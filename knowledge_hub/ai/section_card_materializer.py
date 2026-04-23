from __future__ import annotations

import json
import re
from typing import Any

from knowledge_hub.ai.ask_v2_support import clean_text
from knowledge_hub.ai.section_cards import assess_section_source_quality, project_section_cards, rank_section_cards, section_coverage
from knowledge_hub.infrastructure.persistence.stores.section_card_v1_store import SectionCardV1Store
from knowledge_hub.document_memory import DocumentMemoryBuilder
from knowledge_hub.learning.task_router import get_llm_for_task
from knowledge_hub.papers.card_v2_builder import PaperCardV2Builder


_ROLE_ORDER = ("problem", "method", "results", "limitations")
_ROLE_QUERY = {
    "problem": "paper problem motivation background summary",
    "method": "paper method architecture implementation mechanism",
    "results": "paper results evidence evaluation benchmarks",
    "limitations": "paper limitations caveats failure cases scope",
}


def _extract_json_object(text: Any) -> dict[str, Any]:
    body = str(text or "").strip()
    if not body:
        return {}
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", body, flags=re.DOTALL)
    if fenced:
        body = fenced.group(1)
    else:
        start = body.find("{")
        end = body.rfind("}")
        if start >= 0 and end > start:
            body = body[start : end + 1]
    body = re.sub(r",(\s*[}\]])", r"\1", body)
    try:
        value = json.loads(body)
    except Exception:
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _clean_list(values: Any, *, limit: int = 8) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        raw = [values]
    else:
        try:
            raw = list(values)
        except Exception:
            raw = [values]
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        token = clean_text(item)
        if not token:
            continue
        lowered = token.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(token)
        if len(result) >= max(1, int(limit)):
            break
    return result


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def _role_context(role: str, items: list[dict[str, Any]]) -> str:
    lines = [f"role: {role}"]
    for index, item in enumerate(items, start=1):
        lines.extend(
            [
                f"[unit {index}]",
                f"title: {clean_text(item.get('title'))}",
                f"path: {clean_text(item.get('section_path'))}",
                f"summary: {clean_text(item.get('contextual_summary'))}",
                f"excerpt: {clean_text(item.get('source_excerpt'))}",
                f"claims: {'; '.join(_clean_list(item.get('claims'), limit=4))}",
                f"concepts: {'; '.join(_clean_list(item.get('concepts'), limit=4))}",
            ]
        )
    return "\n".join(lines).strip()


def _prompt(*, paper_title: str, role: str, context: str) -> str:
    return f"""You are creating one inspectable paper SectionCard snapshot.
Return one JSON object only. Do not include markdown or commentary.

Paper title: {paper_title}
Role: {role}

Schema:
{{
  "title": "string",
  "sectionPath": "string",
  "contextualSummary": "2-4 sentence bounded summary grounded only in the provided units",
  "sourceExcerpt": "one short evidence excerpt paraphrase or exact short quote from the provided units",
  "keyPoints": ["string"],
  "scopeNotes": ["string"],
  "confidence": 0.0
}}

Rules:
- Use only the supplied unit context.
- Preserve uncertainty. If scope/limitations are unclear, leave scopeNotes empty rather than inventing.
- Do not introduce external knowledge.
- Keep title and sectionPath short and role-appropriate.

Context:
{context}
"""


def _fallback_card(*, paper_id: str, document_id: str, role: str, units: list[dict[str, Any]]) -> dict[str, Any]:
    primary = units[0]
    key_points = []
    for item in units:
        key_points.extend(_clean_list(item.get("claims"), limit=4))
        if len(key_points) >= 4:
            break
    if not key_points:
        key_points = [clean_text(item.get("contextual_summary")) for item in units if clean_text(item.get("contextual_summary"))][:3]
    return {
        "section_card_id": f"paper-section-card-materialized:{paper_id}:{role}",
        "paper_id": paper_id,
        "document_id": document_id,
        "role": role,
        "title": clean_text(primary.get("title")) or role.title(),
        "section_path": clean_text(primary.get("section_path")) or role.title(),
        "unit_type": clean_text(primary.get("unit_type")) or "section",
        "unit_ids": [clean_text(item.get("unit_id")) for item in units if clean_text(item.get("unit_id"))],
        "contextual_summary": clean_text(primary.get("contextual_summary")),
        "source_excerpt": clean_text(primary.get("source_excerpt")),
        "document_thesis": clean_text(primary.get("document_thesis")),
        "key_points": _clean_list(key_points, limit=4),
        "scope_notes": [],
        "claims": _clean_list([claim for item in units for claim in list(item.get("claims") or [])], limit=8),
        "concepts": _clean_list([concept for item in units for concept in list(item.get("concepts") or [])], limit=8),
        "confidence": max(0.15, min(0.85, sum(_safe_float(item.get("confidence")) for item in units) / max(1, len(units)))),
        "provenance": {
            "builder": "section-card-materializer-v1",
            "mode": "fallback",
            "role": role,
            "sourceUnitCount": len(units),
        },
        "search_text": clean_text(
            " ".join(
                [
                    role,
                    clean_text(primary.get("title")),
                    clean_text(primary.get("section_path")),
                    clean_text(primary.get("contextual_summary")),
                    clean_text(primary.get("source_excerpt")),
                    " ".join(_clean_list(key_points, limit=4)),
                ]
            )
        ),
        "origin": "materialized_v1",
        "generator_model": "",
    }


class PaperSectionCardMaterializer:
    def __init__(self, sqlite_db, config: Any):
        self.sqlite_db = sqlite_db
        self.config = config
        self.store = SectionCardV1Store(sqlite_db.conn)
        self.store.ensure_schema()

    def list_materialized(self, *, paper_id: str) -> list[dict[str, Any]]:
        rows = self.store.list_paper_cards(str(paper_id).strip())
        return [self._row_to_card(row) for row in rows]

    def build_and_store(
        self,
        *,
        paper_id: str,
        allow_external: bool,
        llm_mode: str = "auto",
        max_units_per_role: int = 2,
    ) -> dict[str, Any]:
        token = str(paper_id or "").strip()
        if not token:
            raise ValueError("paper_id is required")
        card = self.sqlite_db.get_paper_card_v2(token)
        if not card:
            card = PaperCardV2Builder(self.sqlite_db).build_and_store(paper_id=token)
        units = list(self.sqlite_db.list_document_memory_units(f"paper:{token}", limit=200) or [])
        if not units:
            DocumentMemoryBuilder(self.sqlite_db, config=self.config).build_and_store_paper(paper_id=token)
            units = list(self.sqlite_db.list_document_memory_units(f"paper:{token}", limit=200) or [])
        projected = project_section_cards(source_kind="paper", source_card=card, units=units)
        projected_cov = section_coverage(section_cards=projected)
        quality_gate = assess_section_source_quality(section_cards=projected, coverage=projected_cov)
        if not quality_gate.get("allowed"):
            return {
                "schema": "knowledge-hub.section-cards.build.result.v1",
                "status": "blocked",
                "paperId": token,
                "count": 0,
                "sectionCoverage": projected_cov,
                "qualityGate": quality_gate,
                "blockReason": clean_text(quality_gate.get("reason")),
                "items": [],
                "warnings": [f"section_quality_gate:{clean_text(quality_gate.get('reason'))}"],
                "llmRoute": {},
            }
        grouped: dict[str, list[dict[str, Any]]] = {}
        for role in _ROLE_ORDER:
            candidates = [item for item in projected if clean_text(item.get("role")) == role]
            ranked = rank_section_cards(query=_ROLE_QUERY[role], section_cards=candidates, intent="implementation" if role == "method" else "paper_summary")
            grouped[role] = ranked[: max(1, int(max_units_per_role))]
        llm, decision, warnings = get_llm_for_task(
            self.config,
            task_type="materialization_summary",
            allow_external=bool(allow_external),
            query=clean_text(card.get("title")) or token,
            context="\n\n".join(_role_context(role, items) for role, items in grouped.items() if items),
            source_count=sum(len(items) for items in grouped.values()),
            force_route=str(llm_mode or "auto").strip().lower() or "auto",
            timeout_sec=90,
        )
        built_cards: list[dict[str, Any]] = []
        for role in _ROLE_ORDER:
            items = grouped.get(role) or []
            if not items:
                continue
            payload = {}
            if llm is not None:
                try:
                    raw = llm.generate(
                        _prompt(paper_title=clean_text(card.get("title")) or token, role=role, context=_role_context(role, items)),
                        max_tokens=700,
                    )
                    payload = _extract_json_object(raw)
                except Exception as error:
                    warnings.append(f"{role}:llm_error:{type(error).__name__}")
            if not payload:
                built = _fallback_card(paper_id=token, document_id=f"paper:{token}", role=role, units=items)
            else:
                primary = items[0]
                key_points = _clean_list(payload.get("keyPoints"), limit=4)
                scope_notes = _clean_list(payload.get("scopeNotes"), limit=4)
                built = {
                    "section_card_id": f"paper-section-card-materialized:{token}:{role}",
                    "paper_id": token,
                    "document_id": f"paper:{token}",
                    "role": role,
                    "title": clean_text(payload.get("title")) or clean_text(primary.get("title")) or role.title(),
                    "section_path": clean_text(payload.get("sectionPath")) or clean_text(primary.get("section_path")) or role.title(),
                    "unit_type": clean_text(primary.get("unit_type")) or "section",
                    "unit_ids": [clean_text(item.get("unit_id")) for item in items if clean_text(item.get("unit_id"))],
                    "contextual_summary": clean_text(payload.get("contextualSummary")) or clean_text(primary.get("contextual_summary")),
                    "source_excerpt": clean_text(payload.get("sourceExcerpt")) or clean_text(primary.get("source_excerpt")),
                    "document_thesis": clean_text(primary.get("document_thesis")) or clean_text(card.get("paper_core")),
                    "key_points": key_points,
                    "scope_notes": scope_notes,
                    "claims": _clean_list([claim for item in items for claim in list(item.get("claims") or [])], limit=8),
                    "concepts": _clean_list([concept for item in items for concept in list(item.get("concepts") or [])], limit=8),
                    "confidence": _safe_float(payload.get("confidence"), max(_safe_float(primary.get("confidence")), 0.25)),
                    "provenance": {
                        "builder": "section-card-materializer-v1",
                        "mode": "llm",
                        "role": role,
                        "sourceUnitCount": len(items),
                        "route": decision.route,
                        "provider": decision.provider,
                    },
                    "search_text": clean_text(
                        " ".join(
                            [
                                role,
                                clean_text(payload.get("title")),
                                clean_text(payload.get("sectionPath")),
                                clean_text(payload.get("contextualSummary")),
                                clean_text(payload.get("sourceExcerpt")),
                                " ".join(key_points),
                                " ".join(scope_notes),
                            ]
                        )
                    ),
                    "origin": "materialized_v1",
                    "generator_model": clean_text(getattr(llm, "model", "")) or clean_text(decision.model),
                }
            built_cards.append(built)
        stored = self.store.replace_paper_cards(paper_id=token, cards=built_cards)
        materialized_cards = [self._row_to_card(row) for row in stored]
        return {
            "schema": "knowledge-hub.section-cards.build.result.v1",
            "status": "ok",
            "paperId": token,
            "count": len(materialized_cards),
            "sectionCoverage": section_coverage(section_cards=materialized_cards),
            "qualityGate": quality_gate,
            "items": materialized_cards,
            "warnings": warnings,
            "llmRoute": decision.to_dict(),
        }

    def _row_to_card(self, row: dict[str, Any]) -> dict[str, Any]:
        unit_ids = _clean_list(row.get("unit_ids"))
        key_points = _clean_list(row.get("key_points"))
        scope_notes = _clean_list(row.get("scope_notes"))
        return {
            "section_card_id": clean_text(row.get("section_card_id")),
            "source_kind": "paper",
            "source_card_id": f"paper-card-v2:{clean_text(row.get('paper_id'))}",
            "source_id": clean_text(row.get("paper_id")),
            "paper_id": clean_text(row.get("paper_id")),
            "document_id": clean_text(row.get("document_id")),
            "unit_id": unit_ids[0] if unit_ids else "",
            "unit_ids": unit_ids,
            "title": clean_text(row.get("title")),
            "section_path": clean_text(row.get("section_path")),
            "unit_type": clean_text(row.get("unit_type")) or "section",
            "role": clean_text(row.get("role")) or "other",
            "order_index": 0,
            "contextual_summary": clean_text(row.get("contextual_summary")),
            "source_excerpt": clean_text(row.get("source_excerpt")),
            "document_thesis": clean_text(row.get("document_thesis")),
            "confidence": _safe_float(row.get("confidence")),
            "claims": _clean_list(row.get("claims"), limit=8),
            "concepts": _clean_list(row.get("concepts"), limit=8),
            "provenance": dict(row.get("provenance") or {}),
            "appendix_like": False,
            "search_text": clean_text(row.get("search_text")),
            "origin": clean_text(row.get("origin")) or "materialized_v1",
            "key_points": key_points,
            "scope_notes": scope_notes,
            "generator_model": clean_text(row.get("generator_model")),
        }


__all__ = ["PaperSectionCardMaterializer"]
