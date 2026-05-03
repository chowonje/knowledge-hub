"""Build additive v2 paper cards from existing paper/document/claim artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any

from knowledge_hub.core.card_v2_common import (
    best_unit as _best_unit,
    claim_is_accepted as _claim_is_accepted,
    clean_text as _clean_text,
    coverage_status as _coverage_status,
    first_nonempty as _first_nonempty,
    slot_excerpt as _slot_excerpt,
    snippet_hash as _snippet_hash,
    stable_anchor_id as _stable_anchor_id,
)
from knowledge_hub.papers.memory_builder import PaperMemoryBuilder
from knowledge_hub.papers.memory_payloads import shared_slot_payload


_SECTION_METHOD_RE = re.compile(r"\b(method|approach|architecture|pipeline|implementation|방법|접근|구현)\b", re.IGNORECASE)
_SECTION_RESULT_RE = re.compile(r"\b(result|finding|evaluation|experiment|benchmark|metric|결과|평가|실험)\b", re.IGNORECASE)
_SECTION_LIMIT_RE = re.compile(r"\b(limit|limitation|future work|risk|caveat|한계)\b", re.IGNORECASE)
_SECTION_PROBLEM_RE = re.compile(r"\b(problem|motivation|background|introduction|abstract|문제|배경|요약|초록)\b", re.IGNORECASE)
_DATASET_RE = re.compile(r"\b(dataset|benchmark|corpus|data|데이터셋|벤치마크)\b", re.IGNORECASE)
_METRIC_RE = re.compile(r"\b(metric|accuracy|f1|auc|auroc|auprc|latency|throughput|precision|recall|지표|정확도)\b", re.IGNORECASE)

def _clean_lines(values: list[Any], *, limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        token = _clean_text(raw)
        if not token:
            continue
        lowered = token.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(token)
        if limit is not None and len(result) >= limit:
            break
    return result


def _is_heuristic_title_fallback_concept(row: dict[str, Any]) -> bool:
    return _clean_text(row.get("source")) == "paper_memory_title_fallback"


def _claim_role(claim: dict[str, Any]) -> str:
    text = _clean_text(claim.get("claim_text"))
    predicate = _clean_text(claim.get("predicate")).casefold()
    haystack = f"{predicate} {text}"
    if _SECTION_LIMIT_RE.search(haystack):
        return "limitation"
    if _SECTION_RESULT_RE.search(haystack) or re.search(r"\d", haystack):
        return "result"
    if _SECTION_METHOD_RE.search(haystack):
        return "method"
    if _DATASET_RE.search(haystack):
        return "dataset"
    if _METRIC_RE.search(haystack):
        return "metric"
    return "supporting"


def _role_slot_key(role: str) -> str:
    mapping = {
        "result": "result_core",
        "method": "method_core",
        "limitation": "limitations_core",
        "dataset": "dataset_core",
        "metric": "metric_core",
        "supporting": "paper_core",
    }
    return mapping.get(str(role or "").strip().lower(), "paper_core")


def _claim_sort_key(claim: dict[str, Any]) -> tuple[float, float, str]:
    confidence = float(claim.get("confidence") or 0.0)
    evidence_ptrs = claim.get("evidence_ptrs") if isinstance(claim.get("evidence_ptrs"), list) else []
    evidence_weight = min(1.0, 0.15 * len([ptr for ptr in evidence_ptrs if isinstance(ptr, dict)]))
    return confidence + evidence_weight, confidence, _clean_text(claim.get("claim_id"))

def _paper_published_at(paper: dict[str, Any], memory_card: dict[str, Any] | None) -> str:
    token = _clean_text((memory_card or {}).get("published_at"))
    if token:
        return token
    year = _clean_text(paper.get("year"))
    if re.fullmatch(r"\d{4}", year):
        return datetime(int(year), 1, 1, tzinfo=timezone.utc).isoformat()
    return ""


def _normalization_by_claim_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        claim_id = _clean_text(row.get("claim_id"))
        if not claim_id or claim_id in by_id:
            continue
        by_id[claim_id] = dict(row)
    return by_id


def _claim_normalization_payload(normalization: dict[str, Any] | None) -> dict[str, Any]:
    row = dict(normalization or {})
    return {
        "task": _clean_text(row.get("task")),
        "dataset": _clean_text(row.get("dataset")),
        "metric": _clean_text(row.get("metric")),
        "comparator": _clean_text(row.get("comparator")),
        "result_direction": _clean_text(row.get("result_direction")),
        "result_value_text": _clean_text(row.get("result_value_text")),
        "result_value_numeric": row.get("result_value_numeric"),
        "condition_text": _clean_text(row.get("condition_text")),
        "scope_text": _clean_text(row.get("scope_text")),
        "limitation_text": _clean_text(row.get("limitation_text")),
        "evidence_strength": _clean_text(row.get("evidence_strength")),
    }


class PaperCardV2Builder:
    def __init__(self, sqlite_db):
        self.sqlite_db = sqlite_db

    def _ensure_memory_card(self, paper_id: str) -> dict[str, Any] | None:
        card = self.sqlite_db.get_paper_memory_card(paper_id)
        if card:
            return dict(card)
        try:
            built = dict(PaperMemoryBuilder(self.sqlite_db).build_and_store(paper_id=paper_id, materialize_card=False))
            return built
        except Exception:
            return None

    def _document_payload(self, paper_id: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        document_id = f"paper:{paper_id}"
        summary = self.sqlite_db.get_document_memory_summary(document_id)
        units = self.sqlite_db.list_document_memory_units(document_id, limit=200)
        return summary, list(units or [])

    def _claims_for_paper(self, paper_id: str) -> list[dict[str, Any]]:
        note_id = f"paper:{paper_id}"
        rows = list(self.sqlite_db.list_claims_by_note(note_id, limit=80))
        rows.extend(self.sqlite_db.list_claims_by_entity(f"paper:{paper_id}", limit=80))
        accepted = [dict(row) for row in rows if _claim_is_accepted(dict(row))]
        accepted.sort(key=_claim_sort_key, reverse=True)
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in accepted:
            claim_id = _clean_text(row.get("claim_id"))
            if not claim_id or claim_id in seen:
                continue
            seen.add(claim_id)
            deduped.append(dict(row))
        return deduped

    def _entity_refs(self, paper_id: str, concepts: list[dict[str, Any]], card_title: str) -> list[dict[str, Any]]:
        if any(not _is_heuristic_title_fallback_concept(concept) for concept in concepts):
            concepts = [concept for concept in concepts if not _is_heuristic_title_fallback_concept(concept)]
        refs: list[dict[str, Any]] = [
            {
                "entity_id": f"paper:{paper_id}",
                "entity_name": card_title,
                "entity_type": "paper",
                "weight": 1.0,
                "role": "paper",
            }
        ]
        for index, concept in enumerate(list(concepts or [])[:10], 1):
            refs.append(
                {
                    "entity_id": _clean_text(concept.get("entity_id") or concept.get("id")),
                    "entity_name": _clean_text(concept.get("canonical_name") or concept.get("id")),
                    "entity_type": _clean_text(concept.get("entity_type") or "concept"),
                    "weight": round(max(0.25, 1.0 - (0.08 * index)), 3),
                    "role": "concept",
                }
            )
        deduped: list[dict[str, Any]] = []
        seen_entity_ids: set[str] = set()
        for ref in refs:
            entity_id = _clean_text(ref.get("entity_id"))
            if not entity_id or entity_id in seen_entity_ids:
                continue
            seen_entity_ids.add(entity_id)
            deduped.append(ref)
        return deduped

    def build(self, *, paper_id: str) -> dict[str, Any]:
        paper = self.sqlite_db.get_paper(paper_id)
        if not paper:
            raise ValueError(f"paper not found: {paper_id}")
        memory_card = self._ensure_memory_card(paper_id)
        summary, units = self._document_payload(paper_id)
        claims = self._claims_for_paper(paper_id)
        claim_ids = [_clean_text(item.get("claim_id")) for item in claims if _clean_text(item.get("claim_id"))]
        normalizations = _normalization_by_claim_id(
            self.sqlite_db.list_claim_normalizations(claim_ids=claim_ids, status="normalized", limit=max(10, len(claim_ids) * 2))
        )
        concepts = list(self.sqlite_db.get_paper_concepts(paper_id) or [])

        document_summary_excerpt = _first_nonempty((summary or {}).get("contextual_summary"), (summary or {}).get("source_excerpt"))
        method_unit = _best_unit(units, _SECTION_METHOD_RE)
        result_unit = _best_unit(units, _SECTION_RESULT_RE)
        limit_unit = _best_unit(units, _SECTION_LIMIT_RE)
        problem_unit = _best_unit(units, _SECTION_PROBLEM_RE)
        dataset_unit = _best_unit(units, _DATASET_RE)
        metric_unit = _best_unit(units, _METRIC_RE)

        claim_texts = _clean_lines([row.get("claim_text") for row in claims], limit=5)
        concept_names = _clean_lines([row.get("canonical_name") or row.get("id") for row in concepts], limit=8)
        datasets = _clean_lines([normalizations.get(claim_id, {}).get("dataset") for claim_id in claim_ids], limit=3)
        metrics = _clean_lines([normalizations.get(claim_id, {}).get("metric") for claim_id in claim_ids], limit=3)
        claim_limitations = _clean_lines([normalizations.get(claim_id, {}).get("limitation_text") for claim_id in claim_ids], limit=2)
        memory_slots = shared_slot_payload(memory_card)

        paper_core = _first_nonempty(memory_slots.get("paper_core"), document_summary_excerpt, paper.get("notes"), paper.get("title"))
        problem_core = _first_nonempty(memory_slots.get("problem_core"), _slot_excerpt(problem_unit or {}), (summary or {}).get("document_thesis"))
        method_core = _first_nonempty(memory_slots.get("method_core"), _slot_excerpt(method_unit or {}), claim_texts[0] if claim_texts else "")
        result_core = _first_nonempty(memory_slots.get("evidence_core"), _slot_excerpt(result_unit or {}), claim_texts[0] if claim_texts else "")
        limitations_core = _first_nonempty(memory_slots.get("limitations_core"), _slot_excerpt(limit_unit or {}), claim_limitations[0] if claim_limitations else "")
        dataset_core = _first_nonempty(datasets[0] if datasets else "", _slot_excerpt(dataset_unit or {}))
        metric_core = _first_nonempty(metrics[0] if metrics else "", _slot_excerpt(metric_unit or {}))
        when_not_to_use = _first_nonempty(limitations_core, claim_limitations[0] if claim_limitations else "")

        card_id = f"paper-card-v2:{paper_id}"
        slot_coverage = {
            "paperCore": _coverage_status(paper_core),
            "problemCore": _coverage_status(problem_core),
            "methodCore": _coverage_status(method_core),
            "resultCore": _coverage_status(result_core),
            "limitationsCore": _coverage_status(limitations_core),
            "datasetCore": _coverage_status(dataset_core),
            "metricCore": _coverage_status(metric_core),
            "whenNotToUse": _coverage_status(when_not_to_use),
        }
        quality_flag = _clean_text(memory_slots.get("quality_flag")) or ("ok" if len(claims) >= 2 else "needs_review" if claims else "unscored")
        card = {
            "card_id": card_id,
            "paper_id": paper_id,
            "title": _clean_text(paper.get("title") or memory_slots.get("title") or paper_id),
            "paper_core": paper_core,
            "problem_core": problem_core,
            "method_core": method_core,
            "result_core": result_core,
            "limitations_core": limitations_core,
            "dataset_core": dataset_core,
            "metric_core": metric_core,
            "when_not_to_use": when_not_to_use,
            "source_memory_id": _clean_text(memory_slots.get("memory_id")),
            "search_text": _clean_text(
                " ".join(
                    [
                        _clean_text(paper.get("title")),
                        paper_core,
                        problem_core,
                        method_core,
                        result_core,
                        limitations_core,
                        dataset_core,
                        metric_core,
                        when_not_to_use,
                        " ".join(claim_texts),
                        " ".join(concept_names),
                    ]
                )
            ),
            "quality_flag": quality_flag,
            "published_at": _paper_published_at(paper, memory_card),
            "version": "paper-card-v2",
            "slot_coverage": slot_coverage,
            "diagnostics": {
                "sourcePaperMemory": bool(memory_card),
                "sourceDocumentSummary": bool(summary),
                "documentUnitCount": len(units),
                "acceptedClaimCount": len(claims),
                "conceptCount": len(concept_names),
            },
        }

        claim_refs: list[dict[str, Any]] = []
        selected_claims: list[dict[str, Any]] = []
        seen_claim_ids: set[str] = set()
        representative_roles = ("result", "method", "limitation", "dataset", "metric")
        for role in representative_roles:
            representative = next((row for row in claims if _claim_role(row) == role), None)
            if representative is None:
                continue
            claim_id = _clean_text(representative.get("claim_id"))
            if not claim_id or claim_id in seen_claim_ids:
                continue
            seen_claim_ids.add(claim_id)
            selected_claims.append(representative)
        for claim in claims:
            claim_id = _clean_text(claim.get("claim_id"))
            if not claim_id or claim_id in seen_claim_ids:
                continue
            seen_claim_ids.add(claim_id)
            selected_claims.append(claim)
            if len(selected_claims) >= 8:
                break
        for rank, claim in enumerate(selected_claims[:8], 1):
            claim_id = _clean_text(claim.get("claim_id"))
            role = _claim_role(claim)
            normalization = _claim_normalization_payload(normalizations.get(claim_id))
            claim_refs.append(
                {
                    "claim_id": claim_id,
                    "role": role,
                    "slot_key": _role_slot_key(role),
                    "confidence": float(claim.get("confidence") or 0.0),
                    "rank": rank,
                    "reason": "representative_claim" if role in representative_roles else "accepted_claim_ranked_by_confidence_and_evidence",
                    "normalization": normalization,
                }
            )

        claim_unit_index: dict[str, dict[str, Any]] = {}
        for unit in units:
            for claim_id in list(unit.get("claims") or []):
                token = _clean_text(claim_id)
                if token and token not in claim_unit_index:
                    claim_unit_index[token] = unit

        anchors: list[dict[str, Any]] = []
        slot_units = [
            ("paper_core", problem_unit or summary or {}, "paper_summary"),
            ("problem_core", problem_unit or summary or {}, "problem"),
            ("method_core", method_unit or summary or {}, "method"),
            ("result_core", result_unit or summary or {}, "result"),
            ("limitations_core", limit_unit or summary or {}, "limitation"),
            ("dataset_core", dataset_unit or result_unit or summary or {}, "dataset"),
            ("metric_core", metric_unit or result_unit or summary or {}, "metric"),
            ("when_not_to_use", limit_unit or summary or {}, "when_not_to_use"),
        ]
        for field_name, unit, role in slot_units:
            excerpt = _first_nonempty(_slot_excerpt(unit), card.get(field_name))
            if not excerpt:
                continue
            section_path = _clean_text(unit.get("section_path") or unit.get("title") or field_name)
            anchors.append(
                {
                    "anchor_id": _stable_anchor_id(card_id, role, field_name, section_path, excerpt),
                    "card_id": card_id,
                    "claim_id": "",
                    "paper_id": paper_id,
                    "document_id": _clean_text(unit.get("document_id") or f"paper:{paper_id}"),
                    "unit_id": _clean_text(unit.get("unit_id")),
                    "chunk_id": "",
                    "title": card["title"],
                    "source_type": "paper",
                    "section_path": section_path,
                    "span_locator": _clean_text(unit.get("unit_id") or field_name),
                    "snippet_hash": _snippet_hash(section_path, excerpt),
                    "evidence_role": role,
                    "excerpt": excerpt,
                    "score": 0.9 if unit else 0.6,
                }
            )

        for ref in claim_refs:
            claim_id = _clean_text(ref.get("claim_id"))
            claim = next((row for row in claims if _clean_text(row.get("claim_id")) == claim_id), {})
            role = _clean_text(ref.get("role"))
            preferred_unit = (
                result_unit
                if role == "result"
                else method_unit
                if role == "method"
                else limit_unit
                if role == "limitation"
                else dataset_unit
                if role == "dataset"
                else metric_unit
                if role == "metric"
                else summary
            )
            unit = preferred_unit or claim_unit_index.get(claim_id) or summary or {}
            excerpt = _first_nonempty(_slot_excerpt(unit), claim.get("claim_text"), card.get("result_core"))
            section_path = _clean_text(unit.get("section_path") or unit.get("title") or ref.get("role") or "claim")
            anchors.append(
                {
                    "anchor_id": _stable_anchor_id(card_id, "claim", claim_id, section_path, excerpt),
                    "card_id": card_id,
                    "claim_id": claim_id,
                    "paper_id": paper_id,
                    "document_id": _clean_text(unit.get("document_id") or f"paper:{paper_id}"),
                    "unit_id": _clean_text(unit.get("unit_id")),
                    "chunk_id": "",
                    "title": card["title"],
                    "source_type": "paper",
                    "section_path": section_path,
                    "span_locator": _clean_text(unit.get("unit_id") or claim_id),
                    "snippet_hash": _snippet_hash(claim_id, excerpt),
                    "evidence_role": role or "supporting",
                    "excerpt": excerpt,
                    "score": round(0.75 + (0.2 * float(ref.get("confidence") or 0.0)), 4),
                }
            )

        entity_refs = self._entity_refs(paper_id, concepts, card["title"])
        return {
            "card": card,
            "claim_refs": claim_refs,
            "anchors": anchors,
            "entity_refs": entity_refs,
        }

    def build_and_store(self, *, paper_id: str) -> dict[str, Any]:
        payload = self.build(paper_id=paper_id)
        card = self.sqlite_db.upsert_paper_card_v2(card=payload["card"])
        card_id = _clean_text(card.get("card_id") or payload["card"].get("card_id"))
        self.sqlite_db.replace_paper_card_claim_refs_v2(card_id=card_id, refs=payload["claim_refs"])
        self.sqlite_db.replace_evidence_anchors_v2(card_id=card_id, anchors=payload["anchors"])
        self.sqlite_db.replace_paper_card_entity_refs_v2(card_id=card_id, refs=payload["entity_refs"])
        from knowledge_hub.domain.ai_papers.claim_cards import ClaimCardBuilder

        claim_cards = ClaimCardBuilder(self.sqlite_db).build_and_store_for_source_card(
            source_kind="paper",
            source_card=card,
        )
        return {
            **card,
            "claim_refs": self.sqlite_db.list_paper_card_claim_refs_v2(card_id=card_id),
            "anchors": self.sqlite_db.list_evidence_anchors_v2(card_id=card_id),
            "entity_refs": self.sqlite_db.list_paper_card_entity_refs_v2(card_id=card_id),
            "claim_cards": claim_cards,
        }


__all__ = ["PaperCardV2Builder"]
