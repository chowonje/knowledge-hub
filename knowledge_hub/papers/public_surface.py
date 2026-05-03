"""User-facing paper reading helpers.

These helpers keep the public CLI surface thin while reusing the structured
summary, paper memory, and search backends.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.papers.memory_builder import _normalize_title_concept, _summary_value_is_unusable
from knowledge_hub.papers.memory_retriever import PaperMemoryRetriever
from knowledge_hub.papers.structured_summary import StructuredPaperSummaryService
from knowledge_hub.vault.concepts import find_concept_note_path


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _summary_service(khub) -> StructuredPaperSummaryService:
    return StructuredPaperSummaryService(khub.sqlite_db(), khub.config)


def _parse_iso(value: Any) -> datetime | None:
    token = _clean_text(value)
    if not token:
        return None
    try:
        if token.endswith("Z"):
            return datetime.fromisoformat(token.replace("Z", "+00:00")).astimezone(timezone.utc)
        parsed = datetime.fromisoformat(token)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _latest_timestamp(*values: Any) -> datetime | None:
    timestamps = [item for item in (_parse_iso(value) for value in values) if item is not None]
    if not timestamps:
        return None
    return max(timestamps)


def _summary_artifact_built_at(service: StructuredPaperSummaryService, *, paper_id: str) -> datetime | None:
    manifest = dict(service.load_manifest(paper_id=paper_id) or {})
    built_at = _parse_iso(manifest.get("built_at"))
    if built_at is not None:
        return built_at
    path = service.artifact_dir_for(paper_id=paper_id) / "summary.json"
    if not path.exists():
        return None
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return None


def _document_memory_updated_at(sqlite_db: Any, *, paper_id: str) -> datetime | None:
    getter = getattr(sqlite_db, "get_document_memory_summary", None)
    if getter is None:
        return None
    row = getter(f"paper:{paper_id}") or {}
    if not isinstance(row, dict):
        row = dict(row or {})
    return _latest_timestamp(
        row.get("updated_at"),
        row.get("updatedAt"),
        row.get("observed_at"),
        row.get("observedAt"),
    )


def _memory_card_updated_at(sqlite_db: Any, *, paper_id: str) -> datetime | None:
    getter = getattr(sqlite_db, "get_paper_memory_card", None)
    if getter is None:
        return None
    row = getter(paper_id) or {}
    if not isinstance(row, dict):
        row = dict(row or {})
    return _latest_timestamp(row.get("updated_at"), row.get("updatedAt"))


def _combine_status(*statuses: str) -> str:
    ranked = [str(item or "").strip() for item in statuses if str(item or "").strip()]
    for candidate in ("missing", "degraded", "stale", "partial", "failed", "ok"):
        if candidate in ranked:
            return candidate
    return ranked[0] if ranked else "ok"


def _append_unique(items: list[str], *values: str) -> list[str]:
    for value in values:
        token = _clean_text(value)
        if token and token not in items:
            items.append(token)
    return items


def _compact_evidence_summary(payload: dict[str, Any], *, include_refs: bool = False) -> dict[str, Any]:
    evidence_summaries = dict(payload.get("evidenceSummaries") or {})
    out: dict[str, dict[str, Any]] = {}
    for field, packet in evidence_summaries.items():
        entry: dict[str, Any] = {
            "summaryLines": _clean_list((packet or {}).get("summaryLines")),
            "claimHintsUsed": int((packet or {}).get("claimHintsUsed") or 0),
            "unitCount": int((packet or {}).get("unitCount") or 0),
        }
        if include_refs:
            entry["evidenceRefs"] = list((packet or {}).get("evidenceRefs") or [])
        out[str(field)] = entry
    return out


def _inspect_public_summary_artifact(khub, *, paper_id: str) -> dict[str, Any]:
    sqlite_db = khub.sqlite_db()
    service = _summary_service(khub)
    payload = dict(service.load_artifact(paper_id=paper_id) or {})
    paper = sqlite_db.get_paper(paper_id) or {}
    warnings: list[str] = []
    if not payload:
        return {
            "payload": {},
            "status": "missing",
            "warnings": ["summary_artifact_missing"],
            "builtAt": "",
            "paperTitle": _clean_text(paper.get("title") or paper_id),
        }
    warnings.extend(str(item) for item in list(payload.get("warnings") or []) if _clean_text(item))
    summary = dict(payload.get("summary") or {})
    summary_present = bool(_clean_text(summary.get("oneLine")) or _clean_text(summary.get("coreIdea")))
    summary_usable = bool(_usable_public_text(summary.get("oneLine")) or _usable_public_text(summary.get("coreIdea")))
    built_at = _summary_artifact_built_at(service, paper_id=paper_id)
    upstream_updated_at = _document_memory_updated_at(sqlite_db, paper_id=paper_id)

    status = _clean_text(payload.get("status")) or "ok"
    if not summary_present:
        status = "missing"
        _append_unique(warnings, "summary_artifact_missing")
    elif not summary_usable:
        status = "degraded"
        _append_unique(warnings, "summary_artifact_unusable")
    elif built_at is not None and upstream_updated_at is not None and upstream_updated_at > built_at:
        status = "stale"
        _append_unique(warnings, "summary_artifact_stale")

    return {
        "payload": payload,
        "status": status,
        "warnings": warnings,
        "builtAt": built_at.isoformat() if built_at is not None else "",
        "paperTitle": _clean_text(paper.get("title") or payload.get("paperTitle") or paper_id),
    }


def _inspect_public_memory_card(khub, *, paper_id: str, include_refs: bool = True, summary_state: dict[str, Any] | None = None) -> dict[str, Any]:
    sqlite_db = khub.sqlite_db()
    card = dict(PaperMemoryRetriever(sqlite_db).get(paper_id, include_refs=include_refs) or {})
    warnings: list[str] = []
    if not card:
        return {
            "card": {},
            "status": "missing",
            "warnings": ["memory_card_missing"],
            "updatedAt": "",
        }

    usable = bool(
        _usable_public_text(card.get("paperCore"))
        or _usable_public_text(card.get("methodCore"))
        or _usable_public_text(card.get("evidenceCore"))
    )
    present = bool(
        _clean_text(card.get("paperCore"))
        or _clean_text(card.get("methodCore"))
        or _clean_text(card.get("evidenceCore"))
    )
    memory_updated_at = _memory_card_updated_at(sqlite_db, paper_id=paper_id)
    summary_state = summary_state or _inspect_public_summary_artifact(khub, paper_id=paper_id)
    upstream_updated_at = _latest_timestamp(
        _document_memory_updated_at(sqlite_db, paper_id=paper_id),
        (summary_state or {}).get("builtAt"),
    )
    status = "ok"
    if not present:
        status = "missing"
        _append_unique(warnings, "memory_card_missing")
    elif not usable:
        status = "degraded"
        _append_unique(warnings, "memory_card_unusable")
    elif memory_updated_at is not None and upstream_updated_at is not None and upstream_updated_at > memory_updated_at:
        status = "stale"
        _append_unique(warnings, "memory_card_stale")

    return {
        "card": card,
        "status": status,
        "warnings": warnings,
        "updatedAt": memory_updated_at.isoformat() if memory_updated_at is not None else "",
    }


def load_public_summary_artifact(khub, *, paper_id: str) -> dict[str, Any]:
    return dict(_summary_service(khub).load_artifact(paper_id=paper_id) or {})


def ensure_public_summary(khub, *, paper_id: str) -> dict[str, Any]:
    return dict(_inspect_public_summary_artifact(khub, paper_id=paper_id).get("payload") or {})


def ensure_public_memory_card(khub, *, paper_id: str, include_refs: bool = True) -> dict[str, Any]:
    return dict(_inspect_public_memory_card(khub, paper_id=paper_id, include_refs=include_refs).get("card") or {})


def load_public_memory_card(khub, *, paper_id: str, include_refs: bool = True) -> dict[str, Any]:
    sqlite_db = khub.sqlite_db()
    retriever = PaperMemoryRetriever(sqlite_db)
    return dict(retriever.get(paper_id, include_refs=include_refs) or {})


def build_public_summary_card(khub, *, paper_id: str) -> dict[str, Any]:
    summary_state = _inspect_public_summary_artifact(khub, paper_id=paper_id)
    payload = dict(summary_state.get("payload") or {})
    summary = dict(payload.get("summary") or {})
    memory_state = _inspect_public_memory_card(khub, paper_id=paper_id, include_refs=False, summary_state=summary_state)
    memory_card = dict(memory_state.get("card") or {})
    concepts_detailed = _build_concepts_detailed(khub, paper_id=paper_id, memory_card=memory_card, limit=8)
    warnings = list(summary_state.get("warnings") or [])
    warnings = _append_unique(warnings, *list(memory_state.get("warnings") or []))
    if not concepts_detailed:
        warnings = _append_unique(warnings, "concept_links_missing")
    quality = _build_paper_quality(
        summary_state=summary_state,
        memory_state=memory_state,
        summary_payload=payload,
        memory_card=memory_card,
        concepts_detailed=concepts_detailed,
        warnings=warnings,
    )
    return {
        "schema": "knowledge-hub.paper.public.summary.v1",
        "status": summary_state.get("status") or payload.get("status") or "missing",
        "paperId": payload.get("paperId") or paper_id,
        "paperTitle": summary_state.get("paperTitle") or payload.get("paperTitle") or "",
        "parserUsed": payload.get("parserUsed") or payload.get("parser_used") or "raw",
        "fallbackUsed": bool(payload.get("fallbackUsed")),
        "llmRoute": _clean_text(payload.get("llmRoute")),
        "summary": summary,
        "evidenceSummary": _compact_evidence_summary(payload, include_refs=False),
        "claimCoverage": _claim_coverage_from_summary(payload),
        "warnings": warnings,
        "conceptsDetailed": concepts_detailed,
        "quality": quality,
        "artifactStatus": {
            "summary": summary_state.get("status") or "missing",
            "memory": memory_state.get("status") or "missing",
            "builtAt": summary_state.get("builtAt") or "",
        },
    }


def build_public_evidence_card(khub, *, paper_id: str) -> dict[str, Any]:
    summary_state = _inspect_public_summary_artifact(khub, paper_id=paper_id)
    payload = dict(summary_state.get("payload") or {})
    memory_state = _inspect_public_memory_card(khub, paper_id=paper_id, include_refs=False, summary_state=summary_state)
    memory_card = dict(memory_state.get("card") or {})
    concepts_detailed = _build_concepts_detailed(khub, paper_id=paper_id, memory_card=memory_card, limit=8)
    warnings = list(summary_state.get("warnings") or [])
    warnings = _append_unique(warnings, *list(memory_state.get("warnings") or []))
    if not concepts_detailed:
        warnings = _append_unique(warnings, "concept_links_missing")
    quality = _build_paper_quality(
        summary_state=summary_state,
        memory_state=memory_state,
        summary_payload=payload,
        memory_card=memory_card,
        concepts_detailed=concepts_detailed,
        warnings=warnings,
    )
    return {
        "schema": "knowledge-hub.paper.public.evidence.v1",
        "status": summary_state.get("status") or payload.get("status") or "missing",
        "paperId": payload.get("paperId") or paper_id,
        "paperTitle": summary_state.get("paperTitle") or payload.get("paperTitle") or "",
        "parserUsed": payload.get("parserUsed") or payload.get("parser_used") or "raw",
        "fallbackUsed": bool(payload.get("fallbackUsed")),
        "evidenceSummary": _compact_evidence_summary(payload, include_refs=True),
        "evidenceMap": list(payload.get("evidenceMap") or []),
        "claimCoverage": _claim_coverage_from_summary(payload),
        "warnings": warnings,
        "conceptsDetailed": concepts_detailed,
        "quality": quality,
        "artifactStatus": {
            "summary": summary_state.get("status") or "missing",
            "memory": memory_state.get("status") or "missing",
            "builtAt": summary_state.get("builtAt") or "",
        },
    }


def build_public_memory_card(khub, *, paper_id: str) -> dict[str, Any]:
    sqlite_db = khub.sqlite_db()
    paper = sqlite_db.get_paper(paper_id) or {}
    summary_state = _inspect_public_summary_artifact(khub, paper_id=paper_id)
    memory_state = _inspect_public_memory_card(khub, paper_id=paper_id, include_refs=True, summary_state=summary_state)
    memory_card = dict(memory_state.get("card") or {})
    summary = dict(summary_state.get("payload") or {})
    warnings = _append_unique(
        list(summary_state.get("warnings") or []),
        *list(memory_state.get("warnings") or []),
    )
    concepts_detailed = _build_concepts_detailed(khub, paper_id=paper_id, memory_card=memory_card, limit=8)
    if not concepts_detailed:
        warnings = _append_unique(warnings, "concept_links_missing")
    quality = _build_paper_quality(
        summary_state=summary_state,
        memory_state=memory_state,
        summary_payload=summary,
        memory_card=memory_card,
        concepts_detailed=concepts_detailed,
        warnings=warnings,
    )
    return {
        "schema": "knowledge-hub.paper.public.memory.v1",
        "status": _combine_status(memory_state.get("status") or "missing", summary_state.get("status") or "missing"),
        "paperId": paper_id,
        "paperTitle": _clean_text(paper.get("title") or summary.get("paperTitle") or paper_id),
        "memory": {
            "paperCore": _clean_text(memory_card.get("paperCore")),
            "problemContext": _clean_text(memory_card.get("problemContext")),
            "methodCore": _clean_text(memory_card.get("methodCore")),
            "evidenceCore": _clean_text(memory_card.get("evidenceCore")),
        },
        "memoryCard": memory_card,
        "conceptsDetailed": concepts_detailed,
        "quality": quality,
        "claimCoverage": _claim_coverage_from_summary(summary),
        "provenance": {
            "paperId": paper_id,
            "paperMemoryAvailable": bool(memory_card),
            "summaryArtifact": bool(summary),
            "summaryArtifactStatus": summary_state.get("status") or "missing",
            "summaryBuiltAt": summary_state.get("builtAt") or "",
            "paperMemoryStatus": memory_state.get("status") or "missing",
            "paperMemoryUpdatedAt": memory_state.get("updatedAt") or "",
        },
        "warnings": warnings,
        "artifactStatus": {
            "summary": summary_state.get("status") or "missing",
            "memory": memory_state.get("status") or "missing",
        },
    }


def _dedupe_by_key(rows: list[dict[str, Any]], *, key: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        token = _clean_text(row.get(key))
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(row)
    return out


def _clean_list(values: Any, *, limit: int | None = None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, list):
        raw = values
    elif isinstance(values, tuple):
        raw = list(values)
    else:
        raw = [values]
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        token = _clean_text(item)
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


def _usable_public_text(value: Any) -> str:
    token = _clean_text(value)
    if not token or _summary_value_is_unusable(token):
        return ""
    return token


def _claim_coverage_from_summary(summary_payload: dict[str, Any]) -> dict[str, Any]:
    context_stats = dict(summary_payload.get("contextStats") or {})
    return dict(context_stats.get("claimCoverage") or summary_payload.get("claimCoverage") or {})


def _search_terms(*parts: Any, limit: int = 6) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for part in parts:
        for candidate in re.split(r"[^0-9A-Za-z가-힣]+", _clean_text(part)):
            token = candidate.strip()
            if len(token) < 3:
                continue
            lowered = token.casefold()
            if lowered in seen:
                continue
            seen.add(lowered)
            tokens.append(token)
            if len(tokens) >= limit:
                return tokens
    return tokens


def _top_concepts(sqlite_db: Any, *, paper_id: str, memory_card: dict[str, Any], limit: int) -> list[str]:
    concepts = _clean_list(memory_card.get("conceptLinks"), limit=limit)
    if concepts:
        return concepts
    rows = getattr(sqlite_db, "get_paper_concepts", lambda _paper_id: [])(paper_id) or []
    return _clean_list(
        [
            row.get("canonical_name")
            or row.get("canonicalName")
            or row.get("name")
            or row.get("entity_id")
            for row in rows
        ],
        limit=limit,
    )


_STATUS_SCORES = {
    "ok": 100,
    "stale": 70,
    "degraded": 35,
    "missing": 0,
}

_QUALITY_BANDS = ("strong", "usable", "weak", "degraded")
_CONCEPT_BANDS = ("verified", "mixed", "heuristic")
_SLOT_FIELD_MAP = {
    "paperCore": "paperCore",
    "problemContext": "problemContext",
    "methodCore": "methodCore",
    "evidenceCore": "evidenceCore",
}
_SLOT_SCORE_WEIGHTS = {
    "paperCore": 0.10,
    "problemContext": 0.10,
    "methodCore": 0.15,
    "evidenceCore": 0.15,
}
_PENALTY_CODES = {
    "summary_artifact_unusable": 15,
    "memory_card_unusable": 15,
    "fallback_used": 10,
    "concept_links_missing": 5,
    "likely_semantic_mismatch": 20,
}


def _status_score(status: Any) -> int:
    return int(_STATUS_SCORES.get(_clean_text(status) or "missing", 0))


def _quality_band(score: int) -> str:
    if score >= 80:
        return "strong"
    if score >= 60:
        return "usable"
    if score >= 40:
        return "weak"
    return "degraded"


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _resolve_vault_concepts_dir(config: Any) -> Path | None:
    vault_path = _clean_text(getattr(config, "vault_path", "") or "")
    if not vault_path:
        return None
    root = Path(vault_path).expanduser().resolve()
    configured = _clean_text(getattr(config, "obsidian_concepts_folder", "") or "")
    candidates: list[Path] = []
    if configured:
        candidates.append(root / configured)
    candidates.extend(
        [
            root / "Projects" / "AI" / "AI_Papers" / "Concepts",
            root / "AI" / "AI_Papers" / "Concepts",
            root / "Papers" / "Concepts",
            root / "Concepts",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def _has_concept_note(config: Any, concept_name: str) -> bool:
    concepts_dir = _resolve_vault_concepts_dir(config)
    if concepts_dir is None:
        return False
    return find_concept_note_path(concepts_dir, concept_name) is not None


def _concept_source_from_row(row: dict[str, Any]) -> str:
    entity_source = _clean_text(row.get("source")).casefold()
    reason = _json_dict(row.get("reason_json"))
    reason_source = _clean_text(reason.get("source")).casefold()
    properties = _json_dict(row.get("properties_json"))
    heuristic_source = _clean_text(properties.get("heuristic_source")).casefold()
    combined = " ".join(token for token in (entity_source, reason_source, heuristic_source) if token)
    if "paper_memory_title_fallback" in combined:
        return "title_fallback"
    if "paper_normalize_concepts" in combined:
        return "normalized"
    return "ontology"


def _concept_confidence(source: str, raw_confidence: Any) -> float:
    try:
        confidence = float(raw_confidence)
    except Exception:
        confidence = 0.0
    if source == "ontology":
        return confidence if confidence > 0 else 0.95
    if source == "normalized":
        return max(confidence, 0.8)
    return 0.45


def _concept_source_rank(source: str) -> int:
    return {"ontology": 3, "normalized": 2, "title_fallback": 1}.get(source, 0)


def _build_concepts_detailed(
    khub,
    *,
    paper_id: str,
    memory_card: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    sqlite_db = khub.sqlite_db()
    rows = getattr(sqlite_db, "get_paper_concepts", lambda _paper_id: [])(paper_id) or []
    grouped: dict[str, dict[str, Any]] = {}

    for row in rows:
        name = _clean_text(
            row.get("canonical_name")
            or row.get("canonicalName")
            or row.get("name")
            or row.get("entity_id")
        )
        if not name:
            continue
        source = _concept_source_from_row(dict(row or {}))
        if source == "title_fallback":
            normalized_name = _normalize_title_concept(name)
            if not normalized_name:
                continue
            name = normalized_name
        key = name.casefold()
        confidence = _concept_confidence(source, row.get("confidence"))
        entry = grouped.setdefault(
            key,
            {
                "name": name,
                "sources": set(),
                "maxConfidence": 0.0,
                "hasConceptNote": _has_concept_note(khub.config, name),
            },
        )
        entry["sources"].add(source)
        entry["maxConfidence"] = max(float(entry["maxConfidence"]), confidence)
        entry["hasConceptNote"] = bool(entry["hasConceptNote"] or _has_concept_note(khub.config, name))

    for name in _clean_list(memory_card.get("conceptLinks"), limit=limit):
        normalized_name = _normalize_title_concept(name)
        if not normalized_name:
            continue
        name = normalized_name
        key = name.casefold()
        existing = grouped.get(key)
        if existing and any(
            _concept_source_rank(str(source)) > _concept_source_rank("title_fallback")
            for source in (existing.get("sources") or set())
        ):
            continue
        entry = grouped.setdefault(
            key,
            {
                "name": name,
                "sources": set(),
                "maxConfidence": 0.0,
                "hasConceptNote": _has_concept_note(khub.config, name),
            },
        )
        entry["sources"].add("title_fallback")
        entry["maxConfidence"] = max(float(entry["maxConfidence"]), 0.45)

    details: list[dict[str, Any]] = []
    for entry in grouped.values():
        sources = set(entry.get("sources") or set())
        source = max(sources or {"title_fallback"}, key=_concept_source_rank)
        confidence = round(float(entry.get("maxConfidence") or 0.45), 2)
        if sources == {"title_fallback"}:
            band = "heuristic"
        elif "title_fallback" in sources or confidence < 0.7:
            band = "mixed"
        else:
            band = "verified"
        details.append(
            {
                "name": str(entry.get("name") or ""),
                "source": source,
                "confidence": confidence,
                "band": band,
                "hasConceptNote": bool(entry.get("hasConceptNote")),
            }
        )

    details.sort(
        key=lambda item: (
            -_concept_source_rank(str(item.get("source") or "")),
            -float(item.get("confidence") or 0.0),
            str(item.get("name") or "").casefold(),
        )
    )
    return details[:limit]


def _slot_status(value: Any) -> str:
    token = _clean_text(value)
    if not token:
        return "missing"
    if _usable_public_text(token):
        return "ok"
    return "weak"


def _source_repair_warning(warning: str) -> bool:
    token = _clean_text(warning).casefold()
    if not token:
        return False
    return token == "likely_semantic_mismatch" or (
        ("source" in token or "parser" in token)
        and ("repair" in token or "unusable" in token or "required" in token)
    )


def _build_paper_quality(
    *,
    summary_state: dict[str, Any],
    memory_state: dict[str, Any],
    summary_payload: dict[str, Any],
    memory_card: dict[str, Any],
    concepts_detailed: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    summary_status = _clean_text(summary_state.get("status")) or "missing"
    memory_status = _clean_text(memory_state.get("status")) or "missing"
    slot_status = {
        slot_name: _slot_status(memory_card.get(field_name))
        for slot_name, field_name in _SLOT_FIELD_MAP.items()
    }
    base_score = (
        0.25 * _status_score(summary_status)
        + 0.25 * _status_score(memory_status)
        + sum(
            weight * (100 if slot_status.get(slot_name) == "ok" else 0)
            for slot_name, weight in _SLOT_SCORE_WEIGHTS.items()
        )
    )
    penalty = 0
    reasons: list[str] = []
    if bool(summary_payload.get("fallbackUsed")):
        penalty += _PENALTY_CODES["fallback_used"]
        _append_unique(reasons, "fallback_used")

    for warning in warnings:
        token = _clean_text(warning)
        if not token:
            continue
        if token in _PENALTY_CODES:
            penalty += _PENALTY_CODES[token]
            _append_unique(reasons, token)
            continue
        if _source_repair_warning(token):
            penalty += 20
            _append_unique(reasons, token)

    if summary_status != "ok":
        _append_unique(reasons, f"summary_{summary_status}")
    if memory_status != "ok":
        _append_unique(reasons, f"memory_{memory_status}")
    for slot_name, status in slot_status.items():
        if status != "ok":
            snake = re.sub(r"([a-z])([A-Z])", r"\1_\2", slot_name).lower()
            _append_unique(reasons, f"{snake}_{status}")

    score = max(0, min(100, int(round(base_score - penalty))))
    band = _quality_band(score)
    has_heuristic_concepts = any(
        str(item.get("source") or "") == "title_fallback" or str(item.get("band") or "") == "heuristic"
        for item in concepts_detailed
    )
    return {
        "band": band,
        "score": score,
        "summaryStatus": summary_status,
        "memoryStatus": memory_status,
        "slotStatus": slot_status,
        "reasons": reasons,
        "displayFlags": {
            "hasFallbackSummary": bool(summary_payload.get("fallbackUsed")),
            "hasHeuristicConcepts": has_heuristic_concepts,
            "needsReview": band in {"weak", "degraded"} or _clean_text(memory_card.get("qualityFlag")).lower() not in {"", "ok"},
        },
    }


def _build_related_papers_read_only(
    khub,
    *,
    paper_id: str,
    paper: dict[str, Any],
    summary_payload: dict[str, Any],
    memory_card: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    sqlite_db = khub.sqlite_db()
    title = _clean_text(paper.get("title") or summary_payload.get("paperTitle") or paper_id)
    field = _clean_text(paper.get("field"))
    concepts = _top_concepts(sqlite_db, paper_id=paper_id, memory_card=memory_card, limit=max(2, top_k))
    terms = _search_terms(title, field, *concepts, limit=6)

    candidates: list[dict[str, Any]] = []
    if field:
        candidates.extend(sqlite_db.list_papers(field=field, limit=max(10, top_k * 4)))
    for term in terms:
        candidates.extend(sqlite_db.search_papers(term, limit=max(4, top_k * 2)))
    if memory_card:
        query = " ".join(concepts[:2] or terms[:2] or [title])
        for row in PaperMemoryRetriever(sqlite_db).search(query, limit=max(4, top_k * 2), include_refs=False):
            target_id = _clean_text(row.get("paperId"))
            if not target_id:
                continue
            target_paper = sqlite_db.get_paper(target_id) or {}
            if target_paper:
                candidates.append(target_paper)

    rows = _dedupe_by_key(
        [
            {
                "paperId": _clean_text(item.get("arxiv_id")),
                "title": _clean_text(item.get("title")),
                "year": item.get("year"),
                "field": _clean_text(item.get("field")),
            }
            for item in candidates
            if _clean_text(item.get("arxiv_id")) and _clean_text(item.get("arxiv_id")) != _clean_text(paper_id)
        ],
        key="paperId",
    )
    rows.sort(
        key=lambda item: (
            -int(item.get("year") or 0),
            item.get("title") or "",
        )
    )
    return rows[:top_k]


def _public_related_knowledge(khub, *, paper_id: str, paper_title: str, top_k: int) -> dict[str, list[dict[str, Any]]]:
    searcher = getattr(khub, "searcher", None)
    if searcher is None:
        return {"papers": [], "concepts": [], "notes": [], "web": []}
    results = searcher().search(
        paper_title or paper_id,
        top_k=max(3, int(top_k) * 2),
        retrieval_mode="hybrid",
        alpha=0.7,
        expand_parent_context=True,
    )
    grouped = {"papers": [], "concepts": [], "notes": [], "web": []}
    for result in results:
        source_type = _clean_text(result.metadata.get("source_type"))
        title = _clean_text(result.metadata.get("title") or "Untitled")
        document_id = _clean_text(result.document_id)
        if document_id == f"paper:{paper_id}" or title == paper_title:
            continue
        item = {
            "title": title,
            "sourceType": source_type,
            "score": float(result.score),
            "documentId": document_id,
        }
        if source_type == "paper":
            grouped["papers"].append(item)
        elif source_type == "concept":
            grouped["concepts"].append(item)
        elif source_type == "vault":
            grouped["notes"].append(item)
        elif source_type == "web":
            grouped["web"].append(item)
    return {key: value[:top_k] for key, value in grouped.items()}


def build_public_board_export(
    khub,
    *,
    field: str | None = None,
    limit: int = 50,
    concept_limit: int = 4,
    related_limit: int = 4,
) -> dict[str, Any]:
    sqlite_db = khub.sqlite_db()
    papers = sqlite_db.list_papers(field=field, limit=max(1, int(limit)))

    items: list[dict[str, Any]] = []
    top_warnings: list[str] = []
    summary_count = 0
    memory_count = 0
    indexed_count = 0
    quality_bands = {band: 0 for band in _QUALITY_BANDS}
    concept_bands = {band: 0 for band in _CONCEPT_BANDS}

    for paper in papers:
        paper_id = _clean_text(paper.get("arxiv_id"))
        summary_state = _inspect_public_summary_artifact(khub, paper_id=paper_id)
        summary_payload = dict(summary_state.get("payload") or {})
        memory_state = _inspect_public_memory_card(khub, paper_id=paper_id, include_refs=False, summary_state=summary_state)
        memory_card = dict(memory_state.get("card") or {})
        summary = dict(summary_payload.get("summary") or {})
        claim_coverage = _claim_coverage_from_summary(summary_payload)
        concepts_detailed = _build_concepts_detailed(
            khub,
            paper_id=paper_id,
            memory_card=memory_card,
            limit=max(1, int(concept_limit)),
        )
        concepts = [str(item.get("name") or "") for item in concepts_detailed if _clean_text(item.get("name"))]
        has_summary = bool(summary_payload) and bool(
            _usable_public_text(summary.get("oneLine")) or _usable_public_text(summary.get("coreIdea"))
        )
        has_memory = bool(memory_card) and bool(
            _usable_public_text(memory_card.get("paperCore"))
            or _usable_public_text(memory_card.get("methodCore"))
            or _usable_public_text(memory_card.get("evidenceCore"))
        )
        warnings: list[str] = list(summary_state.get("warnings") or [])
        warnings = _append_unique(warnings, *list(memory_state.get("warnings") or []))
        if not concepts_detailed:
            warnings = _append_unique(warnings, "concept_links_missing")

        related_papers = _build_related_papers_read_only(
            khub,
            paper_id=paper_id,
            paper=paper,
            summary_payload=summary_payload,
            memory_card=memory_card,
            top_k=max(1, int(related_limit)),
        )
        if not related_papers:
            warnings = _append_unique(warnings, "related_papers_sparse")

        quality = _build_paper_quality(
            summary_state=summary_state,
            memory_state=memory_state,
            summary_payload=summary_payload,
            memory_card=memory_card,
            concepts_detailed=concepts_detailed,
            warnings=warnings,
        )
        quality_bands[str(quality.get("band") or "degraded")] += 1
        for concept in concepts_detailed:
            concept_bands[str(concept.get("band") or "heuristic")] += 1

        if has_summary:
            summary_count += 1
        if has_memory:
            memory_count += 1
        if bool(paper.get("indexed")):
            indexed_count += 1

        items.append(
            {
                "paperId": paper_id,
                "title": _clean_text(paper.get("title")),
                "year": paper.get("year"),
                "field": _clean_text(paper.get("field")),
                "artifactFlags": {
                    "hasPdf": bool(_clean_text(paper.get("pdf_path"))),
                    "hasSummary": has_summary,
                    "hasTranslation": bool(_clean_text(paper.get("translated_path"))),
                    "isIndexed": bool(paper.get("indexed")),
                    "hasMemory": has_memory,
                },
                "artifactStatus": {
                    "summary": summary_state.get("status") or "missing",
                    "memory": memory_state.get("status") or "missing",
                },
                "summary": {
                    "oneLine": _clean_text(summary.get("oneLine")),
                    "coreIdea": _clean_text(summary.get("coreIdea")),
                },
                "memory": {
                    "paperCore": _clean_text(memory_card.get("paperCore")),
                    "methodCore": _clean_text(memory_card.get("methodCore")),
                    "evidenceCore": _clean_text(memory_card.get("evidenceCore")),
                    "qualityFlag": _clean_text(memory_card.get("qualityFlag")) or "missing",
                },
                "concepts": concepts,
                "conceptsDetailed": concepts_detailed,
                "quality": quality,
                "relatedPapers": related_papers,
                "paths": {
                    "pdfPath": _clean_text(paper.get("pdf_path")),
                    "translatedPath": _clean_text(paper.get("translated_path")),
                },
                "warnings": warnings,
                "claimCoverage": claim_coverage,
            }
        )

    payload = {
        "schema": "knowledge-hub.paper.board-export.v1",
        "status": "ok" if items else "empty",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "source": {
            "mode": "read_only",
            "field": _clean_text(field),
            "limit": max(1, int(limit)),
            "conceptLimit": max(1, int(concept_limit)),
            "relatedLimit": max(1, int(related_limit)),
        },
        "papers": items,
        "stats": {
            "returnedPapers": len(items),
            "papersWithSummary": summary_count,
            "papersWithMemory": memory_count,
            "indexedPapers": indexed_count,
            "qualityBands": quality_bands,
            "conceptBands": concept_bands,
        },
        "warnings": top_warnings,
    }
    return payload


def build_public_related_card(khub, *, paper_id: str, top_k: int = 5) -> dict[str, Any]:
    sqlite_db = khub.sqlite_db()
    paper = sqlite_db.get_paper(paper_id) or {}
    summary_state = _inspect_public_summary_artifact(khub, paper_id=paper_id)
    memory_state = _inspect_public_memory_card(khub, paper_id=paper_id, include_refs=True, summary_state=summary_state)
    summary = dict(summary_state.get("payload") or {})
    memory_card = dict(memory_state.get("card") or {})
    concepts_detailed = _build_concepts_detailed(khub, paper_id=paper_id, memory_card=memory_card, limit=max(4, int(top_k)))
    warnings = _append_unique(list(summary_state.get("warnings") or []), *list(memory_state.get("warnings") or []))
    if not concepts_detailed:
        warnings = _append_unique(warnings, "concept_links_missing")
    quality = _build_paper_quality(
        summary_state=summary_state,
        memory_state=memory_state,
        summary_payload=summary,
        memory_card=memory_card,
        concepts_detailed=concepts_detailed,
        warnings=warnings,
    )
    related_groups = _public_related_knowledge(
        khub,
        paper_id=paper_id,
        paper_title=_clean_text(paper.get("title") or summary_state.get("paperTitle") or paper_id),
        top_k=max(1, int(top_k)),
    )
    related_knowledge = [
        {**dict(item or {}), "group": group}
        for group, items in related_groups.items()
        for item in list(items or [])
    ]
    return {
        "schema": "knowledge-hub.paper.public.related.v1",
        "status": "ok" if related_knowledge else _combine_status(summary_state.get("status") or "missing", memory_state.get("status") or "missing"),
        "paperId": paper_id,
        "paperTitle": _clean_text(paper.get("title") or summary_state.get("paperTitle") or paper_id),
        "query": _clean_text(summary_state.get("paperTitle") or paper_id),
        "relatedKnowledge": related_knowledge,
        "conceptsDetailed": concepts_detailed,
        "quality": quality,
        "claimCoverage": _claim_coverage_from_summary(summary),
        "warnings": _append_unique(
            warnings,
            "related_knowledge_missing" if not related_knowledge else "",
        ),
        "artifactStatus": {
            "summary": summary_state.get("status") or "missing",
            "memory": memory_state.get("status") or "missing",
        },
    }


__all__ = [
    "build_public_board_export",
    "build_public_evidence_card",
    "build_public_memory_card",
    "build_public_related_card",
    "build_public_summary_card",
    "ensure_public_memory_card",
    "ensure_public_summary",
    "load_public_memory_card",
    "load_public_summary_artifact",
]
