from __future__ import annotations

from pathlib import Path
from typing import Final
import json

PAPER_UNDERSTANDING_PROFILE_READINESS_SCHEMA_ID: Final = (
    "knowledge-hub.paper-understanding-profile-readiness.v1"
)

PRIVATE_PATH_MARKERS: Final = ("/Users/", "Mobile Documents", "iCloud")
FORBIDDEN_MARKERS: Final = ("paper-card-v2", "card_id", "source_card_id")

CORE_PROFILE_SLOTS: Final = ("claim", "method", "evidence")
BRIEF_PROFILE_SLOTS: Final = (*CORE_PROFILE_SLOTS, "limitation", "purpose_relevance")

_SLOT_KEYWORDS: Final[dict[str, tuple[str, ...]]] = {
    "claim": ("argue", "claim", "contribution", "demonstrate", "propose", "show"),
    "method": ("architecture", "approach", "framework", "method", "retrieval", "generation"),
    "evidence": ("benchmark", "evaluation", "experiment", "result", "study"),
    "limitation": ("challenge", "future work", "limitation", "limited", "risk"),
    "purpose_relevance": ("application", "goal", "purpose", "relevant", "task", "useful"),
}

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonMap = dict[str, JsonValue]


def _as_map(value: JsonValue | None) -> JsonMap:
    return value if isinstance(value, dict) else {}


def _as_maps(value: JsonValue | None) -> list[JsonMap]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _as_strings(value: JsonValue | None) -> list[str]:
    return [str(item) for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _clean_text(value: JsonValue | None, *, limit: int = 900) -> str:
    text = " ".join(str(value or "").strip().split())
    return text[:limit]


def _source_id(value: JsonValue | None) -> str:
    text = _clean_text(value, limit=120)
    for prefix in ("paper:", "arxiv:"):
        if text.startswith(prefix):
            return text.removeprefix(prefix)
    return text


def _unique(values: list[str]) -> list[str]:
    return [item for item in dict.fromkeys(value for value in values if value)]


def _contains_marker(payload: JsonValue, markers: tuple[str, ...]) -> bool:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return any(marker in text for marker in markers)


def _has_warning(row: JsonMap, warning: str) -> bool:
    warnings = _as_strings(row.get("warnings"))
    return any(item == warning or item.endswith(f":{warning}") for item in warnings)


def _expected_source_ids(row: JsonMap) -> list[str]:
    prompt_packet = _as_map(row.get("promptPacket"))
    expected = [_source_id(item) for item in _as_strings(prompt_packet.get("expectedSourceIds"))]
    if expected:
        return _unique(expected)
    source_ids = [_source_id(item) for item in _as_strings(row.get("packetSourceIds"))]
    return _unique(source_ids)


def _spans_for_source(row: JsonMap, source_id: str) -> list[JsonMap]:
    prompt_packet = _as_map(row.get("promptPacket"))
    spans = _as_maps(prompt_packet.get("spans"))
    return [span for span in spans if _source_id(span.get("sourceId") or span.get("source_id")) == source_id]


def _slot_counts(spans: list[JsonMap]) -> dict[str, int]:
    counts = {slot: 0 for slot in BRIEF_PROFILE_SLOTS}
    for span in spans:
        text = _clean_text(span.get("text"), limit=1400).lower()
        for slot, keywords in _SLOT_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                counts[slot] += 1
    return counts


def _has_citation(span: JsonMap) -> bool:
    return bool(_clean_text(span.get("citationLabel") or span.get("citation_label"), limit=80))


def _paper_profile(source_id: str, spans: list[JsonMap]) -> JsonMap:
    slot_counts = _slot_counts(spans)
    available_slots = [slot for slot, count in slot_counts.items() if count > 0]
    missing_core_slots = [slot for slot in CORE_PROFILE_SLOTS if slot not in available_slots]
    missing_brief_slots = [slot for slot in BRIEF_PROFILE_SLOTS if slot not in available_slots]
    citation_span_count = sum(1 for span in spans if _has_citation(span))
    warnings: list[str] = []
    if not spans:
        warnings.append("missing_source_spans")
    if spans and citation_span_count == 0:
        warnings.append("missing_citation_span")
    warnings.extend(f"missing_profile_slot:{slot}" for slot in missing_brief_slots)
    core_ready = bool(spans) and citation_span_count > 0 and not missing_core_slots
    brief_ready = bool(spans) and citation_span_count > 0 and not missing_brief_slots
    return {
        "sourceId": source_id,
        "spanCount": len(spans),
        "citationSpanCount": citation_span_count,
        "availableSlots": available_slots,
        "missingCoreSlots": missing_core_slots,
        "missingBriefSlots": missing_brief_slots,
        "coreReady": core_ready,
        "briefReady": brief_ready,
        "warnings": _unique(warnings),
    }


def _row_profile(row: JsonMap) -> JsonMap:
    expected_ids = _expected_source_ids(row)
    paper_profiles = [_paper_profile(source_id, _spans_for_source(row, source_id)) for source_id in expected_ids]
    row_warnings = list(_as_strings(row.get("warnings")))
    if _clean_text(row.get("status"), limit=20) == "blocked":
        row_warnings.append("packet_input_row_blocked")
    for profile in paper_profiles:
        for warning in _as_strings(profile.get("warnings")):
            row_warnings.append(f"{profile.get('sourceId')}:{warning}")
    if not expected_ids:
        row_warnings.append("missing_expected_source_ids")
    status = "ready" if expected_ids and all(bool(profile.get("briefReady")) for profile in paper_profiles) else "blocked"
    return {
        "runId": _clean_text(row.get("runId"), limit=160),
        "caseId": _clean_text(row.get("caseId"), limit=120),
        "variantId": _clean_text(row.get("variantId"), limit=120),
        "status": status,
        "expectedSourceIds": expected_ids,
        "paperProfiles": paper_profiles,
        "readyPaperCount": sum(1 for profile in paper_profiles if bool(profile.get("briefReady"))),
        "blockedPaperCount": sum(1 for profile in paper_profiles if not bool(profile.get("briefReady"))),
        "warnings": _unique(row_warnings),
    }


def _profile_from_readback_paper(paper: JsonMap) -> JsonMap:
    slots = _as_maps(paper.get("slots"))
    ready_slots = [slot for slot in slots if _clean_text(slot.get("status"), limit=20) == "ready"]
    available_slots = _as_strings(paper.get("availableSlots"))
    missing_brief_slots = _as_strings(paper.get("missingBriefSlots"))
    missing_core_slots = [slot for slot in CORE_PROFILE_SLOTS if slot not in available_slots]
    return {
        "sourceId": _source_id(paper.get("sourceId")),
        "spanCount": len(ready_slots),
        "citationSpanCount": sum(1 for slot in ready_slots if bool(_clean_text(slot.get("citationLabel"), limit=80))),
        "availableSlots": available_slots,
        "missingCoreSlots": missing_core_slots,
        "missingBriefSlots": missing_brief_slots,
        "coreReady": bool(paper.get("coreReady")),
        "briefReady": bool(paper.get("briefReady")),
        "warnings": _as_strings(paper.get("warnings")),
    }


def _row_profile_from_readback(row: JsonMap) -> JsonMap:
    paper_profiles = [_profile_from_readback_paper(paper) for paper in _as_maps(row.get("paperReadbacks"))]
    row_status = _clean_text(row.get("status"), limit=40)
    warnings = _as_strings(row.get("warnings"))
    ready_paper_count = sum(1 for profile in paper_profiles if bool(profile.get("briefReady")))
    blocked_paper_count = sum(1 for profile in paper_profiles if not bool(profile.get("briefReady")))
    if row_status == "not_applicable":
        status = "not_applicable"
    elif row_status == "ready" and blocked_paper_count == 0:
        status = "ready"
    else:
        status = "blocked"
    return {
        "runId": _clean_text(row.get("runId"), limit=160),
        "caseId": _clean_text(row.get("caseId"), limit=120),
        "variantId": _clean_text(row.get("variantId"), limit=120),
        "status": status,
        "expectedSourceIds": [_source_id(source_id) for source_id in _as_strings(row.get("expectedSourceIds"))],
        "paperProfiles": paper_profiles,
        "readyPaperCount": ready_paper_count,
        "blockedPaperCount": blocked_paper_count,
        "warnings": warnings,
    }


def build_paper_understanding_profile_readiness(
    *,
    packet_input_report: JsonMap,
    generated_at: str,
    readback_report: JsonMap | None = None,
) -> JsonMap:
    rows = (
        [_row_profile_from_readback(row) for row in _as_maps(readback_report.get("rows"))]
        if readback_report is not None
        else [_row_profile(row) for row in _as_maps(packet_input_report.get("rows"))]
    )
    private_rows = sum(1 for row in rows if _contains_marker(row, PRIVATE_PATH_MARKERS) or _has_warning(row, "private_path_marker"))
    forbidden_rows = sum(1 for row in rows if _contains_marker(row, FORBIDDEN_MARKERS) or _has_warning(row, "forbidden_raw_marker"))
    paper_profiles = [profile for row in rows for profile in _as_maps(row.get("paperProfiles"))]
    top_level_blocked = bool(private_rows or forbidden_rows)
    return {
        "schema": PAPER_UNDERSTANDING_PROFILE_READINESS_SCHEMA_ID,
        "status": "blocked" if top_level_blocked else "ready",
        "generatedAt": generated_at,
        "profile": "paper-compare-brief-understanding-profile-readiness",
        "policy": {
            "reportOnly": True,
            "localOnlyRawArtifactRefs": True,
            "publicDefaultPromotionAllowed": False,
            "runtimePromotionApplied": False,
        },
        "requirements": {
            "coreProfileSlots": list(CORE_PROFILE_SLOTS),
            "briefProfileSlots": list(BRIEF_PROFILE_SLOTS),
            "requiresCitationSpan": True,
        },
        "counts": {
            "rowCount": len(rows),
            "readyRows": sum(1 for row in rows if row["status"] == "ready"),
            "notApplicableRows": sum(1 for row in rows if row["status"] == "not_applicable"),
            "blockedRows": sum(1 for row in rows if row["status"] == "blocked"),
            "paperProfileRows": len(paper_profiles),
            "briefReadyPaperRows": sum(1 for profile in paper_profiles if bool(profile.get("briefReady"))),
            "briefBlockedPaperRows": sum(1 for profile in paper_profiles if not bool(profile.get("briefReady"))),
            "privatePathLeakRows": private_rows,
            "forbiddenRawMarkerRows": forbidden_rows,
        },
        "rows": rows,
        "warnings": [] if not top_level_blocked else ["paper understanding readiness report contains blocked markers"],
    }


def write_paper_understanding_profile_readiness(report: JsonMap, *, report_json: Path) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"json": str(report_json)}


__all__ = [
    "PAPER_UNDERSTANDING_PROFILE_READINESS_SCHEMA_ID",
    "build_paper_understanding_profile_readiness",
    "write_paper_understanding_profile_readiness",
]
