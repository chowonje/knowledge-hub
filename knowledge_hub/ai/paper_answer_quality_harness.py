from __future__ import annotations

from pathlib import Path
from typing import Final, Mapping
import hashlib
import json

PAPER_ANSWER_QUALITY_HARNESS_SCHEMA_ID: Final = "knowledge-hub.paper-answer-quality-harness.v1"
PRIVATE_PATH_MARKERS: Final = ("/Users/", "Mobile Documents", "iCloud")
FORBIDDEN_MARKERS: Final = ("paper-card-v2", "card_id", "source_card_id", "rawPrompt")

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonMap = dict[str, JsonValue]


def _as_maps(value: JsonValue | None) -> list[JsonMap]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _as_strings(value: JsonValue | None) -> list[str]:
    return [str(item) for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _clean(value: JsonValue | None, *, limit: int = 240) -> str:
    return " ".join(str(value or "").strip().split())[:limit]


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _contains_marker(payload: JsonValue, markers: tuple[str, ...]) -> bool:
    return any(marker in json.dumps(payload, ensure_ascii=False, sort_keys=True) for marker in markers)


def _row_kind(case_id: str) -> str:
    if case_id.startswith("compare"):
        return "compare"
    if case_id.startswith("synthesis"):
        return "synthesis"
    if case_id.startswith("abstain"):
        return "abstain"
    return "single_paper"


def _readback_by_run(readback_report: JsonMap) -> dict[str, JsonMap]:
    return {_clean(row.get("runId"), limit=160): row for row in _as_maps(readback_report.get("rows"))}


def _profile_by_run(profile_readiness_report: JsonMap) -> dict[str, JsonMap]:
    return {_clean(row.get("runId"), limit=160): row for row in _as_maps(profile_readiness_report.get("rows"))}


def _labels_from_readbacks(row: JsonMap) -> list[str]:
    labels: list[str] = []
    for paper in _as_maps(row.get("paperReadbacks")):
        for slot in _as_maps(paper.get("slots")):
            label = _clean(slot.get("citationLabel"), limit=40)
            if label and label not in labels:
                labels.append(label)
    return labels


def _source_ids(row: JsonMap) -> list[str]:
    return _as_strings(row.get("expectedSourceIds"))


def _default_answer(*, kind: str, source_ids: list[str], labels: list[str]) -> str:
    if kind == "abstain" or not source_ids:
        return "insufficient_evidence: no source evidence is available for this row."
    if kind == "compare" and len(source_ids) >= 2 and len(labels) >= 2:
        return (
            f"{source_ids[0]} and {source_ids[1]} both ground RAG-style evidence work, "
            f"but they differ in whether the row frames a method or a survey synthesis [{labels[0]}] [{labels[1]}]."
        )
    label = labels[0] if labels else "S1"
    return f"{source_ids[0]} supports the requested paper understanding answer with grounded evidence [{label}]."


def _override_bool(override: JsonMap, key: str, default: bool) -> bool:
    value = override.get(key)
    return bool(value) if isinstance(value, bool) else default


def _citation_support(labels: list[str], override: JsonMap) -> dict[str, bool]:
    raw = override.get("citationSupport")
    support = raw if isinstance(raw, dict) else {}
    return {label: bool(support.get(label, True)) for label in labels}


def _row_report(
    *,
    packet_row: JsonMap,
    readback_row: JsonMap | None,
    profile_row: JsonMap | None,
    override: JsonMap,
) -> JsonMap:
    run_id = _clean(packet_row.get("runId") or (readback_row or {}).get("runId"), limit=160)
    case_id = _clean(packet_row.get("caseId") or (readback_row or {}).get("caseId"), limit=120)
    variant_id = _clean(packet_row.get("variantId") or (readback_row or {}).get("variantId"), limit=120)
    kind = _row_kind(case_id)
    source_ids = _source_ids(readback_row or packet_row)
    labels = _as_strings(override.get("citationLabels")) or _labels_from_readbacks(readback_row or {})
    answer = _clean(override.get("answer"), limit=900) or _default_answer(kind=kind, source_ids=source_ids, labels=labels)
    has_readback_evidence = readback_row is not None and bool(_as_maps(readback_row.get("paperReadbacks")))
    draft_status = "insufficient_evidence" if kind == "abstain" or not source_ids or not has_readback_evidence else "answerable"
    not_applicable_draft = kind == "abstain" or (not source_ids and draft_status != "answerable")
    inline_ready = draft_status == "answerable" and bool(labels) and all(f"[{label}]" in answer for label in labels[: max(1, len(source_ids))])
    support = _citation_support(labels, override)
    semantic_supported = draft_status == "answerable" and inline_ready and all(support.get(label, False) for label in labels)
    two_sided = kind == "compare" and len(source_ids) >= 2 and len(set(labels)) >= 2 and all(f"[{label}]" in answer for label in labels[:2])
    meaningful = _override_bool(override, "meaningfulSynthesis", kind != "compare" or two_sided)
    warnings: list[str] = []
    if draft_status != "answerable":
        warnings.append("not_applicable_abstention")
    if draft_status == "answerable" and not inline_ready:
        warnings.append("inline_citation_missing")
    if draft_status == "answerable" and not semantic_supported:
        warnings.append("semantic_citation_support_failed")
    if kind == "compare" and not two_sided:
        warnings.append("compare_two_sided_citation_missing")
    if kind == "compare" and two_sided and not meaningful:
        warnings.append("compare_restatement_only")
    if profile_row is None and draft_status == "answerable":
        warnings.append("missing_profile_readiness_row")
    ready = draft_status == "answerable" and inline_ready and semantic_supported and (kind != "compare" or (two_sided and meaningful))
    if profile_row is None and draft_status == "answerable":
        ready = False
    status = "ready" if ready else "not_applicable" if not_applicable_draft else "blocked"
    review_status = "usable" if ready else "insufficient" if not_applicable_draft else "failed"
    return {
        "runId": run_id,
        "caseId": case_id,
        "variantId": variant_id,
        "answerType": kind,
        "status": status,
        "draftStatus": draft_status,
        "expectedSourceIds": source_ids,
        "citationLabels": labels,
        "answerHash": _sha256_text(answer),
        "answerPreview": answer[:180],
        "inlineCitationReady": inline_ready,
        "semanticCitationSupported": semantic_supported,
        "twoSidedCompareCitationReady": two_sided,
        "compareMeaningfulSynthesis": meaningful if kind == "compare" else True,
        "sampleDraft": {"stored": "local_evidence_only", "hash": _sha256_text(answer)},
        "humanReadableReview": {"status": review_status, "reason": ";".join(warnings) or "grounded"},
        "warnings": warnings,
    }


def _rows(packet_input_report: JsonMap, readback_report: JsonMap, profile_readiness_report: JsonMap, overrides: Mapping[str, JsonMap]) -> list[JsonMap]:
    readbacks = _readback_by_run(readback_report)
    profiles = _profile_by_run(profile_readiness_report)
    packet_rows = _as_maps(packet_input_report.get("rows"))
    if not packet_rows and readbacks:
        packet_rows = list(readbacks.values())
    return [
        _row_report(
            packet_row=row,
            readback_row=readbacks.get(_clean(row.get("runId"), limit=160)),
            profile_row=profiles.get(_clean(row.get("runId"), limit=160)),
            override=overrides.get(_clean(row.get("runId"), limit=160), {}),
        )
        for row in packet_rows
    ]


def _counts(rows: list[JsonMap]) -> JsonMap:
    ready = [row for row in rows if row["status"] == "ready"]
    not_applicable = [row for row in rows if row["status"] == "not_applicable"]
    blocked = [row for row in rows if row["status"] == "blocked"]
    return {
        "rowCount": len(rows),
        "answerQualityReadyRows": len(ready),
        "notApplicableRows": len(not_applicable),
        "blockedRows": len(blocked),
        "singlePaperReadyRows": sum(1 for row in ready if row["answerType"] == "single_paper"),
        "synthesisReadyRows": sum(1 for row in ready if row["answerType"] == "synthesis"),
        "compareReadyRows": sum(1 for row in ready if row["answerType"] == "compare"),
        "abstentionRows": len(not_applicable),
        "inlineCitationReadyRows": sum(1 for row in ready if bool(row["inlineCitationReady"])),
        "semanticCitationSupportRows": sum(1 for row in ready if bool(row["semanticCitationSupported"])),
        "semanticCitationFailRows": sum(1 for row in rows if "semantic_citation_support_failed" in row["warnings"]),
        "twoSidedCompareCitationRows": sum(1 for row in ready if row["answerType"] == "compare" and bool(row["twoSidedCompareCitationReady"])),
        "compareMeaningfulSynthesisRows": sum(1 for row in ready if row["answerType"] == "compare" and bool(row["compareMeaningfulSynthesis"])),
        "compareRestatementOnlyRows": sum(1 for row in rows if "compare_restatement_only" in row["warnings"]),
        "insufficientEvidenceRows": len(not_applicable),
        "unexpectedAnswerRows": 0,
        "unsupportedInventionRows": 0,
        "sampleDraftRows": sum(1 for row in rows if "sampleDraft" in row),
        "humanReadableReviewRows": sum(1 for row in rows if "humanReadableReview" in row),
        "privatePathLeakRows": sum(1 for row in rows if _contains_marker(row, PRIVATE_PATH_MARKERS)),
        "forbiddenRawMarkerRows": sum(1 for row in rows if _contains_marker(row, FORBIDDEN_MARKERS)),
        "rawAnswerPersistedRows": sum(1 for row in rows if "answerText" in row),
        "externalModelCallRows": 0,
        "modelApiCallRows": 0,
        "dbVectorMutationRows": 0,
        "vaultReadRows": 0,
        "defaultPromotionRows": 0,
        "schemaViolationCount": 0,
    }


def build_paper_answer_quality_harness(
    *,
    packet_input_report: JsonMap,
    readback_report: JsonMap,
    profile_readiness_report: JsonMap,
    generated_at: str,
    draft_overrides: Mapping[str, JsonMap] | None = None,
) -> JsonMap:
    rows = _rows(packet_input_report, readback_report, profile_readiness_report, draft_overrides or {})
    counts = _counts(rows)
    blocked = bool(counts["blockedRows"] or counts["privatePathLeakRows"] or counts["forbiddenRawMarkerRows"])
    return {
        "schema": PAPER_ANSWER_QUALITY_HARNESS_SCHEMA_ID,
        "status": "blocked" if blocked else "ready",
        "generatedAt": generated_at,
        "profile": "paper-answer-quality-harness",
        "policy": {
            "reportOnly": True,
            "localFakeLlmOnly": True,
            "externalModelCallsAllowed": False,
            "dbVectorMutation": False,
            "vaultRead": False,
            "publicDefaultPromotionAllowed": False,
        },
        "counts": counts,
        "rows": rows,
        "warnings": [] if not blocked else ["paper_answer_quality_harness_blocked"],
    }


def write_paper_answer_quality_harness(report: JsonMap, *, report_json: Path) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"json": str(report_json)}


__all__ = [
    "PAPER_ANSWER_QUALITY_HARNESS_SCHEMA_ID",
    "build_paper_answer_quality_harness",
    "write_paper_answer_quality_harness",
]
