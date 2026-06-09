from __future__ import annotations

from pathlib import Path
from typing import Final, Mapping
import hashlib
import json

PAPER_REAL_ANSWER_QUALITY_GATE_SCHEMA_ID: Final = "knowledge-hub.paper-real-answer-quality-gate.v1"
PRIVATE_PATH_MARKERS: Final = ("/Users/", "Mobile Documents", "iCloud")
FORBIDDEN_MARKERS: Final = ("rawPrompt", "paper-card-v2", "card_id", "source_card_id")
INSUFFICIENT_MARKERS: Final = ("insufficient_evidence", "not_applicable", "no paper evidence", "no source evidence")

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
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return any(marker in serialized for marker in markers)


def _redacted_preview(value: str, *, limit: int = 180) -> str:
    tokens = ["[redacted-local-path]" if any(marker in token for marker in PRIVATE_PATH_MARKERS) else token for token in value.split()]
    return " ".join(tokens)[:limit]


def _row_kind(case_id: str) -> str:
    if case_id.startswith("compare"):
        return "compare"
    if case_id.startswith("synthesis"):
        return "synthesis"
    if case_id.startswith("abstain"):
        return "abstain"
    return "single_paper"


def _answer_type(row: JsonMap) -> str:
    value = _clean(row.get("answerType"), limit=40)
    match value:
        case "single_paper" | "synthesis" | "compare" | "abstain":
            return value
        case _:
            return _row_kind(_clean(row.get("caseId"), limit=120))


def _citations(row: JsonMap) -> list[JsonMap]:
    return _as_maps(row.get("citations"))


def _citation_labels(citations: list[JsonMap]) -> list[str]:
    labels: list[str] = []
    for citation in citations:
        label = _clean(citation.get("label"), limit=40)
        if label and label not in labels:
            labels.append(label)
    return labels


def _citation_source_ids(citations: list[JsonMap]) -> list[str]:
    source_ids: list[str] = []
    for citation in citations:
        source_id = _clean(citation.get("sourceId"), limit=120)
        if source_id and source_id not in source_ids:
            source_ids.append(source_id)
    return source_ids


def _claim_citation_maps(row: JsonMap) -> list[JsonMap]:
    return _as_maps(row.get("claimCitationMap"))


def _semantic_supported(row: JsonMap, citation_labels: list[str]) -> bool:
    claim_maps = _claim_citation_maps(row)
    if not claim_maps:
        return False
    for claim in claim_maps:
        labels = _as_strings(claim.get("citationLabels"))
        if claim.get("supported") is not True or not labels:
            return False
        if any(label not in citation_labels for label in labels):
            return False
    return True


def _insufficient_answer(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in INSUFFICIENT_MARKERS)


def _compare_review(row: JsonMap) -> tuple[bool, bool]:
    review = row.get("compareReview")
    payload = review if isinstance(review, dict) else {}
    return payload.get("meaningfulSynthesis") is True, payload.get("restatementOnly") is True


def _row_report(row: JsonMap) -> JsonMap:
    run_id = _clean(row.get("runId"), limit=160)
    case_id = _clean(row.get("caseId"), limit=120)
    variant_id = _clean(row.get("variantId"), limit=120)
    answer_type = _answer_type(row)
    answer = _clean(row.get("answerText"), limit=1200)
    expected_source_ids = _as_strings(row.get("expectedSourceIds"))
    citations = _citations(row)
    citation_labels = _citation_labels(citations)
    citation_source_ids = _citation_source_ids(citations)
    is_abstain = answer_type == "abstain" or row.get("expectedInsufficientEvidence") is True
    insufficient = _insufficient_answer(answer)
    answerable = not is_abstain and bool(expected_source_ids)
    inline_ready = answerable and bool(citation_labels) and all(f"[{label}]" in answer for label in citation_labels)
    semantic_supported = answerable and inline_ready and _semantic_supported(row, citation_labels)
    compare_meaningful, compare_restatement = _compare_review(row)
    two_sided = answer_type == "compare" and len(set(citation_source_ids) & set(expected_source_ids)) >= 2
    private_leak = _contains_marker(row, PRIVATE_PATH_MARKERS)
    forbidden_marker = _contains_marker(row, FORBIDDEN_MARKERS)
    raw_prompt_present = "rawPrompt" in row
    unsupported_invention = row.get("unsupportedInvention") is True
    warnings: list[str] = []
    if private_leak:
        warnings.append("private_path_leak")
    if forbidden_marker:
        warnings.append("forbidden_raw_marker")
    if raw_prompt_present:
        warnings.append("raw_prompt_present")
    if unsupported_invention:
        warnings.append("unsupported_invention")
    if is_abstain and not insufficient:
        warnings.append("unexpected_answer_for_abstain")
    if answerable and not answer:
        warnings.append("missing_answer")
    if answerable and insufficient:
        warnings.append("unexpected_insufficient_evidence")
    if answerable and not inline_ready:
        warnings.append("inline_citation_missing")
    if answerable and not semantic_supported:
        warnings.append("semantic_citation_support_failed")
    if answer_type == "compare" and not two_sided:
        warnings.append("compare_two_sided_citation_missing")
    if answer_type == "compare" and compare_restatement:
        warnings.append("compare_restatement_only")
    if answer_type == "compare" and not compare_meaningful:
        warnings.append("compare_meaningful_synthesis_missing")
    ready = answerable and not insufficient and inline_ready and semantic_supported and not private_leak and not forbidden_marker and not unsupported_invention
    if answer_type == "compare":
        ready = ready and two_sided and compare_meaningful and not compare_restatement
    status = "ready" if ready else "not_applicable" if is_abstain and insufficient and not private_leak and not forbidden_marker else "blocked"
    review_status = "usable" if status == "ready" else "insufficient" if status == "not_applicable" else "failed"
    return {
        "runId": run_id,
        "caseId": case_id,
        "variantId": variant_id,
        "answerType": answer_type,
        "status": status,
        "draftStatus": "insufficient_evidence" if is_abstain else "answerable",
        "expectedSourceIds": expected_source_ids,
        "citationLabels": citation_labels,
        "citationSourceIds": citation_source_ids,
        "answerHash": _sha256_text(answer),
        "answerPreview": _redacted_preview(answer),
        "inlineCitationReady": inline_ready,
        "semanticCitationSupported": semantic_supported,
        "twoSidedCompareCitationReady": two_sided,
        "compareMeaningfulSynthesis": compare_meaningful if answer_type == "compare" else True,
        "sampleDraft": {"stored": "hash_and_sanitized_preview", "hash": _sha256_text(answer)},
        "humanReadableReview": {"status": review_status, "reason": ";".join(warnings) or "grounded"},
        "warnings": warnings,
    }


def _metadata(report: JsonMap) -> JsonMap:
    value = report.get("runMetadata")
    return value if isinstance(value, dict) else {}


def _counts(rows: list[JsonMap], metadata: Mapping[str, JsonValue]) -> JsonMap:
    ready = [row for row in rows if row["status"] == "ready"]
    not_applicable = [row for row in rows if row["status"] == "not_applicable"]
    blocked = [row for row in rows if row["status"] == "blocked"]
    external_calls = 1 if metadata.get("allowExternal") is True else 0
    return {
        "rowCount": len(rows),
        "answerableRows": sum(1 for row in rows if row["draftStatus"] == "answerable"),
        "readyRows": len(ready),
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
        "unexpectedAnswerRows": sum(1 for row in rows if "unexpected_answer_for_abstain" in row["warnings"]),
        "unsupportedInventionRows": sum(1 for row in rows if "unsupported_invention" in row["warnings"]),
        "sampleDraftRows": sum(1 for row in rows if "sampleDraft" in row),
        "humanReadableReviewRows": sum(1 for row in rows if "humanReadableReview" in row),
        "privatePathLeakRows": sum(1 for row in rows if "private_path_leak" in row["warnings"]),
        "forbiddenRawMarkerRows": sum(1 for row in rows if "forbidden_raw_marker" in row["warnings"]),
        "rawPromptPersistedRows": sum(1 for row in rows if "raw_prompt_present" in row["warnings"]),
        "rawAnswerPersistedRows": sum(1 for row in rows if "answerText" in row),
        "externalModelCallRows": external_calls,
        "modelApiCallRows": external_calls,
        "dbVectorMutationRows": 0,
        "vaultReadRows": 0,
        "defaultPromotionRows": 0,
        "schemaViolationCount": 0,
    }


def build_paper_real_answer_quality_gate(*, answer_payload_report: JsonMap, generated_at: str) -> JsonMap:
    rows = [_row_report(row) for row in _as_maps(answer_payload_report.get("rows"))]
    metadata = _metadata(answer_payload_report)
    counts = _counts(rows, metadata)
    metadata_private_leak = _contains_marker(metadata, PRIVATE_PATH_MARKERS)
    blocked = bool(counts["blockedRows"] or counts["privatePathLeakRows"] or counts["forbiddenRawMarkerRows"] or metadata_private_leak)
    warnings = ["metadata_private_path_leak"] if metadata_private_leak else []
    if blocked:
        warnings.append("paper_real_answer_quality_gate_blocked")
    return {
        "schema": PAPER_REAL_ANSWER_QUALITY_GATE_SCHEMA_ID,
        "status": "blocked" if blocked else "ready",
        "generatedAt": generated_at,
        "profile": "paper-real-answer-quality-gate",
        "querySet": _clean(metadata.get("querySet"), limit=120) or "bounded-real-answer-quality-v0",
        "collector": _clean(metadata.get("collector"), limit=120) or "unknown",
        "runDirectory": _redacted_preview(_clean(metadata.get("runDirectory"), limit=240)) or "unknown",
        "answerRoute": _clean(metadata.get("answerRoute"), limit=80) or "unknown",
        "allowExternal": metadata.get("allowExternal") is True,
        "policy": {
            "reportOnly": True,
            "consumesRealAnswerPayloads": True,
            "localFakeLlmOnly": False,
            "externalModelCallsAllowed": metadata.get("allowExternal") is True,
            "dbVectorMutation": False,
            "vaultRead": False,
            "publicDefaultPromotionAllowed": False,
            "qwenDefaultFallbackAllowed": False,
            "rawPromptPersistenceAllowed": False,
        },
        "counts": counts,
        "rows": rows,
        "warnings": warnings,
    }


def write_paper_real_answer_quality_gate(report: JsonMap, *, report_json: Path) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"json": str(report_json)}


__all__ = [
    "PAPER_REAL_ANSWER_QUALITY_GATE_SCHEMA_ID",
    "build_paper_real_answer_quality_gate",
    "write_paper_real_answer_quality_gate",
]
