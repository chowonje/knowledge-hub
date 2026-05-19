"""Typed strict-evidence policy gate for applied parsed-artifact SourceSpan rows.

This helper consumes the SourceSpan promotion readback-review report and
classifies rows by artifact-type-specific strict-evidence prerequisites. It
does not create StrictEvidence records, does not mutate SourceSpan rows, and
does not enable runtime/parser/answer/DB/index/vault mutation.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_promotion_readback_review import (
    PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_READBACK_REVIEW_SCHEMA_ID,
    READBACK_STATUS_VALIDATED,
)


PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_POLICY_GATE_V2_TYPED_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-source-span-strict-evidence-policy-gate-v2-typed.v1"
)

POLICY_STATUS_STRICT_TEXT = "strict_text_policy_candidate_only"
POLICY_STATUS_STRICT_CAPTION = "strict_caption_policy_candidate_only"
POLICY_STATUS_STRICT_STRUCTURED = "strict_structured_policy_candidate_only"
POLICY_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"
POLICY_STATUS_BLOCKED_READBACK_NOT_READY = "blocked_readback_not_ready"
POLICY_STATUS_BLOCKED_MISSING_SOURCE_HASH = "blocked_missing_source_hash"
POLICY_STATUS_BLOCKED_MISSING_LOCATOR = "blocked_missing_locator"
POLICY_STATUS_BLOCKED_MISSING_OFFSET_AUTHORITY = "blocked_missing_offset_authority"
POLICY_STATUS_BLOCKED_MISSING_STRUCTURED_AUTHORITY = "blocked_missing_structured_authority"
POLICY_STATUS_BLOCKED_UNSUPPORTED_ARTIFACT_TYPE = "blocked_unsupported_artifact_type"
POLICY_STATUS_BLOCKED_RUNTIME_OR_STRICT_FLAG = "blocked_runtime_or_strict_flag_violation"
POLICY_STATUS_BLOCKED_MISSING_RECORD_IDENTITY = "blocked_missing_record_identity"
POLICY_STATUS_BLOCKED_LOCATOR_BASIS_UNKNOWN = "blocked_locator_basis_unknown"
POLICY_STATUS_BLOCKED_NORMALIZATION_MISMATCH = "blocked_normalization_mismatch"
POLICY_STATUS_BLOCKED_TYPE_AUTHORITY_UNDEFINED = "blocked_type_authority_policy_undefined"

TEXT_ARTIFACT_TYPES = {"section"}
CAPTION_ARTIFACT_TYPES = {"figure"}
STRUCTURED_TABLE_TYPES = {"table"}
STRUCTURED_EQUATION_TYPES = {"equation"}
KNOWN_ARTIFACT_TYPES = TEXT_ARTIFACT_TYPES | CAPTION_ARTIFACT_TYPES | STRUCTURED_TABLE_TYPES | STRUCTURED_EQUATION_TYPES

DEFAULT_READBACK_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-source-span-promotion-readback-review"
    / "01-parsed-artifact-source-span-promotion-readback-review"
    / "parsed-artifact-source-span-promotion-readback-review.json"
)

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-source-span-strict-evidence-policy-gate-v2-typed"
    / "01-parsed-artifact-source-span-strict-evidence-policy-gate-v2-typed"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_bool(value: Any) -> bool:
    return bool(value)


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = _safe_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    payload_path = Path(str(path)).expanduser()
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _locator_dict(row: dict[str, Any]) -> dict[str, Any]:
    locator = row.get("locator")
    return locator if isinstance(locator, dict) else {}


def _field_value(row: dict[str, Any], *keys: str) -> str:
    locator = _locator_dict(row)
    for key in keys:
        if key in row and _safe_text(row.get(key)):
            return _safe_text(row.get(key))
        if key in locator and _safe_text(locator.get(key)):
            return _safe_text(locator.get(key))
    return ""


def _chars_dict(row: dict[str, Any]) -> dict[str, Any]:
    chars = _locator_dict(row).get("chars")
    return chars if isinstance(chars, dict) else {}


def _has_any_locator_signal(row: dict[str, Any]) -> bool:
    locator = _locator_dict(row)
    page = _safe_int(locator.get("page"))
    bbox = _safe_list(locator.get("bbox"))
    block_indexes = _safe_list(locator.get("blockIndexes"))
    chars = _chars_dict(row)
    chars_start = _safe_int(chars.get("start"))
    chars_end = _safe_int(chars.get("end"))
    return page is not None or bool(bbox) or bool(block_indexes) or (
        chars_start is not None and chars_end is not None
    )


def _has_authoritative_char_offsets(row: dict[str, Any]) -> bool:
    chars = _chars_dict(row)
    return _safe_int(chars.get("start")) is not None and _safe_int(chars.get("end")) is not None


def _has_page_bbox_fallback(row: dict[str, Any]) -> bool:
    locator = _locator_dict(row)
    page = _safe_int(locator.get("page"))
    bbox = _safe_list(locator.get("bbox"))
    return page is not None and len(bbox) >= 4


def _offset_authority_mode(row: dict[str, Any]) -> str:
    if _has_authoritative_char_offsets(row):
        chars = _chars_dict(row)
        if (
            _safe_text(chars.get("basis")) == "sourceContentHash"
            and _safe_text(chars.get("normalization"))
            and _safe_text(chars.get("expectedSubstringSha256"))
        ):
            return "chars_offset_authority_complete"
        return "chars_partial_missing_basis"
    if _has_page_bbox_fallback(row):
        return "page_bbox_auxiliary_only"
    locator = _locator_dict(row)
    if _safe_int(locator.get("page")) is not None or _safe_list(locator.get("blockIndexes")):
        return "page_or_block_only"
    return "locator_missing"


def _has_figure_region_authority(row: dict[str, Any]) -> bool:
    return bool(
        _field_value(row, "figureId")
        and _field_value(row, "regionContentHash")
        and _field_value(row, "extractionMethod")
        and (_field_value(row, "figureRegionBbox") or _has_page_bbox_fallback(row))
    )


def _has_table_authority(row: dict[str, Any]) -> bool:
    has_coords = bool(
        _field_value(row, "tableRow")
        or _field_value(row, "tableCol")
        or _field_value(row, "rowIndex")
        or _field_value(row, "colIndex")
    )
    return bool(
        _field_value(row, "tableId")
        and has_coords
        and _field_value(row, "cellRawText")
        and _field_value(row, "cellNormalizedValue")
        and _field_value(row, "cellContentHash")
    )


def _has_equation_authority(row: dict[str, Any]) -> bool:
    return bool(
        _field_value(row, "equationId")
        and (_field_value(row, "equationTeXHash") or _field_value(row, "mathmlHash"))
    )


def _validate_text_char_authority(row: dict[str, Any]) -> tuple[str, list[str]]:
    if not _has_authoritative_char_offsets(row):
        return POLICY_STATUS_BLOCKED_MISSING_OFFSET_AUTHORITY, [
            "chars_start_or_end_missing"
        ]

    chars = _chars_dict(row)
    blockers: list[str] = []
    if _safe_text(chars.get("basis")) != "sourceContentHash":
        blockers.append(
            f"chars_basis_invalid:{_safe_text(chars.get('basis')) or 'missing'}"
        )
    if not _safe_text(chars.get("normalization")):
        blockers.append("chars_normalization_missing")
    if not _safe_text(chars.get("expectedSubstringSha256")):
        blockers.append("chars_expectedSubstringSha256_missing")

    if blockers:
        if any("basis" in item for item in blockers) or any("expectedSubstring" in item for item in blockers):
            return POLICY_STATUS_BLOCKED_LOCATOR_BASIS_UNKNOWN, _dedupe(blockers)
        return POLICY_STATUS_BLOCKED_NORMALIZATION_MISMATCH, _dedupe(blockers)

    return "", []


def _runtime_or_strict_flag_violation(row: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for field_name in (
        "sourceSpanCreated",
        "strictEligible",
        "citationGrade",
        "runtimeEvidence",
        "parserRoutingChanged",
        "answerIntegrationChanged",
        "databaseMutation",
        "canonicalParsedArtifactsWritten",
    ):
        if _safe_bool(row.get(field_name)):
            violations.append(f"{field_name}_true")
    return violations


def _classify_policy_row(row: dict[str, Any]) -> tuple[str, list[str], str]:
    blockers: list[str] = []
    offset_mode = _offset_authority_mode(row)
    artifact_type = _safe_text(row.get("artifact_type"))

    if _safe_text(row.get("readback_status")) != READBACK_STATUS_VALIDATED:
        blockers.extend(_safe_list(row.get("review_blockers")))
        blockers.append(f"readback_status={_safe_text(row.get('readback_status')) or 'unknown'}")
        return POLICY_STATUS_BLOCKED_READBACK_NOT_READY, _dedupe(blockers), offset_mode

    missing_identity = [
        field_name
        for field_name, value in (
            ("sourceSpanId", row.get("sourceSpanId")),
            ("candidateRecordId", row.get("candidateRecordId")),
            ("idempotencyKey", row.get("idempotencyKey")),
        )
        if not _safe_text(value)
    ]
    if missing_identity:
        return POLICY_STATUS_BLOCKED_MISSING_RECORD_IDENTITY, [
            f"missing_{field_name}" for field_name in missing_identity
        ], offset_mode

    if artifact_type not in KNOWN_ARTIFACT_TYPES:
        return POLICY_STATUS_BLOCKED_UNSUPPORTED_ARTIFACT_TYPE, [
            f"artifact_type={artifact_type or 'unknown'}"
        ], offset_mode

    if not _safe_text(row.get("sourceContentHash")):
        return POLICY_STATUS_BLOCKED_MISSING_SOURCE_HASH, ["sourceContentHash_missing"], offset_mode

    if not _has_any_locator_signal(row):
        return POLICY_STATUS_BLOCKED_MISSING_LOCATOR, [
            "locator_missing_page_bbox_blockIndexes_or_chars"
        ], offset_mode

    flag_violations = _runtime_or_strict_flag_violation(row)
    if flag_violations:
        return POLICY_STATUS_BLOCKED_RUNTIME_OR_STRICT_FLAG, flag_violations, offset_mode

    if artifact_type in TEXT_ARTIFACT_TYPES:
        text_status, text_blockers = _validate_text_char_authority(row)
        if text_status:
            return text_status, text_blockers, offset_mode
        return POLICY_STATUS_STRICT_TEXT, [], offset_mode

    if artifact_type in CAPTION_ARTIFACT_TYPES:
        if _has_figure_region_authority(row):
            return POLICY_STATUS_STRICT_STRUCTURED, [], "structured_authority_present"
        text_status, text_blockers = _validate_text_char_authority(row)
        if text_status == "":
            return POLICY_STATUS_STRICT_CAPTION, [], offset_mode
        if text_status == POLICY_STATUS_BLOCKED_MISSING_OFFSET_AUTHORITY:
            return POLICY_STATUS_BLOCKED_MISSING_OFFSET_AUTHORITY, [
                *text_blockers,
                "figure_caption_requires_chars_offset_authority",
            ], offset_mode
        return text_status, text_blockers, offset_mode

    if artifact_type in STRUCTURED_TABLE_TYPES:
        if _has_table_authority(row):
            return POLICY_STATUS_STRICT_STRUCTURED, [], "structured_authority_present"
        return POLICY_STATUS_BLOCKED_MISSING_STRUCTURED_AUTHORITY, [
            "table_authority_fields_missing"
        ], offset_mode

    if artifact_type in STRUCTURED_EQUATION_TYPES:
        if _has_equation_authority(row):
            return POLICY_STATUS_STRICT_STRUCTURED, [], "structured_authority_present"
        return POLICY_STATUS_BLOCKED_MISSING_STRUCTURED_AUTHORITY, [
            "equation_authority_fields_missing"
        ], offset_mode

    return POLICY_STATUS_BLOCKED_TYPE_AUTHORITY_UNDEFINED, [
        f"type_authority_policy_undefined:{artifact_type}"
    ], offset_mode


def _nested_counter(rows: list[dict[str, Any]], key_a: str, key_b: str) -> dict[str, dict[str, int]]:
    counter: Counter[tuple[str, str]] = Counter()
    for row in rows:
        counter[(str(row.get(key_a) or ""), str(row.get(key_b) or ""))] += 1
    nested: dict[str, dict[str, int]] = {}
    for (key_a_value, key_b_value), count in counter.items():
        nested.setdefault(key_a_value, {})[key_b_value] = count
    return nested


def _policy_rows(readback_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, readback_row in enumerate(readback_rows):
        source_row = dict(readback_row or {})
        status, blockers, offset_mode = _classify_policy_row(source_row)
        strict_text = status == POLICY_STATUS_STRICT_TEXT
        strict_caption = status == POLICY_STATUS_STRICT_CAPTION
        strict_structured = status == POLICY_STATUS_STRICT_STRUCTURED
        ready = strict_text or strict_caption or strict_structured
        rows.append(
            {
                "policy_gate_row_id": (
                    "parsed-artifact-source-span-strict-evidence-policy-gate-v2-typed:"
                    f"{index:04d}"
                ),
                "readback_review_row_id": _safe_text(source_row.get("review_row_id")),
                "sourceSpanId": _safe_text(source_row.get("sourceSpanId")),
                "candidateRecordId": _safe_text(source_row.get("candidateRecordId")),
                "runId": _safe_text(source_row.get("runId")),
                "paper_id": _safe_text(source_row.get("paper_id")),
                "artifact_type": _safe_text(source_row.get("artifact_type")),
                "source_candidate_id": _safe_text(source_row.get("source_candidate_id")),
                "sourceContentHash": _safe_text(source_row.get("sourceContentHash")),
                "source_file": _safe_text(source_row.get("source_file")),
                "locator": _locator_dict(source_row),
                "idempotencyKey": _safe_text(source_row.get("idempotencyKey")),
                "source_span_store_path": _safe_text(source_row.get("source_span_store_path")),
                "source_span_store_line": _safe_int(source_row.get("source_span_store_line")) or 0,
                "readback_status": _safe_text(source_row.get("readback_status")),
                "offset_authority_mode": offset_mode,
                "policy_gate_status": status,
                "policy_blockers": _dedupe(blockers),
                "strict_text_policy_candidate_only": strict_text,
                "strict_caption_policy_candidate_only": strict_caption,
                "strict_structured_policy_candidate_only": strict_structured,
                "strictEvidenceDesignReviewReady": ready,
                "strictEvidenceCreated": False,
                "strictEligible": False,
                "citationGrade": False,
                "runtimeEvidence": False,
                "parserRoutingChanged": False,
                "answerIntegrationChanged": False,
                "databaseMutation": False,
                "canonicalParsedArtifactsWritten": False,
                "recommended_action": (
                    "queue_for_explicit_strict_evidence_design_review"
                    if ready
                    else "blocked_from_strict_evidence_until_typed_authority_requirements_met"
                ),
            }
        )
    return rows


def _count_rows(
    *,
    rows: list[dict[str, Any]],
    input_schema_violations: list[str],
) -> dict[str, Any]:
    return {
        "inputRows": len(rows),
        "sourceSpanRecordRows": len(rows),
        "readbackValidatedRows": sum(
            1 for row in rows if row.get("readback_status") == READBACK_STATUS_VALIDATED
        ),
        "strictTextPolicyCandidateOnlyRows": sum(
            1 for row in rows if row.get("policy_gate_status") == POLICY_STATUS_STRICT_TEXT
        ),
        "strictCaptionPolicyCandidateOnlyRows": sum(
            1 for row in rows if row.get("policy_gate_status") == POLICY_STATUS_STRICT_CAPTION
        ),
        "strictStructuredPolicyCandidateOnlyRows": sum(
            1 for row in rows if row.get("policy_gate_status") == POLICY_STATUS_STRICT_STRUCTURED
        ),
        "blockedInputSchemaViolationRows": (
            len(rows) if input_schema_violations and rows else int(bool(input_schema_violations))
        ),
        "blockedReadbackNotReadyRows": sum(
            1 for row in rows if row.get("policy_gate_status") == POLICY_STATUS_BLOCKED_READBACK_NOT_READY
        ),
        "blockedMissingSourceHashRows": sum(
            1 for row in rows if row.get("policy_gate_status") == POLICY_STATUS_BLOCKED_MISSING_SOURCE_HASH
        ),
        "blockedMissingLocatorRows": sum(
            1 for row in rows if row.get("policy_gate_status") == POLICY_STATUS_BLOCKED_MISSING_LOCATOR
        ),
        "blockedMissingOffsetAuthorityRows": sum(
            1
            for row in rows
            if row.get("policy_gate_status") == POLICY_STATUS_BLOCKED_MISSING_OFFSET_AUTHORITY
        ),
        "blockedMissingStructuredAuthorityRows": sum(
            1
            for row in rows
            if row.get("policy_gate_status") == POLICY_STATUS_BLOCKED_MISSING_STRUCTURED_AUTHORITY
        ),
        "blockedUnsupportedArtifactTypeRows": sum(
            1
            for row in rows
            if row.get("policy_gate_status") == POLICY_STATUS_BLOCKED_UNSUPPORTED_ARTIFACT_TYPE
        ),
        "blockedRuntimeOrStrictFlagRows": sum(
            1
            for row in rows
            if row.get("policy_gate_status") == POLICY_STATUS_BLOCKED_RUNTIME_OR_STRICT_FLAG
        ),
        "blockedMissingRecordIdentityRows": sum(
            1
            for row in rows
            if row.get("policy_gate_status") == POLICY_STATUS_BLOCKED_MISSING_RECORD_IDENTITY
        ),
        "blockedLocatorBasisUnknownRows": sum(
            1
            for row in rows
            if row.get("policy_gate_status") == POLICY_STATUS_BLOCKED_LOCATOR_BASIS_UNKNOWN
        ),
        "blockedNormalizationMismatchRows": sum(
            1
            for row in rows
            if row.get("policy_gate_status") == POLICY_STATUS_BLOCKED_NORMALIZATION_MISMATCH
        ),
        "blockedTypeAuthorityPolicyUndefinedRows": sum(
            1
            for row in rows
            if row.get("policy_gate_status") == POLICY_STATUS_BLOCKED_TYPE_AUTHORITY_UNDEFINED
        ),
        "sourceSpanCreatedRows": 0,
        "strictEvidenceCreatedRows": 0,
        "citationGradeEvidenceCreatedRows": 0,
        "runtimeEvidenceCreatedRows": 0,
        "parserRoutingChangedRows": 0,
        "answerIntegrationChangedRows": 0,
        "databaseMutationRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
        "schemaViolationCount": len(input_schema_violations),
        "byPaperId": dict(Counter(str(row.get("paper_id") or "") for row in rows)),
        "byArtifactType": dict(Counter(str(row.get("artifact_type") or "") for row in rows)),
        "byPolicyGateStatus": dict(Counter(str(row.get("policy_gate_status") or "") for row in rows)),
        "byArtifactTypeByPolicyGateStatus": _nested_counter(
            rows,
            "artifact_type",
            "policy_gate_status",
        ),
        "byArtifactTypeByOffsetAuthorityMode": _nested_counter(
            rows,
            "artifact_type",
            "offset_authority_mode",
        ),
        "byRecommendedAction": dict(Counter(str(row.get("recommended_action") or "") for row in rows)),
    }


def build_parsed_artifact_source_span_strict_evidence_policy_gate_v2_typed(
    *,
    readback_report_path: str | Path = DEFAULT_READBACK_REPORT_PATH,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    report_path = Path(str(readback_report_path)).expanduser()
    warnings: list[str] = []
    input_schema_violations: list[str] = []
    requested_papers = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}

    readback_report = _read_json(report_path)
    if not readback_report:
        warnings.append("readback_report_missing_or_unreadable")

    validation = validate_payload(
        readback_report,
        PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_READBACK_REVIEW_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        input_schema_violations = [str(error) for error in validation.errors]
        if not readback_report:
            input_schema_violations.append("readback_report_missing_or_unreadable")

    readback_rows = [
        row
        for row in readback_report.get("rows", [])
        if isinstance(row, dict)
        and _safe_text(row.get("readback_status")) == READBACK_STATUS_VALIDATED
    ] if isinstance(readback_report, dict) else []

    if requested_papers:
        found_papers = {
            _safe_text(row.get("paper_id")) for row in readback_rows if _safe_text(row.get("paper_id"))
        }
        if requested_papers - found_papers:
            warnings.append("requested_paper_ids_not_found")
        readback_rows = [
            row for row in readback_rows if _safe_text(row.get("paper_id")) in requested_papers
        ]

    if not readback_rows:
        warnings.append("readback_validated_rows_missing")

    rows = _policy_rows(readback_rows)
    if input_schema_violations:
        for row in rows:
            row["policy_gate_status"] = POLICY_STATUS_BLOCKED_INPUT_SCHEMA
            row["policy_blockers"] = _dedupe(
                [*row.get("policy_blockers", []), *input_schema_violations]
            )
            row["strict_text_policy_candidate_only"] = False
            row["strict_caption_policy_candidate_only"] = False
            row["strict_structured_policy_candidate_only"] = False
            row["strictEvidenceDesignReviewReady"] = False
            row["recommended_action"] = "repair_readback_report_schema_before_typed_strict_policy_gate"

    counts = _count_rows(rows=rows, input_schema_violations=_dedupe(input_schema_violations))
    ready_rows = (
        int(counts.get("strictTextPolicyCandidateOnlyRows") or 0)
        + int(counts.get("strictCaptionPolicyCandidateOnlyRows") or 0)
        + int(counts.get("strictStructuredPolicyCandidateOnlyRows") or 0)
    )
    status = "ok"
    if input_schema_violations or not rows or ready_rows != len(rows):
        status = "blocked"

    return {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_POLICY_GATE_V2_TYPED_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "readbackReportPath": str(report_path),
            "readbackSchema": _safe_text(readback_report.get("schema")) if readback_report else "",
            "requestedPaperIds": sorted(requested_papers),
        },
        "counts": counts,
        "gate": {
            "readyForStrictEvidenceDesignReview": (
                bool(ready_rows) and ready_rows == len(rows) and not input_schema_violations
            ),
            "strictEvidenceCreated": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": _dedupe(input_schema_violations),
            "decision": (
                "parsed_artifact_source_span_strict_evidence_policy_gate_v2_typed_ready"
                if status == "ok"
                else "blocked"
            ),
            "recommendedNextTranche": (
                "parsed_artifact_source_span_original_source_offset_authority_design"
                if int(counts.get("blockedMissingOffsetAuthorityRows") or 0) > 0
                else "parsed_artifact_source_span_strict_evidence_design_review"
                if status == "ok"
                else "parsed_artifact_source_span_promotion_readback_repair"
            ),
        },
        "policy": {
            "reportOnly": True,
            "strictEvidencePolicyGateTypedOnly": True,
            "sourceSpanStoreWrite": False,
            "strictEvidenceCreated": False,
            "strictEligibleMutation": False,
            "citationGradeEvidenceCreated": False,
            "runtimeEvidenceCreated": False,
            "parserRoutingChanged": False,
            "answerIntegrationChanged": False,
            "databaseMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "canonicalParsedArtifactsWritten": False,
        },
        "warnings": _dedupe(warnings),
        "rows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in (
            "schema",
            "status",
            "generatedAt",
            "input",
            "counts",
            "gate",
            "policy",
            "warnings",
            "rows",
        )
        if key in report
    }


def render_parsed_artifact_source_span_strict_evidence_policy_gate_v2_typed_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byPolicyGateStatus") or {})).items())
    ]
    by_artifact_status = []
    for artifact_type, status_counts in sorted(
        (dict(counts.get("byArtifactTypeByPolicyGateStatus") or {})).items()
    ):
        parts = ", ".join(f"{status}={count}" for status, count in sorted(status_counts.items()))
        by_artifact_status.append(f"{artifact_type}: {parts}")
    return "\n".join(
        [
            "# Parsed Artifact SourceSpan Strict Evidence Policy Gate (Typed v2)",
            "",
            f"- status: {report.get('status', '')}",
            f"- report-only: {json.dumps(report.get('policy', {}).get('reportOnly'))}",
            f"- typed policy gate only: {json.dumps(report.get('policy', {}).get('strictEvidencePolicyGateTypedOnly'))}",
            f"- source span records: {int(counts.get('sourceSpanRecordRows') or 0)}",
            f"- strict text policy candidates: {int(counts.get('strictTextPolicyCandidateOnlyRows') or 0)}",
            f"- strict caption policy candidates: {int(counts.get('strictCaptionPolicyCandidateOnlyRows') or 0)}",
            f"- strict structured policy candidates: {int(counts.get('strictStructuredPolicyCandidateOnlyRows') or 0)}",
            f"- blocked missing offset authority: {int(counts.get('blockedMissingOffsetAuthorityRows') or 0)}",
            f"- strict evidence created: {int(counts.get('strictEvidenceCreatedRows') or 0)}",
            "",
            "## Blocked counts by artifact type",
            *[f"- {item}" for item in by_artifact_status],
            "",
            "## Policy gate status breakdown",
            *[f"- {item}" for item in by_status],
        ]
    )


def write_parsed_artifact_source_span_strict_evidence_policy_gate_v2_typed_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-source-span-strict-evidence-policy-gate-v2-typed.json"
    summary_path = root / "parsed-artifact-source-span-strict-evidence-policy-gate-v2-typed-summary.json"
    markdown_path = root / "parsed-artifact-source-span-strict-evidence-policy-gate-v2-typed.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_parsed_artifact_source_span_strict_evidence_policy_gate_v2_typed_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Evaluate applied SourceSpan readback rows against typed strict-evidence "
            "policy prerequisites without creating strict evidence."
        )
    )
    parser.add_argument(
        "--readback-report",
        default=str(DEFAULT_READBACK_REPORT_PATH),
        help="SourceSpan promotion readback-review JSON report.",
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_parsed_artifact_source_span_strict_evidence_policy_gate_v2_typed(
        readback_report_path=args.readback_report,
        paper_ids=args.paper_id or None,
    )

    if args.output_dir:
        paths = write_parsed_artifact_source_span_strict_evidence_policy_gate_v2_typed_reports(
            report,
            args.output_dir,
        )
        print(f"wrote report: {paths['report']}")
        print(f"wrote summary: {paths['summary']}")
        print(f"wrote markdown: {paths['markdown']}")

    if args.json or not args.output_dir:
        print(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_POLICY_GATE_V2_TYPED_SCHEMA_ID",
    "POLICY_STATUS_STRICT_TEXT",
    "POLICY_STATUS_STRICT_CAPTION",
    "POLICY_STATUS_STRICT_STRUCTURED",
    "_classify_policy_row",
    "build_parsed_artifact_source_span_strict_evidence_policy_gate_v2_typed",
    "render_parsed_artifact_source_span_strict_evidence_policy_gate_v2_typed_markdown",
    "write_parsed_artifact_source_span_strict_evidence_policy_gate_v2_typed_reports",
]
