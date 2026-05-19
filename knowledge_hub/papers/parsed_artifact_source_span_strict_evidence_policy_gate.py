"""Strict-evidence policy gate for applied parsed-artifact SourceSpan records.

This helper consumes the SourceSpan promotion readback-review report and
evaluates whether applied SourceSpan rows satisfy minimum strict-evidence
policy prerequisites. It does not create strict evidence, does not set
strictEligible/citationGrade/runtimeEvidence flags, and performs no database,
index, vault, parser-routing, or answer-integration mutation.
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
from knowledge_hub.papers.parsed_artifact_source_span_store_contract import (
    SOURCE_SPAN_STORE_CONTRACT,
)


PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_POLICY_GATE_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-source-span-strict-evidence-policy-gate.v1"
)

POLICY_STATUS_STRICT_POLICY_CANDIDATE_ONLY = "strict_policy_candidate_only"
POLICY_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"
POLICY_STATUS_BLOCKED_READBACK_NOT_READY = "blocked_readback_not_ready"
POLICY_STATUS_BLOCKED_MISSING_SOURCE_HASH = "blocked_missing_source_hash"
POLICY_STATUS_BLOCKED_MISSING_LOCATOR = "blocked_missing_locator"
POLICY_STATUS_BLOCKED_MISSING_OFFSET_AUTHORITY = "blocked_or_needs_offset_authority_review"
POLICY_STATUS_BLOCKED_UNSUPPORTED_ARTIFACT_TYPE = "blocked_unsupported_artifact_type"
POLICY_STATUS_BLOCKED_RUNTIME_OR_STRICT_FLAG = "blocked_runtime_or_strict_flag_violation"
POLICY_STATUS_BLOCKED_MISSING_RECORD_IDENTITY = "blocked_missing_record_identity"

ALLOWED_SOURCE_SPAN_ARTIFACT_TYPES = set(SOURCE_SPAN_STORE_CONTRACT.get("allowedArtifactTypes") or [])

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
    / "parsed-artifact-source-span-strict-evidence-policy-gate"
    / "01-parsed-artifact-source-span-strict-evidence-policy-gate"
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


def _has_authoritative_char_offsets(locator: dict[str, Any]) -> bool:
    chars = locator.get("chars") if isinstance(locator.get("chars"), dict) else {}
    chars_start = _safe_int(chars.get("start"))
    chars_end = _safe_int(chars.get("end"))
    return chars_start is not None and chars_end is not None


def _has_page_bbox_fallback(locator: dict[str, Any]) -> bool:
    page = _safe_int(locator.get("page"))
    bbox = _safe_list(locator.get("bbox"))
    return page is not None and len(bbox) >= 4


def _has_any_locator_signal(locator: dict[str, Any]) -> bool:
    page = _safe_int(locator.get("page"))
    bbox = _safe_list(locator.get("bbox"))
    block_indexes = _safe_list(locator.get("blockIndexes"))
    return (
        _has_authoritative_char_offsets(locator)
        or _has_page_bbox_fallback(locator)
        or page is not None
        or bool(block_indexes)
    )


def _offset_authority_mode(locator: dict[str, Any]) -> str:
    if _has_authoritative_char_offsets(locator):
        return "chars_offset_authority"
    if _has_page_bbox_fallback(locator):
        return "page_bbox_non_strict_candidate_only"
    if _safe_int(locator.get("page")) is not None or _safe_list(locator.get("blockIndexes")):
        return "page_or_block_only_needs_offset_authority_review"
    return "locator_missing"


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
    locator = row.get("locator") if isinstance(row.get("locator"), dict) else {}
    offset_mode = _offset_authority_mode(locator)

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

    if _safe_text(row.get("artifact_type")) not in ALLOWED_SOURCE_SPAN_ARTIFACT_TYPES:
        return POLICY_STATUS_BLOCKED_UNSUPPORTED_ARTIFACT_TYPE, [
            f"artifact_type={_safe_text(row.get('artifact_type')) or 'unknown'}"
        ], offset_mode

    if not _safe_text(row.get("sourceContentHash")):
        return POLICY_STATUS_BLOCKED_MISSING_SOURCE_HASH, ["sourceContentHash_missing"], offset_mode

    if not _has_any_locator_signal(locator):
        return POLICY_STATUS_BLOCKED_MISSING_LOCATOR, [
            "locator_missing_page_bbox_blockIndexes_or_chars"
        ], offset_mode

    flag_violations = _runtime_or_strict_flag_violation(row)
    if flag_violations:
        return POLICY_STATUS_BLOCKED_RUNTIME_OR_STRICT_FLAG, flag_violations, offset_mode

    if offset_mode != "chars_offset_authority":
        blockers.append(f"offset_authority_mode={offset_mode}")
        if offset_mode == "page_bbox_non_strict_candidate_only":
            blockers.append("page_bbox_not_accepted_as_strict_authority")
        return POLICY_STATUS_BLOCKED_MISSING_OFFSET_AUTHORITY, _dedupe(blockers), offset_mode

    return POLICY_STATUS_STRICT_POLICY_CANDIDATE_ONLY, [], offset_mode


def _policy_rows(readback_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, readback_row in enumerate(readback_rows):
        source_row = dict(readback_row or {})
        status, blockers, offset_mode = _classify_policy_row(source_row)
        ready = status == POLICY_STATUS_STRICT_POLICY_CANDIDATE_ONLY
        rows.append(
            {
                "policy_gate_row_id": (
                    f"parsed-artifact-source-span-strict-evidence-policy-gate:{index:04d}"
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
                "locator": (
                    source_row.get("locator") if isinstance(source_row.get("locator"), dict) else {}
                ),
                "idempotencyKey": _safe_text(source_row.get("idempotencyKey")),
                "source_span_store_path": _safe_text(source_row.get("source_span_store_path")),
                "source_span_store_line": _safe_int(source_row.get("source_span_store_line")) or 0,
                "readback_status": _safe_text(source_row.get("readback_status")),
                "offset_authority_mode": offset_mode,
                "policy_gate_status": status,
                "policy_blockers": _dedupe(blockers),
                "strict_policy_candidate_only": ready,
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
                    else "blocked_from_strict_evidence_until_offset_authority_or_readback_repair"
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
        "strictPolicyCandidateOnlyRows": sum(
            1 for row in rows if row.get("policy_gate_status") == POLICY_STATUS_STRICT_POLICY_CANDIDATE_ONLY
        ),
        "strictEvidenceDesignReviewReadyRows": sum(
            1 for row in rows if _safe_bool(row.get("strictEvidenceDesignReviewReady"))
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
        "blockedMissingRecordIdentityRows": sum(
            1
            for row in rows
            if row.get("policy_gate_status") == POLICY_STATUS_BLOCKED_MISSING_RECORD_IDENTITY
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
        "byOffsetAuthorityMode": dict(Counter(str(row.get("offset_authority_mode") or "") for row in rows)),
        "byRecommendedAction": dict(Counter(str(row.get("recommended_action") or "") for row in rows)),
    }


def build_parsed_artifact_source_span_strict_evidence_policy_gate(
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
        row for row in readback_report.get("rows", []) if isinstance(row, dict)
    ] if isinstance(readback_report, dict) else []

    if requested_papers:
        found_papers = {
            _safe_text(row.get("paper_id"))
            for row in readback_rows
            if _safe_text(row.get("paper_id"))
        }
        if requested_papers - found_papers:
            warnings.append("requested_paper_ids_not_found")
        readback_rows = [
            row for row in readback_rows if _safe_text(row.get("paper_id")) in requested_papers
        ]

    if not readback_rows:
        warnings.append("readback_rows_missing")

    rows = _policy_rows(readback_rows)
    if input_schema_violations:
        for row in rows:
            row["policy_gate_status"] = POLICY_STATUS_BLOCKED_INPUT_SCHEMA
            row["policy_blockers"] = _dedupe(
                [*row.get("policy_blockers", []), *input_schema_violations]
            )
            row["strict_policy_candidate_only"] = False
            row["strictEvidenceDesignReviewReady"] = False
            row["recommended_action"] = "repair_readback_report_schema_before_strict_policy_gate"

    counts = _count_rows(rows=rows, input_schema_violations=_dedupe(input_schema_violations))
    ready_rows = int(counts.get("strictPolicyCandidateOnlyRows") or 0)
    status = "ok"
    if input_schema_violations or not rows or ready_rows != len(rows):
        status = "blocked"

    return {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_POLICY_GATE_SCHEMA_ID,
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
                "parsed_artifact_source_span_strict_evidence_policy_gate_ready"
                if status == "ok"
                else "blocked"
            ),
            "recommendedNextTranche": (
                "parsed_artifact_source_span_original_source_offset_authority_design"
                if status == "blocked"
                and int(counts.get("blockedMissingOffsetAuthorityRows") or 0) > 0
                else "parsed_artifact_source_span_strict_evidence_design_review"
                if status == "ok"
                else "parsed_artifact_source_span_promotion_readback_repair"
            ),
        },
        "policy": {
            "reportOnly": True,
            "strictEvidencePolicyGateOnly": True,
            "sourceSpanStoreWrite": False,
            "candidateStoreWrite": False,
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


def render_parsed_artifact_source_span_strict_evidence_policy_gate_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byPolicyGateStatus") or {})).items())
    ]
    by_offset = [
        f"{mode}: {count}"
        for mode, count in sorted((dict(counts.get("byOffsetAuthorityMode") or {})).items())
    ]
    return "\n".join(
        [
            "# Parsed Artifact SourceSpan Strict Evidence Policy Gate",
            "",
            f"- status: {report.get('status', '')}",
            f"- report-only: {json.dumps(report.get('policy', {}).get('reportOnly'))}",
            f"- strict-evidence-policy-gate-only: {json.dumps(report.get('policy', {}).get('strictEvidencePolicyGateOnly'))}",
            f"- source span records: {int(counts.get('sourceSpanRecordRows') or 0)}",
            f"- strict-policy-candidate-only rows: {int(counts.get('strictPolicyCandidateOnlyRows') or 0)}",
            f"- blocked missing offset authority rows: {int(counts.get('blockedMissingOffsetAuthorityRows') or 0)}",
            f"- strict evidence created: {int(counts.get('strictEvidenceCreatedRows') or 0)}",
            f"- database mutations: {int(counts.get('databaseMutationRows') or 0)}",
            "",
            "## Offset authority breakdown",
            *[f"- {item}" for item in by_offset],
            "",
            "## Policy gate status breakdown",
            *[f"- {item}" for item in by_status],
        ]
    )


def write_parsed_artifact_source_span_strict_evidence_policy_gate_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-source-span-strict-evidence-policy-gate.json"
    summary_path = root / "parsed-artifact-source-span-strict-evidence-policy-gate-summary.json"
    markdown_path = root / "parsed-artifact-source-span-strict-evidence-policy-gate.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_parsed_artifact_source_span_strict_evidence_policy_gate_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Evaluate applied SourceSpan readback rows against strict-evidence "
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

    report = build_parsed_artifact_source_span_strict_evidence_policy_gate(
        readback_report_path=args.readback_report,
        paper_ids=args.paper_id or None,
    )

    if args.output_dir:
        paths = write_parsed_artifact_source_span_strict_evidence_policy_gate_reports(
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
    "PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_POLICY_GATE_SCHEMA_ID",
    "POLICY_STATUS_STRICT_POLICY_CANDIDATE_ONLY",
    "_classify_policy_row",
    "build_parsed_artifact_source_span_strict_evidence_policy_gate",
    "render_parsed_artifact_source_span_strict_evidence_policy_gate_markdown",
    "write_parsed_artifact_source_span_strict_evidence_policy_gate_reports",
]
