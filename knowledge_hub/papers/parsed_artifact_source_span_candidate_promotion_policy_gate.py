"""Promotion policy gate for parsed-artifact SourceSpan candidates.

This helper consumes the SourceSpan candidate readback-review report and checks
whether candidate-only rows can be queued for a later explicit promotion
executor dry run. It does not create SourceSpan records and does not create
strict, citation-grade, or runtime evidence.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_candidate_readback_review import (
    PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_READBACK_REVIEW_SCHEMA_ID,
    READBACK_STATUS_PROMOTION_REVIEW_READY,
)


PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_PROMOTION_POLICY_GATE_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-source-span-candidate-promotion-policy-gate.v1"
)

POLICY_STATUS_READY_CANDIDATE_ONLY = "policy_gate_ready_candidate_only"
POLICY_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"
POLICY_STATUS_BLOCKED_READBACK_NOT_READY = "blocked_readback_not_ready"
POLICY_STATUS_BLOCKED_MISSING_SOURCE_HASH = "blocked_missing_source_hash"
POLICY_STATUS_BLOCKED_MISSING_LOCATOR = "blocked_missing_locator"
POLICY_STATUS_BLOCKED_UNSUPPORTED_ARTIFACT_TYPE = "blocked_unsupported_artifact_type"
POLICY_STATUS_BLOCKED_RUNTIME_OR_STRICT_FLAG = "blocked_runtime_or_strict_flag_violation"

ALLOWED_SOURCE_SPAN_ARTIFACT_TYPES = {"section", "figure", "table"}

DEFAULT_READBACK_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-source-span-candidate-readback-review"
    / "01-parsed-artifact-source-span-candidate-readback-review"
    / "parsed-artifact-source-span-candidate-readback-review.json"
)

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-source-span-candidate-promotion-policy-gate"
    / "01-parsed-artifact-source-span-candidate-promotion-policy-gate"
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


def _has_locator(row: dict[str, Any]) -> bool:
    locator = row.get("locator")
    if not isinstance(locator, dict):
        return False
    page = _safe_int(locator.get("page"))
    bbox = _safe_list(locator.get("bbox"))
    block_indexes = _safe_list(locator.get("blockIndexes"))
    chars = locator.get("chars")
    chars_start = _safe_int(chars.get("start")) if isinstance(chars, dict) else None
    chars_end = _safe_int(chars.get("end")) if isinstance(chars, dict) else None
    return page is not None or bool(bbox) or bool(block_indexes) or (
        chars_start is not None and chars_end is not None
    )


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


def _classify_policy_row(row: dict[str, Any]) -> tuple[str, list[str]]:
    blockers: list[str] = []

    if _safe_text(row.get("readback_status")) != READBACK_STATUS_PROMOTION_REVIEW_READY:
        blockers.extend(_safe_list(row.get("review_blockers")))
        blockers.append(f"readback_status={_safe_text(row.get('readback_status')) or 'unknown'}")
        return POLICY_STATUS_BLOCKED_READBACK_NOT_READY, _dedupe(blockers)

    if _safe_text(row.get("artifact_type")) not in ALLOWED_SOURCE_SPAN_ARTIFACT_TYPES:
        return POLICY_STATUS_BLOCKED_UNSUPPORTED_ARTIFACT_TYPE, [
            f"artifact_type={_safe_text(row.get('artifact_type')) or 'unknown'}"
        ]

    if not _safe_text(row.get("sourceContentHash")):
        return POLICY_STATUS_BLOCKED_MISSING_SOURCE_HASH, ["sourceContentHash_missing"]

    if not _has_locator(row):
        return POLICY_STATUS_BLOCKED_MISSING_LOCATOR, [
            "locator_missing_page_bbox_blockIndexes_or_chars"
        ]

    flag_violations = _runtime_or_strict_flag_violation(row)
    if flag_violations:
        return POLICY_STATUS_BLOCKED_RUNTIME_OR_STRICT_FLAG, flag_violations

    return POLICY_STATUS_READY_CANDIDATE_ONLY, []


def _policy_rows(readback_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, readback_row in enumerate(readback_rows):
        source_row = dict(readback_row or {})
        status, blockers = _classify_policy_row(source_row)
        ready = status == POLICY_STATUS_READY_CANDIDATE_ONLY
        rows.append(
            {
                "policy_gate_row_id": (
                    f"parsed-artifact-source-span-candidate-promotion-policy-gate:{index:04d}"
                ),
                "readback_review_row_id": _safe_text(source_row.get("review_row_id")),
                "candidateRecordId": _safe_text(source_row.get("candidateRecordId")),
                "runId": _safe_text(source_row.get("runId")),
                "paper_id": _safe_text(source_row.get("paper_id")),
                "artifact_type": _safe_text(source_row.get("artifact_type")),
                "source_candidate_id": _safe_text(source_row.get("source_candidate_id")),
                "source_readiness_row_id": _safe_text(source_row.get("source_readiness_row_id")),
                "sourceContentHash": _safe_text(source_row.get("sourceContentHash")),
                "source_file": _safe_text(source_row.get("source_file")),
                "locator": (
                    source_row.get("locator") if isinstance(source_row.get("locator"), dict) else {}
                ),
                "idempotencyKey": _safe_text(source_row.get("idempotencyKey")),
                "candidate_store_path": _safe_text(source_row.get("candidate_store_path")),
                "candidate_store_line": _safe_int(source_row.get("candidate_store_line")) or 0,
                "readback_status": _safe_text(source_row.get("readback_status")),
                "policy_gate_status": status,
                "policy_blockers": _dedupe(blockers),
                "promotion_policy_gate_ready": ready,
                "sourceSpanPromotionExecutorDryRunReady": ready,
                "sourceSpanPromotionApproved": False,
                "sourceSpanCreated": False,
                "strictEligible": False,
                "citationGrade": False,
                "runtimeEvidence": False,
                "parserRoutingChanged": False,
                "answerIntegrationChanged": False,
                "databaseMutation": False,
                "canonicalParsedArtifactsWritten": False,
                "recommended_action": (
                    "queue_for_explicit_source_span_promotion_executor_dry_run"
                    if ready
                    else "repair_candidate_or_readback_before_promotion_policy_gate"
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
        "candidateRecordRows": len(rows),
        "promotionReviewReadyCandidateOnlyRows": sum(
            1 for row in rows if row.get("readback_status") == READBACK_STATUS_PROMOTION_REVIEW_READY
        ),
        "promotionPolicyGateReadyRows": sum(
            1 for row in rows if row.get("policy_gate_status") == POLICY_STATUS_READY_CANDIDATE_ONLY
        ),
        "sourceSpanPromotionExecutorDryRunReadyRows": sum(
            1
            for row in rows
            if _safe_bool(row.get("sourceSpanPromotionExecutorDryRunReady"))
        ),
        "sourceSpanPromotionApprovedRows": 0,
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
        "byRecommendedAction": dict(Counter(str(row.get("recommended_action") or "") for row in rows)),
    }


def build_parsed_artifact_source_span_candidate_promotion_policy_gate(
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
        PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_READBACK_REVIEW_SCHEMA_ID,
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
            row["promotion_policy_gate_ready"] = False
            row["sourceSpanPromotionExecutorDryRunReady"] = False
            row["recommended_action"] = "repair_readback_report_schema_before_policy_gate"

    counts = _count_rows(rows=rows, input_schema_violations=_dedupe(input_schema_violations))
    ready_rows = int(counts.get("promotionPolicyGateReadyRows") or 0)
    status = "ok"
    if input_schema_violations or not rows or ready_rows != len(rows):
        status = "blocked"

    return {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_PROMOTION_POLICY_GATE_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "readbackReportPath": str(report_path),
            "readbackSchema": _safe_text(readback_report.get("schema")) if readback_report else "",
            "requestedPaperIds": sorted(requested_papers),
        },
        "counts": counts,
        "gate": {
            "readyForSourceSpanPromotionExecutorDryRun": (
                bool(ready_rows) and ready_rows == len(rows) and not input_schema_violations
            ),
            "sourceSpanPromotionApproved": False,
            "sourceSpanCreated": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": _dedupe(input_schema_violations),
            "decision": (
                "parsed_artifact_source_span_candidate_promotion_policy_gate_ready"
                if status == "ok"
                else "blocked"
            ),
            "recommendedNextTranche": (
                "parsed_artifact_source_span_promotion_executor_dry_run"
                if status == "ok"
                else "parsed_artifact_source_span_candidate_readback_repair"
            ),
        },
        "policy": {
            "reportOnly": True,
            "promotionPolicyGateOnly": True,
            "candidateStoreWrite": False,
            "sourceSpanPromotionApproved": False,
            "sourceSpanCreated": False,
            "strictEvidenceCreated": False,
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


def render_parsed_artifact_source_span_candidate_promotion_policy_gate_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byPolicyGateStatus") or {})).items())
    ]
    return "\n".join(
        [
            "# Parsed Artifact SourceSpan Candidate Promotion Policy Gate",
            "",
            f"- status: {report.get('status', '')}",
            f"- report-only: {json.dumps(report.get('policy', {}).get('reportOnly'))}",
            f"- promotion-policy-gate-only: {json.dumps(report.get('policy', {}).get('promotionPolicyGateOnly'))}",
            f"- candidate records: {int(counts.get('candidateRecordRows') or 0)}",
            f"- policy-gate-ready rows: {int(counts.get('promotionPolicyGateReadyRows') or 0)}",
            f"- source span promotion approved rows: {int(counts.get('sourceSpanPromotionApprovedRows') or 0)}",
            f"- source spans created: {int(counts.get('sourceSpanCreatedRows') or 0)}",
            f"- strict evidence created: {int(counts.get('strictEvidenceCreatedRows') or 0)}",
            f"- database mutations: {int(counts.get('databaseMutationRows') or 0)}",
            "",
            "## Policy gate status breakdown",
            *[f"- {item}" for item in by_status],
        ]
    )


def write_parsed_artifact_source_span_candidate_promotion_policy_gate_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-source-span-candidate-promotion-policy-gate.json"
    summary_path = root / "parsed-artifact-source-span-candidate-promotion-policy-gate-summary.json"
    markdown_path = root / "parsed-artifact-source-span-candidate-promotion-policy-gate.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_parsed_artifact_source_span_candidate_promotion_policy_gate_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Evaluate SourceSpan candidate readback rows against the promotion policy gate "
            "before any explicit promotion executor dry run."
        )
    )
    parser.add_argument(
        "--readback-report",
        default=str(DEFAULT_READBACK_REPORT_PATH),
        help="SourceSpan candidate readback-review JSON report.",
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_parsed_artifact_source_span_candidate_promotion_policy_gate(
        readback_report_path=args.readback_report,
        paper_ids=args.paper_id or None,
    )

    if args.output_dir:
        paths = write_parsed_artifact_source_span_candidate_promotion_policy_gate_reports(
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
    "PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_PROMOTION_POLICY_GATE_SCHEMA_ID",
    "POLICY_STATUS_READY_CANDIDATE_ONLY",
    "build_parsed_artifact_source_span_candidate_promotion_policy_gate",
    "render_parsed_artifact_source_span_candidate_promotion_policy_gate_markdown",
    "write_parsed_artifact_source_span_candidate_promotion_policy_gate_reports",
]
