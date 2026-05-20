"""Report-only table-cell structured-evidence readiness audit helper."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowledge_hub.papers.table_cell_provenance_feasibility_audit import (
    TABLE_CELL_PROVENANCE_FEASIBILITY_AUDIT_SCHEMA_ID,
)


TABLE_CELL_STRUCTURED_EVIDENCE_READINESS_AUDIT_SCHEMA_ID = (
    "knowledge-hub.paper.table-cell-structured-evidence-readiness-audit.v1"
)

STATUS_READY = "table_cell_structured_authority_candidate_only"
STATUS_BLOCKED_MISSING_TABLE_ID = "blocked_missing_table_id"
STATUS_BLOCKED_MISSING_CELL_COORDINATES = "blocked_missing_cell_coordinates"
STATUS_BLOCKED_MISSING_CELL_TEXT = "blocked_missing_cell_text"
STATUS_BLOCKED_MISSING_SOURCE_HASH = "blocked_missing_source_hash"
STATUS_BLOCKED_MISSING_LOCATOR = "blocked_missing_locator"
STATUS_BLOCKED_NUMERIC_NORMALIZATION_UNVERIFIED = "blocked_numeric_normalization_unverified"
STATUS_BLOCKED_INPUT_REPORT_MISSING = "blocked_input_report_missing"
STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION = "blocked_input_schema_violation"

RECOMMENDED_ACTION_BY_STATUS = {
    STATUS_READY: "route_for_table_cell_structured_evidence_execution_plan",
    STATUS_BLOCKED_MISSING_TABLE_ID: "recover_table_region_candidate_id_before_table_cell_readiness",
    STATUS_BLOCKED_MISSING_CELL_COORDINATES: "recover_cell_bbox_or_coordinate_data_before_table_cell_readiness",
    STATUS_BLOCKED_MISSING_CELL_TEXT: "recover_table_cell_row_column_text_before_table_cell_readiness",
    STATUS_BLOCKED_MISSING_SOURCE_HASH: "recover_source_content_hash_before_table_cell_readiness",
    STATUS_BLOCKED_MISSING_LOCATOR: "recover_table_cell_locator_before_table_cell_readiness",
    STATUS_BLOCKED_NUMERIC_NORMALIZATION_UNVERIFIED: "verify_numeric_normalization_before_table_cell_readiness",
    STATUS_BLOCKED_INPUT_REPORT_MISSING: "provide_table_cell_provenance_feasibility_report",
    STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION: "repair_input_report_schema_before_table_cell_readiness",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    if isinstance(value, (int, float)):
        return int(value) != 0
    return bool(value)


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    out: list[str] = []
    try:
        for item in list(value):
            text = str(item or "").strip()
            if text:
                out.append(text)
    except Exception:
        return []
    return out


def _normalize_bbox(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[float] = []
    for item in list(value):
        try:
            out.append(float(item))
        except Exception:
            continue
    return out


def _normalize_indexes(value: Any) -> list[int]:
    if value is None:
        return []
    out: list[int] = []
    try:
        for item in list(value):
            out.append(int(item))
    except Exception:
        return []
    return out


def _dedupe_rows(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = str(value).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    payload_path = Path(str(path)).expanduser()
    if not payload_path.exists():
        return {}
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return []
    return [dict(item) for item in rows if isinstance(item, dict)]


def _extract_source_tex_row_id(row: dict[str, Any]) -> str:
    for key in ("source_tex_row_id", "source_row_id", "sourceCandidateId", "source_cell_row_id"):
        value = _safe_text(row.get(key))
        if value:
            return value
    return ""


def _extract_table_region_candidate_id(row: dict[str, Any]) -> str:
    for key in ("table_region_candidate_id", "tableCandidateId", "source_row_id"):
        value = _safe_text(row.get(key))
        if value:
            return value
    return ""


def _contains_numeric_normalization_unverified(reasons: list[str]) -> bool:
    lower_tokens = [item.lower() for item in reasons]
    return any("numeric" in item and "normal" in item for item in lower_tokens)


def _has_locator(row: dict[str, Any]) -> bool:
    page = _safe_int(row.get("page")) or _safe_int(row.get("caption_page"))
    if page is not None:
        return True

    chars_start = _safe_int(row.get("chars_start")) or _safe_int(row.get("caption_chars_start"))
    chars_end = _safe_int(row.get("chars_end")) or _safe_int(row.get("caption_chars_end"))
    if chars_start is not None and chars_end is not None:
        return True

    selected_pdf_region = row.get("selected_pdf_region")
    if not isinstance(selected_pdf_region, dict):
        selected_pdf_region = {}

    if _normalize_bbox(
        row.get("bbox")
        or row.get("selected_bbox")
        or selected_pdf_region.get("bbox")
    ):
        return True

    if _normalize_indexes(
        row.get("blockIndexes")
        or row.get("block_indexes")
        or selected_pdf_region.get("block_indexes")
    ):
        return True

    return False


def _classify_row(source_row: dict[str, Any]) -> str:
    strict_blockers = _normalize_string_list(source_row.get("strict_blockers"))
    non_strict_reason = _normalize_string_list(source_row.get("non_strict_reason"))
    blockers = [*strict_blockers, *non_strict_reason]

    table_region_candidate_id = _extract_table_region_candidate_id(source_row)
    source_content_hash = _safe_text(source_row.get("sourceContentHash"))

    row_column_text_available = _safe_bool(source_row.get("row_column_text_available"))
    non_empty_count = _safe_int(source_row.get("non_empty_table_cell_count"))
    if non_empty_count is not None:
        row_column_text_available = non_empty_count > 0

    cell_bbox_available = _safe_bool(source_row.get("cell_bbox_available"))
    cell_bbox_count = _safe_int(source_row.get("cell_bbox_count"))
    if cell_bbox_count is not None:
        cell_bbox_available = cell_bbox_available or cell_bbox_count > 0
    if not cell_bbox_available:
        cell_bbox_available = bool(_normalize_bbox(source_row.get("bbox")))

    if not table_region_candidate_id:
        return STATUS_BLOCKED_MISSING_TABLE_ID
    if not source_content_hash:
        return STATUS_BLOCKED_MISSING_SOURCE_HASH
    if _contains_numeric_normalization_unverified(blockers):
        return STATUS_BLOCKED_NUMERIC_NORMALIZATION_UNVERIFIED
    if not row_column_text_available:
        return STATUS_BLOCKED_MISSING_CELL_TEXT
    if not cell_bbox_available:
        return STATUS_BLOCKED_MISSING_CELL_COORDINATES
    if not _has_locator(source_row):
        return STATUS_BLOCKED_MISSING_LOCATOR
    return STATUS_READY


def _build_row(source_row: dict[str, Any]) -> dict[str, Any]:
    source_candidate_id = _safe_text(source_row.get("source_candidate_id"))
    if not source_candidate_id:
        source_candidate_id = _safe_text(
            source_row.get("sourceCandidateId")
            or source_row.get("candidate_id")
            or source_row.get("source_row_id")
        )

    selected_pdf_region = source_row.get("selected_pdf_region")
    if not isinstance(selected_pdf_region, dict):
        selected_pdf_region = {}

    bbox = _normalize_bbox(
        source_row.get("bbox") or source_row.get("selected_bbox") or selected_pdf_region.get("bbox")
    )
    block_indexes = _normalize_indexes(
        source_row.get("blockIndexes")
        or source_row.get("block_indexes")
        or selected_pdf_region.get("block_indexes")
    )
    if not block_indexes:
        block_indexes = _normalize_indexes(source_row.get("selected_pdf_region_block_indexes"))

    source_page = _safe_int(source_row.get("page"))
    if source_page is None:
        source_page = _safe_int(source_row.get("caption_page"))

    strict_blockers = _normalize_string_list(source_row.get("strict_blockers"))
    non_strict_reason = _normalize_string_list(source_row.get("non_strict_reason"))

    status = _classify_row(source_row)
    label_number_hint = source_row.get("labelNumberHint")
    if label_number_hint is None:
        label_number_hint = source_row.get("table_label")

    return {
        "artifact_type": "table",
        "paper_id": _safe_text(source_row.get("paper_id")),
        "source_candidate_id": source_candidate_id,
        "table_region_candidate_id": _extract_table_region_candidate_id(source_row),
        "sourceContentHash": _safe_text(source_row.get("sourceContentHash")),
        "source_tex_row_id": _extract_source_tex_row_id(source_row),
        "source_file": _safe_text(
            source_row.get("source_file")
            or source_row.get("sourceFile")
            or source_row.get("source_pdf_path")
            or source_row.get("sourcePdfPath")
        ),
        "page": source_page,
        "bbox": bbox,
        "blockIndexes": block_indexes,
        "labelNumberHint": label_number_hint,
        "strict_blockers": strict_blockers,
        "non_strict_reason": non_strict_reason,
        "readiness_status": status,
        "recommended_action": RECOMMENDED_ACTION_BY_STATUS.get(status, ""),
        "source_span_created": _safe_bool(source_row.get("source_span_created")),
        "strict_eligible": _safe_bool(source_row.get("strict_eligible")),
        "citation_grade": _safe_bool(source_row.get("citation_grade")),
        "runtime_evidence": _safe_bool(source_row.get("runtime_evidence")),
        "parser_routing_changed": _safe_bool(source_row.get("parser_routing_changed")),
        "answer_integration_changed": _safe_bool(source_row.get("answer_integration_changed")),
    }


def _counts(
    rows: list[dict[str, Any]],
    *,
    input_report_missing: bool,
    input_schema_violation: bool,
) -> dict[str, Any]:
    return {
        "inputRows": len(rows),
        "targetRows": len(rows),
        "tableCellStructuredAuthorityCandidateOnlyRows": sum(
            1 for row in rows if row.get("readiness_status") == STATUS_READY
        ),
        "blockedMissingTableIdRows": sum(
            1 for row in rows if row.get("readiness_status") == STATUS_BLOCKED_MISSING_TABLE_ID
        ),
        "blockedMissingCellCoordinatesRows": sum(
            1 for row in rows if row.get("readiness_status") == STATUS_BLOCKED_MISSING_CELL_COORDINATES
        ),
        "blockedMissingCellTextRows": sum(
            1 for row in rows if row.get("readiness_status") == STATUS_BLOCKED_MISSING_CELL_TEXT
        ),
        "blockedMissingSourceHashRows": sum(
            1 for row in rows if row.get("readiness_status") == STATUS_BLOCKED_MISSING_SOURCE_HASH
        ),
        "blockedMissingLocatorRows": sum(
            1 for row in rows if row.get("readiness_status") == STATUS_BLOCKED_MISSING_LOCATOR
        ),
        "blockedNumericNormalizationUnverifiedRows": sum(
            1
            for row in rows
            if row.get("readiness_status") == STATUS_BLOCKED_NUMERIC_NORMALIZATION_UNVERIFIED
        ),
        "blockedInputReportMissingRows": 1 if input_report_missing else 0,
        "blockedInputSchemaViolationRows": 1 if input_schema_violation else 0,
        "sourceSpanCreatedRows": sum(1 for row in rows if bool(row.get("source_span_created"))),
        "strictEligibleRows": sum(1 for row in rows if bool(row.get("strict_eligible"))),
        "citationGradeRows": sum(1 for row in rows if bool(row.get("citation_grade"))),
        "runtimeEvidenceRows": sum(1 for row in rows if bool(row.get("runtime_evidence"))),
        "parserRoutingChangedRows": sum(1 for row in rows if bool(row.get("parser_routing_changed"))),
        "answerIntegrationChangedRows": sum(1 for row in rows if bool(row.get("answer_integration_changed"))),
        "databaseMutationRows": 0,
        "schemaViolationCount": 1 if input_schema_violation else 0,
        "byArtifactType": dict(Counter(str(row.get("artifact_type") or "") for row in rows)),
        "byReadinessStatus": dict(Counter(str(row.get("readiness_status") or "") for row in rows)),
        "byRecommendedAction": dict(Counter(str(row.get("recommended_action") or "") for row in rows)),
    }


def build_table_cell_structured_evidence_readiness_audit(
    *,
    table_cell_provenance_feasibility_report: str | Path | None = None,
) -> dict[str, Any]:
    report_path = Path(str(table_cell_provenance_feasibility_report)).expanduser() if table_cell_provenance_feasibility_report else None
    input_payload = _read_json(report_path)
    input_schema = _safe_text(input_payload.get("schema"))

    warnings: list[str] = []
    schema_violations: list[str] = []
    rows: list[dict[str, Any]] = []
    input_report_missing = False
    input_schema_violation = False

    if report_path is None or not input_payload:
        input_report_missing = True
        warnings.append("table_cell_provenance_feasibility_report_missing_or_unreadable")
        warnings.append(STATUS_BLOCKED_INPUT_REPORT_MISSING)
    elif input_schema != TABLE_CELL_PROVENANCE_FEASIBILITY_AUDIT_SCHEMA_ID:
        input_schema_violation = True
        warnings.append(STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION)
        warnings.append("table_cell_provenance_feasibility_report_schema_mismatch")
        schema_violations.append("table_cell_provenance_feasibility_report_schema_mismatch")
    else:
        source_rows = _read_rows(input_payload)
        if not source_rows:
            warnings.append("table_cell_provenance_feasibility_rows_missing")
        rows.extend(_build_row(source_row) for source_row in source_rows)

    if not rows and not input_report_missing and not input_schema_violation:
        warnings.append("no_usable_table_cell_structured_evidence_rows")

    ready = bool(rows) and not input_report_missing and not input_schema_violation
    status = "ok" if ready else "blocked"
    decision = (
        "table_cell_structured_evidence_readiness_audit_ready" if status == "ok" else "blocked"
    )

    counts = _counts(
        rows,
        input_report_missing=input_report_missing,
        input_schema_violation=input_schema_violation,
    )

    return {
        "schema": TABLE_CELL_STRUCTURED_EVIDENCE_READINESS_AUDIT_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "tableCellProvenanceFeasibilityReport": str(report_path) if report_path else "",
            "tableCellProvenanceFeasibilitySchema": TABLE_CELL_PROVENANCE_FEASIBILITY_AUDIT_SCHEMA_ID,
        },
        "counts": counts,
        "gate": {
            "readyForTableCellStructuredEvidenceReadinessReview": status == "ok",
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "schemaViolations": schema_violations,
            "decision": decision,
            "recommendedNextTranche": "table_cell_structured_evidence_execution_plan",
        },
        "policy": {
            "reportOnly": True,
            "designOnly": True,
            "databaseMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "canonicalParsedArtifactsWritten": False,
            "sourceSpanCreated": False,
            "strictEvidenceCreated": False,
            "runtimeEvidenceCreated": False,
            "parserRoutingChanged": False,
            "answerIntegrationChanged": False,
        },
        "warnings": _dedupe_rows(warnings),
        "rows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": report["schema"],
        "status": report["status"],
        "generatedAt": report["generatedAt"],
        "input": report["input"],
        "counts": report["counts"],
        "gate": report["gate"],
        "policy": report["policy"],
        "warnings": report["warnings"],
        "rows": report["rows"],
    }


def render_table_cell_structured_evidence_readiness_audit_markdown(report: dict[str, Any]) -> str:
    counts = report.get("counts", {})
    return "\n".join(
        [
            "# TableCell Structured-Evidence Readiness Audit",
            "",
            f"- status: {report.get('status', '')}",
            f"- report-only: {json.dumps(report.get('policy', {}).get('reportOnly'))}",
            f"- input rows: {int(counts.get('inputRows') or 0)}",
            f"- target rows: {int(counts.get('targetRows') or 0)}",
            f"- authority-ready rows: {int(counts.get('tableCellStructuredAuthorityCandidateOnlyRows') or 0)}",
            f"- blocked missing table-id rows: {int(counts.get('blockedMissingTableIdRows') or 0)}",
            f"- blocked missing cell-coordinate rows: {int(counts.get('blockedMissingCellCoordinatesRows') or 0)}",
            f"- blocked missing cell-text rows: {int(counts.get('blockedMissingCellTextRows') or 0)}",
            f"- blocked missing source-hash rows: {int(counts.get('blockedMissingSourceHashRows') or 0)}",
            f"- blocked missing locator rows: {int(counts.get('blockedMissingLocatorRows') or 0)}",
            f"- blocked numeric-normalization rows: {int(counts.get('blockedNumericNormalizationUnverifiedRows') or 0)}",
            f"- blocked input report missing rows: {int(counts.get('blockedInputReportMissingRows') or 0)}",
            f"- blocked input schema violation rows: {int(counts.get('blockedInputSchemaViolationRows') or 0)}",
        "",
        "## Rows",
        *(
            f"- paper={row.get('paper_id','')} status={row.get('readiness_status','')} "
            f"candidate={row.get('source_candidate_id','')}"
            for row in report.get("rows", [])
        ),
    ]
    )


def write_table_cell_structured_evidence_readiness_audit_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "table-cell-structured-evidence-readiness-audit.json"
    summary_path = root / "table-cell-structured-evidence-readiness-audit-summary.json"
    markdown_path = root / "table-cell-structured-evidence-readiness-audit.md"

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = _summary_payload(report)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        render_table_cell_structured_evidence_readiness_audit_markdown(report),
        encoding="utf-8",
    )

    return {
        "report": str(report_path),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(description="Generate a report-only table-cell readiness audit.")
    parser.add_argument("--table-cell-provenance-feasibility-report", default="")
    parser.add_argument("--output-dir")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_table_cell_structured_evidence_readiness_audit(
        table_cell_provenance_feasibility_report=args.table_cell_provenance_feasibility_report or None
    )

    if args.output_dir:
        paths = write_table_cell_structured_evidence_readiness_audit_reports(report, args.output_dir)
        print(f"wrote report: {paths['report']}")
        print(f"wrote summary: {paths['summary']}")
        print(f"wrote markdown: {paths['markdown']}")

    if args.json or not args.output_dir:
        print(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TABLE_CELL_STRUCTURED_EVIDENCE_READINESS_AUDIT_SCHEMA_ID",
    "STATUS_READY",
    "STATUS_BLOCKED_MISSING_TABLE_ID",
    "STATUS_BLOCKED_MISSING_CELL_COORDINATES",
    "STATUS_BLOCKED_MISSING_CELL_TEXT",
    "STATUS_BLOCKED_MISSING_SOURCE_HASH",
    "STATUS_BLOCKED_MISSING_LOCATOR",
    "STATUS_BLOCKED_NUMERIC_NORMALIZATION_UNVERIFIED",
    "STATUS_BLOCKED_INPUT_REPORT_MISSING",
    "STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION",
    "build_table_cell_structured_evidence_readiness_audit",
    "write_table_cell_structured_evidence_readiness_audit_reports",
    "render_table_cell_structured_evidence_readiness_audit_markdown",
    "main",
]
