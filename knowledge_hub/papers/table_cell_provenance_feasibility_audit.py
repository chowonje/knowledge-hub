"""Report-only table-cell provenance feasibility audit.

This helper checks whether TableRegionCandidate rows can be upgraded toward
table-cell evidence.  It may find MinerU-generated row/column text, but it
does not create citation-grade cell evidence because per-cell bbox, canonical
source spans, and cell-level source hashes are still unavailable.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


TABLE_CELL_PROVENANCE_FEASIBILITY_AUDIT_SCHEMA_ID = (
    "knowledge-hub.paper.table-cell-provenance-feasibility-audit.v1"
)
TABLE_REGION_CANDIDATE_REPORT_SCHEMA_ID = "knowledge-hub.paper.table-region-candidate-report.v1"
MINERU_SOURCE_ALIGNMENT_AUDIT_SCHEMA_ID = "knowledge-hub.paper.mineru-source-alignment-audit.v1"
MINERU_NORMALIZER_AUDIT_SCHEMA_ID = "knowledge-hub.paper.mineru-normalizer-audit.v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _candidate_rows(table_region_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in list(table_region_report.get("candidates") or []):
        if isinstance(item, dict) and item.get("candidate_type") == "table_region_candidate":
            rows.append(dict(item))
    rows.sort(
        key=lambda item: (
            str(item.get("paper_id") or ""),
            _safe_int(item.get("page")) or 0,
            _safe_int(item.get("chars_start")) or 0,
            str(item.get("candidate_id") or ""),
        )
    )
    return rows


def _normalizer_paths(source_alignment_report: dict[str, Any]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for paper in list(source_alignment_report.get("papers") or []):
        if not isinstance(paper, dict):
            continue
        paper_id = str(paper.get("paperId") or "")
        path = str((paper.get("input") or {}).get("mineruNormalizerCandidatesPath") or "")
        if paper_id and path:
            paths[paper_id] = path
    return paths


def _normalizer_tables(paths_by_paper: dict[str, str]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    tables: dict[str, dict[str, Any]] = {}
    schemas: dict[str, str] = {}
    for paper_id, path in sorted(paths_by_paper.items()):
        payload = _read_json(path)
        schemas[paper_id] = str(payload.get("schema") or "")
        for item in list(payload.get("candidates") or []):
            if not isinstance(item, dict) or item.get("candidate_type") != "table_candidate":
                continue
            candidate_id = str(item.get("candidate_id") or "")
            if candidate_id:
                row = dict(item)
                row["_normalizer_path"] = path
                row["_normalizer_schema"] = schemas[paper_id]
                tables[candidate_id] = row
    return tables, schemas


def _cell_value(cell: Any, key: str) -> Any:
    return cell.get(key) if isinstance(cell, dict) else None


def _cell_has_bbox(cell: Any) -> bool:
    value = _cell_value(cell, "bbox")
    return isinstance(value, list) and len(value) >= 4


def _cell_has_source_span(cell: Any) -> bool:
    if not isinstance(cell, dict):
        return False
    if cell.get("chars_start") is not None and cell.get("chars_end") is not None:
        return True
    locator = cell.get("source_span_locator")
    chars = locator.get("chars") if isinstance(locator, dict) else None
    return isinstance(chars, dict) and chars.get("start") is not None and chars.get("end") is not None


def _cell_has_source_hash(cell: Any) -> bool:
    return bool(isinstance(cell, dict) and str(cell.get("sourceContentHash") or "").strip())


def _table_structure(normalizer: dict[str, Any] | None) -> dict[str, Any]:
    rows = list((normalizer or {}).get("tableRows") or [])
    row_count = 0
    max_columns = 0
    cell_count = 0
    non_empty = 0
    header_like = 0
    rowspan_cells = 0
    colspan_cells = 0
    cell_bbox_count = 0
    cell_source_span_count = 0
    cell_source_hash_count = 0
    sample_cells: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, list):
            continue
        row_count += 1
        max_columns = max(max_columns, len(row))
        cell_count += len(row)
        for cell in row:
            text = _clean_text(_cell_value(cell, "text"))
            if text:
                non_empty += 1
            if bool(_cell_value(cell, "isHeader")):
                header_like += 1
            if (_safe_int(_cell_value(cell, "rowspan")) or 1) > 1:
                rowspan_cells += 1
            if (_safe_int(_cell_value(cell, "colspan")) or 1) > 1:
                colspan_cells += 1
            if _cell_has_bbox(cell):
                cell_bbox_count += 1
            if _cell_has_source_span(cell):
                cell_source_span_count += 1
            if _cell_has_source_hash(cell):
                cell_source_hash_count += 1
            if text and len(sample_cells) < 8:
                sample_cells.append(
                    {
                        "text": text,
                        "rowspan": _safe_int(_cell_value(cell, "rowspan")) or 1,
                        "colspan": _safe_int(_cell_value(cell, "colspan")) or 1,
                        "has_bbox": _cell_has_bbox(cell),
                        "has_source_span": _cell_has_source_span(cell),
                        "has_source_hash": _cell_has_source_hash(cell),
                    }
                )
    return {
        "row_count": row_count,
        "max_column_count": max_columns,
        "cell_count": cell_count,
        "non_empty_cell_count": non_empty,
        "header_like_cell_count": header_like,
        "rowspan_cell_count": rowspan_cells,
        "colspan_cell_count": colspan_cells,
        "cell_bbox_count": cell_bbox_count,
        "cell_source_span_count": cell_source_span_count,
        "cell_source_hash_count": cell_source_hash_count,
        "sample_cells": sample_cells,
    }


def _has_caption_source_span(item: dict[str, Any]) -> bool:
    return (
        item.get("canonical_alignment_status") == "aligned"
        and item.get("chars_start") is not None
        and item.get("chars_end") is not None
        and item.get("page") is not None
        and bool(str(item.get("sourceContentHash") or "").strip())
    )


def _feasibility_status(
    *,
    normalizer_match: bool,
    table_structure_available: bool,
    row_column_text_available: bool,
    caption_source_span_available: bool,
    cell_bbox_available: bool,
    cell_source_span_available: bool,
    cell_source_hash_backed: bool,
) -> str:
    if not normalizer_match:
        return "blocked_missing_normalizer_table"
    if not table_structure_available:
        return "blocked_missing_table_rows"
    if not row_column_text_available:
        return "blocked_missing_cell_text"
    if not caption_source_span_available:
        return "table_structure_candidate_caption_alignment_blocked"
    if not (cell_bbox_available and cell_source_span_available and cell_source_hash_backed):
        return "table_structure_candidate_no_cell_provenance"
    return "cell_text_structure_candidate_non_strict"


def _strict_blockers(
    item: dict[str, Any],
    *,
    normalizer_match: bool,
    table_structure_available: bool,
    row_column_text_available: bool,
    cell_bbox_available: bool,
    cell_source_span_available: bool,
    cell_source_hash_backed: bool,
) -> list[str]:
    blockers = [str(value) for value in list(item.get("strict_blockers") or []) if str(value)]
    required = [
        "table_cell_provenance_feasibility_audit_only",
        "runtime_promotion_disabled_for_tranche",
        "table_cell_row_column_bbox_provenance_missing",
        "table_cell_text_without_cell_provenance_not_citation_grade",
        "generated_mineru_markdown_rows_are_not_original_source_spans",
    ]
    if not normalizer_match:
        required.append("missing_normalizer_table_candidate")
    if not table_structure_available:
        required.append("missing_table_rows")
    if not row_column_text_available:
        required.append("missing_row_column_cell_text")
    if not cell_bbox_available:
        required.append("table_cell_bbox_missing")
    if not cell_source_span_available:
        required.append("table_cell_chars_start_end_missing")
    if not cell_source_hash_backed:
        required.append("table_cell_source_content_hash_missing")
    if not _has_caption_source_span(item):
        required.append("table_caption_source_span_incomplete")
    return list(dict.fromkeys([*blockers, *required]))


def _row(index: int, item: dict[str, Any], normalizer_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source_candidate_id = str(item.get("source_candidate_id") or "")
    normalizer = normalizer_by_id.get(source_candidate_id)
    structure = _table_structure(normalizer)
    normalizer_match = normalizer is not None
    table_structure_available = structure["row_count"] > 0
    row_column_text_available = structure["non_empty_cell_count"] > 0
    cell_bbox_available = structure["cell_bbox_count"] > 0
    cell_source_span_available = structure["cell_source_span_count"] > 0
    cell_source_hash_backed = structure["cell_source_hash_count"] > 0
    caption_source_span_available = _has_caption_source_span(item)
    status = _feasibility_status(
        normalizer_match=normalizer_match,
        table_structure_available=table_structure_available,
        row_column_text_available=row_column_text_available,
        caption_source_span_available=caption_source_span_available,
        cell_bbox_available=cell_bbox_available,
        cell_source_span_available=cell_source_span_available,
        cell_source_hash_backed=cell_source_hash_backed,
    )
    blockers = _strict_blockers(
        item,
        normalizer_match=normalizer_match,
        table_structure_available=table_structure_available,
        row_column_text_available=row_column_text_available,
        cell_bbox_available=cell_bbox_available,
        cell_source_span_available=cell_source_span_available,
        cell_source_hash_backed=cell_source_hash_backed,
    )
    return {
        "audit_id": f"table-cell-provenance-feasibility:{index:04d}",
        "table_region_candidate_id": str(item.get("candidate_id") or ""),
        "source_candidate_id": source_candidate_id,
        "paper_id": str(item.get("paper_id") or ""),
        "candidate_type": "table_cell_provenance_feasibility_candidate",
        "source_parser": "mineru+pymupdf_alignment",
        "candidate_text": _clean_text(item.get("candidate_text")),
        "table_label": str(item.get("table_label") or ""),
        "caption_text": _clean_text(item.get("caption_text")),
        "caption_alignment_status": str(item.get("canonical_alignment_status") or ""),
        "caption_alignment_method": str(item.get("alignment_method") or ""),
        "caption_chars_start": item.get("chars_start"),
        "caption_chars_end": item.get("chars_end"),
        "caption_page": item.get("page"),
        "sourceContentHash": str(item.get("sourceContentHash") or ""),
        "sourceContentHashSource": str(item.get("sourceContentHashSource") or ""),
        "caption_source_span_available": caption_source_span_available,
        "table_region_bbox_available": bool(item.get("bbox")),
        "layout_element_count": len(list(item.get("layout_element_ids") or [])),
        "normalizer_candidate_id": str((normalizer or {}).get("candidate_id") or ""),
        "normalizer_report_path": str((normalizer or {}).get("_normalizer_path") or ""),
        "normalizer_schema": str((normalizer or {}).get("_normalizer_schema") or ""),
        "normalizer_match": normalizer_match,
        "table_structure_available": table_structure_available,
        "row_column_text_available": row_column_text_available,
        "table_row_count": structure["row_count"],
        "table_max_column_count": structure["max_column_count"],
        "table_cell_count": structure["cell_count"],
        "non_empty_table_cell_count": structure["non_empty_cell_count"],
        "header_like_cell_count": structure["header_like_cell_count"],
        "rowspan_cell_count": structure["rowspan_cell_count"],
        "colspan_cell_count": structure["colspan_cell_count"],
        "cell_bbox_count": structure["cell_bbox_count"],
        "cell_source_span_count": structure["cell_source_span_count"],
        "cell_source_hash_count": structure["cell_source_hash_count"],
        "sample_cells": structure["sample_cells"],
        "cell_bbox_available": cell_bbox_available,
        "cell_source_span_available": cell_source_span_available,
        "cell_source_hash_backed": cell_source_hash_backed,
        "table_cell_citation_grade": False,
        "feasibility_status": status,
        "confidence": float(item.get("confidence") or 0.0),
        "evidence_tier": "table_cell_provenance_feasibility_candidate_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": blockers,
        "non_strict_reason": [
            "table_cell_provenance_feasibility_rows_are_not_evidence",
            "cell_text_without_cell_bbox_source_span_and_source_hash_is_not_citation_grade",
            "later_explicit_promotion_tranche_required",
        ],
    }


def _schema_violations(
    table_region_report: dict[str, Any],
    source_alignment_report: dict[str, Any],
    normalizer_schemas: dict[str, str],
) -> list[str]:
    violations: list[str] = []
    if table_region_report.get("schema") != TABLE_REGION_CANDIDATE_REPORT_SCHEMA_ID:
        violations.append("table_region_candidate_report_schema_mismatch")
    if source_alignment_report.get("schema") != MINERU_SOURCE_ALIGNMENT_AUDIT_SCHEMA_ID:
        violations.append("mineru_source_alignment_report_schema_mismatch")
    for paper_id, schema in sorted(normalizer_schemas.items()):
        if schema and schema != MINERU_NORMALIZER_AUDIT_SCHEMA_ID:
            violations.append(f"mineru_normalizer_schema_mismatch:{paper_id}")
    return violations


def _counts(rows: list[dict[str, Any]], schema_violations: list[str]) -> dict[str, Any]:
    by_status = Counter(str(item.get("feasibility_status") or "") for item in rows)
    by_paper = Counter(str(item.get("paper_id") or "") for item in rows)
    blocker_counts: Counter[str] = Counter()
    for item in rows:
        blocker_counts.update(str(blocker) for blocker in list(item.get("strict_blockers") or []))
    return {
        "inputTableRegionCandidates": len(rows),
        "auditedTableRegionCandidates": len(rows),
        "normalizerTableMatches": sum(1 for item in rows if item.get("normalizer_match")),
        "tableStructureCandidates": sum(1 for item in rows if item.get("table_structure_available")),
        "rowColumnTextCandidates": sum(1 for item in rows if item.get("row_column_text_available")),
        "totalTableRows": sum(int(item.get("table_row_count") or 0) for item in rows),
        "totalTableCells": sum(int(item.get("table_cell_count") or 0) for item in rows),
        "nonEmptyTableCells": sum(int(item.get("non_empty_table_cell_count") or 0) for item in rows),
        "cellBboxCandidates": sum(1 for item in rows if item.get("cell_bbox_available")),
        "cellSourceSpanCandidates": sum(1 for item in rows if item.get("cell_source_span_available")),
        "cellSourceHashBackedCandidates": sum(1 for item in rows if item.get("cell_source_hash_backed")),
        "tableCellCitationGradeCandidates": 0,
        "strictEligibleCandidates": 0,
        "citationGradeCandidates": 0,
        "runtimeEvidenceCandidates": 0,
        "schemaViolationCount": len(schema_violations),
        "byPaper": dict(by_paper),
        "byFeasibilityStatus": dict(by_status),
        "strictBlockerSummary": dict(blocker_counts),
    }


def build_table_cell_provenance_feasibility_audit(
    *,
    table_region_report: str | Path,
    mineru_source_alignment_report: str | Path,
) -> dict[str, Any]:
    """Build a report-only feasibility audit for table-cell provenance."""

    table_region_path = Path(str(table_region_report)).expanduser()
    source_alignment_path = Path(str(mineru_source_alignment_report)).expanduser()
    table_region_payload = _read_json(table_region_path)
    source_alignment_payload = _read_json(source_alignment_path)
    normalizer_paths = _normalizer_paths(source_alignment_payload)
    normalizer_tables, normalizer_schemas = _normalizer_tables(normalizer_paths)
    schema_violations = _schema_violations(table_region_payload, source_alignment_payload, normalizer_schemas)
    candidates = _candidate_rows(table_region_payload)
    rows = [_row(index + 1, item, normalizer_tables) for index, item in enumerate(candidates)]
    counts = _counts(rows, schema_violations)
    status = "blocked" if schema_violations else "ok"
    return {
        "schema": TABLE_CELL_PROVENANCE_FEASIBILITY_AUDIT_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "tableRegionReport": str(table_region_path),
            "mineruSourceAlignmentReport": str(source_alignment_path),
            "tableRegionSchema": str(table_region_payload.get("schema") or ""),
            "mineruSourceAlignmentSchema": str(source_alignment_payload.get("schema") or ""),
            "mineruNormalizerReportPaths": normalizer_paths,
            "mineruNormalizerSchemas": normalizer_schemas,
        },
        "counts": counts,
        "gate": {
            "tableCellProvenanceReviewed": not schema_violations,
            "tableCellCitationGradeReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "decision": "blocked" if schema_violations else "table_cell_provenance_feasibility_reviewed",
            "schemaViolations": schema_violations,
            "recommendedNextTranche": "figure_region_link_feasibility_audit",
        },
        "policy": {
            "auditOnly": True,
            "tableRowsCandidateOnly": True,
            "tableCellEvidenceCreated": False,
            "tableCellCitationGradeEvidenceCreated": False,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "mineru_table_rows_are_generated_parser_candidates_only",
            "cell_text_without_cell_bbox_source_span_and_source_hash_is_not_citation_grade",
            "table_caption_or_region_candidates_do_not_answer_numeric_questions",
            "no_parser_routing_or_runtime_answer_integration_is_changed",
        ],
        "rows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings")
        if key in report
    }


def render_table_cell_provenance_feasibility_audit_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Table Cell Provenance Feasibility Audit",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Audited table region candidates: `{int(counts.get('auditedTableRegionCandidates') or 0)}`",
        f"- Normalizer table matches: `{int(counts.get('normalizerTableMatches') or 0)}`",
        f"- Table structure candidates: `{int(counts.get('tableStructureCandidates') or 0)}`",
        f"- Row/column text candidates: `{int(counts.get('rowColumnTextCandidates') or 0)}`",
        f"- Total table rows: `{int(counts.get('totalTableRows') or 0)}`",
        f"- Total table cells: `{int(counts.get('totalTableCells') or 0)}`",
        f"- Non-empty table cells: `{int(counts.get('nonEmptyTableCells') or 0)}`",
        f"- Cell bbox candidates: `{int(counts.get('cellBboxCandidates') or 0)}`",
        f"- Cell source span candidates: `{int(counts.get('cellSourceSpanCandidates') or 0)}`",
        f"- Table-cell citation-grade candidates: `{int(counts.get('tableCellCitationGradeCandidates') or 0)}`",
        f"- Strict eligible: `{int(counts.get('strictEligibleCandidates') or 0)}`",
        "",
        "## Evidence Tier",
        "",
        "All rows are `table_cell_provenance_feasibility_candidate_only`. They are not strict evidence.",
        "MinerU table rows and cell text are useful review signals, but they are not table-cell citation-grade evidence without per-cell bbox, canonical source spans, and cell-level source hash linkage.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By feasibility status: `{json.dumps(counts.get('byFeasibilityStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Rows",
        "",
    ]
    for item in list(report.get("rows") or []):
        lines.append(
            f"- `{item.get('paper_id')}` `{item.get('table_label')}` rows `{item.get('table_row_count')}` "
            f"cells `{item.get('table_cell_count')}` -> `{item.get('feasibility_status')}`"
        )
    lines.append("")
    return "\n".join(lines)


def write_table_cell_provenance_feasibility_audit_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    audit_path = root / "table-cell-provenance-feasibility-audit.json"
    summary_path = root / "table-cell-provenance-feasibility-summary.json"
    markdown_path = root / "table-cell-provenance-feasibility-audit.md"
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_table_cell_provenance_feasibility_audit_markdown(report), encoding="utf-8")
    return {"audit": str(audit_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only table-cell provenance feasibility audit.")
    parser.add_argument("--table-region-report", required=True, help="Path to table-region-candidates.json.")
    parser.add_argument(
        "--mineru-source-alignment-report",
        required=True,
        help="Path to mineru-source-alignment-report.json.",
    )
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_table_cell_provenance_feasibility_audit(
        table_region_report=args.table_region_report,
        mineru_source_alignment_report=args.mineru_source_alignment_report,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_table_cell_provenance_feasibility_audit_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TABLE_CELL_PROVENANCE_FEASIBILITY_AUDIT_SCHEMA_ID",
    "build_table_cell_provenance_feasibility_audit",
    "render_table_cell_provenance_feasibility_audit_markdown",
    "write_table_cell_provenance_feasibility_audit_reports",
]
