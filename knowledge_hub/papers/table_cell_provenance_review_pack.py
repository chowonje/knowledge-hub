"""Report-only TableCell provenance review pack helpers.

This module turns table-cell feasibility rows plus recovered table-caption PDF
offsets into operator review cards.  Caption offsets and generated table rows
are useful review inputs, but they do not create table-cell citation evidence:
per-cell bbox, source spans, and cell-level source hash linkage remain missing.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


TABLE_CELL_PROVENANCE_REVIEW_PACK_SCHEMA_ID = (
    "knowledge-hub.paper.table-cell-provenance-review-pack.v1"
)
TABLE_CELL_PROVENANCE_FEASIBILITY_AUDIT_SCHEMA_ID = (
    "knowledge-hub.paper.table-cell-provenance-feasibility-audit.v1"
)
TABLE_REGION_PDF_OFFSET_FEASIBILITY_SCHEMA_ID = (
    "knowledge-hub.paper.table-region-pdf-offset-feasibility.v1"
)


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


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _cell_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [dict(item) for item in list(report.get("rows") or []) if isinstance(item, dict)]
    rows.sort(
        key=lambda item: (
            str(item.get("paper_id") or ""),
            _safe_int(item.get("caption_page")),
            str(item.get("table_region_candidate_id") or ""),
        )
    )
    return rows


def _pdf_rows_by_candidate_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for item in list(report.get("feasibilityRows") or []):
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("source_table_region_candidate_id") or "")
        if candidate_id:
            rows[candidate_id] = dict(item)
    return rows


def _schema_violations(cell_report: dict[str, Any], pdf_report: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if cell_report.get("schema") != TABLE_CELL_PROVENANCE_FEASIBILITY_AUDIT_SCHEMA_ID:
        violations.append("table_cell_provenance_feasibility_schema_mismatch")
    if cell_report.get("status") != "ok":
        violations.append("table_cell_provenance_feasibility_not_ok")
    if pdf_report.get("schema") != TABLE_REGION_PDF_OFFSET_FEASIBILITY_SCHEMA_ID:
        violations.append("table_region_pdf_offset_feasibility_schema_mismatch")
    if pdf_report.get("status") != "feasibility_complete":
        violations.append("table_region_pdf_offset_feasibility_not_complete")
    return violations


def _review_status(cell_row: dict[str, Any], pdf_row: dict[str, Any] | None) -> str:
    if not cell_row.get("normalizer_match"):
        return "held_out_normalizer_table_missing"
    if not cell_row.get("table_structure_available"):
        return "held_out_table_structure_missing"
    if not cell_row.get("row_column_text_available"):
        return "held_out_row_column_text_missing"
    if not cell_row.get("caption_source_span_available") or not bool(
        (pdf_row or {}).get("original_pdf_offset_recovered")
    ):
        return "held_out_caption_source_offset_missing"
    return "ready_for_cell_provenance_review"


def _strict_blockers(cell_row: dict[str, Any], pdf_row: dict[str, Any] | None, status: str) -> list[str]:
    blockers = [
        *[str(value) for value in list(cell_row.get("strict_blockers") or []) if str(value)],
        *[str(value) for value in list((pdf_row or {}).get("strict_blockers") or []) if str(value)],
        "table_cell_provenance_review_pack_only",
        "table_cell_row_column_bbox_source_span_provenance_missing",
        "table_cell_source_hash_linkage_missing",
        "caption_offset_does_not_create_table_cell_evidence",
        "runtime_promotion_disabled_for_tranche",
        "strict_promotion_requires_explicit_later_tranche",
    ]
    if status != "ready_for_cell_provenance_review":
        blockers.append(status)
    return list(dict.fromkeys(blockers))


def _original_pdf_span(pdf_row: dict[str, Any] | None) -> dict[str, Any]:
    span = dict((pdf_row or {}).get("original_pdf_span") or {})
    return {
        "originalPdfCharsStart": span.get("originalPdfCharsStart"),
        "originalPdfCharsEnd": span.get("originalPdfCharsEnd"),
        "page": span.get("page"),
        "sourceContentHash": str(span.get("sourceContentHash") or ""),
        "matchMethod": str(span.get("matchMethod") or ""),
        "matchConfidence": span.get("matchConfidence"),
    }


def _card(index: int, cell_row: dict[str, Any], pdf_row: dict[str, Any] | None) -> dict[str, Any]:
    status = _review_status(cell_row, pdf_row)
    blockers = _strict_blockers(cell_row, pdf_row, status)
    return {
        "review_card_id": f"table-cell-provenance-review:{index:04d}",
        "source_table_cell_feasibility_audit_id": str(cell_row.get("audit_id") or ""),
        "source_table_region_candidate_id": str(cell_row.get("table_region_candidate_id") or ""),
        "source_table_region_pdf_offset_row_id": str((pdf_row or {}).get("feasibility_row_id") or ""),
        "paper_id": str(cell_row.get("paper_id") or ""),
        "candidate_type": "table_cell_provenance_review_card",
        "source_parser": "mineru+pymupdf_alignment",
        "table_label": _clean_text(cell_row.get("table_label")),
        "candidate_text": _clean_text(cell_row.get("candidate_text")),
        "caption_text": _clean_text(cell_row.get("caption_text")),
        "caption_source_span_available": bool(cell_row.get("caption_source_span_available")),
        "caption_original_pdf_offset_recovered": bool((pdf_row or {}).get("original_pdf_offset_recovered")),
        "original_pdf_span": _original_pdf_span(pdf_row),
        "page_agrees_with_canonical": bool((pdf_row or {}).get("page_agrees_with_canonical")),
        "source_hash_agrees_with_canonical": bool((pdf_row or {}).get("source_hash_agrees_with_canonical")),
        "layout_element_count": _safe_int(cell_row.get("layout_element_count")),
        "table_region_bbox_available": bool(cell_row.get("table_region_bbox_available")),
        "normalizer_candidate_id": str(cell_row.get("normalizer_candidate_id") or ""),
        "normalizer_report_path": str(cell_row.get("normalizer_report_path") or ""),
        "normalizer_match": bool(cell_row.get("normalizer_match")),
        "table_structure_available": bool(cell_row.get("table_structure_available")),
        "row_column_text_available": bool(cell_row.get("row_column_text_available")),
        "table_row_count": _safe_int(cell_row.get("table_row_count")),
        "table_max_column_count": _safe_int(cell_row.get("table_max_column_count")),
        "table_cell_count": _safe_int(cell_row.get("table_cell_count")),
        "non_empty_table_cell_count": _safe_int(cell_row.get("non_empty_table_cell_count")),
        "header_like_cell_count": _safe_int(cell_row.get("header_like_cell_count")),
        "rowspan_cell_count": _safe_int(cell_row.get("rowspan_cell_count")),
        "colspan_cell_count": _safe_int(cell_row.get("colspan_cell_count")),
        "cell_bbox_count": _safe_int(cell_row.get("cell_bbox_count")),
        "cell_source_span_count": _safe_int(cell_row.get("cell_source_span_count")),
        "cell_source_hash_count": _safe_int(cell_row.get("cell_source_hash_count")),
        "sample_cells": list(cell_row.get("sample_cells") or []),
        "cell_bbox_available": bool(cell_row.get("cell_bbox_available")),
        "cell_source_span_available": bool(cell_row.get("cell_source_span_available")),
        "cell_source_hash_backed": bool(cell_row.get("cell_source_hash_backed")),
        "table_cell_citation_grade": False,
        "table_cell_evidence_verified": False,
        "cell_level_review_required": True,
        "source_feasibility_status": str(cell_row.get("feasibility_status") or ""),
        "review_status": status,
        "recommended_review_action": (
            "inspect_cells_against_original_pdf_and_define_per_cell_bbox_source_span_authority"
            if status == "ready_for_cell_provenance_review"
            else "hold_until_caption_source_offset_table_structure_and_cell_text_are_available"
        ),
        "evidence_tier": "table_cell_provenance_review_card_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": blockers,
        "non_strict_reason": [
            "review_cards_are_not_evidence",
            "caption_offsets_and_generated_cell_text_do_not_create_cell_level_provenance",
            "per_cell_bbox_source_span_and_source_hash_linkage_are_required",
            "no_runtime_or_strict_evidence_created",
        ],
    }


def _counts(cards: list[dict[str, Any]], schema_violations: list[str]) -> dict[str, Any]:
    blocker_counts: Counter[str] = Counter()
    for item in cards:
        blocker_counts.update(str(blocker) for blocker in list(item.get("strict_blockers") or []))
    ready = [item for item in cards if item.get("review_status") == "ready_for_cell_provenance_review"]
    return {
        "reviewCardRows": len(cards),
        "readyForCellProvenanceReviewRows": len(ready),
        "heldOutRows": len(cards) - len(ready),
        "captionSourceSpanRows": sum(1 for item in cards if item.get("caption_source_span_available")),
        "captionOriginalPdfOffsetRecoveredRows": sum(
            1 for item in cards if item.get("caption_original_pdf_offset_recovered")
        ),
        "tableStructureRows": sum(1 for item in cards if item.get("table_structure_available")),
        "rowColumnTextRows": sum(1 for item in cards if item.get("row_column_text_available")),
        "totalTableRows": sum(_safe_int(item.get("table_row_count")) for item in cards),
        "totalTableCells": sum(_safe_int(item.get("table_cell_count")) for item in cards),
        "nonEmptyTableCells": sum(_safe_int(item.get("non_empty_table_cell_count")) for item in cards),
        "cellBboxRows": sum(1 for item in cards if item.get("cell_bbox_available")),
        "cellSourceSpanRows": sum(1 for item in cards if item.get("cell_source_span_available")),
        "cellSourceHashRows": sum(1 for item in cards if item.get("cell_source_hash_backed")),
        "tableCellEvidenceVerifiedRows": 0,
        "tableCellCitationGradeRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "schemaViolationCount": len(schema_violations),
        "byPaper": dict(Counter(str(item.get("paper_id") or "") for item in cards)),
        "byReviewStatus": dict(Counter(str(item.get("review_status") or "") for item in cards)),
        "bySourceFeasibilityStatus": dict(Counter(str(item.get("source_feasibility_status") or "") for item in cards)),
        "strictBlockerSummary": dict(blocker_counts),
    }


def build_table_cell_provenance_review_pack(
    *,
    table_cell_provenance_feasibility_report: str | Path,
    table_region_pdf_offset_feasibility_report: str | Path,
) -> dict[str, Any]:
    """Build report-only TableCell provenance review cards."""

    cell_path = Path(str(table_cell_provenance_feasibility_report)).expanduser()
    pdf_path = Path(str(table_region_pdf_offset_feasibility_report)).expanduser()
    cell_report = _read_json(cell_path)
    pdf_report = _read_json(pdf_path)
    schema_violations = _schema_violations(cell_report, pdf_report)
    pdf_by_id = _pdf_rows_by_candidate_id(pdf_report)
    cards = [
        _card(index + 1, row, pdf_by_id.get(str(row.get("table_region_candidate_id") or "")))
        for index, row in enumerate(_cell_rows(cell_report))
    ]
    counts = _counts(cards, schema_violations)
    blocked = bool(schema_violations) or not cards
    return {
        "schema": TABLE_CELL_PROVENANCE_REVIEW_PACK_SCHEMA_ID,
        "status": "blocked" if blocked else "review_pack_ready",
        "generatedAt": _now(),
        "inputs": {
            "tableCellProvenanceFeasibilityReport": str(cell_path),
            "tableRegionPdfOffsetFeasibilityReport": str(pdf_path),
            "tableCellProvenanceFeasibilitySchema": str(cell_report.get("schema") or ""),
            "tableRegionPdfOffsetFeasibilitySchema": str(pdf_report.get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "reviewPackReady": not blocked,
            "tableCellProvenanceReviewReady": not blocked and bool(cards),
            "tableCellCitationGradeReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "ready_for_table_cell_provenance_human_review" if not blocked else "blocked",
            "schemaViolations": schema_violations,
            "recommendedNextTranche": "table_cell_bbox_source_span_authority_design",
        },
        "policy": {
            "reportOnly": True,
            "reviewOnly": True,
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
            "review_cards_are_not_runtime_evidence",
            "caption_offsets_do_not_create_table_cell_citation_grade_evidence",
            "generated_mineru_table_rows_need_per_cell_bbox_source_span_and_hash_authority",
            "no_strict_evidence_or_parser_routing_is_created",
        ],
        "reviewCards": cards,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings", "reviewCards")
        if key in report
    }


def render_table_cell_provenance_review_pack_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# TableCell Provenance Review Pack",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Review cards: `{int(counts.get('reviewCardRows') or 0)}`",
        f"- Ready for cell provenance review: `{int(counts.get('readyForCellProvenanceReviewRows') or 0)}`",
        f"- Held out: `{int(counts.get('heldOutRows') or 0)}`",
        f"- Caption original PDF offsets recovered: `{int(counts.get('captionOriginalPdfOffsetRecoveredRows') or 0)}`",
        f"- Total table rows: `{int(counts.get('totalTableRows') or 0)}`",
        f"- Total table cells: `{int(counts.get('totalTableCells') or 0)}`",
        f"- Cell bbox rows: `{int(counts.get('cellBboxRows') or 0)}`",
        f"- Cell source span rows: `{int(counts.get('cellSourceSpanRows') or 0)}`",
        f"- Table-cell citation-grade rows: `{int(counts.get('tableCellCitationGradeRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "These cards are for human/operator review only. Recovered table-caption offsets and MinerU-generated table rows do not verify per-cell provenance. No row is strict evidence, citation-grade table-cell evidence, or runtime citation material.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By review status: `{json.dumps(counts.get('byReviewStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By source feasibility status: `{json.dumps(counts.get('bySourceFeasibilityStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Review Cards",
        "",
    ]
    for item in list(report.get("reviewCards") or []):
        lines.append(
            f"- `{item.get('paper_id')}` `{item.get('table_label')}` "
            f"rows `{item.get('table_row_count')}` cells `{item.get('table_cell_count')}` "
            f"-> `{item.get('review_status')}`"
        )
    lines.append("")
    return "\n".join(lines)


def write_table_cell_provenance_review_pack_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    cards_path = root / "table-cell-provenance-review-cards.json"
    summary_path = root / "table-cell-provenance-review-summary.json"
    markdown_path = root / "table-cell-provenance-review-pack.md"
    cards_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_table_cell_provenance_review_pack_markdown(report), encoding="utf-8")
    return {"cards": str(cards_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate report-only TableCell provenance review cards.")
    parser.add_argument("--table-cell-provenance-feasibility-report", required=True)
    parser.add_argument("--table-region-pdf-offset-feasibility-report", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_table_cell_provenance_review_pack(
        table_cell_provenance_feasibility_report=args.table_cell_provenance_feasibility_report,
        table_region_pdf_offset_feasibility_report=args.table_region_pdf_offset_feasibility_report,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_table_cell_provenance_review_pack_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TABLE_CELL_PROVENANCE_REVIEW_PACK_SCHEMA_ID",
    "build_table_cell_provenance_review_pack",
    "render_table_cell_provenance_review_pack_markdown",
    "write_table_cell_provenance_review_pack_reports",
]
