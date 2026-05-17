"""Report-only TableCell bbox/source-span extractor pilot.

This helper probes whether the current local PyMuPDF runtime can detect table
cell bbox candidates on the bounded set of TableCell authority-design rows.
Detected cells remain diagnostic-only: no cell source spans, citation-grade
table evidence, runtime citations, parser routing, DB/index changes, or
canonical parsed-artifact writes are created here.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import io
import json
from pathlib import Path
from typing import Any, Callable

from knowledge_hub.papers.sectionspan_pdf_offset_recovery_dry_run import (
    _exact_matches,
    _extract_pdf_pages,
    _normalized_matches,
    _safe_int,
    _with_offsets,
)


TABLE_CELL_BBOX_SOURCE_SPAN_EXTRACTOR_PILOT_SCHEMA_ID = (
    "knowledge-hub.paper.table-cell-bbox-source-span-extractor-pilot.v1"
)
TABLE_CELL_BBOX_SOURCE_SPAN_AUTHORITY_DESIGN_SCHEMA_ID = (
    "knowledge-hub.paper.table-cell-bbox-source-span-authority-design.v1"
)
TABLE_REGION_PDF_OFFSET_FEASIBILITY_SCHEMA_ID = (
    "knowledge-hub.paper.table-region-pdf-offset-feasibility.v1"
)


TableProbeLoader = Callable[[str | Path, int], dict[str, Any]]
PageTextLoader = Callable[[str | Path], list[dict[str, Any]]]


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


def _bbox(value: Any) -> list[float] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
        except Exception:
            return None
    return None


def _source_rows_by_candidate_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for item in list(report.get("feasibilityRows") or []):
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("source_table_region_candidate_id") or "")
        if candidate_id:
            rows[candidate_id] = dict(item)
    return rows


def _schema_violations(authority_design: dict[str, Any], source_report: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if authority_design.get("schema") != TABLE_CELL_BBOX_SOURCE_SPAN_AUTHORITY_DESIGN_SCHEMA_ID:
        violations.append("table_cell_bbox_source_span_authority_design_schema_mismatch")
    if authority_design.get("status") != "design_ready":
        violations.append("table_cell_bbox_source_span_authority_design_not_ready")
    if source_report.get("schema") != TABLE_REGION_PDF_OFFSET_FEASIBILITY_SCHEMA_ID:
        violations.append("table_region_pdf_offset_feasibility_schema_mismatch")
    if source_report.get("status") != "feasibility_complete":
        violations.append("table_region_pdf_offset_feasibility_not_complete")
    return violations


def _unsafe_flags(authority_design: dict[str, Any]) -> list[str]:
    unsafe: list[str] = []
    counts = dict(authority_design.get("counts") or {})
    gate = dict(authority_design.get("gate") or {})
    policy = dict(authority_design.get("policy") or {})
    for key in (
        "tableCellEvidenceReadyRows",
        "tableCellCitationGradeRows",
        "strictEligibleRows",
        "citationGradeRows",
        "runtimeEvidenceRows",
    ):
        if _safe_int(counts.get(key)) > 0:
            unsafe.append(f"{key}_nonzero")
    for key in (
        "authorityDecisionMade",
        "tableCellEvidenceReady",
        "tableCellCitationGradeReady",
        "strictEvidenceReady",
        "parserRoutingReady",
        "answerIntegrationReady",
        "runtimePromotionAllowed",
    ):
        if bool(gate.get(key)):
            unsafe.append(f"{key}_true")
    for key in (
        "authorityDecisionMade",
        "tableCellEvidenceCreated",
        "tableCellCitationGradeEvidenceCreated",
        "strictEvidenceCreated",
        "runtimePromotionAllowed",
        "parserRoutingChanged",
        "canonicalParsedArtifactsWritten",
        "databaseMutation",
        "reindexOrReembed",
        "answerIntegrationChanged",
    ):
        if bool(policy.get(key)):
            unsafe.append(f"{key}_true")
    return list(dict.fromkeys(unsafe))


def _default_table_probe(source_pdf: str | Path, page_number: int) -> dict[str, Any]:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        return {"status": "blocked_pymupdf_unavailable", "failureReason": str(exc), "tables": []}
    path = Path(str(source_pdf)).expanduser()
    if not path.exists():
        return {"status": "blocked_source_pdf_missing", "failureReason": str(path), "tables": []}
    try:
        document = fitz.open(str(path))
    except Exception as exc:
        return {"status": "blocked_source_pdf_open_failed", "failureReason": str(exc), "tables": []}
    try:
        if page_number < 1 or page_number > int(getattr(document, "page_count", 0) or 0):
            return {"status": "blocked_page_out_of_range", "failureReason": str(page_number), "tables": []}
        page = document.load_page(page_number - 1)
        if not hasattr(page, "find_tables"):
            return {"status": "blocked_pymupdf_find_tables_unavailable", "failureReason": "", "tables": []}
        try:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = page.find_tables()
        except Exception as exc:
            return {"status": "failed_find_tables", "failureReason": str(exc), "tables": []}
        tables: list[dict[str, Any]] = []
        for index, table in enumerate(list(getattr(result, "tables", []) or []), start=1):
            cells = list(getattr(table, "cells", []) or [])
            extracted: list[list[str]] = []
            try:
                extracted_raw = table.extract()
                extracted = [
                    [str(cell or "") for cell in list(row or [])]
                    for row in list(extracted_raw or [])
                    if isinstance(row, list)
                ]
            except Exception:
                extracted = []
            tables.append(
                {
                    "table_index": index,
                    "bbox": _bbox(getattr(table, "bbox", None)),
                    "row_count": _safe_int(getattr(table, "row_count", 0)),
                    "column_count": _safe_int(getattr(table, "col_count", 0)),
                    "cell_bbox_count": sum(1 for item in cells if _bbox(item) is not None),
                    "cell_bboxes_sample": [_bbox(item) for item in cells[:8] if _bbox(item) is not None],
                    "extracted_rows": extracted,
                }
            )
        return {"status": "ok", "failureReason": "", "tables": tables}
    finally:
        try:
            document.close()
        except Exception:
            pass


def _page_text_context(
    *,
    source_pdf: str | Path,
    page_number: int,
    page_text_loader: PageTextLoader,
) -> dict[str, Any]:
    pages = _with_offsets(page_text_loader(source_pdf))
    for page in pages:
        if _safe_int(page.get("page")) == page_number:
            return page
    return {}


def _extracted_cell_texts(selected_table: dict[str, Any] | None) -> list[str]:
    if not selected_table:
        return []
    values: list[str] = []
    for row in list(selected_table.get("extracted_rows") or []):
        if not isinstance(row, list):
            continue
        for cell in row:
            text = _clean_text(cell)
            if text:
                values.append(text)
    return values


def _diagnostic_cell_text_matches(
    *,
    selected_table: dict[str, Any] | None,
    page_text: dict[str, Any],
) -> dict[str, Any]:
    cells = _extracted_cell_texts(selected_table)
    if not cells or not page_text:
        return {
            "cell_text_candidates": len(cells),
            "diagnostic_unique_cell_text_matches": 0,
            "diagnostic_ambiguous_cell_text_matches": 0,
            "diagnostic_no_match_cell_texts": len(cells),
            "sample_matches": [],
        }
    sample: list[dict[str, Any]] = []
    unique = 0
    ambiguous = 0
    missing = 0
    for text in cells[:50]:
        matches = _exact_matches([page_text], text)
        method = "exact"
        if not matches:
            matches = _normalized_matches([page_text], text)
            method = "normalized_whitespace_case"
        if len(matches) == 1:
            unique += 1
            if len(sample) < 8:
                match = dict(matches[0])
                sample.append(
                    {
                        "cell_text": text,
                        "method": method,
                        "page": match.get("page"),
                        "chars_start": match.get("chars_start"),
                        "chars_end": match.get("chars_end"),
                        "strict_source_span": False,
                    }
                )
        elif len(matches) > 1:
            ambiguous += 1
        else:
            missing += 1
    return {
        "cell_text_candidates": len(cells),
        "diagnostic_unique_cell_text_matches": unique,
        "diagnostic_ambiguous_cell_text_matches": ambiguous,
        "diagnostic_no_match_cell_texts": missing,
        "sample_matches": sample,
    }


def _selected_table(tables: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not tables:
        return None
    return max(tables, key=lambda item: (_safe_int(item.get("cell_bbox_count")), _safe_int(item.get("row_count"))))


def _probe_status(row: dict[str, Any], source_row: dict[str, Any] | None, probe: dict[str, Any]) -> str:
    if row.get("authority_design_status") != "ready_for_cell_bbox_source_span_authority_design":
        return "held_out_authority_design_not_ready"
    if not source_row:
        return "blocked_source_pdf_context_missing"
    if not bool(row.get("caption_original_pdf_offset_recovered")):
        return "blocked_caption_source_offset_missing"
    if _safe_int((row.get("original_pdf_span") or {}).get("page")) <= 0:
        return "blocked_page_missing"
    status = str(probe.get("status") or "")
    if status != "ok":
        return status or "failed_table_probe"
    tables = list(probe.get("tables") or [])
    if not tables:
        return "no_tables_detected"
    if sum(_safe_int(item.get("cell_bbox_count")) for item in tables) <= 0:
        return "tables_detected_without_cell_bboxes"
    return "cell_bbox_candidates_detected_non_strict"


def _pilot_row(
    index: int,
    row: dict[str, Any],
    source_row: dict[str, Any] | None,
    *,
    table_probe_loader: TableProbeLoader,
    page_text_loader: PageTextLoader,
    should_probe: bool,
) -> dict[str, Any]:
    page_number = _safe_int((row.get("original_pdf_span") or {}).get("page"))
    source_pdf = str((source_row or {}).get("source_pdf_path") or "")
    probe = (
        table_probe_loader(source_pdf, page_number)
        if should_probe and source_pdf and page_number > 0
        else {"status": "not_probed", "failureReason": "", "tables": []}
    )
    tables = list(probe.get("tables") or [])
    selected = _selected_table([dict(item) for item in tables if isinstance(item, dict)])
    page_text = (
        _page_text_context(source_pdf=source_pdf, page_number=page_number, page_text_loader=page_text_loader)
        if should_probe and source_pdf and page_number > 0 and selected
        else {}
    )
    cell_match_stats = _diagnostic_cell_text_matches(selected_table=selected, page_text=page_text)
    status = (
        "held_out_max_ready_rows_exceeded"
        if (
            row.get("authority_design_status") == "ready_for_cell_bbox_source_span_authority_design"
            and not should_probe
        )
        else _probe_status(row, source_row, probe)
    )
    return {
        "pilot_row_id": f"table-cell-extractor-pilot:{index:04d}",
        "source_authority_design_id": str(row.get("design_id") or ""),
        "source_review_card_id": str(row.get("source_review_card_id") or ""),
        "source_table_region_candidate_id": str(row.get("source_table_region_candidate_id") or ""),
        "paper_id": str(row.get("paper_id") or ""),
        "candidate_type": "table_cell_bbox_source_span_extractor_pilot_row",
        "source_parser": "pymupdf_table_probe",
        "table_label": _clean_text(row.get("table_label")),
        "candidate_text": _clean_text(row.get("candidate_text")),
        "caption_original_pdf_offset_recovered": bool(row.get("caption_original_pdf_offset_recovered")),
        "source_pdf_path": source_pdf,
        "page": page_number or None,
        "sourceContentHash": str((row.get("original_pdf_span") or {}).get("sourceContentHash") or ""),
        "authority_design_status": str(row.get("authority_design_status") or ""),
        "probe_attempted": should_probe and source_pdf and page_number > 0,
        "probe_status": status,
        "probe_failure_reason": str(probe.get("failureReason") or ""),
        "detected_table_count": len(tables),
        "selected_table_index": _safe_int((selected or {}).get("table_index")),
        "selected_table_bbox": (selected or {}).get("bbox"),
        "selected_table_row_count": _safe_int((selected or {}).get("row_count")),
        "selected_table_column_count": _safe_int((selected or {}).get("column_count")),
        "selected_table_cell_bbox_count": _safe_int((selected or {}).get("cell_bbox_count")),
        "selected_table_cell_text_count": int(cell_match_stats.get("cell_text_candidates") or 0),
        "diagnostic_unique_cell_text_matches": int(cell_match_stats.get("diagnostic_unique_cell_text_matches") or 0),
        "diagnostic_ambiguous_cell_text_matches": int(
            cell_match_stats.get("diagnostic_ambiguous_cell_text_matches") or 0
        ),
        "diagnostic_no_match_cell_texts": int(cell_match_stats.get("diagnostic_no_match_cell_texts") or 0),
        "diagnostic_cell_text_matches_are_strict": False,
        "sample_cell_bboxes": list((selected or {}).get("cell_bboxes_sample") or []),
        "sample_cell_text_matches": list(cell_match_stats.get("sample_matches") or []),
        "table_cell_bbox_candidates_found": bool(
            status == "cell_bbox_candidates_detected_non_strict"
            and _safe_int((selected or {}).get("cell_bbox_count")) > 0
        ),
        "cell_bbox_text_pairing_verified": False,
        "cell_source_spans_created": 0,
        "cell_source_hash_linkages_created": 0,
        "table_cell_evidence_created": False,
        "table_cell_citation_grade": False,
        "evidence_tier": "table_cell_extractor_pilot_diagnostic_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": list(
            dict.fromkeys(
                [
                    *[str(value) for value in list(row.get("strict_blockers") or []) if str(value)],
                    "table_cell_extractor_pilot_only",
                    "cell_bbox_text_pairing_not_verified",
                    "cell_source_spans_not_created",
                    "table_cell_source_hash_linkage_not_created",
                    "runtime_promotion_disabled_for_tranche",
                    "strict_promotion_requires_explicit_later_tranche",
                    status,
                ]
            )
        ),
        "non_strict_reason": [
            "table_probe_output_is_diagnostic_only",
            "cell_bboxes_are_not_linked_to_original_source_spans",
            "diagnostic_cell_text_matches_are_not_strict_source_spans",
            "no_runtime_or_strict_evidence_created",
        ],
    }


def _counts(rows: list[dict[str, Any]], violations: list[str]) -> dict[str, Any]:
    blocker_counts: Counter[str] = Counter()
    for item in rows:
        blocker_counts.update(str(blocker) for blocker in list(item.get("strict_blockers") or []))
    return {
        "inputAuthorityDesignRows": len(rows),
        "pilotRows": len(rows),
        "probeAttemptedRows": sum(1 for item in rows if item.get("probe_attempted")),
        "heldOutRows": sum(1 for item in rows if str(item.get("probe_status") or "").startswith("held_out")),
        "blockedRows": sum(1 for item in rows if str(item.get("probe_status") or "").startswith("blocked")),
        "noTableDetectedRows": sum(1 for item in rows if item.get("probe_status") == "no_tables_detected"),
        "tableDetectedRows": sum(1 for item in rows if _safe_int(item.get("detected_table_count")) > 0),
        "cellBboxCandidateRows": sum(1 for item in rows if item.get("table_cell_bbox_candidates_found")),
        "selectedTableCellBboxCandidates": sum(_safe_int(item.get("selected_table_cell_bbox_count")) for item in rows),
        "selectedTableCellTextCandidates": sum(_safe_int(item.get("selected_table_cell_text_count")) for item in rows),
        "diagnosticUniqueCellTextMatches": sum(
            _safe_int(item.get("diagnostic_unique_cell_text_matches")) for item in rows
        ),
        "diagnosticAmbiguousCellTextMatches": sum(
            _safe_int(item.get("diagnostic_ambiguous_cell_text_matches")) for item in rows
        ),
        "cellBboxTextPairingVerifiedRows": 0,
        "cellSourceSpanCreatedRows": 0,
        "cellSourceHashLinkedRows": 0,
        "tableCellEvidenceCreatedRows": 0,
        "tableCellCitationGradeRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "schemaViolationCount": len([item for item in violations if item.endswith("_mismatch")]),
        "unsafeUpstreamFlagCount": len([item for item in violations if not item.endswith("_mismatch")]),
        "byPaper": dict(Counter(str(item.get("paper_id") or "") for item in rows)),
        "byProbeStatus": dict(Counter(str(item.get("probe_status") or "") for item in rows)),
        "strictBlockerSummary": dict(blocker_counts),
    }


def build_table_cell_bbox_source_span_extractor_pilot(
    *,
    table_cell_authority_design_report: str | Path,
    table_region_pdf_offset_feasibility_report: str | Path,
    max_ready_rows: int = 10,
    table_probe_loader: TableProbeLoader | None = None,
    page_text_loader: PageTextLoader | None = None,
) -> dict[str, Any]:
    """Build a bounded, report-only TableCell extractor pilot."""

    design_path = Path(str(table_cell_authority_design_report)).expanduser()
    source_path = Path(str(table_region_pdf_offset_feasibility_report)).expanduser()
    design = _read_json(design_path)
    source_report = _read_json(source_path)
    violations = [*_schema_violations(design, source_report), *_unsafe_flags(design)]
    source_by_id = _source_rows_by_candidate_id(source_report)
    ready_seen = 0
    rows: list[dict[str, Any]] = []
    probe_loader = table_probe_loader or _default_table_probe
    text_loader = page_text_loader or _extract_pdf_pages
    for index, row in enumerate([dict(item) for item in list(design.get("designRows") or []) if isinstance(item, dict)], start=1):
        is_ready = row.get("authority_design_status") == "ready_for_cell_bbox_source_span_authority_design"
        should_probe = False
        if not violations and is_ready and ready_seen < max_ready_rows:
            ready_seen += 1
            should_probe = True
        rows.append(
            _pilot_row(
                index,
                row,
                source_by_id.get(str(row.get("source_table_region_candidate_id") or "")),
                table_probe_loader=probe_loader,
                page_text_loader=text_loader,
                should_probe=should_probe,
            )
        )
    counts = _counts(rows, violations)
    ready = not violations and bool(rows)
    return {
        "schema": TABLE_CELL_BBOX_SOURCE_SPAN_EXTRACTOR_PILOT_SCHEMA_ID,
        "status": "pilot_complete" if ready else "blocked",
        "generatedAt": _now(),
        "inputs": {
            "tableCellAuthorityDesignReport": str(design_path),
            "tableRegionPdfOffsetFeasibilityReport": str(source_path),
            "tableCellAuthorityDesignSchema": str(design.get("schema") or ""),
            "tableRegionPdfOffsetFeasibilitySchema": str(source_report.get("schema") or ""),
            "maxReadyRows": max_ready_rows,
        },
        "counts": counts,
        "gate": {
            "pilotComplete": ready,
            "cellBboxCandidatesObserved": counts["cellBboxCandidateRows"] > 0,
            "cellBboxTextPairingVerified": False,
            "cellSourceSpansCreated": False,
            "cellSourceHashLinked": False,
            "tableCellEvidenceReady": False,
            "tableCellCitationGradeReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "table_cell_extractor_pilot_complete_non_strict" if ready else "blocked",
            "schemaViolations": [item for item in violations if item.endswith("_mismatch")],
            "unsafeUpstreamFlags": [item for item in violations if not item.endswith("_mismatch")],
            "recommendedNextTranche": "review_table_cell_probe_results_before_any_extractor_choice",
        },
        "policy": {
            "reportOnly": True,
            "pilotOnly": True,
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
            "pymupdf_table_probe_output_is_diagnostic_only",
            "cell_bboxes_without_verified_text_pairing_and_source_spans_are_not_citation_grade",
            "diagnostic_cell_text_matches_do_not_create_source_spans",
            "no_strict_evidence_or_parser_routing_is_created",
        ],
        "pilotRows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings", "pilotRows")
        if key in report
    }


def render_table_cell_bbox_source_span_extractor_pilot_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# TableCell BBox / Source-Span Extractor Pilot",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Pilot rows: `{int(counts.get('pilotRows') or 0)}`",
        f"- Probe attempted rows: `{int(counts.get('probeAttemptedRows') or 0)}`",
        f"- Table detected rows: `{int(counts.get('tableDetectedRows') or 0)}`",
        f"- Cell bbox candidate rows: `{int(counts.get('cellBboxCandidateRows') or 0)}`",
        f"- Selected table cell bbox candidates: `{int(counts.get('selectedTableCellBboxCandidates') or 0)}`",
        f"- Diagnostic unique cell text matches: `{int(counts.get('diagnosticUniqueCellTextMatches') or 0)}`",
        f"- Cell source span created rows: `{int(counts.get('cellSourceSpanCreatedRows') or 0)}`",
        f"- Table-cell citation-grade rows: `{int(counts.get('tableCellCitationGradeRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This is a bounded diagnostic pilot. PyMuPDF table bboxes and diagnostic cell text matches are not source-aligned table-cell evidence until cell/text pairing, original PDF source spans, and cell-level source hash linkage are verified.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By probe status: `{json.dumps(counts.get('byProbeStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_table_cell_bbox_source_span_extractor_pilot_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    pilot_path = root / "table-cell-bbox-source-span-extractor-pilot.json"
    summary_path = root / "table-cell-bbox-source-span-extractor-pilot-summary.json"
    markdown_path = root / "table-cell-bbox-source-span-extractor-pilot.md"
    pilot_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_table_cell_bbox_source_span_extractor_pilot_markdown(report), encoding="utf-8")
    return {"pilot": str(pilot_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate report-only TableCell bbox/source-span extractor pilot.")
    parser.add_argument("--table-cell-authority-design-report", required=True)
    parser.add_argument("--table-region-pdf-offset-feasibility-report", required=True)
    parser.add_argument("--max-ready-rows", type=int, default=10)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_table_cell_bbox_source_span_extractor_pilot(
        table_cell_authority_design_report=args.table_cell_authority_design_report,
        table_region_pdf_offset_feasibility_report=args.table_region_pdf_offset_feasibility_report,
        max_ready_rows=max(0, args.max_ready_rows),
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_table_cell_bbox_source_span_extractor_pilot_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TABLE_CELL_BBOX_SOURCE_SPAN_EXTRACTOR_PILOT_SCHEMA_ID",
    "build_table_cell_bbox_source_span_extractor_pilot",
    "render_table_cell_bbox_source_span_extractor_pilot_markdown",
    "write_table_cell_bbox_source_span_extractor_pilot_reports",
]
