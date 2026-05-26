"""Report-only PyMuPDF table-cell bbox/text pairing diagnostics.

This helper reruns the bounded PyMuPDF table probe for overlay-review rows and
records row/column text plus bbox candidates.  The output is diagnostic only:
cell/text pairing is not verified, no cell source spans are created, no
table-cell evidence is produced, parser routing is unchanged, and no DB/index
or canonical parsed-artifact writes occur.
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


TABLE_CELL_PYMUPDF_PAIRING_DIAGNOSTIC_SCHEMA_ID = (
    "knowledge-hub.paper.table-cell-pymupdf-pairing-diagnostic.v1"
)
TABLE_CELL_PYMUPDF_OVERLAY_REVIEW_PACK_SCHEMA_ID = (
    "knowledge-hub.paper.table-cell-pymupdf-overlay-review-pack.v1"
)
TABLE_CELL_BBOX_SOURCE_SPAN_EXTRACTOR_PILOT_SCHEMA_ID = (
    "knowledge-hub.paper.table-cell-bbox-source-span-extractor-pilot.v1"
)


PairingProbeLoader = Callable[[str | Path, int], dict[str, Any]]
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


def _schema_violations(overlay_report: dict[str, Any], pilot_report: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if overlay_report.get("schema") != TABLE_CELL_PYMUPDF_OVERLAY_REVIEW_PACK_SCHEMA_ID:
        violations.append("table_cell_pymupdf_overlay_review_pack_schema_mismatch")
    if overlay_report.get("status") != "review_pack_ready":
        violations.append("table_cell_pymupdf_overlay_review_pack_not_ready")
    if pilot_report.get("schema") != TABLE_CELL_BBOX_SOURCE_SPAN_EXTRACTOR_PILOT_SCHEMA_ID:
        violations.append("table_cell_bbox_source_span_extractor_pilot_schema_mismatch")
    if pilot_report.get("status") != "pilot_complete":
        violations.append("table_cell_bbox_source_span_extractor_pilot_not_complete")
    return violations


def _unsafe_flags(report: dict[str, Any]) -> list[str]:
    unsafe: list[str] = []
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    policy = dict(report.get("policy") or {})
    for key in (
        "cellBboxTextPairingVerifiedRows",
        "visualPairingReviewCompletedRows",
        "cellSourceSpanCreatedRows",
        "cellSourceHashLinkedRows",
        "tableCellEvidenceCreatedRows",
        "tableCellCitationGradeRows",
        "strictEligibleRows",
        "citationGradeRows",
        "runtimeEvidenceRows",
    ):
        if _safe_int(counts.get(key)) > 0:
            unsafe.append(f"{key}_nonzero")
    for key in (
        "visualPairingReviewCompleted",
        "cellBboxTextPairingVerified",
        "cellSourceSpansCreated",
        "cellSourceHashLinked",
        "extractorChoiceMade",
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
        "extractorChoiceMade",
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


def _pilot_rows_by_candidate_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for item in list(report.get("pilotRows") or []):
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("source_table_region_candidate_id") or "")
        if candidate_id:
            rows[candidate_id] = dict(item)
    return rows


def _default_pairing_probe(source_pdf: str | Path, page_number: int) -> dict[str, Any]:
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
            cells = [_bbox(item) for item in list(getattr(table, "cells", []) or [])]
            extracted: list[list[str]] = []
            try:
                raw = table.extract()
                extracted = [[str(cell or "") for cell in list(row or [])] for row in list(raw or [])]
            except Exception:
                extracted = []
            tables.append(
                {
                    "table_index": index,
                    "bbox": _bbox(getattr(table, "bbox", None)),
                    "row_count": _safe_int(getattr(table, "row_count", 0)),
                    "column_count": _safe_int(getattr(table, "col_count", 0)),
                    "cell_bboxes": [item for item in cells if item is not None],
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


def _selected_table(tables: list[dict[str, Any]], selected_index: int) -> dict[str, Any] | None:
    for table in tables:
        if _safe_int(table.get("table_index")) == selected_index and selected_index > 0:
            return table
    if not tables:
        return None
    return max(
        tables,
        key=lambda item: (
            len(list(item.get("cell_bboxes") or [])),
            _safe_int(item.get("row_count")),
            _safe_int(item.get("column_count")),
        ),
    )


def _cell_text_at(rows: list[list[str]], row_index: int, column_index: int) -> str:
    if row_index < len(rows) and column_index < len(rows[row_index]):
        return _clean_text(rows[row_index][column_index])
    return ""


def _diagnostic_match(page_text: dict[str, Any], text: str) -> dict[str, Any]:
    if not text or not page_text:
        return {
            "status": "no_match",
            "method": "",
            "page": None,
            "chars_start": None,
            "chars_end": None,
            "strict_source_span": False,
        }
    matches = _exact_matches([page_text], text)
    method = "exact"
    if not matches:
        matches = _normalized_matches([page_text], text)
        method = "normalized_whitespace_case"
    if len(matches) == 1:
        match = dict(matches[0])
        return {
            "status": "unique_match_non_strict",
            "method": method,
            "page": match.get("page"),
            "chars_start": match.get("chars_start"),
            "chars_end": match.get("chars_end"),
            "strict_source_span": False,
        }
    if len(matches) > 1:
        return {
            "status": "ambiguous_match_non_strict",
            "method": method,
            "page": None,
            "chars_start": None,
            "chars_end": None,
            "strict_source_span": False,
        }
    return {
        "status": "no_match",
        "method": method,
        "page": None,
        "chars_start": None,
        "chars_end": None,
        "strict_source_span": False,
    }


def _cell_candidates(selected: dict[str, Any] | None, page_text: dict[str, Any]) -> list[dict[str, Any]]:
    if not selected:
        return []
    row_count = _safe_int(selected.get("row_count"))
    column_count = _safe_int(selected.get("column_count"))
    rows = [[str(cell or "") for cell in list(row or [])] for row in list(selected.get("extracted_rows") or [])]
    bboxes = list(selected.get("cell_bboxes") or [])
    inferred_rows = row_count or len(rows)
    inferred_columns = column_count or max([len(row) for row in rows] or [0])
    cells: list[dict[str, Any]] = []
    for row_index in range(inferred_rows):
        for column_index in range(inferred_columns):
            flat_index = row_index * max(inferred_columns, 1) + column_index
            bbox = bboxes[flat_index] if flat_index < len(bboxes) else None
            text = _cell_text_at(rows, row_index, column_index)
            match = _diagnostic_match(page_text, text)
            cells.append(
                {
                    "cell_candidate_id": f"cell:{row_index + 1}:{column_index + 1}",
                    "row_index": row_index + 1,
                    "column_index": column_index + 1,
                    "bbox": bbox if _bbox(bbox) is not None else None,
                    "text": text,
                    "diagnostic_match_status": match["status"],
                    "diagnostic_match_method": match["method"],
                    "diagnostic_page": match["page"],
                    "diagnostic_chars_start": match["chars_start"],
                    "diagnostic_chars_end": match["chars_end"],
                    "diagnostic_match_is_strict": False,
                    "pairing_status": (
                        "bbox_text_pairing_candidate_non_strict"
                        if bbox and text
                        else "bbox_without_text" if bbox else "text_without_bbox" if text else "empty_cell_candidate"
                    ),
                    "strict_eligible": False,
                    "strict_blockers": [
                        "cell_bbox_text_pairing_not_verified",
                        "diagnostic_match_is_not_cell_source_span",
                        "cell_source_hash_linkage_not_created",
                    ],
                }
            )
    return cells


def _row_status(*, selected: dict[str, Any] | None, probe: dict[str, Any], cells: list[dict[str, Any]]) -> str:
    if probe.get("status") != "ok":
        return str(probe.get("status") or "failed_pairing_probe")
    if not selected:
        return "blocked_selected_table_missing"
    if not cells:
        return "blocked_no_cell_candidates"
    return "pairing_diagnostic_ready_non_strict"


def _diagnostic_row(
    index: int,
    overlay_card: dict[str, Any],
    pilot_row: dict[str, Any] | None,
    *,
    pairing_probe_loader: PairingProbeLoader,
    page_text_loader: PageTextLoader,
) -> dict[str, Any]:
    source_pdf = str((pilot_row or {}).get("source_pdf_path") or "")
    page_number = _safe_int(overlay_card.get("page") or (pilot_row or {}).get("page"))
    selected_index = _safe_int((pilot_row or {}).get("selected_table_index"))
    probe = (
        pairing_probe_loader(source_pdf, page_number)
        if source_pdf and page_number > 0
        else {"status": "blocked_source_pdf_or_page_missing", "failureReason": "", "tables": []}
    )
    tables = [dict(item) for item in list(probe.get("tables") or []) if isinstance(item, dict)]
    selected = _selected_table(tables, selected_index)
    page_text = (
        _page_text_context(source_pdf=source_pdf, page_number=page_number, page_text_loader=page_text_loader)
        if source_pdf and page_number > 0 and selected
        else {}
    )
    cells = _cell_candidates(selected, page_text)
    status = _row_status(selected=selected, probe=probe, cells=cells)
    match_counts = Counter(str(cell.get("diagnostic_match_status") or "") for cell in cells)
    pairing_counts = Counter(str(cell.get("pairing_status") or "") for cell in cells)
    strict_blockers = list(
        dict.fromkeys(
            [
                *[str(value) for value in list(overlay_card.get("strict_blockers") or []) if str(value)],
                "table_cell_pymupdf_pairing_diagnostic_only",
                "cell_bbox_text_pairing_not_verified",
                "diagnostic_matches_are_not_cell_source_spans",
                "cell_source_spans_not_created",
                "table_cell_source_hash_linkage_not_created",
                "table_cell_citation_grade_evidence_not_created",
                "runtime_promotion_disabled_for_tranche",
                "strict_promotion_requires_explicit_later_tranche",
                status,
            ]
        )
    )
    return {
        "diagnostic_row_id": f"table-cell-pymupdf-pairing-diagnostic:{index:04d}",
        "source_overlay_review_card_id": str(overlay_card.get("overlay_review_card_id") or ""),
        "source_pilot_row_id": str((pilot_row or {}).get("pilot_row_id") or ""),
        "source_table_region_candidate_id": str(overlay_card.get("source_table_region_candidate_id") or ""),
        "paper_id": str(overlay_card.get("paper_id") or ""),
        "candidate_type": "table_cell_pymupdf_pairing_diagnostic_row",
        "source_parser": "pymupdf_table_probe",
        "table_label": _clean_text(overlay_card.get("table_label")),
        "candidate_text": _clean_text(overlay_card.get("candidate_text")),
        "source_pdf_path": source_pdf,
        "page": page_number or None,
        "sourceContentHash": str(overlay_card.get("sourceContentHash") or ""),
        "diagnostic_status": status,
        "probe_failure_reason": str(probe.get("failureReason") or ""),
        "selected_table_index": _safe_int((selected or {}).get("table_index")),
        "selected_table_bbox": (selected or {}).get("bbox"),
        "selected_table_row_count": _safe_int((selected or {}).get("row_count")),
        "selected_table_column_count": _safe_int((selected or {}).get("column_count")),
        "cell_bbox_candidate_count": len([cell for cell in cells if cell.get("bbox")]),
        "cell_text_candidate_count": len([cell for cell in cells if cell.get("text")]),
        "cell_pairing_candidate_count": len(
            [cell for cell in cells if cell.get("pairing_status") == "bbox_text_pairing_candidate_non_strict"]
        ),
        "diagnostic_unique_cell_text_matches": int(match_counts.get("unique_match_non_strict", 0)),
        "diagnostic_ambiguous_cell_text_matches": int(match_counts.get("ambiguous_match_non_strict", 0)),
        "diagnostic_no_match_cell_texts": int(match_counts.get("no_match", 0)),
        "diagnostic_pairing_status_counts": dict(pairing_counts),
        "cell_candidates": cells,
        "cell_bbox_text_pairing_verified": False,
        "cell_source_spans_created": 0,
        "cell_source_hash_linkages_created": 0,
        "table_cell_evidence_created": False,
        "table_cell_citation_grade": False,
        "extractor_choice_made": False,
        "evidence_tier": "table_cell_pymupdf_pairing_diagnostic_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": strict_blockers,
        "non_strict_reason": [
            "pymupdf_pairing_diagnostic_only",
            "cell_bbox_text_pairing_not_verified",
            "diagnostic_matches_are_not_cell_source_spans",
            "cell_source_spans_not_created",
            "no_runtime_or_strict_evidence_created",
        ],
    }


def _counts(rows: list[dict[str, Any]], violations: list[str]) -> dict[str, Any]:
    blocker_counts: Counter[str] = Counter()
    for item in rows:
        blocker_counts.update(str(blocker) for blocker in list(item.get("strict_blockers") or []))
    return {
        "diagnosticRows": len(rows),
        "pairingDiagnosticReadyRows": sum(
            1 for item in rows if item.get("diagnostic_status") == "pairing_diagnostic_ready_non_strict"
        ),
        "blockedRows": sum(1 for item in rows if str(item.get("diagnostic_status") or "").startswith("blocked")),
        "cellBboxCandidateCount": sum(_safe_int(item.get("cell_bbox_candidate_count")) for item in rows),
        "cellTextCandidateCount": sum(_safe_int(item.get("cell_text_candidate_count")) for item in rows),
        "cellPairingCandidateCount": sum(_safe_int(item.get("cell_pairing_candidate_count")) for item in rows),
        "diagnosticUniqueCellTextMatches": sum(
            _safe_int(item.get("diagnostic_unique_cell_text_matches")) for item in rows
        ),
        "diagnosticAmbiguousCellTextMatches": sum(
            _safe_int(item.get("diagnostic_ambiguous_cell_text_matches")) for item in rows
        ),
        "diagnosticNoMatchCellTexts": sum(_safe_int(item.get("diagnostic_no_match_cell_texts")) for item in rows),
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
        "byDiagnosticStatus": dict(Counter(str(item.get("diagnostic_status") or "") for item in rows)),
        "strictBlockerSummary": dict(blocker_counts),
    }


def build_table_cell_pymupdf_pairing_diagnostic(
    *,
    table_cell_pymupdf_overlay_review_pack: str | Path,
    table_cell_extractor_pilot_report: str | Path,
    pairing_probe_loader: PairingProbeLoader | None = None,
    page_text_loader: PageTextLoader | None = None,
) -> dict[str, Any]:
    """Build report-only PyMuPDF table-cell bbox/text pairing diagnostics."""

    overlay_path = Path(str(table_cell_pymupdf_overlay_review_pack)).expanduser()
    pilot_path = Path(str(table_cell_extractor_pilot_report)).expanduser()
    overlay_report = _read_json(overlay_path)
    pilot_report = _read_json(pilot_path)
    violations = [
        *_schema_violations(overlay_report, pilot_report),
        *_unsafe_flags(overlay_report),
        *_unsafe_flags(pilot_report),
    ]
    pilot_by_id = _pilot_rows_by_candidate_id(pilot_report)
    probe_loader = pairing_probe_loader or _default_pairing_probe
    text_loader = page_text_loader or _extract_pdf_pages
    rows = [
        _diagnostic_row(
            index + 1,
            dict(card),
            pilot_by_id.get(str(card.get("source_table_region_candidate_id") or "")),
            pairing_probe_loader=probe_loader,
            page_text_loader=text_loader,
        )
        for index, card in enumerate(
            [dict(item) for item in list(overlay_report.get("overlayReviewCards") or []) if isinstance(item, dict)]
        )
        if card.get("overlay_review_status") in {
            "manual_overlay_pairing_review_required",
            "overlay_candidate_ready_for_visual_pairing_review",
        }
    ]
    counts = _counts(rows, violations)
    blocked = bool(violations) or not rows
    return {
        "schema": TABLE_CELL_PYMUPDF_PAIRING_DIAGNOSTIC_SCHEMA_ID,
        "status": "blocked" if blocked else "diagnostic_ready",
        "generatedAt": _now(),
        "inputs": {
            "tableCellPymupdfOverlayReviewPack": str(overlay_path),
            "tableCellExtractorPilotReport": str(pilot_path),
            "tableCellPymupdfOverlayReviewPackSchema": str(overlay_report.get("schema") or ""),
            "tableCellExtractorPilotSchema": str(pilot_report.get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "pairingDiagnosticReady": not blocked,
            "cellBboxTextPairingVerified": False,
            "cellSourceSpansCreated": False,
            "cellSourceHashLinked": False,
            "extractorChoiceMade": False,
            "tableCellEvidenceReady": False,
            "tableCellCitationGradeReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "ready_for_table_cell_pairing_human_review" if not blocked else "blocked",
            "schemaViolations": [item for item in violations if item.endswith("_mismatch")],
            "unsafeUpstreamFlags": [item for item in violations if not item.endswith("_mismatch")],
            "recommendedNextTranche": "human_review_pairing_diagnostic_or_isolated_alternative_extractor_pilot",
        },
        "policy": {
            "reportOnly": True,
            "diagnosticOnly": True,
            "extractorChoiceMade": False,
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
            "pairing_diagnostics_are_not_runtime_evidence",
            "row_column_bbox_text_candidates_require_human_or_visual_verification",
            "diagnostic_text_matches_are_not_cell_source_spans",
            "no_strict_evidence_or_parser_routing_is_created",
        ],
        "diagnosticRows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in (
            "schema",
            "status",
            "generatedAt",
            "inputs",
            "counts",
            "gate",
            "policy",
            "warnings",
            "diagnosticRows",
        )
        if key in report
    }


def render_table_cell_pymupdf_pairing_diagnostic_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# TableCell PyMuPDF Pairing Diagnostic",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Diagnostic rows: `{int(counts.get('diagnosticRows') or 0)}`",
        f"- Pairing diagnostic ready rows: `{int(counts.get('pairingDiagnosticReadyRows') or 0)}`",
        f"- Cell bbox candidates: `{int(counts.get('cellBboxCandidateCount') or 0)}`",
        f"- Cell text candidates: `{int(counts.get('cellTextCandidateCount') or 0)}`",
        f"- Cell pairing candidates: `{int(counts.get('cellPairingCandidateCount') or 0)}`",
        f"- Diagnostic unique text matches: `{int(counts.get('diagnosticUniqueCellTextMatches') or 0)}`",
        f"- Diagnostic ambiguous text matches: `{int(counts.get('diagnosticAmbiguousCellTextMatches') or 0)}`",
        f"- Diagnostic no-match texts: `{int(counts.get('diagnosticNoMatchCellTexts') or 0)}`",
        f"- Cell source span created rows: `{int(counts.get('cellSourceSpanCreatedRows') or 0)}`",
        f"- Table-cell citation-grade rows: `{int(counts.get('tableCellCitationGradeRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This diagnostic captures PyMuPDF row/column/bbox/text candidates only. It does not verify cell/text pairing, create source spans, create cell-level source hash links, or authorize table-cell citation-grade evidence.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By diagnostic status: `{json.dumps(counts.get('byDiagnosticStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Diagnostic Rows",
        "",
    ]
    for item in list(report.get("diagnosticRows") or []):
        lines.append(
            f"- `{item.get('paper_id')}` `{item.get('table_label')}` page `{item.get('page')}` "
            f"pairs `{item.get('cell_pairing_candidate_count')}` "
            f"unique `{item.get('diagnostic_unique_cell_text_matches')}` "
            f"ambiguous `{item.get('diagnostic_ambiguous_cell_text_matches')}` "
            f"missing `{item.get('diagnostic_no_match_cell_texts')}` "
            f"-> `{item.get('diagnostic_status')}`"
        )
    lines.append("")
    return "\n".join(lines)


def write_table_cell_pymupdf_pairing_diagnostic_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    diagnostic_path = root / "table-cell-pymupdf-pairing-diagnostic.json"
    summary_path = root / "table-cell-pymupdf-pairing-diagnostic-summary.json"
    markdown_path = root / "table-cell-pymupdf-pairing-diagnostic.md"
    diagnostic_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_table_cell_pymupdf_pairing_diagnostic_markdown(report), encoding="utf-8")
    return {"diagnostic": str(diagnostic_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate report-only PyMuPDF TableCell pairing diagnostics.")
    parser.add_argument("--table-cell-pymupdf-overlay-review-pack", required=True)
    parser.add_argument("--table-cell-extractor-pilot-report", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_table_cell_pymupdf_pairing_diagnostic(
        table_cell_pymupdf_overlay_review_pack=args.table_cell_pymupdf_overlay_review_pack,
        table_cell_extractor_pilot_report=args.table_cell_extractor_pilot_report,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_table_cell_pymupdf_pairing_diagnostic_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TABLE_CELL_PYMUPDF_PAIRING_DIAGNOSTIC_SCHEMA_ID",
    "build_table_cell_pymupdf_pairing_diagnostic",
    "render_table_cell_pymupdf_pairing_diagnostic_markdown",
    "write_table_cell_pymupdf_pairing_diagnostic_reports",
]
