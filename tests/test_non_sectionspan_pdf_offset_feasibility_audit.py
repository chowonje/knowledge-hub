from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.non_sectionspan_pdf_offset_feasibility_audit import (
    NON_SECTIONSPAN_PDF_OFFSET_FEASIBILITY_AUDIT_SCHEMA_ID,
    build_non_sectionspan_pdf_offset_feasibility_audit,
    write_non_sectionspan_pdf_offset_feasibility_audit_reports,
)


def _write_report(tmp_path: Path, name: str, schema: str, rows: list[dict], *, status: str = "feasibility_complete") -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                "schema": schema,
                "status": status,
                "counts": {
                    "feasibilityRows": len(rows),
                    "originalPdfOffsetRecoveredRows": sum(
                        1 for row in rows if row.get("original_pdf_offset_recovered")
                    ),
                    "blockedRows": sum(1 for row in rows if not row.get("original_pdf_offset_recovered")),
                    "schemaViolationCount": 0,
                    "strictEligibleRows": 0,
                    "citationGradeRows": 0,
                    "runtimeEvidenceRows": 0,
                    "byPaper": {"paper-1": len(rows)},
                    "byFeasibilityStatus": {
                        str(row.get("feasibility_status") or ""): 1 for row in rows
                    },
                    "strictBlockerSummary": {"report_only": len(rows)},
                },
                "feasibilityRows": rows,
            }
        ),
        encoding="utf-8",
    )
    return path


def _base_row(layer: str, *, recovered: bool, paper_id: str = "paper-1") -> dict:
    source_keys = {
        "figure_caption": "source_figure_caption_candidate_id",
        "table_region": "source_table_region_candidate_id",
        "equation_quote": "source_equation_quote_candidate_id",
    }
    span = {
        "originalPdfCharsStart": 10 if recovered else None,
        "originalPdfCharsEnd": 24 if recovered else None,
        "page": 2 if recovered else None,
        "sourceContentHash": "hash-1",
        "matchMethod": "exact" if recovered else "",
        "matchConfidence": 1.0 if recovered else 0.0,
    }
    row = {
        "feasibility_row_id": f"{layer}:row-1",
        source_keys[layer]: f"{layer}:candidate-1",
        "paper_id": paper_id,
        "candidate_text": "Candidate text",
        "feasibility_status": (
            "recovered_exact"
            if recovered
            else "diagnostic_page_context_candidate_only"
            if layer == "equation_quote"
            else "blocked_no_match"
        ),
        "original_pdf_offset_recovered": recovered,
        "original_pdf_span": span,
        "sourceContentHash": "hash-1",
        "layout_region_candidate_present": True,
        "page_agrees_with_canonical": recovered,
        "source_hash_agrees_with_canonical": recovered,
        "evidence_tier": f"{layer}_pdf_offset_feasibility_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": ["report_only"],
        "non_strict_reason": [f"{layer}_pdf_offset_feasibility_only"],
    }
    if layer == "figure_caption":
        row["figure_region_link_verified"] = False
    if layer == "table_region":
        row["table_region_link_verified"] = False
        row["table_cell_evidence_available"] = False
        row["table_cell_citation_grade"] = False
    if layer == "equation_quote":
        row["equation_region_link_verified"] = False
        row["equation_semantics_interpreted"] = False
        row["diagnostic_page_candidates"] = [{"page": 2, "coverage": 0.8}] if not recovered else []
        row["diagnostic_best_page_coverage"] = 0.8 if not recovered else None
    return row


def _reports(tmp_path: Path) -> tuple[Path, Path, Path]:
    figure = _write_report(
        tmp_path,
        "figure",
        "knowledge-hub.paper.figure-caption-pdf-offset-feasibility.v1",
        [_base_row("figure_caption", recovered=True)],
    )
    table = _write_report(
        tmp_path,
        "table",
        "knowledge-hub.paper.table-region-pdf-offset-feasibility.v1",
        [_base_row("table_region", recovered=True)],
    )
    equation = _write_report(
        tmp_path,
        "equation",
        "knowledge-hub.paper.equation-quote-pdf-offset-feasibility.v1",
        [_base_row("equation_quote", recovered=False)],
    )
    return figure, table, equation


def test_non_sectionspan_pdf_offset_audit_consolidates_rows_as_non_strict(tmp_path: Path) -> None:
    figure, table, equation = _reports(tmp_path)

    payload = build_non_sectionspan_pdf_offset_feasibility_audit(
        figure_caption_pdf_offset_feasibility_report=figure,
        table_region_pdf_offset_feasibility_report=table,
        equation_quote_pdf_offset_feasibility_report=equation,
    )

    assert payload["schema"] == NON_SECTIONSPAN_PDF_OFFSET_FEASIBILITY_AUDIT_SCHEMA_ID
    assert validate_payload(payload, NON_SECTIONSPAN_PDF_OFFSET_FEASIBILITY_AUDIT_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "ok"
    assert payload["counts"]["totalRows"] == 3
    assert payload["counts"]["recoveredRows"] == 2
    assert payload["counts"]["blockedRows"] == 1
    assert payload["counts"]["readyForRegionReviewRows"] == 2
    assert payload["counts"]["needsFigureRegionReviewRows"] == 1
    assert payload["counts"]["needsTableCellProvenanceReviewRows"] == 1
    assert payload["counts"]["needsEquationAlignmentReviewRows"] == 1
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["counts"]["citationGradeRows"] == 0
    assert payload["counts"]["runtimeEvidenceRows"] == 0
    assert payload["policy"]["strictEvidenceCreated"] is False
    assert payload["policy"]["parserRoutingChanged"] is False
    assert payload["policy"]["databaseMutation"] is False
    assert payload["policy"]["canonicalParsedArtifactsWritten"] is False
    assert all(row["strict_eligible"] is False for row in payload["auditRows"])
    assert all(row["evidence_tier"] == "non_sectionspan_pdf_offset_feasibility_audit_only" for row in payload["auditRows"])


def test_recovered_offsets_do_not_become_runtime_or_strict_evidence(tmp_path: Path) -> None:
    figure, table, equation = _reports(tmp_path)

    payload = build_non_sectionspan_pdf_offset_feasibility_audit(
        figure_caption_pdf_offset_feasibility_report=figure,
        table_region_pdf_offset_feasibility_report=table,
        equation_quote_pdf_offset_feasibility_report=equation,
    )

    recovered_rows = [row for row in payload["auditRows"] if row["original_pdf_offset_recovered"]]
    assert recovered_rows
    for row in recovered_rows:
        assert row["page"] == 2
        assert row["sourceContentHash"] == "hash-1"
        assert row["runtime_promotion_allowed"] is False
        assert row["strict_eligible"] is False
        assert row["citation_grade"] is False
        assert row["runtime_evidence"] is False
        assert "runtime_promotion_disabled_for_tranche" in row["strict_blockers"]


def test_equation_diagnostic_page_context_remains_blocked_non_strict(tmp_path: Path) -> None:
    figure, table, equation = _reports(tmp_path)

    payload = build_non_sectionspan_pdf_offset_feasibility_audit(
        figure_caption_pdf_offset_feasibility_report=figure,
        table_region_pdf_offset_feasibility_report=table,
        equation_quote_pdf_offset_feasibility_report=equation,
    )

    equation_rows = [row for row in payload["auditRows"] if row["candidate_layer"] == "equation_quote"]
    assert len(equation_rows) == 1
    row = equation_rows[0]
    assert row["readiness"] == "diagnostic_page_context_only_non_strict"
    assert row["diagnostic_page_context_available"] is True
    assert row["original_pdf_offset_recovered"] is False
    assert row["recommended_next_action"] == "equation_quote_normalization_or_layout_review"
    assert row["strict_eligible"] is False


def test_schema_mismatch_blocks_audit_without_green_status(tmp_path: Path) -> None:
    figure, table, equation = _reports(tmp_path)
    figure.write_text(
        json.dumps(
            {
                "schema": "wrong.schema",
                "status": "feasibility_complete",
                "counts": {"schemaViolationCount": 0},
                "feasibilityRows": [],
            }
        ),
        encoding="utf-8",
    )

    payload = build_non_sectionspan_pdf_offset_feasibility_audit(
        figure_caption_pdf_offset_feasibility_report=figure,
        table_region_pdf_offset_feasibility_report=table,
        equation_quote_pdf_offset_feasibility_report=equation,
    )

    assert payload["status"] == "blocked_upstream_schema"
    assert payload["gate"]["auditComplete"] is False
    assert "figure_caption_schema_mismatch" in payload["gate"]["schemaViolations"]
    assert payload["counts"]["strictEligibleRows"] == 0


def test_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    figure, table, equation = _reports(tmp_path)
    payload = build_non_sectionspan_pdf_offset_feasibility_audit(
        figure_caption_pdf_offset_feasibility_report=figure,
        table_region_pdf_offset_feasibility_report=table,
        equation_quote_pdf_offset_feasibility_report=equation,
    )

    paths = write_non_sectionspan_pdf_offset_feasibility_audit_reports(payload, tmp_path / "reports")

    assert set(paths) == {"report", "summary", "markdown"}
    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(report, NON_SECTIONSPAN_PDF_OFFSET_FEASIBILITY_AUDIT_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["totalRows"] == 3
    assert "Non-SectionSpan PDF Offset Feasibility Audit" in markdown
