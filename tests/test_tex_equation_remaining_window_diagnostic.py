from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.tex_equation_pdf_region_anchor_audit import (
    TEX_EQUATION_PDF_REGION_ANCHOR_AUDIT_SCHEMA_ID,
)
from knowledge_hub.papers.tex_equation_remaining_window_diagnostic import (
    TEX_EQUATION_REMAINING_WINDOW_DIAGNOSTIC_SCHEMA_ID,
    build_tex_equation_remaining_window_diagnostic,
    write_tex_equation_remaining_window_diagnostic_reports,
)


def _write_json(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _pdf_region_row(
    row_id: str,
    text: str,
    *,
    paper_id: str = "paper-1",
    status: str = "blocked_no_line_local_normalized_window",
    environment: str = "align",
    terms: list[str] | None = None,
) -> dict:
    return {
        "pdf_region_anchor_id": row_id,
        "source_line_local_anchor_id": row_id.replace("pdf", "line"),
        "source_design_id": row_id.replace("pdf", "design"),
        "source_candidate_id": row_id.replace("pdf", "candidate"),
        "paper_id": paper_id,
        "source_file": "main.tex",
        "equation_environment": environment,
        "candidate_text": text,
        "normalized_terms": terms if terms is not None else ["mbf", "z0", "class", "MSA"],
        "normalized_window_count": 0,
        "pdf_region_candidate_count": 0,
        "line_local_anchor_status": "no_normalized_windows",
        "pdf_region_anchor_status": status,
        "strict_blockers": ["line_local_normalized_window_missing"],
    }


def _pdf_region_report(root: Path, rows: list[dict], *, schema: str | None = None) -> Path:
    return _write_json(
        root,
        "pdf-region.json",
        {
            "schema": schema or TEX_EQUATION_PDF_REGION_ANCHOR_AUDIT_SCHEMA_ID,
            "status": "ok",
            "generatedAt": "2026-05-18T00:00:00Z",
            "input": {},
            "counts": {},
            "gate": {},
            "policy": {},
            "warnings": [],
            "rows": rows,
        },
    )


def test_remaining_window_diagnostic_classifies_blocked_multiline_macro_rows(tmp_path: Path) -> None:
    report_path = _pdf_region_report(
        tmp_path,
        [
            _pdf_region_row(
                "pdf:0001",
                r"\mbf{z}_0 &= [ \mbf{x}_\text{class}; \mbf{x}^1_p \mbf{E} ] \\ \op{MSA}(\op{LN}(\mbf{z}))",
                terms=["mbf", "z0", "class", "op", "MSA", "LN"],
            ),
            _pdf_region_row("pdf:0002", "x = y", status="unique_pdf_region_anchor_candidate_only"),
        ],
    )

    payload = build_tex_equation_remaining_window_diagnostic(report_path)

    assert payload["schema"] == TEX_EQUATION_REMAINING_WINDOW_DIAGNOSTIC_SCHEMA_ID
    assert validate_payload(payload, TEX_EQUATION_REMAINING_WINDOW_DIAGNOSTIC_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["inputRows"] == 2
    assert payload["counts"]["diagnosticRows"] == 1
    assert payload["counts"]["strictEligibleRows"] == 0
    row = payload["rows"][0]
    assert row["candidate_type"] == "tex_equation_remaining_window_diagnostic"
    assert row["strict_eligible"] is False
    assert row["runtime_evidence"] is False
    assert "multiline_equation_environment" in row["diagnoses"]
    assert "custom_or_rendered_latex_macro_gap" in row["diagnoses"]
    assert row["recommended_action"] == "design_segmented_multiline_equation_matching"


def test_remaining_window_diagnostic_classifies_text_heavy_rows(tmp_path: Path) -> None:
    report_path = _pdf_region_report(
        tmp_path,
        [
            _pdf_region_row(
                "pdf:0001",
                r"\text{Input} &= \text{\tt [CLS] the man went to [MASK] store [SEP]}",
                environment="align*",
                terms=["Input", "tt", "CLS", "MASK", "SEP"],
            )
        ],
    )

    payload = build_tex_equation_remaining_window_diagnostic(report_path)

    row = payload["rows"][0]
    assert "text_heavy_equation_environment" in row["diagnoses"]
    assert row["recommended_action"] == "keep_blocked_or_handle_as_text_example_not_equation_quote"
    assert payload["counts"]["byRecommendedAction"] == {
        "keep_blocked_or_handle_as_text_example_not_equation_quote": 1
    }


def test_remaining_window_diagnostic_blocks_wrong_parent_schema(tmp_path: Path) -> None:
    report_path = _pdf_region_report(
        tmp_path,
        [_pdf_region_row("pdf:0001", "x = y")],
        schema="example.wrong",
    )

    payload = build_tex_equation_remaining_window_diagnostic(report_path)

    assert validate_payload(payload, TEX_EQUATION_REMAINING_WINDOW_DIAGNOSTIC_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "blocked"
    assert payload["counts"]["diagnosticRows"] == 0
    assert payload["gate"]["schemaViolations"] == [
        "tex_equation_pdf_region_anchor_audit_schema_mismatch"
    ]


def test_remaining_window_diagnostic_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    report_path = _pdf_region_report(tmp_path, [_pdf_region_row("pdf:0001", r"\ve{y}=x")])
    payload = build_tex_equation_remaining_window_diagnostic(report_path)

    paths = write_tex_equation_remaining_window_diagnostic_reports(payload, tmp_path / "reports")

    assert set(paths) == {"report", "summary", "markdown"}
    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(report, TEX_EQUATION_REMAINING_WINDOW_DIAGNOSTIC_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["sourceSpanCreatedRows"] == 0
    assert "TeX Equation Remaining Window Diagnostic" in markdown
