from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.tex_equation_remaining_window_diagnostic import (
    TEX_EQUATION_REMAINING_WINDOW_DIAGNOSTIC_SCHEMA_ID,
)
from knowledge_hub.papers.tex_equation_rendered_macro_term_profile_design import (
    TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_SCHEMA_ID,
    build_tex_equation_rendered_macro_term_profile_design,
    write_tex_equation_rendered_macro_term_profile_design_reports,
)


def _write_json(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _parsed_root(root: Path, *, paper_id: str = "paper-1") -> Path:
    parsed = root / "parsed"
    paper = parsed / paper_id
    paper.mkdir(parents=True)
    source_pdf = root / f"{paper_id}.pdf"
    source_pdf.write_bytes(b"%PDF fake fixture")
    (paper / "document.md").write_text(
        "## Page 1\n\nThe projection form is y = F(x, {Wi}) + Wsx. (2)\n",
        encoding="utf-8",
    )
    (paper / "manifest.json").write_text(
        json.dumps({"parser_meta": {"source_pdf": str(source_pdf)}}),
        encoding="utf-8",
    )
    return parsed


def _diagnostic_row(
    row_id: str,
    text: str,
    *,
    paper_id: str = "paper-1",
    action: str = "design_rendered_macro_term_profile",
) -> dict:
    return {
        "diagnostic_id": row_id,
        "candidate_type": "tex_equation_remaining_window_diagnostic",
        "source_pdf_region_anchor_id": row_id.replace("diagnostic", "pdf"),
        "source_line_local_anchor_id": row_id.replace("diagnostic", "line"),
        "source_design_id": row_id.replace("diagnostic", "design"),
        "source_candidate_id": row_id.replace("diagnostic", "candidate"),
        "paper_id": paper_id,
        "source_parser": "arxiv_tex+pymupdf_pdf_blocks",
        "source_file": "main.tex",
        "equation_environment": "equation",
        "candidate_text": text,
        "candidate_text_length": len(text),
        "normalized_terms": ["ve", "mathcal", "Wi", "Ws"],
        "normalized_term_count": 4,
        "custom_latex_macros": ["ve", "mathcal"],
        "custom_latex_macro_count": 2,
        "line_local_anchor_status": "no_normalized_windows",
        "pdf_region_anchor_status": "blocked_no_line_local_normalized_window",
        "normalized_window_count": 0,
        "pdf_region_candidate_count": 0,
        "diagnoses": ["custom_or_rendered_latex_macro_gap"],
        "recommended_action": action,
        "evidence_tier": "remaining_window_diagnostic_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "source_span_created": False,
        "parser_routing_changed": False,
        "answer_integration_changed": False,
        "strict_blockers": ["remaining_window_rows_are_diagnostic_only"],
        "non_strict_reason": ["remaining_window_rows_are_diagnostic_only"],
    }


def _diagnostic_report(root: Path, rows: list[dict], *, schema: str | None = None) -> Path:
    return _write_json(
        root,
        "remaining-window.json",
        {
            "schema": schema or TEX_EQUATION_REMAINING_WINDOW_DIAGNOSTIC_SCHEMA_ID,
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


def _fake_pdf_blocks(_source_pdf: str | Path) -> list[dict]:
    return [
        {
            "page": 1,
            "blocks": [
                {"block_index": 1, "bbox": [10.0, 10.0, 140.0, 20.0], "text": "y = F(x, {Wi}) + Wsx. (2)"},
            ],
        }
    ]


def test_rendered_macro_profile_emits_alias_term_candidates(tmp_path: Path) -> None:
    parsed = _parsed_root(tmp_path)
    report_path = _diagnostic_report(
        tmp_path,
        [_diagnostic_row("diagnostic:0001", r"\label{eq:transform} \ve{y}= \mathcal{F}(\ve{x}, \{W_{i}\}) + W_{s}\ve{x}.")],
    )

    payload = build_tex_equation_rendered_macro_term_profile_design(
        report_path,
        parsed_root=parsed,
        pdf_block_loader=_fake_pdf_blocks,
    )

    assert payload["schema"] == TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_SCHEMA_ID
    assert validate_payload(payload, TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["targetRows"] == 1
    assert payload["counts"]["candidateReadyRows"] == 1
    assert payload["counts"]["strictEligibleRows"] == 0
    row = payload["rows"][0]
    assert row["rendered_alias_text"] == "y = F ( x , Wi ) + Ws x ."
    assert row["latex_labels"] == ["eq:transform"]
    assert row["recommended_profile"] == "rendered_macro_alias_terms_v1"
    rendered = [profile for profile in row["profile_results"] if profile["profile_name"] == "rendered_macro_alias_terms_v1"][0]
    assert rendered["normalized_terms"] == ["y", "F", "x", "Wi", "Ws"]
    assert rendered["canonical_match_status"] == "unique_rendered_macro_canonical_window_candidate_only"
    assert rendered["pdf_region_match_status"] == "unique_rendered_macro_pdf_region_candidate_only"
    assert row["source_span_created"] is False
    assert row["runtime_evidence"] is False


def test_rendered_macro_profile_ignores_other_recommended_actions(tmp_path: Path) -> None:
    parsed = _parsed_root(tmp_path)
    report_path = _diagnostic_report(
        tmp_path,
        [
            _diagnostic_row(
                "diagnostic:0001",
                r"alpha = beta \\ gamma = delta",
                action="design_segmented_multiline_equation_matching",
            )
        ],
    )

    payload = build_tex_equation_rendered_macro_term_profile_design(
        report_path,
        parsed_root=parsed,
        pdf_block_loader=_fake_pdf_blocks,
    )

    assert validate_payload(payload, TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "blocked"
    assert payload["counts"]["inputRows"] == 1
    assert payload["counts"]["targetRows"] == 0
    assert payload["rows"] == []


def test_rendered_macro_profile_blocks_wrong_parent_schema(tmp_path: Path) -> None:
    parsed = _parsed_root(tmp_path)
    report_path = _diagnostic_report(
        tmp_path,
        [_diagnostic_row("diagnostic:0001", r"\ve{y}= \mathcal{F}(\ve{x})")],
        schema="example.wrong",
    )

    payload = build_tex_equation_rendered_macro_term_profile_design(
        report_path,
        parsed_root=parsed,
        pdf_block_loader=_fake_pdf_blocks,
    )

    assert validate_payload(payload, TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "blocked"
    assert payload["counts"]["schemaViolationCount"] == 1
    assert payload["gate"]["schemaViolations"] == [
        "tex_equation_remaining_window_diagnostic_schema_mismatch"
    ]


def test_rendered_macro_profile_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    parsed = _parsed_root(tmp_path)
    report_path = _diagnostic_report(
        tmp_path,
        [_diagnostic_row("diagnostic:0001", r"\ve{y}= \mathcal{F}(\ve{x}, \{W_{i}\}) + W_{s}\ve{x}.")],
    )
    payload = build_tex_equation_rendered_macro_term_profile_design(
        report_path,
        parsed_root=parsed,
        pdf_block_loader=_fake_pdf_blocks,
    )

    paths = write_tex_equation_rendered_macro_term_profile_design_reports(payload, tmp_path / "reports")

    assert set(paths) == {"report", "summary", "markdown"}
    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(report, TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["sourceSpanCreatedRows"] == 0
    assert "TeX Equation Rendered Macro Term Profile Design" in markdown
