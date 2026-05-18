from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.arxiv_source_tex_availability_audit import (
    ARXIV_SOURCE_TEX_AVAILABILITY_AUDIT_SCHEMA_ID,
)
from knowledge_hub.papers.tex_equation_label_number_pdf_region_disambiguation_design import (
    TEX_EQUATION_LABEL_NUMBER_PDF_REGION_DISAMBIGUATION_DESIGN_SCHEMA_ID,
    build_tex_equation_label_number_pdf_region_disambiguation_design,
    write_tex_equation_label_number_pdf_region_disambiguation_design_reports,
)
from knowledge_hub.papers.tex_equation_rendered_macro_term_profile_design import (
    TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_SCHEMA_ID,
)


def _write_json(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _parsed_root(root: Path, *, paper_id: str = "1512.03385") -> Path:
    parsed = root / "parsed"
    paper = parsed / paper_id
    paper.mkdir(parents=True)
    source_pdf = root / f"{paper_id}.pdf"
    source_pdf.write_bytes(b"%PDF fake fixture")
    (paper / "manifest.json").write_text(
        json.dumps({"parser_meta": {"source_pdf": str(source_pdf)}}),
        encoding="utf-8",
    )
    (paper / "document.md").write_text(
        "## Page 3\n\ny = F(x, {Wi}) + x. (1)\n\ny = F(x, {Wi}) + Wsx. (2)\n",
        encoding="utf-8",
    )
    return parsed


def _rendered_macro_row(
    *,
    action: str = "requires_equation_number_or_label_to_pdf_region_disambiguation_design",
    source_candidate_id: str = "arxiv-source-structure:1512.03385:0040",
) -> dict:
    return {
        "design_id": "tex-equation-rendered-macro-term-profile-design:0001",
        "candidate_type": "tex_equation_rendered_macro_term_profile_design",
        "source_diagnostic_id": "tex-equation-remaining-window-diagnostic:0006",
        "source_pdf_region_anchor_id": "tex-equation-pdf-region-anchor:0016",
        "source_line_local_anchor_id": "tex-equation-line-local-anchor:0016",
        "source_design_id": "tex-equation-canonical-text-normalizer-design:0016",
        "source_candidate_id": source_candidate_id,
        "paper_id": "1512.03385",
        "source_parser": "arxiv_tex+pymupdf_pdf_blocks",
        "source_file": "residual_v1_arxiv_release.tex",
        "equation_environment": "equation",
        "candidate_text": r"\label{eq:identity} \ve{y}= \mathcal{F}(\ve{x}, \{W_{i}\}) + \ve{x}.",
        "rendered_alias_text": "y = F ( x , Wi ) + x .",
        "latex_labels": ["eq:identity"],
        "custom_latex_macros": ["ve", "mathcal"],
        "source_context_status": "ok",
        "sourceContentHash": "hash",
        "source_pdf_path": "paper.pdf",
        "source_manifest_path": "manifest.json",
        "profile_results": [
            {
                "profile_name": "rendered_macro_alias_terms_v1",
                "description": "test",
                "proposed_rules": ["ordered_anchor_token_window"],
                "normalized_terms": ["y", "F", "x", "Wi"],
                "normalized_term_count": 4,
                "canonical_match_status": "ambiguous_rendered_macro_canonical_window_candidate_only",
                "canonical_window_count": 3,
                "pdf_region_match_status": "ambiguous_rendered_macro_pdf_region_candidate_only",
                "pdf_region_candidate_count": 2,
                "selected_pdf_region": {
                    "page": 3,
                    "bbox": [375.0, 264.0, 545.0, 281.0],
                    "blockIndexes": [10],
                    "matchedTerms": ["y", "F", "x", "Wi"],
                    "coverage": 1.0,
                    "formulaScore": 1.7,
                    "textPreview": "y = F(x, {Wi}) + Wsx. (2)",
                },
                "profile_status": "ambiguous_rendered_macro_profile_candidate_only",
                "profile_candidate_ready": False,
            }
        ],
        "recommended_profile": "rendered_macro_alias_terms_v1",
        "recommended_status": "ambiguous_rendered_macro_profile_candidate_only",
        "candidate_ready": False,
        "evidence_tier": "rendered_macro_term_profile_design_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "source_span_created": False,
        "parser_routing_changed": False,
        "answer_integration_changed": False,
        "recommended_action": action,
        "strict_blockers": ["rendered_macro_term_profile_design_only"],
        "non_strict_reason": ["rendered_macro_term_profile_design_only"],
    }


def _rendered_macro_report(root: Path, rows: list[dict], *, schema: str | None = None) -> Path:
    return _write_json(
        root,
        "rendered-macro.json",
        {
            "schema": schema or TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_SCHEMA_ID,
            "status": "ok",
            "generatedAt": "2026-05-18T00:00:00Z",
            "input": {},
            "counts": {},
            "gate": {},
            "policy": {},
            "profiles": [],
            "warnings": [],
            "rows": rows,
        },
    )


def _source_row(row_id: str, text: str, *, start: int) -> dict:
    return {
        "structure_row_id": row_id,
        "paper_id": "1512.03385",
        "source_file": "residual_v1_arxiv_release.tex",
        "structure_type": "equation_environment",
        "tex_environment": "equation",
        "tex_chars_start": start,
        "tex_chars_end": start + len(text),
        "candidate_text": text,
    }


def _source_report(root: Path, rows: list[dict], *, schema: str | None = None) -> Path:
    return _write_json(
        root,
        "source-report.json",
        {
            "schema": schema or ARXIV_SOURCE_TEX_AVAILABILITY_AUDIT_SCHEMA_ID,
            "status": "ok",
            "generatedAt": "2026-05-18T00:00:00Z",
            "inputs": {},
            "counts": {},
            "gate": {},
            "policy": {},
            "warnings": [],
            "papers": [],
            "structureRows": rows,
        },
    )


def _fake_pdf_blocks(_source_pdf: str | Path) -> list[dict]:
    return [
        {
            "page": 3,
            "blocks": [
                {"block_index": 10, "bbox": [375.0, 264.0, 545.0, 281.0], "text": "y = F(x, {Wi}) + Wsx. (2)"},
                {"block_index": 6, "bbox": [123.0, 626.0, 286.0, 643.0], "text": "y = F(x, {Wi}) + x. (1)"},
            ],
        }
    ]


def test_label_number_disambiguates_rendered_macro_pdf_region(tmp_path: Path) -> None:
    parsed = _parsed_root(tmp_path)
    rendered_report = _rendered_macro_report(tmp_path, [_rendered_macro_row()])
    source_report = _source_report(
        tmp_path,
        [
            _source_row(
                "arxiv-source-structure:1512.03385:0040",
                r"\label{eq:identity} \ve{y}= \mathcal{F}(\ve{x}, \{W_{i}\}) + \ve{x}.",
                start=10,
            ),
            _source_row(
                "arxiv-source-structure:1512.03385:0041",
                r"\label{eq:transform} \ve{y}= \mathcal{F}(\ve{x}, \{W_{i}\}) + W_{s}\ve{x}.",
                start=20,
            ),
        ],
    )

    payload = build_tex_equation_label_number_pdf_region_disambiguation_design(
        rendered_report,
        arxiv_source_tex_availability_report=source_report,
        parsed_root=parsed,
        pdf_block_loader=_fake_pdf_blocks,
    )

    assert payload["schema"] == TEX_EQUATION_LABEL_NUMBER_PDF_REGION_DISAMBIGUATION_DESIGN_SCHEMA_ID
    assert validate_payload(payload, TEX_EQUATION_LABEL_NUMBER_PDF_REGION_DISAMBIGUATION_DESIGN_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["targetRows"] == 1
    assert payload["counts"]["candidateReadyRows"] == 1
    row = payload["rows"][0]
    assert row["source_label_number_hint"]["latexLabels"] == ["eq:identity"]
    assert row["source_label_number_hint"]["inferredEquationNumbers"] == ["1"]
    assert row["pdf_region_candidate_count"] == 2
    assert row["label_number_matching_candidate_count"] == 1
    assert row["selected_pdf_region"]["equationNumbers"] == ["1"]
    assert row["selected_pdf_region"]["textPreview"] == "y = F(x, {Wi}) + x. (1)"
    assert row["disambiguation_status"] == "unique_label_number_pdf_region_candidate_only"
    assert row["source_span_created"] is False
    assert row["runtime_evidence"] is False


def test_label_number_ignores_non_target_rows(tmp_path: Path) -> None:
    parsed = _parsed_root(tmp_path)
    rendered_report = _rendered_macro_report(
        tmp_path,
        [_rendered_macro_row(action="review_rendered_macro_unique_candidate_before_any_later_promotion_design")],
    )
    source_report = _source_report(tmp_path, [])

    payload = build_tex_equation_label_number_pdf_region_disambiguation_design(
        rendered_report,
        arxiv_source_tex_availability_report=source_report,
        parsed_root=parsed,
        pdf_block_loader=_fake_pdf_blocks,
    )

    assert validate_payload(payload, TEX_EQUATION_LABEL_NUMBER_PDF_REGION_DISAMBIGUATION_DESIGN_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "blocked"
    assert payload["counts"]["inputRows"] == 1
    assert payload["counts"]["targetRows"] == 0
    assert payload["rows"] == []


def test_label_number_blocks_wrong_parent_schema(tmp_path: Path) -> None:
    parsed = _parsed_root(tmp_path)
    rendered_report = _rendered_macro_report(tmp_path, [_rendered_macro_row()], schema="example.wrong")
    source_report = _source_report(tmp_path, [])

    payload = build_tex_equation_label_number_pdf_region_disambiguation_design(
        rendered_report,
        arxiv_source_tex_availability_report=source_report,
        parsed_root=parsed,
        pdf_block_loader=_fake_pdf_blocks,
    )

    assert validate_payload(payload, TEX_EQUATION_LABEL_NUMBER_PDF_REGION_DISAMBIGUATION_DESIGN_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "blocked"
    assert payload["counts"]["schemaViolationCount"] == 1
    assert payload["gate"]["schemaViolations"] == [
        "tex_equation_rendered_macro_term_profile_design_schema_mismatch"
    ]


def test_label_number_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    parsed = _parsed_root(tmp_path)
    rendered_report = _rendered_macro_report(tmp_path, [_rendered_macro_row()])
    source_report = _source_report(
        tmp_path,
        [
            _source_row(
                "arxiv-source-structure:1512.03385:0040",
                r"\label{eq:identity} \ve{y}= \mathcal{F}(\ve{x}, \{W_{i}\}) + \ve{x}.",
                start=10,
            )
        ],
    )
    payload = build_tex_equation_label_number_pdf_region_disambiguation_design(
        rendered_report,
        arxiv_source_tex_availability_report=source_report,
        parsed_root=parsed,
        pdf_block_loader=_fake_pdf_blocks,
    )

    paths = write_tex_equation_label_number_pdf_region_disambiguation_design_reports(payload, tmp_path / "reports")

    assert set(paths) == {"report", "summary", "markdown"}
    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(report, TEX_EQUATION_LABEL_NUMBER_PDF_REGION_DISAMBIGUATION_DESIGN_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["sourceSpanCreatedRows"] == 0
    assert "TeX Equation Label/Number PDF-Region Disambiguation Design" in markdown
