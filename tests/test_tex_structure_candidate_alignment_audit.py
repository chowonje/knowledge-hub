from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.tex_structure_candidate_alignment_audit import (
    TEX_STRUCTURE_CANDIDATE_ALIGNMENT_AUDIT_SCHEMA_ID,
    build_tex_structure_candidate_alignment_audit,
    write_tex_structure_candidate_alignment_audit_reports,
)


def _write_json(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _parsed_paper(root: Path, paper_id: str, markdown: str) -> Path:
    parsed = root / paper_id
    parsed.mkdir(parents=True)
    source_pdf = root / f"{paper_id}.pdf"
    source_pdf.write_bytes(b"fake pdf bytes")
    _write_json(
        parsed,
        "manifest.json",
        {
            "paper_id": paper_id,
            "parser_meta": {
                "parser": "pymupdf",
                "source_pdf": str(source_pdf),
            },
        },
    )
    parsed.joinpath("document.md").write_text(markdown, encoding="utf-8")
    return parsed


def _availability_report(root: Path, paper_id: str) -> Path:
    return _write_json(
        root,
        "arxiv-source-tex-availability-report.json",
        {
            "schema": "knowledge-hub.paper.arxiv-source-tex-availability-audit.v1",
            "status": "ok",
            "structureRows": [
                {
                    "structure_row_id": "tex:0001",
                    "paper_id": paper_id,
                    "source_file": "main.tex",
                    "structure_type": "section",
                    "tex_command": "\\section",
                    "tex_environment": "",
                    "tex_chars_start": 10,
                    "tex_chars_end": 32,
                    "candidate_text": "Introduction",
                    "mineru_layout_link_status": "failed",
                    "mineru_layout_link_method": "none",
                    "mineru_candidate_ids": [],
                    "mineru_bbox_link_count": 0,
                },
                {
                    "structure_row_id": "tex:0002",
                    "paper_id": paper_id,
                    "source_file": "figures.tex",
                    "structure_type": "figure_caption",
                    "tex_command": "\\caption",
                    "tex_environment": "figure",
                    "tex_chars_start": 40,
                    "tex_chars_end": 70,
                    "candidate_text": "Model overview.",
                    "mineru_layout_link_status": "linked",
                    "mineru_layout_link_method": "text_match",
                    "mineru_candidate_ids": ["mineru:figure:1"],
                    "mineru_bbox_link_count": 1,
                },
                {
                    "structure_row_id": "tex:0003",
                    "paper_id": paper_id,
                    "source_file": "tables.tex",
                    "structure_type": "table_caption",
                    "tex_command": "\\caption",
                    "tex_environment": "table",
                    "tex_chars_start": 80,
                    "tex_chars_end": 100,
                    "candidate_text": "Scores.",
                    "mineru_layout_link_status": "failed",
                    "mineru_layout_link_method": "none",
                    "mineru_candidate_ids": [],
                    "mineru_bbox_link_count": 0,
                },
                {
                    "structure_row_id": "tex:0004",
                    "paper_id": paper_id,
                    "source_file": "math.tex",
                    "structure_type": "equation_environment",
                    "tex_command": "\\begin",
                    "tex_environment": "equation",
                    "tex_chars_start": 110,
                    "tex_chars_end": 120,
                    "candidate_text": "",
                    "mineru_layout_link_status": "blocked",
                    "mineru_layout_link_method": "none",
                    "mineru_candidate_ids": [],
                    "mineru_bbox_link_count": 0,
                },
                {
                    "structure_row_id": "tex:0005",
                    "paper_id": paper_id,
                    "source_file": "main.tex",
                    "structure_type": "subsection",
                    "tex_command": "\\subsection",
                    "tex_environment": "",
                    "tex_chars_start": 130,
                    "tex_chars_end": 150,
                    "candidate_text": "Missing heading",
                    "mineru_layout_link_status": "failed",
                    "mineru_layout_link_method": "none",
                    "mineru_candidate_ids": [],
                    "mineru_bbox_link_count": 0,
                },
            ],
        },
    )


def test_tex_structure_alignment_audit_recovers_source_span_candidates_without_runtime_authority(tmp_path: Path) -> None:
    paper_id = "1234.5678"
    _parsed_paper(
        tmp_path / "parsed",
        paper_id,
        "# Paper\n\n## Page 1\n\nIntroduction\n\nModel overview.\n\n## Page 2\n\nScores.\n\nScores.\n",
    )
    report_path = _availability_report(tmp_path, paper_id)

    payload = build_tex_structure_candidate_alignment_audit(
        input_report=report_path,
        parsed_root=tmp_path / "parsed",
    )

    assert payload["schema"] == TEX_STRUCTURE_CANDIDATE_ALIGNMENT_AUDIT_SCHEMA_ID
    assert validate_payload(payload, TEX_STRUCTURE_CANDIDATE_ALIGNMENT_AUDIT_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["totalRows"] == 5
    assert payload["counts"]["canonicalAlignedRows"] == 2
    assert payload["counts"]["pageRecoveredRows"] == 2
    assert payload["counts"]["sourceSpanCandidateReadyRows"] == 2
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["policy"]["databaseMutation"] is False
    assert payload["policy"]["answerIntegrationChanged"] is False

    section = next(item for item in payload["candidates"] if item["candidate_id"] == "tex:0001")
    assert section["alignment_status"] == "aligned"
    assert section["alignment_method"] == "exact"
    assert section["page"] == 1
    assert section["sourceContentHash"]
    assert section["source_span_candidate_ready"] is True
    assert section["strict_eligible"] is False
    assert "source_structure_candidate_only" in section["strict_blockers"]

    figure = next(item for item in payload["candidates"] if item["candidate_id"] == "tex:0002")
    assert figure["classification"] == "source_span_and_layout_candidate_only"
    assert figure["mineru_bbox_link_count"] == 1
    assert "figure_region_link_unverified" in figure["strict_blockers"]

    table = next(item for item in payload["candidates"] if item["candidate_id"] == "tex:0003")
    assert table["alignment_status"] == "ambiguous"
    assert table["source_span_candidate_ready"] is False
    assert "table_cell_row_column_bbox_provenance_missing" in table["strict_blockers"]

    equation = next(item for item in payload["candidates"] if item["candidate_id"] == "tex:0004")
    assert equation["classification"] == "raw_tex_environment_only"
    assert "equation_text_or_semantics_not_citation_grade" in equation["strict_blockers"]


def test_tex_structure_alignment_audit_filters_paper_ids_and_keeps_missing_unaligned(tmp_path: Path) -> None:
    _parsed_paper(tmp_path / "parsed", "1234.5678", "# Paper\n\n## Page 1\n\nIntroduction\n")
    report_path = _availability_report(tmp_path, "1234.5678")

    payload = build_tex_structure_candidate_alignment_audit(
        input_report=report_path,
        parsed_root=tmp_path / "parsed",
        paper_ids=["missing-paper"],
    )

    assert validate_payload(payload, TEX_STRUCTURE_CANDIDATE_ALIGNMENT_AUDIT_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "blocked"
    assert payload["counts"]["totalRows"] == 0
    assert payload["gate"]["runtimePromotionAllowed"] is False


def test_tex_structure_alignment_audit_does_not_align_section_to_body_text_only(tmp_path: Path) -> None:
    paper_id = "1234.5678"
    _parsed_paper(
        tmp_path / "parsed",
        paper_id,
        "# Paper\n\n## Page 1\n\nThis paragraph mentions Introduction exactly once, but it is not a heading.\n",
    )
    report_path = _availability_report(tmp_path, paper_id)

    payload = build_tex_structure_candidate_alignment_audit(
        input_report=report_path,
        parsed_root=tmp_path / "parsed",
    )

    section = next(item for item in payload["candidates"] if item["candidate_id"] == "tex:0001")
    assert section["alignment_status"] == "failed"
    assert section["alignment_reason"] == "no_heading_context_text_match"
    assert section["source_span_candidate_ready"] is False
    assert section["classification"] == "blocked_no_canonical_match"
    assert "canonical_text_alignment_not_available" in section["strict_blockers"]


def test_tex_structure_alignment_audit_aligns_numbered_heading_inside_page_blob(tmp_path: Path) -> None:
    paper_id = "1234.5678"
    _parsed_paper(
        tmp_path / "parsed",
        paper_id,
        "# Paper\n\n## Page 1\n\nOpening prose. 1 Introduction This starts a section in page-blob text.\n",
    )
    report_path = _availability_report(tmp_path, paper_id)

    payload = build_tex_structure_candidate_alignment_audit(
        input_report=report_path,
        parsed_root=tmp_path / "parsed",
    )

    section = next(item for item in payload["candidates"] if item["candidate_id"] == "tex:0001")
    assert section["alignment_status"] == "aligned"
    assert section["alignment_method"] == "exact"
    assert section["alignment_reason"] == "single_heading_context_exact_match"
    assert section["source_span_candidate_ready"] is True


def test_tex_structure_alignment_audit_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    paper_id = "1234.5678"
    _parsed_paper(tmp_path / "parsed", paper_id, "# Paper\n\n## Page 1\n\nIntroduction\n")
    report_path = _availability_report(tmp_path, paper_id)
    payload = build_tex_structure_candidate_alignment_audit(
        input_report=report_path,
        parsed_root=tmp_path / "parsed",
    )

    paths = write_tex_structure_candidate_alignment_audit_reports(payload, tmp_path / "reports")

    assert set(paths) == {"report", "summary", "markdown"}
    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(report, TEX_STRUCTURE_CANDIDATE_ALIGNMENT_AUDIT_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["runtimeEvidenceRows"] == 0
    assert "This audit is report-only" in markdown
