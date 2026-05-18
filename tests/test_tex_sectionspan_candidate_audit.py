from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.tex_sectionspan_candidate_audit import (
    TEX_SECTIONSPAN_CANDIDATE_REPORT_SCHEMA_ID,
    build_tex_sectionspan_candidate_report,
    write_tex_sectionspan_candidate_reports,
)


def _write_json(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _row(
    candidate_id: str,
    *,
    paper_id: str = "1234.5678",
    structure_type: str = "section",
    text: str = "Introduction",
    status: str = "aligned",
    method: str = "exact",
    page: int | None = 1,
    source_hash: str | None = "hash-source",
) -> dict:
    return {
        "candidate_id": candidate_id,
        "candidate_type": "tex_structure_candidate",
        "paper_id": paper_id,
        "source_parser": "arxiv_tex",
        "source_file": "main.tex",
        "structure_type": structure_type,
        "tex_command": "\\section",
        "tex_environment": "",
        "candidate_text": text,
        "tex_locator": {"source_file": "main.tex", "chars": {"start": 0, "end": 10}},
        "alignment_status": status,
        "alignment_method": method,
        "alignment_reason": "test",
        "chars_start": 20 if status == "aligned" else None,
        "chars_end": 20 + len(text) if status == "aligned" else None,
        "page": page,
        "sourceContentHash": source_hash,
        "sourceContentHashSource": "manifest",
        "confidence": 0.99 if method == "exact" else 0.82,
        "source_span_locator": {"path": "document.md", "chars": {"start": 20, "end": 20 + len(text)}} if status == "aligned" else {},
        "source_span_candidate_ready": status == "aligned" and page is not None and bool(source_hash),
        "classification": "source_span_candidate_only",
        "mineru_layout_link_status": "failed",
        "mineru_layout_link_method": "none",
        "mineru_candidate_ids": [],
        "mineru_bbox_link_count": 0,
        "evidence_tier": "source_structure_candidate_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": ["runtime_promotion_disabled_for_tranche"],
        "non_strict_reason": ["runtime_promotion_disabled_for_tranche"],
    }


def _alignment_report(root: Path) -> Path:
    return _write_json(
        root,
        "tex-structure-candidate-alignment-report.json",
        {
            "schema": "knowledge-hub.paper.tex-structure-candidate-alignment-audit.v1",
            "status": "ok",
            "candidates": [
                _row("tex:0001", structure_type="section", text="Introduction"),
                _row("tex:0002", structure_type="subsection", text="1.1 Background"),
                _row("tex:0003", structure_type="subsubsection", text="Details", method="normalized"),
                _row("tex:0004", structure_type="section", text="Repeated", status="ambiguous"),
                _row("tex:0005", structure_type="figure_caption", text="A figure caption."),
                _row("tex:0006", structure_type="section", text="No page", page=None),
                _row("tex:0007", structure_type="section", text="No hash", source_hash=""),
            ],
        },
    )


def test_tex_sectionspan_candidate_report_emits_exact_heading_candidates_only(tmp_path: Path) -> None:
    report_path = _alignment_report(tmp_path)

    payload = build_tex_sectionspan_candidate_report(report_path)

    assert payload["schema"] == TEX_SECTIONSPAN_CANDIDATE_REPORT_SCHEMA_ID
    assert validate_payload(payload, TEX_SECTIONSPAN_CANDIDATE_REPORT_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "ok"
    assert payload["counts"]["inputRows"] == 7
    assert payload["counts"]["headingRows"] == 6
    assert payload["counts"]["sectionSpanCandidates"] == 2
    assert payload["counts"]["heldOutCandidates"] == 4
    assert payload["counts"]["strictEligibleCandidates"] == 0
    assert payload["counts"]["runtimeEvidenceCandidates"] == 0
    assert payload["policy"]["answerIntegrationChanged"] is False

    section = payload["candidates"][0]
    assert section["candidate_type"] == "section_span_candidate"
    assert section["source_parser"] == "arxiv_tex+pymupdf_alignment"
    assert section["alignment_method"] == "exact"
    assert section["evidence_tier"] == "sectionspan_candidate_only"
    assert section["strict_eligible"] is False
    assert section["runtime_evidence"] is False
    assert "source_structure_candidate_only" in section["strict_blockers"]

    subsection = payload["candidates"][1]
    assert subsection["section_label"] == "1.1"
    assert subsection["section_title"] == "Background"
    assert subsection["section_level"] == 2

    reasons = payload["counts"]["heldOutByReason"]
    assert reasons["non_exact_alignment"] == 1
    assert reasons["ambiguous_canonical_match"] == 1
    assert reasons["missing_page"] == 1
    assert reasons["missing_source_content_hash"] == 1


def test_tex_sectionspan_candidate_report_filters_paper_id(tmp_path: Path) -> None:
    report_path = _alignment_report(tmp_path)

    payload = build_tex_sectionspan_candidate_report(report_path, paper_ids=["missing-paper"])

    assert validate_payload(payload, TEX_SECTIONSPAN_CANDIDATE_REPORT_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "empty"
    assert payload["counts"]["inputRows"] == 0
    assert payload["counts"]["sectionSpanCandidates"] == 0
    assert payload["policy"]["runtimePromotionAllowed"] is False


def test_tex_sectionspan_candidate_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    report_path = _alignment_report(tmp_path)
    payload = build_tex_sectionspan_candidate_report(report_path)

    paths = write_tex_sectionspan_candidate_reports(payload, tmp_path / "reports")

    assert set(paths) == {"candidates", "summary", "markdown"}
    report = json.loads(Path(paths["candidates"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(report, TEX_SECTIONSPAN_CANDIDATE_REPORT_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["citationGradeCandidates"] == 0
    assert "not strict evidence" in markdown
