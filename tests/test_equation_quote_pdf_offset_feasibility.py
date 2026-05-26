from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.equation_quote_pdf_offset_feasibility import (
    EQUATION_QUOTE_PDF_OFFSET_FEASIBILITY_SCHEMA_ID,
    build_equation_quote_pdf_offset_feasibility_report,
    write_equation_quote_pdf_offset_feasibility_reports,
)
from knowledge_hub.papers.source_text import source_hash_for_path


def _source_pdf(root: Path, paper_id: str) -> Path:
    path = root / "sources" / f"{paper_id}.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"fake pdf bytes for {paper_id}".encode("utf-8"))
    return path


def _parsed_manifest(root: Path, paper_id: str, source_pdf: Path) -> None:
    parsed_dir = root / "parsed" / paper_id
    parsed_dir.mkdir(parents=True, exist_ok=True)
    (parsed_dir / "manifest.json").write_text(
        json.dumps({"parser_meta": {"parser": "pymupdf", "source_pdf": str(source_pdf)}}),
        encoding="utf-8",
    )


def _candidate_report(root: Path, candidates: list[dict]) -> Path:
    path = root / "equation-quote-candidates.json"
    path.write_text(
        json.dumps(
            {
                "schema": "knowledge-hub.paper.equation-quote-candidate-report.v1",
                "status": "ok",
                "candidates": candidates,
            }
        ),
        encoding="utf-8",
    )
    return path


def _candidate(
    *,
    paper_id: str,
    source_hash: str,
    candidate_id: str = "equationquote:paper-1:0001",
    text: str = "x = y + z",
    page: int | None = 2,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "candidate_type": "equation_quote_candidate",
        "paper_id": paper_id,
        "source_parser": "mineru+pymupdf_alignment",
        "candidate_text": text,
        "equation_text": text,
        "equation_label": "tag:1",
        "canonical_alignment_status": "failed" if page is None else "aligned",
        "alignment_method": "none" if page is None else "exact",
        "chars_start": None if page is None else 10,
        "chars_end": None if page is None else 15,
        "page": page,
        "sourceContentHash": source_hash,
        "layout_element_ids": ["mineru:element:1"],
        "bbox": [10.0, 20.0, 200.0, 240.0],
        "layout_region_candidate_present": True,
        "equation_region_link_verified": False,
        "equation_semantics_interpreted": False,
        "strict_eligible": False,
        "citation_grade": False,
        "strict_blockers": ["equation_alignment_missing"],
    }


def test_equation_quote_pdf_offset_recovers_compact_match_but_creates_no_source_span(tmp_path: Path) -> None:
    paper_id = "paper-1"
    source_pdf = _source_pdf(tmp_path, paper_id)
    source_hash = source_hash_for_path(str(source_pdf))
    _parsed_manifest(tmp_path, paper_id, source_pdf)
    candidate_path = _candidate_report(tmp_path, [_candidate(paper_id=paper_id, source_hash=source_hash)])

    payload = build_equation_quote_pdf_offset_feasibility_report(
        equation_quote_report=candidate_path,
        pymupdf_parsed_root=tmp_path / "parsed",
        pdf_page_text_loader=lambda _: [{"page": 2, "text": "The equation is x=y+z in this setup."}],
    )

    assert payload["schema"] == EQUATION_QUOTE_PDF_OFFSET_FEASIBILITY_SCHEMA_ID
    assert validate_payload(payload, EQUATION_QUOTE_PDF_OFFSET_FEASIBILITY_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "feasibility_complete"
    assert payload["counts"]["originalPdfOffsetRecoveredRows"] == 1
    assert payload["counts"]["sourceSpanCreatedRows"] == 0
    row = payload["feasibilityRows"][0]
    assert row["original_pdf_offset_recovered"] is True
    assert row["original_pdf_span"]["matchMethod"] == "compact_whitespace_removed"
    assert row["source_span_created"] is False
    assert row["equation_semantics_interpreted"] is False
    assert row["strict_eligible"] is False
    assert payload["policy"]["sourceSpanCreated"] is False
    assert payload["policy"]["equationSemanticsInterpreted"] is False


def test_diagnostic_page_context_remains_blocked_non_strict(tmp_path: Path) -> None:
    paper_id = "paper-1"
    source_pdf = _source_pdf(tmp_path, paper_id)
    source_hash = source_hash_for_path(str(source_pdf))
    _parsed_manifest(tmp_path, paper_id, source_pdf)
    candidate_path = _candidate_report(
        tmp_path,
        [
            _candidate(
                paper_id=paper_id,
                source_hash=source_hash,
                text=r"\mathrm { A t t e n t i o n } ( Q , K , V ) = \mathrm { s o f t m a x }",
                page=None,
            )
        ],
    )

    payload = build_equation_quote_pdf_offset_feasibility_report(
        equation_quote_report=candidate_path,
        pymupdf_parsed_root=tmp_path / "parsed",
        pdf_page_text_loader=lambda _: [{"page": 3, "text": "Attention and softmax are discussed here."}],
    )

    row = payload["feasibilityRows"][0]
    assert row["original_pdf_offset_recovered"] is False
    assert row["feasibility_status"] == "diagnostic_page_context_candidate_only"
    assert row["diagnostic_page_candidates"][0]["page"] == 3
    assert row["source_span_created"] is False
    assert row["strict_eligible"] is False
    assert "original_pdf_offset_not_recovered" in row["strict_blockers"]


def test_ambiguous_equation_quote_match_is_blocked(tmp_path: Path) -> None:
    paper_id = "paper-1"
    source_pdf = _source_pdf(tmp_path, paper_id)
    source_hash = source_hash_for_path(str(source_pdf))
    _parsed_manifest(tmp_path, paper_id, source_pdf)
    candidate_path = _candidate_report(tmp_path, [_candidate(paper_id=paper_id, source_hash=source_hash)])

    payload = build_equation_quote_pdf_offset_feasibility_report(
        equation_quote_report=candidate_path,
        pymupdf_parsed_root=tmp_path / "parsed",
        pdf_page_text_loader=lambda _: [{"page": 2, "text": "x = y + z and later x = y + z again."}],
    )

    row = payload["feasibilityRows"][0]
    assert row["original_pdf_offset_recovered"] is False
    assert row["feasibility_status"] == "blocked_ambiguous_match"
    assert row["strict_eligible"] is False


def test_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    paper_id = "paper-1"
    source_pdf = _source_pdf(tmp_path, paper_id)
    source_hash = source_hash_for_path(str(source_pdf))
    _parsed_manifest(tmp_path, paper_id, source_pdf)
    candidate_path = _candidate_report(tmp_path, [_candidate(paper_id=paper_id, source_hash=source_hash)])
    payload = build_equation_quote_pdf_offset_feasibility_report(
        equation_quote_report=candidate_path,
        pymupdf_parsed_root=tmp_path / "parsed",
        pdf_page_text_loader=lambda _: [{"page": 2, "text": "x = y + z"}],
    )

    paths = write_equation_quote_pdf_offset_feasibility_reports(payload, tmp_path / "reports")

    assert set(paths) == {"report", "summary", "markdown"}
    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(report, EQUATION_QUOTE_PDF_OFFSET_FEASIBILITY_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, EQUATION_QUOTE_PDF_OFFSET_FEASIBILITY_SCHEMA_ID, strict=True).ok
    assert "does not interpret equations" in markdown
