from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.source_text import source_hash_for_path
from knowledge_hub.papers.table_region_pdf_offset_feasibility import (
    TABLE_REGION_PDF_OFFSET_FEASIBILITY_SCHEMA_ID,
    build_table_region_pdf_offset_feasibility_report,
    write_table_region_pdf_offset_feasibility_reports,
)


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
    path = root / "table-region-candidates.json"
    path.write_text(
        json.dumps(
            {
                "schema": "knowledge-hub.paper.table-region-candidate-report.v1",
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
    candidate_id: str = "tableregion:paper-1:0001",
    text: str = "Table 1: Model results.",
    caption_text: str = "Model results.",
    page: int | None = 2,
    alignment_status: str = "aligned",
) -> dict:
    return {
        "candidate_id": candidate_id,
        "candidate_type": "table_region_candidate",
        "paper_id": paper_id,
        "source_parser": "mineru+pymupdf_alignment",
        "candidate_text": text,
        "table_label": "Table 1",
        "caption_text": caption_text,
        "canonical_alignment_status": alignment_status,
        "alignment_method": "exact" if alignment_status == "aligned" else "none",
        "chars_start": 10 if alignment_status == "aligned" else None,
        "chars_end": 32 if alignment_status == "aligned" else None,
        "page": page,
        "sourceContentHash": source_hash,
        "layout_element_ids": ["mineru:element:1"],
        "bbox": [10.0, 20.0, 200.0, 240.0],
        "layout_region_candidate_present": True,
        "table_region_link_verified": False,
        "table_cell_evidence_available": False,
        "table_cell_citation_grade": False,
        "strict_eligible": False,
        "citation_grade": False,
        "strict_blockers": ["table_cell_provenance_missing"],
    }


def test_table_region_pdf_offset_recovers_unique_caption_but_no_cell_evidence(tmp_path: Path) -> None:
    paper_id = "paper-1"
    source_pdf = _source_pdf(tmp_path, paper_id)
    source_hash = source_hash_for_path(str(source_pdf))
    _parsed_manifest(tmp_path, paper_id, source_pdf)
    candidate_path = _candidate_report(tmp_path, [_candidate(paper_id=paper_id, source_hash=source_hash)])

    payload = build_table_region_pdf_offset_feasibility_report(
        table_region_candidate_report=candidate_path,
        pymupdf_parsed_root=tmp_path / "parsed",
        pdf_page_text_loader=lambda _: [{"page": 2, "text": "Table 1: Model results.\nA B\n1 2"}],
    )

    assert payload["schema"] == TABLE_REGION_PDF_OFFSET_FEASIBILITY_SCHEMA_ID
    assert validate_payload(payload, TABLE_REGION_PDF_OFFSET_FEASIBILITY_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "feasibility_complete"
    assert payload["counts"]["originalPdfOffsetRecoveredRows"] == 1
    assert payload["counts"]["tableCellEvidenceRows"] == 0
    row = payload["feasibilityRows"][0]
    assert row["original_pdf_offset_recovered"] is True
    assert row["original_pdf_span"]["page"] == 2
    assert row["match_target"] == "full_candidate_text"
    assert row["table_cell_evidence_available"] is False
    assert row["table_cell_citation_grade"] is False
    assert row["strict_eligible"] is False
    assert "table_cell_row_column_bbox_provenance_missing" in row["strict_blockers"]
    assert payload["policy"]["tableCellEvidenceCreated"] is False
    assert payload["policy"]["strictEvidenceCreated"] is False


def test_caption_body_fallback_remains_non_strict(tmp_path: Path) -> None:
    paper_id = "paper-1"
    source_pdf = _source_pdf(tmp_path, paper_id)
    source_hash = source_hash_for_path(str(source_pdf))
    _parsed_manifest(tmp_path, paper_id, source_pdf)
    candidate_path = _candidate_report(tmp_path, [_candidate(paper_id=paper_id, source_hash=source_hash)])

    payload = build_table_region_pdf_offset_feasibility_report(
        table_region_candidate_report=candidate_path,
        pymupdf_parsed_root=tmp_path / "parsed",
        pdf_page_text_loader=lambda _: [{"page": 2, "text": "Model results.\nA B\n1 2"}],
    )

    row = payload["feasibilityRows"][0]
    assert row["original_pdf_offset_recovered"] is True
    assert row["match_target"] == "caption_body_text"
    assert row["strict_eligible"] is False
    assert "full_table_caption_label_not_recovered_in_original_pdf_match" in row["strict_blockers"]


def test_ambiguous_table_caption_match_is_blocked(tmp_path: Path) -> None:
    paper_id = "paper-1"
    source_pdf = _source_pdf(tmp_path, paper_id)
    source_hash = source_hash_for_path(str(source_pdf))
    _parsed_manifest(tmp_path, paper_id, source_pdf)
    candidate_path = _candidate_report(tmp_path, [_candidate(paper_id=paper_id, source_hash=source_hash)])

    payload = build_table_region_pdf_offset_feasibility_report(
        table_region_candidate_report=candidate_path,
        pymupdf_parsed_root=tmp_path / "parsed",
        pdf_page_text_loader=lambda _: [{"page": 2, "text": "Table 1: Model results.\nTable 1: Model results."}],
    )

    row = payload["feasibilityRows"][0]
    assert row["original_pdf_offset_recovered"] is False
    assert row["feasibility_status"] == "blocked_ambiguous_match"
    assert row["strict_eligible"] is False
    assert "original_pdf_offset_not_recovered" in row["strict_blockers"]


def test_failed_canonical_alignment_can_only_be_pdf_feasibility(tmp_path: Path) -> None:
    paper_id = "paper-1"
    source_pdf = _source_pdf(tmp_path, paper_id)
    source_hash = source_hash_for_path(str(source_pdf))
    _parsed_manifest(tmp_path, paper_id, source_pdf)
    candidate_path = _candidate_report(
        tmp_path,
        [_candidate(paper_id=paper_id, source_hash=source_hash, alignment_status="failed", page=None)],
    )

    payload = build_table_region_pdf_offset_feasibility_report(
        table_region_candidate_report=candidate_path,
        pymupdf_parsed_root=tmp_path / "parsed",
        pdf_page_text_loader=lambda _: [{"page": 4, "text": "Table 1: Model results."}],
    )

    row = payload["feasibilityRows"][0]
    assert row["original_pdf_offset_recovered"] is True
    assert row["canonical_alignment_status"] == "failed"
    assert row["page_agrees_with_canonical"] is False
    assert row["evidence_tier"] == "table_region_pdf_offset_feasibility_only"
    assert row["strict_eligible"] is False


def test_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    paper_id = "paper-1"
    source_pdf = _source_pdf(tmp_path, paper_id)
    source_hash = source_hash_for_path(str(source_pdf))
    _parsed_manifest(tmp_path, paper_id, source_pdf)
    candidate_path = _candidate_report(tmp_path, [_candidate(paper_id=paper_id, source_hash=source_hash)])
    payload = build_table_region_pdf_offset_feasibility_report(
        table_region_candidate_report=candidate_path,
        pymupdf_parsed_root=tmp_path / "parsed",
        pdf_page_text_loader=lambda _: [{"page": 2, "text": "Table 1: Model results."}],
    )

    paths = write_table_region_pdf_offset_feasibility_reports(payload, tmp_path / "reports")

    assert set(paths) == {"report", "summary", "markdown"}
    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(report, TABLE_REGION_PDF_OFFSET_FEASIBILITY_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, TABLE_REGION_PDF_OFFSET_FEASIBILITY_SCHEMA_ID, strict=True).ok
    assert "does not create table-cell evidence" in markdown
