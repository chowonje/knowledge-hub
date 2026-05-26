from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.figure_caption_artifact_vertical_slice import PaperSpec
from knowledge_hub.papers.text_section_paragraph_span_artifacts import (
    TEXT_SECTION_PARAGRAPH_SPAN_REPORT_SCHEMA_ID,
    TEXT_SPAN_ARTIFACT_CANDIDATE_SCHEMA_ID,
    build_text_section_paragraph_span_report,
    extract_text_span_candidates_from_blocks,
    write_report,
)


def _blocks() -> list[tuple[int, list[tuple[float, float, float, float, str, int, int]]]]:
    return [
        (
            1,
            [
                (10.0, 10.0, 200.0, 25.0, "Abstract", 0, 0),
                (
                    10.0,
                    30.0,
                    200.0,
                    80.0,
                    "We propose a local-first evidence runtime for auditable paper question answering with stable provenance.",
                    1,
                    0,
                ),
                (10.0, 90.0, 200.0, 110.0, "Figure 1: This caption should not become a paragraph.", 2, 0),
            ],
        ),
        (
            2,
            [
                (10.0, 10.0, 200.0, 25.0, "1 Introduction", 0, 0),
                (
                    10.0,
                    30.0,
                    200.0,
                    80.0,
                    "The introduction explains why source hashes, page locators, and character offsets are required.",
                    1,
                    0,
                ),
            ],
        ),
    ]


def test_extracts_section_and_paragraph_candidates_with_char_and_page_provenance() -> None:
    candidates, diagnostics = extract_text_span_candidates_from_blocks(
        paper_id="sample-paper",
        paper_ref="papers_dir/sample.pdf",
        source_content_hash="sha256:" + "1" * 64,
        blocks_by_page=_blocks(),
    )

    section_rows = [row for row in candidates if row["spanType"] == "section"]
    paragraph_rows = [row for row in candidates if row["spanType"] == "paragraph"]

    assert len(section_rows) == 2
    assert len(paragraph_rows) == 2
    assert diagnostics["sectionSpanRows"] == 2
    assert diagnostics["paragraphSpanRows"] == 2
    assert diagnostics["textAssemblyHash"].startswith("sha256:")
    assert all(row["sourceContentHash"] == "sha256:" + "1" * 64 for row in candidates)
    assert all(row["charEnd"] > row["charStart"] for row in candidates)
    assert all(row["page"] in {1, 2} for row in candidates)
    assert all(len(row["bbox"]) == 4 for row in candidates)
    assert all(validate_payload(row, TEXT_SPAN_ARTIFACT_CANDIDATE_SCHEMA_ID, strict=True).ok for row in candidates)


def test_build_report_records_blocker_for_missing_source(tmp_path: Path) -> None:
    report = build_text_section_paragraph_span_report(
        papers_root=tmp_path,
        paper_specs=[PaperSpec(paper_id="missing-paper", filename="missing.pdf")],
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["candidateRows"] == 0
    assert report["blockerRows"] == 1
    assert report["blockers"][0]["blockerReason"] == "source_pdf_missing"
    assert validate_payload(report, TEXT_SECTION_PARAGRAPH_SPAN_REPORT_SCHEMA_ID, strict=True).ok


def test_report_schema_accepts_fake_block_candidates() -> None:
    candidates, diagnostics = extract_text_span_candidates_from_blocks(
        paper_id="sample-paper",
        paper_ref="papers_dir/sample.pdf",
        source_content_hash="sha256:" + "1" * 64,
        blocks_by_page=_blocks(),
    )
    diagnostics["pageCount"] = 2
    diagnostics["candidateRows"] = len(candidates)
    report = {
        "schema": TEXT_SECTION_PARAGRAPH_SPAN_REPORT_SCHEMA_ID,
        "status": "ready",
        "generatedAt": "2026-05-26T00:00:00Z",
        "scope": {
            "paperRows": 1,
            "paperRefs": ["papers_dir/sample.pdf"],
            "writes": "report_only",
            "canonicalParsedArtifactWriteRows": 0,
            "charBasis": "pymupdf_block_reading_order_normalized_text_v1",
        },
        "paperDiagnostics": [diagnostics],
        "candidateRows": len(candidates),
        "sectionSpanRows": 2,
        "paragraphSpanRows": 2,
        "candidates": candidates,
        "blockerRows": 0,
        "blockers": [],
        "mutationCounters": {
            "canonicalParsedArtifactWriteRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "vaultScanRows": 0,
            "externalDownloadRows": 0,
            "strictEvidencePromotionRows": 0,
            "runtimeAnswerVisibleExposureRows": 0,
        },
        "schemaViolationCount": 0,
        "privatePathLeakRows": 0,
        "warnings": [],
        "schemaErrors": [],
    }

    assert validate_payload(report, TEXT_SECTION_PARAGRAPH_SPAN_REPORT_SCHEMA_ID, strict=True).ok


def test_report_writer_uses_sanitized_refs(tmp_path: Path) -> None:
    report = build_text_section_paragraph_span_report(
        papers_root=tmp_path,
        paper_specs=[PaperSpec(paper_id="missing-paper", filename="missing.pdf")],
        generated_at="2026-05-26T00:00:00Z",
    )
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    write_report(report, json_path=json_path, markdown_path=md_path)

    combined = json_path.read_text(encoding="utf-8") + md_path.read_text(encoding="utf-8")
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "papers_dir/missing.pdf" in combined
    assert json.loads(json_path.read_text(encoding="utf-8"))["privatePathLeakRows"] == 0
