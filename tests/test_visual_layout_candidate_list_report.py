from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.visual_layout_candidate_list_report import (
    VISUAL_LAYOUT_CANDIDATE_LIST_REPORT_SCHEMA_ID,
    build_report_from_candidate_rows,
    extract_candidates_from_blocks,
    extract_candidates_from_images,
    extract_candidates_from_parsed_elements,
    write_visual_layout_candidate_list_reports,
)


def _source_hash() -> str:
    return "sha256:" + "1" * 64


def _sample_rows() -> list[dict[str, object]]:
    block_rows = extract_candidates_from_blocks(
        paper_id="sample-paper",
        paper_ref="papers_dir/sample.pdf",
        source_content_hash=_source_hash(),
        blocks_by_page=[
            (
                2,
                [
                    (10, 10, 240, 30, "1 Introduction", 0, 0),
                    (10, 40, 240, 80, "Figure 1: A model diagram with visual-only labels.", 1, 0),
                    (
                        10,
                        90,
                        240,
                        140,
                        "metric  baseline  model\naccuracy  73.2  81.5\nerror  26.8  18.5",
                        2,
                        0,
                    ),
                    (10, 150, 240, 180, "y = Wx + b", 3, 0),
                ],
            )
        ],
    )
    image_rows = extract_candidates_from_images(
        paper_id="sample-paper",
        paper_ref="papers_dir/sample.pdf",
        source_content_hash=_source_hash(),
        image_refs_by_page=[
            (
                2,
                [
                    {
                        "bbox": [12, 200, 220, 340],
                        "imageHash": "sha256:" + "2" * 64,
                    }
                ],
            )
        ],
    )
    return [*block_rows, *image_rows]


def test_extracts_visual_layout_candidate_types_from_blocks_and_images() -> None:
    rows = _sample_rows()
    by_type = {str(row["candidateType"]) for row in rows}

    assert "figure_caption_region" in by_type
    assert "table_region" in by_type
    assert "equation_region" in by_type
    assert "layout_region" in by_type
    assert "image_region" in by_type
    for row in rows:
        assert row["paperRef"] == "papers_dir/sample.pdf"
        assert row["sourceContentHash"] == _source_hash()
        assert row["page"] == 2
        assert row["bbox"]
        plan = row["retrievalHintPlan"]
        assert plan["allowedUse"] == "retrieval_hint_only"
        assert plan["strictEvidence"] is False
        assert plan["citationGrade"] is False
        assert plan["answerableWithoutTextEvidence"] is False


def test_parsed_elements_can_contribute_normalized_candidates() -> None:
    rows = extract_candidates_from_parsed_elements(
        paper_id="parsed-paper",
        paper_ref="papers_dir/parsed.pdf",
        source_content_hash=_source_hash(),
        extraction_method="parsed_artifact_mineru_element_metadata_v1",
        elements=[
            {
                "type": "image",
                "text": "diagram panel",
                "page": 4,
                "bbox": [1, 2, 3, 4],
                "heading_path": ["Results"],
            },
            {
                "type": "table",
                "text": "Table 2: accuracy by split",
                "page": 5,
                "bbox": [10, 20, 30, 40],
                "heading_path": ["Evaluation"],
            },
        ],
    )

    assert [row["candidateType"] for row in rows] == ["image_region", "table_region"]
    assert rows[0]["textContext"]["headingPath"] == ["Results"]
    assert rows[1]["textContext"]["captionText"] == "accuracy by split"


def test_report_schema_validates_and_preserves_no_mutation_contract() -> None:
    report = build_report_from_candidate_rows(
        input_paper_rows=1,
        candidate_rows=_sample_rows(),
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "ready"
    assert report["decision"] == "ready_for_visual_annotation_request_pack_design"
    assert report["nextRecommendedTranche"] == "visual_annotation_request_pack_design"
    assert report["scope"]["writes"] == "report_only"
    assert report["scope"]["modelCalls"] is False
    assert report["scope"]["vectorIndexing"] is False
    assert report["scope"]["strictEvidencePromotionRows"] == 0
    assert report["scope"]["runtimeAnswerVisibleExposureRows"] == 0
    assert report["scope"]["databaseMutationRows"] == 0
    assert report["scope"]["indexMutationRows"] == 0
    assert report["scope"]["reindexOrReembedRows"] == 0
    assert report["scope"]["vaultScanRows"] == 0
    assert report["scope"]["externalDownloadRows"] == 0
    assert report["scope"]["answerabilityGateBypassRows"] == 0
    assert report["scope"]["cropWriteRows"] == 0
    assert report["scope"]["canonicalParsedArtifactWriteRows"] == 0
    assert report["counts"]["privatePathLeakRows"] == 0
    assert report["counts"]["schemaViolationCount"] == 0

    validation = validate_payload(report, VISUAL_LAYOUT_CANDIDATE_LIST_REPORT_SCHEMA_ID, strict=True)
    assert validation.ok, validation.errors


def test_writer_outputs_only_sanitized_refs(tmp_path: Path) -> None:
    report = build_report_from_candidate_rows(
        input_paper_rows=1,
        candidate_rows=_sample_rows(),
        generated_at="2026-05-26T00:00:00Z",
    )
    report_json = tmp_path / "report.json"
    report_md = tmp_path / "report.md"

    write_visual_layout_candidate_list_reports(report, report_json=report_json, report_md=report_md)

    combined = report_json.read_text(encoding="utf-8") + report_md.read_text(encoding="utf-8")
    parsed = json.loads(report_json.read_text(encoding="utf-8"))
    assert parsed["candidateRowsDetail"][0]["paperRef"].startswith("papers_dir/")
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "i" + "Cloud" not in combined
