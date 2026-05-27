from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.visual_annotation_web_pack import (
    VISUAL_ANNOTATION_WEB_PACK_SCHEMA_ID,
    build_visual_annotation_web_pack,
    sanitized_report_ref,
    select_web_pack_candidates,
    write_visual_annotation_web_pack,
)
from knowledge_hub.papers.visual_layout_candidate_list_report import (
    VISUAL_LAYOUT_CANDIDATE_LIST_REPORT_SCHEMA_ID,
    build_report_from_candidate_rows,
    extract_candidates_from_blocks,
    extract_candidates_from_images,
)


def _source_hash() -> str:
    return "sha256:" + "1" * 64


def _source_report() -> dict[str, object]:
    alexnet_rows = extract_candidates_from_blocks(
        paper_id="alexnet-2012",
        paper_ref="papers_dir/alexnet.pdf",
        source_content_hash=_source_hash(),
        blocks_by_page=[
            (
                2,
                [
                    (10, 10, 240, 30, "1 Introduction", 0, 0),
                    (10, 40, 240, 80, "Figure 1: AlexNet architecture overview.", 1, 0),
                    (
                        10,
                        90,
                        260,
                        150,
                        "model  top1  top5\nbaseline  63.3  84.6\nours  66.5  87.2",
                        2,
                        0,
                    ),
                    (10, 160, 260, 190, "y = Wx + b", 3, 0),
                ],
            )
        ],
    )
    resnet_rows = extract_candidates_from_blocks(
        paper_id="resnet-2015",
        paper_ref="papers_dir/resnet.pdf",
        source_content_hash=_source_hash(),
        blocks_by_page=[
            (
                3,
                [
                    (10, 10, 240, 40, "Figure 2: Residual block.", 0, 0),
                    (
                        10,
                        50,
                        260,
                        110,
                        "depth  error  params\n18  30.24  11.7\n34  28.54  21.8",
                        1,
                        0,
                    ),
                    (10, 120, 240, 150, "H(x) = F(x) + x", 2, 0),
                ],
            )
        ],
    )
    image_rows = extract_candidates_from_images(
        paper_id="mae-2021",
        paper_ref="papers_dir/mae.pdf",
        source_content_hash=_source_hash(),
        image_refs_by_page=[
            (
                4,
                [
                    {
                        "bbox": [12, 200, 220, 340],
                        "imageHash": "sha256:" + "2" * 64,
                    }
                ],
            )
        ],
    )
    return build_report_from_candidate_rows(
        input_paper_rows=3,
        candidate_rows=[*alexnet_rows, *resnet_rows, *image_rows],
        generated_at="2026-05-26T00:00:00Z",
    )


def test_select_web_pack_candidates_prefers_small_priority_scope() -> None:
    report = _source_report()
    selected = select_web_pack_candidates(
        report["candidateRowsDetail"],
        max_candidates=4,
        preferred_paper_ids=("alexnet-2012", "resnet-2015"),
    )

    assert len(selected) == 4
    assert [row["paperId"] for row in selected[:3]] == ["alexnet-2012"] * 3
    assert {row["candidateType"] for row in selected}.issubset(
        {"figure_caption_region", "table_region", "equation_region"}
    )
    assert "image_region" not in {row["candidateType"] for row in selected}


def test_build_web_pack_validates_and_preserves_no_mutation_contract() -> None:
    pack = build_visual_annotation_web_pack(
        _source_report(),
        pack_id="visual_annotation_web_pack_test",
        max_candidates=5,
        generated_at="2026-05-26T00:00:00Z",
    )

    assert pack["schema"] == VISUAL_ANNOTATION_WEB_PACK_SCHEMA_ID
    assert pack["status"] == "ready"
    assert pack["decision"] == "ready_for_manual_web_vlm_calibration"
    assert pack["sourceReport"]["schema"] == VISUAL_LAYOUT_CANDIDATE_LIST_REPORT_SCHEMA_ID
    assert pack["counts"]["selectedCandidateRows"] == 5
    assert pack["counts"]["imageCandidateRows"] == 0
    assert pack["counts"]["privatePathLeakRows"] == 0
    assert pack["scope"]["writes"] == "report_only"
    assert pack["scope"]["apiCalls"] is False
    assert pack["scope"]["modelCalls"] is False
    assert pack["scope"]["webModelCalls"] is False
    assert pack["scope"]["vectorIndexing"] is False
    assert pack["scope"]["strictEvidencePromotionRows"] == 0
    assert pack["scope"]["runtimeAnswerVisibleExposureRows"] == 0
    assert pack["scope"]["databaseMutationRows"] == 0
    assert pack["scope"]["indexMutationRows"] == 0
    assert pack["scope"]["reindexOrReembedRows"] == 0
    assert pack["scope"]["vaultScanRows"] == 0
    assert pack["scope"]["externalDownloadRows"] == 0
    assert pack["scope"]["answerabilityGateBypassRows"] == 0
    assert pack["scope"]["cropWriteRows"] == 0

    row = pack["packRowsDetail"][0]
    assert row["retrievalHintPlan"]["allowedUse"] == "retrieval_hint_only"
    assert row["retrievalHintPlan"]["strictEvidence"] is False
    assert row["expectedOutputContract"]["citationGrade"] is False
    assert "derivedTextForRetrieval" in row["expectedOutputContract"]["requiredFields"]
    assert "Do not create citation-grade evidence" in pack["prompt"]["systemPrompt"]

    validation = validate_payload(pack, VISUAL_ANNOTATION_WEB_PACK_SCHEMA_ID, strict=True)
    assert validation.ok, validation.errors


def test_web_pack_blocks_on_non_ready_source_report() -> None:
    source_report = _source_report()
    source_report["status"] = "blocked"

    pack = build_visual_annotation_web_pack(
        source_report,
        max_candidates=2,
        generated_at="2026-05-26T00:00:00Z",
    )

    assert pack["status"] == "blocked"
    assert pack["decision"] == "blocked"


def test_writer_outputs_only_sanitized_refs(tmp_path: Path) -> None:
    pack = build_visual_annotation_web_pack(
        _source_report(),
        max_candidates=3,
        source_report_ref=sanitized_report_ref(
            tmp_path / "visual_layout_candidate_list_report.v1.json",
            project_root=tmp_path,
        ),
        generated_at="2026-05-26T00:00:00Z",
    )
    report_json = tmp_path / "pack.json"
    report_md = tmp_path / "pack.md"

    write_visual_annotation_web_pack(pack, report_json=report_json, report_md=report_md)

    combined = report_json.read_text(encoding="utf-8") + report_md.read_text(encoding="utf-8")
    parsed = json.loads(report_json.read_text(encoding="utf-8"))
    assert parsed["sourceReport"]["reportRef"] == "visual_layout_candidate_list_report.v1.json"
    assert parsed["packRowsDetail"][0]["paperRef"].startswith("papers_dir/")
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "i" + "Cloud" not in combined
