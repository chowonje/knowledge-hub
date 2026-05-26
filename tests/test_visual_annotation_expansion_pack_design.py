from __future__ import annotations

import copy
import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.visual_annotation_expansion_pack_design import (
    VISUAL_ANNOTATION_EXPANSION_PACK_DESIGN_SCHEMA_ID,
    build_visual_annotation_expansion_pack_design,
    select_expansion_candidates,
    write_visual_annotation_expansion_pack_design,
)
from knowledge_hub.papers.visual_annotation_web_pack import VISUAL_ANNOTATION_WEB_PACK_SCHEMA_ID
from knowledge_hub.papers.visual_layout_candidate_list_report import (
    build_report_from_candidate_rows,
    extract_candidates_from_blocks,
    extract_candidates_from_images,
)
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_dry_run import (
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DRY_RUN_SCHEMA_ID,
)


def _hash(seed: str = "1") -> str:
    return "sha256:" + seed * 64


def _block_rows(paper_id: str, *, figure_rows: int, table_rows: int, equation_rows: int) -> list[dict[str, object]]:
    blocks_by_page: list[tuple[int, list[tuple[float, float, float, float, str, int, int]]]] = []
    for index in range(max(figure_rows, table_rows, equation_rows)):
        page = index + 1
        blocks: list[tuple[float, float, float, float, str, int, int]] = []
        if index < figure_rows:
            blocks.append(
                (
                    10.0,
                    20.0,
                    260.0,
                    42.0,
                    f"Figure {index + 1}: {paper_id} visual comparison panel.",
                    0,
                    0,
                )
            )
        if index < table_rows:
            blocks.append(
                (
                    10.0,
                    60.0,
                    260.0,
                    86.0,
                    f"Table {index + 1}: {paper_id} benchmark summary.",
                    1,
                    0,
                )
            )
        if index < equation_rows:
            blocks.append(
                (
                    10.0,
                    104.0,
                    260.0,
                    130.0,
                    f"z = W{index + 1} x + b{index + 1}",
                    2,
                    0,
                )
            )
        blocks_by_page.append((page, blocks))
    return extract_candidates_from_blocks(
        paper_id=paper_id,
        paper_ref=f"papers_dir/{paper_id}.pdf",
        source_content_hash=_hash(),
        blocks_by_page=blocks_by_page,
    )


def _image_rows(paper_id: str, *, page_count: int, images_per_page: int) -> list[dict[str, object]]:
    return extract_candidates_from_images(
        paper_id=paper_id,
        paper_ref=f"papers_dir/{paper_id}.pdf",
        source_content_hash=_hash(),
        image_refs_by_page=[
            (
                page,
                [
                    {
                        "bbox": [20 + (slot * 30), 160, 42 + (slot * 30), 198],
                        "imageHash": _hash(str((page + slot) % 9 or 9)),
                    }
                    for slot in range(images_per_page)
                ],
            )
            for page in range(1, page_count + 1)
        ],
    )


def _candidate_report() -> dict[str, object]:
    rows = [
        *_image_rows("clip-2021", page_count=4, images_per_page=3),
        *_block_rows("clip-2021", figure_rows=6, table_rows=3, equation_rows=2),
        *_block_rows("mae-2021", figure_rows=4, table_rows=3, equation_rows=2),
        *_image_rows("mae-2021", page_count=2, images_per_page=2),
    ]
    return build_report_from_candidate_rows(
        input_paper_rows=2,
        candidate_rows=rows,
        generated_at="2026-05-26T00:00:00Z",
    )


def _web_pack(previous_ids: list[str] | None = None) -> dict[str, object]:
    return {
        "schema": VISUAL_ANNOTATION_WEB_PACK_SCHEMA_ID,
        "status": "ready",
        "packRowsDetail": [{"sourceCandidateId": candidate_id} for candidate_id in list(previous_ids or [])],
    }


def _dry_run(previous_ids: list[str] | None = None) -> dict[str, object]:
    return {
        "schema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DRY_RUN_SCHEMA_ID,
        "status": "ready",
        "dryRunRowsDetail": [{"sourceCandidateId": candidate_id} for candidate_id in list(previous_ids or [])],
    }


def test_select_expansion_candidates_excludes_previous_rows_and_caps_per_page() -> None:
    source = _candidate_report()
    rows = list(source["candidateRowsDetail"])
    previous_ids = {str(rows[0]["candidateId"]), str(rows[1]["candidateId"])}

    selected = select_expansion_candidates(
        rows,
        previous_candidate_ids=previous_ids,
        max_candidates=4,
        type_quotas={"image_region": 4},
        preferred_paper_ids=("clip-2021", "mae-2021"),
        max_per_paper_type_page=2,
    )

    assert len(selected) == 4
    assert not previous_ids.intersection(str(row["candidateId"]) for row in selected)
    assert {row["candidateType"] for row in selected} == {"image_region"}
    per_page = {}
    for row in selected:
        key = (row["paperId"], row["candidateType"], row["page"])
        per_page[key] = per_page.get(key, 0) + 1
    assert max(per_page.values()) <= 2


def test_build_expansion_pack_validates_and_preserves_report_only_policy() -> None:
    source = _candidate_report()
    previous_ids = [
        source["candidateRowsDetail"][0]["candidateId"],
        source["candidateRowsDetail"][1]["candidateId"],
    ]

    report = build_visual_annotation_expansion_pack_design(
        source,
        _web_pack([previous_ids[0]]),
        _dry_run([previous_ids[1]]),
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["schema"] == VISUAL_ANNOTATION_EXPANSION_PACK_DESIGN_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == "ready_for_visual_annotation_expansion_attachment_pack"
    assert report["nextRecommendedTranche"] == "visual_annotation_expansion_attachment_pack"
    assert report["counts"]["selectedExpansionRows"] == 24
    assert report["counts"]["imageCandidateRows"] == 8
    assert report["counts"]["figureCandidateRows"] == 8
    assert report["counts"]["tableCandidateRows"] == 5
    assert report["counts"]["equationCandidateRows"] == 3
    assert report["counts"]["wholeImageRows"] == 0
    assert report["counts"]["pageImageRows"] == 0
    assert report["counts"]["privatePathLeakRows"] == 0
    assert report["scope"]["writes"] == "report_only"
    assert report["scope"]["apiCalls"] is False
    assert report["scope"]["modelCalls"] is False
    assert report["scope"]["webModelCalls"] is False
    assert report["scope"]["wholeImageGptRows"] == 0
    assert report["scope"]["cropWriteRows"] == 0
    assert report["scope"]["candidateStoreMutationRows"] == 0
    assert report["scope"]["vectorIndexing"] is False
    assert report["scope"]["runtimeAnswerVisibleExposureRows"] == 0
    assert report["scope"]["strictEvidencePromotionRows"] == 0
    assert report["scope"]["databaseMutationRows"] == 0
    assert report["scope"]["indexMutationRows"] == 0
    assert report["scope"]["vaultScanRows"] == 0
    assert report["scope"]["externalDownloadRows"] == 0
    assert report["selectionPolicy"]["wholeImagePolicy"] == "deferred_to_visual_full_image_annotation_pack_design"

    row = report["packRowsDetail"][0]
    assert row["webInput"]["attachmentGuidance"]["recommendedAttachmentKind"] == "context_crop_png"
    assert row["webInput"]["attachmentGuidance"]["wholeImageAllowedInThisPack"] is False
    assert row["webInput"]["attachmentGuidance"]["pageImageAllowedInThisPack"] is False
    assert row["visualContext"]["wholeImageAllowedInThisPack"] is False
    assert row["retrievalHintPlan"]["allowedUse"] == "retrieval_hint_only"
    assert row["retrievalHintPlan"]["strictEvidence"] is False
    assert row["expectedOutputContract"]["citationGrade"] is False

    validation = validate_payload(
        report,
        VISUAL_ANNOTATION_EXPANSION_PACK_DESIGN_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors


def test_expansion_pack_blocks_on_non_ready_source_report() -> None:
    source = _candidate_report()
    source["status"] = "blocked"

    report = build_visual_annotation_expansion_pack_design(
        source,
        _web_pack(),
        _dry_run(),
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["decision"] == "blocked"


def test_expansion_pack_detects_private_path_leak() -> None:
    source = copy.deepcopy(_candidate_report())
    for row in source["candidateRowsDetail"]:
        row["textContext"]["nearbyText"] = "Do not leak /" + "Users" + "/won/private.pdf"

    report = build_visual_annotation_expansion_pack_design(
        source,
        _web_pack(),
        _dry_run(),
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] == 1


def test_expansion_pack_writer_uses_sanitized_refs(tmp_path: Path) -> None:
    report = build_visual_annotation_expansion_pack_design(
        _candidate_report(),
        _web_pack(),
        _dry_run(),
        source_candidate_report_ref="eval/knowledgeos/reports/visual_layout_candidate_list_report.v1.json",
        source_web_pack_ref="eval/knowledgeos/reports/visual_annotation_web_pack_001.v1.json",
        source_dry_run_report_ref="eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_dry_run.v1.json",
        generated_at="2026-05-26T00:00:00Z",
    )
    report_json = tmp_path / "report.json"
    report_md = tmp_path / "report.md"

    write_visual_annotation_expansion_pack_design(
        report,
        report_json=report_json,
        report_md=report_md,
    )

    combined = report_json.read_text(encoding="utf-8") + report_md.read_text(encoding="utf-8")
    parsed = json.loads(report_json.read_text(encoding="utf-8"))
    assert parsed["sourceCandidateReport"]["reportRef"].startswith("eval/knowledgeos/reports/")
    assert parsed["sourceWebPack"]["reportRef"].startswith("eval/knowledgeos/reports/")
    assert parsed["sourceDryRunReport"]["reportRef"].startswith("eval/knowledgeos/reports/")
    assert parsed["packRowsDetail"][0]["paperRef"].startswith("papers_dir/")
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "i" + "Cloud" not in combined
