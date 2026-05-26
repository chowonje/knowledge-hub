from __future__ import annotations

import copy
import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.visual_annotation_expansion_manual_output_capture import (
    VISUAL_ANNOTATION_EXPANSION_WEB_OUTPUT_VALIDATION_SCHEMA_ID,
)
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_expansion_design import (
    PLANNED_STORE_REF,
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DESIGN_SCHEMA_ID,
    build_visual_retrieval_hint_candidate_store_expansion_design,
    write_visual_retrieval_hint_candidate_store_expansion_design,
)
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_expansion_dry_run import (
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DRY_RUN_SCHEMA_ID,
    build_visual_retrieval_hint_candidate_store_expansion_dry_run,
    write_visual_retrieval_hint_candidate_store_expansion_dry_run,
)


def _hash() -> str:
    return "sha256:" + "1" * 64


def _captured_row(candidate_id: str, *, page: int = 1) -> dict[str, object]:
    return {
        "schema": "knowledge-hub.paper.visual-annotation-expansion-captured-row.v1",
        "sourceCandidateId": candidate_id,
        "sourcePackCandidateId": "visual-annotation-expansion-pack:test:1111111111111111",
        "paperId": "sample-paper",
        "paperRef": "papers_dir/sample.pdf",
        "sourceContentHash": _hash(),
        "page": page,
        "bbox": [10.0, 20.0, 200.0, 260.0],
        "candidateType": "figure_caption_region",
        "attachmentRef": "eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/01.png",
        "visualObservationStatus": "image_attached",
        "derivedTextForRetrieval": "Retrieval hint only: sample visual figure region.",
        "visibleText": "Visible fragments include: Figure 1 and sample labels.",
        "retrievalKeywords": ["sample", "figure", "retrieval"],
        "uncertainty": "Low.",
        "limitations": "Retrieval hint only, not evidence.",
        "retrievalHintPlan": {
            "targetDerivedTextField": "derivedTextForRetrieval",
            "allowedUse": "retrieval_hint_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
        },
        "provenance": {
            "sourceExpansionPackSchema": "knowledge-hub.paper.visual-annotation-expansion-pack-design.v1",
            "sourceAttachmentPackSchema": "knowledge-hub.paper.visual-annotation-expansion-attachment-pack.v1",
            "sourceCandidateId": candidate_id,
            "sourceContentHash": _hash(),
            "page": page,
            "bbox": [10.0, 20.0, 200.0, 260.0],
            "extractionMethod": "manual_web_vlm_expansion_output_capture_v1",
        },
        "validation": {
            "matchedSourcePackRow": True,
            "matchedAttachmentRow": True,
            "policyCompliant": True,
            "violationReasons": [],
        },
    }


def _validation_report() -> dict[str, object]:
    rows = [
        _captured_row("visual-layout:sample-paper:figure_caption_region:1:1111111111111111", page=1),
        _captured_row("visual-layout:sample-paper:figure_caption_region:2:2222222222222222", page=2),
    ]
    return {
        "schema": VISUAL_ANNOTATION_EXPANSION_WEB_OUTPUT_VALIDATION_SCHEMA_ID,
        "status": "ready",
        "counts": {
            "matchedRows": len(rows),
            "blockedRows": 0,
        },
        "capturedRowsDetail": rows,
    }


def test_expansion_design_projects_rows_without_index_or_runtime_visibility() -> None:
    report = build_visual_retrieval_hint_candidate_store_expansion_design(
        _validation_report(),
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["schema"] == VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DESIGN_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == "ready_for_visual_retrieval_hint_candidate_store_expansion_dry_run"
    assert report["nextRecommendedTranche"] == "visual_retrieval_hint_candidate_store_expansion_dry_run"
    assert report["counts"]["sourceValidationRows"] == 2
    assert report["counts"]["candidateRows"] == 2
    assert report["counts"]["eligibleRows"] == 2
    assert report["counts"]["blockedRows"] == 0
    assert report["counts"]["indexEligibleRows"] == 0
    assert report["counts"]["runtimeVisibleRows"] == 0
    assert report["counts"]["strictEvidenceRows"] == 0
    assert report["counts"]["citationGradeRows"] == 0
    assert report["counts"]["privatePathLeakRows"] == 0
    assert report["scope"]["writes"] == "report_only"
    assert report["scope"]["candidateStoreMutationRows"] == 0
    assert report["scope"]["vectorIndexing"] is False

    row = report["candidateRowsDetail"][0]
    assert row["storeProjection"]["plannedStoreRef"] == PLANNED_STORE_REF
    assert row["storeProjection"]["writeStatus"] == "not_written_design_only"
    assert row["policy"]["allowedUse"] == "retrieval_hint_only"
    assert row["policy"]["strictEvidence"] is False
    assert row["policy"]["citationGrade"] is False
    assert row["policy"]["runtimeVisible"] is False
    assert row["policy"]["indexEligible"] is False
    assert row["provenance"]["sourceValidationReportSchema"] == (
        VISUAL_ANNOTATION_EXPANSION_WEB_OUTPUT_VALIDATION_SCHEMA_ID
    )
    assert report["fullImageEscalationPolicy"]["currentTrancheSendsWholeImagesToGpt"] is False

    validation = validate_payload(
        report,
        VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DESIGN_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors


def test_expansion_dry_run_previews_jsonl_records_without_store_writes() -> None:
    design = build_visual_retrieval_hint_candidate_store_expansion_design(
        _validation_report(),
        generated_at="2026-05-27T00:00:00Z",
    )

    report = build_visual_retrieval_hint_candidate_store_expansion_dry_run(
        design,
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["schema"] == VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DRY_RUN_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == "ready_for_visual_retrieval_hint_candidate_store_expansion_review"
    assert report["nextRecommendedTranche"] == "visual_retrieval_hint_candidate_store_expansion_review"
    assert report["counts"]["sourceDesignRows"] == 2
    assert report["counts"]["dryRunRows"] == 2
    assert report["counts"]["plannedWriteRows"] == 2
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert report["counts"]["jsonlSerializableRows"] == 2
    assert report["counts"]["indexEligibleRows"] == 0
    assert report["counts"]["runtimeVisibleRows"] == 0
    assert report["counts"]["strictEvidenceRows"] == 0
    assert report["counts"]["citationGradeRows"] == 0
    assert report["counts"]["privatePathLeakRows"] == 0
    assert report["scope"]["writes"] == "report_only"
    assert report["scope"]["candidateStoreWriteRows"] == 0
    assert report["scope"]["vectorIndexing"] is False

    row = report["dryRunRowsDetail"][0]
    assert row["plannedStoreRef"] == PLANNED_STORE_REF
    assert row["plannedJsonlRecordSha256"].startswith("sha256:")
    assert row["dryRunResult"]["wouldWriteOnApply"] is True
    assert row["dryRunResult"]["actualStoreWrite"] is False
    assert row["dryRunResult"]["indexEligible"] is False
    assert row["dryRunResult"]["runtimeVisible"] is False
    assert row["plannedJsonlRecordPreview"]["policy"]["allowedUse"] == "retrieval_hint_only"

    validation = validate_payload(
        report,
        VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DRY_RUN_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors


def test_expansion_design_blocks_wrong_source_schema() -> None:
    source = _validation_report()
    source["schema"] = "knowledge-hub.paper.visual-annotation-web-output-validation.v1"

    report = build_visual_retrieval_hint_candidate_store_expansion_design(
        source,
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["decision"] == "blocked"
    assert report["counts"]["blockedRows"] >= 1


def test_expansion_dry_run_blocks_non_quarantined_policy() -> None:
    design = build_visual_retrieval_hint_candidate_store_expansion_design(
        _validation_report(),
        generated_at="2026-05-27T00:00:00Z",
    )
    design["candidateRowsDetail"][0]["policy"] = copy.deepcopy(
        design["candidateRowsDetail"][0]["policy"]
    )
    design["candidateRowsDetail"][0]["policy"]["indexEligible"] = True

    report = build_visual_retrieval_hint_candidate_store_expansion_dry_run(
        design,
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["dryRunRowsDetail"][0]["blockerReason"] == "policy_not_quarantined"
    assert report["counts"]["candidateStoreWriteRows"] == 0


def test_expansion_reports_detect_private_path_leaks() -> None:
    source = _validation_report()
    source["capturedRowsDetail"][0]["derivedTextForRetrieval"] = (
        "Retrieval hint only: /" + "Users" + "/won/private.pdf"
    )

    design = build_visual_retrieval_hint_candidate_store_expansion_design(
        source,
        generated_at="2026-05-27T00:00:00Z",
    )

    assert design["status"] == "blocked"
    assert design["counts"]["privatePathLeakRows"] == 1


def test_expansion_writers_use_sanitized_refs(tmp_path: Path) -> None:
    design = build_visual_retrieval_hint_candidate_store_expansion_design(
        _validation_report(),
        source_validation_report_ref=(
            "eval/knowledgeos/reports/visual_annotation_expansion_web_output_002.validation.v1.json"
        ),
        generated_at="2026-05-27T00:00:00Z",
    )
    dry_run = build_visual_retrieval_hint_candidate_store_expansion_dry_run(
        design,
        source_design_report_ref=(
            "eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_design.v1.json"
        ),
        generated_at="2026-05-27T00:00:00Z",
    )
    design_json = tmp_path / "design.json"
    design_md = tmp_path / "design.md"
    dry_run_json = tmp_path / "dry-run.json"
    dry_run_md = tmp_path / "dry-run.md"

    write_visual_retrieval_hint_candidate_store_expansion_design(
        design,
        report_json=design_json,
        report_md=design_md,
    )
    write_visual_retrieval_hint_candidate_store_expansion_dry_run(
        dry_run,
        report_json=dry_run_json,
        report_md=dry_run_md,
    )

    combined = (
        design_json.read_text(encoding="utf-8")
        + design_md.read_text(encoding="utf-8")
        + dry_run_json.read_text(encoding="utf-8")
        + dry_run_md.read_text(encoding="utf-8")
    )
    parsed = json.loads(dry_run_json.read_text(encoding="utf-8"))
    assert parsed["sourceDesignReport"]["reportRef"].startswith("eval/knowledgeos/reports/")
    assert parsed["dryRunRowsDetail"][0]["paperRef"] == "papers_dir/sample.pdf"
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "i" + "Cloud" not in combined
