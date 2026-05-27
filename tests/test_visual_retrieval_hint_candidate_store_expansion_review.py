from __future__ import annotations

import copy
import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.visual_annotation_expansion_manual_output_capture import (
    VISUAL_ANNOTATION_EXPANSION_WEB_OUTPUT_VALIDATION_SCHEMA_ID,
)
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_expansion_design import (
    build_visual_retrieval_hint_candidate_store_expansion_design,
)
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_expansion_dry_run import (
    build_visual_retrieval_hint_candidate_store_expansion_dry_run,
)
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_expansion_review import (
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_REVIEW_SCHEMA_ID,
    build_visual_retrieval_hint_candidate_store_expansion_review,
    write_visual_retrieval_hint_candidate_store_expansion_review,
)


def _hash() -> str:
    return "sha256:" + "2" * 64


def _captured_row(candidate_id: str, *, page: int = 1) -> dict[str, object]:
    return {
        "schema": "knowledge-hub.paper.visual-annotation-expansion-captured-row.v1",
        "sourceCandidateId": candidate_id,
        "sourcePackCandidateId": "visual-annotation-expansion-pack:test:2222222222222222",
        "paperId": "sample-paper",
        "paperRef": "papers_dir/sample.pdf",
        "sourceContentHash": _hash(),
        "page": page,
        "bbox": [11.0, 21.0, 201.0, 261.0],
        "candidateType": "table_region",
        "attachmentRef": "eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/01.png",
        "visualObservationStatus": "image_attached",
        "derivedTextForRetrieval": "Retrieval hint only: sample visual table region.",
        "visibleText": "Visible fragments include: Table 1 and sample labels.",
        "retrievalKeywords": ["sample", "table", "retrieval"],
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
            "bbox": [11.0, 21.0, 201.0, 261.0],
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
        _captured_row("visual-layout:sample-paper:table_region:1:1111111111111111", page=1),
        _captured_row("visual-layout:sample-paper:table_region:2:2222222222222222", page=2),
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


def _dry_run_report() -> dict[str, object]:
    design = build_visual_retrieval_hint_candidate_store_expansion_design(
        _validation_report(),
        generated_at="2026-05-27T00:00:00Z",
    )
    return build_visual_retrieval_hint_candidate_store_expansion_dry_run(
        design,
        generated_at="2026-05-27T00:00:00Z",
    )


def test_expansion_review_marks_dry_run_rows_ready_for_human_product_review() -> None:
    report = build_visual_retrieval_hint_candidate_store_expansion_review(
        _dry_run_report(),
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["schema"] == VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_REVIEW_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == (
        "ready_for_visual_retrieval_hint_candidate_store_expansion_human_product_review"
    )
    assert report["nextRecommendedTranche"] == (
        "visual_retrieval_hint_candidate_store_expansion_human_product_decision_record"
    )
    assert report["counts"]["sourceDryRunRows"] == 2
    assert report["counts"]["reviewRows"] == 2
    assert report["counts"]["reviewReadyRows"] == 2
    assert report["counts"]["humanDecisionRows"] == 0
    assert report["counts"]["applyReadyRows"] == 0
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert report["counts"]["indexEligibleRows"] == 0
    assert report["counts"]["runtimeVisibleRows"] == 0
    assert report["counts"]["strictEvidenceRows"] == 0
    assert report["counts"]["citationGradeRows"] == 0
    assert report["counts"]["privatePathLeakRows"] == 0
    assert report["scope"]["writes"] == "report_only"
    assert report["reviewPolicy"]["actualApproval"] is False
    assert report["reviewPolicy"]["applyAllowedByThisReport"] is False
    assert report["reviewPolicy"]["requiresHumanProductDecisionRecord"] is True

    row = report["reviewRowsDetail"][0]
    assert row["reviewStatus"] == "ready_for_human_product_review"
    assert row["defaultHumanDecision"] == "hold_pending_human_product_review"
    assert row["policySummary"]["allowedUse"] == "retrieval_hint_only"
    assert row["policySummary"]["strictEvidence"] is False
    assert row["policySummary"]["runtimeVisible"] is False
    assert row["policySummary"]["indexEligible"] is False
    assert row["checks"]["wouldWriteOnlyOnFutureApply"] is True
    assert row["checks"]["noPrivatePathLeak"] is True

    validation = validate_payload(
        report,
        VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_REVIEW_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors


def test_expansion_review_blocks_wrong_source_schema() -> None:
    source = _dry_run_report()
    source["schema"] = "knowledge-hub.paper.visual-retrieval-hint-candidate-store-dry-run.v1"

    report = build_visual_retrieval_hint_candidate_store_expansion_review(
        source,
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["decision"] == "blocked"


def test_expansion_review_blocks_candidate_store_write_signal() -> None:
    source = _dry_run_report()
    source["counts"] = copy.deepcopy(source["counts"])
    source["counts"]["candidateStoreWriteRows"] = 1

    report = build_visual_retrieval_hint_candidate_store_expansion_review(
        source,
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["candidateStoreWriteRows"] == 0


def test_expansion_review_blocks_non_quarantined_preview_policy() -> None:
    source = _dry_run_report()
    row = source["dryRunRowsDetail"][0]
    row["plannedJsonlRecordPreview"]["policy"] = copy.deepcopy(
        row["plannedJsonlRecordPreview"]["policy"]
    )
    row["plannedJsonlRecordPreview"]["policy"]["runtimeVisible"] = True

    report = build_visual_retrieval_hint_candidate_store_expansion_review(
        source,
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["reviewRowsDetail"][0]["reviewStatus"] == "blocked"
    assert "runtimeVisibleDisabled" in report["reviewRowsDetail"][0]["blockerReason"]


def test_expansion_review_detects_private_path_leaks() -> None:
    source = _dry_run_report()
    source["dryRunRowsDetail"][0]["plannedJsonlRecordPreview"]["derivedTextForRetrieval"] = (
        "Retrieval hint only: /" + "Users" + "/won/private.pdf"
    )

    report = build_visual_retrieval_hint_candidate_store_expansion_review(
        source,
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] == 1


def test_expansion_review_writer_uses_sanitized_refs(tmp_path: Path) -> None:
    report = build_visual_retrieval_hint_candidate_store_expansion_review(
        _dry_run_report(),
        source_dry_run_report_ref=(
            "eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_dry_run.v1.json"
        ),
        generated_at="2026-05-27T00:00:00Z",
    )
    report_json = tmp_path / "review.json"
    report_md = tmp_path / "review.md"

    write_visual_retrieval_hint_candidate_store_expansion_review(
        report,
        report_json=report_json,
        report_md=report_md,
    )

    combined = report_json.read_text(encoding="utf-8") + report_md.read_text(encoding="utf-8")
    parsed = json.loads(report_json.read_text(encoding="utf-8"))
    assert parsed["sourceDryRunReport"]["reportRef"].startswith("eval/knowledgeos/reports/")
    assert parsed["reviewRowsDetail"][0]["paperRef"] == "papers_dir/sample.pdf"
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "i" + "Cloud" not in combined
