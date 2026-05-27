from __future__ import annotations

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
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_expansion_human_product_decision_record import (
    DEFAULT_DECISION,
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DECISION_FILE_SCHEMA_ID,
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DECISION_RECORD_SCHEMA_ID,
    build_visual_retrieval_hint_candidate_store_expansion_human_product_decision_record,
    write_visual_retrieval_hint_candidate_store_expansion_human_product_decision_record,
)
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_expansion_review import (
    build_visual_retrieval_hint_candidate_store_expansion_review,
)


def _hash() -> str:
    return "sha256:" + "3" * 64


def _captured_row(candidate_id: str, *, page: int = 1) -> dict[str, object]:
    return {
        "schema": "knowledge-hub.paper.visual-annotation-expansion-captured-row.v1",
        "sourceCandidateId": candidate_id,
        "sourcePackCandidateId": "visual-annotation-expansion-pack:test:3333333333333333",
        "paperId": "sample-paper",
        "paperRef": "papers_dir/sample.pdf",
        "sourceContentHash": _hash(),
        "page": page,
        "bbox": [12.0, 22.0, 202.0, 262.0],
        "candidateType": "image_region",
        "attachmentRef": "eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/01.png",
        "visualObservationStatus": "image_attached",
        "derivedTextForRetrieval": "Retrieval hint only: sample visual image region.",
        "visibleText": "Visible fragments include: sample image labels.",
        "retrievalKeywords": ["sample", "image", "retrieval"],
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
            "bbox": [12.0, 22.0, 202.0, 262.0],
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
        _captured_row("visual-layout:sample-paper:image_region:1:1111111111111111", page=1),
        _captured_row("visual-layout:sample-paper:image_region:2:2222222222222222", page=2),
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


def _review_report() -> dict[str, object]:
    design = build_visual_retrieval_hint_candidate_store_expansion_design(
        _validation_report(),
        generated_at="2026-05-27T00:00:00Z",
    )
    dry_run = build_visual_retrieval_hint_candidate_store_expansion_dry_run(
        design,
        generated_at="2026-05-27T00:00:00Z",
    )
    return build_visual_retrieval_hint_candidate_store_expansion_review(
        dry_run,
        generated_at="2026-05-27T00:00:00Z",
    )


def test_decision_record_defaults_all_rows_to_hold_template() -> None:
    report = build_visual_retrieval_hint_candidate_store_expansion_human_product_decision_record(
        _review_report(),
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["schema"] == VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DECISION_RECORD_SCHEMA_ID
    assert report["status"] == "decision_record_template_ready"
    assert report["decision"] == "manual_human_product_decisions_required"
    assert report["nextRecommendedTranche"] == (
        "manual_edit_visual_retrieval_hint_candidate_store_expansion_decision_record"
    )
    assert report["counts"]["sourceReviewRows"] == 2
    assert report["counts"]["decisionRows"] == 2
    assert report["counts"]["defaultHoldRows"] == 2
    assert report["counts"]["humanDecisionRows"] == 0
    assert report["counts"]["approvedRows"] == 0
    assert report["counts"]["applyDesignCandidateRows"] == 0
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert report["counts"]["blockedRows"] == 0
    assert report["scope"]["writes"] == "report_only"
    assert report["decisionPolicy"]["applyAllowedByThisReport"] is False
    assert report["decisionFileTemplate"]["schema"] == (
        VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DECISION_FILE_SCHEMA_ID
    )

    row = report["decisionRowsDetail"][0]
    assert row["decision"] == DEFAULT_DECISION
    assert row["decisionStatus"] == "default_hold_pending_manual_review"
    assert row["acceptedAsHumanDecision"] is False
    assert row["applyDesignCandidate"] is False
    assert row["candidateStoreWrite"] is False
    assert row["policySummary"]["strictEvidence"] is False
    assert row["policySummary"]["runtimeVisible"] is False
    assert row["policySummary"]["indexEligible"] is False

    validation = validate_payload(
        report,
        VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DECISION_RECORD_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors


def test_decision_record_accepts_explicit_human_approval_as_apply_design_candidate_only() -> None:
    review = _review_report()
    hint_id = review["reviewRowsDetail"][0]["hintCandidateId"]
    decision_file = {
        "schema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DECISION_FILE_SCHEMA_ID,
        "decisions": [
            {
                "hintCandidateId": hint_id,
                "decision": "approve_store_candidate_only",
                "reviewer": "product-reviewer",
                "notes": "Accept as quarantined retrieval hint only.",
            }
        ],
    }

    report = build_visual_retrieval_hint_candidate_store_expansion_human_product_decision_record(
        review,
        decision_file=decision_file,
        decision_file_ref="eval/knowledgeos/reports/manual_decisions.json",
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "decision_record_validated"
    assert report["decision"] == "ready_for_visual_retrieval_hint_candidate_store_expansion_apply_design"
    assert report["counts"]["humanDecisionRows"] == 1
    assert report["counts"]["approvedRows"] == 1
    assert report["counts"]["applyDesignCandidateRows"] == 1
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert report["decisionRowsDetail"][0]["acceptedAsHumanDecision"] is True
    assert report["decisionRowsDetail"][0]["applyDesignCandidate"] is True
    assert report["decisionRowsDetail"][0]["candidateStoreWrite"] is False


def test_decision_record_blocks_approval_without_reviewer() -> None:
    review = _review_report()
    hint_id = review["reviewRowsDetail"][0]["hintCandidateId"]
    decision_file = {
        "schema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DECISION_FILE_SCHEMA_ID,
        "decisions": [
            {
                "hintCandidateId": hint_id,
                "decision": "approve_store_candidate_only",
                "reviewer": "",
                "notes": "Missing reviewer.",
            }
        ],
    }

    report = build_visual_retrieval_hint_candidate_store_expansion_human_product_decision_record(
        review,
        decision_file=decision_file,
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["invalidDecisionRows"] == 1
    assert "reviewer_required_for_human_decision" in report["decisionRowsDetail"][0]["blockerReason"]
    assert report["counts"]["candidateStoreWriteRows"] == 0


def test_decision_record_blocks_wrong_review_schema() -> None:
    review = _review_report()
    review["schema"] = "knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-dry-run.v1"

    report = build_visual_retrieval_hint_candidate_store_expansion_human_product_decision_record(
        review,
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["decision"] == "blocked"


def test_decision_record_detects_private_path_in_decision_file() -> None:
    review = _review_report()
    hint_id = review["reviewRowsDetail"][0]["hintCandidateId"]
    decision_file = {
        "schema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DECISION_FILE_SCHEMA_ID,
        "decisions": [
            {
                "hintCandidateId": hint_id,
                "decision": "hold_pending_more_context",
                "reviewer": "product-reviewer",
                "notes": "/" + "Users" + "/won/private.pdf",
            }
        ],
    }

    report = build_visual_retrieval_hint_candidate_store_expansion_human_product_decision_record(
        review,
        decision_file=decision_file,
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] == 1


def test_decision_record_writer_uses_sanitized_refs(tmp_path: Path) -> None:
    report = build_visual_retrieval_hint_candidate_store_expansion_human_product_decision_record(
        _review_report(),
        source_review_report_ref=(
            "eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_review.v1.json"
        ),
        generated_at="2026-05-27T00:00:00Z",
    )
    report_json = tmp_path / "decision-record.json"
    report_md = tmp_path / "decision-record.md"

    write_visual_retrieval_hint_candidate_store_expansion_human_product_decision_record(
        report,
        report_json=report_json,
        report_md=report_md,
    )

    combined = report_json.read_text(encoding="utf-8") + report_md.read_text(encoding="utf-8")
    parsed = json.loads(report_json.read_text(encoding="utf-8"))
    assert parsed["sourceReviewReport"]["reportRef"].startswith("eval/knowledgeos/reports/")
    assert parsed["decisionRowsDetail"][0]["paperRef"] == "papers_dir/sample.pdf"
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "i" + "Cloud" not in combined
