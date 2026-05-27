from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.visual_annotation_manual_output_capture import (
    VISUAL_ANNOTATION_WEB_OUTPUT_VALIDATION_SCHEMA_ID,
)
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_design import (
    PLANNED_STORE_REF,
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DESIGN_SCHEMA_ID,
    build_visual_retrieval_hint_candidate_store_design,
    write_visual_retrieval_hint_candidate_store_design,
)


def _hash() -> str:
    return "sha256:" + "1" * 64


def _captured_row(candidate_id: str) -> dict[str, object]:
    return {
        "schema": "knowledge-hub.paper.visual-annotation-captured-row.v1",
        "sourceCandidateId": candidate_id,
        "paperId": "sample-paper",
        "paperRef": "papers_dir/sample.pdf",
        "sourceContentHash": _hash(),
        "page": 1,
        "bbox": [10.0, 20.0, 200.0, 260.0],
        "candidateType": "figure_caption_region",
        "attachmentRef": "eval/knowledgeos/reports/visual_annotation_attachment_pack_001/assets/01.png",
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
            "sourceValidationReportSchema": VISUAL_ANNOTATION_WEB_OUTPUT_VALIDATION_SCHEMA_ID,
            "sourceCandidateId": candidate_id,
            "sourceContentHash": _hash(),
            "page": 1,
            "bbox": [10.0, 20.0, 200.0, 260.0],
            "extractionMethod": "manual_web_vlm_output_capture_v1",
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
        _captured_row("visual-layout:sample-paper:figure_caption_region:1:1111111111111111"),
        _captured_row("visual-layout:sample-paper:figure_caption_region:2:2222222222222222"),
    ]
    return {
        "schema": VISUAL_ANNOTATION_WEB_OUTPUT_VALIDATION_SCHEMA_ID,
        "status": "ready",
        "counts": {
            "matchedRows": len(rows),
            "blockedRows": 0,
        },
        "capturedRowsDetail": rows,
    }


def test_candidate_store_design_projects_rows_without_index_or_runtime_visibility() -> None:
    report = build_visual_retrieval_hint_candidate_store_design(
        _validation_report(),
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["schema"] == VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DESIGN_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == "ready_for_visual_retrieval_hint_candidate_store_dry_run"
    assert report["nextRecommendedTranche"] == "visual_retrieval_hint_candidate_store_dry_run"
    assert report["counts"]["candidateRows"] == 2
    assert report["counts"]["indexEligibleRows"] == 0
    assert report["counts"]["runtimeVisibleRows"] == 0
    assert report["counts"]["strictEvidenceRows"] == 0
    assert report["counts"]["citationGradeRows"] == 0
    assert report["counts"]["privatePathLeakRows"] == 0
    assert report["scope"]["writes"] == "report_only"
    assert report["scope"]["candidateStoreMutationRows"] == 0
    assert report["scope"]["vectorIndexing"] is False
    assert report["scope"]["databaseMutationRows"] == 0
    assert report["scope"]["indexMutationRows"] == 0
    assert report["scope"]["reindexOrReembedRows"] == 0
    assert report["scope"]["vaultScanRows"] == 0
    assert report["scope"]["externalDownloadRows"] == 0
    assert report["scope"]["answerabilityGateBypassRows"] == 0

    row = report["candidateRowsDetail"][0]
    assert row["storeProjection"]["plannedStoreRef"] == PLANNED_STORE_REF
    assert row["storeProjection"]["writeStatus"] == "not_written_design_only"
    assert row["policy"]["allowedUse"] == "retrieval_hint_only"
    assert row["policy"]["strictEvidence"] is False
    assert row["policy"]["citationGrade"] is False
    assert row["policy"]["runtimeVisible"] is False
    assert row["policy"]["indexEligible"] is False
    assert report["fullImageEscalationPolicy"]["currentTrancheSendsWholeImagesToGpt"] is False

    validation = validate_payload(
        report,
        VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DESIGN_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors


def test_candidate_store_design_blocks_when_source_validation_is_blocked() -> None:
    source = _validation_report()
    source["status"] = "blocked"
    source["counts"]["blockedRows"] = 1

    report = build_visual_retrieval_hint_candidate_store_design(
        source,
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["decision"] == "blocked"
    assert report["counts"]["blockedRows"] == 1


def test_candidate_store_design_detects_private_path_leak() -> None:
    source = _validation_report()
    source["capturedRowsDetail"][0]["derivedTextForRetrieval"] = (
        "Retrieval hint only: /" + "Users" + "/won/private.pdf"
    )

    report = build_visual_retrieval_hint_candidate_store_design(
        source,
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] == 1


def test_candidate_store_design_writer_uses_sanitized_refs(tmp_path: Path) -> None:
    report = build_visual_retrieval_hint_candidate_store_design(
        _validation_report(),
        source_validation_report_ref="eval/knowledgeos/reports/visual_annotation_web_output_001.validation.v1.json",
        generated_at="2026-05-26T00:00:00Z",
    )
    report_json = tmp_path / "report.json"
    report_md = tmp_path / "report.md"

    write_visual_retrieval_hint_candidate_store_design(
        report,
        report_json=report_json,
        report_md=report_md,
    )

    combined = report_json.read_text(encoding="utf-8") + report_md.read_text(encoding="utf-8")
    parsed = json.loads(report_json.read_text(encoding="utf-8"))
    assert parsed["sourceValidationReport"]["reportRef"].startswith("eval/knowledgeos/reports/")
    assert parsed["candidateRowsDetail"][0]["paperRef"] == "papers_dir/sample.pdf"
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "i" + "Cloud" not in combined
