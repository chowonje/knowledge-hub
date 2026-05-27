from __future__ import annotations

import copy
import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_design import (
    PLANNED_STORE_REF,
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DESIGN_SCHEMA_ID,
)
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_dry_run import (
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DRY_RUN_SCHEMA_ID,
    build_visual_retrieval_hint_candidate_store_dry_run,
    write_visual_retrieval_hint_candidate_store_dry_run,
)


def _hash() -> str:
    return "sha256:" + "1" * 64


def _candidate_row(candidate_id: str, hint_id: str) -> dict[str, object]:
    return {
        "schema": "knowledge-hub.paper.visual-retrieval-hint-candidate-row.v1",
        "hintCandidateId": hint_id,
        "sourceCandidateId": candidate_id,
        "paperId": "sample-paper",
        "paperRef": "papers_dir/sample.pdf",
        "sourceContentHash": _hash(),
        "page": 1,
        "bbox": [10.0, 20.0, 200.0, 260.0],
        "candidateType": "figure_caption_region",
        "sourceAttachmentRef": "eval/knowledgeos/reports/visual_annotation_attachment_pack_001/assets/01.png",
        "derivedTextForRetrieval": "Retrieval hint only: sample visual figure region.",
        "visibleText": "Visible fragments include: Figure 1 and sample labels.",
        "retrievalKeywords": ["sample", "figure", "retrieval"],
        "uncertainty": "Low.",
        "limitations": "Retrieval hint only, not evidence.",
        "policy": {
            "allowedUse": "retrieval_hint_only",
            "targetDerivedTextField": "derivedTextForRetrieval",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": False,
            "indexEligible": False,
            "requiresFutureIndexEligibilityGate": True,
            "answerabilityGateBypassAllowed": False,
        },
        "storeProjection": {
            "plannedStoreRef": PLANNED_STORE_REF,
            "writeStatus": "not_written_design_only",
            "indexStatus": "not_indexed",
            "runtimeVisibility": "not_runtime_visible",
        },
        "provenance": {
            "sourceValidationReportSchema": "knowledge-hub.paper.visual-annotation-web-output-validation.v1",
            "sourceValidationReportRef": "eval/knowledgeos/reports/visual_annotation_web_output_001.validation.v1.json",
            "sourceCandidateId": candidate_id,
            "sourceContentHash": _hash(),
            "page": 1,
            "bbox": [10.0, 20.0, 200.0, 260.0],
            "extractionMethod": "visual_annotation_manual_output_capture_to_candidate_store_design_v1",
        },
        "blockerReason": "",
    }


def _design_report() -> dict[str, object]:
    rows = [
        _candidate_row(
            "visual-layout:sample-paper:figure_caption_region:1:1111111111111111",
            "visual-retrieval-hint:sample-paper:figure_caption_region:1:1111111111111111",
        ),
        _candidate_row(
            "visual-layout:sample-paper:figure_caption_region:2:2222222222222222",
            "visual-retrieval-hint:sample-paper:figure_caption_region:2:2222222222222222",
        ),
    ]
    return {
        "schema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DESIGN_SCHEMA_ID,
        "status": "ready",
        "counts": {
            "candidateRows": len(rows),
            "blockedRows": 0,
        },
        "candidateRowsDetail": rows,
    }


def test_candidate_store_dry_run_previews_jsonl_records_without_writes() -> None:
    report = build_visual_retrieval_hint_candidate_store_dry_run(
        _design_report(),
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["schema"] == VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DRY_RUN_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == "ready_for_visual_annotation_expansion_pack_design"
    assert report["nextRecommendedTranche"] == "visual_annotation_expansion_pack_design"
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
    assert report["scope"]["databaseMutationRows"] == 0
    assert report["scope"]["indexMutationRows"] == 0
    assert report["scope"]["reindexOrReembedRows"] == 0
    assert report["scope"]["vaultScanRows"] == 0
    assert report["scope"]["externalDownloadRows"] == 0
    assert report["scope"]["answerabilityGateBypassRows"] == 0

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
        VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DRY_RUN_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors


def test_candidate_store_dry_run_blocks_duplicate_hint_ids() -> None:
    source = _design_report()
    source["candidateRowsDetail"][1]["hintCandidateId"] = source["candidateRowsDetail"][0]["hintCandidateId"]

    report = build_visual_retrieval_hint_candidate_store_dry_run(
        source,
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["duplicateHintCandidateIdRows"] == 1
    assert report["counts"]["blockedRows"] >= 1
    validation = validate_payload(
        report,
        VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DRY_RUN_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors


def test_candidate_store_dry_run_blocks_non_quarantined_policy() -> None:
    source = _design_report()
    source["candidateRowsDetail"][0]["policy"] = copy.deepcopy(source["candidateRowsDetail"][0]["policy"])
    source["candidateRowsDetail"][0]["policy"]["indexEligible"] = True

    report = build_visual_retrieval_hint_candidate_store_dry_run(
        source,
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["dryRunRowsDetail"][0]["blockerReason"] == "policy_not_quarantined"
    assert report["counts"]["candidateStoreWriteRows"] == 0


def test_candidate_store_dry_run_detects_private_path_leak() -> None:
    source = _design_report()
    source["candidateRowsDetail"][0]["derivedTextForRetrieval"] = (
        "Retrieval hint only: /" + "Users" + "/won/private.pdf"
    )

    report = build_visual_retrieval_hint_candidate_store_dry_run(
        source,
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] == 1
    assert "private_path_leak" in report["dryRunRowsDetail"][0]["blockerReason"]


def test_candidate_store_dry_run_writer_uses_sanitized_refs(tmp_path: Path) -> None:
    report = build_visual_retrieval_hint_candidate_store_dry_run(
        _design_report(),
        source_design_report_ref="eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_design.v1.json",
        generated_at="2026-05-26T00:00:00Z",
    )
    report_json = tmp_path / "report.json"
    report_md = tmp_path / "report.md"

    write_visual_retrieval_hint_candidate_store_dry_run(
        report,
        report_json=report_json,
        report_md=report_md,
    )

    combined = report_json.read_text(encoding="utf-8") + report_md.read_text(encoding="utf-8")
    parsed = json.loads(report_json.read_text(encoding="utf-8"))
    assert parsed["sourceDesignReport"]["reportRef"].startswith("eval/knowledgeos/reports/")
    assert parsed["dryRunRowsDetail"][0]["paperRef"] == "papers_dir/sample.pdf"
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "i" + "Cloud" not in combined
