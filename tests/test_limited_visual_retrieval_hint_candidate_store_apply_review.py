from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_apply_review import (
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_REVIEW_SCHEMA_ID,
    build_limited_visual_retrieval_hint_candidate_store_apply_review,
    write_limited_visual_retrieval_hint_candidate_store_apply_review,
)


def _hash() -> str:
    return "sha256:" + "8" * 64


def _apply_design_row(*, unsafe_policy: bool = False) -> dict[str, object]:
    return {
        "schema": "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-apply-design-row.v1",
        "applyDesignRowId": "limited-visual-retrieval-hint-apply-design:fixture",
        "hintCandidateId": "visual-retrieval-hint:sample-paper:figure_caption_region:2:abcdefabcdefabcd",
        "sourceCandidateId": "visual-layout:sample-paper:figure_caption_region:2:aaaaaaaaaaaaaaaa",
        "paperId": "sample-paper",
        "paperRef": "papers_dir/sample.pdf",
        "sourceContentHash": _hash(),
        "page": 2,
        "bbox": [10.0, 20.0, 120.0, 180.0],
        "candidateType": "figure_caption_region",
        "sourceDryRunReportRef": "eval/knowledgeos/reports/dry_run_fixture.v1.json",
        "sourceDryRunRowId": "visual-retrieval-hint-dry-run:fixture",
        "plannedStoreRef": "papers_dir/visual_retrieval_hints/visual_retrieval_hint_candidates.v1.jsonl",
        "idempotencyKey": "visual-retrieval-hint-idempotency:fixture",
        "plannedJsonlRecordSha256": "sha256:" + "9" * 64,
        "recordPreview": {
            "derivedTextForRetrievalPreview": "Retrieval hint only: a ReLU tanh CIFAR-10 training error plot.",
            "visibleTextPreview": "Visible fragments include: ReLU, tanh, CIFAR-10, training error.",
            "retrievalKeywords": ["ReLU", "tanh", "CIFAR-10", "training error"],
            "limitations": "Retrieval hint only, not evidence.",
        },
        "searchEvalSummary": {
            "sourceSearchEvalQueryRowIds": [
                "visual-retrieval-hint-search-query:00000000000000000001",
                "visual-retrieval-hint-search-query:00000000000000000002",
            ],
            "queryRows": 2,
            "textOnlyBestRank": 20,
            "visualHintAugmentedBestRank": 1,
            "augmentedHitAt5Rows": 2,
            "improvedQueryRows": 2,
            "regressedQueryRows": 0,
            "strongLiftQueryRows": 1,
            "blockedQueryRows": 0,
            "passesLimitedApplySearchGate": True,
        },
        "applyPlan": {
            "limitedApplyDesignCandidate": True,
            "wouldWriteOnSeparateExplicitApply": True,
            "applyAllowedByThisReport": False,
            "candidateStoreWrite": False,
            "requiresSeparateApplyExecutor": True,
            "requiresSeparateIndexingGate": True,
            "writeModeIfLaterApproved": "append_jsonl_to_visual_retrieval_hint_candidate_store",
        },
        "policy": {
            "allowedUse": "retrieval_hint_only",
            "strictEvidence": unsafe_policy,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": False,
            "indexEligible": False,
        },
        "blockerReason": "",
    }


def _apply_design_report(*, unsafe_policy: bool = False, store_writes: int = 0) -> dict[str, object]:
    return {
        "schema": "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-apply-design.v1",
        "status": "ready",
        "decision": "ready_for_limited_visual_retrieval_hint_candidate_store_apply_review",
        "counts": {
            "applyDesignRows": 1,
            "limitedApplyDesignCandidateRows": 1,
            "plannedSeparateApplyWriteRows": 1,
            "blockedRows": 0,
            "candidateStoreWriteRows": store_writes,
        },
        "applyDesignRowsDetail": [_apply_design_row(unsafe_policy=unsafe_policy)],
    }


def test_apply_review_projects_ready_rows_without_store_writes() -> None:
    report = build_limited_visual_retrieval_hint_candidate_store_apply_review(
        _apply_design_report(),
        source_apply_design_report_ref="eval/knowledgeos/reports/apply_design_fixture.v1.json",
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["schema"] == LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_REVIEW_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == "ready_for_limited_visual_retrieval_hint_candidate_store_apply_executor_dry_run"
    assert report["nextRecommendedTranche"] == "limited_visual_retrieval_hint_candidate_store_apply_executor_dry_run"
    assert report["counts"]["reviewRows"] == 1
    assert report["counts"]["reviewReadyRows"] == 1
    assert report["counts"]["applyExecutorDryRunCandidateRows"] == 1
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert report["scope"]["candidateStoreApplyExecutorRows"] == 0

    row = report["reviewRowsDetail"][0]
    assert row["reviewStatus"] == "ready_for_apply_executor_dry_run"
    assert row["reviewPlan"]["applyExecutorDryRunCandidate"] is True
    assert row["reviewPlan"]["candidateStoreWrite"] is False
    assert row["reviewPlan"]["applyAllowedByThisReport"] is False
    assert all(row["checks"].values())

    validation = validate_payload(
        report,
        LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_REVIEW_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors


def test_apply_review_blocks_unsafe_policy_row() -> None:
    report = build_limited_visual_retrieval_hint_candidate_store_apply_review(
        _apply_design_report(unsafe_policy=True),
        source_apply_design_report_ref="eval/knowledgeos/reports/apply_design_fixture.v1.json",
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["reviewReadyRows"] == 0
    assert report["counts"]["blockedRows"] == 1
    assert "strictEvidenceFalse" in report["reviewRowsDetail"][0]["blockerReason"]


def test_apply_review_blocks_source_report_with_store_writes() -> None:
    report = build_limited_visual_retrieval_hint_candidate_store_apply_review(
        _apply_design_report(store_writes=1),
        source_apply_design_report_ref="eval/knowledgeos/reports/apply_design_fixture.v1.json",
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert report["sourceBlockers"] == ["apply_design_has_store_writes"]


def test_apply_review_writer_uses_sanitized_refs(tmp_path: Path) -> None:
    report = build_limited_visual_retrieval_hint_candidate_store_apply_review(
        _apply_design_report(),
        source_apply_design_report_ref="eval/knowledgeos/reports/apply_design_fixture.v1.json",
        generated_at="2026-05-27T00:00:00Z",
    )
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    write_limited_visual_retrieval_hint_candidate_store_apply_review(
        report,
        report_json=json_path,
        report_md=md_path,
    )

    combined = json_path.read_text(encoding="utf-8") + md_path.read_text(encoding="utf-8")
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["sourceApplyDesignReport"]["reportRef"].startswith("eval/knowledgeos/reports/")
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "i" + "Cloud" not in combined
