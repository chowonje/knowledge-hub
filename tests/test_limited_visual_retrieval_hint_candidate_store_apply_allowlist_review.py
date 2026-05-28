from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_apply_allowlist_review import (
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_ALLOWLIST_REVIEW_SCHEMA_ID,
    build_limited_visual_retrieval_hint_candidate_store_apply_allowlist_review,
    write_limited_visual_retrieval_hint_candidate_store_apply_allowlist_review,
)


def _hash() -> str:
    return "sha256:" + "a" * 64


def _row(index: int, *, candidate: bool, unsafe_policy: bool = False) -> dict[str, object]:
    hint_id = f"visual-retrieval-hint:sample-paper:figure_caption_region:{index}:abcdefabcdefab{index:02d}"
    source_id = f"visual-layout:sample-paper:figure_caption_region:{index}:aaaaaaaaaaaaaa{index:02d}"
    blocker = "" if candidate else "search_eval_gate_not_passed"
    return {
        "schema": "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-apply-design-row.v1",
        "applyDesignRowId": f"limited-visual-retrieval-hint-apply-design:fixture{index}",
        "hintCandidateId": hint_id,
        "sourceCandidateId": source_id,
        "paperId": "sample-paper",
        "paperRef": "papers_dir/sample.pdf",
        "sourceContentHash": _hash(),
        "page": index,
        "bbox": [10.0, 20.0, 120.0, 180.0],
        "candidateType": "figure_caption_region",
        "sourceDryRunReportRef": "eval/knowledgeos/reports/dry_run_fixture.v1.json",
        "sourceDryRunRowId": f"visual-retrieval-hint-dry-run:fixture{index}",
        "plannedStoreRef": "papers_dir/visual_retrieval_hints/visual_retrieval_hint_candidates.v1.jsonl",
        "idempotencyKey": f"visual-retrieval-hint-idempotency:fixture{index}",
        "plannedJsonlRecordSha256": "sha256:" + f"{index}" * 64,
        "recordPreview": {
            "derivedTextForRetrievalPreview": "Retrieval hint only: a ReLU tanh CIFAR-10 training error plot.",
            "visibleTextPreview": "Visible fragments include: ReLU, tanh, CIFAR-10, training error.",
            "retrievalKeywords": ["ReLU", "tanh", "CIFAR-10", "training error"],
            "limitations": "Retrieval hint only, not evidence.",
        },
        "searchEvalSummary": {
            "sourceSearchEvalQueryRowIds": [
                f"visual-retrieval-hint-search-query:{index:020d}",
                f"visual-retrieval-hint-search-query:{index + 100:020d}",
            ],
            "queryRows": 2,
            "textOnlyBestRank": 20,
            "visualHintAugmentedBestRank": 1 if candidate else 7,
            "augmentedHitAt5Rows": 2 if candidate else 1,
            "improvedQueryRows": 2 if candidate else 0,
            "regressedQueryRows": 0,
            "strongLiftQueryRows": 1 if candidate else 0,
            "blockedQueryRows": 0,
            "passesLimitedApplySearchGate": candidate,
        },
        "applyPlan": {
            "limitedApplyDesignCandidate": candidate,
            "wouldWriteOnSeparateExplicitApply": candidate,
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
        "blockerReason": blocker,
    }


def _apply_design_report(*, unsafe_policy: bool = False, store_writes: int = 0) -> dict[str, object]:
    rows = [
        _row(1, candidate=True, unsafe_policy=unsafe_policy),
        _row(2, candidate=False),
    ]
    return {
        "schema": "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-apply-design.v1",
        "status": "blocked",
        "decision": "blocked",
        "sourceBlockers": ["search_eval_has_regressions"],
        "counts": {
            "applyDesignRows": 2,
            "limitedApplyDesignCandidateRows": 1,
            "plannedSeparateApplyWriteRows": 1,
            "blockedRows": 1,
            "candidateStoreWriteRows": store_writes,
            "privatePathLeakRows": 0,
        },
        "applyDesignRowsDetail": rows,
    }


def test_allowlist_review_splits_candidates_and_holdouts_without_store_writes() -> None:
    report = build_limited_visual_retrieval_hint_candidate_store_apply_allowlist_review(
        _apply_design_report(),
        source_apply_design_report_ref="eval/knowledgeos/reports/apply_design_fixture.v1.json",
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["schema"] == LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_ALLOWLIST_REVIEW_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == (
        "ready_for_limited_visual_retrieval_hint_candidate_store_allowlist_apply_executor_dry_run"
    )
    assert report["counts"]["reviewRows"] == 2
    assert report["counts"]["allowlistRows"] == 1
    assert report["counts"]["holdoutRows"] == 1
    assert report["counts"]["blockedRows"] == 0
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert report["scope"]["candidateStoreApplyExecutorRows"] == 0

    rows = report["reviewRowsDetail"]
    assert rows[0]["reviewStatus"] == "allowlisted_for_apply_executor_dry_run"
    assert rows[0]["reviewPlan"]["applyExecutorDryRunCandidate"] is True
    assert rows[0]["reviewPlan"]["candidateStoreWrite"] is False
    assert rows[1]["reviewStatus"] == "holdout_pending_search_gate_review"
    assert rows[1]["reviewPlan"]["holdoutFromApplyExecutorDryRun"] is True
    assert rows[1]["blockerReason"] == "search_eval_gate_not_passed"

    validation = validate_payload(
        report,
        LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_ALLOWLIST_REVIEW_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors


def test_allowlist_review_blocks_unsafe_allowlist_policy_row() -> None:
    report = build_limited_visual_retrieval_hint_candidate_store_apply_allowlist_review(
        _apply_design_report(unsafe_policy=True),
        source_apply_design_report_ref="eval/knowledgeos/reports/apply_design_fixture.v1.json",
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["allowlistRows"] == 0
    assert report["counts"]["holdoutRows"] == 1
    assert report["counts"]["blockedRows"] == 1
    assert "strictEvidenceFalse" in report["reviewRowsDetail"][0]["blockerReason"]


def test_allowlist_review_blocks_source_report_with_store_writes() -> None:
    report = build_limited_visual_retrieval_hint_candidate_store_apply_allowlist_review(
        _apply_design_report(store_writes=1),
        source_apply_design_report_ref="eval/knowledgeos/reports/apply_design_fixture.v1.json",
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert report["sourceBlockers"] == ["apply_design_has_store_writes"]


def test_allowlist_review_writer_uses_sanitized_refs(tmp_path: Path) -> None:
    report = build_limited_visual_retrieval_hint_candidate_store_apply_allowlist_review(
        _apply_design_report(),
        source_apply_design_report_ref="eval/knowledgeos/reports/apply_design_fixture.v1.json",
        generated_at="2026-05-27T00:00:00Z",
    )
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    write_limited_visual_retrieval_hint_candidate_store_apply_allowlist_review(
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
