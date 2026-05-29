from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.knowledgeos_v01_rc_post_merge_convergence_cleanup_decision import (
    DEFAULT_BASE_BRANCH,
    DEFAULT_HEAD_BRANCH,
    DEFAULT_PR_NUMBER,
    DEFAULT_VISION_BOTTLENECK_REPORT,
    KNOWLEDGEOS_V01_RC_POST_MERGE_CONVERGENCE_CLEANUP_DECISION_SCHEMA_ID,
    READY_DECISION,
    build_knowledgeos_v01_rc_post_merge_convergence_cleanup_decision,
    write_knowledgeos_v01_rc_post_merge_convergence_cleanup_decision,
)


MERGE_SHA = "026f7e7c53f3431d3846155ac25a8899ecf6f6ce"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _vision_report(**updates: Any) -> dict[str, Any]:
    payload = _json(DEFAULT_VISION_BOTTLENECK_REPORT)
    for key, value in updates.items():
        payload[key] = value
    return payload


def _checks(*, total: int = 7, success: int = 7) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(total):
        rows.append(
            {
                "__typename": "CheckRun",
                "name": f"check-{index + 1}",
                "workflowName": "CI",
                "status": "COMPLETED",
                "conclusion": "SUCCESS" if index < success else "FAILURE",
            }
        )
    return rows


def _pr_state(**updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "lookupStatus": "ok",
        "number": DEFAULT_PR_NUMBER,
        "title": "KnowledgeOS v0.1 RC public default promotion decision gate",
        "url": "https://github.com/chowonje/knowledge-hub/pull/171",
        "state": "MERGED",
        "isDraft": False,
        "mergedAt": "2026-05-29T08:17:57Z",
        "mergeCommit": {"oid": MERGE_SHA},
        "headRefName": DEFAULT_HEAD_BRANCH,
        "baseRefName": DEFAULT_BASE_BRANCH,
        "statusCheckRollup": _checks(),
    }
    payload.update(updates)
    return payload


def _git_state(**updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "branchName": "codex/v01-rc-post-merge-convergence-pr171-20260529",
        "headSha": MERGE_SHA,
        "headShortSha": MERGE_SHA[:7],
        "originMainSha": MERGE_SHA,
        "originMainShortSha": MERGE_SHA[:7],
        "remoteMainSha": MERGE_SHA,
        "remoteMainShortSha": MERGE_SHA[:7],
        "remoteFeatureBranchSha": "b7d8475c11111111111111111111111111111111",
        "remoteFeatureBranchShortSha": "b7d8475",
        "statusRows": [],
    }
    payload.update(updates)
    return payload


def _build(**updates: Any) -> dict[str, Any]:
    inputs = {
        "vision_bottleneck_report": _vision_report(),
        "git_state": _git_state(),
        "github_pr_state": _pr_state(),
        "release_smoke_result": {"status": "ok", "checkedCount": 10, "passedCount": 10},
        "hygiene_result": {"status": "ok", "issueCount": 0},
        "generated_at": "2026-05-29T00:00:00Z",
    }
    inputs.update(updates)
    return build_knowledgeos_v01_rc_post_merge_convergence_cleanup_decision(**inputs)


def test_post_merge_convergence_ready_after_pr_171_merge() -> None:
    report = _build()

    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["nextRecommendedTranche"] == "corpus_scale_answer_quality_gate"
    assert report["postMergeDecision"]["postMergeConvergence"] == "research_preview_rc_merged"
    assert report["postMergeDecision"]["branchCleanupDecision"] == "cleanup_recommended_not_applied"
    assert report["counts"]["prMergedRows"] == 1
    assert report["counts"]["mainMergeCommitMatchRows"] == 1
    assert report["counts"]["remoteMainMatchesLocalOriginRows"] == 1
    assert report["counts"]["ciCheckRows"] == 7
    assert report["counts"]["ciCheckSuccessRows"] == 7
    assert report["counts"]["releaseSmokePassedRows"] == 10
    assert report["counts"]["publicDefaultPromotionReadyRows"] == 0
    assert report["counts"]["publicDefaultPromotionHeldRows"] == 1
    assert report["counts"]["generalRcReadyRows"] == 0
    assert report["counts"]["corpusScaleClaimProvenRows"] == 0
    assert report["counts"]["branchCleanupRecommendedRows"] == 1
    assert report["counts"]["branchCleanupAppliedRows"] == 0
    assert report["counts"]["branchDeletionRows"] == 0
    assert report["counts"]["privatePathLeakRows"] == 0
    assert report["counts"]["schemaViolationCount"] == 0
    assert report["gate"]["publicDefaultPromotionAllowed"] is False
    assert report["gate"]["generalRcReady"] is False
    assert validate_payload(
        report,
        KNOWLEDGEOS_V01_RC_POST_MERGE_CONVERGENCE_CLEANUP_DECISION_SCHEMA_ID,
        strict=True,
    ).ok


def test_post_merge_convergence_blocks_when_pr_not_merged() -> None:
    report = _build(github_pr_state=_pr_state(state="OPEN"))

    assert report["status"] == "blocked"
    assert "github_pr_not_merged" in report["gate"]["semanticViolations"]


def test_post_merge_convergence_blocks_when_pr_is_still_draft() -> None:
    report = _build(github_pr_state=_pr_state(isDraft=True))

    assert report["status"] == "blocked"
    assert "github_pr_still_draft" in report["gate"]["semanticViolations"]


def test_post_merge_convergence_blocks_when_ci_not_green() -> None:
    report = _build(github_pr_state=_pr_state(statusCheckRollup=_checks(success=6)))

    assert report["status"] == "blocked"
    assert report["counts"]["ciCheckSuccessRows"] == 6
    assert "github_pr_ci_checks_not_all_success" in report["gate"]["semanticViolations"]


def test_post_merge_convergence_blocks_when_main_does_not_match_merge_commit() -> None:
    report = _build(git_state=_git_state(originMainSha="9" * 40, originMainShortSha="9999999"))

    assert report["status"] == "blocked"
    assert report["counts"]["mainMergeCommitMatchRows"] == 0
    assert "origin_main_does_not_match_pr_merge_commit" in report["gate"]["semanticViolations"]


def test_post_merge_convergence_blocks_dirty_rows_outside_self_review_paths() -> None:
    report = _build(git_state=_git_state(statusRows=[{"statusCode": "M", "path": "knowledge_hub/runtime.py"}]))

    assert report["status"] == "blocked"
    assert "blocking_dirty_worktree_rows_present" in report["gate"]["semanticViolations"]


def test_post_merge_convergence_allows_self_review_dirty_paths() -> None:
    report = _build(
        git_state=_git_state(
            statusRows=[
                {"statusCode": "A", "path": "tests/test_knowledgeos_v01_rc_post_merge_convergence_cleanup_decision.py"},
                {"statusCode": "M", "path": "docs/PROJECT_STATE.md"},
            ]
        )
    )

    assert report["status"] == "ready"


def test_post_merge_convergence_blocks_release_smoke_failure() -> None:
    report = _build(release_smoke_result={"status": "failed", "checkedCount": 10, "passedCount": 9})

    assert report["status"] == "blocked"
    assert "release_smoke_not_ok" in report["gate"]["semanticViolations"]
    assert "release_smoke_not_all_passed" in report["gate"]["semanticViolations"]


def test_post_merge_convergence_blocks_public_hygiene_issue() -> None:
    report = _build(hygiene_result={"status": "ok", "issueCount": 1})

    assert report["status"] == "blocked"
    assert report["counts"]["publicHygieneIssueRows"] == 1
    assert "public_hygiene_issues_present" in report["gate"]["semanticViolations"]


def test_post_merge_convergence_blocks_when_vision_review_not_ready() -> None:
    report = _build(vision_bottleneck_report=_vision_report(status="blocked"))

    assert report["status"] == "blocked"
    assert "vision_bottleneck_definition_not_ready" in report["gate"]["semanticViolations"]


def test_post_merge_convergence_blocks_unsafe_counter_in_upstream_report() -> None:
    vision = _vision_report()
    vision["counts"] = deepcopy(vision["counts"])
    vision["counts"]["candidateStoreWriteRows"] = 1
    report = _build(vision_bottleneck_report=vision)

    assert report["status"] == "blocked"
    assert "unsafe_counter_nonzero:report1:candidateStoreWriteRows" in report["gate"]["semanticViolations"]


def test_post_merge_convergence_handles_no_remote_feature_branch() -> None:
    report = _build(git_state=_git_state(remoteFeatureBranchSha="", remoteFeatureBranchShortSha=""))

    assert report["status"] == "ready"
    assert report["postMergeDecision"]["branchCleanupDecision"] == "no_remote_branch_cleanup_needed"
    assert report["counts"]["branchCleanupRecommendedRows"] == 0
    assert report["cleanupRows"][0]["status"] == "not_needed"


def test_post_merge_convergence_blocks_private_path_marker() -> None:
    marker = "/" + "Users" + "/example/private"
    report = _build(github_pr_state=_pr_state(title=marker))

    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] == 1
    assert "post_merge_convergence_cleanup_private_path_marker" in report["gate"]["semanticViolations"]


def test_post_merge_convergence_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    report = _build()

    paths = write_knowledgeos_v01_rc_post_merge_convergence_cleanup_decision(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )

    parsed = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert parsed["status"] == "ready"
    assert Path(paths["markdown"]).read_text(encoding="utf-8").startswith(
        "# KnowledgeOS v0.1 RC Post-Merge Convergence Cleanup Decision"
    )
    assert validate_payload(
        parsed,
        KNOWLEDGEOS_V01_RC_POST_MERGE_CONVERGENCE_CLEANUP_DECISION_SCHEMA_ID,
        strict=True,
    ).ok
