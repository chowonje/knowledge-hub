from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_post_open_review import (
    DEFAULT_BRANCH,
    DEFAULT_PR_NUMBER,
    DEFAULT_PUBLIC_DEFAULT_PROMOTION_GATE_REPORT,
    DEFAULT_RELEASE_NOTES_REVIEW_REPORT,
    DEFAULT_RELEASE_PACKAGE_HANDOFF_REPORT,
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_DRAFT_PR_POST_OPEN_REVIEW_SCHEMA_ID,
    READY_DECISION,
    build_parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_post_open_review,
    write_parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_post_open_review,
)


HEAD_SHA = "a496b9673d915af099074d95222326526de94024"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _handoff_report(**updates: Any) -> dict[str, Any]:
    payload = _json(DEFAULT_RELEASE_PACKAGE_HANDOFF_REPORT)
    for key, value in updates.items():
        payload[key] = value
    return payload


def _promotion_report(**updates: Any) -> dict[str, Any]:
    payload = _json(DEFAULT_PUBLIC_DEFAULT_PROMOTION_GATE_REPORT)
    for key, value in updates.items():
        payload[key] = value
    return payload


def _release_notes_report(**updates: Any) -> dict[str, Any]:
    payload = _json(DEFAULT_RELEASE_NOTES_REVIEW_REPORT)
    for key, value in updates.items():
        payload[key] = value
    return payload


def _current_state(**updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "branchName": DEFAULT_BRANCH,
        "headSha": HEAD_SHA,
        "headShortSha": "a496b96",
        "originMainSha": "ef7138b8b2c81c85611d3085a719658cd4cd7110",
        "originMainShortSha": "ef7138b",
        "remoteMainSha": "ef7138b8b2c81c85611d3085a719658cd4cd7110",
        "remoteMainShortSha": "ef7138b",
        "remoteMainVerified": True,
        "remoteBranchSha": HEAD_SHA,
        "remoteBranchShortSha": "a496b96",
        "aheadCommits": 3,
        "behindCommits": 0,
        "statusRows": [],
    }
    payload.update(updates)
    return payload


def _check(name: str) -> dict[str, Any]:
    return {
        "__typename": "CheckRun",
        "name": name,
        "workflowName": "CI",
        "status": "COMPLETED",
        "conclusion": "SUCCESS",
    }


def _pr_state(**updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "lookupStatus": "ok",
        "number": DEFAULT_PR_NUMBER,
        "title": "KnowledgeOS v0.1 RC Research Preview package",
        "url": "https://github.com/chowonje/knowledge-hub/pull/171",
        "state": "OPEN",
        "isDraft": True,
        "mergeStateStatus": "CLEAN",
        "headRefName": DEFAULT_BRANCH,
        "baseRefName": "main",
        "headRefOid": HEAD_SHA,
        "baseRefOid": "ef7138b8b2c81c85611d3085a719658cd4cd7110",
        "statusCheckRollup": [
            _check("static-guards"),
            _check("python-hermetic (3.10)"),
            _check("python-hermetic (3.11)"),
            _check("python-hermetic (3.12)"),
            _check("python-hermetic (3.13)"),
            _check("authority-contracts"),
            _check("evidence-first-gates"),
        ],
    }
    payload.update(updates)
    return payload


def _build(**updates: Any) -> dict[str, Any]:
    inputs = {
        "release_package_handoff_report": _handoff_report(),
        "public_default_promotion_gate_report": _promotion_report(),
        "release_notes_review_report": _release_notes_report(),
        "current_state": _current_state(),
        "github_pr_state": _pr_state(),
        "generated_at": "2026-05-29T00:00:00Z",
    }
    inputs.update(updates)
    return build_parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_post_open_review(**inputs)


def test_post_open_review_ready_for_human_review_not_merge() -> None:
    report = _build()

    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["postOpenReviewDecision"]["draftPrPostOpenReview"] == "ready_for_human_review_not_merge"
    assert report["postOpenReviewDecision"]["publicDefaultDecision"] == "hold_public_default_promotion"
    assert report["counts"]["openPrRows"] == 1
    assert report["counts"]["draftPrRows"] == 1
    assert report["counts"]["mergeStateCleanRows"] == 1
    assert report["counts"]["ciCheckRows"] == 7
    assert report["counts"]["ciCheckSuccessRows"] == 7
    assert report["counts"]["branchAheadCommitRows"] == 3
    assert report["counts"]["branchBehindCommitRows"] == 0
    assert report["counts"]["releasePackageHandoffReadyRows"] == 1
    assert report["counts"]["publicDefaultPromotionHeldRows"] == 1
    assert report["counts"]["readyForHumanReviewRows"] == 1
    assert report["counts"]["readyForMergeRows"] == 0
    assert report["counts"]["githubPrMutationRows"] == 0
    assert report["gate"]["readyForHumanReview"] is True
    assert report["gate"]["readyForMerge"] is False
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_DRAFT_PR_POST_OPEN_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_post_open_review_blocks_when_pr_is_missing_or_unavailable() -> None:
    report = _build(github_pr_state={"lookupStatus": "unavailable", "number": DEFAULT_PR_NUMBER})

    assert report["status"] == "blocked"
    assert "github_pr_lookup_failed" in report["gate"]["semanticViolations"]


def test_post_open_review_blocks_closed_pr() -> None:
    report = _build(github_pr_state=_pr_state(state="CLOSED"))

    assert report["status"] == "blocked"
    assert "github_pr_not_open" in report["gate"]["semanticViolations"]


def test_post_open_review_blocks_wrong_base_or_head() -> None:
    report = _build(github_pr_state=_pr_state(baseRefName="develop", headRefName="codex/other"))

    assert report["status"] == "blocked"
    assert "github_pr_base_not_main" in report["gate"]["semanticViolations"]
    assert "github_pr_head_branch_mismatch" in report["gate"]["semanticViolations"]


def test_post_open_review_blocks_non_clean_merge_state() -> None:
    report = _build(github_pr_state=_pr_state(mergeStateStatus="DIRTY"))

    assert report["status"] == "blocked"
    assert "github_pr_merge_state_not_clean" in report["gate"]["semanticViolations"]


def test_post_open_review_blocks_failed_or_pending_ci() -> None:
    checks = [_check("static-guards"), _check("python-hermetic (3.10)")]
    checks[1]["conclusion"] = "FAILURE"
    report = _build(github_pr_state=_pr_state(statusCheckRollup=checks))

    assert report["status"] == "blocked"
    assert "github_pr_ci_check_count_not_seven" in report["gate"]["semanticViolations"]
    assert "github_pr_ci_checks_not_all_success" in report["gate"]["semanticViolations"]


def test_post_open_review_blocks_when_release_package_handoff_not_ready() -> None:
    report = _build(release_package_handoff_report=_handoff_report(status="blocked"))

    assert report["status"] == "blocked"
    assert "release_package_handoff_not_ready" in report["gate"]["semanticViolations"]


def test_post_open_review_blocks_when_release_notes_review_not_ready() -> None:
    report = _build(release_notes_review_report=_release_notes_report(status="blocked"))

    assert report["status"] == "blocked"
    assert "release_notes_review_not_ready" in report["gate"]["semanticViolations"]


def test_post_open_review_blocks_public_default_promotion_ready() -> None:
    promotion = _promotion_report()
    promotion["counts"]["publicDefaultPromotionReadyRows"] = 1
    report = _build(public_default_promotion_gate_report=promotion)

    assert report["status"] == "blocked"
    assert "public_default_promotion_ready_unexpected" in report["gate"]["semanticViolations"]


def test_post_open_review_tolerates_pre_pr_handoff_snapshot_as_stale() -> None:
    report = _build()

    assert report["status"] == "ready"
    assert report["counts"]["stalePrePrHandoffRows"] == 1
    assert report["staleArtifactRows"][0]["staleReason"] == "pre_pr_open_handoff_snapshot"
    assert report["staleArtifactRows"][0]["currentAuthority"] == "live_github_pr_state"


def test_post_open_review_blocks_private_path_marker() -> None:
    marker = "/" + "Users" + "/example/private"
    report = _build(current_state=_current_state(statusRows=[{"statusCode": "M", "path": marker}]))

    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] == 1
    assert "draft_pr_post_open_review_private_path_marker" in report["gate"]["semanticViolations"]


def test_post_open_review_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    report = _build()

    paths = write_parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_post_open_review(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )

    parsed = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert parsed["status"] == "ready"
    assert Path(paths["markdown"]).read_text(encoding="utf-8").startswith(
        "# Parsed Artifact Evidence Chunk Answer Path v0.1 RC Draft PR Post-Open Review"
    )
    assert validate_payload(
        parsed,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_DRAFT_PR_POST_OPEN_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok
