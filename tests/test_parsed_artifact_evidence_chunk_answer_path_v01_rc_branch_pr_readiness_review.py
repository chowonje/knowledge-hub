from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_branch_pr_readiness_review import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_BRANCH_PR_READINESS_REVIEW_SCHEMA_ID,
    READY_DECISION,
    SELF_REVIEW_ALLOWED_DIRTY_PATHS,
    build_parsed_artifact_evidence_chunk_answer_path_v01_rc_branch_pr_readiness_review,
    write_parsed_artifact_evidence_chunk_answer_path_v01_rc_branch_pr_readiness_review,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_LABS_RELEASE_GATE_SCHEMA_ID,
    READY_DECISION as RELEASE_GATE_READY_DECISION,
)


def _release_gate() -> dict[str, Any]:
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_LABS_RELEASE_GATE_SCHEMA_ID,
        "status": "ready",
        "decision": RELEASE_GATE_READY_DECISION,
        "counts": {
            "releaseGatePassRows": 1,
            "releaseGateBlockedRows": 0,
            "releaseSmokePassedRows": 10,
            "publicHygieneIssueRows": 0,
            "noAnswerPassRows": 3,
            "labsSurfaceSmokePassRows": 1,
            "publicDefaultPromotionReadyRows": 0,
            "publicDefaultPromotionHeldRows": 1,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
        },
        "gate": {
            "readyForV01RcBranchPrReadinessReview": True,
            "publicDefaultPromotionAllowed": False,
        },
    }


def _git_state(**updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "branchName": "codex/next-implementation-20260528",
        "baseRef": "refs/remotes/origin/main",
        "headShortSha": "abc1234",
        "baseShortSha": "def5678",
        "aheadCommits": 33,
        "behindCommits": 0,
        "remoteMainMatchesLocalOriginMain": True,
        "statusRows": [],
        "changedPaths": [
            "docs/knowledge_os_definition.md",
            "knowledge_hub/ai/rag.py",
            "tests/test_paper_evidence_chunk_answer_preview.py",
        ],
        "commitRows": [
            {"shortSha": "abc1234", "subject": "eval(papers): gate labs evidence chunk rc readiness"},
            {"shortSha": "def5678", "subject": "docs: define KnowledgeOS product goal"},
        ],
    }
    payload.update(updates)
    return payload


def _github_pr_state(**updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"lookupStatus": "ok", "openPrRows": []}
    payload.update(updates)
    return payload


def _build(**updates: Any) -> dict[str, Any]:
    inputs = {
        "release_gate_report": _release_gate(),
        "git_state": _git_state(),
        "github_pr_state": _github_pr_state(),
        "generated_at": "2026-05-29T00:00:00Z",
    }
    inputs.update(updates)
    return build_parsed_artifact_evidence_chunk_answer_path_v01_rc_branch_pr_readiness_review(**inputs)


def test_branch_pr_readiness_ready_for_draft_pr_review_not_merge() -> None:
    report = _build()

    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["readinessDecision"]["branchPrReadiness"] == "ready_for_draft_pr_review"
    assert report["readinessDecision"]["readyForMerge"] is False
    assert report["counts"]["branchAheadCommitRows"] == 33
    assert report["counts"]["branchBehindCommitRows"] == 0
    assert report["counts"]["openPrRows"] == 0
    assert report["counts"]["readyForDraftPrRows"] == 1
    assert report["counts"]["readyForMergeRows"] == 0
    assert report["gate"]["readyForDraftPrReview"] is True
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_BRANCH_PR_READINESS_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_branch_pr_readiness_allows_self_review_dirty_paths() -> None:
    self_path = next(iter(SELF_REVIEW_ALLOWED_DIRTY_PATHS))
    report = _build(git_state=_git_state(statusRows=[{"statusCode": "M", "path": self_path}]))

    assert report["status"] == "ready"
    assert report["counts"]["worktreeDirtyRows"] == 1
    assert report["counts"]["selfReviewDirtyRows"] == 1
    assert report["counts"]["blockingDirtyRows"] == 0


def test_branch_pr_readiness_blocks_unrelated_dirty_paths() -> None:
    report = _build(git_state=_git_state(statusRows=[{"statusCode": "M", "path": "knowledge_hub/ai/rag.py"}]))

    assert report["status"] == "blocked"
    assert "blocking_dirty_worktree_rows_present" in report["gate"]["semanticViolations"]
    assert report["counts"]["blockingDirtyRows"] == 1


def test_branch_pr_readiness_blocks_when_branch_behind_origin_main() -> None:
    report = _build(git_state=_git_state(behindCommits=1))

    assert report["status"] == "blocked"
    assert "branch_is_behind_origin_main" in report["gate"]["semanticViolations"]


def test_branch_pr_readiness_blocks_release_gate_failure() -> None:
    release_gate = _release_gate()
    release_gate["status"] = "blocked"

    report = _build(release_gate_report=release_gate)

    assert report["status"] == "blocked"
    assert "release_gate_not_ready" in report["gate"]["semanticViolations"]


def test_branch_pr_readiness_blocks_conflicting_open_pr() -> None:
    pr_state = _github_pr_state(
        openPrRows=[
            {
                "number": 149,
                "state": "OPEN",
                "isDraft": True,
                "mergeStateStatus": "DIRTY",
                "headRefName": "codex/next-implementation-20260528",
                "baseRefName": "main",
            }
        ]
    )

    report = _build(github_pr_state=pr_state)

    assert report["status"] == "blocked"
    assert "open_pr_not_merge_clean" in report["gate"]["semanticViolations"]


def test_branch_pr_readiness_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    report = _build()

    paths = write_parsed_artifact_evidence_chunk_answer_path_v01_rc_branch_pr_readiness_review(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )

    parsed = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert parsed["status"] == "ready"
    assert Path(paths["markdown"]).read_text(encoding="utf-8").startswith(
        "# Parsed Artifact Evidence Chunk Answer Path v0.1 RC Branch PR Readiness Review"
    )
    assert validate_payload(
        parsed,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_BRANCH_PR_READINESS_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok
