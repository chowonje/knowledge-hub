from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_branch_pr_readiness_review import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_BRANCH_PR_READINESS_REVIEW_SCHEMA_ID,
    READY_DECISION as BRANCH_READY_DECISION,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_opening_or_handoff import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_DRAFT_PR_OPENING_OR_HANDOFF_SCHEMA_ID,
    READY_DECISION,
    SELF_HANDOFF_ALLOWED_DIRTY_PATHS,
    build_parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_opening_or_handoff,
    write_parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_opening_or_handoff,
)


def _branch_report(**updates: Any) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_BRANCH_PR_READINESS_REVIEW_SCHEMA_ID,
        "status": "ready",
        "decision": BRANCH_READY_DECISION,
        "readinessDecision": {
            "branchPrReadiness": "ready_for_draft_pr_review",
            "readyForMerge": False,
            "publicDefaultDecision": "hold_public_default_promotion",
        },
        "counts": {
            "readyForDraftPrRows": 1,
            "readyForMergeRows": 0,
            "blockingDirtyRows": 0,
            "branchBehindCommitRows": 0,
            "releaseSmokePassedRows": 10,
            "publicHygieneIssueRows": 0,
            "noAnswerPassRows": 3,
            "changedFileRows": 184,
            "publicDefaultPromotionReadyRows": 0,
            "publicDefaultPromotionHeldRows": 1,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
        },
        "gate": {
            "readyForDraftPrReview": True,
            "readyForMerge": False,
            "publicDefaultPromotionAllowed": False,
        },
    }
    report.update(updates)
    return report


def _current_state(**updates: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "branchName": "codex/next-implementation-20260528",
        "headShortSha": "abc1234",
        "aheadCommits": 34,
        "behindCommits": 0,
        "statusRows": [],
        "remoteMainVerified": True,
    }
    state.update(updates)
    return state


def _github_pr_state(**updates: Any) -> dict[str, Any]:
    state: dict[str, Any] = {"lookupStatus": "ok", "openPrRows": []}
    state.update(updates)
    return state


def _build(**updates: Any) -> dict[str, Any]:
    inputs = {
        "branch_pr_readiness_report": _branch_report(),
        "current_state": _current_state(),
        "github_pr_state": _github_pr_state(),
        "generated_at": "2026-05-29T00:00:00Z",
    }
    inputs.update(updates)
    return build_parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_opening_or_handoff(**inputs)


def test_draft_pr_handoff_ready_without_mutating_github() -> None:
    report = _build()

    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["handoffDecision"]["operatorAction"] == "push_branch_and_open_draft_pr"
    assert report["handoffDecision"]["readyForDraftPrCreation"] is True
    assert report["handoffDecision"]["readyForMerge"] is False
    assert report["counts"]["currentBranchAheadCommitRows"] == 34
    assert report["counts"]["currentDirtyRows"] == 0
    assert report["counts"]["currentBlockingDirtyRows"] == 0
    assert report["counts"]["openPrRows"] == 0
    assert report["counts"]["pushRows"] == 0
    assert report["counts"]["githubPrMutationRows"] == 0
    assert "public default promotion ready rows: `0`" in report["draftPr"]["bodyMarkdown"]
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_DRAFT_PR_OPENING_OR_HANDOFF_SCHEMA_ID,
        strict=True,
    ).ok


def test_draft_pr_handoff_blocks_when_branch_readiness_not_ready() -> None:
    branch = _branch_report(status="blocked")

    report = _build(branch_pr_readiness_report=branch)

    assert report["status"] == "blocked"
    assert "branch_pr_readiness_not_ready" in report["gate"]["semanticViolations"]
    assert report["draftPr"]["bodyMarkdown"] == ""


def test_draft_pr_handoff_allows_self_handoff_dirty_rows() -> None:
    self_path = next(iter(SELF_HANDOFF_ALLOWED_DIRTY_PATHS))
    report = _build(current_state=_current_state(statusRows=[{"statusCode": "M", "path": self_path}]))

    assert report["status"] == "ready"
    assert report["counts"]["currentDirtyRows"] == 1
    assert report["counts"]["currentSelfHandoffDirtyRows"] == 1
    assert report["counts"]["currentBlockingDirtyRows"] == 0


def test_draft_pr_handoff_blocks_unrelated_dirty_current_worktree() -> None:
    report = _build(current_state=_current_state(statusRows=[{"statusCode": "M", "path": "knowledge_hub/ai/rag.py"}]))

    assert report["status"] == "blocked"
    assert "current_worktree_has_blocking_dirty_rows" in report["gate"]["semanticViolations"]
    assert report["counts"]["currentBlockingDirtyRows"] == 1


def test_draft_pr_handoff_blocks_when_current_branch_behind() -> None:
    report = _build(current_state=_current_state(behindCommits=1))

    assert report["status"] == "blocked"
    assert "current_branch_behind_origin_main" in report["gate"]["semanticViolations"]


def test_draft_pr_handoff_handles_existing_open_pr() -> None:
    pr_state = _github_pr_state(
        openPrRows=[
            {
                "number": 151,
                "state": "OPEN",
                "isDraft": True,
                "mergeStateStatus": "CLEAN",
                "headRefName": "codex/next-implementation-20260528",
                "baseRefName": "main",
            }
        ]
    )

    report = _build(github_pr_state=pr_state)

    assert report["status"] == "ready"
    assert report["handoffDecision"]["operatorAction"] == "review_existing_draft_pr"
    assert report["handoffDecision"]["readyForDraftPrCreation"] is False
    assert report["handoffDecision"]["readyForExistingDraftPrReview"] is True


def test_draft_pr_handoff_blocks_conflicting_open_pr() -> None:
    pr_state = _github_pr_state(
        openPrRows=[
            {
                "number": 151,
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


def test_draft_pr_handoff_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    report = _build()

    paths = write_parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_opening_or_handoff(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )

    parsed = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert parsed["status"] == "ready"
    assert Path(paths["markdown"]).read_text(encoding="utf-8").startswith(
        "# Parsed Artifact Evidence Chunk Answer Path v0.1 RC Draft PR Handoff"
    )
    assert validate_payload(
        parsed,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_DRAFT_PR_OPENING_OR_HANDOFF_SCHEMA_ID,
        strict=True,
    ).ok
