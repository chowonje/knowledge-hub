from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_release_package_handoff import (
    DEFAULT_BRANCH,
    DEFAULT_RELEASE_NOTES_REVIEW_REPORT,
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_RELEASE_PACKAGE_HANDOFF_SCHEMA_ID,
    READY_DECISION,
    build_parsed_artifact_evidence_chunk_answer_path_v01_rc_release_package_handoff,
    write_parsed_artifact_evidence_chunk_answer_path_v01_rc_release_package_handoff,
)


def _release_notes_report(**updates: Any) -> dict[str, Any]:
    report = json.loads(DEFAULT_RELEASE_NOTES_REVIEW_REPORT.read_text(encoding="utf-8"))
    for key, value in updates.items():
        report[key] = value
    return report


def _current_state(**updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "branchName": DEFAULT_BRANCH,
        "headSha": "05eeef4aa0000000000000000000000000000000",
        "headShortSha": "05eeef4",
        "originMainSha": "ef7138baa0000000000000000000000000000000",
        "originMainShortSha": "ef7138b",
        "remoteMainSha": "ef7138baa0000000000000000000000000000000",
        "remoteMainShortSha": "ef7138b",
        "remoteMainVerified": True,
        "aheadCommits": 2,
        "behindCommits": 0,
        "remoteBranchExists": False,
        "statusRows": [],
    }
    payload.update(updates)
    return payload


def _github_pr_state(**updates: Any) -> dict[str, Any]:
    payload = {"lookupStatus": "ok", "openPrRows": []}
    payload.update(updates)
    return payload


def _build(
    *,
    release_report: dict[str, Any] | None = None,
    current_state: dict[str, Any] | None = None,
    github_pr_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_parsed_artifact_evidence_chunk_answer_path_v01_rc_release_package_handoff(
        release_notes_review_report=release_report or _release_notes_report(),
        current_state=current_state or _current_state(),
        github_pr_state=github_pr_state or _github_pr_state(),
        generated_at="2026-05-29T00:00:00Z",
    )


def test_release_package_handoff_ready_for_operator_push_and_draft_pr_only() -> None:
    report = _build()

    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["handoffDecision"]["releasePackageHandoff"] == "ready_for_operator_push_and_draft_pr"
    assert report["handoffDecision"]["mergeDecision"] == "not_ready_for_merge"
    assert report["handoffDecision"]["releaseTagDecision"] == "not_ready_for_release_tag"
    assert report["counts"]["currentBranchAheadCommitRows"] == 2
    assert report["counts"]["currentBranchBehindCommitRows"] == 0
    assert report["counts"]["openPrRows"] == 0
    assert report["counts"]["draftPrBodyRows"] == 1
    assert report["counts"]["pushRows"] == 0
    assert report["counts"]["githubPrMutationRows"] == 0
    assert report["counts"]["readyForMergeRows"] == 0
    assert report["counts"]["readyForReleaseTagRows"] == 0
    assert report["gate"]["publicDefaultPromotionAllowed"] is False
    assert "git push -u origin" in report["operatorCommands"][0]["command"]
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_RELEASE_PACKAGE_HANDOFF_SCHEMA_ID,
        strict=True,
    ).ok


def test_release_package_handoff_blocks_when_release_notes_review_not_ready() -> None:
    release_report = _release_notes_report(status="blocked")
    report = _build(release_report=release_report)

    assert report["status"] == "blocked"
    assert "release_notes_review_not_ready" in report["gate"]["semanticViolations"]


def test_release_package_handoff_blocks_dirty_rows_outside_self_files() -> None:
    report = _build(current_state=_current_state(statusRows=[{"statusCode": "M", "path": "README.md"}]))

    assert report["status"] == "blocked"
    assert "current_worktree_has_blocking_dirty_rows" in report["gate"]["semanticViolations"]


def test_release_package_handoff_allows_self_handoff_dirty_rows() -> None:
    report = _build(
        current_state=_current_state(
            statusRows=[
                {
                    "statusCode": "M",
                    "path": "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_release_package_handoff.v1.json",
                }
            ]
        )
    )

    assert report["status"] == "ready"
    assert report["counts"]["currentDirtyRows"] == 1
    assert report["counts"]["currentBlockingDirtyRows"] == 0


def test_release_package_handoff_blocks_when_branch_is_behind() -> None:
    report = _build(current_state=_current_state(behindCommits=1))

    assert report["status"] == "blocked"
    assert "current_branch_behind_origin_main" in report["gate"]["semanticViolations"]


def test_release_package_handoff_blocks_unsafe_release_notes_counter() -> None:
    release_report = _release_notes_report()
    release_report["counts"]["pushRows"] = 1
    report = _build(release_report=release_report)

    assert report["status"] == "blocked"
    assert "unsafe_counter_nonzero:pushRows" in report["gate"]["semanticViolations"]


def test_release_package_handoff_blocks_private_path_marker() -> None:
    marker = "/" + "Users" + "/example/private"
    report = _build(current_state=_current_state(statusRows=[{"statusCode": "M", "path": marker}]))

    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] == 1
    assert "release_handoff_private_path_marker" in report["gate"]["semanticViolations"]


def test_release_package_handoff_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    report = _build()

    paths = write_parsed_artifact_evidence_chunk_answer_path_v01_rc_release_package_handoff(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )

    parsed = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert parsed["status"] == "ready"
    assert Path(paths["markdown"]).read_text(encoding="utf-8").startswith(
        "# Parsed Artifact Evidence Chunk Answer Path v0.1 RC Release Package Handoff"
    )
    assert validate_payload(
        parsed,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_RELEASE_PACKAGE_HANDOFF_SCHEMA_ID,
        strict=True,
    ).ok
