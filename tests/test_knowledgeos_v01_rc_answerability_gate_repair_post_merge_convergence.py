from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.knowledgeos_v01_rc_answerability_gate_repair_post_merge_convergence import (
    BLOCKED_DECISION,
    DEFAULT_CONTROLLED_EXECUTION_REPORT,
    KNOWLEDGEOS_V01_RC_ANSWERABILITY_GATE_REPAIR_POST_MERGE_CONVERGENCE_SCHEMA_ID,
    READY_DECISION,
    build_knowledgeos_v01_rc_answerability_gate_repair_post_merge_convergence,
    write_knowledgeos_v01_rc_answerability_gate_repair_post_merge_convergence,
)


MERGE_SHA = "7c200381a3839446460c8ff6d924c35b26b1d08f"
HEAD_BRANCH = "codex/corpus-scale-answerability-gate-repair-20260530"


def _controlled_report() -> dict[str, Any]:
    return json.loads(Path(DEFAULT_CONTROLLED_EXECUTION_REPORT).read_text(encoding="utf-8"))


def _pr_state(**updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "lookupStatus": "ok",
        "number": 178,
        "title": "Repair corpus-scale evidence chunk answerability gate",
        "url": "https://github.com/chowonje/knowledge-hub/pull/178",
        "state": "MERGED",
        "isDraft": False,
        "mergedAt": "2026-05-29T15:27:40Z",
        "mergeCommit": {"oid": MERGE_SHA},
        "headRefName": HEAD_BRANCH,
        "baseRefName": "main",
        "statusCheckRollup": [
            {
                "__typename": "CheckRun",
                "name": f"check-{index}",
                "workflowName": "CI",
                "status": "COMPLETED",
                "conclusion": "SUCCESS",
            }
            for index in range(7)
        ],
    }
    payload.update(updates)
    return payload


def _git_state(**updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "branchName": "codex/answerability-gate-repair-post-merge-20260530",
        "headSha": MERGE_SHA,
        "headShortSha": MERGE_SHA[:7],
        "originMainSha": MERGE_SHA,
        "originMainShortSha": MERGE_SHA[:7],
        "remoteMainSha": MERGE_SHA,
        "remoteMainShortSha": MERGE_SHA[:7],
        "remoteFeatureBranchSha": "0ac7157331c6878920d3004fa5acaee9d4f410fd",
        "remoteFeatureBranchShortSha": "0ac7157",
        "statusRows": [],
    }
    payload.update(updates)
    return payload


def _build(**updates: Any) -> dict[str, Any]:
    return build_knowledgeos_v01_rc_answerability_gate_repair_post_merge_convergence(
        controlled_execution_report=updates.pop("controlled_execution_report", _controlled_report()),
        git_state=updates.pop("git_state", _git_state()),
        github_pr_state=updates.pop("github_pr_state", _pr_state()),
        release_smoke_result=updates.pop("release_smoke_result", {"status": "ok", "checkedCount": 10, "passedCount": 10}),
        hygiene_result=updates.pop("hygiene_result", {"status": "ok", "issueCount": 0}),
        generated_at="2026-05-30T00:00:00Z",
        **updates,
    )


def test_post_merge_convergence_ready_and_points_to_positive_section_seed() -> None:
    report = _build()

    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["nextRecommendedTranche"] == "corpus_scale_answer_quality_positive_section_paragraph_seed"
    assert report["postMergeDecision"]["controlledExecutionDecision"] == "ready_after_answerability_gate_repair"
    assert report["counts"]["controlledExecutionReadyRows"] == 1
    assert report["counts"]["controlledExecutionPassRows"] == 50
    assert report["counts"]["controlledExecutionFailRows"] == 0
    assert report["counts"]["controlledExecutionUnexpectedAnswerableRows"] == 0
    assert report["counts"]["controlledExecutionNoAnswerSafetyFailRows"] == 0
    assert report["counts"]["prMergedRows"] == 1
    assert report["counts"]["mainMergeCommitMatchRows"] == 1
    assert report["counts"]["ciCheckSuccessRows"] == 7
    assert report["counts"]["answerabilityGateRepairCompleteRows"] == 1
    assert report["counts"]["positiveSectionParagraphSeedRecommendedRows"] == 1
    assert report["counts"]["publicDefaultPromotionHeldRows"] == 1
    assert report["counts"]["branchCleanupRecommendedRows"] == 1
    assert report["counts"]["branchCleanupAppliedRows"] == 0
    assert report["counts"]["privatePathLeakRows"] == 0
    assert report["counts"]["schemaViolationCount"] == 0
    assert validate_payload(
        report,
        KNOWLEDGEOS_V01_RC_ANSWERABILITY_GATE_REPAIR_POST_MERGE_CONVERGENCE_SCHEMA_ID,
        strict=True,
    ).ok


def test_post_merge_convergence_blocks_when_pr_is_not_merged() -> None:
    report = _build(github_pr_state=_pr_state(state="OPEN", mergedAt="", mergeCommit={}))

    assert report["status"] == "blocked"
    assert report["decision"] == BLOCKED_DECISION
    assert "github_pr_not_merged" in report["gate"]["semanticViolations"]


def test_post_merge_convergence_blocks_when_main_does_not_match_merge_commit() -> None:
    report = _build(git_state=_git_state(originMainSha="0" * 40, originMainShortSha="0000000"))

    assert report["status"] == "blocked"
    assert "origin_main_does_not_match_pr_merge_commit" in report["gate"]["semanticViolations"]


def test_post_merge_convergence_blocks_when_controlled_execution_is_not_ready() -> None:
    controlled = _controlled_report()
    controlled["status"] = "blocked"

    report = _build(controlled_execution_report=controlled)

    assert report["status"] == "blocked"
    assert "controlled_execution_not_ready_after_repair" in report["gate"]["semanticViolations"]


def test_post_merge_convergence_blocks_release_smoke_failure() -> None:
    report = _build(release_smoke_result={"status": "failed", "checkedCount": 10, "passedCount": 9})

    assert report["status"] == "blocked"
    assert "release_smoke_not_ok" in report["gate"]["semanticViolations"]
    assert "release_smoke_not_all_passed" in report["gate"]["semanticViolations"]


def test_post_merge_convergence_blocks_public_hygiene_issue() -> None:
    report = _build(hygiene_result={"status": "ok", "issueCount": 1})

    assert report["status"] == "blocked"
    assert "public_hygiene_issues_present" in report["gate"]["semanticViolations"]


def test_post_merge_convergence_blocks_private_path_marker() -> None:
    git_state = _git_state()
    marker = "/" + "Users" + "/example/private"
    git_state["statusRows"] = [{"statusCode": "??", "path": marker}]

    report = _build(git_state=git_state)

    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] == 1
    assert "answerability_gate_repair_post_merge_private_path_marker" in report["gate"]["semanticViolations"]


def test_post_merge_convergence_writer_outputs_schema_valid_report(tmp_path: Path) -> None:
    report = _build()

    paths = write_knowledgeos_v01_rc_answerability_gate_repair_post_merge_convergence(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )

    parsed = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert parsed["status"] == "ready"
    assert markdown.startswith("# KnowledgeOS v0.1 RC Corpus-Scale Answerability Gate Repair Post-Merge Convergence")
    assert validate_payload(
        parsed,
        KNOWLEDGEOS_V01_RC_ANSWERABILITY_GATE_REPAIR_POST_MERGE_CONVERGENCE_SCHEMA_ID,
        strict=True,
    ).ok


def test_post_merge_convergence_keeps_mutation_counters_zero() -> None:
    report = _build()
    counts = report["counts"]

    for field in (
        "githubPrMutationRows",
        "mergeRows",
        "branchDeletionRows",
        "releaseTagRows",
        "packagePublishRows",
        "databaseMutationRows",
        "indexMutationRows",
        "vaultScanRows",
        "externalDownloadRows",
        "defaultOnRows",
    ):
        assert counts[field] == 0


def test_post_merge_convergence_blocks_controlled_execution_semantic_violations() -> None:
    controlled = deepcopy(_controlled_report())
    controlled["counts"]["schemaViolationCount"] = 1

    report = _build(controlled_execution_report=controlled)

    assert report["status"] == "blocked"
    assert report["counts"]["controlledExecutionSemanticViolationRows"] == 1
    assert "controlled_execution_schema_violations_present" in report["gate"]["semanticViolations"]
