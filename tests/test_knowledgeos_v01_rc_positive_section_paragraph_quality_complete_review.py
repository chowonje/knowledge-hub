from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.knowledgeos_v01_rc_positive_section_paragraph_quality_complete_review import (
    BLOCKED_DECISION,
    DEFAULT_POSITIVE_EXECUTION_REPORT,
    DEFAULT_POSITIVE_SEED_REPORT,
    DEFAULT_PROVENANCE_REFRESH_REPORT,
    KNOWLEDGEOS_V01_RC_POSITIVE_SECTION_PARAGRAPH_QUALITY_COMPLETE_REVIEW_SCHEMA_ID,
    READY_DECISION,
    build_knowledgeos_v01_rc_positive_section_paragraph_quality_complete_review,
    write_knowledgeos_v01_rc_positive_section_paragraph_quality_complete_review,
)


MERGE_180 = "12530d7473f91372ff4501324c0d757c8f3aeaa0"
MERGE_181 = "2dfc109a15644d9b68bff806449004f7c8ed44c0"
MERGE_182 = "1f74352494a65cd5c0c0ff462aa1fda1cfeee5a7"


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _positive_seed_report() -> dict[str, Any]:
    return _read(DEFAULT_POSITIVE_SEED_REPORT)


def _positive_execution_report() -> dict[str, Any]:
    return _read(DEFAULT_POSITIVE_EXECUTION_REPORT)


def _provenance_report() -> dict[str, Any]:
    return _read(DEFAULT_PROVENANCE_REFRESH_REPORT)


def _check_runs() -> list[dict[str, Any]]:
    return [
        {
            "__typename": "CheckRun",
            "name": f"check-{index}",
            "workflowName": "CI",
            "status": "COMPLETED",
            "conclusion": "SUCCESS",
        }
        for index in range(7)
    ]


def _pr_state(number: int, merge_sha: str, head_branch: str, **updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "lookupStatus": "ok",
        "number": number,
        "title": f"PR {number}",
        "url": f"https://github.com/chowonje/knowledge-hub/pull/{number}",
        "state": "MERGED",
        "isDraft": False,
        "mergedAt": "2026-05-30T03:28:07Z",
        "mergeCommit": {"oid": merge_sha},
        "headRefName": head_branch,
        "baseRefName": "main",
        "statusCheckRollup": _check_runs(),
    }
    payload.update(updates)
    return payload


def _pr_states(**updates: Any) -> list[dict[str, Any]]:
    states = [
        _pr_state(180, MERGE_180, "codex/corpus-scale-answer-quality-positive-section-paragraph-seed-20260530-20260530"),
        _pr_state(181, MERGE_181, "codex/corpus-scale-answer-quality-positive-answer-execution-gate-20260530-20260530"),
        _pr_state(182, MERGE_182, "codex/corpus-scale-answer-quality-positive-provenance-refresh-20260530-20260530"),
    ]
    for index, row_updates in updates.items():
        states[int(index)] = {**states[int(index)], **row_updates}
    return states


def _git_state(**updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "branchName": "codex/quality-complete-review",
        "headSha": MERGE_182,
        "headShortSha": MERGE_182[:7],
        "originMainSha": MERGE_182,
        "originMainShortSha": MERGE_182[:7],
        "remoteMainSha": MERGE_182,
        "remoteMainShortSha": MERGE_182[:7],
        "remoteFeatureBranches": [
            {"headBranch": "branch-180", "sha": "a" * 40, "shortSha": "a" * 7},
            {"headBranch": "branch-181", "sha": "b" * 40, "shortSha": "b" * 7},
            {"headBranch": "branch-182", "sha": "c" * 40, "shortSha": "c" * 7},
        ],
        "statusRows": [],
    }
    payload.update(updates)
    return payload


def _build(**updates: Any) -> dict[str, Any]:
    return build_knowledgeos_v01_rc_positive_section_paragraph_quality_complete_review(
        positive_seed_report=updates.pop("positive_seed_report", _positive_seed_report()),
        positive_execution_report=updates.pop("positive_execution_report", _positive_execution_report()),
        provenance_refresh_report=updates.pop("provenance_refresh_report", _provenance_report()),
        git_state=updates.pop("git_state", _git_state()),
        github_pr_states=updates.pop("github_pr_states", _pr_states()),
        release_smoke_result=updates.pop("release_smoke_result", {"status": "ok", "checkedCount": 10, "passedCount": 10}),
        hygiene_result=updates.pop("hygiene_result", {"status": "ok", "issueCount": 0}),
        generated_at="2026-05-30T00:00:00Z",
        **updates,
    )


def test_complete_review_ready_for_positive_section_paragraph_slice() -> None:
    report = _build()

    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["nextRecommendedTranche"] == "knowledgeos_v01_rc_research_preview_release_readiness_decision_gate"
    assert report["qualityDecision"]["sectionParagraphQuality"] == "complete"
    assert report["counts"]["positiveSeedRows"] == 7
    assert report["counts"]["positiveAnswerPassRows"] == 7
    assert report["counts"]["provenancePassRows"] == 7
    assert report["counts"]["strictProvenanceSpanRows"] == 20
    assert report["counts"]["sourceContentHashRows"] == 20
    assert report["counts"]["charsLocatorRows"] == 20
    assert report["counts"]["answerContractCitationProvenanceRows"] == 20
    assert report["counts"]["heldExpectedNoAnswerRows"] == 17
    assert report["counts"]["heldStructuredModalityRows"] == 34
    assert report["counts"]["prMergedRows"] == 3
    assert report["counts"]["ciCheckSuccessRows"] == 21
    assert report["counts"]["positiveSectionParagraphQualityCompleteRows"] == 1
    assert report["counts"]["corpusScalePositiveSliceProvenRows"] == 1
    assert report["counts"]["corpusScaleClaimProvenRows"] == 0
    assert report["counts"]["publicDefaultPromotionHeldRows"] == 1
    assert report["counts"]["publicDefaultPromotionReadyRows"] == 0
    assert report["counts"]["schemaViolationCount"] == 0
    assert validate_payload(
        report,
        KNOWLEDGEOS_V01_RC_POSITIVE_SECTION_PARAGRAPH_QUALITY_COMPLETE_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_complete_review_blocks_when_seed_report_not_ready() -> None:
    seed = _positive_seed_report()
    seed["status"] = "blocked"

    report = _build(positive_seed_report=seed)

    assert report["status"] == "blocked"
    assert report["decision"] == BLOCKED_DECISION
    assert "positive_seed_not_ready" in report["gate"]["semanticViolations"]


def test_complete_review_blocks_when_positive_execution_has_failures() -> None:
    execution = deepcopy(_positive_execution_report())
    execution["counts"]["positiveAnswerFailRows"] = 1

    report = _build(positive_execution_report=execution)

    assert report["status"] == "blocked"
    assert "positive_execution_fail_rows_present" in report["gate"]["semanticViolations"]


def test_complete_review_blocks_when_provenance_contract_regresses() -> None:
    provenance = deepcopy(_provenance_report())
    provenance["counts"]["charsLocatorRows"] = 19

    report = _build(provenance_refresh_report=provenance)

    assert report["status"] == "blocked"
    assert "provenance_refresh_chars_locator_rows_not_20" in report["gate"]["semanticViolations"]


def test_complete_review_blocks_cross_report_mismatch() -> None:
    provenance = deepcopy(_provenance_report())
    provenance["counts"]["heldExpectedNoAnswerRows"] = 16

    report = _build(provenance_refresh_report=provenance)

    assert report["status"] == "blocked"
    assert "cross_report_heldExpectedNoAnswerRows_mismatch" in report["gate"]["semanticViolations"]


def test_complete_review_blocks_when_quality_pr_not_merged() -> None:
    states = _pr_states(**{"2": {"state": "OPEN", "mergedAt": "", "mergeCommit": {}}})

    report = _build(github_pr_states=states)

    assert report["status"] == "blocked"
    assert "pr_182_not_merged" in report["gate"]["semanticViolations"]


def test_complete_review_blocks_when_main_not_at_latest_quality_merge() -> None:
    report = _build(git_state=_git_state(originMainSha="0" * 40, originMainShortSha="0000000"))

    assert report["status"] == "blocked"
    assert "origin_main_does_not_match_latest_quality_merge_commit" in report["gate"]["semanticViolations"]


def test_complete_review_blocks_hygiene_or_smoke_failures() -> None:
    report = _build(
        release_smoke_result={"status": "failed", "checkedCount": 10, "passedCount": 9},
        hygiene_result={"status": "ok", "issueCount": 1},
    )

    assert report["status"] == "blocked"
    assert "release_smoke_not_ok" in report["gate"]["semanticViolations"]
    assert "public_hygiene_issues_present" in report["gate"]["semanticViolations"]


def test_complete_review_blocks_raw_payload_key_in_provenance_report() -> None:
    provenance = deepcopy(_provenance_report())
    provenance["provenanceRows"][0]["quote"] = "raw quote"

    report = _build(provenance_refresh_report=provenance)

    assert report["status"] == "blocked"
    assert "provenance_refresh_raw_payload_key_present" in report["gate"]["semanticViolations"]


def test_complete_review_blocks_private_path_marker() -> None:
    report = _build(git_state=_git_state(statusRows=[{"statusCode": "??", "path": "/" + "Users" + "/example/private"}]))

    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] == 1
    assert "positive_section_paragraph_quality_complete_private_path_marker" in report["gate"]["semanticViolations"]


def test_complete_review_keeps_mutation_and_default_promotion_counters_zero() -> None:
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
        "defaultMcpToolRows",
        "defaultKhubAskRouteRows",
    ):
        assert counts[field] == 0
    assert counts["publicDefaultPromotionReadyRows"] == 0
    assert counts["publicDefaultPromotionHeldRows"] == 1


def test_complete_review_writer_outputs_schema_valid_report(tmp_path: Path) -> None:
    report = _build()

    paths = write_knowledgeos_v01_rc_positive_section_paragraph_quality_complete_review(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )

    parsed = _read(paths["json"])
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert parsed["status"] == "ready"
    assert "positiveSectionParagraphQualityCompleteRows" in markdown
    assert validate_payload(
        parsed,
        KNOWLEDGEOS_V01_RC_POSITIVE_SECTION_PARAGRAPH_QUALITY_COMPLETE_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok
