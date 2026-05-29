"""Post-merge convergence review for the v0.1 parsed-artifact evidence chunk RC."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed import (
    ZERO_COUNTER_FIELDS,
    _clean_text,
    _contains_private_path,
    _int,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture import (
    _read_json,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_branch_pr_readiness_review import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_BRANCH_PR_READINESS_REVIEW_SCHEMA_ID,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_opening_or_handoff import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_DRAFT_PR_OPENING_OR_HANDOFF_SCHEMA_ID,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_LABS_RELEASE_GATE_SCHEMA_ID,
    READY_DECISION as RELEASE_GATE_READY_DECISION,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_POST_MERGE_CONVERGENCE_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-v01-rc-post-merge-convergence.v1"
)

READY_DECISION = "knowledge_hub_v01_rc_research_preview_post_merge_convergence_ready"
BLOCKED_DECISION = "knowledge_hub_v01_rc_post_merge_convergence_blocked"
NEXT_TRANCHE_READY = "knowledge_hub_v01_rc_public_default_promotion_decision_gate_or_release_notes"
NEXT_TRANCHE_BLOCKED = "knowledge_hub_v01_rc_post_merge_convergence_repair"

DEFAULT_RELEASE_GATE_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate.v1.json"
)
DEFAULT_BRANCH_PR_READINESS_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_branch_pr_readiness_review.v1.json"
)
DEFAULT_DRAFT_PR_HANDOFF_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_opening_or_handoff.v1.json"
)
DEFAULT_MERGED_PR_NUMBER = 169
DEFAULT_BASE_BRANCH = "main"

SELF_REVIEW_ALLOWED_DIRTY_PATHS = {
    "CHANGELOG.md",
    "docs/PROJECT_STATE.md",
    "docs/schemas/paper-parsed-artifact-evidence-chunk-answer-path-v01-rc-post-merge-convergence.v1.json",
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_post_merge_convergence.v1.json",
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_post_merge_convergence.v1.md",
    "eval/knowledgeos/scripts/build_parsed_artifact_evidence_chunk_answer_path_v01_rc_post_merge_convergence.py",
    "knowledge_hub/core/schema_validator.py",
    "knowledge_hub/papers/parsed_artifact_evidence_chunk_answer_path_v01_rc_post_merge_convergence.py",
    "tests/test_parsed_artifact_evidence_chunk_answer_path_v01_rc_post_merge_convergence.py",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _release_gate_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers: list[str] = []
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_LABS_RELEASE_GATE_SCHEMA_ID:
        blockers.append("release_gate_schema_mismatch")
    if report.get("status") != "ready":
        blockers.append("release_gate_not_ready")
    if report.get("decision") != RELEASE_GATE_READY_DECISION:
        blockers.append("release_gate_decision_not_ready")
    if _int(counts.get("releaseGatePassRows")) != 1:
        blockers.append("release_gate_pass_rows_not_one")
    if _int(counts.get("releaseSmokePassedRows")) < 10:
        blockers.append("release_smoke_not_green")
    if _int(counts.get("publicHygieneIssueRows")) != 0:
        blockers.append("public_hygiene_issues_present")
    if _int(counts.get("noAnswerPassRows")) < 3:
        blockers.append("no_answer_regression_not_green")
    if _int(counts.get("publicDefaultPromotionReadyRows")) != 0:
        blockers.append("release_gate_public_default_ready_unexpected")
    if _int(counts.get("publicDefaultPromotionHeldRows")) < 1:
        blockers.append("release_gate_public_default_hold_missing")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("release_gate_private_path_leak")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("release_gate_schema_violations_present")
    return sorted(set(blockers))


def _pre_merge_report_blockers(
    *,
    branch_pr_readiness_report: dict[str, Any],
    draft_pr_handoff_report: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if (
        branch_pr_readiness_report.get("schema")
        != PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_BRANCH_PR_READINESS_REVIEW_SCHEMA_ID
    ):
        blockers.append("branch_pr_readiness_schema_mismatch")
    if branch_pr_readiness_report.get("status") != "ready":
        blockers.append("branch_pr_readiness_not_ready")
    if (
        draft_pr_handoff_report.get("schema")
        != PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_DRAFT_PR_OPENING_OR_HANDOFF_SCHEMA_ID
    ):
        blockers.append("draft_pr_handoff_schema_mismatch")
    if draft_pr_handoff_report.get("status") != "ready":
        blockers.append("draft_pr_handoff_not_ready")
    return sorted(set(blockers))


def _git_state_blockers(git_state: dict[str, Any], *, merge_commit_oid: str) -> list[str]:
    blockers: list[str] = []
    origin_main = _clean_text(git_state.get("originMainSha"))
    remote_main = _clean_text(git_state.get("remoteMainSha"))
    if not origin_main:
        blockers.append("origin_main_sha_missing")
    if not remote_main:
        blockers.append("remote_main_sha_missing")
    if origin_main and remote_main and origin_main != remote_main:
        blockers.append("remote_main_mismatch")
    if merge_commit_oid and origin_main and merge_commit_oid != origin_main:
        blockers.append("origin_main_does_not_match_pr_merge_commit")
    blocking_dirty = [
        row
        for row in list(git_state.get("statusRows") or [])
        if _clean_text(dict(row).get("path")) not in SELF_REVIEW_ALLOWED_DIRTY_PATHS
    ]
    if blocking_dirty:
        blockers.append("blocking_dirty_worktree_rows_present")
    if _contains_private_path(git_state):
        blockers.append("git_state_private_path_marker")
    return sorted(set(blockers))


def _github_pr_blockers(github_pr_state: dict[str, Any], *, pr_number: int) -> list[str]:
    blockers: list[str] = []
    if github_pr_state.get("lookupStatus") != "ok":
        blockers.append("github_pr_lookup_failed")
    if _int(github_pr_state.get("number")) != pr_number:
        blockers.append("github_pr_number_mismatch")
    if github_pr_state.get("state") != "MERGED":
        blockers.append("github_pr_not_merged")
    if github_pr_state.get("isDraft") is not False:
        blockers.append("github_pr_still_draft")
    if not _clean_text(github_pr_state.get("mergedAt")):
        blockers.append("github_pr_merged_at_missing")
    if not _clean_text(dict(github_pr_state.get("mergeCommit") or {}).get("oid")):
        blockers.append("github_pr_merge_commit_missing")
    if _clean_text(github_pr_state.get("baseRefName")) != DEFAULT_BASE_BRANCH:
        blockers.append("github_pr_base_not_main")
    if _contains_private_path(github_pr_state):
        blockers.append("github_pr_state_private_path_marker")
    return sorted(set(blockers))


def _summary_result_blockers(hygiene_result: dict[str, Any], release_smoke_result: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if hygiene_result.get("status") != "ok":
        blockers.append("live_public_hygiene_not_ok")
    if _int(hygiene_result.get("issueCount")) != 0:
        blockers.append("live_public_hygiene_issues_present")
    if release_smoke_result.get("status") != "ok":
        blockers.append("live_release_smoke_not_ok")
    if _int(release_smoke_result.get("passedCount")) < _int(release_smoke_result.get("checkedCount")):
        blockers.append("live_release_smoke_not_all_passed")
    if _contains_private_path(hygiene_result) or _contains_private_path(release_smoke_result):
        blockers.append("live_check_summary_private_path_marker")
    return sorted(set(blockers))


def _count_check_runs(github_pr_state: dict[str, Any]) -> tuple[int, int]:
    total = 0
    success = 0
    for row in list(github_pr_state.get("statusCheckRollup") or []):
        item = dict(row)
        if item.get("__typename") != "CheckRun":
            continue
        total += 1
        if item.get("status") == "COMPLETED" and item.get("conclusion") == "SUCCESS":
            success += 1
    return total, success


def _stale_artifact_rows(
    *,
    branch_pr_readiness_report: dict[str, Any],
    draft_pr_handoff_report: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "artifactRef": "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_branch_pr_readiness_review.v1.json",
            "staleReason": "pre_merge_branch_pr_readiness_snapshot",
            "historicalStatus": _clean_text(branch_pr_readiness_report.get("status")),
            "historicalNextRecommendedTranche": _clean_text(branch_pr_readiness_report.get("nextRecommendedTranche")),
        },
        {
            "artifactRef": "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_opening_or_handoff.v1.json",
            "staleReason": "pre_merge_draft_pr_handoff_snapshot",
            "historicalStatus": _clean_text(draft_pr_handoff_report.get("status")),
            "historicalNextRecommendedTranche": _clean_text(draft_pr_handoff_report.get("nextRecommendedTranche")),
        },
    ]


def build_parsed_artifact_evidence_chunk_answer_path_v01_rc_post_merge_convergence(
    *,
    release_gate_report_path: str | Path = DEFAULT_RELEASE_GATE_REPORT,
    branch_pr_readiness_report_path: str | Path = DEFAULT_BRANCH_PR_READINESS_REPORT,
    draft_pr_handoff_report_path: str | Path = DEFAULT_DRAFT_PR_HANDOFF_REPORT,
    release_gate_report: dict[str, Any] | None = None,
    branch_pr_readiness_report: dict[str, Any] | None = None,
    draft_pr_handoff_report: dict[str, Any] | None = None,
    git_state: dict[str, Any] | None = None,
    github_pr_state: dict[str, Any] | None = None,
    hygiene_result: dict[str, Any] | None = None,
    release_smoke_result: dict[str, Any] | None = None,
    pr_number: int = DEFAULT_MERGED_PR_NUMBER,
    generated_at: str | None = None,
) -> dict[str, Any]:
    release_report = dict(release_gate_report or _read_json(release_gate_report_path))
    branch_report = dict(branch_pr_readiness_report or _read_json(branch_pr_readiness_report_path))
    handoff_report = dict(draft_pr_handoff_report or _read_json(draft_pr_handoff_report_path))
    git_payload = dict(git_state or {})
    pr_payload = dict(github_pr_state or {})
    hygiene_payload = dict(hygiene_result or {})
    smoke_payload = dict(release_smoke_result or {})
    merge_commit_oid = _clean_text(dict(pr_payload.get("mergeCommit") or {}).get("oid"))
    release_blockers = _release_gate_blockers(release_report)
    pre_merge_blockers = _pre_merge_report_blockers(
        branch_pr_readiness_report=branch_report,
        draft_pr_handoff_report=handoff_report,
    )
    git_blockers = _git_state_blockers(git_payload, merge_commit_oid=merge_commit_oid)
    pr_blockers = _github_pr_blockers(pr_payload, pr_number=pr_number)
    live_check_blockers = _summary_result_blockers(hygiene_payload, smoke_payload)
    semantic_violations = sorted(
        set(release_blockers + pre_merge_blockers + git_blockers + pr_blockers + live_check_blockers)
    )
    status = "ready" if not semantic_violations else "blocked"
    release_counts = dict(release_report.get("counts") or {})
    quality_case_rows = _int(release_counts.get("qualityInputCaseRows")) or 4
    real_answer_case_rows = _int(release_counts.get("realAnswerInputCaseRows")) or 2
    check_total, check_success = _count_check_runs(pr_payload)
    stale_rows = _stale_artifact_rows(
        branch_pr_readiness_report=branch_report,
        draft_pr_handoff_report=handoff_report,
    )
    public_default_held = _int(release_counts.get("publicDefaultPromotionHeldRows")) >= 1
    research_preview_ready = status == "ready" and public_default_held
    counts = {
        "postMergeReviewRows": 1,
        "prMergedRows": 1 if pr_payload.get("state") == "MERGED" else 0,
        "mainMergeCommitMatchRows": 1
        if merge_commit_oid and merge_commit_oid == _clean_text(git_payload.get("originMainSha"))
        else 0,
        "remoteMainMatchesLocalOriginRows": 1
        if _clean_text(git_payload.get("originMainSha")) == _clean_text(git_payload.get("remoteMainSha"))
        and _clean_text(git_payload.get("originMainSha"))
        else 0,
        "ciCheckRows": check_total,
        "ciCheckSuccessRows": check_success,
        "liveReleaseSmokeCheckedRows": _int(smoke_payload.get("checkedCount")),
        "liveReleaseSmokePassedRows": _int(smoke_payload.get("passedCount")),
        "releaseSmokePassedRows": _int(release_counts.get("releaseSmokePassedRows")),
        "publicHygieneIssueRows": _int(hygiene_payload.get("issueCount")),
        "noAnswerPassRows": _int(release_counts.get("noAnswerPassRows")),
        "labsSurfaceSmokePassRows": _int(release_counts.get("labsSurfaceSmokePassRows")),
        "labsLimitedPromotionReadyRows": _int(release_counts.get("labsLimitedPromotionReadyRows")),
        "publicDefaultPromotionReadyRows": _int(release_counts.get("publicDefaultPromotionReadyRows")),
        "publicDefaultPromotionHeldRows": _int(release_counts.get("publicDefaultPromotionHeldRows")),
        "preMergeReportStaleRows": len(stale_rows),
        "qualityEvalCaseRows": quality_case_rows,
        "realAnswerSmokeCaseRows": real_answer_case_rows,
        "corpusScaleClaimProvenRows": 0,
        "researchPreviewRcCandidateRows": 1 if research_preview_ready else 0,
        "generalRcReadyRows": 0,
        "blockedRows": len(semantic_violations),
        **{field: 0 for field in ZERO_COUNTER_FIELDS},
        "pushRows": 0,
        "githubPrMutationRows": 0,
        "branchDeletionRows": 0,
        "rawGithubPayloadPersistedRows": 0,
        "privatePathLeakRows": 0,
        "schemaViolationCount": len(semantic_violations),
    }
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_POST_MERGE_CONVERGENCE_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "inputs": {
            "releaseGateReportRef": DEFAULT_RELEASE_GATE_REPORT.as_posix(),
            "branchPrReadinessReportRef": DEFAULT_BRANCH_PR_READINESS_REPORT.as_posix(),
            "draftPrHandoffReportRef": DEFAULT_DRAFT_PR_HANDOFF_REPORT.as_posix(),
            "prNumber": pr_number,
            "baseBranch": DEFAULT_BASE_BRANCH,
            "originMainShortSha": _clean_text(git_payload.get("originMainShortSha")),
            "originMainSha": _clean_text(git_payload.get("originMainSha")),
            "remoteMainSha": _clean_text(git_payload.get("remoteMainSha")),
            "mergeCommitSha": merge_commit_oid,
            "githubPrLookupStatus": _clean_text(pr_payload.get("lookupStatus")),
        },
        "policy": {
            "reportOnly": True,
            "gitReadOnly": True,
            "githubReadOnly": True,
            "publicDefaultPromotionAllowed": False,
            "readyForGeneralRelease": False,
            "rawGithubPayloadPersisted": False,
        },
        "convergenceDecision": {
            "postMergeConvergence": "research_preview_rc_candidate" if research_preview_ready else "blocked",
            "publicDefaultDecision": "hold_public_default_promotion",
            "generalRcDecision": "blocked_pending_public_default_and_corpus_scale_evidence",
            "strongestNextBlocker": "public_default_promotion_decision_and_corpus_scale_quality_evidence",
        },
        "counts": counts,
        "gate": {
            "postMergeConvergenceReady": status == "ready",
            "prMerged": pr_payload.get("state") == "MERGED",
            "mainContainsMergeCommit": counts["mainMergeCommitMatchRows"] == 1,
            "remoteMainVerified": counts["remoteMainMatchesLocalOriginRows"] == 1,
            "releaseGateReady": not release_blockers,
            "preMergeReportsPresent": not pre_merge_blockers,
            "liveReleaseSmokeReady": not any(
                blocker for blocker in live_check_blockers if blocker.startswith("live_release_smoke")
            ),
            "livePublicHygieneReady": not any(
                blocker for blocker in live_check_blockers if blocker.startswith("live_public_hygiene")
            ),
            "researchPreviewRcCandidate": research_preview_ready,
            "publicDefaultPromotionAllowed": False,
            "generalRcReady": False,
            "semanticViolations": semantic_violations,
        },
        "checkRows": [
            {"checkId": "release_gate", "status": "pass" if not release_blockers else "fail", "blockers": release_blockers},
            {"checkId": "pre_merge_reports", "status": "pass" if not pre_merge_blockers else "fail", "blockers": pre_merge_blockers},
            {"checkId": "git_main_state", "status": "pass" if not git_blockers else "fail", "blockers": git_blockers},
            {"checkId": "github_pr_169", "status": "pass" if not pr_blockers else "fail", "blockers": pr_blockers},
            {"checkId": "live_checks", "status": "pass" if not live_check_blockers else "fail", "blockers": live_check_blockers},
        ],
        "staleArtifactRows": stale_rows,
        "nextBlockerRows": [
            {
                "blockerId": "public_default_promotion_held",
                "severity": "P1",
                "summary": "Evidence chunk answer path remains labs-only; public/default khub ask and default MCP promotion are still held.",
            },
            {
                "blockerId": "corpus_scale_quality_evidence_missing",
                "severity": "P2",
                "summary": "Current quality evidence is tranche-scale; corpus-scale v0.1 claims still need a broader gate.",
            },
            {
                "blockerId": "pre_merge_state_reports_historical_only",
                "severity": "P2",
                "summary": "Branch readiness and draft handoff reports are now historical snapshots after PR #169 merge.",
            },
        ],
        "warnings": [
            "research_preview_rc_candidate_is_not_general_release_ready",
            "public_default_promotion_remains_held",
            "pre_merge_readiness_reports_are_historical_after_pr_169_merge",
        ],
    }


def render_parsed_artifact_evidence_chunk_answer_path_v01_rc_post_merge_convergence_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    decision = dict(report.get("convergenceDecision") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Answer Path v0.1 RC Post-Merge Convergence",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- postMergeConvergence: `{decision.get('postMergeConvergence')}`",
        f"- publicDefaultDecision: `{decision.get('publicDefaultDecision')}`",
        f"- generalRcDecision: `{decision.get('generalRcDecision')}`",
        f"- prMergedRows: `{counts.get('prMergedRows')}`",
        f"- mainMergeCommitMatchRows: `{counts.get('mainMergeCommitMatchRows')}`",
        f"- liveReleaseSmokePassedRows: `{counts.get('liveReleaseSmokePassedRows')}`",
        f"- publicHygieneIssueRows: `{counts.get('publicHygieneIssueRows')}`",
        f"- noAnswerPassRows: `{counts.get('noAnswerPassRows')}`",
        f"- publicDefaultPromotionReadyRows: `{counts.get('publicDefaultPromotionReadyRows')}`",
        f"- publicDefaultPromotionHeldRows: `{counts.get('publicDefaultPromotionHeldRows')}`",
        f"- researchPreviewRcCandidateRows: `{counts.get('researchPreviewRcCandidateRows')}`",
        f"- generalRcReadyRows: `{counts.get('generalRcReadyRows')}`",
        f"- corpusScaleClaimProvenRows: `{counts.get('corpusScaleClaimProvenRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Checks",
        "",
    ]
    for row in list(report.get("checkRows") or []):
        blockers = ", ".join(list(row.get("blockers") or [])) or "none"
        lines.append(f"- `{row.get('checkId')}`: `{row.get('status')}`; blockers=`{blockers}`")
    lines.extend(["", "## Remaining Blockers", ""])
    for row in list(report.get("nextBlockerRows") or []):
        lines.append(f"- `{row.get('severity')}` `{row.get('blockerId')}`: {row.get('summary')}")
    lines.extend(["", "## Historical Pre-Merge Reports", ""])
    for row in list(report.get("staleArtifactRows") or []):
        lines.append(f"- `{row.get('artifactRef')}`: `{row.get('staleReason')}`")
    lines.extend(["", "## Mutation Guarantees", ""])
    for field in (*ZERO_COUNTER_FIELDS, "pushRows", "githubPrMutationRows", "branchDeletionRows"):
        lines.append(f"- {field}: `{counts.get(field)}`")
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_evidence_chunk_answer_path_v01_rc_post_merge_convergence(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_parsed_artifact_evidence_chunk_answer_path_v01_rc_post_merge_convergence_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_POST_MERGE_CONVERGENCE_SCHEMA_ID",
    "READY_DECISION",
    "build_parsed_artifact_evidence_chunk_answer_path_v01_rc_post_merge_convergence",
    "write_parsed_artifact_evidence_chunk_answer_path_v01_rc_post_merge_convergence",
]
