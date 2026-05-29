"""Post-merge convergence and cleanup decision for KnowledgeOS v0.1 RC PR #171."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.knowledgeos_v01_rc_vision_bottleneck_definition_review import (
    KNOWLEDGEOS_V01_RC_VISION_BOTTLENECK_DEFINITION_REVIEW_SCHEMA_ID,
    READY_DECISION as VISION_BOTTLENECK_READY_DECISION,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed import (
    ZERO_COUNTER_FIELDS,
    _clean_text,
    _contains_private_path,
    _int,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture import (
    _read_json,
)


KNOWLEDGEOS_V01_RC_POST_MERGE_CONVERGENCE_CLEANUP_DECISION_SCHEMA_ID = (
    "knowledge-hub.product.knowledgeos-v01-rc-post-merge-convergence-cleanup-decision.v1"
)

READY_DECISION = "knowledgeos_v01_rc_post_merge_convergence_cleanup_decision_ready"
BLOCKED_DECISION = "knowledgeos_v01_rc_post_merge_convergence_cleanup_decision_blocked"
NEXT_TRANCHE_READY = "corpus_scale_answer_quality_gate"
NEXT_TRANCHE_BLOCKED = "knowledgeos_v01_rc_post_merge_convergence_cleanup_repair"

DEFAULT_VISION_BOTTLENECK_REPORT = Path(
    "eval/knowledgeos/reports/knowledgeos_v01_rc_vision_bottleneck_definition_review.v1.json"
)
DEFAULT_PR_NUMBER = 171
DEFAULT_HEAD_BRANCH = "codex/v01-rc-public-default-promotion-decision-gate-20260529"
DEFAULT_BASE_BRANCH = "main"

EXTRA_ZERO_COUNTER_FIELDS = (
    "githubPrMutationRows",
    "mergeRows",
    "branchDeletionRows",
    "releaseTagRows",
    "packagePublishRows",
    "rawGithubPayloadPersistedRows",
    "defaultMcpToolRows",
    "defaultKhubAskRouteRows",
)

SELF_REVIEW_ALLOWED_DIRTY_PATHS = {
    "CHANGELOG.md",
    "docs/PROJECT_STATE.md",
    "docs/schemas/knowledgeos-v01-rc-post-merge-convergence-cleanup-decision.v1.json",
    "eval/knowledgeos/reports/knowledgeos_v01_rc_post_merge_convergence_cleanup_decision.v1.json",
    "eval/knowledgeos/reports/knowledgeos_v01_rc_post_merge_convergence_cleanup_decision.v1.md",
    "eval/knowledgeos/scripts/build_knowledgeos_v01_rc_post_merge_convergence_cleanup_decision.py",
    "knowledge_hub/core/schema_validator.py",
    "knowledge_hub/papers/knowledgeos_v01_rc_post_merge_convergence_cleanup_decision.py",
    "tests/test_knowledgeos_v01_rc_post_merge_convergence_cleanup_decision.py",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _schema_blockers(report: dict[str, Any], schema_id: str, prefix: str) -> list[str]:
    if report.get("schema") != schema_id:
        return [f"{prefix}_schema_mismatch"]
    validation = validate_payload(report, schema_id, strict=True)
    if not validation.ok:
        return [f"{prefix}_schema_validation_failed"]
    return []


def _vision_bottleneck_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers = _schema_blockers(
        report,
        KNOWLEDGEOS_V01_RC_VISION_BOTTLENECK_DEFINITION_REVIEW_SCHEMA_ID,
        "vision_bottleneck_definition",
    )
    if report.get("status") != "ready":
        blockers.append("vision_bottleneck_definition_not_ready")
    if report.get("decision") != VISION_BOTTLENECK_READY_DECISION:
        blockers.append("vision_bottleneck_definition_decision_not_ready")
    if _int(counts.get("readyForHumanReviewRows")) != 1:
        blockers.append("vision_ready_for_human_review_missing")
    if _int(counts.get("readyForMergeRows")) != 0:
        blockers.append("vision_ready_for_merge_unexpected")
    if _int(counts.get("publicDefaultPromotionHeldRows")) < 1:
        blockers.append("vision_public_default_hold_missing")
    if _int(counts.get("corpusScaleClaimProvenRows")) != 0:
        blockers.append("vision_corpus_scale_claim_proven_unexpected")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("vision_private_path_leak")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("vision_schema_violations")
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


def _github_pr_blockers(github_pr_state: dict[str, Any], *, pr_number: int, head_branch: str, base_branch: str) -> list[str]:
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
    if _clean_text(github_pr_state.get("headRefName")) != head_branch:
        blockers.append("github_pr_head_branch_mismatch")
    if _clean_text(github_pr_state.get("baseRefName")) != base_branch:
        blockers.append("github_pr_base_not_main")
    check_total, check_success = _count_check_runs(github_pr_state)
    if check_total != 7:
        blockers.append("github_pr_ci_check_count_not_seven")
    if check_success != 7:
        blockers.append("github_pr_ci_checks_not_all_success")
    if _contains_private_path(github_pr_state):
        blockers.append("github_pr_state_private_path_marker")
    return sorted(set(blockers))


def _status_rows(git_state: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in list(git_state.get("statusRows") or []):
        row = dict(item)
        rows.append({"statusCode": _clean_text(row.get("statusCode")), "path": _clean_text(row.get("path"))})
    return rows


def _blocking_dirty_rows(git_state: dict[str, Any]) -> list[dict[str, str]]:
    return [row for row in _status_rows(git_state) if row.get("path") not in SELF_REVIEW_ALLOWED_DIRTY_PATHS]


def _git_state_blockers(git_state: dict[str, Any], *, merge_commit_oid: str) -> list[str]:
    blockers: list[str] = []
    origin_main = _clean_text(git_state.get("originMainSha"))
    remote_main = _clean_text(git_state.get("remoteMainSha"))
    head = _clean_text(git_state.get("headSha"))
    if not origin_main:
        blockers.append("origin_main_sha_missing")
    if not remote_main:
        blockers.append("remote_main_sha_missing")
    if origin_main and remote_main and origin_main != remote_main:
        blockers.append("origin_main_remote_main_mismatch")
    if merge_commit_oid and origin_main and merge_commit_oid != origin_main:
        blockers.append("origin_main_does_not_match_pr_merge_commit")
    if head and origin_main and head != origin_main:
        blockers.append("worktree_head_not_at_origin_main")
    if _blocking_dirty_rows(git_state):
        blockers.append("blocking_dirty_worktree_rows_present")
    if _contains_private_path(git_state):
        blockers.append("git_state_private_path_marker")
    return sorted(set(blockers))


def _release_smoke_blockers(release_smoke_result: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if release_smoke_result.get("status") != "ok":
        blockers.append("release_smoke_not_ok")
    if _int(release_smoke_result.get("checkedCount")) <= 0:
        blockers.append("release_smoke_no_checks")
    if _int(release_smoke_result.get("checkedCount")) != _int(release_smoke_result.get("passedCount")):
        blockers.append("release_smoke_not_all_passed")
    if _contains_private_path(release_smoke_result):
        blockers.append("release_smoke_private_path_marker")
    return sorted(set(blockers))


def _hygiene_blockers(hygiene_result: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if hygiene_result.get("status") != "ok":
        blockers.append("public_hygiene_not_ok")
    if _int(hygiene_result.get("issueCount")) != 0:
        blockers.append("public_hygiene_issues_present")
    if _contains_private_path(hygiene_result):
        blockers.append("public_hygiene_private_path_marker")
    return sorted(set(blockers))


def _unsafe_counter_blockers(*reports: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for index, report in enumerate(reports, start=1):
        counts = dict(report.get("counts") or {})
        for field in (*ZERO_COUNTER_FIELDS, *EXTRA_ZERO_COUNTER_FIELDS):
            if _int(counts.get(field)) != 0:
                blockers.append(f"unsafe_counter_nonzero:report{index}:{field}")
    return sorted(set(blockers))


def build_knowledgeos_v01_rc_post_merge_convergence_cleanup_decision(
    *,
    vision_bottleneck_report_path: str | Path = DEFAULT_VISION_BOTTLENECK_REPORT,
    vision_bottleneck_report: dict[str, Any] | None = None,
    git_state: dict[str, Any] | None = None,
    github_pr_state: dict[str, Any] | None = None,
    release_smoke_result: dict[str, Any] | None = None,
    hygiene_result: dict[str, Any] | None = None,
    pr_number: int = DEFAULT_PR_NUMBER,
    head_branch: str = DEFAULT_HEAD_BRANCH,
    base_branch: str = DEFAULT_BASE_BRANCH,
    generated_at: str | None = None,
) -> dict[str, Any]:
    vision_report = dict(vision_bottleneck_report or _read_json(vision_bottleneck_report_path))
    git_payload = dict(git_state or {})
    pr_payload = dict(github_pr_state or {})
    smoke_payload = dict(release_smoke_result or {})
    hygiene_payload = dict(hygiene_result or {})
    merge_commit_oid = _clean_text(dict(pr_payload.get("mergeCommit") or {}).get("oid"))
    vision_blockers = _vision_bottleneck_blockers(vision_report)
    pr_blockers = _github_pr_blockers(pr_payload, pr_number=pr_number, head_branch=head_branch, base_branch=base_branch)
    git_blockers = _git_state_blockers(git_payload, merge_commit_oid=merge_commit_oid)
    smoke_blockers = _release_smoke_blockers(smoke_payload)
    hygiene_blockers = _hygiene_blockers(hygiene_payload)
    unsafe_blockers = _unsafe_counter_blockers(vision_report)
    semantic_violations = sorted(
        set(vision_blockers + pr_blockers + git_blockers + smoke_blockers + hygiene_blockers + unsafe_blockers)
    )
    private_path_leak_rows = (
        1
        if _contains_private_path(vision_report)
        or _contains_private_path(git_payload)
        or _contains_private_path(pr_payload)
        or _contains_private_path(smoke_payload)
        or _contains_private_path(hygiene_payload)
        else 0
    )
    if private_path_leak_rows:
        semantic_violations.append("post_merge_convergence_cleanup_private_path_marker")
    semantic_violations = sorted(set(semantic_violations))
    status = "ready" if not semantic_violations else "blocked"
    check_total, check_success = _count_check_runs(pr_payload)
    remote_branch_exists = bool(_clean_text(git_payload.get("remoteFeatureBranchSha")))
    counts = {
        "postMergeConvergenceRows": 1,
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
        "releaseSmokeCheckedRows": _int(smoke_payload.get("checkedCount")),
        "releaseSmokePassedRows": _int(smoke_payload.get("passedCount")),
        "publicHygieneIssueRows": _int(hygiene_payload.get("issueCount")),
        "readyForHumanReviewRowsFromVision": _int(dict(vision_report.get("counts") or {}).get("readyForHumanReviewRows")),
        "publicDefaultPromotionReadyRows": 0,
        "publicDefaultPromotionHeldRows": 1,
        "generalRcReadyRows": 0,
        "corpusScaleClaimProvenRows": 0,
        "researchPreviewRcCandidateRows": 1 if status == "ready" else 0,
        "remoteFeatureBranchStillExistsRows": 1 if remote_branch_exists else 0,
        "branchCleanupRecommendedRows": 1 if status == "ready" and remote_branch_exists else 0,
        "branchCleanupAppliedRows": 0,
        "blockedRows": len(semantic_violations),
        **{field: 0 for field in ZERO_COUNTER_FIELDS},
        **{field: 0 for field in EXTRA_ZERO_COUNTER_FIELDS},
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(semantic_violations),
    }
    return {
        "schema": KNOWLEDGEOS_V01_RC_POST_MERGE_CONVERGENCE_CLEANUP_DECISION_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "inputs": {
            "visionBottleneckReportRef": DEFAULT_VISION_BOTTLENECK_REPORT.as_posix(),
            "prNumber": pr_number,
            "headBranch": head_branch,
            "baseBranch": base_branch,
            "headShortSha": _clean_text(git_payload.get("headShortSha")),
            "originMainShortSha": _clean_text(git_payload.get("originMainShortSha")),
            "remoteMainShortSha": _clean_text(git_payload.get("remoteMainShortSha")),
            "mergeCommitShortSha": merge_commit_oid[:7] if merge_commit_oid else "",
        },
        "postMergeDecision": {
            "postMergeConvergence": "research_preview_rc_merged" if status == "ready" else "blocked",
            "branchCleanupDecision": "cleanup_recommended_not_applied"
            if counts["branchCleanupRecommendedRows"]
            else "no_remote_branch_cleanup_needed",
            "publicDefaultDecision": "hold_public_default_promotion",
            "generalRcDecision": "blocked_pending_corpus_scale_quality_gate",
            "nextGate": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        },
        "counts": counts,
        "gate": {
            "postMergeConvergenceReady": status == "ready",
            "prMerged": pr_payload.get("state") == "MERGED",
            "mainContainsMergeCommit": counts["mainMergeCommitMatchRows"] == 1,
            "remoteMainVerified": counts["remoteMainMatchesLocalOriginRows"] == 1,
            "ciChecksGreen": check_total == 7 and check_success == 7,
            "releaseSmokeReady": not smoke_blockers,
            "publicHygieneReady": not hygiene_blockers,
            "branchCleanupRecommended": counts["branchCleanupRecommendedRows"] == 1,
            "branchCleanupApplied": False,
            "publicDefaultPromotionAllowed": False,
            "generalRcReady": False,
            "semanticViolations": semantic_violations,
        },
        "checkRows": [
            {"checkId": "vision_bottleneck_definition", "status": "pass" if not vision_blockers else "fail", "blockers": vision_blockers},
            {"checkId": "github_pr_171", "status": "pass" if not pr_blockers else "fail", "blockers": pr_blockers},
            {"checkId": "git_main_state", "status": "pass" if not git_blockers else "fail", "blockers": git_blockers},
            {"checkId": "release_smoke", "status": "pass" if not smoke_blockers else "fail", "blockers": smoke_blockers},
            {"checkId": "public_hygiene", "status": "pass" if not hygiene_blockers else "fail", "blockers": hygiene_blockers},
            {"checkId": "unsafe_counters", "status": "pass" if not unsafe_blockers else "fail", "blockers": unsafe_blockers},
        ],
        "cleanupRows": [
            {
                "cleanupId": "remote_feature_branch",
                "targetRef": f"origin/{head_branch}",
                "status": "recommended_not_applied" if remote_branch_exists else "not_needed",
                "requiresExplicitApproval": True,
                "summary": "Remote feature branch cleanup is safe to consider after merged main convergence, but this tranche does not delete it.",
            }
        ],
        "warnings": [
            "research_preview_rc_merged_is_not_public_default_promotion",
            "branch_cleanup_not_applied_in_this_tranche",
            "corpus_scale_quality_gate_required_before_default_surface_promotion",
        ],
    }


def render_knowledgeos_v01_rc_post_merge_convergence_cleanup_decision_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    decision = dict(report.get("postMergeDecision") or {})
    lines = [
        "# KnowledgeOS v0.1 RC Post-Merge Convergence Cleanup Decision",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- postMergeConvergence: `{decision.get('postMergeConvergence')}`",
        f"- branchCleanupDecision: `{decision.get('branchCleanupDecision')}`",
        f"- publicDefaultDecision: `{decision.get('publicDefaultDecision')}`",
        f"- prMergedRows: `{counts.get('prMergedRows')}`",
        f"- mainMergeCommitMatchRows: `{counts.get('mainMergeCommitMatchRows')}`",
        f"- ciCheckSuccessRows: `{counts.get('ciCheckSuccessRows')}`",
        f"- releaseSmokePassedRows: `{counts.get('releaseSmokePassedRows')}`",
        f"- publicHygieneIssueRows: `{counts.get('publicHygieneIssueRows')}`",
        f"- remoteFeatureBranchStillExistsRows: `{counts.get('remoteFeatureBranchStillExistsRows')}`",
        f"- branchCleanupRecommendedRows: `{counts.get('branchCleanupRecommendedRows')}`",
        f"- branchCleanupAppliedRows: `{counts.get('branchCleanupAppliedRows')}`",
        f"- publicDefaultPromotionHeldRows: `{counts.get('publicDefaultPromotionHeldRows')}`",
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
    lines.extend(["", "## Cleanup", ""])
    for row in list(report.get("cleanupRows") or []):
        lines.append(
            f"- `{row.get('cleanupId')}` `{row.get('targetRef')}`: `{row.get('status')}`; "
            f"requiresExplicitApproval=`{row.get('requiresExplicitApproval')}`"
        )
    lines.extend(["", "## Mutation Guarantees", ""])
    for field in (*ZERO_COUNTER_FIELDS, *EXTRA_ZERO_COUNTER_FIELDS):
        lines.append(f"- {field}: `{counts.get(field)}`")
    return "\n".join(lines).rstrip() + "\n"


def write_knowledgeos_v01_rc_post_merge_convergence_cleanup_decision(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_knowledgeos_v01_rc_post_merge_convergence_cleanup_decision_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "KNOWLEDGEOS_V01_RC_POST_MERGE_CONVERGENCE_CLEANUP_DECISION_SCHEMA_ID",
    "READY_DECISION",
    "build_knowledgeos_v01_rc_post_merge_convergence_cleanup_decision",
    "write_knowledgeos_v01_rc_post_merge_convergence_cleanup_decision",
]
