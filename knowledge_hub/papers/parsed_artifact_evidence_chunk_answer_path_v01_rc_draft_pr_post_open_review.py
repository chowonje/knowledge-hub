"""Post-open review for the v0.1 RC Research Preview draft PR."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed import (
    ZERO_COUNTER_FIELDS,
    _clean_text,
    _contains_private_path,
    _int,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture import (
    _read_json,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_public_default_promotion_decision_gate import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_PUBLIC_DEFAULT_PROMOTION_DECISION_GATE_SCHEMA_ID,
    READY_DECISION as PUBLIC_DEFAULT_PROMOTION_READY_DECISION,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_release_package_handoff import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_RELEASE_PACKAGE_HANDOFF_SCHEMA_ID,
    READY_DECISION as RELEASE_PACKAGE_HANDOFF_READY_DECISION,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_research_preview_release_notes_review import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_RESEARCH_PREVIEW_RELEASE_NOTES_REVIEW_SCHEMA_ID,
    READY_DECISION as RELEASE_NOTES_REVIEW_READY_DECISION,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_DRAFT_PR_POST_OPEN_REVIEW_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-v01-rc-draft-pr-post-open-review.v1"
)

READY_DECISION = "knowledge_hub_v01_rc_draft_pr_post_open_review_ready"
BLOCKED_DECISION = "knowledge_hub_v01_rc_draft_pr_post_open_review_blocked"
NEXT_TRANCHE_READY = "operator_mark_pr_ready_or_merge_decision_after_review"
NEXT_TRANCHE_BLOCKED = "knowledge_hub_v01_rc_draft_pr_post_open_review_repair"

DEFAULT_RELEASE_PACKAGE_HANDOFF_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_release_package_handoff.v1.json"
)
DEFAULT_PUBLIC_DEFAULT_PROMOTION_GATE_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_public_default_promotion_decision_gate.v1.json"
)
DEFAULT_RELEASE_NOTES_REVIEW_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_research_preview_release_notes_review.v1.json"
)
DEFAULT_PR_NUMBER = 171
DEFAULT_BRANCH = "codex/v01-rc-public-default-promotion-decision-gate-20260529"
DEFAULT_BASE = "main"

EXTRA_ZERO_COUNTER_FIELDS = (
    "pushRows",
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
    "docs/schemas/paper-parsed-artifact-evidence-chunk-answer-path-v01-rc-draft-pr-post-open-review.v1.json",
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_post_open_review.v1.json",
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_post_open_review.v1.md",
    "eval/knowledgeos/scripts/build_parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_post_open_review.py",
    "knowledge_hub/core/schema_validator.py",
    "knowledge_hub/papers/parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_post_open_review.py",
    "tests/test_parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_post_open_review.py",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _schema_blocker(report: dict[str, Any], schema_id: str, prefix: str) -> list[str]:
    if report.get("schema") != schema_id:
        return [f"{prefix}_schema_mismatch"]
    validation = validate_payload(report, schema_id, strict=True)
    if not validation.ok:
        return [f"{prefix}_schema_validation_failed"]
    return []


def _release_package_handoff_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    blockers = _schema_blocker(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_RELEASE_PACKAGE_HANDOFF_SCHEMA_ID,
        "release_package_handoff",
    )
    if report.get("status") != "ready":
        blockers.append("release_package_handoff_not_ready")
    if report.get("decision") != RELEASE_PACKAGE_HANDOFF_READY_DECISION:
        blockers.append("release_package_handoff_decision_not_ready")
    if _int(counts.get("releasePackageHandoffReadyRows")) != 1:
        blockers.append("release_package_handoff_ready_rows_not_one")
    if _int(counts.get("readyForMergeRows")) != 0:
        blockers.append("release_package_handoff_ready_for_merge_unexpected")
    if _int(counts.get("readyForReleaseTagRows")) != 0:
        blockers.append("release_package_handoff_ready_for_release_tag_unexpected")
    if _int(counts.get("publicDefaultPromotionReadyRows")) != 0:
        blockers.append("release_package_handoff_public_default_ready_unexpected")
    if _int(counts.get("publicDefaultPromotionHeldRows")) < 1:
        blockers.append("release_package_handoff_public_default_hold_missing")
    if gate.get("publicDefaultPromotionAllowed") is not False:
        blockers.append("release_package_handoff_public_default_allowed_unexpected")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("release_package_handoff_private_path_leak_reported")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("release_package_handoff_schema_violations_present")
    return sorted(set(blockers))


def _public_default_promotion_gate_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    blockers = _schema_blocker(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_PUBLIC_DEFAULT_PROMOTION_DECISION_GATE_SCHEMA_ID,
        "public_default_promotion_gate",
    )
    if report.get("status") != "ready":
        blockers.append("public_default_promotion_gate_not_ready")
    if report.get("decision") != PUBLIC_DEFAULT_PROMOTION_READY_DECISION:
        blockers.append("public_default_promotion_gate_decision_not_ready")
    if _int(counts.get("publicDefaultPromotionReadyRows")) != 0:
        blockers.append("public_default_promotion_ready_unexpected")
    if _int(counts.get("publicDefaultPromotionHeldRows")) < 1:
        blockers.append("public_default_promotion_hold_missing")
    if _int(counts.get("releaseNotesPathAllowedRows")) != 1:
        blockers.append("release_notes_path_not_allowed")
    if gate.get("publicDefaultPromotionAllowed") is not False:
        blockers.append("public_default_promotion_allowed_unexpected")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("public_default_promotion_gate_private_path_leak_reported")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("public_default_promotion_gate_schema_violations_present")
    return sorted(set(blockers))


def _release_notes_review_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    blockers = _schema_blocker(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_RESEARCH_PREVIEW_RELEASE_NOTES_REVIEW_SCHEMA_ID,
        "release_notes_review",
    )
    if report.get("status") != "ready":
        blockers.append("release_notes_review_not_ready")
    if report.get("decision") != RELEASE_NOTES_REVIEW_READY_DECISION:
        blockers.append("release_notes_review_decision_not_ready")
    if _int(counts.get("releaseNotesReadyRows")) != 1:
        blockers.append("release_notes_ready_rows_not_one")
    if _int(counts.get("publicDefaultPromotionReadyRows")) != 0:
        blockers.append("release_notes_public_default_ready_unexpected")
    if _int(counts.get("publicDefaultPromotionHeldRows")) < 1:
        blockers.append("release_notes_public_default_hold_missing")
    if gate.get("publicDefaultPromotionAllowed") is not False:
        blockers.append("release_notes_public_default_allowed_unexpected")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("release_notes_private_path_leak_reported")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("release_notes_schema_violations_present")
    return sorted(set(blockers))


def _unsafe_counter_blockers(*reports: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for report_index, report in enumerate(reports, start=1):
        counts = dict(report.get("counts") or {})
        for field in (*ZERO_COUNTER_FIELDS, *EXTRA_ZERO_COUNTER_FIELDS):
            if _int(counts.get(field)) != 0:
                blockers.append(f"unsafe_counter_nonzero:report{report_index}:{field}")
    return sorted(set(blockers))


def _status_rows(current_state: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in list(current_state.get("statusRows") or []):
        row = dict(item)
        rows.append({"statusCode": _clean_text(row.get("statusCode")), "path": _clean_text(row.get("path"))})
    return rows


def _blocking_dirty_rows(current_state: dict[str, Any]) -> list[dict[str, str]]:
    return [row for row in _status_rows(current_state) if row.get("path") not in SELF_REVIEW_ALLOWED_DIRTY_PATHS]


def _current_state_blockers(current_state: dict[str, Any], *, branch_name: str) -> list[str]:
    blockers: list[str] = []
    if _clean_text(current_state.get("branchName")) != branch_name:
        blockers.append("current_branch_mismatch")
    if _int(current_state.get("aheadCommits")) < 1:
        blockers.append("current_branch_has_no_pr_commits")
    if _int(current_state.get("behindCommits")) != 0:
        blockers.append("current_branch_behind_origin_main")
    if current_state.get("remoteMainVerified") is not True:
        blockers.append("remote_main_not_verified")
    if not _clean_text(current_state.get("remoteBranchSha")):
        blockers.append("remote_feature_branch_missing")
    if _blocking_dirty_rows(current_state):
        blockers.append("current_worktree_has_blocking_dirty_rows")
    if _contains_private_path(current_state):
        blockers.append("current_state_private_path_marker")
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


def _github_pr_state_blockers(
    github_pr_state: dict[str, Any],
    *,
    pr_number: int,
    branch_name: str,
    base_branch: str,
    remote_branch_sha: str,
) -> list[str]:
    blockers: list[str] = []
    if github_pr_state.get("lookupStatus") != "ok":
        blockers.append("github_pr_lookup_failed")
    if _int(github_pr_state.get("number")) != pr_number:
        blockers.append("github_pr_number_mismatch")
    if github_pr_state.get("state") != "OPEN":
        blockers.append("github_pr_not_open")
    if github_pr_state.get("isDraft") is not True:
        blockers.append("github_pr_not_draft")
    if _clean_text(github_pr_state.get("baseRefName")) != base_branch:
        blockers.append("github_pr_base_not_main")
    if _clean_text(github_pr_state.get("headRefName")) != branch_name:
        blockers.append("github_pr_head_branch_mismatch")
    if _clean_text(github_pr_state.get("mergeStateStatus")) != "CLEAN":
        blockers.append("github_pr_merge_state_not_clean")
    if remote_branch_sha and _clean_text(github_pr_state.get("headRefOid")) != remote_branch_sha:
        blockers.append("github_pr_head_oid_remote_branch_mismatch")
    check_total, check_success = _count_check_runs(github_pr_state)
    if check_total != 7:
        blockers.append("github_pr_ci_check_count_not_seven")
    if check_success != check_total or check_success != 7:
        blockers.append("github_pr_ci_checks_not_all_success")
    if _contains_private_path(github_pr_state):
        blockers.append("github_pr_state_private_path_marker")
    return sorted(set(blockers))


def _stale_handoff_rows(release_package_handoff_report: dict[str, Any]) -> list[dict[str, Any]]:
    counts = dict(release_package_handoff_report.get("counts") or {})
    if _int(counts.get("openPrRows")) != 0:
        return []
    return [
        {
            "artifactRef": DEFAULT_RELEASE_PACKAGE_HANDOFF_REPORT.as_posix(),
            "staleReason": "pre_pr_open_handoff_snapshot",
            "historicalStatus": _clean_text(release_package_handoff_report.get("status")),
            "historicalOpenPrRows": _int(counts.get("openPrRows")),
            "currentAuthority": "live_github_pr_state",
        }
    ]


def build_parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_post_open_review(
    *,
    release_package_handoff_report_path: str | Path = DEFAULT_RELEASE_PACKAGE_HANDOFF_REPORT,
    public_default_promotion_gate_report_path: str | Path = DEFAULT_PUBLIC_DEFAULT_PROMOTION_GATE_REPORT,
    release_notes_review_report_path: str | Path = DEFAULT_RELEASE_NOTES_REVIEW_REPORT,
    release_package_handoff_report: dict[str, Any] | None = None,
    public_default_promotion_gate_report: dict[str, Any] | None = None,
    release_notes_review_report: dict[str, Any] | None = None,
    current_state: dict[str, Any] | None = None,
    github_pr_state: dict[str, Any] | None = None,
    pr_number: int = DEFAULT_PR_NUMBER,
    branch_name: str = DEFAULT_BRANCH,
    base_branch: str = DEFAULT_BASE,
    generated_at: str | None = None,
) -> dict[str, Any]:
    handoff_report = dict(release_package_handoff_report or _read_json(release_package_handoff_report_path))
    promotion_report = dict(public_default_promotion_gate_report or _read_json(public_default_promotion_gate_report_path))
    release_notes_report = dict(release_notes_review_report or _read_json(release_notes_review_report_path))
    current = dict(current_state or {})
    pr_state = dict(github_pr_state or {})
    handoff_blockers = _release_package_handoff_blockers(handoff_report)
    promotion_blockers = _public_default_promotion_gate_blockers(promotion_report)
    release_notes_blockers = _release_notes_review_blockers(release_notes_report)
    unsafe_blockers = _unsafe_counter_blockers(handoff_report, promotion_report, release_notes_report)
    current_blockers = _current_state_blockers(current, branch_name=branch_name)
    remote_branch_sha = _clean_text(current.get("remoteBranchSha"))
    pr_blockers = _github_pr_state_blockers(
        pr_state,
        pr_number=pr_number,
        branch_name=branch_name,
        base_branch=base_branch,
        remote_branch_sha=remote_branch_sha,
    )
    semantic_violations = sorted(
        set(
            handoff_blockers
            + promotion_blockers
            + release_notes_blockers
            + unsafe_blockers
            + current_blockers
            + pr_blockers
        )
    )
    private_path_leak_rows = (
        1
        if _contains_private_path(handoff_report)
        or _contains_private_path(promotion_report)
        or _contains_private_path(release_notes_report)
        or _contains_private_path(current)
        or _contains_private_path(pr_state)
        else 0
    )
    if private_path_leak_rows:
        semantic_violations.append("draft_pr_post_open_review_private_path_marker")
    semantic_violations = sorted(set(semantic_violations))
    status = "ready" if not semantic_violations else "blocked"
    handoff_counts = dict(handoff_report.get("counts") or {})
    promotion_counts = dict(promotion_report.get("counts") or {})
    release_notes_counts = dict(release_notes_report.get("counts") or {})
    status_rows = _status_rows(current)
    blocking_dirty_rows = _blocking_dirty_rows(current)
    check_total, check_success = _count_check_runs(pr_state)
    stale_rows = _stale_handoff_rows(handoff_report)
    pr_is_open = pr_state.get("state") == "OPEN"
    pr_is_draft = pr_state.get("isDraft") is True
    merge_clean = _clean_text(pr_state.get("mergeStateStatus")) == "CLEAN"
    public_default_held = (
        _int(promotion_counts.get("publicDefaultPromotionHeldRows")) >= 1
        and _int(release_notes_counts.get("publicDefaultPromotionHeldRows")) >= 1
        and _int(handoff_counts.get("publicDefaultPromotionHeldRows")) >= 1
    )
    ready_for_human_review = status == "ready" and public_default_held
    counts = {
        "postOpenReviewRows": 1,
        "githubPrLookupRows": 1 if pr_state.get("lookupStatus") == "ok" else 0,
        "openPrRows": 1 if pr_is_open else 0,
        "draftPrRows": 1 if pr_is_draft else 0,
        "baseMainRows": 1 if _clean_text(pr_state.get("baseRefName")) == base_branch else 0,
        "headBranchMatchRows": 1 if _clean_text(pr_state.get("headRefName")) == branch_name else 0,
        "mergeStateCleanRows": 1 if merge_clean else 0,
        "ciCheckRows": check_total,
        "ciCheckSuccessRows": check_success,
        "branchAheadCommitRows": _int(current.get("aheadCommits")),
        "branchBehindCommitRows": _int(current.get("behindCommits")),
        "currentDirtyRows": len(status_rows),
        "currentBlockingDirtyRows": len(blocking_dirty_rows),
        "remoteBranchRows": 1 if remote_branch_sha else 0,
        "releasePackageHandoffReadyRows": 1 if not handoff_blockers else 0,
        "releaseNotesReadyRows": 1 if not release_notes_blockers else 0,
        "publicDefaultPromotionGateReadyRows": 1 if not promotion_blockers else 0,
        "publicDefaultPromotionReadyRows": 0,
        "publicDefaultPromotionHeldRows": 1 if public_default_held else 0,
        "readyForHumanReviewRows": 1 if ready_for_human_review else 0,
        "readyForMergeRows": 0,
        "readyForReleaseTagRows": 0,
        "generalRcReadyRows": 0,
        "corpusScaleClaimProvenRows": 0,
        "stalePrePrHandoffRows": len(stale_rows),
        "blockedRows": len(semantic_violations),
        **{field: 0 for field in ZERO_COUNTER_FIELDS},
        **{field: 0 for field in EXTRA_ZERO_COUNTER_FIELDS},
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(semantic_violations),
    }
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_DRAFT_PR_POST_OPEN_REVIEW_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "inputs": {
            "releasePackageHandoffReportRef": DEFAULT_RELEASE_PACKAGE_HANDOFF_REPORT.as_posix(),
            "publicDefaultPromotionGateReportRef": DEFAULT_PUBLIC_DEFAULT_PROMOTION_GATE_REPORT.as_posix(),
            "releaseNotesReviewReportRef": DEFAULT_RELEASE_NOTES_REVIEW_REPORT.as_posix(),
            "prNumber": pr_number,
            "branchName": branch_name,
            "baseBranch": base_branch,
            "headShortSha": _clean_text(current.get("headShortSha")),
            "originMainShortSha": _clean_text(current.get("originMainShortSha")),
            "remoteMainShortSha": _clean_text(current.get("remoteMainShortSha")),
            "remoteBranchShortSha": _clean_text(current.get("remoteBranchShortSha")),
        },
        "policy": {
            "reportOnly": True,
            "gitReadOnly": True,
            "githubReadOnly": True,
            "githubMutationAllowed": False,
            "markReadyAllowed": False,
            "mergeAllowed": False,
            "branchDeletionAllowed": False,
            "releaseTagAllowed": False,
            "packagePublishAllowed": False,
            "publicDefaultPromotionAllowed": False,
            "readyForGeneralRelease": False,
            "rawGithubPayloadPersisted": False,
        },
        "postOpenReviewDecision": {
            "draftPrPostOpenReview": "ready_for_human_review_not_merge" if ready_for_human_review else "blocked",
            "publicDefaultDecision": "hold_public_default_promotion",
            "defaultSurfaceDecision": "do_not_enable_default_ask_or_default_mcp",
            "mergeDecision": "not_ready_for_merge_in_this_tranche",
            "releaseTagDecision": "not_ready_for_release_tag",
            "nextGate": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        },
        "counts": counts,
        "gate": {
            "postOpenReviewReady": status == "ready",
            "releasePackageHandoffReady": not handoff_blockers,
            "releaseNotesReviewReady": not release_notes_blockers,
            "publicDefaultPromotionGateReady": not promotion_blockers,
            "currentBranchReady": not current_blockers,
            "githubPrStateReady": not pr_blockers,
            "ciChecksGreen": check_total == 7 and check_success == 7,
            "readyForHumanReview": ready_for_human_review,
            "readyForMerge": False,
            "publicDefaultPromotionAllowed": False,
            "generalRcReady": False,
            "semanticViolations": semantic_violations,
        },
        "githubPr": {
            "number": _int(pr_state.get("number")),
            "title": _clean_text(pr_state.get("title")),
            "url": _clean_text(pr_state.get("url")),
            "state": _clean_text(pr_state.get("state")),
            "isDraft": pr_is_draft,
            "mergeStateStatus": _clean_text(pr_state.get("mergeStateStatus")),
            "baseRefName": _clean_text(pr_state.get("baseRefName")),
            "headRefName": _clean_text(pr_state.get("headRefName")),
        },
        "checkRows": [
            {
                "checkId": "release_package_handoff",
                "status": "pass" if not handoff_blockers else "fail",
                "blockers": handoff_blockers,
            },
            {
                "checkId": "release_notes_review",
                "status": "pass" if not release_notes_blockers else "fail",
                "blockers": release_notes_blockers,
            },
            {
                "checkId": "public_default_promotion_gate",
                "status": "pass" if not promotion_blockers else "fail",
                "blockers": promotion_blockers,
            },
            {
                "checkId": "unsafe_counters",
                "status": "pass" if not unsafe_blockers else "fail",
                "blockers": unsafe_blockers,
            },
            {
                "checkId": "current_branch_state",
                "status": "pass" if not current_blockers else "fail",
                "blockers": current_blockers,
            },
            {
                "checkId": "github_pr_171",
                "status": "pass" if not pr_blockers else "fail",
                "blockers": pr_blockers,
            },
        ],
        "ciCheckRows": [
            {
                "name": _clean_text(dict(row).get("name")),
                "workflowName": _clean_text(dict(row).get("workflowName")),
                "status": _clean_text(dict(row).get("status")),
                "conclusion": _clean_text(dict(row).get("conclusion")),
            }
            for row in list(pr_state.get("statusCheckRollup") or [])
            if dict(row).get("__typename") == "CheckRun"
        ],
        "staleArtifactRows": stale_rows,
        "nextActionRows": [
            {
                "actionId": "operator_review_draft_pr",
                "requiresExplicitApproval": True,
                "summary": "Review PR #171 and decide whether to mark ready or keep collecting corpus-scale evidence.",
            },
            {
                "actionId": "merge_decision_separate_tranche",
                "requiresExplicitApproval": True,
                "summary": "Merge, branch deletion, release tags, and package publishing remain out of scope.",
            },
        ]
        if status == "ready"
        else [],
        "warnings": [
            "post_open_review_only_no_pr_mutation",
            "draft_pr_ready_for_human_review_is_not_merge_approval",
            "public_default_promotion_remains_held",
            "corpus_scale_quality_gate_required_before_general_rc_language",
        ],
    }


def render_parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_post_open_review_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    decision = dict(report.get("postOpenReviewDecision") or {})
    pr = dict(report.get("githubPr") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Answer Path v0.1 RC Draft PR Post-Open Review",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- draftPrPostOpenReview: `{decision.get('draftPrPostOpenReview')}`",
        f"- publicDefaultDecision: `{decision.get('publicDefaultDecision')}`",
        f"- mergeDecision: `{decision.get('mergeDecision')}`",
        f"- pr: `#{pr.get('number')}` `{pr.get('state')}` draft=`{pr.get('isDraft')}` mergeState=`{pr.get('mergeStateStatus')}`",
        f"- prUrl: `{pr.get('url')}`",
        f"- branchAheadCommitRows: `{counts.get('branchAheadCommitRows')}`",
        f"- branchBehindCommitRows: `{counts.get('branchBehindCommitRows')}`",
        f"- currentBlockingDirtyRows: `{counts.get('currentBlockingDirtyRows')}`",
        f"- ciCheckRows: `{counts.get('ciCheckRows')}`",
        f"- ciCheckSuccessRows: `{counts.get('ciCheckSuccessRows')}`",
        f"- releasePackageHandoffReadyRows: `{counts.get('releasePackageHandoffReadyRows')}`",
        f"- publicDefaultPromotionHeldRows: `{counts.get('publicDefaultPromotionHeldRows')}`",
        f"- readyForHumanReviewRows: `{counts.get('readyForHumanReviewRows')}`",
        f"- readyForMergeRows: `{counts.get('readyForMergeRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Checks",
        "",
    ]
    for row in list(report.get("checkRows") or []):
        blockers = ", ".join(list(row.get("blockers") or [])) or "none"
        lines.append(f"- `{row.get('checkId')}`: `{row.get('status')}`; blockers=`{blockers}`")
    lines.extend(["", "## CI", ""])
    for row in list(report.get("ciCheckRows") or []):
        lines.append(
            f"- `{row.get('name')}`: status=`{row.get('status')}` conclusion=`{row.get('conclusion')}`"
        )
    lines.extend(["", "## Historical Handoff Snapshot", ""])
    for row in list(report.get("staleArtifactRows") or []):
        lines.append(
            f"- `{row.get('artifactRef')}`: `{row.get('staleReason')}`, "
            f"historicalOpenPrRows=`{row.get('historicalOpenPrRows')}`"
        )
    lines.extend(["", "## Mutation Guarantees", ""])
    for field in (*ZERO_COUNTER_FIELDS, *EXTRA_ZERO_COUNTER_FIELDS):
        lines.append(f"- {field}: `{counts.get(field)}`")
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_post_open_review(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_post_open_review_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_DRAFT_PR_POST_OPEN_REVIEW_SCHEMA_ID",
    "READY_DECISION",
    "build_parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_post_open_review",
    "write_parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_post_open_review",
]
