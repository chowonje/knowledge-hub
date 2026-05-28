"""Branch and PR readiness review for the v0.1 parsed-artifact evidence chunk RC."""

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
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_LABS_RELEASE_GATE_SCHEMA_ID,
    READY_DECISION as RELEASE_GATE_READY_DECISION,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_BRANCH_PR_READINESS_REVIEW_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-v01-rc-branch-pr-readiness-review.v1"
)

READY_DECISION = "knowledge_hub_v01_rc_branch_pr_readiness_review_ready"
BLOCKED_DECISION = "knowledge_hub_v01_rc_branch_pr_readiness_review_blocked"
NEXT_TRANCHE_READY = "knowledge_hub_v01_rc_draft_pr_opening_or_handoff"
NEXT_TRANCHE_BLOCKED = "knowledge_hub_v01_rc_branch_pr_readiness_repair"
DEFAULT_RELEASE_GATE_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate.v1.json"
)
BASE_REF = "refs/remotes/origin/main"
EXPECTED_BRANCH_PREFIX = "codex/"

SELF_REVIEW_ALLOWED_DIRTY_PATHS = {
    "CHANGELOG.md",
    "docs/PROJECT_STATE.md",
    "docs/schemas/paper-parsed-artifact-evidence-chunk-answer-path-v01-rc-branch-pr-readiness-review.v1.json",
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_branch_pr_readiness_review.v1.json",
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_branch_pr_readiness_review.v1.md",
    "eval/knowledgeos/scripts/build_parsed_artifact_evidence_chunk_answer_path_v01_rc_branch_pr_readiness_review.py",
    "knowledge_hub/core/schema_validator.py",
    "knowledge_hub/papers/parsed_artifact_evidence_chunk_answer_path_v01_rc_branch_pr_readiness_review.py",
    "tests/test_parsed_artifact_evidence_chunk_answer_path_v01_rc_branch_pr_readiness_review.py",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _release_gate_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    blockers: list[str] = []
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_LABS_RELEASE_GATE_SCHEMA_ID:
        blockers.append("release_gate_schema_mismatch")
    if report.get("status") != "ready":
        blockers.append("release_gate_not_ready")
    if report.get("decision") != RELEASE_GATE_READY_DECISION:
        blockers.append("release_gate_decision_not_ready")
    if gate.get("readyForV01RcBranchPrReadinessReview") is not True:
        blockers.append("release_gate_not_ready_for_branch_pr_review")
    if gate.get("publicDefaultPromotionAllowed") is not False:
        blockers.append("release_gate_allowed_public_default")
    if _int(counts.get("releaseGatePassRows")) != 1:
        blockers.append("release_gate_pass_rows_not_one")
    if _int(counts.get("releaseGateBlockedRows")) != 0:
        blockers.append("release_gate_blocked_rows_present")
    if _int(counts.get("publicDefaultPromotionReadyRows")) != 0:
        blockers.append("release_gate_public_default_ready")
    if _int(counts.get("publicDefaultPromotionHeldRows")) < 1:
        blockers.append("release_gate_public_default_hold_missing")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("release_gate_private_path_leak")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("release_gate_schema_violations_present")
    return sorted(set(blockers))


def _status_rows(git_state: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in list(git_state.get("statusRows") or []):
        row = dict(item)
        rows.append(
            {
                "statusCode": _clean_text(row.get("statusCode")),
                "path": _clean_text(row.get("path")),
            }
        )
    return rows


def _blocking_dirty_rows(git_state: dict[str, Any]) -> list[dict[str, str]]:
    return [row for row in _status_rows(git_state) if row.get("path") not in SELF_REVIEW_ALLOWED_DIRTY_PATHS]


def _git_state_blockers(git_state: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    branch = _clean_text(git_state.get("branchName"))
    if not branch.startswith(EXPECTED_BRANCH_PREFIX):
        blockers.append("branch_name_not_codex_prefixed")
    if _clean_text(git_state.get("baseRef")) != BASE_REF:
        blockers.append("base_ref_not_origin_main")
    if _int(git_state.get("aheadCommits")) <= 0:
        blockers.append("branch_has_no_commits_ahead_of_origin_main")
    if _int(git_state.get("behindCommits")) != 0:
        blockers.append("branch_is_behind_origin_main")
    if git_state.get("remoteMainMatchesLocalOriginMain") is not True:
        blockers.append("remote_main_not_verified_against_local_origin_main")
    if _blocking_dirty_rows(git_state):
        blockers.append("blocking_dirty_worktree_rows_present")
    if _contains_private_path(git_state):
        blockers.append("git_state_private_path_marker")
    return sorted(set(blockers))


def _github_pr_blockers(github_pr_state: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if github_pr_state.get("lookupStatus") not in {"ok", "not_performed"}:
        blockers.append("github_pr_lookup_failed")
    open_rows = list(github_pr_state.get("openPrRows") or [])
    if len(open_rows) > 1:
        blockers.append("multiple_open_prs_for_branch")
    for row in open_rows:
        pr = dict(row)
        if _clean_text(pr.get("baseRefName")) not in {"main", ""}:
            blockers.append("open_pr_base_not_main")
        if _clean_text(pr.get("mergeStateStatus")) in {"DIRTY", "UNKNOWN"}:
            blockers.append("open_pr_not_merge_clean")
    if _contains_private_path(github_pr_state):
        blockers.append("github_pr_state_private_path_marker")
    return sorted(set(blockers))


def _commit_rows(git_state: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in list(git_state.get("commitRows") or []):
        item = dict(row)
        rows.append({"shortSha": _clean_text(item.get("shortSha")), "subject": _clean_text(item.get("subject"))})
    return rows


def _changed_path_summary_rows(git_state: dict[str, Any]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for path in list(git_state.get("changedPaths") or []):
        text = _clean_text(path)
        if not text:
            continue
        top = text.split("/", 1)[0]
        counts[top] = counts.get(top, 0) + 1
    return [{"pathPrefix": key, "changedFileRows": counts[key]} for key in sorted(counts)]


def build_parsed_artifact_evidence_chunk_answer_path_v01_rc_branch_pr_readiness_review(
    *,
    release_gate_report_path: str | Path = DEFAULT_RELEASE_GATE_REPORT,
    release_gate_report: dict[str, Any] | None = None,
    git_state: dict[str, Any] | None = None,
    github_pr_state: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    release_report = dict(release_gate_report or _read_json(release_gate_report_path))
    git_payload = dict(git_state or {})
    pr_payload = dict(github_pr_state or {"lookupStatus": "not_performed", "openPrRows": []})
    release_blockers = _release_gate_blockers(release_report)
    git_blockers = _git_state_blockers(git_payload)
    pr_blockers = _github_pr_blockers(pr_payload)
    semantic_violations = sorted(set(release_blockers + git_blockers + pr_blockers))
    status = "ready" if not semantic_violations else "blocked"
    release_counts = dict(release_report.get("counts") or {})
    commit_rows = _commit_rows(git_payload)
    changed_paths = [_clean_text(path) for path in list(git_payload.get("changedPaths") or []) if _clean_text(path)]
    self_dirty_rows = [row for row in _status_rows(git_payload) if row.get("path") in SELF_REVIEW_ALLOWED_DIRTY_PATHS]
    blocking_dirty_rows = _blocking_dirty_rows(git_payload)
    open_pr_rows = list(pr_payload.get("openPrRows") or [])
    counts = {
        "releaseGateReadyRows": 1 if not release_blockers else 0,
        "branchAheadCommitRows": _int(git_payload.get("aheadCommits")),
        "branchBehindCommitRows": _int(git_payload.get("behindCommits")),
        "changedFileRows": len(changed_paths),
        "commitRows": len(commit_rows),
        "worktreeDirtyRows": len(_status_rows(git_payload)),
        "selfReviewDirtyRows": len(self_dirty_rows),
        "blockingDirtyRows": len(blocking_dirty_rows),
        "remoteMainMatchesLocalOriginRows": 1
        if git_payload.get("remoteMainMatchesLocalOriginMain") is True
        else 0,
        "githubPrLookupRows": 1 if pr_payload.get("lookupStatus") == "ok" else 0,
        "openPrRows": len(open_pr_rows),
        "readyForDraftPrRows": 1 if status == "ready" else 0,
        "readyForMergeRows": 0,
        "releaseSmokePassedRows": _int(release_counts.get("releaseSmokePassedRows")),
        "publicHygieneIssueRows": _int(release_counts.get("publicHygieneIssueRows")),
        "noAnswerPassRows": _int(release_counts.get("noAnswerPassRows")),
        "labsSurfaceSmokePassRows": _int(release_counts.get("labsSurfaceSmokePassRows")),
        "publicDefaultPromotionReadyRows": _int(release_counts.get("publicDefaultPromotionReadyRows")),
        "publicDefaultPromotionHeldRows": _int(release_counts.get("publicDefaultPromotionHeldRows")),
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
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_BRANCH_PR_READINESS_REVIEW_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "inputs": {
            "releaseGateReportRef": (
                "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate.v1.json"
            ),
            "baseRef": BASE_REF,
            "branchName": _clean_text(git_payload.get("branchName")),
            "headShortSha": _clean_text(git_payload.get("headShortSha")),
            "baseShortSha": _clean_text(git_payload.get("baseShortSha")),
            "githubPrLookupStatus": _clean_text(pr_payload.get("lookupStatus")),
        },
        "policy": {
            "reportOnly": True,
            "gitReadOnly": True,
            "pushAllowed": False,
            "githubPrMutationAllowed": False,
            "branchDeletionAllowed": False,
            "publicDefaultPromotionAllowed": False,
            "readyForMerge": False,
            "rawGithubPayloadPersisted": False,
        },
        "readinessDecision": {
            "branchPrReadiness": "ready_for_draft_pr_review" if status == "ready" else "blocked",
            "readyForMerge": False,
            "publicDefaultDecision": "hold_public_default_promotion",
            "nextGate": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        },
        "counts": counts,
        "gate": {
            "readyForDraftPrReview": status == "ready",
            "releaseGateReady": not release_blockers,
            "branchCleanOrSelfReviewOnlyDirty": not blocking_dirty_rows,
            "branchAheadOfOriginMain": _int(git_payload.get("aheadCommits")) > 0,
            "branchNotBehindOriginMain": _int(git_payload.get("behindCommits")) == 0,
            "remoteMainVerified": git_payload.get("remoteMainMatchesLocalOriginMain") is True,
            "githubPrStateAcceptable": not pr_blockers,
            "publicDefaultPromotionAllowed": False,
            "readyForMerge": False,
            "semanticViolations": semantic_violations,
        },
        "checkRows": [
            {"checkId": "release_gate", "status": "pass" if not release_blockers else "fail", "blockers": release_blockers},
            {"checkId": "git_branch_state", "status": "pass" if not git_blockers else "fail", "blockers": git_blockers},
            {"checkId": "github_pr_state", "status": "pass" if not pr_blockers else "fail", "blockers": pr_blockers},
        ],
        "changedPathSummaryRows": _changed_path_summary_rows(git_payload),
        "commitRowsDetail": commit_rows,
        "warnings": [
            "this_review_does_not_push_create_pr_or_delete_branches",
            "ready_for_draft_pr_review_is_not_ready_for_merge",
            "public_default_promotion_remains_held",
            "self_review_dirty_rows_are_expected_until_this_review_commit_lands",
        ],
    }


def render_parsed_artifact_evidence_chunk_answer_path_v01_rc_branch_pr_readiness_review_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    readiness = dict(report.get("readinessDecision") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Answer Path v0.1 RC Branch PR Readiness Review",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- branchPrReadiness: `{readiness.get('branchPrReadiness')}`",
        f"- readyForMerge: `{readiness.get('readyForMerge')}`",
        f"- branchAheadCommitRows: `{counts.get('branchAheadCommitRows')}`",
        f"- branchBehindCommitRows: `{counts.get('branchBehindCommitRows')}`",
        f"- changedFileRows: `{counts.get('changedFileRows')}`",
        f"- commitRows: `{counts.get('commitRows')}`",
        f"- blockingDirtyRows: `{counts.get('blockingDirtyRows')}`",
        f"- openPrRows: `{counts.get('openPrRows')}`",
        f"- releaseSmokePassedRows: `{counts.get('releaseSmokePassedRows')}`",
        f"- publicHygieneIssueRows: `{counts.get('publicHygieneIssueRows')}`",
        f"- noAnswerPassRows: `{counts.get('noAnswerPassRows')}`",
        f"- publicDefaultPromotionReadyRows: `{counts.get('publicDefaultPromotionReadyRows')}`",
        f"- publicDefaultPromotionHeldRows: `{counts.get('publicDefaultPromotionHeldRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Checks",
        "",
    ]
    for row in list(report.get("checkRows") or []):
        blockers = ", ".join(list(row.get("blockers") or [])) or "none"
        lines.append(f"- `{row.get('checkId')}`: `{row.get('status')}`; blockers=`{blockers}`")
    lines.extend(["", "## Changed Path Summary", ""])
    for row in list(report.get("changedPathSummaryRows") or []):
        lines.append(f"- `{row.get('pathPrefix')}`: `{row.get('changedFileRows')}`")
    lines.extend(["", "## Mutation Guarantees", ""])
    for field in (*ZERO_COUNTER_FIELDS, "pushRows", "githubPrMutationRows", "branchDeletionRows"):
        lines.append(f"- {field}: `{counts.get(field)}`")
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_evidence_chunk_answer_path_v01_rc_branch_pr_readiness_review(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_parsed_artifact_evidence_chunk_answer_path_v01_rc_branch_pr_readiness_review_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_BRANCH_PR_READINESS_REVIEW_SCHEMA_ID",
    "READY_DECISION",
    "build_parsed_artifact_evidence_chunk_answer_path_v01_rc_branch_pr_readiness_review",
    "write_parsed_artifact_evidence_chunk_answer_path_v01_rc_branch_pr_readiness_review",
]
