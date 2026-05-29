"""Draft PR handoff for the v0.1 parsed-artifact evidence chunk RC."""

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
    READY_DECISION as BRANCH_PR_READY_DECISION,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_DRAFT_PR_OPENING_OR_HANDOFF_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-v01-rc-draft-pr-opening-or-handoff.v1"
)

READY_DECISION = "knowledge_hub_v01_rc_draft_pr_opening_or_handoff_ready"
BLOCKED_DECISION = "knowledge_hub_v01_rc_draft_pr_opening_or_handoff_blocked"
NEXT_TRANCHE_READY = "operator_push_and_open_draft_pr_or_request_codex_pr_creation"
NEXT_TRANCHE_BLOCKED = "knowledge_hub_v01_rc_draft_pr_handoff_repair"
DEFAULT_BRANCH_PR_READINESS_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_branch_pr_readiness_review.v1.json"
)

DEFAULT_BRANCH = "codex/next-implementation-20260528"
DEFAULT_BASE = "main"
DEFAULT_DRAFT_PR_TITLE = "KnowledgeOS v0.1 RC labs evidence chunk preview"
SELF_HANDOFF_ALLOWED_DIRTY_PATHS = {
    "CHANGELOG.md",
    "docs/PROJECT_STATE.md",
    "docs/schemas/paper-parsed-artifact-evidence-chunk-answer-path-v01-rc-draft-pr-opening-or-handoff.v1.json",
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_opening_or_handoff.v1.json",
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_opening_or_handoff.v1.md",
    "eval/knowledgeos/scripts/build_parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_opening_or_handoff.py",
    "knowledge_hub/core/schema_validator.py",
    "knowledge_hub/papers/parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_opening_or_handoff.py",
    "tests/test_parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_opening_or_handoff.py",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _branch_readiness_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    readiness = dict(report.get("readinessDecision") or {})
    blockers: list[str] = []
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_BRANCH_PR_READINESS_REVIEW_SCHEMA_ID:
        blockers.append("branch_pr_readiness_schema_mismatch")
    if report.get("status") != "ready":
        blockers.append("branch_pr_readiness_not_ready")
    if report.get("decision") != BRANCH_PR_READY_DECISION:
        blockers.append("branch_pr_readiness_decision_not_ready")
    if readiness.get("branchPrReadiness") != "ready_for_draft_pr_review":
        blockers.append("branch_pr_not_ready_for_draft_review")
    if gate.get("readyForDraftPrReview") is not True:
        blockers.append("branch_pr_gate_not_ready_for_draft_review")
    if gate.get("readyForMerge") is not False:
        blockers.append("branch_pr_gate_marked_ready_for_merge")
    if gate.get("publicDefaultPromotionAllowed") is not False:
        blockers.append("branch_pr_gate_allowed_public_default")
    if _int(counts.get("readyForDraftPrRows")) != 1:
        blockers.append("branch_pr_ready_for_draft_rows_not_one")
    if _int(counts.get("readyForMergeRows")) != 0:
        blockers.append("branch_pr_ready_for_merge_rows_present")
    if _int(counts.get("blockingDirtyRows")) != 0:
        blockers.append("branch_pr_blocking_dirty_rows_present")
    if _int(counts.get("branchBehindCommitRows")) != 0:
        blockers.append("branch_pr_branch_behind_origin_main")
    if _int(counts.get("publicDefaultPromotionReadyRows")) != 0:
        blockers.append("branch_pr_public_default_ready")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("branch_pr_private_path_leak")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("branch_pr_schema_violations_present")
    return sorted(set(blockers))


def _status_rows(current_state: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in list(current_state.get("statusRows") or []):
        row = dict(item)
        rows.append({"statusCode": _clean_text(row.get("statusCode")), "path": _clean_text(row.get("path"))})
    return rows


def _blocking_dirty_rows(current_state: dict[str, Any]) -> list[dict[str, str]]:
    return [row for row in _status_rows(current_state) if row.get("path") not in SELF_HANDOFF_ALLOWED_DIRTY_PATHS]


def _current_state_blockers(current_state: dict[str, Any], *, branch_name: str) -> list[str]:
    blockers: list[str] = []
    if _clean_text(current_state.get("branchName")) != branch_name:
        blockers.append("current_branch_mismatch")
    if _int(current_state.get("aheadCommits")) <= 0:
        blockers.append("current_branch_not_ahead_of_origin_main")
    if _int(current_state.get("behindCommits")) != 0:
        blockers.append("current_branch_behind_origin_main")
    if _blocking_dirty_rows(current_state):
        blockers.append("current_worktree_has_blocking_dirty_rows")
    if current_state.get("remoteMainVerified") is not True:
        blockers.append("current_remote_main_not_verified")
    if _contains_private_path(current_state):
        blockers.append("current_state_private_path_marker")
    return sorted(set(blockers))


def _github_pr_state_blockers(github_pr_state: dict[str, Any]) -> list[str]:
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


def _draft_pr_body_lines(
    *,
    title: str,
    branch_name: str,
    base_branch: str,
    branch_report: dict[str, Any],
    current_state: dict[str, Any],
) -> list[str]:
    counts = dict(branch_report.get("counts") or {})
    return [
        f"# {title}",
        "",
        "## Summary",
        "",
        "- Defines the KnowledgeOS v0.1 RC product scope as a section/paragraph evidence-first paper QA and comparison Research Preview.",
        "- Adds the parsed-artifact evidence chunk candidate path, opt-in runtime adapter, labs-only CLI/MCP preview surface, quality/user-test reports, and release/readiness gates.",
        "- Keeps public/default `khub ask` and default MCP promotion held; labs activation remains explicit.",
        "",
        "## Scope",
        "",
        f"- Base: `{base_branch}`",
        f"- Head: `{branch_name}`",
        f"- Current branch commits ahead of origin/main: `{_int(current_state.get('aheadCommits'))}`",
        f"- Changed files recorded by readiness review: `{_int(counts.get('changedFileRows'))}`",
        "",
        "## Verification",
        "",
        f"- release smoke passed rows: `{_int(counts.get('releaseSmokePassedRows'))}`",
        f"- public hygiene issue rows: `{_int(counts.get('publicHygieneIssueRows'))}`",
        f"- no-answer pass rows: `{_int(counts.get('noAnswerPassRows'))}`",
        f"- public default promotion ready rows: `{_int(counts.get('publicDefaultPromotionReadyRows'))}`",
        f"- public default promotion held rows: `{_int(counts.get('publicDefaultPromotionHeldRows'))}`",
        "",
        "## Non-Goals",
        "",
        "- Does not make parsed-artifact evidence chunks default-on.",
        "- Does not promote visual/table/equation/figure evidence to public/default answerability.",
        "- Does not claim merge readiness; this is a draft PR review handoff.",
        "",
        "## Reviewer Notes",
        "",
        "- Review public/default surface boundaries first.",
        "- Check schema-backed reports before discussing answer-quality expansion.",
        "- Treat public/default promotion as a later explicit tranche.",
    ]


def build_parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_opening_or_handoff(
    *,
    branch_pr_readiness_report_path: str | Path = DEFAULT_BRANCH_PR_READINESS_REPORT,
    branch_pr_readiness_report: dict[str, Any] | None = None,
    current_state: dict[str, Any] | None = None,
    github_pr_state: dict[str, Any] | None = None,
    branch_name: str = DEFAULT_BRANCH,
    base_branch: str = DEFAULT_BASE,
    draft_pr_title: str = DEFAULT_DRAFT_PR_TITLE,
    generated_at: str | None = None,
) -> dict[str, Any]:
    branch_report = dict(branch_pr_readiness_report or _read_json(branch_pr_readiness_report_path))
    current = dict(current_state or {})
    pr_state = dict(github_pr_state or {"lookupStatus": "not_performed", "openPrRows": []})
    branch_blockers = _branch_readiness_blockers(branch_report)
    current_blockers = _current_state_blockers(current, branch_name=branch_name)
    pr_blockers = _github_pr_state_blockers(pr_state)
    semantic_violations = sorted(set(branch_blockers + current_blockers + pr_blockers))
    status = "ready" if not semantic_violations else "blocked"
    branch_counts = dict(branch_report.get("counts") or {})
    open_pr_rows = list(pr_state.get("openPrRows") or [])
    status_rows = _status_rows(current)
    self_dirty_rows = [row for row in status_rows if row.get("path") in SELF_HANDOFF_ALLOWED_DIRTY_PATHS]
    blocking_dirty_rows = _blocking_dirty_rows(current)
    operator_action = "push_branch_and_open_draft_pr" if not open_pr_rows else "review_existing_draft_pr"
    body_lines = (
        _draft_pr_body_lines(
            title=draft_pr_title,
            branch_name=branch_name,
            base_branch=base_branch,
            branch_report=branch_report,
            current_state=current,
        )
        if status == "ready"
        else []
    )
    counts = {
        "branchReadinessReadyRows": 1 if not branch_blockers else 0,
        "currentBranchAheadCommitRows": _int(current.get("aheadCommits")),
        "currentBranchBehindCommitRows": _int(current.get("behindCommits")),
        "currentDirtyRows": len(status_rows),
        "currentSelfHandoffDirtyRows": len(self_dirty_rows),
        "currentBlockingDirtyRows": len(blocking_dirty_rows),
        "openPrRows": len(open_pr_rows),
        "draftPrHandoffReadyRows": 1 if status == "ready" else 0,
        "draftPrBodyRows": 1 if body_lines else 0,
        "operatorActionRows": 1 if status == "ready" else 0,
        "readyForMergeRows": 0,
        "releaseSmokePassedRows": _int(branch_counts.get("releaseSmokePassedRows")),
        "publicHygieneIssueRows": _int(branch_counts.get("publicHygieneIssueRows")),
        "noAnswerPassRows": _int(branch_counts.get("noAnswerPassRows")),
        "publicDefaultPromotionReadyRows": _int(branch_counts.get("publicDefaultPromotionReadyRows")),
        "publicDefaultPromotionHeldRows": _int(branch_counts.get("publicDefaultPromotionHeldRows")),
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
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_DRAFT_PR_OPENING_OR_HANDOFF_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "inputs": {
            "branchPrReadinessReportRef": (
                "eval/knowledgeos/reports/"
                "parsed_artifact_evidence_chunk_answer_path_v01_rc_branch_pr_readiness_review.v1.json"
            ),
            "branchName": branch_name,
            "baseBranch": base_branch,
            "headShortSha": _clean_text(current.get("headShortSha")),
            "githubPrLookupStatus": _clean_text(pr_state.get("lookupStatus")),
        },
        "policy": {
            "reportOnly": True,
            "pushAllowed": False,
            "githubPrMutationAllowed": False,
            "branchDeletionAllowed": False,
            "publicDefaultPromotionAllowed": False,
            "readyForMerge": False,
            "rawGithubPayloadPersisted": False,
            "requiresExplicitOperatorApproval": True,
        },
        "handoffDecision": {
            "operatorAction": operator_action if status == "ready" else "blocked",
            "readyForDraftPrCreation": status == "ready" and not open_pr_rows,
            "readyForExistingDraftPrReview": status == "ready" and bool(open_pr_rows),
            "readyForMerge": False,
            "publicDefaultDecision": "hold_public_default_promotion",
        },
        "counts": counts,
        "gate": {
            "readyForOperatorPushAndDraftPr": status == "ready",
            "branchReadinessReady": not branch_blockers,
            "currentBranchCleanOrSelfHandoffOnly": not blocking_dirty_rows,
            "currentBranchAheadOfOriginMain": _int(current.get("aheadCommits")) > 0,
            "currentBranchNotBehindOriginMain": _int(current.get("behindCommits")) == 0,
            "githubPrStateAcceptable": not pr_blockers,
            "publicDefaultPromotionAllowed": False,
            "readyForMerge": False,
            "semanticViolations": semantic_violations,
        },
        "draftPr": {
            "title": draft_pr_title,
            "baseBranch": base_branch,
            "headBranch": branch_name,
            "draft": True,
            "bodyMarkdown": "\n".join(body_lines).rstrip() + ("\n" if body_lines else ""),
        },
        "operatorCommands": [
            {
                "commandId": "push_branch",
                "command": f"git push -u origin {branch_name}",
                "requiresExplicitApproval": True,
                "executedByThisReport": False,
            },
            {
                "commandId": "create_draft_pr",
                "command": (
                    f"gh pr create --draft --base {base_branch} --head {branch_name} "
                    f"--title \"{draft_pr_title}\" --body-file <prepared-pr-body.md>"
                ),
                "requiresExplicitApproval": True,
                "executedByThisReport": False,
            },
        ],
        "warnings": [
            "this_handoff_does_not_push_or_create_a_pr",
            "ready_for_draft_pr_creation_is_not_ready_for_merge",
            "public_default_promotion_remains_held",
            "self_handoff_dirty_rows_are_expected_until_this_handoff_commit_lands",
        ],
    }


def render_parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_opening_or_handoff_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    handoff = dict(report.get("handoffDecision") or {})
    draft_pr = dict(report.get("draftPr") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Answer Path v0.1 RC Draft PR Handoff",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- operatorAction: `{handoff.get('operatorAction')}`",
        f"- readyForDraftPrCreation: `{handoff.get('readyForDraftPrCreation')}`",
        f"- readyForMerge: `{handoff.get('readyForMerge')}`",
        f"- currentBranchAheadCommitRows: `{counts.get('currentBranchAheadCommitRows')}`",
        f"- currentBranchBehindCommitRows: `{counts.get('currentBranchBehindCommitRows')}`",
        f"- currentDirtyRows: `{counts.get('currentDirtyRows')}`",
        f"- openPrRows: `{counts.get('openPrRows')}`",
        f"- publicDefaultPromotionReadyRows: `{counts.get('publicDefaultPromotionReadyRows')}`",
        f"- publicDefaultPromotionHeldRows: `{counts.get('publicDefaultPromotionHeldRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Prepared Draft PR Body",
        "",
        draft_pr.get("bodyMarkdown") or "_No body generated because the handoff is blocked._",
        "",
        "## Operator Commands",
        "",
    ]
    for row in list(report.get("operatorCommands") or []):
        lines.append(
            f"- `{row.get('commandId')}`: `{row.get('command')}`; "
            f"executedByThisReport=`{row.get('executedByThisReport')}`"
        )
    lines.extend(["", "## Mutation Guarantees", ""])
    for field in (*ZERO_COUNTER_FIELDS, "pushRows", "githubPrMutationRows", "branchDeletionRows"):
        lines.append(f"- {field}: `{counts.get(field)}`")
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_opening_or_handoff(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_opening_or_handoff_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_DRAFT_PR_OPENING_OR_HANDOFF_SCHEMA_ID",
    "READY_DECISION",
    "build_parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_opening_or_handoff",
    "write_parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_opening_or_handoff",
]
