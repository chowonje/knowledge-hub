"""Release package handoff for the v0.1 parsed-artifact evidence chunk Research Preview."""

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
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_research_preview_release_notes_review import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_RESEARCH_PREVIEW_RELEASE_NOTES_REVIEW_SCHEMA_ID,
    READY_DECISION as RELEASE_NOTES_READY_DECISION,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_RELEASE_PACKAGE_HANDOFF_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-v01-rc-release-package-handoff.v1"
)

READY_DECISION = "knowledge_hub_v01_rc_release_package_handoff_ready"
BLOCKED_DECISION = "knowledge_hub_v01_rc_release_package_handoff_blocked"
NEXT_TRANCHE_READY = "operator_push_and_open_draft_pr_or_corpus_scale_quality_gate"
NEXT_TRANCHE_BLOCKED = "knowledge_hub_v01_rc_release_package_handoff_repair"
DEFAULT_RELEASE_NOTES_REVIEW_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_research_preview_release_notes_review.v1.json"
)
DEFAULT_BRANCH = "codex/v01-rc-public-default-promotion-decision-gate-20260529"
DEFAULT_BASE = "main"
DEFAULT_PR_TITLE = "KnowledgeOS v0.1 RC Research Preview package"

EXTRA_ZERO_COUNTER_FIELDS = (
    "pushRows",
    "githubPrMutationRows",
    "branchDeletionRows",
    "rawGithubPayloadPersistedRows",
    "defaultMcpToolRows",
    "defaultKhubAskRouteRows",
    "releaseTagRows",
    "packagePublishRows",
)

SELF_HANDOFF_ALLOWED_DIRTY_PATHS = {
    "CHANGELOG.md",
    "docs/PROJECT_STATE.md",
    "docs/schemas/paper-parsed-artifact-evidence-chunk-answer-path-v01-rc-release-package-handoff.v1.json",
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_release_package_handoff.v1.json",
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_release_package_handoff.v1.md",
    "eval/knowledgeos/scripts/build_parsed_artifact_evidence_chunk_answer_path_v01_rc_release_package_handoff.py",
    "knowledge_hub/core/schema_validator.py",
    "knowledge_hub/papers/parsed_artifact_evidence_chunk_answer_path_v01_rc_release_package_handoff.py",
    "tests/test_parsed_artifact_evidence_chunk_answer_path_v01_rc_release_package_handoff.py",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _release_notes_review_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    blockers: list[str] = []
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_RESEARCH_PREVIEW_RELEASE_NOTES_REVIEW_SCHEMA_ID:
        blockers.append("release_notes_review_schema_mismatch")
    else:
        validation = validate_payload(
            report,
            PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_RESEARCH_PREVIEW_RELEASE_NOTES_REVIEW_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            blockers.append("release_notes_review_schema_validation_failed")
    if report.get("status") != "ready":
        blockers.append("release_notes_review_not_ready")
    if report.get("decision") != RELEASE_NOTES_READY_DECISION:
        blockers.append("release_notes_review_decision_not_ready")
    if _int(counts.get("releaseNotesReadyRows")) != 1:
        blockers.append("release_notes_ready_rows_not_one")
    if _int(counts.get("releaseNotesPathAllowedRows")) != 1:
        blockers.append("release_notes_path_not_allowed")
    if _int(counts.get("publicDefaultPromotionReadyRows")) != 0:
        blockers.append("public_default_ready_unexpected")
    if _int(counts.get("publicDefaultPromotionHeldRows")) < 1:
        blockers.append("public_default_hold_missing")
    if _int(counts.get("generalRcReadyRows")) != 0:
        blockers.append("general_rc_ready_unexpected")
    if _int(counts.get("corpusScaleClaimProvenRows")) != 0:
        blockers.append("corpus_scale_claim_proven_unexpected")
    if gate.get("publicDefaultPromotionAllowed") is not False:
        blockers.append("public_default_allowed_unexpected")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("release_notes_private_path_leak_reported")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("release_notes_schema_violations_present")
    return sorted(set(blockers))


def _unsafe_counter_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers: list[str] = []
    for field in (*ZERO_COUNTER_FIELDS, *EXTRA_ZERO_COUNTER_FIELDS):
        if _int(counts.get(field)) != 0:
            blockers.append(f"unsafe_counter_nonzero:{field}")
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
    if _int(current_state.get("aheadCommits")) < 2:
        blockers.append("current_branch_missing_release_package_commits")
    if _int(current_state.get("behindCommits")) != 0:
        blockers.append("current_branch_behind_origin_main")
    if _blocking_dirty_rows(current_state):
        blockers.append("current_worktree_has_blocking_dirty_rows")
    if current_state.get("remoteMainVerified") is not True:
        blockers.append("remote_main_not_verified")
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
    if _contains_private_path(github_pr_state):
        blockers.append("github_pr_state_private_path_marker")
    return sorted(set(blockers))


def _body_markdown(*, branch_name: str, base_branch: str, release_notes_report: dict[str, Any]) -> str:
    counts = dict(release_notes_report.get("counts") or {})
    lines = [
        "# KnowledgeOS v0.1 RC Research Preview package",
        "",
        "## Summary",
        "",
        "- Packages the v0.1 RC Research Preview release notes and evidence reports for review.",
        "- Keeps parsed-artifact evidence chunks limited to labs/explicit opt-in surfaces.",
        "- Keeps public/default `khub ask`, default MCP activation, and general RC language held.",
        "",
        "## Scope",
        "",
        f"- Base: `{base_branch}`",
        f"- Head: `{branch_name}`",
        "- Release note: `docs/releases/knowledge-hub-v0.1-rc-research-preview.md`",
        "- Public path: `discover -> index -> search/ask -> evidence review`",
        "",
        "## Gate Evidence",
        "",
        f"- releaseNotesReadyRows: `{_int(counts.get('releaseNotesReadyRows'))}`",
        f"- publicDefaultPromotionReadyRows: `{_int(counts.get('publicDefaultPromotionReadyRows'))}`",
        f"- publicDefaultPromotionHeldRows: `{_int(counts.get('publicDefaultPromotionHeldRows'))}`",
        f"- generalRcReadyRows: `{_int(counts.get('generalRcReadyRows'))}`",
        f"- corpusScaleClaimProvenRows: `{_int(counts.get('corpusScaleClaimProvenRows'))}`",
        f"- privatePathLeakRows: `{_int(counts.get('privatePathLeakRows'))}`",
        f"- schemaViolationCount: `{_int(counts.get('schemaViolationCount'))}`",
        "",
        "## Non-Goals",
        "",
        "- Does not publish a package, create a release tag, or mark a GitHub release.",
        "- Does not make evidence chunks default-on.",
        "- Does not promote default MCP or public `khub ask` behavior.",
        "- Does not claim corpus-scale answer quality.",
        "",
        "## Reviewer Focus",
        "",
        "- Check Research Preview wording and known limits.",
        "- Check public/default surface boundaries.",
        "- Check generated report counters before considering broader release claims.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def build_parsed_artifact_evidence_chunk_answer_path_v01_rc_release_package_handoff(
    *,
    release_notes_review_report_path: str | Path = DEFAULT_RELEASE_NOTES_REVIEW_REPORT,
    release_notes_review_report: dict[str, Any] | None = None,
    current_state: dict[str, Any] | None = None,
    github_pr_state: dict[str, Any] | None = None,
    branch_name: str = DEFAULT_BRANCH,
    base_branch: str = DEFAULT_BASE,
    pr_title: str = DEFAULT_PR_TITLE,
    generated_at: str | None = None,
) -> dict[str, Any]:
    release_report = dict(release_notes_review_report or _read_json(release_notes_review_report_path))
    current = dict(current_state or {})
    pr_state = dict(github_pr_state or {"lookupStatus": "not_performed", "openPrRows": []})
    release_blockers = _release_notes_review_blockers(release_report)
    unsafe_blockers = _unsafe_counter_blockers(release_report)
    current_blockers = _current_state_blockers(current, branch_name=branch_name)
    pr_blockers = _github_pr_state_blockers(pr_state)
    semantic_violations = sorted(set(release_blockers + unsafe_blockers + current_blockers + pr_blockers))
    private_path_leak_rows = 1 if _contains_private_path(release_report) or _contains_private_path(current) or _contains_private_path(pr_state) else 0
    if private_path_leak_rows:
        semantic_violations.append("release_handoff_private_path_marker")
    semantic_violations = sorted(set(semantic_violations))
    status = "ready" if not semantic_violations else "blocked"
    release_counts = dict(release_report.get("counts") or {})
    status_rows = _status_rows(current)
    blocking_dirty_rows = _blocking_dirty_rows(current)
    open_pr_rows = list(pr_state.get("openPrRows") or [])
    body = _body_markdown(branch_name=branch_name, base_branch=base_branch, release_notes_report=release_report)
    counts = {
        "releasePackageHandoffRows": 1,
        "releaseNotesReviewReadyRows": 1 if not release_blockers else 0,
        "releaseNotesReadyRows": _int(release_counts.get("releaseNotesReadyRows")),
        "currentBranchAheadCommitRows": _int(current.get("aheadCommits")),
        "currentBranchBehindCommitRows": _int(current.get("behindCommits")),
        "currentDirtyRows": len(status_rows),
        "currentBlockingDirtyRows": len(blocking_dirty_rows),
        "remoteBranchRows": 1 if current.get("remoteBranchExists") else 0,
        "openPrRows": len(open_pr_rows),
        "draftPrBodyRows": 1 if status == "ready" else 0,
        "operatorCommandRows": 2 if status == "ready" else 0,
        "releasePackageHandoffReadyRows": 1 if status == "ready" else 0,
        "readyForMergeRows": 0,
        "readyForReleaseTagRows": 0,
        "publicDefaultPromotionReadyRows": 0,
        "publicDefaultPromotionHeldRows": _int(release_counts.get("publicDefaultPromotionHeldRows")),
        "generalRcReadyRows": 0,
        "corpusScaleClaimProvenRows": 0,
        "blockedRows": len(semantic_violations),
        **{field: 0 for field in ZERO_COUNTER_FIELDS},
        **{field: 0 for field in EXTRA_ZERO_COUNTER_FIELDS},
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(semantic_violations),
    }
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_RELEASE_PACKAGE_HANDOFF_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "inputs": {
            "releaseNotesReviewReportRef": DEFAULT_RELEASE_NOTES_REVIEW_REPORT.as_posix(),
            "branchName": branch_name,
            "baseBranch": base_branch,
            "headShortSha": _clean_text(current.get("headShortSha")),
            "originMainShortSha": _clean_text(current.get("originMainShortSha")),
            "remoteMainShortSha": _clean_text(current.get("remoteMainShortSha")),
        },
        "policy": {
            "reportOnly": True,
            "operatorHandoffOnly": True,
            "gitReadOnly": True,
            "githubReadOnly": True,
            "pushAllowed": False,
            "prCreationAllowed": False,
            "mergeAllowed": False,
            "releaseTagAllowed": False,
            "packagePublishAllowed": False,
            "publicDefaultPromotionAllowed": False,
            "readyForGeneralRelease": False,
            "rawGithubPayloadPersisted": False,
        },
        "handoffDecision": {
            "releasePackageHandoff": "ready_for_operator_push_and_draft_pr" if status == "ready" else "blocked",
            "publicDefaultDecision": "hold_public_default_promotion",
            "defaultSurfaceDecision": "do_not_enable_default_ask_or_default_mcp",
            "mergeDecision": "not_ready_for_merge",
            "releaseTagDecision": "not_ready_for_release_tag",
            "nextGate": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        },
        "counts": counts,
        "gate": {
            "releasePackageHandoffReady": status == "ready",
            "releaseNotesReviewReady": not release_blockers,
            "currentBranchReady": not current_blockers,
            "githubPrStateReady": not pr_blockers,
            "publicDefaultPromotionAllowed": False,
            "readyForMerge": False,
            "readyForReleaseTag": False,
            "semanticViolations": semantic_violations,
        },
        "draftPr": {
            "title": pr_title,
            "baseBranch": base_branch,
            "headBranch": branch_name,
            "bodyMarkdown": body if status == "ready" else "",
        },
        "operatorCommands": [
            {"commandId": "push_branch", "command": f"git push -u origin {branch_name}"},
            {
                "commandId": "open_draft_pr",
                "command": (
                    "gh pr create --draft --base "
                    f"{base_branch} --head {branch_name} --title \"{pr_title}\" --body-file <prepared-body-file>"
                ),
            },
        ]
        if status == "ready"
        else [],
        "checkRows": [
            {
                "checkId": "release_notes_review",
                "status": "pass" if not release_blockers else "fail",
                "blockers": release_blockers,
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
                "checkId": "github_pr_state",
                "status": "pass" if not pr_blockers else "fail",
                "blockers": pr_blockers,
            },
        ],
        "warnings": [
            "handoff_only_no_push_or_pr_created",
            "research_preview_only_not_general_release",
            "public_default_promotion_remains_held",
            "corpus_scale_quality_gate_required_before_default_promotion",
        ],
    }


def render_parsed_artifact_evidence_chunk_answer_path_v01_rc_release_package_handoff_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    decision = dict(report.get("handoffDecision") or {})
    draft = dict(report.get("draftPr") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Answer Path v0.1 RC Release Package Handoff",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- releasePackageHandoff: `{decision.get('releasePackageHandoff')}`",
        f"- publicDefaultDecision: `{decision.get('publicDefaultDecision')}`",
        f"- mergeDecision: `{decision.get('mergeDecision')}`",
        f"- releaseTagDecision: `{decision.get('releaseTagDecision')}`",
        f"- currentBranchAheadCommitRows: `{counts.get('currentBranchAheadCommitRows')}`",
        f"- currentBranchBehindCommitRows: `{counts.get('currentBranchBehindCommitRows')}`",
        f"- currentBlockingDirtyRows: `{counts.get('currentBlockingDirtyRows')}`",
        f"- openPrRows: `{counts.get('openPrRows')}`",
        f"- draftPrBodyRows: `{counts.get('draftPrBodyRows')}`",
        f"- publicDefaultPromotionReadyRows: `{counts.get('publicDefaultPromotionReadyRows')}`",
        f"- publicDefaultPromotionHeldRows: `{counts.get('publicDefaultPromotionHeldRows')}`",
        f"- readyForMergeRows: `{counts.get('readyForMergeRows')}`",
        f"- readyForReleaseTagRows: `{counts.get('readyForReleaseTagRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Draft PR",
        "",
        f"- title: `{draft.get('title')}`",
        f"- base: `{draft.get('baseBranch')}`",
        f"- head: `{draft.get('headBranch')}`",
        "",
        "## Operator Commands",
        "",
    ]
    for row in list(report.get("operatorCommands") or []):
        lines.append(f"- `{row.get('commandId')}`: `{row.get('command')}`")
    lines.extend(["", "## Checks", ""])
    for row in list(report.get("checkRows") or []):
        blockers = ", ".join(list(row.get("blockers") or [])) or "none"
        lines.append(f"- `{row.get('checkId')}`: `{row.get('status')}`; blockers=`{blockers}`")
    lines.extend(["", "## Mutation Guarantees", ""])
    for field in (*ZERO_COUNTER_FIELDS, *EXTRA_ZERO_COUNTER_FIELDS):
        lines.append(f"- {field}: `{counts.get(field)}`")
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_evidence_chunk_answer_path_v01_rc_release_package_handoff(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_parsed_artifact_evidence_chunk_answer_path_v01_rc_release_package_handoff_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_RELEASE_PACKAGE_HANDOFF_SCHEMA_ID",
    "READY_DECISION",
    "build_parsed_artifact_evidence_chunk_answer_path_v01_rc_release_package_handoff",
    "write_parsed_artifact_evidence_chunk_answer_path_v01_rc_release_package_handoff",
]
