"""Report-only convergence board for the v0.1 text-evidence roadmap."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Sequence

from knowledge_hub.papers.figure_caption_artifact_vertical_slice import utc_now_iso

TEXT_EVIDENCE_RC_CONVERGENCE_SCHEMA_ID = "knowledge-hub.paper.text-evidence-rc-convergence-report.v1"
PR149_DISPOSITION_REPORT_REF = "text_evidence_pr149_disposition.v1.json"
CANONICAL_DIRTY_INVENTORY_REPORT_REF = "text_evidence_canonical_dirty_inventory.v1.json"
CANONICAL_DIRTY_BUCKET_DECISION_REPORT_REF = "text_evidence_canonical_dirty_bucket_decision.v1.json"

PRIVATE_PATH_TOKENS = (
    "/" + "Users" + "/",
    "/" + "Volumes" + "/",
    "Mobile " + "Documents",
    "i" + "Cloud",
)
PRIVATE_PATH_RE = re.compile("|".join(re.escape(token) for token in PRIVATE_PATH_TOKENS), re.IGNORECASE)

PHASE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "phase": "figure_caption_artifact_vertical_slice",
        "role": "prerequisite",
        "branch": "codex/figure-caption-artifact-vertical-slice-20260526",
        "commit": "b5ffdf5",
        "report": "figure_caption_artifact_vertical_slice.v1.json",
    },
    {
        "phase": "text_evidence_v01_roadmap",
        "role": "scope_decision",
        "branch": "codex/text-evidence-v01-roadmap-20260526",
        "commit": "a4e7f68",
        "report": "",
    },
    {
        "phase": "text_figure_caption_qa_path",
        "role": "roadmap_phase",
        "branch": "codex/text-figure-caption-qa-path-20260526",
        "commit": "9c1cf9a",
        "report": "figure_caption_text_qa_readback.v1.json",
    },
    {
        "phase": "text_section_paragraph_span_artifacts",
        "role": "roadmap_phase",
        "branch": "codex/text-section-paragraph-span-artifacts-20260526",
        "commit": "41679e6",
        "report": "text_section_paragraph_span_artifacts.v1.json",
    },
    {
        "phase": "text_table_caption_candidate_artifacts",
        "role": "roadmap_phase",
        "branch": "codex/text-table-caption-candidate-artifacts-20260526",
        "commit": "f2ebb44",
        "report": "text_table_caption_candidate_artifacts.v1.json",
    },
    {
        "phase": "text_equation_locator_context_artifacts",
        "role": "roadmap_phase",
        "branch": "codex/text-equation-locator-context-artifacts-20260526",
        "commit": "153eb79",
        "report": "text_equation_locator_context_artifacts.v1.json",
    },
    {
        "phase": "text_complex_qa_eval_alignment",
        "role": "roadmap_phase",
        "branch": "codex/text-complex-qa-eval-alignment-20260526",
        "commit": "90585c8",
        "report": "text_complex_qa_eval_alignment.v1.json",
    },
    {
        "phase": "source_alias_normalization",
        "role": "roadmap_phase",
        "branch": "codex/source-alias-normalization-20260526",
        "commit": "ac88f19",
        "report": "source_alias_normalization.v1.json",
    },
    {
        "phase": "public_operator_surface_cleanup",
        "role": "roadmap_phase",
        "branch": "codex/public-operator-surface-cleanup-20260526",
        "commit": "ca4d794",
        "report": "",
    },
)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _private_path_leak_count(payload: dict[str, Any]) -> int:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return 1 if PRIVATE_PATH_RE.search(encoded) else 0


def _run_text(args: Sequence[str], *, cwd: Path | None = None, timeout: int = 8) -> str:
    try:
        result = subprocess.run(
            list(args),
            cwd=str(cwd) if cwd else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _git_text(repo: Path, *args: str) -> str:
    return _run_text(["git", "-C", str(repo), *args])


def _git_is_ancestor(repo: Path, ancestor: str, ref: str = "HEAD") -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", ancestor, ref],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception:
        return False
    return result.returncode == 0


def _status_dirty_count(repo: Path) -> int:
    output = _git_text(repo, "status", "--short")
    if not output:
        return 0
    return len([line for line in output.splitlines() if line.strip()])


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _count_from_report(report: dict[str, Any]) -> dict[str, int]:
    keys = [
        "candidateRows",
        "caseRows",
        "inputRows",
        "rows",
        "answerableRows",
        "noAnswerRows",
        "textAnswerableRows",
        "candidateOnlyRows",
        "visualUnsupportedRows",
        "unsafeDirectAliasRows",
        "expectationFailureRows",
        "privatePathLeakRows",
    ]
    counts: dict[str, int] = {}
    for key in keys:
        value = report.get(key)
        if key == "rows" and isinstance(value, list):
            counts["rowCount"] = len(value)
            continue
        if isinstance(value, int):
            counts[key] = value
    return counts


def _phase_row(spec: dict[str, Any], *, reports_root: Path, repo: Path) -> dict[str, Any]:
    report_name = str(spec.get("report") or "")
    report = _load_json(reports_root / report_name) if report_name else {}
    report_status = _clean_text(report.get("status") or "accepted" if not report_name else report.get("status"))
    commit = _clean_text(spec.get("commit"))
    ancestor_ok = _git_is_ancestor(repo, commit)
    row = {
        "phase": _clean_text(spec.get("phase")),
        "role": _clean_text(spec.get("role")),
        "branch": _clean_text(spec.get("branch")),
        "commit": commit,
        "reportRef": report_name,
        "reportSchema": _clean_text(report.get("schema")),
        "reportStatus": report_status,
        "keyCounts": _count_from_report(report),
        "ancestorOfHead": bool(ancestor_ok),
        "mergeDisposition": "included_in_stacked_branch" if ancestor_ok else "not_in_current_stack",
        "mutationCountersClean": all(int(value or 0) == 0 for value in dict(report.get("mutationCounters") or {}).values()),
        "privatePathLeakRows": int(report.get("privatePathLeakRows") or 0),
    }
    if not report_name:
        row["mutationCountersClean"] = True
        row["privatePathLeakRows"] = 0
    return row


def _current_branch_state(repo: Path) -> dict[str, Any]:
    return {
        "branch": _git_text(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "head": _git_text(repo, "rev-parse", "--short", "HEAD"),
        "dirtyCount": _status_dirty_count(repo),
    }


def _canonical_state(canonical_repo: Path | None) -> dict[str, Any]:
    if canonical_repo is None or not canonical_repo.exists():
        return {"available": False, "branch": "", "head": "", "dirtyCount": 0}
    return {
        "available": True,
        "branch": _git_text(canonical_repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "head": _git_text(canonical_repo, "rev-parse", "--short", "HEAD"),
        "dirtyCount": _status_dirty_count(canonical_repo),
    }


def _pr_149_state(*, include_pr_state: bool, repo: Path) -> dict[str, Any]:
    if not include_pr_state:
        return {"available": False, "number": 149, "state": "not_checked"}
    output = _run_text(
        [
            "gh",
            "pr",
            "view",
            "149",
            "--repo",
            "chowonje/knowledge-hub",
            "--json",
            "number,title,headRefName,baseRefName,isDraft,mergeable,mergeStateStatus,headRefOid,url,updatedAt",
        ],
        cwd=repo,
        timeout=10,
    )
    if not output:
        return {"available": False, "number": 149, "state": "unavailable"}
    try:
        data = json.loads(output)
    except Exception:
        return {"available": False, "number": 149, "state": "parse_failed"}
    mergeable = _clean_text(data.get("mergeable"))
    merge_state = _clean_text(data.get("mergeStateStatus"))
    is_draft = bool(data.get("isDraft"))
    blocked = is_draft or mergeable != "MERGEABLE" or merge_state != "CLEAN"
    return {
        "available": True,
        "number": int(data.get("number") or 149),
        "title": _clean_text(data.get("title")),
        "headRefName": _clean_text(data.get("headRefName")),
        "headRefOid": _clean_text(data.get("headRefOid")),
        "baseRefName": _clean_text(data.get("baseRefName")),
        "isDraft": is_draft,
        "mergeable": mergeable,
        "mergeStateStatus": merge_state,
        "updatedAt": _clean_text(data.get("updatedAt")),
        "url": _clean_text(data.get("url")),
        "blocked": blocked,
        "decision": "recut_or_abandon_before_rc" if blocked else "eligible_for_merge_review",
    }


def _pr149_disposition_state(reports_root: Path) -> dict[str, Any]:
    payload = _load_json(reports_root / PR149_DISPOSITION_REPORT_REF)
    if not payload:
        return {"available": False}
    decision = dict(payload.get("decision") or {})
    return {
        "available": True,
        "reportRef": PR149_DISPOSITION_REPORT_REF,
        "status": _clean_text(payload.get("status")),
        "decision": _clean_text(decision.get("decision")),
        "mergeRecommended": bool(decision.get("mergeRecommended")),
        "recutRecommendedForV01": bool(decision.get("recutRecommendedForV01")),
        "laterSideTrackAllowed": bool(decision.get("laterSideTrackAllowed")),
        "nextAction": _clean_text(payload.get("nextAction")),
        "privatePathLeakRows": int(payload.get("privatePathLeakRows") or 0),
    }


def _canonical_dirty_inventory_state(reports_root: Path) -> dict[str, Any]:
    payload = _load_json(reports_root / CANONICAL_DIRTY_INVENTORY_REPORT_REF)
    if not payload:
        return {"available": False}
    return {
        "available": True,
        "reportRef": CANONICAL_DIRTY_INVENTORY_REPORT_REF,
        "status": _clean_text(payload.get("status")),
        "dirtyRows": int(payload.get("dirtyRows") or 0),
        "unknownRows": int(payload.get("unknownRows") or 0),
        "nextAction": _clean_text(payload.get("nextAction")),
        "privatePathLeakRows": int(payload.get("privatePathLeakRows") or 0),
    }


def _canonical_dirty_bucket_decision_state(reports_root: Path) -> dict[str, Any]:
    payload = _load_json(reports_root / CANONICAL_DIRTY_BUCKET_DECISION_REPORT_REF)
    if not payload:
        return {"available": False}
    decision = dict(payload.get("publicRcDecision") or {})
    return {
        "available": True,
        "reportRef": CANONICAL_DIRTY_BUCKET_DECISION_REPORT_REF,
        "status": _clean_text(payload.get("status")),
        "decision": _clean_text(decision.get("decision")),
        "directIncludeRows": int(payload.get("directIncludeRows") or 0),
        "blockRows": int(payload.get("blockRows") or 0),
        "unknownBucketRows": int(payload.get("unknownBucketRows") or 0),
        "nextAction": _clean_text(payload.get("nextAction")),
        "privatePathLeakRows": int(payload.get("privatePathLeakRows") or 0),
    }


def build_text_evidence_rc_convergence_report(
    *,
    project_root: Path,
    canonical_repo: Path | None = None,
    reports_root: Path | None = None,
    include_pr_state: bool = True,
    generated_at: str | None = None,
) -> dict[str, Any]:
    reports_root = reports_root or project_root / "eval" / "knowledgeos" / "reports"
    phase_rows = [_phase_row(spec, reports_root=reports_root, repo=project_root) for spec in PHASE_SPECS]
    blocker_rows = [
        {
            "blockerId": "canonical_checkout_dirty",
            "severity": "hold",
            "reason": "canonical checkout is dirty and remains evidence-only",
        },
        {
            "blockerId": "pr_149_conflicting_or_draft",
            "severity": "hold",
            "reason": "open strict-evidence audit PR is not clean/mergeable",
        },
    ]
    canonical = _canonical_state(canonical_repo)
    pr_149 = _pr_149_state(include_pr_state=include_pr_state, repo=project_root)
    pr149_disposition = _pr149_disposition_state(reports_root)
    canonical_dirty_inventory = _canonical_dirty_inventory_state(reports_root)
    canonical_dirty_bucket_decision = _canonical_dirty_bucket_decision_state(reports_root)
    if canonical_dirty_inventory.get("available"):
        canonical["dirtyInventoryReportRef"] = _clean_text(canonical_dirty_inventory.get("reportRef"))
        canonical["dirtyInventoryStatus"] = _clean_text(canonical_dirty_inventory.get("status"))
        canonical["dirtyInventoryUnknownRows"] = int(canonical_dirty_inventory.get("unknownRows") or 0)
        canonical["dirtyInventoryNextAction"] = _clean_text(canonical_dirty_inventory.get("nextAction"))
    if canonical_dirty_bucket_decision.get("available"):
        canonical["dirtyBucketDecisionReportRef"] = _clean_text(canonical_dirty_bucket_decision.get("reportRef"))
        canonical["dirtyBucketDecisionStatus"] = _clean_text(canonical_dirty_bucket_decision.get("status"))
        canonical["dirtyBucketDecision"] = _clean_text(canonical_dirty_bucket_decision.get("decision"))
        canonical["dirtyBucketDirectIncludeRows"] = int(canonical_dirty_bucket_decision.get("directIncludeRows") or 0)
        canonical["dirtyBucketNextAction"] = _clean_text(canonical_dirty_bucket_decision.get("nextAction"))
    if pr149_disposition.get("available"):
        pr_149["dispositionReportRef"] = _clean_text(pr149_disposition.get("reportRef"))
        pr_149["dispositionStatus"] = _clean_text(pr149_disposition.get("status"))
        pr_149["dispositionDecision"] = _clean_text(pr149_disposition.get("decision"))
        pr_149["dispositionNextAction"] = _clean_text(pr149_disposition.get("nextAction"))
    if not canonical.get("dirtyCount"):
        blocker_rows = [row for row in blocker_rows if row["blockerId"] != "canonical_checkout_dirty"]
    elif canonical_dirty_bucket_decision.get("decision") == "do_not_merge_canonical_dirty_checkout_into_text_rc":
        for row in blocker_rows:
            if row["blockerId"] == "canonical_checkout_dirty":
                row["reason"] = "canonical dirty buckets are decided for exclusion/hold/replay; physical cleanup remains pending"
    elif canonical_dirty_inventory.get("available"):
        for row in blocker_rows:
            if row["blockerId"] == "canonical_checkout_dirty":
                row["reason"] = "canonical checkout dirty inventory is available; bucket-level keep/drop/replay decision remains pending"
    if not pr_149.get("blocked"):
        blocker_rows = [row for row in blocker_rows if row["blockerId"] != "pr_149_conflicting_or_draft"]
    elif pr149_disposition.get("decision") == "abandon_current_pr_before_public_rc":
        for row in blocker_rows:
            if row["blockerId"] == "pr_149_conflicting_or_draft":
                row["reason"] = "PR #149 has an abandon-before-RC disposition; external PR closure remains pending"

    ready_phase_rows = sum(
        1
        for row in phase_rows
        if row["ancestorOfHead"] and row["reportStatus"] in {"ready", "accepted"} and row["privatePathLeakRows"] == 0
    )
    report: dict[str, Any] = {
        "schema": TEXT_EVIDENCE_RC_CONVERGENCE_SCHEMA_ID,
        "status": "ready_for_integration_review" if ready_phase_rows == len(phase_rows) else "blocked",
        "generatedAt": generated_at or utc_now_iso(),
        "scope": {
            "writes": "report_only",
            "mergePerformed": False,
            "cherryPickPerformed": False,
            "worktreeDeletionPerformed": False,
            "canonicalCheckoutEdited": False,
            "visualLayoutBranchDeferred": True,
            "runtimeAnswerVisibleExposureRows": 0,
            "strictEvidencePromotionRows": 0,
        },
        "currentStack": _current_branch_state(project_root),
        "canonicalCheckout": canonical,
        "pullRequest149": pr_149,
        "phaseRows": len(phase_rows),
        "readyPhaseRows": ready_phase_rows,
        "blockedPhaseRows": len(phase_rows) - ready_phase_rows,
        "publicRcReady": False,
        "publicRcBlockerRows": len(blocker_rows),
        "blockers": blocker_rows,
        "mergeQueue": [
            {
                "order": 1,
                "branch": "codex/rc-hygiene-and-convergence-20260526",
                "head": _git_text(project_root, "rev-parse", "--short", "HEAD"),
                "decision": "review_as_single_stacked_text_evidence_candidate",
            },
            {
                "order": 2,
                "branch": "codex/complex-qa-real-strict-evidence-availability-bridge-audit-20260520",
                "head": _clean_text(pr_149.get("headRefOid")),
                "decision": _clean_text(
                    pr_149.get("dispositionDecision") or pr_149.get("decision") or "hold_until_pr_state_checked"
                ),
            },
        ],
        "phaseReports": phase_rows,
        "mutationCounters": {
            "canonicalParsedArtifactWriteRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "vaultScanRows": 0,
            "externalDownloadRows": 0,
            "mergeRows": 0,
            "cherryPickRows": 0,
            "worktreeDeletionRows": 0,
            "runtimeAnswerVisibleExposureRows": 0,
            "strictEvidencePromotionRows": 0,
        },
        "schemaViolationCount": 0,
        "privatePathLeakRows": 0,
        "reportHash": "",
        "nextAction": _clean_text(
            canonical_dirty_bucket_decision.get("nextAction")
            or canonical_dirty_inventory.get("nextAction")
            or pr149_disposition.get("nextAction")
            or "resolve_canonical_dirty_checkout_and_pr149_before_public_rc"
        ),
        "warnings": [
            "publicRcReady remains false while canonical checkout is dirty or PR #149 is conflicting/draft",
            "visual/layout/VLM work remains deferred outside the v0.1 text-evidence mainline",
        ],
        "schemaErrors": [],
    }
    report["privatePathLeakRows"] = _private_path_leak_count(report)
    if report["privatePathLeakRows"]:
        report["status"] = "blocked"
    payload_for_hash = dict(report)
    payload_for_hash["reportHash"] = ""
    report["reportHash"] = _sha256_json(payload_for_hash)
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Text Evidence RC Convergence",
        "",
        f"- status: `{report.get('status')}`",
        f"- publicRcReady: `{report.get('publicRcReady')}`",
        f"- phaseRows: `{report.get('phaseRows')}`",
        f"- readyPhaseRows: `{report.get('readyPhaseRows')}`",
        f"- blockedPhaseRows: `{report.get('blockedPhaseRows')}`",
        f"- publicRcBlockerRows: `{report.get('publicRcBlockerRows')}`",
        f"- privatePathLeakRows: `{report.get('privatePathLeakRows')}`",
        f"- reportHash: `{report.get('reportHash')}`",
        "",
        "## Blockers",
        "",
        "| blockerId | severity | reason |",
        "|---|---|---|",
    ]
    for row in report.get("blockers", []):
        lines.append(f"| `{row.get('blockerId')}` | `{row.get('severity')}` | {row.get('reason')} |")
    lines.extend(
        [
            "",
            "## Phase Reports",
            "",
            "| phase | commit | reportStatus | disposition |",
            "|---|---:|---|---|",
        ]
    )
    for row in report.get("phaseReports", []):
        lines.append(
            f"| `{row.get('phase')}` | `{row.get('commit')}` | `{row.get('reportStatus')}` | "
            f"`{row.get('mergeDisposition')}` |"
        )
    lines.extend(
        [
            "",
            "## Merge Queue",
            "",
            "| order | branch | decision |",
            "|---:|---|---|",
        ]
    )
    for row in report.get("mergeQueue", []):
        lines.append(f"| `{row.get('order')}` | `{row.get('branch')}` | `{row.get('decision')}` |")
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: dict[str, Any], *, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")


__all__ = [
    "TEXT_EVIDENCE_RC_CONVERGENCE_SCHEMA_ID",
    "build_text_evidence_rc_convergence_report",
    "render_markdown_report",
    "write_report",
]
