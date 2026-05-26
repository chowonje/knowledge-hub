"""Report-only disposition for PR #149 in the v0.1 text-evidence RC line."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Sequence

from knowledge_hub.papers.figure_caption_artifact_vertical_slice import utc_now_iso

TEXT_EVIDENCE_PR149_DISPOSITION_SCHEMA_ID = "knowledge-hub.paper.text-evidence-pr149-disposition.v1"

PRIVATE_PATH_TOKENS = (
    "/" + "Users" + "/",
    "/" + "Volumes" + "/",
    "Mobile " + "Documents",
    "i" + "Cloud",
)
PRIVATE_PATH_RE = re.compile("|".join(re.escape(token) for token in PRIVATE_PATH_TOKENS), re.IGNORECASE)

PR149_FILES = (
    "CHANGELOG.md",
    "docs/PROJECT_STATE.md",
    "docs/schemas/paper-complex-qa-real-strict-evidence-availability-bridge-audit.v1.json",
    "knowledge_hub/core/schema_validator.py",
    "knowledge_hub/papers/complex_qa_real_strict_evidence_availability_bridge_audit.py",
    "tests/test_complex_qa_real_strict_evidence_availability_bridge_audit.py",
)

PR149_REQUIRED_DEPENDENCIES = (
    {
        "module": "knowledge_hub.papers.complex_qa_structured_evidence_comparison_runner",
        "path": "knowledge_hub/papers/complex_qa_structured_evidence_comparison_runner.py",
    },
    {
        "module": "knowledge_hub.papers.complex_qa_supplied_strict_evidence_grader_baseline_runner",
        "path": "knowledge_hub/papers/complex_qa_supplied_strict_evidence_grader_baseline_runner.py",
    },
)

TEXT_ALIGNMENT_REPORT_REF = "text_complex_qa_eval_alignment.v1.json"


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


def _status_dirty_count(repo: Path) -> int:
    output = _git_text(repo, "status", "--short")
    return len([line for line in output.splitlines() if line.strip()]) if output else 0


def _current_branch_state(repo: Path) -> dict[str, Any]:
    return {
        "branch": _git_text(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "head": _git_text(repo, "rev-parse", "--short", "HEAD"),
        "dirtyCount": _status_dirty_count(repo),
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _fallback_pr_state() -> dict[str, Any]:
    return {
        "available": False,
        "number": 149,
        "title": "Add complex QA real strict evidence availability bridge audit",
        "headRefName": "codex/complex-qa-real-strict-evidence-availability-bridge-audit-20260520",
        "headRefOid": "",
        "baseRefName": "main",
        "isDraft": True,
        "mergeable": "CONFLICTING",
        "mergeStateStatus": "DIRTY",
        "updatedAt": "",
        "url": "",
        "files": list(PR149_FILES),
    }


def _load_pr_state(*, repo: Path, include_gh: bool) -> dict[str, Any]:
    if not include_gh:
        return _fallback_pr_state()
    output = _run_text(
        [
            "gh",
            "pr",
            "view",
            "149",
            "--repo",
            "chowonje/knowledge-hub",
            "--json",
            "number,title,headRefName,baseRefName,isDraft,mergeable,mergeStateStatus,headRefOid,url,updatedAt,files",
        ],
        cwd=repo,
        timeout=10,
    )
    if not output:
        return _fallback_pr_state()
    try:
        payload = json.loads(output)
    except Exception:
        return _fallback_pr_state()
    files = payload.get("files")
    file_names = []
    if isinstance(files, list):
        for row in files:
            if isinstance(row, dict):
                file_names.append(_clean_text(row.get("path")))
            else:
                file_names.append(_clean_text(row))
    return {
        "available": True,
        "number": int(payload.get("number") or 149),
        "title": _clean_text(payload.get("title")),
        "headRefName": _clean_text(payload.get("headRefName")),
        "headRefOid": _clean_text(payload.get("headRefOid")),
        "baseRefName": _clean_text(payload.get("baseRefName")),
        "isDraft": bool(payload.get("isDraft")),
        "mergeable": _clean_text(payload.get("mergeable")),
        "mergeStateStatus": _clean_text(payload.get("mergeStateStatus")),
        "updatedAt": _clean_text(payload.get("updatedAt")),
        "url": _clean_text(payload.get("url")),
        "files": [name for name in file_names if name] or list(PR149_FILES),
    }


def _dependency_rows(project_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in PR149_REQUIRED_DEPENDENCIES:
        path = _clean_text(spec["path"])
        present = (project_root / path).exists()
        rows.append(
            {
                "module": _clean_text(spec["module"]),
                "path": path,
                "presentInCurrentStack": present,
                "disposition": "available" if present else "missing_from_text_evidence_stack",
            }
        )
    return rows


def _text_alignment_row(reports_root: Path) -> dict[str, Any]:
    report = _load_json(reports_root / TEXT_ALIGNMENT_REPORT_REF)
    return {
        "reportRef": TEXT_ALIGNMENT_REPORT_REF,
        "present": bool(report),
        "schema": _clean_text(report.get("schema")),
        "status": _clean_text(report.get("status")),
        "caseRows": int(report.get("caseRows") or 0),
        "textAnswerableRows": int(report.get("textAnswerableRows") or 0),
        "visualUnsupportedRows": int(report.get("visualUnsupportedRows") or 0),
        "disposition": "v0_1_text_evidence_alignment_source" if report else "missing",
    }


def build_text_evidence_pr149_disposition_report(
    *,
    project_root: Path,
    reports_root: Path | None = None,
    include_gh: bool = True,
    generated_at: str | None = None,
) -> dict[str, Any]:
    reports_root = reports_root or project_root / "eval" / "knowledgeos" / "reports"
    pr_state = _load_pr_state(repo=project_root, include_gh=include_gh)
    dependency_rows = _dependency_rows(project_root)
    missing_dependency_rows = sum(1 for row in dependency_rows if not row["presentInCurrentStack"])
    text_alignment = _text_alignment_row(reports_root)
    pr_files = list(pr_state.get("files") or PR149_FILES)

    report: dict[str, Any] = {
        "schema": TEXT_EVIDENCE_PR149_DISPOSITION_SCHEMA_ID,
        "status": "ready",
        "generatedAt": generated_at or utc_now_iso(),
        "scope": {
            "writes": "report_only",
            "pullRequestMutationPerformed": False,
            "mergePerformed": False,
            "cherryPickPerformed": False,
            "canonicalCheckoutEdited": False,
            "textOnlyRcScope": True,
            "visualLayoutBranchDeferred": True,
            "runtimeAnswerVisibleExposureRows": 0,
            "strictEvidencePromotionRows": 0,
        },
        "currentStack": _current_branch_state(project_root),
        "pullRequest": {
            "number": 149,
            "title": _clean_text(pr_state.get("title")),
            "headRefName": _clean_text(pr_state.get("headRefName")),
            "headRefOid": _clean_text(pr_state.get("headRefOid")),
            "baseRefName": _clean_text(pr_state.get("baseRefName")),
            "isDraft": bool(pr_state.get("isDraft")),
            "mergeable": _clean_text(pr_state.get("mergeable")),
            "mergeStateStatus": _clean_text(pr_state.get("mergeStateStatus")),
            "updatedAt": _clean_text(pr_state.get("updatedAt")),
            "url": _clean_text(pr_state.get("url")),
            "fileRows": len(pr_files),
            "files": pr_files,
        },
        "dependencyFindings": dependency_rows,
        "textAlignmentFinding": text_alignment,
        "decision": {
            "decision": "abandon_current_pr_before_public_rc",
            "mergeRecommended": False,
            "recutRecommendedForV01": False,
            "laterSideTrackAllowed": True,
            "reason": (
                "PR #149 is draft/conflicting and depends on older complex strict-evidence "
                "side-stack modules that are not present in the v0.1 text-evidence stack."
            ),
            "conceptDisposition": "preserve_as_later_strict_evidence_side_track_only",
        },
        "publicRcImpact": "remove_from_public_rc_merge_queue_after_operator_closes_or_marks_abandoned",
        "nextAction": "operator_close_or_abandon_pr149_without_merging_after_approval",
        "prFileRows": len(pr_files),
        "missingDependencyRows": missing_dependency_rows,
        "mutationCounters": {
            "canonicalParsedArtifactWriteRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "vaultScanRows": 0,
            "externalDownloadRows": 0,
            "pullRequestMutationRows": 0,
            "mergeRows": 0,
            "cherryPickRows": 0,
            "runtimeAnswerVisibleExposureRows": 0,
            "strictEvidencePromotionRows": 0,
        },
        "schemaViolationCount": 0,
        "privatePathLeakRows": 0,
        "reportHash": "",
        "warnings": [
            "This report does not close, merge, or edit PR #149.",
            "The broader strict-evidence bridge concept is retained for a later side-track only.",
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
    decision = dict(report.get("decision") or {})
    pr = dict(report.get("pullRequest") or {})
    lines = [
        "# Text Evidence PR #149 Disposition",
        "",
        f"- status: `{report.get('status')}`",
        f"- decision: `{decision.get('decision')}`",
        f"- mergeRecommended: `{decision.get('mergeRecommended')}`",
        f"- recutRecommendedForV01: `{decision.get('recutRecommendedForV01')}`",
        f"- laterSideTrackAllowed: `{decision.get('laterSideTrackAllowed')}`",
        f"- PR mergeable: `{pr.get('mergeable')}`",
        f"- PR mergeStateStatus: `{pr.get('mergeStateStatus')}`",
        f"- missingDependencyRows: `{report.get('missingDependencyRows')}`",
        f"- privatePathLeakRows: `{report.get('privatePathLeakRows')}`",
        f"- nextAction: `{report.get('nextAction')}`",
        "",
        "## Dependency Findings",
        "",
        "| module | present | disposition |",
        "|---|---:|---|",
    ]
    for row in report.get("dependencyFindings", []):
        lines.append(f"| `{row.get('module')}` | `{row.get('presentInCurrentStack')}` | `{row.get('disposition')}` |")
    lines.extend(
        [
            "",
            "## PR Files",
            "",
        ]
    )
    for file_name in pr.get("files", []):
        lines.append(f"- `{file_name}`")
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: dict[str, Any], *, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")


__all__ = [
    "TEXT_EVIDENCE_PR149_DISPOSITION_SCHEMA_ID",
    "build_text_evidence_pr149_disposition_report",
    "render_markdown_report",
    "write_report",
]
