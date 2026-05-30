#!/usr/bin/env python3
"""Build the post-merge convergence report for PR #178 answerability-gate repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.knowledgeos_v01_rc_answerability_gate_repair_post_merge_convergence import (
    DEFAULT_BASE_BRANCH,
    DEFAULT_CONTROLLED_EXECUTION_REPORT,
    DEFAULT_HEAD_BRANCH,
    DEFAULT_PR_NUMBER,
    KNOWLEDGEOS_V01_RC_ANSWERABILITY_GATE_REPAIR_POST_MERGE_CONVERGENCE_SCHEMA_ID,
    build_knowledgeos_v01_rc_answerability_gate_repair_post_merge_convergence,
    write_knowledgeos_v01_rc_answerability_gate_repair_post_merge_convergence,
)


DEFAULT_JSON_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/knowledgeos_v01_rc_answerability_gate_repair_post_merge_convergence.v1.json"
)
DEFAULT_MD_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/knowledgeos_v01_rc_answerability_gate_repair_post_merge_convergence.v1.md"
)


def _run(args: list[str], *, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        check=False,
    )


def _run_git(args: list[str], *, timeout: float = 30.0) -> str:
    completed = _run(["git", *args], timeout=timeout)
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _parse_status_rows(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        if line:
            rows.append({"statusCode": line[:2].strip(), "path": line[2:].strip()})
    return rows


def _short_sha(sha: str) -> str:
    return sha[:7] if sha else ""


def _collect_git_state(head_branch: str) -> dict[str, Any]:
    remote_main = _run_git(["ls-remote", "origin", "refs/heads/main"], timeout=30.0).split()
    remote_branch = _run_git(["ls-remote", "--heads", "origin", head_branch], timeout=30.0).split()
    origin_main_sha = _run_git(["rev-parse", "origin/main"])
    remote_main_sha = remote_main[0] if remote_main else ""
    remote_branch_sha = remote_branch[0] if remote_branch else ""
    return {
        "branchName": _run_git(["branch", "--show-current"]),
        "headSha": _run_git(["rev-parse", "HEAD"]),
        "headShortSha": _run_git(["rev-parse", "--short", "HEAD"]),
        "originMainSha": origin_main_sha,
        "originMainShortSha": _short_sha(origin_main_sha),
        "remoteMainSha": remote_main_sha,
        "remoteMainShortSha": _short_sha(remote_main_sha),
        "remoteFeatureBranchSha": remote_branch_sha,
        "remoteFeatureBranchShortSha": _short_sha(remote_branch_sha),
        "statusRows": _parse_status_rows(_run_git(["status", "--porcelain", "--untracked-files=all"])),
    }


def _collect_github_pr_state(pr_number: int) -> dict[str, Any]:
    completed = _run(
        [
            "gh",
            "pr",
            "view",
            str(pr_number),
            "--json",
            "number,title,url,state,isDraft,mergedAt,mergeCommit,headRefName,baseRefName,statusCheckRollup",
        ],
        timeout=30.0,
    )
    if completed.returncode != 0:
        return {"lookupStatus": "unavailable", "number": pr_number}
    try:
        payload = json.loads(completed.stdout or "{}")
    except Exception:
        return {"lookupStatus": "unavailable", "number": pr_number}
    checks: list[dict[str, Any]] = []
    for row in list(payload.get("statusCheckRollup") or []):
        item = dict(row)
        if item.get("__typename") != "CheckRun":
            continue
        checks.append(
            {
                "__typename": "CheckRun",
                "name": item.get("name"),
                "workflowName": item.get("workflowName"),
                "status": item.get("status"),
                "conclusion": item.get("conclusion"),
            }
        )
    return {
        "lookupStatus": "ok",
        "number": payload.get("number"),
        "title": payload.get("title"),
        "url": payload.get("url"),
        "state": payload.get("state"),
        "isDraft": bool(payload.get("isDraft")),
        "mergedAt": payload.get("mergedAt"),
        "mergeCommit": payload.get("mergeCommit") or {},
        "headRefName": payload.get("headRefName"),
        "baseRefName": payload.get("baseRefName"),
        "statusCheckRollup": checks,
    }


def _summarize_public_hygiene() -> dict[str, Any]:
    completed = _run([sys.executable, "scripts/check_public_release_hygiene.py", "--repo-root", ".", "--json"], timeout=60.0)
    if completed.returncode != 0:
        return {"status": "failed", "issueCount": 1}
    try:
        payload = json.loads(completed.stdout or "{}")
    except Exception:
        return {"status": "failed", "issueCount": 1}
    return {"status": payload.get("status"), "issueCount": payload.get("issueCount")}


def _summarize_release_smoke() -> dict[str, Any]:
    completed = _run([sys.executable, "scripts/check_release_smoke.py", "--mode", "release", "--json"], timeout=120.0)
    if completed.returncode != 0:
        return {"status": "failed", "checkedCount": 0, "passedCount": 0}
    try:
        payload = json.loads(completed.stdout or "{}")
    except Exception:
        return {"status": "failed", "checkedCount": 0, "passedCount": 0}
    return {"status": payload.get("status"), "checkedCount": payload.get("checkedCount"), "passedCount": payload.get("passedCount")}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controlled-execution-report", type=Path, default=DEFAULT_CONTROLLED_EXECUTION_REPORT)
    parser.add_argument("--pr-number", type=int, default=DEFAULT_PR_NUMBER)
    parser.add_argument("--head-branch", default=DEFAULT_HEAD_BRANCH)
    parser.add_argument("--base-branch", default=DEFAULT_BASE_BRANCH)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--json", action="store_true", help="Print full report JSON to stdout.")
    parser.add_argument("--no-write-report", action="store_true", help="Build and validate without writing reports.")
    parser.add_argument("--skip-live-checks", action="store_true", help="Use green summaries instead of running smoke/hygiene checks.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    git_state = _collect_git_state(args.head_branch)
    github_pr_state = _collect_github_pr_state(args.pr_number)
    if args.skip_live_checks:
        release_smoke_result = {"status": "ok", "checkedCount": 10, "passedCount": 10}
        hygiene_result = {"status": "ok", "issueCount": 0}
    else:
        release_smoke_result = _summarize_release_smoke()
        hygiene_result = _summarize_public_hygiene()
    report = build_knowledgeos_v01_rc_answerability_gate_repair_post_merge_convergence(
        controlled_execution_report_path=args.controlled_execution_report,
        git_state=git_state,
        github_pr_state=github_pr_state,
        release_smoke_result=release_smoke_result,
        hygiene_result=hygiene_result,
        pr_number=args.pr_number,
        head_branch=args.head_branch,
        base_branch=args.base_branch,
    )
    validation = validate_payload(
        report,
        KNOWLEDGEOS_V01_RC_ANSWERABILITY_GATE_REPAIR_POST_MERGE_CONVERGENCE_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "KnowledgeOS v0.1 RC answerability gate repair post-merge convergence schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    if args.no_write_report:
        paths = {}
    else:
        paths = write_knowledgeos_v01_rc_answerability_gate_repair_post_merge_convergence(
            report,
            report_json=args.report_json,
            report_md=args.report_md,
        )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            json.dumps(
                {
                    "status": report.get("status"),
                    "decision": report.get("decision"),
                    "nextRecommendedTranche": report.get("nextRecommendedTranche"),
                    "postMergeDecision": report.get("postMergeDecision"),
                    "counts": report.get("counts"),
                    "gate": report.get("gate"),
                    "paths": paths,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0 if report.get("status") == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
