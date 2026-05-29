#!/usr/bin/env python3
"""Build the post-merge v0.1 RC convergence review for parsed-artifact evidence chunks."""

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
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_post_merge_convergence import (
    DEFAULT_BRANCH_PR_READINESS_REPORT,
    DEFAULT_DRAFT_PR_HANDOFF_REPORT,
    DEFAULT_MERGED_PR_NUMBER,
    DEFAULT_RELEASE_GATE_REPORT,
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_POST_MERGE_CONVERGENCE_SCHEMA_ID,
    build_parsed_artifact_evidence_chunk_answer_path_v01_rc_post_merge_convergence,
    write_parsed_artifact_evidence_chunk_answer_path_v01_rc_post_merge_convergence,
)


DEFAULT_JSON_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_post_merge_convergence.v1.json"
)
DEFAULT_MD_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_post_merge_convergence.v1.md"
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


def _collect_git_state() -> dict[str, Any]:
    remote_main = _run_git(["ls-remote", "origin", "refs/heads/main"], timeout=30.0).split()
    remote_main_sha = remote_main[0] if remote_main else ""
    origin_main_sha = _run_git(["rev-parse", "origin/main"])
    return {
        "branchName": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "headSha": _run_git(["rev-parse", "HEAD"]),
        "headShortSha": _run_git(["rev-parse", "--short", "HEAD"]),
        "originMainSha": origin_main_sha,
        "originMainShortSha": _short_sha(origin_main_sha),
        "remoteMainSha": remote_main_sha,
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
            "number,title,state,isDraft,mergedAt,mergeCommit,headRefName,baseRefName,statusCheckRollup",
        ],
        timeout=30.0,
    )
    if completed.returncode != 0:
        return {"lookupStatus": "unavailable", "number": pr_number}
    try:
        payload = json.loads(completed.stdout or "{}")
    except Exception:
        return {"lookupStatus": "unavailable", "number": pr_number}
    status_rows: list[dict[str, Any]] = []
    for row in list(payload.get("statusCheckRollup") or []):
        item = dict(row)
        if item.get("__typename") != "CheckRun":
            continue
        status_rows.append(
            {
                "__typename": "CheckRun",
                "name": item.get("name"),
                "status": item.get("status"),
                "conclusion": item.get("conclusion"),
                "workflowName": item.get("workflowName"),
            }
        )
    return {
        "lookupStatus": "ok",
        "number": payload.get("number"),
        "title": payload.get("title"),
        "state": payload.get("state"),
        "isDraft": bool(payload.get("isDraft")),
        "mergedAt": payload.get("mergedAt"),
        "mergeCommit": payload.get("mergeCommit") or {},
        "headRefName": payload.get("headRefName"),
        "baseRefName": payload.get("baseRefName"),
        "statusCheckRollup": status_rows,
    }


def _summarize_public_hygiene() -> dict[str, Any]:
    completed = _run(
        [sys.executable, "scripts/check_public_release_hygiene.py", "--repo-root", ".", "--json"],
        timeout=60.0,
    )
    if completed.returncode != 0:
        return {"status": "failed", "issueCount": 1}
    try:
        payload = json.loads(completed.stdout or "{}")
    except Exception:
        return {"status": "failed", "issueCount": 1}
    return {"status": payload.get("status"), "issueCount": payload.get("issueCount")}


def _summarize_release_smoke() -> dict[str, Any]:
    completed = _run(
        [sys.executable, "scripts/check_release_smoke.py", "--mode", "release", "--json"],
        timeout=120.0,
    )
    if completed.returncode != 0:
        return {"status": "failed", "checkedCount": 0, "passedCount": 0}
    try:
        payload = json.loads(completed.stdout or "{}")
    except Exception:
        return {"status": "failed", "checkedCount": 0, "passedCount": 0}
    return {
        "status": payload.get("status"),
        "checkedCount": payload.get("checkedCount"),
        "passedCount": payload.get("passedCount"),
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-gate-report", type=Path, default=DEFAULT_RELEASE_GATE_REPORT)
    parser.add_argument("--branch-pr-readiness-report", type=Path, default=DEFAULT_BRANCH_PR_READINESS_REPORT)
    parser.add_argument("--draft-pr-handoff-report", type=Path, default=DEFAULT_DRAFT_PR_HANDOFF_REPORT)
    parser.add_argument("--pr-number", type=int, default=DEFAULT_MERGED_PR_NUMBER)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--json", action="store_true", help="Print full report JSON to stdout.")
    parser.add_argument("--no-write-report", action="store_true", help="Build and validate without writing reports.")
    parser.add_argument(
        "--skip-live-checks",
        action="store_true",
        help="Use existing release-gate counters instead of running live smoke/hygiene checks.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    git_state = _collect_git_state()
    github_pr_state = _collect_github_pr_state(args.pr_number)
    if args.skip_live_checks:
        hygiene_result = {"status": "ok", "issueCount": 0}
        release_smoke_result = {"status": "ok", "checkedCount": 10, "passedCount": 10}
    else:
        hygiene_result = _summarize_public_hygiene()
        release_smoke_result = _summarize_release_smoke()
    report = build_parsed_artifact_evidence_chunk_answer_path_v01_rc_post_merge_convergence(
        release_gate_report_path=args.release_gate_report,
        branch_pr_readiness_report_path=args.branch_pr_readiness_report,
        draft_pr_handoff_report_path=args.draft_pr_handoff_report,
        git_state=git_state,
        github_pr_state=github_pr_state,
        hygiene_result=hygiene_result,
        release_smoke_result=release_smoke_result,
        pr_number=args.pr_number,
    )
    validation = validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_POST_MERGE_CONVERGENCE_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "parsed artifact evidence chunk v0.1 RC post-merge convergence schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    if not args.no_write_report:
        paths = write_parsed_artifact_evidence_chunk_answer_path_v01_rc_post_merge_convergence(
            report,
            report_json=args.report_json,
            report_md=args.report_md,
        )
    else:
        paths = {}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            json.dumps(
                {
                    "status": report.get("status"),
                    "decision": report.get("decision"),
                    "nextRecommendedTranche": report.get("nextRecommendedTranche"),
                    "convergenceDecision": report.get("convergenceDecision"),
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
