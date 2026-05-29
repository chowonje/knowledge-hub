#!/usr/bin/env python3
"""Build the v0.1 RC Research Preview release package handoff."""

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
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_release_package_handoff import (
    DEFAULT_BASE,
    DEFAULT_BRANCH,
    DEFAULT_PR_TITLE,
    DEFAULT_RELEASE_NOTES_REVIEW_REPORT,
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_RELEASE_PACKAGE_HANDOFF_SCHEMA_ID,
    build_parsed_artifact_evidence_chunk_answer_path_v01_rc_release_package_handoff,
    write_parsed_artifact_evidence_chunk_answer_path_v01_rc_release_package_handoff,
)


DEFAULT_JSON_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_release_package_handoff.v1.json"
)
DEFAULT_MD_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_release_package_handoff.v1.md"
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


def _collect_current_state(branch_name: str) -> dict[str, Any]:
    ahead_behind = _run_git(["rev-list", "--left-right", "--count", "origin/main...HEAD"]).split()
    ahead = int(ahead_behind[1]) if len(ahead_behind) == 2 else 0
    behind = int(ahead_behind[0]) if len(ahead_behind) == 2 else 0
    origin_main_sha = _run_git(["rev-parse", "origin/main"])
    remote_main_row = _run_git(["ls-remote", "origin", "refs/heads/main"]).split()
    remote_main_sha = remote_main_row[0] if remote_main_row else ""
    remote_branch_row = _run_git(["ls-remote", "--heads", "origin", branch_name]).split()
    return {
        "branchName": _run_git(["branch", "--show-current"]),
        "headSha": _run_git(["rev-parse", "HEAD"]),
        "headShortSha": _run_git(["rev-parse", "--short", "HEAD"]),
        "originMainSha": origin_main_sha,
        "originMainShortSha": _short_sha(origin_main_sha),
        "remoteMainSha": remote_main_sha,
        "remoteMainShortSha": _short_sha(remote_main_sha),
        "remoteMainVerified": bool(origin_main_sha and remote_main_sha and origin_main_sha == remote_main_sha),
        "aheadCommits": ahead,
        "behindCommits": behind,
        "remoteBranchExists": bool(remote_branch_row),
        "statusRows": _parse_status_rows(_run_git(["status", "--porcelain", "--untracked-files=all"])),
    }


def _collect_github_pr_state(branch_name: str) -> dict[str, Any]:
    completed = _run(
        [
            "gh",
            "pr",
            "list",
            "--head",
            branch_name,
            "--state",
            "open",
            "--json",
            "number,title,url,isDraft,baseRefName,headRefName",
        ],
        timeout=30.0,
    )
    if completed.returncode != 0:
        return {"lookupStatus": "unavailable", "openPrRows": []}
    try:
        payload = json.loads(completed.stdout or "[]")
    except Exception:
        return {"lookupStatus": "unavailable", "openPrRows": []}
    rows = []
    for row in list(payload):
        item = dict(row)
        rows.append(
            {
                "number": item.get("number"),
                "title": item.get("title"),
                "url": item.get("url"),
                "isDraft": item.get("isDraft"),
                "baseRefName": item.get("baseRefName"),
                "headRefName": item.get("headRefName"),
            }
        )
    return {"lookupStatus": "ok", "openPrRows": rows}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-notes-review-report", type=Path, default=DEFAULT_RELEASE_NOTES_REVIEW_REPORT)
    parser.add_argument("--branch-name", default=DEFAULT_BRANCH)
    parser.add_argument("--base-branch", default=DEFAULT_BASE)
    parser.add_argument("--pr-title", default=DEFAULT_PR_TITLE)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--json", action="store_true", help="Print full report JSON to stdout.")
    parser.add_argument("--no-write-report", action="store_true", help="Build and validate without writing reports.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    current_state = _collect_current_state(args.branch_name)
    github_pr_state = _collect_github_pr_state(args.branch_name)
    report = build_parsed_artifact_evidence_chunk_answer_path_v01_rc_release_package_handoff(
        release_notes_review_report_path=args.release_notes_review_report,
        current_state=current_state,
        github_pr_state=github_pr_state,
        branch_name=args.branch_name,
        base_branch=args.base_branch,
        pr_title=args.pr_title,
    )
    validation = validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_RELEASE_PACKAGE_HANDOFF_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "parsed artifact evidence chunk v0.1 RC release package handoff schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    if not args.no_write_report:
        paths = write_parsed_artifact_evidence_chunk_answer_path_v01_rc_release_package_handoff(
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
                    "handoffDecision": report.get("handoffDecision"),
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
