#!/usr/bin/env python3
"""Build the v0.1 RC draft PR handoff for parsed-artifact evidence chunks."""

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
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_opening_or_handoff import (
    DEFAULT_BASE,
    DEFAULT_BRANCH,
    DEFAULT_BRANCH_PR_READINESS_REPORT,
    DEFAULT_DRAFT_PR_TITLE,
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_DRAFT_PR_OPENING_OR_HANDOFF_SCHEMA_ID,
    build_parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_opening_or_handoff,
    write_parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_opening_or_handoff,
)


DEFAULT_JSON_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_opening_or_handoff.v1.json"
)
DEFAULT_MD_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_opening_or_handoff.v1.md"
)


def _run_git(args: list[str], *, timeout: float = 20.0) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _parse_int(text: str) -> int:
    try:
        return int(text.strip() or "0")
    except Exception:
        return 0


def _collect_current_state() -> dict[str, Any]:
    remote_main = _run_git(["ls-remote", "origin", "refs/heads/main"], timeout=30.0).split()
    remote_main_sha = remote_main[0] if remote_main else ""
    local_origin_main_sha = _run_git(["rev-parse", "origin/main"])
    status_rows = [
        {"statusCode": line[:2].strip(), "path": line[2:].strip()}
        for line in _run_git(["status", "--porcelain", "--untracked-files=all"]).splitlines()
        if line
    ]
    return {
        "branchName": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "headShortSha": _run_git(["rev-parse", "--short", "HEAD"]),
        "aheadCommits": _parse_int(_run_git(["rev-list", "--count", "origin/main..HEAD"])),
        "behindCommits": _parse_int(_run_git(["rev-list", "--count", "HEAD..origin/main"])),
        "statusRows": status_rows,
        "remoteMainVerified": bool(remote_main_sha and remote_main_sha == local_origin_main_sha),
    }


def _collect_github_pr_state(branch_name: str) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "gh",
                "pr",
                "list",
                "--head",
                branch_name,
                "--base",
                DEFAULT_BASE,
                "--json",
                "number,state,isDraft,mergeStateStatus,mergeable,headRefName,baseRefName,title",
                "--limit",
                "10",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30.0,
            check=False,
        )
    except Exception:
        return {"lookupStatus": "unavailable", "openPrRows": []}
    if completed.returncode != 0:
        return {"lookupStatus": "unavailable", "openPrRows": []}
    try:
        rows = json.loads(completed.stdout or "[]")
    except Exception:
        rows = []
    sanitized_rows: list[dict[str, Any]] = []
    iterable_rows = rows if isinstance(rows, list) else []
    for row in iterable_rows:
        item = dict(row)
        sanitized_rows.append(
            {
                "number": item.get("number"),
                "state": item.get("state"),
                "isDraft": bool(item.get("isDraft")),
                "mergeStateStatus": item.get("mergeStateStatus"),
                "mergeable": item.get("mergeable"),
                "headRefName": item.get("headRefName"),
                "baseRefName": item.get("baseRefName"),
            }
        )
    return {"lookupStatus": "ok", "openPrRows": sanitized_rows}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-pr-readiness-report", type=Path, default=DEFAULT_BRANCH_PR_READINESS_REPORT)
    parser.add_argument("--branch-name", default=DEFAULT_BRANCH)
    parser.add_argument("--base-branch", default=DEFAULT_BASE)
    parser.add_argument("--draft-pr-title", default=DEFAULT_DRAFT_PR_TITLE)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--json", action="store_true", help="Print full report JSON to stdout.")
    parser.add_argument("--no-write-report", action="store_true", help="Build and validate without writing reports.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    current_state = _collect_current_state()
    github_pr_state = _collect_github_pr_state(str(args.branch_name))
    report = build_parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_opening_or_handoff(
        branch_pr_readiness_report_path=args.branch_pr_readiness_report,
        current_state=current_state,
        github_pr_state=github_pr_state,
        branch_name=str(args.branch_name),
        base_branch=str(args.base_branch),
        draft_pr_title=str(args.draft_pr_title),
    )
    validation = validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_DRAFT_PR_OPENING_OR_HANDOFF_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "parsed artifact evidence chunk v0.1 RC draft PR handoff schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    if not args.no_write_report:
        paths = write_parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_opening_or_handoff(
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
