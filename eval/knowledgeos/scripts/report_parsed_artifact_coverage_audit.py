#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge_hub.application.corpus_artifacts import load_corpus_manifest
from knowledge_hub.core.config import Config
from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.papers.parsed_artifact_coverage_audit import (
    DEFAULT_INCLUDED_CORPUS_TIERS,
    PARSED_ARTIFACT_COVERAGE_AUDIT_SCHEMA_ID,
    build_blocked_parsed_artifact_coverage_audit,
    build_parsed_artifact_coverage_audit,
    write_parsed_artifact_coverage_audit_reports,
)


def _safe_reason(code: str) -> str:
    return code.strip() or "audit_blocked"


def _relative_for_print(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return candidate.name


def build_report_from_runtime(args: argparse.Namespace) -> dict:
    paper_ids = [str(item).strip() for item in list(args.paper_id or []) if str(item).strip()]
    try:
        manifest = load_corpus_manifest(args.corpus_manifest)
    except Exception:
        return build_blocked_parsed_artifact_coverage_audit(
            reason=_safe_reason("corpus_manifest_unavailable"),
            paper_ids=paper_ids,
        )

    config = Config()
    try:
        sqlite_db = SQLiteDatabase(
            config.sqlite_path,
            enable_event_store=False,
            bootstrap=False,
            read_only=True,
        )
    except Exception:
        return build_blocked_parsed_artifact_coverage_audit(
            reason=_safe_reason("sqlite_db_unavailable"),
            paper_ids=paper_ids,
        )

    try:
        return build_parsed_artifact_coverage_audit(
            sqlite_db=sqlite_db,
            papers_dir=config.papers_dir,
            corpus_manifest=manifest,
            paper_ids=paper_ids,
            included_corpus_tiers=args.corpus_tier or DEFAULT_INCLUDED_CORPUS_TIERS,
        )
    finally:
        sqlite_db.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a report-only parsed artifact coverage audit for the eval-critical paper corpus."
    )
    parser.add_argument(
        "--corpus-manifest",
        default=str(REPO_ROOT / "eval" / "knowledgeos" / "fixtures" / "corpus_manifest.json"),
        help="Corpus manifest to audit. Defaults to the eval-critical corpus manifest.",
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Restrict to one source id; repeat as needed.")
    parser.add_argument(
        "--corpus-tier",
        action="append",
        default=[],
        help="Restrict to a corpus tier; defaults to local_corpus and repo_fixture.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "eval" / "knowledgeos" / "runs" / "reports" / "parsed_artifact_coverage_audit"),
        help="Directory for JSON and Markdown report outputs.",
    )
    parser.add_argument("--json", action="store_true", help="Print the schema-backed JSON report.")
    parser.add_argument("--no-write", action="store_true", help="Do not write JSON/Markdown report files.")
    args = parser.parse_args(argv)

    report = build_report_from_runtime(args)
    validation = validate_payload(report, PARSED_ARTIFACT_COVERAGE_AUDIT_SCHEMA_ID, strict=True)
    if not validation.ok:
        report = build_blocked_parsed_artifact_coverage_audit(reason="schema_validation_failed", paper_ids=args.paper_id)

    if not args.no_write:
        paths = write_parsed_artifact_coverage_audit_reports(report, args.output_dir)
        if not args.json:
            print(
                "wrote parsed artifact coverage audit: "
                f"{_relative_for_print(paths['json'])}, {_relative_for_print(paths['markdown'])}"
            )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
