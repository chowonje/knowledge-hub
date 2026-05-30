#!/usr/bin/env python3
"""Build the KnowledgeOS v0.1 RC positive section/paragraph seed report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.knowledgeos_v01_rc_corpus_scale_answer_quality_positive_section_paragraph_seed import (
    DEFAULT_CONTROLLED_EXECUTION_REPORT,
    DEFAULT_LIVE_RUNNER_DRY_RUN_REPORT,
    DEFAULT_POST_MERGE_REPORT,
    KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_POSITIVE_SECTION_PARAGRAPH_SEED_SCHEMA_ID,
    build_knowledgeos_v01_rc_corpus_scale_answer_quality_positive_section_paragraph_seed,
    write_knowledgeos_v01_rc_corpus_scale_answer_quality_positive_section_paragraph_seed,
)


DEFAULT_JSON_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/knowledgeos_v01_rc_corpus_scale_answer_quality_positive_section_paragraph_seed.v1.json"
)
DEFAULT_MD_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/knowledgeos_v01_rc_corpus_scale_answer_quality_positive_section_paragraph_seed.v1.md"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--post-merge-report", type=Path, default=DEFAULT_POST_MERGE_REPORT)
    parser.add_argument("--controlled-execution-report", type=Path, default=DEFAULT_CONTROLLED_EXECUTION_REPORT)
    parser.add_argument("--live-runner-dry-run-report", type=Path, default=DEFAULT_LIVE_RUNNER_DRY_RUN_REPORT)
    parser.add_argument("--papers-dir", type=Path, default=None)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--json", action="store_true", help="Print full report JSON to stdout.")
    parser.add_argument("--no-write-report", action="store_true", help="Build and validate without writing reports.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    kwargs = {
        "post_merge_report_path": args.post_merge_report,
        "controlled_execution_report_path": args.controlled_execution_report,
        "live_runner_dry_run_report_path": args.live_runner_dry_run_report,
    }
    if args.papers_dir is not None:
        kwargs["papers_dir"] = args.papers_dir
    report = build_knowledgeos_v01_rc_corpus_scale_answer_quality_positive_section_paragraph_seed(**kwargs)
    validation = validate_payload(
        report,
        KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_POSITIVE_SECTION_PARAGRAPH_SEED_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "KnowledgeOS v0.1 RC positive section/paragraph seed schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    if args.no_write_report:
        paths = {}
    else:
        paths = write_knowledgeos_v01_rc_corpus_scale_answer_quality_positive_section_paragraph_seed(
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
