#!/usr/bin/env python3
"""Build parsed-artifact evidence chunk candidate dry-run report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_dry_run import (
    DEFAULT_PAPERS_DIR,
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_DRY_RUN_SCHEMA_ID,
    build_parsed_artifact_evidence_chunk_candidate_dry_run,
    load_json,
    sanitized_report_ref,
    write_parsed_artifact_evidence_chunk_candidate_dry_run,
)


DEFAULT_SOURCE_CONTRACT_REVIEW_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "paper_retrieval_lane_split_parsed_artifact_evidence_chunk_contract_review.v1.json"
)
DEFAULT_JSON_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_dry_run.v1.json"
)
DEFAULT_MD_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_dry_run.v1.md"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-contract-review", type=Path, default=DEFAULT_SOURCE_CONTRACT_REVIEW_PATH)
    parser.add_argument("--papers-dir", type=Path, default=DEFAULT_PAPERS_DIR)
    parser.add_argument("--max-rows-per-paper", type=int, default=4)
    parser.add_argument("--max-total-rows", type=int, default=1200)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--json", action="store_true", help="Print full report JSON to stdout.")
    parser.add_argument("--no-write", action="store_true", help="Build and validate without writing reports.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    source_contract_path = args.source_contract_review.expanduser()
    report = build_parsed_artifact_evidence_chunk_candidate_dry_run(
        contract_review=load_json(source_contract_path),
        source_contract_review_ref=sanitized_report_ref(source_contract_path, project_root=PROJECT_ROOT),
        papers_dir=args.papers_dir,
        max_rows_per_paper=args.max_rows_per_paper,
        max_total_rows=args.max_total_rows,
    )
    validation = validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_DRY_RUN_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "parsed artifact evidence chunk candidate dry-run schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    if not args.no_write:
        paths = write_parsed_artifact_evidence_chunk_candidate_dry_run(
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
