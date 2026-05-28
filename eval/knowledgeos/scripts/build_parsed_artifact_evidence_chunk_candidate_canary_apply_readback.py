#!/usr/bin/env python3
"""Build parsed-artifact evidence chunk candidate canary apply/readback report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_canary_apply_readback import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_CANARY_APPLY_READBACK_SCHEMA_ID,
    build_parsed_artifact_evidence_chunk_candidate_canary_apply_readback,
    load_json,
    sanitized_report_ref,
    write_parsed_artifact_evidence_chunk_candidate_canary_apply_readback,
)


DEFAULT_SOURCE_CANDIDATE_DRY_RUN_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_dry_run.v1.json"
)
DEFAULT_JSON_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_canary_apply_readback.v1.json"
)
DEFAULT_MD_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_canary_apply_readback.v1.md"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-candidate-dry-run", type=Path, default=DEFAULT_SOURCE_CANDIDATE_DRY_RUN_PATH)
    parser.add_argument("--papers-dir", type=Path, default=None)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--canary-record-limit", type=int, default=10)
    parser.add_argument("--apply", action="store_true", help="Write canary candidate-store JSONL records.")
    parser.add_argument("--all", action="store_true", help="Blocked in this canary tranche.")
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--json", action="store_true", help="Print full report JSON to stdout.")
    parser.add_argument("--no-write", action="store_true", help="Build and validate without writing reports.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    source_path = args.source_candidate_dry_run.expanduser()
    report = build_parsed_artifact_evidence_chunk_candidate_canary_apply_readback(
        candidate_dry_run_report=load_json(source_path),
        source_candidate_dry_run_report_ref=sanitized_report_ref(source_path, project_root=PROJECT_ROOT),
        papers_dir=args.papers_dir,
        run_id=args.run_id or None,
        apply=bool(args.apply),
        all_requested=bool(args.all),
        canary_record_limit=args.canary_record_limit,
    )
    validation = validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_CANARY_APPLY_READBACK_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "parsed artifact evidence chunk candidate canary apply/readback schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    if not args.no_write:
        paths = write_parsed_artifact_evidence_chunk_candidate_canary_apply_readback(
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
    return 0 if report.get("status") in {"ready", "applied"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
