#!/usr/bin/env python3
"""Build parsed-artifact evidence chunk candidate full-apply readiness gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_full_apply_readiness_gate import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_READINESS_GATE_SCHEMA_ID,
    build_parsed_artifact_evidence_chunk_candidate_full_apply_readiness_gate,
    load_json,
    sanitized_report_ref,
    write_parsed_artifact_evidence_chunk_candidate_full_apply_readiness_gate,
)


DEFAULT_CANDIDATE_DRY_RUN_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_dry_run.v1.json"
)
DEFAULT_CANARY_APPLY_READBACK_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_canary_apply_readback.v1.json"
)
DEFAULT_JSON_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_full_apply_readiness_gate.v1.json"
)
DEFAULT_MD_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_full_apply_readiness_gate.v1.md"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dry-run", type=Path, default=DEFAULT_CANDIDATE_DRY_RUN_PATH)
    parser.add_argument("--canary-apply-readback", type=Path, default=DEFAULT_CANARY_APPLY_READBACK_PATH)
    parser.add_argument("--papers-dir", type=Path, default=Path.home() / ".khub" / "papers")
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--json", action="store_true", help="Print full report JSON to stdout.")
    parser.add_argument("--no-write-report", action="store_true", help="Build and validate without writing report files.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    dry_run_path = args.candidate_dry_run.expanduser()
    canary_path = args.canary_apply_readback.expanduser()
    report = build_parsed_artifact_evidence_chunk_candidate_full_apply_readiness_gate(
        candidate_dry_run_report=load_json(dry_run_path),
        source_candidate_dry_run_report_ref=sanitized_report_ref(dry_run_path, project_root=PROJECT_ROOT),
        canary_apply_readback_report=load_json(canary_path),
        source_canary_apply_readback_report_ref=sanitized_report_ref(canary_path, project_root=PROJECT_ROOT),
        papers_dir=args.papers_dir,
    )
    validation = validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_READINESS_GATE_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "parsed artifact evidence chunk candidate full-apply readiness gate schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    if not args.no_write_report:
        paths = write_parsed_artifact_evidence_chunk_candidate_full_apply_readiness_gate(
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
