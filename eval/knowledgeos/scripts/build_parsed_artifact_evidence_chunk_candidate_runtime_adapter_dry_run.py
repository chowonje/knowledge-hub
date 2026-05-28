#!/usr/bin/env python3
"""Build parsed-artifact evidence chunk candidate runtime adapter dry-run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_runtime_adapter_dry_run import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_ADAPTER_DRY_RUN_SCHEMA_ID,
    build_parsed_artifact_evidence_chunk_candidate_runtime_adapter_dry_run,
    load_json,
    sanitized_report_ref,
    write_parsed_artifact_evidence_chunk_candidate_runtime_adapter_dry_run,
)


DEFAULT_ADAPTER_DESIGN_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_runtime_adapter_design.v1.json"
)
DEFAULT_FULL_APPLY_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_full_apply_executor_apply_readback.v1.json"
)
DEFAULT_FULL_APPLY_READBACK_REVIEW_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_full_apply_readback_review.v1.json"
)
DEFAULT_JSON_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_runtime_adapter_dry_run.v1.json"
)
DEFAULT_MD_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_runtime_adapter_dry_run.v1.md"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-design", type=Path, default=DEFAULT_ADAPTER_DESIGN_PATH)
    parser.add_argument("--full-apply", type=Path, default=DEFAULT_FULL_APPLY_PATH)
    parser.add_argument("--full-apply-readback-review", type=Path, default=DEFAULT_FULL_APPLY_READBACK_REVIEW_PATH)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--json", action="store_true", help="Print full report JSON to stdout.")
    parser.add_argument("--no-write-report", action="store_true", help="Build and validate without writing report files.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    adapter_design_path = args.adapter_design.expanduser()
    full_apply_path = args.full_apply.expanduser()
    full_apply_readback_review_path = args.full_apply_readback_review.expanduser()
    report = build_parsed_artifact_evidence_chunk_candidate_runtime_adapter_dry_run(
        runtime_adapter_design_report=load_json(adapter_design_path),
        full_apply_report=load_json(full_apply_path),
        full_apply_readback_review_report=load_json(full_apply_readback_review_path),
        source_runtime_adapter_design_report_ref=sanitized_report_ref(
            adapter_design_path,
            project_root=PROJECT_ROOT,
        ),
        source_full_apply_report_ref=sanitized_report_ref(
            full_apply_path,
            project_root=PROJECT_ROOT,
        ),
        source_full_apply_readback_review_report_ref=sanitized_report_ref(
            full_apply_readback_review_path,
            project_root=PROJECT_ROOT,
        ),
    )
    validation = validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_ADAPTER_DRY_RUN_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "parsed artifact evidence chunk candidate runtime adapter dry-run schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    if not args.no_write_report:
        paths = write_parsed_artifact_evidence_chunk_candidate_runtime_adapter_dry_run(
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
