#!/usr/bin/env python3
"""Build the v0.1 RC labs release gate for parsed-artifact evidence chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.application.public_release_hygiene import check_public_release_hygiene
from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate import (
    DEFAULT_DEFAULT_OFF_REPORT,
    DEFAULT_PROMOTION_REVIEW_REPORT,
    DEFAULT_SURFACE_LIVE_SMOKE_REPORT,
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_LABS_RELEASE_GATE_SCHEMA_ID,
    build_parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate,
    write_parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate,
)
from scripts.check_release_smoke import run_smoke


DEFAULT_JSON_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate.v1.json"
)
DEFAULT_MD_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate.v1.md"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promotion-review-report", type=Path, default=DEFAULT_PROMOTION_REVIEW_REPORT)
    parser.add_argument("--default-off-no-answer-report", type=Path, default=DEFAULT_DEFAULT_OFF_REPORT)
    parser.add_argument("--surface-live-smoke-report", type=Path, default=DEFAULT_SURFACE_LIVE_SMOKE_REPORT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--json", action="store_true", help="Print full report JSON to stdout.")
    parser.add_argument("--no-write-report", action="store_true", help="Build and validate without writing reports.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    release_smoke_payload = run_smoke(mode="release", keep_temp_dir=False)
    public_hygiene_payload = check_public_release_hygiene(PROJECT_ROOT)
    report = build_parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate(
        promotion_review_report_path=args.promotion_review_report,
        default_off_no_answer_report_path=args.default_off_no_answer_report,
        surface_live_smoke_report_path=args.surface_live_smoke_report,
        release_smoke_payload=release_smoke_payload,
        public_hygiene_payload=public_hygiene_payload,
    )
    validation = validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_LABS_RELEASE_GATE_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "parsed artifact evidence chunk v0.1 RC labs release gate schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    if not args.no_write_report:
        paths = write_parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate(
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
                    "releaseDecision": report.get("releaseDecision"),
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
