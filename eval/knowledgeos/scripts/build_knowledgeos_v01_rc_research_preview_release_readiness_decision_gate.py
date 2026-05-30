#!/usr/bin/env python3
"""Build the KnowledgeOS v0.1 RC Research Preview release-readiness decision gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.knowledgeos_v01_rc_research_preview_release_readiness_decision_gate import (
    DEFAULT_POSITIVE_COMPLETE_REVIEW_REPORT,
    DEFAULT_PRODUCT_DEFINITION_DOC,
    DEFAULT_RELEASE_NOTES_DOC,
    KNOWLEDGEOS_V01_RC_RESEARCH_PREVIEW_RELEASE_READINESS_DECISION_GATE_SCHEMA_ID,
    build_knowledgeos_v01_rc_research_preview_release_readiness_decision_gate,
    write_knowledgeos_v01_rc_research_preview_release_readiness_decision_gate,
)


DEFAULT_JSON_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/knowledgeos_v01_rc_research_preview_release_readiness_decision_gate.v1.json"
)
DEFAULT_MD_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/knowledgeos_v01_rc_research_preview_release_readiness_decision_gate.v1.md"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positive-complete-review-report",
        type=Path,
        default=DEFAULT_POSITIVE_COMPLETE_REVIEW_REPORT,
    )
    parser.add_argument("--product-definition-doc", type=Path, default=DEFAULT_PRODUCT_DEFINITION_DOC)
    parser.add_argument("--release-notes-doc", type=Path, default=DEFAULT_RELEASE_NOTES_DOC)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--json", action="store_true", help="Print full report JSON to stdout.")
    parser.add_argument("--no-write-report", action="store_true", help="Build and validate without writing reports.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    report = build_knowledgeos_v01_rc_research_preview_release_readiness_decision_gate(
        positive_complete_review_report_path=args.positive_complete_review_report,
        product_definition_doc_path=args.product_definition_doc,
        release_notes_doc_path=args.release_notes_doc,
    )
    validation = validate_payload(
        report,
        KNOWLEDGEOS_V01_RC_RESEARCH_PREVIEW_RELEASE_READINESS_DECISION_GATE_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "KnowledgeOS v0.1 RC Research Preview release-readiness decision gate schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    if args.no_write_report:
        paths: dict[str, str] = {}
    else:
        paths = write_knowledgeos_v01_rc_research_preview_release_readiness_decision_gate(
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
                    "readinessDecision": report.get("readinessDecision"),
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
