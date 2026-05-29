#!/usr/bin/env python3
"""Build the KnowledgeOS v0.1 RC vision bottleneck definition review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.knowledgeos_v01_rc_vision_bottleneck_definition_review import (
    DEFAULT_DRAFT_PR_POST_OPEN_REVIEW_REPORT,
    DEFAULT_LABS_RELEASE_GATE_REPORT,
    DEFAULT_PRODUCT_DEFINITION_DOC,
    DEFAULT_PUBLIC_DEFAULT_PROMOTION_GATE_REPORT,
    KNOWLEDGEOS_V01_RC_VISION_BOTTLENECK_DEFINITION_REVIEW_SCHEMA_ID,
    build_knowledgeos_v01_rc_vision_bottleneck_definition_review,
    write_knowledgeos_v01_rc_vision_bottleneck_definition_review,
)


DEFAULT_JSON_PATH = PROJECT_ROOT / "eval/knowledgeos/reports/knowledgeos_v01_rc_vision_bottleneck_definition_review.v1.json"
DEFAULT_MD_PATH = PROJECT_ROOT / "eval/knowledgeos/reports/knowledgeos_v01_rc_vision_bottleneck_definition_review.v1.md"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-definition-doc", type=Path, default=DEFAULT_PRODUCT_DEFINITION_DOC)
    parser.add_argument("--draft-pr-post-open-review-report", type=Path, default=DEFAULT_DRAFT_PR_POST_OPEN_REVIEW_REPORT)
    parser.add_argument("--public-default-promotion-gate-report", type=Path, default=DEFAULT_PUBLIC_DEFAULT_PROMOTION_GATE_REPORT)
    parser.add_argument("--labs-release-gate-report", type=Path, default=DEFAULT_LABS_RELEASE_GATE_REPORT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--json", action="store_true", help="Print full report JSON to stdout.")
    parser.add_argument("--no-write-report", action="store_true", help="Build and validate without writing reports.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    report = build_knowledgeos_v01_rc_vision_bottleneck_definition_review(
        product_definition_doc_path=args.product_definition_doc,
        draft_pr_post_open_review_report_path=args.draft_pr_post_open_review_report,
        public_default_promotion_gate_report_path=args.public_default_promotion_gate_report,
        labs_release_gate_report_path=args.labs_release_gate_report,
    )
    validation = validate_payload(report, KNOWLEDGEOS_V01_RC_VISION_BOTTLENECK_DEFINITION_REVIEW_SCHEMA_ID, strict=True)
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "KnowledgeOS v0.1 RC vision bottleneck definition review schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    if args.no_write_report:
        paths = {}
    else:
        paths = write_knowledgeos_v01_rc_vision_bottleneck_definition_review(
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
                    "productDecision": report.get("productDecision"),
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
