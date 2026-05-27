#!/usr/bin/env python3
"""Build the GPT-facing visual hint decision recommendation pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack import (
    DEFAULT_PACK_DIR_REF,
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_REVIEW_PACK_SCHEMA_ID,
    build_visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack,
    load_json,
    sanitized_report_ref,
    write_visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack,
)


DEFAULT_SOURCE_DECISION_RECORD_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_human_product_decision_record.v1.json"
)
DEFAULT_SOURCE_REVIEW_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_review.v1.json"
)
DEFAULT_JSON_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack.v1.json"
)
DEFAULT_MD_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack.v1.md"
)
DEFAULT_PACK_DIR = PROJECT_ROOT / DEFAULT_PACK_DIR_REF


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-decision-record", type=Path, default=DEFAULT_SOURCE_DECISION_RECORD_PATH)
    parser.add_argument("--source-review-report", type=Path, default=DEFAULT_SOURCE_REVIEW_PATH)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--json", action="store_true", help="Print full report JSON to stdout.")
    parser.add_argument("--no-write", action="store_true", help="Build and validate without writing reports.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    decision_path = args.source_decision_record.expanduser()
    review_path = args.source_review_report.expanduser()
    decision_record = load_json(decision_path)
    review_report = load_json(review_path)
    report = build_visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack(
        decision_record,
        review_report,
        source_decision_record_ref=sanitized_report_ref(decision_path, project_root=PROJECT_ROOT),
        source_review_report_ref=sanitized_report_ref(review_path, project_root=PROJECT_ROOT),
    )
    validation = validate_payload(
        report,
        VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_REVIEW_PACK_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "visual retrieval hint GPT decision review pack schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    if not args.no_write:
        paths = write_visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack(
            report,
            report_json=args.report_json,
            report_md=args.report_md,
            pack_dir=args.pack_dir,
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
                    "paths": paths,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0 if report.get("status") == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
