#!/usr/bin/env python3
"""Validate manually supplied GPT visual hint recommendation output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_expansion_gpt_recommendation_output_capture import (
    DEFAULT_OUTPUT_REF,
    DEFAULT_PACK_REF,
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_RECOMMENDATION_OUTPUT_VALIDATION_SCHEMA_ID,
    build_visual_retrieval_hint_candidate_store_expansion_gpt_recommendation_output_validation,
    load_json,
    sanitized_report_ref,
    write_visual_retrieval_hint_candidate_store_expansion_gpt_recommendation_output_validation,
)


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / DEFAULT_OUTPUT_REF
DEFAULT_GPT_REVIEW_PACK_PATH = PROJECT_ROOT / DEFAULT_PACK_REF
DEFAULT_JSON_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "visual_retrieval_hint_candidate_store_expansion_gpt_recommendation_output_001.validation.v1.json"
)
DEFAULT_MD_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "visual_retrieval_hint_candidate_store_expansion_gpt_recommendation_output_001.validation.v1.md"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--source-gpt-review-pack", type=Path, default=DEFAULT_GPT_REVIEW_PACK_PATH)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--json", action="store_true", help="Print full report JSON to stdout.")
    parser.add_argument("--no-write", action="store_true", help="Build and validate without writing reports.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    output_path = args.output.expanduser()
    pack_path = args.source_gpt_review_pack.expanduser()
    output = load_json(output_path)
    gpt_review_pack = load_json(pack_path)
    report = build_visual_retrieval_hint_candidate_store_expansion_gpt_recommendation_output_validation(
        output,
        gpt_review_pack,
        output_ref=sanitized_report_ref(output_path, project_root=PROJECT_ROOT),
        source_gpt_review_pack_ref=sanitized_report_ref(pack_path, project_root=PROJECT_ROOT),
    )
    validation = validate_payload(
        report,
        VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_RECOMMENDATION_OUTPUT_VALIDATION_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "visual retrieval hint GPT recommendation output validation schema failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    if not args.no_write:
        paths = write_visual_retrieval_hint_candidate_store_expansion_gpt_recommendation_output_validation(
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
                    "paths": paths,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0 if report.get("status") == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
