#!/usr/bin/env python3
"""Build the third bounded visual annotation expansion pack design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.visual_annotation_expansion_pack_design import (
    VISUAL_ANNOTATION_EXPANSION_PACK_DESIGN_SCHEMA_ID,
    build_visual_annotation_expansion_pack_design,
    load_json,
    sanitized_report_ref,
    write_visual_annotation_expansion_pack_design,
)


DEFAULT_CANDIDATE_REPORT_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_layout_candidate_list_report.v1.json"
)
DEFAULT_WEB_PACK_PATH = PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_web_pack_001.v1.json"
DEFAULT_SOURCE_DRY_RUN_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_dry_run.v1.json"
)
DEFAULT_JSON_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_expansion_pack_design_003.v1.json"
)
DEFAULT_MD_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_expansion_pack_design_003.v1.md"
)
DEFAULT_TYPE_QUOTAS = {
    "table_region": 8,
    "figure_caption_region": 8,
    "equation_region": 4,
    "layout_region": 4,
    "image_region": 0,
}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", type=Path, default=DEFAULT_CANDIDATE_REPORT_PATH)
    parser.add_argument("--source-web-pack", type=Path, default=DEFAULT_WEB_PACK_PATH)
    parser.add_argument("--source-dry-run", type=Path, default=DEFAULT_SOURCE_DRY_RUN_PATH)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--pack-id", default="visual_annotation_expansion_pack_003")
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--json", action="store_true", help="Print full report JSON to stdout.")
    parser.add_argument("--no-write", action="store_true", help="Build and validate without writing reports.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    candidate_report_path = args.candidate_report.expanduser()
    web_pack_path = args.source_web_pack.expanduser()
    dry_run_path = args.source_dry_run.expanduser()
    candidate_report = load_json(candidate_report_path)
    web_pack = load_json(web_pack_path)
    dry_run_report = load_json(dry_run_path)
    report = build_visual_annotation_expansion_pack_design(
        candidate_report,
        web_pack,
        dry_run_report,
        pack_id=args.pack_id,
        source_candidate_report_ref=sanitized_report_ref(candidate_report_path, project_root=PROJECT_ROOT),
        source_web_pack_ref=sanitized_report_ref(web_pack_path, project_root=PROJECT_ROOT),
        source_dry_run_report_ref=sanitized_report_ref(dry_run_path, project_root=PROJECT_ROOT),
        max_candidates=args.max_candidates,
        type_quotas=DEFAULT_TYPE_QUOTAS,
    )
    validation = validate_payload(
        report,
        VISUAL_ANNOTATION_EXPANSION_PACK_DESIGN_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "visual annotation expansion pack 003 design schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    if not args.no_write:
        paths = write_visual_annotation_expansion_pack_design(
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
