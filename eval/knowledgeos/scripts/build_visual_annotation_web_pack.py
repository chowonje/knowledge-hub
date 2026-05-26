#!/usr/bin/env python3
"""Build a small manual web/VLM annotation pack from visual layout candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.visual_annotation_web_pack import (
    DEFAULT_CANDIDATE_TYPES,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_PACK_ID,
    DEFAULT_PAPER_IDS,
    VISUAL_ANNOTATION_WEB_PACK_SCHEMA_ID,
    build_visual_annotation_web_pack,
    load_candidate_report,
    sanitized_report_ref,
    write_visual_annotation_web_pack,
)


DEFAULT_SOURCE_REPORT_PATH = PROJECT_ROOT / "eval/knowledgeos/reports/visual_layout_candidate_list_report.v1.json"
DEFAULT_JSON_PATH = PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_web_pack_001.v1.json"
DEFAULT_MD_PATH = PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_web_pack_001.v1.md"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE_REPORT_PATH)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--pack-id", default=DEFAULT_PACK_ID)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument(
        "--paper-id",
        action="append",
        dest="paper_ids",
        help="Preferred paper id. Repeat to set order. Defaults to AlexNet then ResNet.",
    )
    parser.add_argument(
        "--candidate-type",
        action="append",
        dest="candidate_types",
        help="Candidate type to include. Repeat to set allowed types.",
    )
    parser.add_argument("--json", action="store_true", help="Print pack JSON to stdout.")
    parser.add_argument("--no-write", action="store_true", help="Build and validate without writing reports.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    source_report_path = args.source_report.expanduser()
    source_report = load_candidate_report(source_report_path)
    report = build_visual_annotation_web_pack(
        source_report,
        pack_id=args.pack_id,
        source_report_ref=sanitized_report_ref(source_report_path, project_root=PROJECT_ROOT),
        max_candidates=args.max_candidates,
        preferred_paper_ids=tuple(args.paper_ids or DEFAULT_PAPER_IDS),
        included_candidate_types=tuple(args.candidate_types or DEFAULT_CANDIDATE_TYPES),
    )
    validation = validate_payload(report, VISUAL_ANNOTATION_WEB_PACK_SCHEMA_ID, strict=True)
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "visual annotation web pack schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    if not args.no_write:
        paths = write_visual_annotation_web_pack(
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
