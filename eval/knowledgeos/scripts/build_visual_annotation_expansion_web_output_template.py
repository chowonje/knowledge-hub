#!/usr/bin/env python3
"""Build the fill-only template for expansion web/VLM output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.visual_annotation_expansion_manual_output_capture import (
    DEFAULT_WEB_OUTPUT_TEMPLATE_ID,
    VISUAL_ANNOTATION_EXPANSION_WEB_OUTPUT_TEMPLATE_SCHEMA_ID,
    build_visual_annotation_expansion_web_output_template,
    load_json,
    sanitized_report_ref,
    write_visual_annotation_expansion_web_output_template,
)


DEFAULT_SOURCE_HANDOFF_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_expansion_operator_handoff_002.v1.json"
)
DEFAULT_TEMPLATE_JSON_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_expansion_web_output_template_002.v1.json"
)
DEFAULT_TEMPLATE_MD_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_expansion_web_output_template_002.v1.md"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-operator-handoff", type=Path, default=DEFAULT_SOURCE_HANDOFF_PATH)
    parser.add_argument("--template-json", type=Path, default=DEFAULT_TEMPLATE_JSON_PATH)
    parser.add_argument("--template-md", type=Path, default=DEFAULT_TEMPLATE_MD_PATH)
    parser.add_argument("--template-id", default=DEFAULT_WEB_OUTPUT_TEMPLATE_ID)
    parser.add_argument("--json", action="store_true", help="Print template JSON to stdout.")
    parser.add_argument("--no-write", action="store_true", help="Build and validate without writing reports.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    handoff_path = args.source_operator_handoff.expanduser()
    handoff = load_json(handoff_path)
    report = build_visual_annotation_expansion_web_output_template(
        handoff,
        template_id=args.template_id,
        source_operator_handoff_ref=sanitized_report_ref(handoff_path, project_root=PROJECT_ROOT),
    )
    validation = validate_payload(
        report,
        VISUAL_ANNOTATION_EXPANSION_WEB_OUTPUT_TEMPLATE_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "visual annotation expansion web output template schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    if not args.no_write:
        paths = write_visual_annotation_expansion_web_output_template(
            report,
            report_json=args.template_json,
            report_md=args.template_md,
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
                    "targetOutput": report.get("targetOutput"),
                    "paths": paths,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0 if report.get("status") == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
