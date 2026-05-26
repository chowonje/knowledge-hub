#!/usr/bin/env python3
"""Validate manually supplied visual annotation web/VLM output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.visual_annotation_manual_output_capture import (
    VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID,
    VISUAL_ANNOTATION_WEB_OUTPUT_VALIDATION_SCHEMA_ID,
    build_visual_annotation_web_output_validation,
    load_json,
    sanitized_report_ref,
    write_visual_annotation_web_output_validation,
)


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_web_output_001.manual.json"
DEFAULT_SOURCE_WEB_PACK_PATH = PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_web_pack_001.v1.json"
DEFAULT_SOURCE_ATTACHMENT_PACK_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_attachment_pack_001.v1.json"
)
DEFAULT_VALIDATION_JSON_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_web_output_001.validation.v1.json"
)
DEFAULT_VALIDATION_MD_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_web_output_001.validation.v1.md"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--source-web-pack", type=Path, default=DEFAULT_SOURCE_WEB_PACK_PATH)
    parser.add_argument(
        "--source-attachment-pack",
        type=Path,
        default=DEFAULT_SOURCE_ATTACHMENT_PACK_PATH,
    )
    parser.add_argument("--validation-json", type=Path, default=DEFAULT_VALIDATION_JSON_PATH)
    parser.add_argument("--validation-md", type=Path, default=DEFAULT_VALIDATION_MD_PATH)
    parser.add_argument("--json", action="store_true", help="Print validation report JSON to stdout.")
    parser.add_argument("--no-write", action="store_true", help="Validate without writing reports.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    output_path = args.output.expanduser()
    web_pack_path = args.source_web_pack.expanduser()
    attachment_pack_path = args.source_attachment_pack.expanduser()

    output = load_json(output_path)
    web_pack = load_json(web_pack_path)
    attachment_pack = load_json(attachment_pack_path)

    output_validation = validate_payload(output, VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID, strict=True)
    report = build_visual_annotation_web_output_validation(
        output,
        web_pack,
        attachment_pack,
        output_ref=sanitized_report_ref(output_path, project_root=PROJECT_ROOT),
        source_web_pack_ref=sanitized_report_ref(web_pack_path, project_root=PROJECT_ROOT),
        source_attachment_pack_ref=sanitized_report_ref(attachment_pack_path, project_root=PROJECT_ROOT),
    )
    if not output_validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(output_validation.errors)

    report_validation = validate_payload(
        report,
        VISUAL_ANNOTATION_WEB_OUTPUT_VALIDATION_SCHEMA_ID,
        strict=True,
    )
    if not report_validation.ok:
        raise ValueError(
            "visual annotation web output validation report schema failed: "
            + "; ".join(str(error) for error in report_validation.errors)
        )

    if not args.no_write:
        paths = write_visual_annotation_web_output_validation(
            report,
            report_json=args.validation_json,
            report_md=args.validation_md,
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
