#!/usr/bin/env python3
"""Build per-batch web-run prompts and fill templates for expansion annotation."""

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
    DEFAULT_WEB_RUN_BUNDLE_ID,
    VISUAL_ANNOTATION_EXPANSION_WEB_RUN_BATCH_TEMPLATE_SCHEMA_ID,
    VISUAL_ANNOTATION_EXPANSION_WEB_RUN_BUNDLE_SCHEMA_ID,
    build_visual_annotation_expansion_web_run_batch_template,
    build_visual_annotation_expansion_web_run_bundle,
    load_json,
    sanitized_report_ref,
    write_visual_annotation_expansion_web_run_bundle,
)


DEFAULT_SOURCE_TEMPLATE_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_expansion_web_output_template_002.v1.json"
)
DEFAULT_BUNDLE_JSON_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_expansion_web_run_bundle_002.v1.json"
)
DEFAULT_BUNDLE_MD_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_expansion_web_run_bundle_002.v1.md"
)
DEFAULT_BUNDLE_DIR = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_expansion_web_run_bundle_002"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-web-output-template", type=Path, default=DEFAULT_SOURCE_TEMPLATE_PATH)
    parser.add_argument("--bundle-json", type=Path, default=DEFAULT_BUNDLE_JSON_PATH)
    parser.add_argument("--bundle-md", type=Path, default=DEFAULT_BUNDLE_MD_PATH)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--bundle-id", default=DEFAULT_WEB_RUN_BUNDLE_ID)
    parser.add_argument(
        "--target-output-ref",
        default="eval/knowledgeos/reports/visual_annotation_expansion_web_output_002.manual.json",
    )
    parser.add_argument(
        "--validation-command",
        default="PYTHONPATH=. python eval/knowledgeos/scripts/validate_visual_annotation_expansion_web_output.py",
    )
    parser.add_argument("--json", action="store_true", help="Print bundle JSON to stdout.")
    parser.add_argument("--no-write", action="store_true", help="Build and validate without writing reports.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    template_path = args.source_web_output_template.expanduser()
    template = load_json(template_path)
    bundle_dir_ref = sanitized_report_ref(args.bundle_dir, project_root=PROJECT_ROOT)
    report = build_visual_annotation_expansion_web_run_bundle(
        template,
        bundle_id=args.bundle_id,
        source_web_output_template_ref=sanitized_report_ref(template_path, project_root=PROJECT_ROOT),
        bundle_dir_ref=bundle_dir_ref,
        target_output_ref=args.target_output_ref,
        validation_command=args.validation_command,
    )
    validation = validate_payload(
        report,
        VISUAL_ANNOTATION_EXPANSION_WEB_RUN_BUNDLE_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "visual annotation expansion web run bundle schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    for batch in report.get("batchBundles", []):
        batch_template = build_visual_annotation_expansion_web_run_batch_template(batch)
        batch_validation = validate_payload(
            batch_template,
            VISUAL_ANNOTATION_EXPANSION_WEB_RUN_BATCH_TEMPLATE_SCHEMA_ID,
            strict=True,
        )
        if not batch_validation.ok:
            raise ValueError(
                "visual annotation expansion web run batch template schema validation failed: "
                + "; ".join(str(error) for error in batch_validation.errors)
            )
    if not args.no_write:
        paths = write_visual_annotation_expansion_web_run_bundle(
            report,
            report_json=args.bundle_json,
            report_md=args.bundle_md,
            bundle_dir=args.bundle_dir,
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
