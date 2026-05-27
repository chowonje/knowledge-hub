#!/usr/bin/env python3
"""Collect and validate per-batch manual web/VLM outputs for expansion annotation."""

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
    DEFAULT_WEB_BATCH_OUTPUT_COLLECTOR_ID,
    VISUAL_ANNOTATION_EXPANSION_WEB_BATCH_OUTPUT_COLLECTOR_SCHEMA_ID,
    VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID,
    build_visual_annotation_expansion_web_batch_output_collector,
    combine_visual_annotation_expansion_web_batch_outputs,
    load_json,
    sanitized_report_ref,
    write_visual_annotation_expansion_web_batch_output_collector,
)


DEFAULT_SOURCE_BUNDLE_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_expansion_web_run_bundle_002.v1.json"
)
DEFAULT_BATCH_OUTPUT_DIR = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_expansion_web_batch_outputs_002"
)
DEFAULT_COLLECTOR_JSON_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_expansion_web_batch_output_collector_002.v1.json"
)
DEFAULT_COLLECTOR_MD_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_expansion_web_batch_output_collector_002.v1.md"
)
DEFAULT_COMBINED_OUTPUT_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_expansion_web_output_002.manual.json"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-web-run-bundle", type=Path, default=DEFAULT_SOURCE_BUNDLE_PATH)
    parser.add_argument("--batch-output-dir", type=Path, default=DEFAULT_BATCH_OUTPUT_DIR)
    parser.add_argument("--collector-json", type=Path, default=DEFAULT_COLLECTOR_JSON_PATH)
    parser.add_argument("--collector-md", type=Path, default=DEFAULT_COLLECTOR_MD_PATH)
    parser.add_argument("--combined-output", type=Path, default=DEFAULT_COMBINED_OUTPUT_PATH)
    parser.add_argument("--collector-id", default=DEFAULT_WEB_BATCH_OUTPUT_COLLECTOR_ID)
    parser.add_argument("--write-combined-output", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print collector report JSON to stdout.")
    parser.add_argument("--no-write", action="store_true", help="Build and validate without writing reports.")
    return parser.parse_args(argv)


def _batch_output_ref(batch_output_dir_ref: str, batch_number: int) -> str:
    return f"{batch_output_dir_ref}/batch_{batch_number:02d}_web_output.manual.json"


def _load_present_batch_outputs(
    web_run_bundle: dict[str, object],
    *,
    batch_output_dir: Path,
    batch_output_dir_ref: str,
) -> dict[str, dict[str, object]]:
    outputs: dict[str, dict[str, object]] = {}
    for batch in list(web_run_bundle.get("batchBundles") or []):
        if not isinstance(batch, dict):
            continue
        batch_number = int(batch.get("batchNumber") or 0)
        ref = _batch_output_ref(batch_output_dir_ref, batch_number)
        path = batch_output_dir / f"batch_{batch_number:02d}_web_output.manual.json"
        if path.exists():
            outputs[ref] = load_json(path)
    return outputs


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    bundle_path = args.source_web_run_bundle.expanduser()
    batch_output_dir = args.batch_output_dir.expanduser()
    batch_output_dir_ref = sanitized_report_ref(batch_output_dir, project_root=PROJECT_ROOT)
    web_run_bundle = load_json(bundle_path)
    batch_outputs = _load_present_batch_outputs(
        web_run_bundle,
        batch_output_dir=batch_output_dir,
        batch_output_dir_ref=batch_output_dir_ref,
    )
    report = build_visual_annotation_expansion_web_batch_output_collector(
        web_run_bundle,
        batch_outputs,
        collector_id=args.collector_id,
        source_web_run_bundle_ref=sanitized_report_ref(bundle_path, project_root=PROJECT_ROOT),
        batch_output_dir_ref=batch_output_dir_ref,
        target_output_ref=sanitized_report_ref(args.combined_output, project_root=PROJECT_ROOT),
    )
    validation = validate_payload(
        report,
        VISUAL_ANNOTATION_EXPANSION_WEB_BATCH_OUTPUT_COLLECTOR_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "visual annotation expansion web batch output collector schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    if args.write_combined_output:
        if report.get("status") != "ready":
            raise ValueError("cannot write combined output until all batch outputs are present and valid")
        combined_output = combine_visual_annotation_expansion_web_batch_outputs(
            web_run_bundle,
            batch_outputs,
            batch_output_dir_ref=batch_output_dir_ref,
        )
        combined_validation = validate_payload(
            combined_output,
            VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID,
            strict=True,
        )
        if not combined_validation.ok:
            raise ValueError(
                "combined visual annotation expansion web output schema validation failed: "
                + "; ".join(str(error) for error in combined_validation.errors)
            )
        args.combined_output.parent.mkdir(parents=True, exist_ok=True)
        args.combined_output.write_text(
            json.dumps(combined_output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if not args.no_write:
        paths = write_visual_annotation_expansion_web_batch_output_collector(
            report,
            report_json=args.collector_json,
            report_md=args.collector_md,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
