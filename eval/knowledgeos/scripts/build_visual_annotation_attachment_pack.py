#!/usr/bin/env python3
"""Build local PNG attachments for visual annotation web pack 001."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.visual_annotation_attachment_pack import (
    DEFAULT_ATTACHMENT_PACK_ID,
    VISUAL_ANNOTATION_ATTACHMENT_PACK_SCHEMA_ID,
    build_visual_annotation_attachment_pack,
    default_papers_root,
    load_web_pack,
    sanitized_report_ref,
    write_visual_annotation_attachment_pack,
)


DEFAULT_SOURCE_WEB_PACK_PATH = PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_web_pack_001.v1.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_attachment_pack_001"
DEFAULT_JSON_PATH = PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_attachment_pack_001.v1.json"
DEFAULT_MD_PATH = PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_attachment_pack_001.v1.md"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-web-pack", type=Path, default=DEFAULT_SOURCE_WEB_PACK_PATH)
    parser.add_argument("--papers-root", type=Path, default=default_papers_root())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--attachment-pack-id", default=DEFAULT_ATTACHMENT_PACK_ID)
    parser.add_argument("--zoom", type=float, default=2.0)
    parser.add_argument("--json", action="store_true", help="Print report JSON to stdout.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    source_web_pack_path = args.source_web_pack.expanduser()
    output_dir = args.output_dir.expanduser()
    web_pack = load_web_pack(source_web_pack_path)
    report = build_visual_annotation_attachment_pack(
        web_pack,
        papers_root=args.papers_root.expanduser(),
        output_dir=output_dir,
        output_dir_ref=sanitized_report_ref(output_dir, project_root=PROJECT_ROOT),
        attachment_pack_id=args.attachment_pack_id,
        source_web_pack_ref=sanitized_report_ref(source_web_pack_path, project_root=PROJECT_ROOT),
        zoom=args.zoom,
    )
    validation = validate_payload(report, VISUAL_ANNOTATION_ATTACHMENT_PACK_SCHEMA_ID, strict=True)
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "visual annotation attachment pack schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    paths = write_visual_annotation_attachment_pack(
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
                    "counts": report.get("counts"),
                    "assetRootRef": report.get("assetRootRef"),
                    "paths": paths,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0 if report.get("status") == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
