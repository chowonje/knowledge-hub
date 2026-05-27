#!/usr/bin/env python3
"""Build the manual web/VLM run packet for expansion attachment pack 002."""

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
    DEFAULT_MANUAL_RUN_PACKET_ID,
    VISUAL_ANNOTATION_EXPANSION_MANUAL_RUN_PACKET_SCHEMA_ID,
    build_visual_annotation_expansion_manual_run_packet,
    load_json,
    sanitized_report_ref,
    write_visual_annotation_expansion_manual_run_packet,
)


DEFAULT_SOURCE_EXPANSION_PACK_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_expansion_pack_design.v1.json"
)
DEFAULT_SOURCE_ATTACHMENT_PACK_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002.v1.json"
)
DEFAULT_PACKET_JSON_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_expansion_manual_run_packet_002.v1.json"
)
DEFAULT_PACKET_MD_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_expansion_manual_run_packet_002.v1.md"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-expansion-pack", type=Path, default=DEFAULT_SOURCE_EXPANSION_PACK_PATH)
    parser.add_argument("--source-attachment-pack", type=Path, default=DEFAULT_SOURCE_ATTACHMENT_PACK_PATH)
    parser.add_argument("--packet-json", type=Path, default=DEFAULT_PACKET_JSON_PATH)
    parser.add_argument("--packet-md", type=Path, default=DEFAULT_PACKET_MD_PATH)
    parser.add_argument("--packet-id", default=DEFAULT_MANUAL_RUN_PACKET_ID)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--json", action="store_true", help="Print packet JSON to stdout.")
    parser.add_argument("--no-write", action="store_true", help="Build and validate without writing reports.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    expansion_pack_path = args.source_expansion_pack.expanduser()
    attachment_pack_path = args.source_attachment_pack.expanduser()
    expansion_pack = load_json(expansion_pack_path)
    attachment_pack = load_json(attachment_pack_path)
    report = build_visual_annotation_expansion_manual_run_packet(
        expansion_pack,
        attachment_pack,
        packet_id=args.packet_id,
        source_expansion_pack_ref=sanitized_report_ref(expansion_pack_path, project_root=PROJECT_ROOT),
        source_attachment_pack_ref=sanitized_report_ref(attachment_pack_path, project_root=PROJECT_ROOT),
        batch_size=args.batch_size,
    )
    validation = validate_payload(
        report,
        VISUAL_ANNOTATION_EXPANSION_MANUAL_RUN_PACKET_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "visual annotation expansion manual run packet schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    if not args.no_write:
        paths = write_visual_annotation_expansion_manual_run_packet(
            report,
            report_json=args.packet_json,
            report_md=args.packet_md,
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
