#!/usr/bin/env python3
"""Build the operator handoff for expansion web/VLM output assembly."""

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
    DEFAULT_OPERATOR_HANDOFF_ID,
    VISUAL_ANNOTATION_EXPANSION_OPERATOR_HANDOFF_SCHEMA_ID,
    build_visual_annotation_expansion_operator_handoff,
    load_json,
    sanitized_report_ref,
    write_visual_annotation_expansion_operator_handoff,
)


DEFAULT_SOURCE_PACKET_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_expansion_manual_run_packet_002.v1.json"
)
DEFAULT_HANDOFF_JSON_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_expansion_operator_handoff_002.v1.json"
)
DEFAULT_HANDOFF_MD_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_annotation_expansion_operator_handoff_002.v1.md"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manual-run-packet", type=Path, default=DEFAULT_SOURCE_PACKET_PATH)
    parser.add_argument("--handoff-json", type=Path, default=DEFAULT_HANDOFF_JSON_PATH)
    parser.add_argument("--handoff-md", type=Path, default=DEFAULT_HANDOFF_MD_PATH)
    parser.add_argument("--handoff-id", default=DEFAULT_OPERATOR_HANDOFF_ID)
    parser.add_argument(
        "--expected-output-ref",
        default="eval/knowledgeos/reports/visual_annotation_expansion_web_output_002.manual.json",
    )
    parser.add_argument(
        "--validation-command",
        default="PYTHONPATH=. python eval/knowledgeos/scripts/validate_visual_annotation_expansion_web_output.py",
    )
    parser.add_argument("--json", action="store_true", help="Print handoff JSON to stdout.")
    parser.add_argument("--no-write", action="store_true", help="Build and validate without writing reports.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    packet_path = args.source_manual_run_packet.expanduser()
    packet = load_json(packet_path)
    report = build_visual_annotation_expansion_operator_handoff(
        packet,
        handoff_id=args.handoff_id,
        source_manual_run_packet_ref=sanitized_report_ref(packet_path, project_root=PROJECT_ROOT),
        expected_output_ref=args.expected_output_ref,
        validation_command=args.validation_command,
    )
    validation = validate_payload(
        report,
        VISUAL_ANNOTATION_EXPANSION_OPERATOR_HANDOFF_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "visual annotation expansion operator handoff schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    if not args.no_write:
        paths = write_visual_annotation_expansion_operator_handoff(
            report,
            report_json=args.handoff_json,
            report_md=args.handoff_md,
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
                    "expectedOutputRef": report.get("expectedOutputRef"),
                    "paths": paths,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0 if report.get("status") == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
