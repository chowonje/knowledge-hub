#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.ai.paper_answer_quality_harness import (  # noqa: E402
    PAPER_ANSWER_QUALITY_HARNESS_SCHEMA_ID,
    build_paper_answer_quality_harness,
    write_paper_answer_quality_harness,
)
from knowledge_hub.core.schema_validator import validate_payload  # noqa: E402

DEFAULT_PACKET_REPORT = PROJECT_ROOT / "eval/knowledgeos/reports/evidence_packet_input_completeness.v1.json"
DEFAULT_READBACK_REPORT = PROJECT_ROOT / "eval/knowledgeos/reports/paper_understanding_readback.v1.json"
DEFAULT_PROFILE_REPORT = PROJECT_ROOT / "eval/knowledgeos/reports/paper_understanding_profile_readiness.v1.json"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "eval/knowledgeos/reports/paper_answer_quality_harness.v1.json"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the bounded paper answer quality harness report.")
    parser.add_argument("--packet-input-report", type=Path, default=DEFAULT_PACKET_REPORT)
    parser.add_argument("--readback-report", type=Path, default=DEFAULT_READBACK_REPORT)
    parser.add_argument("--profile-readiness-report", type=Path, default=DEFAULT_PROFILE_REPORT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--no-write-report", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    report = build_paper_answer_quality_harness(
        packet_input_report=_read_json(args.packet_input_report),
        readback_report=_read_json(args.readback_report),
        profile_readiness_report=_read_json(args.profile_readiness_report),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    validation = validate_payload(report, PAPER_ANSWER_QUALITY_HARNESS_SCHEMA_ID, strict=True)
    if not validation.ok:
        report["counts"]["schemaViolationCount"] = len(validation.errors)
        raise ValueError("paper answer quality harness schema validation failed: " + "; ".join(validation.errors))
    paths: dict[str, str] = {}
    if not args.no_write_report:
        paths = write_paper_answer_quality_harness(report, report_json=args.report_json)
    payload = report if args.as_json else {"status": report["status"], "counts": report["counts"], "paths": paths}
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
