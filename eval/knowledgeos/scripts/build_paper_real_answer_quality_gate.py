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

from knowledge_hub.ai.paper_real_answer_quality_gate import (  # noqa: E402
    PAPER_REAL_ANSWER_QUALITY_GATE_SCHEMA_ID,
    build_paper_real_answer_quality_gate,
    write_paper_real_answer_quality_gate,
)
from knowledge_hub.core.schema_validator import validate_payload  # noqa: E402

DEFAULT_REPORT_JSON = PROJECT_ROOT / "eval/knowledgeos/reports/paper_real_answer_quality_gate.v1.json"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the bounded paper real-answer quality gate report.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--no-write-report", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    report = build_paper_real_answer_quality_gate(
        answer_payload_report=_read_json(args.input_json),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    validation = validate_payload(report, PAPER_REAL_ANSWER_QUALITY_GATE_SCHEMA_ID, strict=True)
    if not validation.ok:
        report["counts"]["schemaViolationCount"] = len(validation.errors)
        raise ValueError("paper real answer quality gate schema validation failed: " + "; ".join(validation.errors))
    paths: dict[str, str] = {}
    if not args.no_write_report:
        paths = write_paper_real_answer_quality_gate(report, report_json=args.report_json)
    payload = report if args.as_json else {"status": report["status"], "counts": report["counts"], "paths": paths}
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
