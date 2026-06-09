#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_JSON_PATH = PROJECT_ROOT / "eval/knowledgeos/reports/paper_understanding_profile_readiness.v1.json"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build report-only Paper Understanding Profile readiness rows from sanitized packet input."
    )
    parser.add_argument("--packet-input-report", type=Path, required=True)
    parser.add_argument("--readback-report", type=Path)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-write-report", action="store_true")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    from knowledge_hub.ai.paper_understanding_profile_readiness import (
        PAPER_UNDERSTANDING_PROFILE_READINESS_SCHEMA_ID,
        build_paper_understanding_profile_readiness,
        write_paper_understanding_profile_readiness,
    )
    from knowledge_hub.core.schema_validator import validate_payload

    args = _parse_args(list(argv or sys.argv[1:]))
    generated_at = datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    report = build_paper_understanding_profile_readiness(
        packet_input_report=_load_json(args.packet_input_report),
        readback_report=_load_json(args.readback_report) if args.readback_report else None,
        generated_at=generated_at,
    )
    validation = validate_payload(report, PAPER_UNDERSTANDING_PROFILE_READINESS_SCHEMA_ID, strict=True)
    if not validation.ok:
        raise RuntimeError("paper understanding readiness schema validation failed: " + "; ".join(validation.errors))
    paths = {} if args.no_write_report else write_paper_understanding_profile_readiness(report, report_json=args.report_json)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {"status": report.get("status"), "counts": report.get("counts"), "paths": paths},
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0 if report.get("status") == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
