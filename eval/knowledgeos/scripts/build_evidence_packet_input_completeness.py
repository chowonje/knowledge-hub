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

DEFAULT_JSON_PATH = PROJECT_ROOT / "eval/knowledgeos/reports/evidence_packet_input_completeness.v1.json"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build report-only Evidence Packet input completeness rows.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-write-report", action="store_true")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_raw_payloads(manifest: dict[str, object], raw_dir: Path) -> dict[str, dict[str, object]]:
    payloads: dict[str, dict[str, object]] = {}
    for run in list(manifest.get("runs") or []):
        if not isinstance(run, dict):
            continue
        run_id = str(run.get("runId") or "").strip()
        if not run_id:
            continue
        path = raw_dir / f"{run_id}.json"
        payloads[run_id] = _load_json(path) if path.exists() else {}
    return payloads


def main(argv: list[str] | None = None) -> int:
    from knowledge_hub.ai.evidence_packet_input_completeness import (
        EVIDENCE_PACKET_INPUT_COMPLETENESS_SCHEMA_ID,
        build_evidence_packet_input_completeness,
        write_evidence_packet_input_completeness,
    )
    from knowledge_hub.core.schema_validator import validate_payload

    args = _parse_args(list(argv or sys.argv[1:]))
    generated_at = datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    manifest = _load_json(args.manifest)
    report = build_evidence_packet_input_completeness(
        manifest=manifest,
        raw_payloads=_load_raw_payloads(manifest, args.raw_dir),
        generated_at=generated_at,
    )
    validation = validate_payload(report, EVIDENCE_PACKET_INPUT_COMPLETENESS_SCHEMA_ID, strict=True)
    if not validation.ok:
        raise RuntimeError("packet input completeness schema validation failed: " + "; ".join(validation.errors))
    paths = {} if args.no_write_report else write_evidence_packet_input_completeness(report, report_json=args.report_json)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps({"status": report.get("status"), "counts": report.get("counts"), "paths": paths}, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
