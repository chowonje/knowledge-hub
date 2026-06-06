#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


READY_DECISION = "ready_for_removal_tranche"
SCHEMA = "knowledge-hub.legacy-runtime-removal-gate.result.v1"
SOURCE_ORDER = ("paper", "vault", "web")
TREND_METRICS = ("legacy_runtime_rate", "capability_missing_rate", "forced_legacy_rate")


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _hits(readiness: dict[str, Any], group: str, category: str) -> list[dict[str, Any]]:
    return [
        dict(item or {})
        for item in list(
            (((readiness.get("callsites") or {}).get(group) or {}).get(category) or [])
        )
    ]


def _trend_values(readiness: dict[str, Any], source: str, metric: str) -> list[Any]:
    return list((((readiness.get("readiness_trends") or {}).get(source) or {}).get(metric) or []))


def build_gate_result(readiness: dict[str, Any], *, readiness_path: str = "") -> dict[str, Any]:
    errors: list[str] = []
    callsite_checks: list[dict[str, Any]] = []
    trend_checks: list[dict[str, Any]] = []

    schema = _clean_text(readiness.get("schema"))
    if schema != "knowledge-hub.legacy-runtime-readiness.report.v1":
        errors.append(f"schema_mismatch:{schema or 'missing'}")

    decision = _clean_text(readiness.get("decision"))
    if decision != READY_DECISION:
        errors.append(f"decision_not_ready:{decision or 'missing'}")

    run_count = int(readiness.get("run_count") or 0)
    required_run_count = int(readiness.get("required_run_count") or 0)
    if required_run_count <= 0:
        errors.append("required_run_count_missing")
    elif run_count < required_run_count:
        errors.append(f"insufficient_run_count:{run_count}/{required_run_count}")

    for group in ("legacy_runtime_symbol", "ask_v2_mode_legacy_literal"):
        for category in ("runtime", "eval", "scripts"):
            category_hits = _hits(readiness, group, category)
            callsite_checks.append(
                {
                    "group": group,
                    "category": category,
                    "hitCount": len(category_hits),
                    "status": "pass" if not category_hits else "fail",
                }
            )
            if category_hits:
                errors.append(f"{group}_{category}_hits_present")

    test_literal_hits = _hits(readiness, "ask_v2_mode_legacy_literal", "tests")
    for source in SOURCE_ORDER:
        for metric in TREND_METRICS:
            values = _trend_values(readiness, source, metric)
            numeric_values = [_as_float(value) for value in values]
            non_zero_values = [value for value in numeric_values if value is None or value != 0.0]
            trend_checks.append(
                {
                    "source": source,
                    "metric": metric,
                    "pointCount": len(values),
                    "values": values,
                    "status": "pass" if not non_zero_values and len(values) >= required_run_count > 0 else "fail",
                }
            )
            if non_zero_values:
                errors.append(f"{source}_{metric}_above_gate")
            if required_run_count > 0 and len(values) < required_run_count:
                errors.append(f"{source}_{metric}_insufficient_points:{len(values)}/{required_run_count}")

    status = "ok" if not errors else "failed"
    return {
        "schema": SCHEMA,
        "status": status,
        "decision": decision,
        "runCount": run_count,
        "requiredRunCount": required_run_count,
        "readinessReportPath": readiness_path,
        "callsiteChecks": callsite_checks,
        "trendChecks": trend_checks,
        "testOnlyLegacyLiteralHits": len(test_literal_hits),
        "errors": errors,
    }


def render_human(result: dict[str, Any]) -> str:
    lines = [
        f"legacy runtime removal gate: {result.get('status')}",
        f"decision: {result.get('decision')}",
        f"run_count: {result.get('runCount')} / {result.get('requiredRunCount')}",
        f"readiness_report: {result.get('readinessReportPath')}",
        f"test_only_legacy_literal_hits: {result.get('testOnlyLegacyLiteralHits')}",
    ]
    errors = list(result.get("errors") or [])
    if errors:
        lines.append("errors:")
        lines.extend(f"- {error}" for error in errors)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail unless the legacy answer runtime removal gate is clear.")
    parser.add_argument("--runs-root", default="eval/knowledgeos/runs")
    parser.add_argument("--readiness-json", default="")
    parser.add_argument("--json", action="store_true", dest="as_json", default=False)
    args = parser.parse_args(argv)

    runs_root = Path(args.runs_root).expanduser().resolve()
    readiness_path = (
        Path(args.readiness_json).expanduser().resolve()
        if _clean_text(args.readiness_json)
        else runs_root / "reports" / "legacy_runtime_readiness_latest.json"
    )
    try:
        readiness = _read_json(readiness_path)
        result = build_gate_result(readiness, readiness_path=str(readiness_path))
    except Exception as exc:  # noqa: BLE001
        result = {
            "schema": SCHEMA,
            "status": "failed",
            "decision": "",
            "runCount": 0,
            "requiredRunCount": 0,
            "readinessReportPath": str(readiness_path),
            "callsiteChecks": [],
            "trendChecks": [],
            "testOnlyLegacyLiteralHits": 0,
            "errors": [f"readiness_read_failed:{exc}"],
        }

    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(render_human(result))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
