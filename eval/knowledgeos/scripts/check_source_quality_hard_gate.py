#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


SOURCE_ORDER = ("paper", "vault", "web")
READY_DECISION = "ready_for_hard_gate_review"
SCHEMA = "knowledge-hub.source-quality-hard-gate.result.v1"


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


def _passed_min(value: Any, threshold: float) -> bool:
    numeric = _as_float(value)
    return numeric is not None and numeric >= threshold


def _passed_max(value: Any, threshold: float) -> bool:
    numeric = _as_float(value)
    return numeric is not None and numeric <= threshold


def build_gate_result(
    observation: dict[str, Any],
    *,
    observation_path: str = "",
    route_threshold: float = 1.0,
    vault_stale_threshold: float = 0.0,
) -> dict[str, Any]:
    errors: list[str] = []
    metric_checks: list[dict[str, Any]] = []

    schema = _clean_text(observation.get("schema"))
    if schema != "knowledge-hub.source-quality-observation.report.v1":
        errors.append(f"schema_mismatch:{schema or 'missing'}")

    decision = _clean_text(observation.get("decision"))
    if decision != READY_DECISION:
        errors.append(f"decision_not_ready:{decision or 'missing'}")

    blockers = [str(item).strip() for item in list(observation.get("blockers") or []) if str(item).strip()]
    if blockers:
        errors.append("blockers_present")

    run_count = int(observation.get("run_count") or 0)
    required_runs = int(observation.get("required_runs") or 0)
    if required_runs <= 0:
        errors.append("required_runs_missing")
    elif run_count < required_runs:
        errors.append(f"insufficient_run_count:{run_count}/{required_runs}")

    sources = dict(observation.get("sources") or {})
    for source in SOURCE_ORDER:
        payload = dict(sources.get(source) or {})
        value = payload.get("route_correctness")
        passed = _passed_min(value, route_threshold)
        metric_checks.append(
            {
                "source": source,
                "metric": "route_correctness",
                "value": value,
                "threshold": route_threshold,
                "operator": ">=",
                "status": "pass" if passed else "fail",
            }
        )
        if not passed:
            errors.append(f"{source}_route_correctness_below_gate")

        for metric in ("legacy_runtime_rate", "capability_missing_rate"):
            metric_value = payload.get(metric)
            metric_passed = _passed_max(metric_value, 0.0)
            metric_checks.append(
                {
                    "source": source,
                    "metric": metric,
                    "value": metric_value,
                    "threshold": 0.0,
                    "operator": "<=",
                    "status": "pass" if metric_passed else "fail",
                }
            )
            if not metric_passed:
                errors.append(f"{source}_{metric}_above_gate")

    vault_payload = dict(sources.get("vault") or {})
    stale_value = vault_payload.get("stale_citation_rate")
    stale_passed = _passed_max(stale_value, vault_stale_threshold)
    metric_checks.append(
        {
            "source": "vault",
            "metric": "stale_citation_rate",
            "value": stale_value,
            "threshold": vault_stale_threshold,
            "operator": "<=",
            "status": "pass" if stale_passed else "fail",
        }
    )
    if not stale_passed:
        errors.append("vault_stale_citation_rate_above_gate")

    status = "ok" if not errors else "failed"
    return {
        "schema": SCHEMA,
        "status": status,
        "decision": decision,
        "blockers": blockers,
        "runCount": run_count,
        "requiredRuns": required_runs,
        "latestRunDir": _clean_text(observation.get("latest_run_dir")),
        "observationReportPath": observation_path,
        "routeThreshold": route_threshold,
        "vaultStaleThreshold": vault_stale_threshold,
        "checks": metric_checks,
        "errors": errors,
    }


def render_human(result: dict[str, Any]) -> str:
    lines = [
        f"source-quality hard gate: {result.get('status')}",
        f"decision: {result.get('decision')}",
        f"run_count: {result.get('runCount')} / {result.get('requiredRuns')}",
        f"latest_run_dir: {result.get('latestRunDir')}",
    ]
    blockers = list(result.get("blockers") or [])
    lines.append(f"blockers: {blockers if blockers else []}")
    errors = list(result.get("errors") or [])
    if errors:
        lines.append("errors:")
        lines.extend(f"- {error}" for error in errors)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail unless the source-quality observation report clears the hard gate.")
    parser.add_argument("--runs-root", default="eval/knowledgeos/runs")
    parser.add_argument("--observation-json", default="")
    parser.add_argument("--route-threshold", type=float, default=1.0)
    parser.add_argument("--vault-stale-threshold", type=float, default=0.0)
    parser.add_argument("--json", action="store_true", dest="as_json", default=False)
    args = parser.parse_args(argv)

    runs_root = Path(args.runs_root).expanduser().resolve()
    observation_path = (
        Path(args.observation_json).expanduser().resolve()
        if _clean_text(args.observation_json)
        else runs_root / "reports" / "source_quality_observation_latest.json"
    )
    try:
        observation = _read_json(observation_path)
        result = build_gate_result(
            observation,
            observation_path=str(observation_path),
            route_threshold=float(args.route_threshold),
            vault_stale_threshold=float(args.vault_stale_threshold),
        )
    except Exception as exc:  # noqa: BLE001
        result = {
            "schema": SCHEMA,
            "status": "failed",
            "decision": "",
            "blockers": [],
            "runCount": 0,
            "requiredRuns": 0,
            "latestRunDir": "",
            "observationReportPath": str(observation_path),
            "routeThreshold": float(args.route_threshold),
            "vaultStaleThreshold": float(args.vault_stale_threshold),
            "checks": [],
            "errors": [f"read_observation_failed:{exc}"],
        }

    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(render_human(result))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
