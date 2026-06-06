#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


READY_DECISION = "ready_for_detail_gate_review"
SCHEMA = "knowledge-hub.source-quality-detail-gate.result.v1"
DETAIL_METRICS = (
    {
        "source": "paper",
        "metric": "paper_citation_correctness",
        "threshold": 1.0,
        "operator": ">=",
    },
    {
        "source": "vault",
        "metric": "vault_abstention_correctness",
        "threshold": 1.0,
        "operator": ">=",
    },
    {
        "source": "web",
        "metric": "web_recency_violation",
        "threshold": 0.0,
        "operator": "<=",
    },
)


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


def _passes_gate(value: Any, *, operator: str, threshold: float) -> bool:
    numeric = _as_float(value)
    if numeric is None:
        return False
    if operator == ">=":
        return numeric >= threshold
    if operator == "<=":
        return numeric <= threshold
    raise ValueError(f"unsupported operator: {operator}")


def _error_code(*, source: str, metric: str, operator: str) -> str:
    suffix = "below_gate" if operator == ">=" else "above_gate"
    return f"{source}_{metric}_{suffix}"


def build_gate_result(
    observation: dict[str, Any],
    *,
    observation_path: str = "",
) -> dict[str, Any]:
    errors: list[str] = []
    metric_checks: list[dict[str, Any]] = []

    schema = _clean_text(observation.get("schema"))
    if schema != "knowledge-hub.source-quality-detail-observation.report.v1":
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

    base_decision = _clean_text(observation.get("base_observation_decision"))
    if base_decision != "ready_for_hard_gate_review":
        errors.append(f"base_observation_not_ready:{base_decision or 'missing'}")

    check_lookup = {
        (_clean_text(item.get("source")), _clean_text(item.get("metric"))): dict(item)
        for item in list(observation.get("checks") or [])
    }
    for metric_spec in DETAIL_METRICS:
        source = metric_spec["source"]
        metric = metric_spec["metric"]
        operator = metric_spec["operator"]
        threshold = float(metric_spec["threshold"])
        payload = check_lookup.get((source, metric), {})
        value = payload.get("latest")
        point_count = int(payload.get("numericPointCount") or 0)
        check_status = _clean_text(payload.get("status"))
        check_blockers = [str(item).strip() for item in list(payload.get("blockers") or []) if str(item).strip()]
        passed = _passes_gate(value, operator=operator, threshold=threshold) and point_count >= required_runs > 0
        detail_passed = passed and (not check_status or check_status == "pass") and not check_blockers
        metric_checks.append(
            {
                "source": source,
                "metric": metric,
                "value": value,
                "threshold": threshold,
                "operator": operator,
                "numericPointCount": point_count,
                "detailStatus": check_status,
                "blockers": check_blockers,
                "status": "pass" if detail_passed else "fail",
            }
        )
        if check_status and check_status != "pass":
            errors.append(f"{source}_{metric}_status_{check_status}")
        if check_blockers:
            errors.append(f"{source}_{metric}_blockers_present")
        if not passed:
            errors.append(_error_code(source=source, metric=metric, operator=operator))

    status = "ok" if not errors else "failed"
    return {
        "schema": SCHEMA,
        "status": status,
        "decision": decision,
        "blockers": blockers,
        "runCount": run_count,
        "requiredRuns": required_runs,
        "latestRunDir": _clean_text(observation.get("latest_run_dir")),
        "baseObservationDecision": base_decision,
        "observationReportPath": observation_path,
        "checks": metric_checks,
        "errors": errors,
    }


def render_human(result: dict[str, Any]) -> str:
    lines = [
        f"source-quality detail gate: {result.get('status')}",
        f"decision: {result.get('decision')}",
        f"base_observation_decision: {result.get('baseObservationDecision')}",
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
    parser = argparse.ArgumentParser(description="Fail unless the source-quality detail observation clears the promotion gate.")
    parser.add_argument("--runs-root", default="eval/knowledgeos/runs")
    parser.add_argument("--observation-json", default="")
    parser.add_argument("--json", action="store_true", dest="as_json", default=False)
    args = parser.parse_args(argv)

    runs_root = Path(args.runs_root).expanduser().resolve()
    observation_path = (
        Path(args.observation_json).expanduser().resolve()
        if _clean_text(args.observation_json)
        else runs_root / "reports" / "source_quality_detail_observation_latest.json"
    )
    try:
        observation = _read_json(observation_path)
        result = build_gate_result(observation, observation_path=str(observation_path))
    except Exception as exc:  # noqa: BLE001
        result = {
            "schema": SCHEMA,
            "status": "failed",
            "decision": "",
            "blockers": [],
            "runCount": 0,
            "requiredRuns": 0,
            "latestRunDir": "",
            "baseObservationDecision": "",
            "observationReportPath": str(observation_path),
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
