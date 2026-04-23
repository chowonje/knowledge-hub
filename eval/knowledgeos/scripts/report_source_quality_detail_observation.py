#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "knowledge-hub.source-quality-detail-observation.report.v1"
READY_DECISION = "ready_for_detail_gate_review"
SOURCE_ORDER = ("paper", "vault", "web")
DETAIL_METRICS = (
    {
        "source": "paper",
        "group": "soft",
        "metric": "paper_citation_correctness",
        "label": "paper citation correctness",
        "operator": ">=",
        "threshold": 1.0,
        "gate": "evidence",
        "promotion_candidate": True,
    },
    {
        "source": "vault",
        "group": "soft",
        "metric": "vault_abstention_correctness",
        "label": "vault abstention correctness",
        "operator": ">=",
        "threshold": 1.0,
        "gate": "answer/evidence",
        "promotion_candidate": True,
    },
    {
        "source": "web",
        "group": "soft",
        "metric": "web_recency_violation",
        "label": "web recency violation",
        "operator": "<=",
        "threshold": 0.0,
        "gate": "temporal evidence",
        "promotion_candidate": True,
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


def _metric_passes(value: float, *, operator: str, threshold: float) -> bool:
    if operator == ">=":
        return value >= threshold
    if operator == "<=":
        return value <= threshold
    raise ValueError(f"unsupported operator: {operator}")


def _metric_payload(trend_report: dict[str, Any], spec: dict[str, Any], *, required_runs: int) -> tuple[dict[str, Any], list[str]]:
    source = str(spec["source"])
    group = str(spec["group"])
    metric = str(spec["metric"])
    metric_report = (
        ((trend_report.get("sources") or {}).get(source) or {})
        .get(group, {})
        .get(metric, {})
    )
    points = list(metric_report.get("points") or [])
    numeric_points = []
    for point in points:
        value = _as_float(dict(point or {}).get("value"))
        if value is not None:
            numeric_points.append(
                {
                    "created_at": dict(point or {}).get("created_at"),
                    "run_dir": dict(point or {}).get("run_dir"),
                    "value": value,
                }
            )

    blockers: list[str] = []
    if len(points) < required_runs:
        blockers.append(f"{source}_{metric}_need_{required_runs}_points_have_{len(points)}")
    if not numeric_points:
        blockers.append(f"{source}_{metric}_unobserved")
    elif len(numeric_points) < required_runs:
        blockers.append(f"{source}_{metric}_need_{required_runs}_numeric_points_have_{len(numeric_points)}")

    failed_values = [
        point for point in numeric_points if not _metric_passes(point["value"], operator=str(spec["operator"]), threshold=float(spec["threshold"]))
    ]
    if failed_values:
        blockers.append(f"{source}_{metric}_not_stable")

    status = "pass" if not blockers else "blocked"
    return (
        {
            "source": source,
            "metric": metric,
            "label": spec["label"],
            "gate": spec["gate"],
            "operator": spec["operator"],
            "threshold": spec["threshold"],
            "promotionCandidate": bool(spec["promotion_candidate"]),
            "status": status,
            "latest": metric_report.get("latest"),
            "min": metric_report.get("min"),
            "max": metric_report.get("max"),
            "pointCount": len(points),
            "numericPointCount": len(numeric_points),
            "points": numeric_points,
            "blockers": blockers,
        },
        blockers,
    )


def build_report(
    *,
    trend_report: dict[str, Any],
    base_observation_report: dict[str, Any],
    required_runs: int,
) -> dict[str, Any]:
    blockers: list[str] = []
    run_count = int(trend_report.get("run_count") or 0)
    if run_count < required_runs:
        blockers.append(f"need_{required_runs}_runs_have_{run_count}")

    base_decision = _clean_text(base_observation_report.get("decision"))
    if base_decision != "ready_for_hard_gate_review":
        blockers.append(f"base_source_quality_not_ready:{base_decision or 'missing'}")
    base_blockers = [str(item).strip() for item in list(base_observation_report.get("blockers") or []) if str(item).strip()]
    if base_blockers:
        blockers.append("base_source_quality_blockers_present")

    checks = []
    for spec in DETAIL_METRICS:
        check, metric_blockers = _metric_payload(trend_report, spec, required_runs=required_runs)
        checks.append(check)
        blockers.extend(metric_blockers)

    unique_blockers = list(dict.fromkeys(blockers))
    if run_count < required_runs:
        decision = "observe_more"
    elif unique_blockers:
        decision = "not_ready_for_detail_gate_review"
    else:
        decision = READY_DECISION

    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "blockers": unique_blockers,
        "run_count": run_count,
        "required_runs": required_runs,
        "latest_run_dir": trend_report.get("latest_run_dir"),
        "base_observation_decision": base_decision,
        "base_observation_blockers": base_blockers,
        "checks": checks,
        "promotion_scope": "detail_quality_candidates",
        "promotion_note": (
            "This report observes detail-quality candidates only. Promotion should be a later explicit change "
            "after every candidate check is stable across the required window."
        ),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Source Quality Detail Observation",
        "",
        f"- decision: `{report['decision']}`",
        f"- run_count: `{report['run_count']}` / `{report['required_runs']}`",
        f"- latest_run: `{report.get('latest_run_dir') or ''}`",
        f"- base_observation_decision: `{report.get('base_observation_decision') or ''}`",
        "",
        "## Detail Checks",
    ]
    for check in list(report.get("checks") or []):
        payload = dict(check or {})
        lines.append(
            f"- {payload.get('source')}.{payload.get('metric')}: status=`{payload.get('status')}`, "
            f"latest=`{_fmt(payload.get('latest'))}`, threshold=`{payload.get('operator')} {payload.get('threshold')}`, "
            f"numeric_points=`{payload.get('numericPointCount')}/{report['required_runs']}`"
        )
    lines.extend(["", "## Blockers"])
    blockers = list(report.get("blockers") or [])
    if blockers:
        lines.extend(f"- {blocker}" for blocker in blockers)
    else:
        lines.append("- none")
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: dict[str, Any], *, out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the detail-quality observation verdict above source-quality trend reports.")
    parser.add_argument("--runs-root", default="eval/knowledgeos/runs")
    parser.add_argument("--required-runs", type=int, default=7)
    parser.add_argument("--trend-json", default="")
    parser.add_argument("--base-observation-json", default="")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    args = parser.parse_args(argv)

    runs_root = Path(args.runs_root).expanduser().resolve()
    reports_root = runs_root / "reports"
    trend_json = Path(args.trend_json).expanduser().resolve() if _clean_text(args.trend_json) else reports_root / "source_quality_trend_latest.json"
    base_json = (
        Path(args.base_observation_json).expanduser().resolve()
        if _clean_text(args.base_observation_json)
        else reports_root / "source_quality_observation_latest.json"
    )
    trend_report = _read_json(trend_json)
    base_observation_report = _read_json(base_json)
    report = build_report(
        trend_report=trend_report,
        base_observation_report=base_observation_report,
        required_runs=max(1, int(args.required_runs)),
    )
    out_json = (
        Path(args.out_json).expanduser().resolve()
        if _clean_text(args.out_json)
        else reports_root / "source_quality_detail_observation_latest.json"
    )
    out_md = (
        Path(args.out_md).expanduser().resolve()
        if _clean_text(args.out_md)
        else reports_root / "source_quality_detail_observation_latest.md"
    )
    write_report(report, out_json=out_json, out_md=out_md)
    print(json.dumps({"status": "ok", "decision": report["decision"], "run_count": report["run_count"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
