#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_ORDER = ("paper", "vault", "web")


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _route_points(report: dict[str, Any], source: str) -> list[dict[str, Any]]:
    return list((((report.get("sources") or {}).get(source) or {}).get("hard") or {}).get("route_correctness", {}).get("points") or [])


def _vault_stale_points(report: dict[str, Any]) -> list[dict[str, Any]]:
    return list((((report.get("sources") or {}).get("vault") or {}).get("hard") or {}).get("vault_stale_citation_rate", {}).get("points") or [])


def _readiness_latest(report: dict[str, Any], source: str) -> dict[str, Any]:
    return dict((report.get("readiness_latest_by_source") or {}).get(source) or {})


def _all_zero(values: list[Any]) -> bool:
    cleaned = [value for value in values if value is not None]
    return bool(cleaned) and all(float(value) == 0.0 for value in cleaned)


def _decision(
    *,
    trend_report: dict[str, Any],
    readiness_report: dict[str, Any],
    required_runs: int,
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    run_count = int(trend_report.get("run_count") or 0)
    if run_count < required_runs:
        blockers.append(f"need_{required_runs}_runs_have_{run_count}")
        return "observe_more", blockers

    for source in SOURCE_ORDER:
        points = _route_points(trend_report, source)
        if not points or not all(float(point.get("value") or 0.0) == 1.0 for point in points):
            blockers.append(f"{source}_route_correctness_not_stable")

    stale_points = _vault_stale_points(trend_report)
    if not stale_points or not all(float(point.get("value") or 0.0) == 0.0 for point in stale_points):
        blockers.append("vault_stale_citation_rate_not_stable")

    readiness_decision = _clean_text(readiness_report.get("decision"))
    if readiness_decision != "ready_for_removal_tranche":
        blockers.append(f"legacy_readiness_{readiness_decision or 'unknown'}")

    readiness_trends = dict(readiness_report.get("readiness_trends") or {})
    for source in SOURCE_ORDER:
        source_trends = dict(readiness_trends.get(source) or {})
        if not _all_zero(list(source_trends.get("legacy_runtime_rate") or [])):
            blockers.append(f"{source}_legacy_runtime_rate_nonzero")
        if not _all_zero(list(source_trends.get("capability_missing_rate") or [])):
            blockers.append(f"{source}_capability_missing_rate_nonzero")

    if blockers:
        return "not_ready_for_hard_gate_review", blockers
    return "ready_for_hard_gate_review", blockers


def build_report(
    *,
    trend_report: dict[str, Any],
    readiness_report: dict[str, Any],
    required_runs: int,
) -> dict[str, Any]:
    decision, blockers = _decision(
        trend_report=trend_report,
        readiness_report=readiness_report,
        required_runs=required_runs,
    )
    latest_by_source = {
        source: {
            "route_correctness": (((trend_report.get("sources") or {}).get(source) or {}).get("hard") or {}).get("route_correctness", {}).get("latest"),
            "legacy_runtime_rate": _readiness_latest(trend_report, source).get("legacy_runtime_rate"),
            "capability_missing_rate": _readiness_latest(trend_report, source).get("capability_missing_rate"),
            "forced_legacy_rate": _readiness_latest(trend_report, source).get("forced_legacy_rate"),
            "runtime_used_counts": dict(_readiness_latest(trend_report, source).get("runtime_used_counts") or {}),
            "fallback_reason_counts": dict(_readiness_latest(trend_report, source).get("fallback_reason_counts") or {}),
        }
        for source in SOURCE_ORDER
    }
    latest_by_source["vault"]["stale_citation_rate"] = (
        (((trend_report.get("sources") or {}).get("vault") or {}).get("hard") or {}).get("vault_stale_citation_rate", {}).get("latest")
    )
    return {
        "schema": "knowledge-hub.source-quality-observation.report.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "required_runs": required_runs,
        "run_count": int(trend_report.get("run_count") or 0),
        "latest_run_dir": trend_report.get("latest_run_dir"),
        "decision": decision,
        "blockers": blockers,
        "legacy_readiness_decision": readiness_report.get("decision"),
        "sources": latest_by_source,
        "trend_report_path": _clean_text(trend_report.get("_report_path")),
        "readiness_report_path": _clean_text(readiness_report.get("_report_path")),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Source Quality Observation",
        "",
        f"- decision: `{report['decision']}`",
        f"- run_count: `{report['run_count']}` / `{report['required_runs']}`",
        f"- latest_run: `{report.get('latest_run_dir') or ''}`",
        f"- legacy_readiness_decision: `{report.get('legacy_readiness_decision') or ''}`",
        "",
        "## Latest Metrics",
    ]
    for source in SOURCE_ORDER:
        payload = dict((report.get("sources") or {}).get(source) or {})
        metric_line = (
            f"- {source}: route_correctness=`{payload.get('route_correctness')}`"
            f", legacy_runtime_rate=`{payload.get('legacy_runtime_rate')}`"
            f", capability_missing_rate=`{payload.get('capability_missing_rate')}`"
            f", forced_legacy_rate=`{payload.get('forced_legacy_rate')}`"
        )
        if source == "vault":
            metric_line += f", stale_citation_rate=`{payload.get('stale_citation_rate')}`"
        lines.append(metric_line)
    lines.extend(["", "## Blockers"])
    if report["blockers"]:
        for blocker in report["blockers"]:
            lines.append(f"- {blocker}")
    else:
        lines.append("- none")
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: dict[str, Any], *, out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the weekly source-quality observation verdict.")
    parser.add_argument("--runs-root", default="eval/knowledgeos/runs")
    parser.add_argument("--required-runs", type=int, default=7)
    parser.add_argument("--trend-json", default="")
    parser.add_argument("--readiness-json", default="")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    args = parser.parse_args(argv)

    runs_root = Path(args.runs_root).expanduser().resolve()
    reports_root = runs_root / "reports"
    trend_json = Path(args.trend_json).expanduser().resolve() if _clean_text(args.trend_json) else reports_root / "source_quality_trend_latest.json"
    readiness_json = Path(args.readiness_json).expanduser().resolve() if _clean_text(args.readiness_json) else reports_root / "legacy_runtime_readiness_latest.json"
    trend_report = _read_json(trend_json)
    readiness_report = _read_json(readiness_json)
    trend_report["_report_path"] = str(trend_json)
    readiness_report["_report_path"] = str(readiness_json)
    report = build_report(
        trend_report=trend_report,
        readiness_report=readiness_report,
        required_runs=max(1, int(args.required_runs)),
    )
    out_json = Path(args.out_json).expanduser().resolve() if _clean_text(args.out_json) else reports_root / "source_quality_observation_latest.json"
    out_md = Path(args.out_md).expanduser().resolve() if _clean_text(args.out_md) else reports_root / "source_quality_observation_latest.md"
    write_report(report, out_json=out_json, out_md=out_md)
    print(json.dumps({"status": "ok", "decision": report["decision"], "run_count": report["run_count"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
