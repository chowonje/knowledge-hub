#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_ORDER = ("paper", "vault", "web")
HARD_METRICS = (
    "route_correctness",
    "vault_stale_citation_rate",
)
SOFT_METRICS = (
    "paper_citation_correctness",
    "vault_abstention_correctness",
    "web_recency_violation",
)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_timestamp(value: str) -> datetime:
    text = _clean_text(value)
    if not text:
        raise ValueError("missing timestamp")
    normalized = text.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _summary_candidates(runs_root: Path) -> list[Path]:
    return sorted(runs_root.glob("source_quality_battery_*/source_quality_battery_summary.json"))


def _load_run_summaries(runs_root: Path, *, limit: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in _summary_candidates(runs_root):
        payload = _read_json(path)
        created_at = _parse_timestamp(payload.get("created_at"))
        entries.append(
            {
                "created_at": created_at,
                "run_dir": str(path.parent),
                "summary_path": str(path),
                "payload": payload,
            }
        )
    entries.sort(key=lambda item: item["created_at"])
    latest_by_date: dict[str, dict[str, Any]] = {}
    for item in entries:
        latest_by_date[item["created_at"].date().isoformat()] = item
    entries = sorted(latest_by_date.values(), key=lambda item: item["created_at"])
    if limit > 0:
        entries = entries[-limit:]
    return entries


def _metric_stats(points: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = [float(point["value"]) for point in points if point.get("value") is not None]
    latest = points[-1]["value"] if points else None
    previous = points[-2]["value"] if len(points) >= 2 else None
    delta = None
    if latest is not None and previous is not None:
        delta = round(float(latest) - float(previous), 6)
    direction = "flat"
    if delta is not None:
        if delta > 0:
            direction = "up"
        elif delta < 0:
            direction = "down"
    return {
        "latest": latest,
        "previous": previous,
        "delta_vs_previous": delta,
        "direction": direction,
        "min": None if not numeric else round(min(numeric), 6),
        "max": None if not numeric else round(max(numeric), 6),
        "points": points,
    }


def build_trend_report(run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    latest_run = run_summaries[-1] if run_summaries else None
    sources: dict[str, dict[str, Any]] = {}
    readiness_latest_by_source: dict[str, dict[str, Any]] = {}
    for source in SOURCE_ORDER:
        source_payload: dict[str, Any] = {"hard": {}, "soft": {}, "readiness": {}}
        route_points = []
        stale_points = []
        readiness_points: dict[str, list[dict[str, Any]]] = {
            "legacy_runtime_rate": [],
            "capability_missing_rate": [],
            "forced_legacy_rate": [],
        }
        for run in run_summaries:
            created_at = run["payload"]["created_at"]
            per_source = dict(run["payload"].get("per_source") or {})
            summary = dict(per_source.get(source) or {})
            hard = dict(run["payload"].get("hard_gates") or {})
            route_gate = dict(hard.get("route_correctness") or {}).get(source, {})
            route_points.append(
                {
                    "created_at": created_at,
                    "run_dir": run["run_dir"],
                    "value": summary.get("route_correctness"),
                    "passed": route_gate.get("passed"),
                }
            )
            if source == "vault":
                stale_gate = dict(hard.get("vault_stale_citation_rate") or {}).get("vault", {})
                stale_points.append(
                    {
                        "created_at": created_at,
                        "run_dir": run["run_dir"],
                        "value": summary.get("stale_citation_rate"),
                        "passed": stale_gate.get("passed"),
                    }
                )
            for metric_name, points in readiness_points.items():
                points.append(
                    {
                        "created_at": created_at,
                        "run_dir": run["run_dir"],
                        "value": summary.get(metric_name),
                    }
                )
        source_payload["hard"]["route_correctness"] = _metric_stats(route_points)
        if source == "vault":
            source_payload["hard"]["vault_stale_citation_rate"] = _metric_stats(stale_points)

        soft_metric_name = {
            "paper": "citation_correctness_soft",
            "vault": "abstention_correctness_soft",
            "web": "recency_violation_soft",
        }[source]
        soft_report_name = {
            "paper": "paper_citation_correctness",
            "vault": "vault_abstention_correctness",
            "web": "web_recency_violation",
        }[source]
        soft_points = []
        for run in run_summaries:
            created_at = run["payload"]["created_at"]
            summary = dict((run["payload"].get("per_source") or {}).get(source) or {})
            soft_points.append(
                {
                    "created_at": created_at,
                    "run_dir": run["run_dir"],
                    "value": summary.get(soft_metric_name),
                }
            )
        source_payload["soft"][soft_report_name] = _metric_stats(soft_points)
        for metric_name, points in readiness_points.items():
            source_payload["readiness"][metric_name] = _metric_stats(points)
        latest_summary = {} if latest_run is None else dict((latest_run["payload"].get("per_source") or {}).get(source) or {})
        readiness_latest_by_source[source] = {
            "runtime_used_counts": dict(latest_summary.get("runtime_used_counts") or {}),
            "fallback_reason_counts": dict(latest_summary.get("fallback_reason_counts") or {}),
            "legacy_runtime_rate": latest_summary.get("legacy_runtime_rate"),
            "capability_missing_rate": latest_summary.get("capability_missing_rate"),
            "forced_legacy_rate": latest_summary.get("forced_legacy_rate"),
        }
        sources[source] = source_payload

    return {
        "schema": "knowledge-hub.source-quality-trend.report.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_count": len(run_summaries),
        "latest_run_dir": None if latest_run is None else latest_run["run_dir"],
        "runs": [
            {
                "created_at": item["payload"]["created_at"],
                "run_date": item["created_at"].date().isoformat(),
                "run_dir": item["run_dir"],
                "summary_path": item["summary_path"],
            }
            for item in run_summaries
        ],
        "sources": sources,
        "readiness_latest_by_source": readiness_latest_by_source,
    }


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _fmt_delta(value: Any) -> str:
    if value is None:
        return "n/a"
    numeric = float(value)
    sign = "+" if numeric >= 0 else ""
    return f"{sign}{numeric:.3f}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Source Quality Trend",
        "",
        f"- runs analyzed: `{report['run_count']}`",
        f"- latest run: `{report['latest_run_dir'] or ''}`",
        "",
    ]
    for source in SOURCE_ORDER:
        payload = report["sources"][source]
        lines.append(f"## {source}")
        route = payload["hard"]["route_correctness"]
        lines.append(
            f"- route_correctness: latest `{_fmt_metric(route['latest'])}`, "
            f"prev `{_fmt_metric(route['previous'])}`, delta `{_fmt_delta(route['delta_vs_previous'])}`, "
            f"range `{_fmt_metric(route['min'])}`..`{_fmt_metric(route['max'])}`"
        )
        if source == "vault":
            stale = payload["hard"]["vault_stale_citation_rate"]
            lines.append(
                f"- vault_stale_citation_rate: latest `{_fmt_metric(stale['latest'])}`, "
                f"prev `{_fmt_metric(stale['previous'])}`, delta `{_fmt_delta(stale['delta_vs_previous'])}`, "
                f"range `{_fmt_metric(stale['min'])}`..`{_fmt_metric(stale['max'])}`"
            )
        soft_name, soft_payload = next(iter(payload["soft"].items()))
        lines.append(
            f"- {soft_name}: latest `{_fmt_metric(soft_payload['latest'])}`, "
            f"prev `{_fmt_metric(soft_payload['previous'])}`, delta `{_fmt_delta(soft_payload['delta_vs_previous'])}`, "
            f"range `{_fmt_metric(soft_payload['min'])}`..`{_fmt_metric(soft_payload['max'])}`"
        )
        for metric_name, readiness_payload in payload["readiness"].items():
            lines.append(
                f"- {metric_name}: latest `{_fmt_metric(readiness_payload['latest'])}`, "
                f"prev `{_fmt_metric(readiness_payload['previous'])}`, delta `{_fmt_delta(readiness_payload['delta_vs_previous'])}`, "
                f"range `{_fmt_metric(readiness_payload['min'])}`..`{_fmt_metric(readiness_payload['max'])}`"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: dict[str, Any], *, json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a source-quality trend report from recent nightly run summaries.")
    parser.add_argument("--runs-root", default="eval/knowledgeos/runs")
    parser.add_argument("--limit", type=int, default=7)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    args = parser.parse_args(argv)

    runs_root = Path(args.runs_root).expanduser().resolve()
    run_summaries = _load_run_summaries(runs_root, limit=max(1, int(args.limit)))
    if not run_summaries:
        raise SystemExit(f"no source quality battery summaries found under {runs_root}")
    report = build_trend_report(run_summaries)

    default_root = runs_root / "reports"
    json_path = Path(args.out_json).expanduser().resolve() if _clean_text(args.out_json) else default_root / "source_quality_trend_latest.json"
    md_path = Path(args.out_md).expanduser().resolve() if _clean_text(args.out_md) else default_root / "source_quality_trend_latest.md"
    write_report(report, json_path=json_path, md_path=md_path)
    print(
        json.dumps(
            {
                "status": "ok",
                "schema": report["schema"],
                "run_count": report["run_count"],
                "latest_run_dir": report["latest_run_dir"],
                "json_path": str(json_path),
                "md_path": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
