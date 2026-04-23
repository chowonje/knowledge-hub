#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LEGACY_RUNTIME_SYMBOL_RE = re.compile(r"\bLegacyRAGRuntime\b")
ASK_V2_MODE_LEGACY_RE = re.compile(r"ask_v2_mode\s*=\s*['\"]legacy['\"]")
SOURCE_ORDER = ("paper", "vault", "web")
LEGACY_REMOVAL_READY_RUNS = 7


@dataclass(frozen=True)
class ScanHit:
    path: str
    line: int
    text: str
    category: str


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _classify_path(path: Path, repo_root: Path) -> str:
    rel = path.relative_to(repo_root)
    parts = rel.parts
    if "tests" in parts:
        return "tests"
    if "eval" in parts:
        return "eval"
    if parts and parts[0] == "scripts":
        return "scripts"
    return "runtime"


def _scan_python_files(repo_root: Path) -> list[Path]:
    roots = (
        repo_root / "knowledge_hub",
        repo_root / "tests",
        repo_root / "eval",
        repo_root / "scripts",
    )
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        files.extend(sorted(root.rglob("*.py")))
    return files


def _scan_hits(repo_root: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    legacy_hits: dict[str, list[dict[str, Any]]] = {category: [] for category in ("runtime", "tests", "eval", "scripts")}
    ask_v2_legacy_hits: dict[str, list[dict[str, Any]]] = {category: [] for category in ("runtime", "tests", "eval", "scripts")}
    for path in _scan_python_files(repo_root):
        category = _classify_path(path, repo_root)
        rel = str(path.relative_to(repo_root))
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            text = line.strip()
            if path.name != "rag_legacy_runtime.py" and LEGACY_RUNTIME_SYMBOL_RE.search(line):
                legacy_hits[category].append(
                    {"path": rel, "line": line_no, "text": text}
                )
            if ASK_V2_MODE_LEGACY_RE.search(line) and "was removed" not in line:
                ask_v2_legacy_hits[category].append(
                    {"path": rel, "line": line_no, "text": text}
                )
    return {
        "legacy_runtime_symbol": legacy_hits,
        "ask_v2_mode_legacy_literal": ask_v2_legacy_hits,
    }


def _summary_candidates(runs_root: Path) -> list[Path]:
    return sorted(runs_root.glob("source_quality_battery_*/source_quality_battery_summary.json"))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _recent_summaries(runs_root: Path, limit: int) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in _summary_candidates(runs_root):
        payload = _read_json(path)
        summaries.append(
            {
                "run_dir": str(path.parent),
                "summary_path": str(path),
                "payload": payload,
                "created_at": _clean_text(payload.get("created_at")),
            }
        )
    return summaries[-max(1, int(limit)) :]


def _readiness_trends(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    per_source: dict[str, dict[str, Any]] = {}
    for source in SOURCE_ORDER:
        legacy_rates: list[Any] = []
        capability_rates: list[Any] = []
        forced_rates: list[Any] = []
        for summary in summaries:
            payload = dict((summary["payload"].get("per_source") or {}).get(source) or {})
            legacy_rates.append(payload.get("legacy_runtime_rate"))
            capability_rates.append(payload.get("capability_missing_rate"))
            forced_rates.append(payload.get("forced_legacy_rate"))
        per_source[source] = {
            "legacy_runtime_rate": legacy_rates,
            "capability_missing_rate": capability_rates,
            "forced_legacy_rate": forced_rates,
        }
    return per_source


def _decision(
    *,
    recent_summaries: list[dict[str, Any]],
    callsites: dict[str, dict[str, list[dict[str, Any]]]],
) -> str:
    if len(recent_summaries) < LEGACY_REMOVAL_READY_RUNS:
        return "observe_more"
    readiness = _readiness_trends(recent_summaries)
    for source in SOURCE_ORDER:
        if any(float(value or 0.0) != 0.0 for value in readiness[source]["legacy_runtime_rate"]):
            return "not_ready"
        if any(float(value or 0.0) != 0.0 for value in readiness[source]["capability_missing_rate"]):
            return "not_ready"
    runtime_legacy = callsites["legacy_runtime_symbol"]["runtime"]
    if runtime_legacy != [
        hit
        for hit in runtime_legacy
        if hit["path"] == "knowledge_hub/ai/rag_answer_runtime.py"
    ]:
        return "not_ready"
    if callsites["ask_v2_mode_legacy_literal"]["runtime"] or callsites["ask_v2_mode_legacy_literal"]["scripts"]:
        return "not_ready"
    return "ready_for_removal_tranche"


def build_report(
    *,
    repo_root: Path,
    runs_root: Path,
    limit: int,
) -> dict[str, Any]:
    callsites = _scan_hits(repo_root)
    recent_summaries = _recent_summaries(runs_root, limit)
    trend_report_path = runs_root / "reports" / "source_quality_trend_latest.json"
    return {
        "schema": "knowledge-hub.legacy-runtime-readiness.report.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "readiness_only",
        "required_run_count": LEGACY_REMOVAL_READY_RUNS,
        "observation_window": f"{max(1, int(limit))}_runs",
        "decision": _decision(recent_summaries=recent_summaries, callsites=callsites),
        "trend_report_path": str(trend_report_path) if trend_report_path.exists() else "",
        "run_count": len(recent_summaries),
        "callsites": callsites,
        "recent_runs": [
            {
                "created_at": summary["created_at"],
                "run_dir": summary["run_dir"],
                "summary_path": summary["summary_path"],
            }
            for summary in recent_summaries
        ],
        "readiness_trends": _readiness_trends(recent_summaries),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Legacy Runtime Readiness",
        "",
        f"- decision: `{report['decision']}`",
        f"- run_count: `{report['run_count']}`",
        f"- required_run_count: `{report['required_run_count']}`",
        f"- policy: `7_consecutive_source_quality_runs`",
        f"- trend_report_path: `{report.get('trend_report_path') or ''}`",
        "",
        "## Runtime Readiness Trends",
    ]
    for source in SOURCE_ORDER:
        payload = report["readiness_trends"][source]
        lines.append(
            f"- {source}: legacy_runtime_rate={payload['legacy_runtime_rate']}, "
            f"capability_missing_rate={payload['capability_missing_rate']}, "
            f"forced_legacy_rate={payload['forced_legacy_rate']}"
        )
    lines.extend(["", "## Callsites"])
    for group_name, categories in report["callsites"].items():
        lines.append(f"### {group_name}")
        for category in ("runtime", "tests", "eval", "scripts"):
            hits = categories.get(category) or []
            lines.append(f"- {category}: {len(hits)}")
            for hit in hits:
                lines.append(f"  - `{hit['path']}:{hit['line']}` {hit['text']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: dict[str, Any], *, out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a legacy-runtime removal-readiness report.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--runs-root", default="eval/knowledgeos/runs")
    parser.add_argument("--limit", type=int, default=LEGACY_REMOVAL_READY_RUNS)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve()
    runs_root = Path(args.runs_root).expanduser().resolve()
    report = build_report(repo_root=repo_root, runs_root=runs_root, limit=max(1, int(args.limit)))
    default_root = runs_root / "reports"
    out_json = (
        Path(args.out_json).expanduser().resolve()
        if _clean_text(args.out_json)
        else default_root / "legacy_runtime_readiness_latest.json"
    )
    out_md = (
        Path(args.out_md).expanduser().resolve()
        if _clean_text(args.out_md)
        else default_root / "legacy_runtime_readiness_latest.md"
    )
    write_report(report, out_json=out_json, out_md=out_md)
    print(json.dumps({"status": "ok", "decision": report["decision"], "run_count": report["run_count"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
