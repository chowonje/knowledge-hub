#!/usr/bin/env python3
"""Run the local daily source-quality observation loop and optional docs writeback."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from typing import Any
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "eval" / "knowledgeos" / "runs"
REPORTS_ROOT = RUNS_ROOT / "reports"
OBSERVATION_REPORT_PATH = REPORTS_ROOT / "source_quality_observation_latest.json"
DETAIL_OBSERVATION_REPORT_PATH = REPORTS_ROOT / "source_quality_detail_observation_latest.json"
SOURCE_ORDER = ("paper", "vault", "web")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local source-quality daily loop and optional docs/status + worklog writeback"
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="knowledge-hub repo root")
    parser.add_argument("--runs-root", default=str(RUNS_ROOT), help="eval runs root")
    parser.add_argument("--gate-mode", default="stub_hard", choices=["standard", "stub_hard", "live_smoke"])
    parser.add_argument("--mode", default="hybrid", help="retrieval mode for the source battery")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--limit", type=int, default=7, help="trend/readiness history window")
    parser.add_argument("--required-runs", type=int, default=7)
    parser.add_argument(
        "--skip-if-local-date-already-covered",
        action="store_true",
        default=False,
        help="skip when the latest source-quality run already belongs to the current local date",
    )
    parser.add_argument(
        "--local-timezone",
        default="",
        help="optional IANA timezone name used by --skip-if-local-date-already-covered",
    )
    parser.add_argument("--writeback", action="store_true", default=False, help="run the docs/status + worklog consumer loop")
    parser.add_argument(
        "--enforce-hard-gate",
        action="store_true",
        default=False,
        help="fail unless the refreshed source-quality observation report clears the hard gate",
    )
    parser.add_argument(
        "--apply-writeback",
        action="store_true",
        default=False,
        help="ack and execute the docs/status + worklog consumer loop instead of preview only",
    )
    parser.add_argument("--actor", default="daily-runner")
    include_group = parser.add_mutually_exclusive_group()
    include_group.add_argument("--include-workspace", dest="include_workspace", action="store_true")
    include_group.add_argument("--no-include-workspace", dest="include_workspace", action="store_false")
    parser.set_defaults(include_workspace=None)
    parser.add_argument("--max-workspace-files", type=int, default=8)
    parser.add_argument(
        "--writeback-goal",
        default="",
        help="override the generated writeback goal text for the docs/status + worklog consumer",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", default=False)
    return parser


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _run_json_command(argv: list[str], *, cwd: Path) -> dict[str, Any]:
    result = subprocess.run(
        argv,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(argv)} :: {stderr}")
    stdout = result.stdout or ""
    json_text = stdout
    if stdout and not stdout.lstrip().startswith("{"):
        for line_index, line in enumerate(stdout.splitlines()):
            if line.lstrip().startswith("{"):
                json_text = "\n".join(stdout.splitlines()[line_index:])
                break
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid json from command: {' '.join(argv)} :: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid payload type from command: {' '.join(argv)}")
    return payload


def _local_timezone(tz_name: str) -> ZoneInfo:
    cleaned = _clean_text(tz_name)
    if cleaned:
        return ZoneInfo(cleaned)
    current = datetime.now().astimezone().tzinfo
    if isinstance(current, ZoneInfo):
        return current
    return ZoneInfo("Asia/Seoul")


def _latest_run_summary_path(runs_root: Path) -> Path | None:
    observation_path = runs_root / "reports" / OBSERVATION_REPORT_PATH.name
    if observation_path.exists():
        try:
            observation = json.loads(observation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            observation = {}
        latest_run_dir = Path(str((observation or {}).get("latest_run_dir") or "").strip())
        if latest_run_dir:
            candidate = latest_run_dir / "source_quality_battery_summary.json"
            if candidate.exists():
                return candidate

    candidates = sorted(runs_root.glob("source_quality_battery_*/source_quality_battery_summary.json"))
    if candidates:
        return candidates[-1]
    return None


def _latest_run_local_date(runs_root: Path, *, tz_name: str) -> dict[str, str] | None:
    summary_path = _latest_run_summary_path(runs_root)
    if summary_path is None:
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    created_at = _clean_text(payload.get("created_at"))
    if not created_at:
        return None
    created_dt = datetime.fromisoformat(created_at)
    tz = _local_timezone(tz_name)
    local_date = created_dt.astimezone(tz).date().isoformat()
    return {
        "summaryPath": str(summary_path),
        "createdAt": created_at,
        "localDate": local_date,
        "timezone": getattr(tz, "key", str(tz)),
    }


def _build_observation_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "decision": _clean_text(report.get("decision")),
        "blockers": [str(item).strip() for item in list(report.get("blockers") or []) if str(item).strip()],
        "runCount": int(report.get("run_count") or 0),
        "requiredRuns": int(report.get("required_runs") or 0),
        "latestRunDir": _clean_text(report.get("latest_run_dir")),
        "legacyReadinessDecision": _clean_text(report.get("legacy_readiness_decision")),
        "sources": {},
    }
    for source in SOURCE_ORDER:
        payload = dict((report.get("sources") or {}).get(source) or {})
        source_summary = {
            "routeCorrectness": payload.get("route_correctness"),
            "legacyRuntimeRate": payload.get("legacy_runtime_rate"),
            "capabilityMissingRate": payload.get("capability_missing_rate"),
            "forcedLegacyRate": payload.get("forced_legacy_rate"),
        }
        if source == "vault":
            source_summary["staleCitationRate"] = payload.get("stale_citation_rate")
        summary["sources"][source] = source_summary
    return summary


def _build_detail_observation_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision": _clean_text(report.get("decision")),
        "blockers": [str(item).strip() for item in list(report.get("blockers") or []) if str(item).strip()],
        "runCount": int(report.get("run_count") or 0),
        "requiredRuns": int(report.get("required_runs") or 0),
        "latestRunDir": _clean_text(report.get("latest_run_dir")),
        "baseObservationDecision": _clean_text(report.get("base_observation_decision")),
        "checks": [
            {
                "source": _clean_text(dict(item or {}).get("source")),
                "metric": _clean_text(dict(item or {}).get("metric")),
                "status": _clean_text(dict(item or {}).get("status")),
                "latest": dict(item or {}).get("latest"),
                "numericPointCount": int(dict(item or {}).get("numericPointCount") or 0),
            }
            for item in list(report.get("checks") or [])
        ],
    }


def _metric_text(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value)


def build_writeback_goal(summary: dict[str, Any]) -> str:
    blockers = list(summary.get("blockers") or [])
    blocker_text = ", ".join(blockers) if blockers else "none"
    sources = dict(summary.get("sources") or {})
    return (
        "오늘 source-quality 관찰 결과를 docs/status와 worklog에 정리해줘. "
        f"decision={_clean_text(summary.get('decision'))}; "
        f"run_count={summary.get('runCount')}/{summary.get('requiredRuns')}; "
        f"blockers={blocker_text}; "
        f"paper route_correctness={_metric_text(dict(sources.get('paper') or {}).get('routeCorrectness'))}; "
        f"vault route_correctness={_metric_text(dict(sources.get('vault') or {}).get('routeCorrectness'))}; "
        f"web route_correctness={_metric_text(dict(sources.get('web') or {}).get('routeCorrectness'))}; "
        f"vault stale_citation_rate={_metric_text(dict(sources.get('vault') or {}).get('staleCitationRate'))}. "
        "hard gate 승격 판단과 blocker를 짧게 남겨줘."
    )


def build_writeback_argv(args: argparse.Namespace, *, goal: str, repo_root: Path) -> list[str]:
    argv = [
        sys.executable,
        str(repo_root / "scripts" / "run_agent_docs_writeback_loop.py"),
        goal,
        "--repo-path",
        str(repo_root),
        "--max-workspace-files",
        str(max(1, int(args.max_workspace_files))),
        "--actor",
        str(args.actor),
        "--json",
    ]
    if args.include_workspace is True:
        argv.append("--include-workspace")
    elif args.include_workspace is False:
        argv.append("--no-include-workspace")
    if args.apply_writeback:
        argv.append("--apply")
    return argv


def build_hard_gate_argv(*, repo_root: Path, runs_root: Path) -> list[str]:
    return [
        sys.executable,
        str(repo_root / "eval" / "knowledgeos" / "scripts" / "check_source_quality_hard_gate.py"),
        "--runs-root",
        str(runs_root),
        "--json",
    ]


def _load_detail_observation_summary_if_exists(runs_root: Path) -> dict[str, Any] | None:
    path = runs_root / "reports" / DETAIL_OBSERVATION_REPORT_PATH.name
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return _build_detail_observation_summary(payload)


def build_daily_commands(args: argparse.Namespace, *, repo_root: Path, runs_root: Path) -> list[tuple[str, list[str]]]:
    return [
        (
            "battery",
            [
                sys.executable,
                str(repo_root / "eval" / "knowledgeos" / "scripts" / "run_source_quality_battery.py"),
                "--repo-root",
                str(repo_root),
                "--gate-mode",
                str(args.gate_mode),
                "--mode",
                str(args.mode),
                "--top-k",
                str(max(1, int(args.top_k))),
            ],
        ),
        (
            "trend",
            [
                sys.executable,
                str(repo_root / "eval" / "knowledgeos" / "scripts" / "report_source_quality_trend.py"),
                "--runs-root",
                str(runs_root),
                "--limit",
                str(max(1, int(args.limit))),
            ],
        ),
        (
            "readiness",
            [
                sys.executable,
                str(repo_root / "eval" / "knowledgeos" / "scripts" / "report_legacy_runtime_readiness.py"),
                "--repo-root",
                str(repo_root),
                "--runs-root",
                str(runs_root),
                "--limit",
                str(max(1, int(args.limit))),
            ],
        ),
        (
            "observation",
            [
                sys.executable,
                str(repo_root / "eval" / "knowledgeos" / "scripts" / "report_source_quality_observation.py"),
                "--runs-root",
                str(runs_root),
                "--required-runs",
                str(max(1, int(args.required_runs))),
            ],
        ),
        (
            "detail_observation",
            [
                sys.executable,
                str(repo_root / "eval" / "knowledgeos" / "scripts" / "report_source_quality_detail_observation.py"),
                "--runs-root",
                str(runs_root),
                "--required-runs",
                str(max(1, int(args.required_runs))),
            ],
        ),
    ]


def run_daily_loop(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    runs_root = Path(args.runs_root).expanduser().resolve()
    if args.skip_if_local_date_already_covered:
        latest_run = _latest_run_local_date(runs_root, tz_name=str(args.local_timezone))
        tz = _local_timezone(str(args.local_timezone))
        today_local = datetime.now(tz).date().isoformat()
        if latest_run is not None and latest_run.get("localDate") == today_local:
            result = {
                "skipped": True,
                "skipReason": "already_ran_for_local_date",
                "localDate": today_local,
                "timezone": latest_run.get("timezone", getattr(tz, "key", str(tz))),
                "latestRun": latest_run,
            }
            if args.enforce_hard_gate:
                result["hardGate"] = _run_json_command(
                    build_hard_gate_argv(repo_root=repo_root, runs_root=runs_root),
                    cwd=repo_root,
                )
            detail_summary = _load_detail_observation_summary_if_exists(runs_root)
            if detail_summary is not None:
                result["detailObservationSummary"] = detail_summary
            return result

    commands = build_daily_commands(args, repo_root=repo_root, runs_root=runs_root)
    command_results: dict[str, Any] = {}
    for name, argv in commands:
        command_results[name] = _run_json_command(argv, cwd=repo_root)

    observation_report_path = runs_root / "reports" / OBSERVATION_REPORT_PATH.name
    observation_report = json.loads(observation_report_path.read_text(encoding="utf-8"))
    if not isinstance(observation_report, dict):
        raise RuntimeError(f"invalid observation report payload: {observation_report_path}")
    observation_summary = _build_observation_summary(observation_report)
    detail_observation_report_path = runs_root / "reports" / DETAIL_OBSERVATION_REPORT_PATH.name
    detail_observation_report = json.loads(detail_observation_report_path.read_text(encoding="utf-8"))
    if not isinstance(detail_observation_report, dict):
        raise RuntimeError(f"invalid detail observation report payload: {detail_observation_report_path}")
    detail_observation_summary = _build_detail_observation_summary(detail_observation_report)

    result: dict[str, Any] = {
        "commands": command_results,
        "observationReportPath": str(observation_report_path),
        "observationSummary": observation_summary,
        "detailObservationReportPath": str(detail_observation_report_path),
        "detailObservationSummary": detail_observation_summary,
    }

    if args.enforce_hard_gate:
        result["hardGate"] = _run_json_command(
            build_hard_gate_argv(repo_root=repo_root, runs_root=runs_root),
            cwd=repo_root,
        )

    if args.writeback:
        goal = _clean_text(args.writeback_goal) or build_writeback_goal(observation_summary)
        writeback_payload = _run_json_command(
            build_writeback_argv(args, goal=goal, repo_root=repo_root),
            cwd=repo_root,
        )
        result["writebackGoal"] = goal
        result["writeback"] = writeback_payload

    return result


def render_result(result: dict[str, Any]) -> str:
    if result.get("skipped") is True:
        latest_run = dict(result.get("latestRun") or {})
        lines = [
            "skipped: True",
            f"reason: {result.get('skipReason', '')}",
            f"local_date: {result.get('localDate', '')}",
            f"timezone: {result.get('timezone', '')}",
            f"latest_run_local_date: {latest_run.get('localDate', '')}",
            f"latest_run_created_at: {latest_run.get('createdAt', '')}",
            f"latest_run_summary: {latest_run.get('summaryPath', '')}",
        ]
        if isinstance(result.get("hardGate"), dict):
            lines.append(f"hard_gate_status: {dict(result.get('hardGate') or {}).get('status', '')}")
        detail_summary = dict(result.get("detailObservationSummary") or {})
        if detail_summary:
            lines.append(f"detail_decision: {detail_summary.get('decision', '')}")
        return "\n".join(lines)

    summary = dict(result.get("observationSummary") or {})
    blockers = list(summary.get("blockers") or [])
    sources = dict(summary.get("sources") or {})
    lines = [
        f"decision: {summary.get('decision', '')}",
        f"blockers: {blockers if blockers else []}",
        f"run_count: {summary.get('runCount', 0)} / {summary.get('requiredRuns', 0)}",
        f"latest_run_dir: {summary.get('latestRunDir', '')}",
        f"paper route_correctness: {_metric_text(dict(sources.get('paper') or {}).get('routeCorrectness'))}",
        f"vault route_correctness: {_metric_text(dict(sources.get('vault') or {}).get('routeCorrectness'))}",
        f"web route_correctness: {_metric_text(dict(sources.get('web') or {}).get('routeCorrectness'))}",
        f"vault stale_citation_rate: {_metric_text(dict(sources.get('vault') or {}).get('staleCitationRate'))}",
        f"legacy_readiness_decision: {summary.get('legacyReadinessDecision', '')}",
    ]
    detail_summary = dict(result.get("detailObservationSummary") or {})
    if detail_summary:
        detail_blockers = list(detail_summary.get("blockers") or [])
        lines.extend(
            [
                f"detail_decision: {detail_summary.get('decision', '')}",
                f"detail_blockers: {detail_blockers if detail_blockers else []}",
            ]
        )
    if isinstance(result.get("hardGate"), dict):
        lines.append(f"hard_gate_status: {dict(result.get('hardGate') or {}).get('status', '')}")
    if isinstance(result.get("writeback"), dict):
        writeback_summary = dict((result.get("writeback") or {}).get("summary") or {})
        lines.extend(
            [
                f"writeback_goal: {result.get('writebackGoal', '')}",
                f"writeback_applied: {writeback_summary.get('applied', False)}",
                f"writeback_targets: {', '.join(str(item) for item in list(writeback_summary.get('targets') or [])) or '-'}",
                f"writeback_receipt_id: {writeback_summary.get('receiptId', '')}",
            ]
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_daily_loop(args)
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(render_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
