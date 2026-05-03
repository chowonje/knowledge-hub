#!/usr/bin/env python3
"""Build a daily read-only Eval Center snapshot."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from knowledge_hub.application.eval_center import EVAL_CENTER_SUMMARY_SCHEMA, build_eval_center_summary
from knowledge_hub.core.schema_validator import validate_payload


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = Path("~/.khub/eval/knowledgeos/runs")
REPORTS_ROOT = RUNS_ROOT / "reports"
LATEST_JSON_NAME = "eval_center_latest.json"
LATEST_MD_NAME = "eval_center_latest.md"
SNAPSHOT_JSON_NAME = "eval_center_summary.json"
SNAPSHOT_MD_NAME = "eval_center_summary.md"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a daily read-only Eval Center snapshot.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="knowledge-hub repo root")
    parser.add_argument("--runs-root", default=str(RUNS_ROOT), help="eval runs root")
    parser.add_argument("--queries-dir", default="eval/knowledgeos/queries", help="eval query CSV directory")
    parser.add_argument(
        "--failure-bank-path",
        default="~/.khub/eval/knowledgeos/failures/failure_bank.jsonl",
        help="Failure Bank JSONL path",
    )
    parser.add_argument(
        "--skip-if-local-date-already-covered",
        action="store_true",
        default=False,
        help="skip when the latest Eval Center snapshot already belongs to the current local date",
    )
    parser.add_argument(
        "--local-timezone",
        default="",
        help="optional IANA timezone name used by --skip-if-local-date-already-covered",
    )
    parser.add_argument(
        "--require-today-source-quality",
        dest="require_today_source_quality",
        action="store_true",
        default=True,
        help="warn when the latest successful source-quality run is not from today's local date",
    )
    parser.add_argument(
        "--no-require-today-source-quality",
        dest="require_today_source_quality",
        action="store_false",
        help="do not check whether source-quality is fresh for today's local date",
    )
    parser.add_argument(
        "--wait-for-today-source-quality-seconds",
        type=float,
        default=0.0,
        help=(
            "wait up to this many seconds for today's source-quality observation before building the "
            "Eval Center snapshot; this is read-only and does not run source-quality"
        ),
    )
    parser.add_argument(
        "--source-quality-wait-poll-seconds",
        type=float,
        default=30.0,
        help="poll interval used by --wait-for-today-source-quality-seconds",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", default=False)
    return parser


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _local_timezone(tz_name: str) -> ZoneInfo:
    cleaned = _clean_text(tz_name)
    if cleaned:
        return ZoneInfo(cleaned)
    current = datetime.now().astimezone().tzinfo
    if isinstance(current, ZoneInfo):
        return current
    return ZoneInfo("Asia/Seoul")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _validate_eval_center_payload(payload: dict[str, Any], *, label: str) -> None:
    schema = _clean_text(payload.get("schema"))
    if schema != EVAL_CENTER_SUMMARY_SCHEMA:
        raise ValueError(f"{label} has unexpected schema: {schema or 'missing'}")
    result = validate_payload(payload, schema, strict=True)
    if not result.ok:
        raise ValueError(f"{label} failed schema validation: {result.errors}")


def _latest_snapshot_summary_path(runs_root: Path) -> Path | None:
    latest_path = runs_root / "reports" / LATEST_JSON_NAME
    if latest_path.exists():
        return latest_path
    candidates = sorted(runs_root.glob("eval_center_snapshot_*/eval_center_summary.json"))
    if candidates:
        return candidates[-1]
    return None


def _latest_snapshot_local_date(runs_root: Path, *, tz_name: str) -> dict[str, str] | None:
    summary_path = _latest_snapshot_summary_path(runs_root)
    if summary_path is None:
        return None
    try:
        payload = _read_json(summary_path)
    except Exception:
        return None
    try:
        _validate_eval_center_payload(payload, label=str(summary_path))
    except Exception:
        return None
    generated_at = _clean_text(payload.get("generatedAt"))
    if not generated_at:
        return None
    generated_dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    if generated_dt.tzinfo is None:
        generated_dt = generated_dt.replace(tzinfo=timezone.utc)
    tz = _local_timezone(tz_name)
    local_date = generated_dt.astimezone(tz).date().isoformat()
    freshness = dict(dict(payload.get("sourceQuality") or {}).get("freshness") or {})
    return {
        "summaryPath": str(summary_path),
        "generatedAt": generated_at,
        "localDate": local_date,
        "timezone": getattr(tz, "key", str(tz)),
        "sourceQualityFreshnessStatus": _clean_text(freshness.get("status")),
    }


def _build_snapshot_summary(payload: dict[str, Any]) -> dict[str, Any]:
    source_quality = dict(payload.get("sourceQuality") or {})
    answer_loop = dict(payload.get("answerLoop") or {})
    answer_summary = dict(answer_loop.get("summary") or {})
    overall = dict(answer_summary.get("overall") or {})
    gaps = [str(item.get("id") or "").strip() for item in list(payload.get("gaps") or []) if str(item.get("id") or "").strip()]
    return {
        "status": _clean_text(payload.get("status")),
        "warningCount": len(list(payload.get("warnings") or [])),
        "sourceQualityBaseDecision": _clean_text(dict(source_quality.get("baseObservation") or {}).get("decision")),
        "sourceQualityDetailDecision": _clean_text(dict(source_quality.get("detailObservation") or {}).get("decision")),
        "sourceQualityLatestRun": _clean_text(dict(source_quality.get("latestRun") or {}).get("path")),
        "sourceQualityFreshnessStatus": _clean_text(dict(source_quality.get("freshness") or {}).get("status")),
        "sourceQualityFreshnessExpectedDate": _clean_text(
            dict(source_quality.get("freshness") or {}).get("expectedLocalDate")
        ),
        "sourceQualityFreshnessLatestDate": _clean_text(
            dict(source_quality.get("freshness") or {}).get("latestRunLocalDate")
        ),
        "answerLoopStatus": _clean_text(answer_summary.get("status")),
        "answerLoopLatestRun": _clean_text(answer_loop.get("latestRunDir")),
        "answerLoopRowCount": int(answer_summary.get("rowCount") or 0),
        "answerLoopPredLabelScore": overall.get("predLabelScore"),
        "gapIds": gaps,
    }


def _now_in_timezone(now: datetime | None, tzinfo: Any) -> datetime:
    if now is None:
        return datetime.now(tzinfo)
    if now.tzinfo is None:
        return now.replace(tzinfo=tzinfo)
    return now.astimezone(tzinfo)


def _parse_timestamp(value: Any) -> datetime | None:
    token = _clean_text(value)
    if not token:
        return None
    try:
        parsed = datetime.fromisoformat(token.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _source_quality_freshness_probe(runs_root: Path, *, expected_local_date: str, tz: ZoneInfo) -> dict[str, Any]:
    report_path = runs_root / "reports" / "source_quality_observation_latest.json"
    probe: dict[str, Any] = {
        "status": "missing",
        "expectedLocalDate": expected_local_date,
        "latestRunLocalDate": "",
        "latestRunDir": "",
        "reportPath": str(report_path),
    }
    if not report_path.exists():
        return probe
    try:
        report = _read_json(report_path)
    except Exception as exc:
        probe["status"] = "unreadable"
        probe["error"] = str(exc)
        return probe
    latest_run_dir = _clean_text(report.get("latest_run_dir"))
    probe["latestRunDir"] = latest_run_dir
    if not latest_run_dir:
        probe["status"] = "missing_run_dir"
        return probe
    summary_path = Path(latest_run_dir) / "source_quality_battery_summary.json"
    if not summary_path.exists():
        probe["status"] = "missing_summary"
        return probe
    try:
        summary = _read_json(summary_path)
    except Exception as exc:
        probe["status"] = "unreadable_summary"
        probe["error"] = str(exc)
        return probe
    timestamp = _parse_timestamp(summary.get("created_at") or summary.get("generatedAt"))
    if timestamp is None:
        probe["status"] = "missing_timestamp"
        return probe
    latest_local_date = timestamp.astimezone(tz).date().isoformat()
    probe["latestRunLocalDate"] = latest_local_date
    probe["latestRunTimestamp"] = timestamp.isoformat()
    probe["status"] = "fresh" if latest_local_date == expected_local_date else "stale"
    return probe


def _wait_for_source_quality_freshness(
    runs_root: Path,
    *,
    expected_local_date: str,
    tz: ZoneInfo,
    max_seconds: float,
    poll_seconds: float,
    sleep_fn: Callable[[float], None],
    monotonic_fn: Callable[[], float],
) -> dict[str, Any]:
    max_wait = max(0.0, float(max_seconds or 0.0))
    poll = max(1.0, float(poll_seconds or 1.0))
    started = monotonic_fn()
    probe = _source_quality_freshness_probe(runs_root, expected_local_date=expected_local_date, tz=tz)
    if probe.get("status") == "fresh" or max_wait <= 0.0:
        probe["waitedSeconds"] = 0.0
        probe["timedOut"] = False
        return probe

    deadline = started + max_wait
    while monotonic_fn() < deadline:
        remaining = max(0.0, deadline - monotonic_fn())
        sleep_fn(min(poll, remaining))
        probe = _source_quality_freshness_probe(runs_root, expected_local_date=expected_local_date, tz=tz)
        if probe.get("status") == "fresh":
            probe["waitedSeconds"] = round(max(0.0, monotonic_fn() - started), 3)
            probe["timedOut"] = False
            return probe
    probe["waitedSeconds"] = round(max(0.0, monotonic_fn() - started), 3)
    probe["timedOut"] = True
    return probe


def render_markdown(payload: dict[str, Any]) -> str:
    summary = _build_snapshot_summary(payload)
    warnings = list(payload.get("warnings") or [])
    recommendations = list(payload.get("recommendations") or [])
    operator_brief = dict(payload.get("operatorBrief") or {})
    brief_summary = dict(operator_brief.get("summary") or {})
    sections = [dict(item) for item in list(operator_brief.get("sections") or []) if isinstance(item, dict)]
    findings = [dict(item) for item in list(operator_brief.get("findings") or []) if isinstance(item, dict)]
    lines = [
        "# Daily Eval Center Brief",
        "",
        f"- status: `{summary.get('status')}`",
        f"- generated_at: `{payload.get('generatedAt') or ''}`",
        f"- priority: `{brief_summary.get('priority') or 'unknown'}`",
        f"- source_quality_base: `{summary.get('sourceQualityBaseDecision') or 'unknown'}`",
        f"- source_quality_detail: `{summary.get('sourceQualityDetailDecision') or 'unknown'}`",
        f"- source_quality_freshness: `{summary.get('sourceQualityFreshnessStatus') or 'not_checked'}`",
        f"- answer_loop_status: `{summary.get('answerLoopStatus') or 'missing'}`",
        f"- answer_loop_rows: `{summary.get('answerLoopRowCount', 0)}`",
        f"- answer_loop_pred_label_score: `{summary.get('answerLoopPredLabelScore')}`",
        f"- warnings: `{summary.get('warningCount', 0)}`",
        "",
        "## Part Status",
    ]
    if sections:
        for section in sections:
            lines.extend(
                [
                    "",
                    f"### {section.get('title') or section.get('id') or 'Unknown'}",
                    f"- status: `{section.get('status') or 'unknown'}`",
                    f"- ran: {'; '.join(str(item) for item in list(section.get('ran') or [])) or 'none'}",
                    f"- problem: {'; '.join(str(item) for item in list(section.get('problem') or [])) or 'none'}",
                    f"- next_action: {section.get('nextAction') or 'none'}",
                ]
            )
            details = [str(item) for item in list(section.get("details") or []) if str(item)]
            if details:
                lines.append("- details:")
                lines.extend(f"  - {item}" for item in details[:10])
    else:
        lines.append("- none")
    lines.extend(["", "## Findings"])
    if findings:
        for item in findings[:12]:
            severity = item.get("severity") or "P3"
            part = item.get("part") or "unknown"
            title = item.get("title") or "finding"
            body = item.get("body") or ""
            lines.extend([f"- [{severity}] {part}: {title}", f"  {body}"])
    else:
        lines.append("- none")
    lines.extend(["", "## Gaps"])
    gap_ids = list(summary.get("gapIds") or [])
    if gap_ids:
        lines.extend(f"- {item}" for item in gap_ids)
    else:
        lines.append("- none")
    lines.extend(["", "## Recommendations"])
    if recommendations:
        lines.extend(f"- {item}" for item in recommendations[:10])
    else:
        lines.append("- none")
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {item}" for item in warnings[:10])
    return "\n".join(lines).rstrip() + "\n"


def run_daily_snapshot(
    args: argparse.Namespace,
    *,
    now: datetime | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    runs_root = Path(args.runs_root).expanduser().resolve()
    queries_dir = Path(args.queries_dir).expanduser()
    if not queries_dir.is_absolute():
        queries_dir = (repo_root / queries_dir).resolve()
    failure_bank_path = Path(args.failure_bank_path).expanduser()
    if not failure_bank_path.is_absolute():
        failure_bank_path = (repo_root / failure_bank_path).resolve()

    tz = _local_timezone(str(args.local_timezone))
    today_local = _now_in_timezone(now, tz).date().isoformat()
    if args.skip_if_local_date_already_covered:
        latest_snapshot = _latest_snapshot_local_date(runs_root, tz_name=str(args.local_timezone))
        if latest_snapshot is not None and latest_snapshot.get("localDate") == today_local:
            freshness_status = _clean_text(latest_snapshot.get("sourceQualityFreshnessStatus"))
            if freshness_status not in {"missing", "stale"}:
                return {
                    "skipped": True,
                    "skipReason": "already_ran_for_local_date",
                    "localDate": today_local,
                    "timezone": latest_snapshot.get("timezone", getattr(tz, "key", str(tz))),
                    "latestSnapshot": latest_snapshot,
                }

    source_quality_wait: dict[str, Any] | None = None
    if args.require_today_source_quality:
        source_quality_wait = _wait_for_source_quality_freshness(
            runs_root,
            expected_local_date=today_local,
            tz=tz,
            max_seconds=float(args.wait_for_today_source_quality_seconds),
            poll_seconds=float(args.source_quality_wait_poll_seconds),
            sleep_fn=sleep_fn,
            monotonic_fn=monotonic_fn,
        )

    now_utc = _now_in_timezone(now, timezone.utc)
    generated_at = now_utc.isoformat()
    stamp = now_utc.strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / f"eval_center_snapshot_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    reports_root = runs_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    payload = build_eval_center_summary(
        runs_root=runs_root,
        queries_dir=queries_dir,
        failure_bank_path=failure_bank_path,
        repo_root=repo_root,
        generated_at=generated_at,
        expected_source_quality_local_date=today_local if args.require_today_source_quality else None,
        freshness_timezone=getattr(tz, "key", str(tz)),
    )
    _validate_eval_center_payload(payload, label="daily Eval Center payload")

    snapshot_json_path = run_dir / SNAPSHOT_JSON_NAME
    snapshot_md_path = run_dir / SNAPSHOT_MD_NAME
    latest_json_path = reports_root / LATEST_JSON_NAME
    latest_md_path = reports_root / LATEST_MD_NAME

    snapshot_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_markdown(payload)
    snapshot_md_path.write_text(markdown, encoding="utf-8")
    latest_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_md_path.write_text(markdown, encoding="utf-8")

    return {
        "status": "ok",
        "generatedAt": generated_at,
        "snapshotDir": str(run_dir),
        "snapshotJsonPath": str(snapshot_json_path),
        "snapshotMarkdownPath": str(snapshot_md_path),
        "latestJsonPath": str(latest_json_path),
        "latestMarkdownPath": str(latest_md_path),
        "sourceQualityWait": source_quality_wait or {},
        "summary": _build_snapshot_summary(payload),
    }


def render_result(result: dict[str, Any]) -> str:
    if result.get("skipped") is True:
        latest_snapshot = dict(result.get("latestSnapshot") or {})
        return "\n".join(
            [
                "skipped: True",
                f"reason: {result.get('skipReason', '')}",
                f"local_date: {result.get('localDate', '')}",
                f"timezone: {result.get('timezone', '')}",
                f"latest_snapshot_local_date: {latest_snapshot.get('localDate', '')}",
                f"latest_snapshot_generated_at: {latest_snapshot.get('generatedAt', '')}",
                f"latest_snapshot_summary: {latest_snapshot.get('summaryPath', '')}",
            ]
        )

    summary = dict(result.get("summary") or {})
    source_quality_wait = dict(result.get("sourceQualityWait") or {})
    return "\n".join(
        [
            f"status: {summary.get('status', '')}",
            f"warnings: {summary.get('warningCount', 0)}",
            f"source_quality_base: {summary.get('sourceQualityBaseDecision', '')}",
            f"source_quality_detail: {summary.get('sourceQualityDetailDecision', '')}",
            f"source_quality_freshness: {summary.get('sourceQualityFreshnessStatus', '')}",
            f"source_quality_wait_status: {source_quality_wait.get('status', '')}",
            f"source_quality_wait_seconds: {source_quality_wait.get('waitedSeconds', 0)}",
            f"answer_loop_status: {summary.get('answerLoopStatus', '')}",
            f"answer_loop_rows: {summary.get('answerLoopRowCount', 0)}",
            f"snapshot_json: {result.get('snapshotJsonPath', '')}",
            f"latest_json: {result.get('latestJsonPath', '')}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_daily_snapshot(args)
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(render_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
