#!/usr/bin/env python3
"""Run the local RAG vNext observation loop and store a snapshot."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from knowledge_hub.application.context import AppContextFactory
from knowledge_hub.application.rag_observation_loop import (
    DEFAULT_CORRECTIVE_EVAL_PATH,
    RAG_VNEXT_OBSERVATION_SCHEMA,
    build_rag_vnext_observation_report,
)
from knowledge_hub.core.schema_validator import validate_payload


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = Path("~/.khub/eval/knowledgeos/runs/rag_vnext")
REPORTS_DIR_NAME = "reports"
LATEST_JSON_NAME = "rag_vnext_latest.json"
LATEST_MD_NAME = "rag_vnext_latest.md"
SNAPSHOT_JSON_NAME = "rag_vnext_observation.json"
SNAPSHOT_MD_NAME = "rag_vnext_observation.md"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local RAG vNext observation loop and store a snapshot.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="knowledge-hub repo root")
    parser.add_argument("--config", default=None, help="optional khub config path")
    parser.add_argument("--runs-root", default=str(RUNS_ROOT), help="RAG vNext observation runs root")
    parser.add_argument("--queries", default=DEFAULT_CORRECTIVE_EVAL_PATH, help="corrective eval query CSV")
    parser.add_argument("-k", "--top-k", type=int, default=5)
    parser.add_argument("--mode", dest="retrieval_mode", default="hybrid", choices=["semantic", "keyword", "hybrid"])
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry-limit", type=int, default=None)
    parser.add_argument("--rerank-limit", type=int, default=None)
    parser.add_argument("--graph-limit", type=int, default=None)
    parser.add_argument(
        "--skip-if-local-date-already-covered",
        action="store_true",
        default=False,
        help="skip when the latest RAG vNext observation already belongs to the current local date",
    )
    parser.add_argument("--local-timezone", default="", help="optional IANA timezone for local-date skip checks")
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


def _now_in_timezone(now: datetime | None, tzinfo: Any) -> datetime:
    if now is None:
        return datetime.now(tzinfo)
    if now.tzinfo is None:
        return now.replace(tzinfo=tzinfo)
    return now.astimezone(tzinfo)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _validate_observation_payload(payload: dict[str, Any], *, label: str) -> None:
    schema = _clean_text(payload.get("schema"))
    if schema != RAG_VNEXT_OBSERVATION_SCHEMA:
        raise ValueError(f"{label} has unexpected schema: {schema or 'missing'}")
    result = validate_payload(payload, schema, strict=True)
    if not result.ok:
        raise ValueError(f"{label} failed schema validation: {result.errors}")


def _latest_snapshot_path(runs_root: Path) -> Path | None:
    latest = runs_root / REPORTS_DIR_NAME / LATEST_JSON_NAME
    if latest.exists():
        return latest
    candidates = sorted(runs_root.glob("rag_vnext_observation_*/rag_vnext_observation.json"))
    return candidates[-1] if candidates else None


def _latest_snapshot_local_date(runs_root: Path, *, tz_name: str) -> dict[str, str] | None:
    path = _latest_snapshot_path(runs_root)
    if path is None:
        return None
    try:
        payload = _read_json(path)
        _validate_observation_payload(payload, label=str(path))
    except Exception:
        return None
    generated_at = _clean_text(payload.get("generatedAt"))
    if not generated_at:
        return None
    generated_dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    if generated_dt.tzinfo is None:
        generated_dt = generated_dt.replace(tzinfo=timezone.utc)
    tz = _local_timezone(tz_name)
    return {
        "summaryPath": str(path),
        "generatedAt": generated_at,
        "localDate": generated_dt.astimezone(tz).date().isoformat(),
        "timezone": getattr(tz, "key", str(tz)),
    }


def _observation_count(runs_root: Path) -> int:
    return len(list(runs_root.glob("rag_vnext_observation_*/rag_vnext_observation.json")))


def _resolve_path(repo_root: Path, path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _build_searcher(repo_root: Path, config_path: str | None):
    factory = AppContextFactory(config_path=config_path, project_root=repo_root)
    return factory.get_searcher()


def render_markdown(payload: dict[str, Any]) -> str:
    summary = dict(payload.get("summary") or {})
    readiness = dict(payload.get("promotionReadiness") or {})
    metrics = dict((payload.get("correctiveEval") or {}).get("metrics") or {})
    lines = [
        "# RAG vNext Observation",
        "",
        f"- status: `{payload.get('status')}`",
        f"- generated_at: `{payload.get('generatedAt')}`",
        f"- rows: `{summary.get('rowCount', 0)}`",
        f"- corrective_pass_rate: `{summary.get('correctivePassRate')}`",
        f"- retry_candidates: `{summary.get('retryCandidateCount', 0)}`",
        f"- retry_applied: `{summary.get('retryAppliedCount', 0)}`",
        f"- retry_improvement_rate: `{summary.get('retryImprovementRate')}`",
        f"- rerank_changed_rank_count: `{summary.get('rerankChangedRankCount', 0)}`",
        f"- graph_candidate_count: `{summary.get('graphCandidateCount', 0)}`",
        f"- readiness: `{readiness.get('status')}`",
        "",
        "## Corrective Eval",
        "",
        f"- complexity_class_accuracy: `{metrics.get('complexityClassAccuracy')}`",
        f"- retry_candidate_accuracy: `{metrics.get('retryCandidateAccuracy')}`",
        f"- candidate_action_accuracy: `{metrics.get('candidateActionAccuracy')}`",
        "",
        "## Readiness Blockers",
    ]
    blockers = [str(item) for item in list(readiness.get("blockers") or [])]
    lines.extend(f"- {item}" for item in blockers) if blockers else lines.append("- none")
    failed_rows = [dict(row) for row in list((payload.get("correctiveEval") or {}).get("rows") or []) if not row.get("passed")]
    lines.extend(["", "## Failed Rows"])
    if failed_rows:
        for row in failed_rows[:20]:
            lines.append(
                f"- row={row.get('row')} scenario={row.get('scenario')} "
                f"expected={row.get('expected')} observed={row.get('observed')}"
            )
    else:
        lines.append("- none")
    warnings = [str(item) for item in list(payload.get("warnings") or []) if str(item)]
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {item}" for item in warnings[:20])
    return "\n".join(lines).rstrip() + "\n"


def run_daily_observation(args: argparse.Namespace, *, now: datetime | None = None, searcher: Any | None = None) -> dict[str, Any]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    runs_root = Path(args.runs_root).expanduser().resolve()
    queries_path = _resolve_path(repo_root, str(args.queries))

    if args.skip_if_local_date_already_covered:
        latest = _latest_snapshot_local_date(runs_root, tz_name=str(args.local_timezone))
        tz = _local_timezone(str(args.local_timezone))
        today_local = _now_in_timezone(now, tz).date().isoformat()
        if latest is not None and latest.get("localDate") == today_local:
            return {
                "skipped": True,
                "skipReason": "already_ran_for_local_date",
                "localDate": today_local,
                "timezone": latest.get("timezone", getattr(tz, "key", str(tz))),
                "latestSnapshot": latest,
            }

    now_utc = _now_in_timezone(now, timezone.utc)
    generated_at = now_utc.isoformat()
    stamp = now_utc.strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / f"rag_vnext_observation_{stamp}"
    reports_dir = runs_root / REPORTS_DIR_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    active_searcher = searcher if searcher is not None else _build_searcher(repo_root, args.config)
    payload = build_rag_vnext_observation_report(
        active_searcher,
        queries_path=queries_path,
        top_k=max(1, int(args.top_k)),
        retrieval_mode=str(args.retrieval_mode),
        alpha=float(args.alpha),
        limit=args.limit,
        retry_limit=args.retry_limit,
        rerank_limit=args.rerank_limit,
        graph_limit=args.graph_limit,
        observation_count=_observation_count(runs_root) + 1,
        generated_at=generated_at,
    )
    _validate_observation_payload(payload, label="RAG vNext observation payload")

    snapshot_json_path = run_dir / SNAPSHOT_JSON_NAME
    snapshot_md_path = run_dir / SNAPSHOT_MD_NAME
    latest_json_path = reports_dir / LATEST_JSON_NAME
    latest_md_path = reports_dir / LATEST_MD_NAME
    markdown = render_markdown(payload)
    snapshot_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
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
        "summary": dict(payload.get("summary") or {}),
        "promotionReadiness": dict(payload.get("promotionReadiness") or {}),
    }


def render_result(result: dict[str, Any]) -> str:
    if result.get("skipped") is True:
        latest = dict(result.get("latestSnapshot") or {})
        return "\n".join(
            [
                "skipped: True",
                f"reason: {result.get('skipReason', '')}",
                f"local_date: {result.get('localDate', '')}",
                f"timezone: {result.get('timezone', '')}",
                f"latest_snapshot_local_date: {latest.get('localDate', '')}",
                f"latest_snapshot_generated_at: {latest.get('generatedAt', '')}",
                f"latest_snapshot_summary: {latest.get('summaryPath', '')}",
            ]
        )
    summary = dict(result.get("summary") or {})
    readiness = dict(result.get("promotionReadiness") or {})
    return "\n".join(
        [
            f"status: {result.get('status', '')}",
            f"rows: {summary.get('rowCount', 0)}",
            f"corrective_pass_rate: {summary.get('correctivePassRate')}",
            f"retry_applied: {summary.get('retryAppliedCount', 0)}",
            f"graph_candidates: {summary.get('graphCandidateCount', 0)}",
            f"readiness: {readiness.get('status', '')}",
            f"snapshot_json: {result.get('snapshotJsonPath', '')}",
            f"latest_json: {result.get('latestJsonPath', '')}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_daily_observation(args)
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(render_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
