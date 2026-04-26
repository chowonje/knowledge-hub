"""Read-only Eval Center summary assembly."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVAL_CENTER_SUMMARY_SCHEMA = "knowledge-hub.eval-center.summary.result.v1"

_QUERY_SURFACES = {
    "paper_default_eval_queries_v1.csv": ["source-quality:paper", "collect_paper_default_eval.py"],
    "vault_default_eval_queries_v1.csv": ["source-quality:vault", "collect_vault_default_eval.py"],
    "knowledgeos_eval_queries_100_v1.csv": ["source-quality:web", "collect_web_default_eval.py"],
    "knowledgeos_ask_v2_eval_queries_v1.csv": ["khub labs eval run --profile ask-v2"],
    "user_answer_eval_queries_v1.csv": ["khub labs eval answer-loop collect"],
    "user_answer_optimize_seed_queries_v1.csv": ["khub labs eval answer-loop optimize"],
    "knowledgeos_embedding_eval_queries_30_v1.csv": ["embedding review seed"],
    "knowledgeos_embedding_eval_queries_20_v1.csv": ["embedding review seed"],
    "knowledgeos_hard_regression_pack_v0.csv": ["legacy report_eval hard pack"],
    "knowledgeos_paper_topic_eval_queries_20_v1.csv": ["paper topic eval"],
    "knowledgeos_paper_topic_eval_bad10_v1.csv": ["paper topic regression"],
    "paper_regression_eval_queries_v1.csv": ["paper regression collector"],
    "youtube_default_eval_queries_v1.csv": ["youtube default collector"],
}

_LATEST_REPORTS = {
    "sourceQualityObservation": "source_quality_observation_latest.json",
    "sourceQualityDetailObservation": "source_quality_detail_observation_latest.json",
    "sourceQualityTrend": "source_quality_trend_latest.json",
    "legacyRuntimeReadiness": "legacy_runtime_readiness_latest.json",
}

_COVERAGE_TESTS = [
    "tests/test_source_quality_battery.py",
    "tests/test_source_quality_trend_report.py",
    "tests/test_source_quality_observation_report.py",
    "tests/test_source_quality_detail_observation_report.py",
    "tests/test_source_quality_hard_gate.py",
    "tests/test_daily_source_quality_runner.py",
    "tests/test_eval_report_script.py",
    "tests/test_eval_gate.py",
    "tests/test_answer_loop.py",
    "tests/test_eval_cmd.py",
]


def build_eval_center_summary(
    *,
    runs_root: str | Path = "eval/knowledgeos/runs",
    queries_dir: str | Path = "eval/knowledgeos/queries",
    failure_bank_path: str | Path = "~/.khub/eval/knowledgeos/failures/failure_bank.jsonl",
    repo_root: str | Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a read-only inventory of eval assets and current eval status."""

    root = _resolve_path(repo_root or Path.cwd())
    runs_path = _resolve_path(runs_root, base=root)
    queries_path = _resolve_path(queries_dir, base=root)
    warnings: list[str] = []

    source_quality = _build_source_quality_status(runs_path, warnings)
    answer_loop = _build_answer_loop_status(runs_path, warnings)
    failure_bank = _build_failure_bank_status(_resolve_path(failure_bank_path, base=root), warnings)
    query_inventory = _build_query_inventory(queries_path, warnings)
    report_index = _build_report_index(root, runs_path, warnings)
    coverage = _build_coverage(root)
    gaps = _build_gaps(source_quality, answer_loop, failure_bank)
    recommendations = _build_recommendations(source_quality, answer_loop, failure_bank, gaps)

    status = (
        "missing"
        if not runs_path.exists() or not queries_path.exists()
        else "warn"
        if warnings
        else "ok"
    )
    return {
        "schema": EVAL_CENTER_SUMMARY_SCHEMA,
        "status": status,
        "generatedAt": generated_at or datetime.now(timezone.utc).isoformat(),
        "repoRoot": str(root),
        "runsRoot": str(runs_path),
        "runsRootPhysical": str(runs_path.resolve()) if runs_path.exists() else "",
        "runsRootIsSymlink": runs_path.is_symlink(),
        "queriesDir": str(queries_path),
        "sourceQuality": source_quality,
        "answerLoop": answer_loop,
        "failureBank": failure_bank,
        "queryInventory": query_inventory,
        "reportIndex": report_index,
        "coverage": coverage,
        "gaps": gaps,
        "recommendations": recommendations,
        "warnings": warnings,
    }


def _resolve_path(value: str | Path, *, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return ((base or Path.cwd()) / path).resolve()


def _read_json(path: Path, warnings: list[str], *, label: str) -> dict[str, Any]:
    if not path.exists():
        warnings.append(f"missing {label}: {path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:  # pragma: no cover - exact JSON error text varies by Python version.
        warnings.append(f"failed to read {label}: {path}: {error}")
        return {}
    if not isinstance(payload, dict):
        warnings.append(f"{label} is not a JSON object: {path}")
        return {}
    return payload


def _read_jsonl(path: Path, warnings: list[str], *, label: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        raw_lines = path.read_bytes().splitlines()
    except Exception as error:  # pragma: no cover - exact OS error text varies.
        warnings.append(f"failed to read {label}: {path}: {error}")
        return []
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.strip():
            continue
        try:
            row_text = raw_line.decode("utf-8").strip()
        except UnicodeDecodeError as error:
            warnings.append(f"failed to read {label}: {path}: line {line_number}: invalid utf-8: {error}")
            continue
        if not row_text:
            continue
        try:
            payload = json.loads(row_text)
        except Exception as error:
            warnings.append(f"failed to read {label}: {path}: line {line_number}: {error}")
            continue
        if isinstance(payload, dict):
            rows.append(payload)
        else:
            warnings.append(f"{label} line is not a JSON object: {path}: line {line_number}")
    return rows


def _latest_existing(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: (path.stat().st_mtime, str(path)))


def _path_summary(path: Path) -> dict[str, Any]:
    modified_at = ""
    if path.exists():
        modified_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    return {
        "path": str(path),
        "physicalPath": str(path.resolve()) if path.exists() else "",
        "exists": path.exists(),
        "modifiedAt": modified_at,
    }


def _compact_report(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **_path_summary(path),
        "schema": payload.get("schema", ""),
        "status": payload.get("status", ""),
        "decision": payload.get("decision", ""),
        "createdAt": payload.get("created_at", "") or payload.get("createdAt", ""),
        "latestRunDir": payload.get("latest_run_dir", "") or payload.get("latestRunDir", ""),
        "runCount": payload.get("run_count", "") or payload.get("runCount", ""),
        "requiredRuns": payload.get("required_runs", "") or payload.get("requiredRuns", ""),
        "blockers": list(payload.get("blockers") or []),
    }


def _build_source_quality_status(runs_root: Path, warnings: list[str]) -> dict[str, Any]:
    reports_root = runs_root / "reports"
    report_payloads: dict[str, dict[str, Any]] = {}
    reports: dict[str, dict[str, Any]] = {}
    for key, file_name in _LATEST_REPORTS.items():
        path = reports_root / file_name
        payload = _read_json(path, warnings, label=key)
        report_payloads[key] = payload
        reports[key] = _compact_report(path, payload)

    observation_latest_dir = _text(
        report_payloads.get("sourceQualityObservation", {}).get("latest_run_dir")
        or report_payloads.get("sourceQualityObservation", {}).get("latestRunDir")
    )
    latest_battery_path = None
    if observation_latest_dir:
        candidate_dir = Path(observation_latest_dir).expanduser()
        if not candidate_dir.is_absolute():
            candidate_dir = runs_root / candidate_dir
        candidate_summary = candidate_dir / "source_quality_battery_summary.json"
        if candidate_summary.exists():
            latest_battery_path = candidate_summary
    if latest_battery_path is None:
        battery_paths = list(runs_root.glob("source_quality_battery_*/source_quality_battery_summary.json"))
        latest_battery_path = _latest_existing(battery_paths)
    latest_battery = (
        _read_json(latest_battery_path, warnings, label="latest source-quality battery")
        if latest_battery_path
        else {}
    )
    if latest_battery_path is None:
        warnings.append(f"missing latest source-quality battery under {runs_root}")

    detail_blockers = list(report_payloads.get("sourceQualityDetailObservation", {}).get("blockers") or [])
    if detail_blockers:
        warnings.append(
            "source-quality detail observation has blockers: "
            + ", ".join(str(item) for item in detail_blockers)
        )

    base_blockers = list(report_payloads.get("sourceQualityObservation", {}).get("blockers") or [])
    if base_blockers:
        warnings.append("source-quality observation has blockers: " + ", ".join(str(item) for item in base_blockers))

    per_source_payload = latest_battery.get("per_source") or {}
    per_source = []
    for source, item in sorted(per_source_payload.items()):
        if not isinstance(item, dict):
            continue
        per_source.append(
            {
                "source": source,
                "rows": item.get("rows", 0),
                "routeCorrectness": item.get("route_correctness"),
                "noResultRate": item.get("no_result_rate"),
                "runtimeUsedCounts": item.get("runtime_used_counts") or {},
                "fallbackReasonCounts": item.get("fallback_reason_counts") or {},
                "legacyRuntimeRate": item.get("legacy_runtime_rate"),
                "capabilityMissingRate": item.get("capability_missing_rate"),
                "forcedLegacyRate": item.get("forced_legacy_rate"),
                "staleCitationRate": item.get("stale_citation_rate"),
                "softMetrics": {
                    key: value
                    for key, value in item.items()
                    if key.endswith("_soft") or key in {"latest_source_age_days_p50"}
                },
            }
        )

    return {
        "latestRun": {
            **(
                _path_summary(latest_battery_path)
                if latest_battery_path
                else {"path": "", "physicalPath": "", "exists": False}
            ),
            "createdAt": latest_battery.get("created_at", ""),
            "gateMode": latest_battery.get("gate_mode", ""),
            "retrievalMode": latest_battery.get("retrieval_mode", ""),
            "topK": latest_battery.get("top_k", ""),
        },
        "baseObservation": reports["sourceQualityObservation"],
        "detailObservation": reports["sourceQualityDetailObservation"],
        "trend": reports["sourceQualityTrend"],
        "legacyRuntimeReadiness": reports["legacyRuntimeReadiness"],
        "hardGates": latest_battery.get("hard_gates") or {},
        "softGates": latest_battery.get("soft_gates") or {},
        "perSource": per_source,
    }


def _build_answer_loop_status(runs_root: Path, warnings: list[str]) -> dict[str, Any]:
    answer_root = runs_root / "answer_loop"
    latest_alias = answer_root / "latest"
    if answer_root.exists() and not latest_alias.exists():
        warnings.append(f"missing answer-loop latest alias: {latest_alias}")

    summary_path = None
    latest_alias_summary = latest_alias / "answer_loop_summary.json"
    if latest_alias_summary.exists():
        summary_path = latest_alias_summary
    if summary_path is None:
        summary_path = _latest_existing(list(answer_root.glob("**/answer_loop_summary.json")))
    if summary_path is None:
        warnings.append(f"missing answer-loop summary under {answer_root}")
        run_dir = None
    else:
        run_dir = summary_path.parent

    collect_path = (run_dir / "answer_loop_collect_manifest.json") if run_dir else None
    judge_path = (run_dir / "answer_loop_judge_manifest.json") if run_dir else None
    if collect_path is None or not collect_path.exists():
        collect_path = _latest_existing(list(answer_root.glob("**/answer_loop_collect_manifest.json")))
    if judge_path is None or not judge_path.exists():
        judge_path = _latest_existing(list(answer_root.glob("**/answer_loop_judge_manifest.json")))

    summary = _read_json(summary_path, warnings, label="latest answer-loop summary") if summary_path else {}
    collect = _read_json(collect_path, warnings, label="latest answer-loop collect manifest") if collect_path else {}
    judge = _read_json(judge_path, warnings, label="latest answer-loop judge manifest") if judge_path else {}
    request = collect.get("request") or {}

    return {
        "root": _path_summary(answer_root),
        "latestAlias": _path_summary(latest_alias),
        "latestRunDir": str(run_dir) if run_dir else "",
        "collect": {
            **(
                _path_summary(collect_path)
                if collect_path
                else {"path": "", "physicalPath": "", "exists": False}
            ),
            "status": collect.get("status", ""),
            "rowCount": collect.get("rowCount", 0),
            "packetCount": collect.get("packetCount", 0),
            "querySet": request.get("queriesPath", ""),
            "answerBackends": list(request.get("answerBackends") or []),
            "backendModels": request.get("backendModels") or {},
            "retrievalMode": request.get("retrievalMode", ""),
            "topK": request.get("topK", ""),
        },
        "judge": {
            **(
                _path_summary(judge_path)
                if judge_path
                else {"path": "", "physicalPath": "", "exists": False}
            ),
            "status": judge.get("status", ""),
            "rowCount": judge.get("rowCount", 0),
            "judgeProvider": judge.get("judgeProvider", ""),
            "judgeModel": judge.get("judgeModel", ""),
        },
        "summary": {
            **(
                _path_summary(summary_path)
                if summary_path
                else {"path": "", "physicalPath": "", "exists": False}
            ),
            "status": summary.get("status", ""),
            "rowCount": summary.get("rowCount", 0),
            "overall": summary.get("overall") or {},
            "failureBucketCounts": summary.get("failureBucketCounts") or {},
            "failureCardCount": summary.get("failureCardCount", 0),
        },
    }


def _build_failure_bank_status(path: Path, warnings: list[str]) -> dict[str, Any]:
    rows = _read_jsonl(path, warnings, label="failure bank")
    bucket_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for row in rows:
        bucket = str(row.get("bucket") or "unknown")
        status = str(row.get("status") or "unknown")
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        **_path_summary(path),
        "recordCount": len(rows),
        "bucketCounts": bucket_counts,
        "statusCounts": status_counts,
    }


def _count_csv(path: Path) -> tuple[int, list[str], list[str]]:
    parse_warnings: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [str(item) for item in (reader.fieldnames or [])]
        count = 0
        for row_number, row in enumerate(reader, start=2):
            count += 1
            extras = row.get(None)
            if extras:
                parse_warnings.append(f"row {row_number} has {len(extras)} extra field(s)")
    return count, fieldnames, parse_warnings


def _build_query_inventory(queries_dir: Path, warnings: list[str]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    if not queries_dir.exists():
        warnings.append(f"missing queries dir: {queries_dir}")
        return {"path": str(queries_dir), "exists": False, "count": 0, "items": items}

    for path in sorted(queries_dir.glob("*.csv")):
        try:
            row_count, fieldnames, parse_warnings = _count_csv(path)
        except Exception as error:  # pragma: no cover - exact CSV error text varies by Python version.
            row_count, fieldnames, parse_warnings = 0, [], [f"failed to parse CSV: {error}"]
        for warning in parse_warnings:
            warnings.append(f"{path.name}: {warning}")
        items.append(
            {
                "path": str(path),
                "fileName": path.name,
                "rowCount": row_count,
                "fieldNames": fieldnames,
                "surfaces": list(_QUERY_SURFACES.get(path.name, ["manual/unknown"])),
                "parseWarnings": parse_warnings,
            }
        )
    return {"path": str(queries_dir), "exists": True, "count": len(items), "items": items}


def _build_report_index(repo_root: Path, runs_root: Path, warnings: list[str]) -> dict[str, Any]:
    reports_root = runs_root / "reports"
    latest = []
    for key, file_name in _LATEST_REPORTS.items():
        path = reports_root / file_name
        latest.append({"name": key, **_path_summary(path)})

    if not reports_root.exists():
        warnings.append(f"missing eval reports dir: {reports_root}")
        report_files: list[Path] = []
    else:
        report_files = [path for path in reports_root.iterdir() if path.is_file()]

    manual_root = repo_root / "eval" / "knowledgeos" / "reports"
    manual_reports = [path for path in sorted(manual_root.glob("*")) if path.is_file()] if manual_root.exists() else []
    return {
        "reportsRoot": _path_summary(reports_root),
        "latestReports": latest,
        "jsonReportCount": sum(1 for path in report_files if path.suffix == ".json"),
        "markdownReportCount": sum(1 for path in report_files if path.suffix == ".md"),
        "manualReportsRoot": _path_summary(manual_root),
        "manualReportCount": len(manual_reports),
        "manualReports": [str(path) for path in manual_reports[:20]],
    }


def _build_coverage(repo_root: Path) -> dict[str, Any]:
    items = []
    for rel_path in _COVERAGE_TESTS:
        path = repo_root / rel_path
        items.append({"path": str(path), "exists": path.exists()})
    return {"tests": items}


def _build_gaps(
    source_quality: dict[str, Any],
    answer_loop: dict[str, Any],
    failure_bank: dict[str, Any],
) -> list[dict[str, str]]:
    gaps = [
        {
            "id": "eval_cases_store",
            "status": "csv_inventory_only",
            "summary": "Eval cases are still CSV-backed; there is no schema-backed EvalCase table or durable run ledger.",
        },
    ]
    if not failure_bank.get("exists"):
        gaps.insert(
            0,
            {
                "id": "failure_bank",
                "status": "missing_first_class_store",
                "summary": "Answer-loop failure cards exist as artifacts, but there is no first-class Failure Bank store yet.",
            },
        )
    if (source_quality.get("detailObservation") or {}).get("decision") != "ready_for_detail_gate_review":
        gaps.append(
            {
                "id": "detail_quality_gate",
                "status": "observation_only",
                "summary": "Detail-quality promotion is not ready yet; keep it out of hard-gate enforcement.",
            }
        )
    if not (answer_loop.get("latestAlias") or {}).get("exists"):
        gaps.append(
            {
                "id": "answer_loop_latest_alias",
                "status": "missing",
                "summary": "Docs mention answer_loop/latest, but current discovery must use artifact mtimes.",
            }
        )
    return gaps


def _build_recommendations(
    source_quality: dict[str, Any],
    answer_loop: dict[str, Any],
    failure_bank: dict[str, Any],
    gaps: list[dict[str, str]],
) -> list[str]:
    recommendations = [
        "Keep Eval Center read-only until the summary contract is stable.",
        "Move CSV eval rows toward schema-backed EvalCase records before adding heavier optimizer workflows.",
    ]
    if failure_bank.get("exists"):
        recommendations.append("Keep Failure Bank synced after answer-loop summarize runs.")
    else:
        recommendations.append("Promote a first-class Failure Bank after answer-loop failure cards are consistently produced.")
    if any(item.get("id") == "detail_quality_gate" for item in gaps):
        recommendations.append("Wait for seven stable detail-quality points before enforcing the detail gate.")
    if not (answer_loop.get("summary") or {}).get("exists"):
        recommendations.append("Run `khub labs eval answer-loop summarize` after the next judged answer-loop run.")
    if (source_quality.get("baseObservation") or {}).get("decision") == "ready_for_hard_gate_review":
        recommendations.append("Base source-quality can remain a local hard gate while detail-quality stays observation-only.")
    return recommendations


def _text(value: Any) -> str:
    return str(value or "").strip()
