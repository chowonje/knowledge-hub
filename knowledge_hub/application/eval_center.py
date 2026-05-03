"""Read-only Eval Center summary assembly."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from knowledge_hub.application.eval_cases import DEFAULT_EVAL_CASE_REGISTRY_PATH, list_eval_cases


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
    "vault_compare_answer_eval_queries_v1.csv": ["khub labs eval answer-loop collect"],
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
    eval_case_registry_path: str | Path = DEFAULT_EVAL_CASE_REGISTRY_PATH,
    repo_root: str | Path | None = None,
    generated_at: str | None = None,
    expected_source_quality_local_date: str | None = None,
    freshness_timezone: str = "UTC",
) -> dict[str, Any]:
    """Build a read-only inventory of eval assets and current eval status."""

    root = _resolve_path(repo_root or Path.cwd())
    runs_path = _resolve_path(runs_root, base=root)
    queries_path = _resolve_path(queries_dir, base=root)
    generated_at_value = generated_at or datetime.now(timezone.utc).isoformat()
    warnings: list[str] = []

    source_quality = _build_source_quality_status(
        runs_path,
        warnings,
        expected_local_date=expected_source_quality_local_date,
        freshness_timezone=freshness_timezone,
    )
    answer_loop = _build_answer_loop_status(runs_path, warnings)
    failure_bank = _build_failure_bank_status(_resolve_path(failure_bank_path, base=root), warnings)
    eval_cases = _build_eval_case_registry_status(_resolve_path(eval_case_registry_path, base=root), warnings)
    query_inventory = _build_query_inventory(queries_path, warnings)
    report_index = _build_report_index(root, runs_path, warnings)
    coverage = _build_coverage(root)
    gaps = _build_gaps(source_quality, answer_loop, failure_bank, eval_cases)
    recommendations = _build_recommendations(source_quality, answer_loop, failure_bank, eval_cases, gaps)
    operator_brief = _build_operator_brief(
        source_quality=source_quality,
        answer_loop=answer_loop,
        failure_bank=failure_bank,
        eval_cases=eval_cases,
        query_inventory=query_inventory,
        gaps=gaps,
        recommendations=recommendations,
        warnings=warnings,
    )

    status = (
        "missing"
        if not runs_path.exists() or not queries_path.exists()
        else "warn"
        if warnings or gaps
        else "ok"
    )
    return {
        "schema": EVAL_CENTER_SUMMARY_SCHEMA,
        "status": status,
        "generatedAt": generated_at_value,
        "repoRoot": str(root),
        "runsRoot": str(runs_path),
        "runsRootPhysical": str(runs_path.resolve()) if runs_path.exists() else "",
        "runsRootIsSymlink": runs_path.is_symlink(),
        "queriesDir": str(queries_path),
        "sourceQuality": source_quality,
        "answerLoop": answer_loop,
        "failureBank": failure_bank,
        "evalCases": eval_cases,
        "queryInventory": query_inventory,
        "reportIndex": report_index,
        "coverage": coverage,
        "gaps": gaps,
        "recommendations": recommendations,
        "operatorBrief": operator_brief,
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
            token = raw_line.decode("utf-8").strip()
        except UnicodeDecodeError as error:
            warnings.append(f"failed to read {label}: {path}: line {line_number}: invalid utf-8: {error}")
            continue
        if not token:
            continue
        try:
            payload = json.loads(token)
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


def _parse_datetime(value: Any) -> datetime | None:
    token = str(value or "").strip()
    if not token:
        return None
    try:
        parsed = datetime.fromisoformat(token.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _safe_zoneinfo(name: str, warnings: list[str]) -> ZoneInfo:
    token = str(name or "").strip() or "UTC"
    try:
        return ZoneInfo(token)
    except ZoneInfoNotFoundError:
        warnings.append(f"unknown freshness timezone {token}; using UTC")
        return ZoneInfo("UTC")


def _source_quality_freshness(
    *,
    latest_battery_path: Path | None,
    latest_battery: dict[str, Any],
    expected_local_date: str | None,
    freshness_timezone: str,
    warnings: list[str],
) -> dict[str, Any]:
    expected = _text(expected_local_date)
    timezone_name = _text(freshness_timezone) or "UTC"
    freshness = {
        "status": "not_checked",
        "timezone": timezone_name,
        "expectedLocalDate": expected,
        "latestRunLocalDate": "",
        "latestRunTimestamp": "",
    }
    if not expected:
        return freshness

    if latest_battery_path is None or not latest_battery_path.exists():
        freshness["status"] = "missing"
        warnings.append(f"source-quality latest run missing for local date {expected} ({timezone_name})")
        return freshness

    latest_dt = _parse_datetime(latest_battery.get("created_at") or latest_battery.get("createdAt"))
    if latest_dt is None:
        latest_dt = datetime.fromtimestamp(latest_battery_path.stat().st_mtime, timezone.utc)
    tz = _safe_zoneinfo(timezone_name, warnings)
    latest_local_date = latest_dt.astimezone(tz).date().isoformat()
    freshness["latestRunLocalDate"] = latest_local_date
    freshness["latestRunTimestamp"] = latest_dt.isoformat()
    if latest_local_date == expected:
        freshness["status"] = "fresh"
        return freshness

    freshness["status"] = "stale"
    warnings.append(
        "source-quality latest run is stale for local date "
        f"{expected} ({timezone_name}); latest run local date is {latest_local_date}"
    )
    return freshness


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


def _build_source_quality_status(
    runs_root: Path,
    warnings: list[str],
    *,
    expected_local_date: str | None = None,
    freshness_timezone: str = "UTC",
) -> dict[str, Any]:
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

    freshness = _source_quality_freshness(
        latest_battery_path=latest_battery_path,
        latest_battery=latest_battery,
        expected_local_date=expected_local_date,
        freshness_timezone=freshness_timezone,
        warnings=warnings,
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
        "freshness": freshness,
    }


def _build_answer_loop_status(runs_root: Path, warnings: list[str]) -> dict[str, Any]:
    answer_root = runs_root / "answer_loop"
    latest_alias = answer_root / "latest"

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


def _build_eval_case_registry_status(path: Path, warnings: list[str]) -> dict[str, Any]:
    payload = list_eval_cases(registry_path=path, limit=0)
    for warning in list(payload.get("warnings") or []):
        warnings.append(f"eval case registry: {warning}")
    return {
        **_path_summary(Path(str(payload.get("registryPath") or path))),
        "status": payload.get("status", ""),
        "recordCount": payload.get("totalRecordCount", payload.get("recordCount", 0)),
        "laneCounts": payload.get("laneCounts") or {},
        "sourceTypeCounts": payload.get("sourceTypeCounts") or {},
        "statusCounts": payload.get("statusCounts") or {},
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
    eval_cases: dict[str, Any],
) -> list[dict[str, str]]:
    gaps = []
    if not eval_cases.get("exists"):
        gaps.append(
            {
                "id": "eval_cases_store",
                "status": "missing_registry",
                "summary": "Eval cases are still CSV-backed; there is no schema-backed EvalCase registry yet.",
            }
        )
    elif int(eval_cases.get("recordCount") or 0) <= 0:
        gaps.append(
            {
                "id": "eval_cases_store",
                "status": "empty_registry",
                "summary": "EvalCase registry exists but has no active records yet.",
            }
        )
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
    if not (answer_loop.get("latestAlias") or {}).get("exists") and not (answer_loop.get("summary") or {}).get("exists"):
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
    eval_cases: dict[str, Any],
    gaps: list[dict[str, str]],
) -> list[str]:
    recommendations = [
        "Keep Eval Center read-only until the summary contract is stable.",
    ]
    if eval_cases.get("exists") and int(eval_cases.get("recordCount") or 0) > 0:
        recommendations.append("Keep EvalCase registry synced when canonical CSV query sets change.")
    else:
        recommendations.append("Import canonical CSV eval rows into the schema-backed EvalCase registry before adding heavier optimizer workflows.")
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


def _strings(value: Any) -> list[str]:
    return [str(item).strip() for item in list(value or []) if str(item).strip()]


def _section(
    *,
    section_id: str,
    title: str,
    status: str,
    ran: list[str],
    problem: list[str],
    next_action: str,
    details: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": section_id,
        "title": title,
        "status": status,
        "ran": ran,
        "problem": problem or ["none"],
        "nextAction": next_action,
        "details": details or [],
    }


def _finding(severity: str, part: str, title: str, body: str) -> dict[str, str]:
    return {"severity": severity, "part": part, "title": title, "body": body}


def _metric(value: Any) -> Any:
    return "n/a" if value is None or value == "" else value


def _build_operator_brief(
    *,
    source_quality: dict[str, Any],
    answer_loop: dict[str, Any],
    failure_bank: dict[str, Any],
    eval_cases: dict[str, Any],
    query_inventory: dict[str, Any],
    gaps: list[dict[str, str]],
    recommendations: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    sections: list[dict[str, Any]] = []

    latest_run = dict(source_quality.get("latestRun") or {})
    freshness = dict(source_quality.get("freshness") or {})
    base = dict(source_quality.get("baseObservation") or {})
    detail = dict(source_quality.get("detailObservation") or {})
    base_blockers = _strings(base.get("blockers"))
    detail_blockers = _strings(detail.get("blockers"))
    base_decision = _text(base.get("decision") or base.get("status") or "unknown")
    detail_decision = _text(detail.get("decision") or detail.get("status") or "unknown")
    freshness_status = _text(freshness.get("status") or "not_checked")
    source_status = "ok"
    source_problem: list[str] = []
    source_next = "Keep source-quality as the local daily hard gate."
    if freshness_status in {"missing", "stale"}:
        source_status = "failed"
        source_problem = [
            "source-quality latest run is "
            f"{freshness_status} for {freshness.get('expectedLocalDate') or 'the expected local date'}"
        ]
        source_next = "Repair and rerun source-quality before trusting the daily brief."
        findings.append(
            _finding(
                "P1",
                "source_quality",
                "Daily source-quality result is not fresh",
                "; ".join(source_problem),
            )
        )
    elif base_blockers or base_decision not in {"ready_for_hard_gate_review", "unknown"}:
        source_status = "failed" if base_blockers else "warn"
        source_problem = base_blockers or [f"base observation decision is {base_decision}"]
        source_next = "Triage source-quality before treating research or answer-loop results as canonical."
        findings.append(
            _finding(
                "P1",
                "source_quality",
                "Base source-quality gate is not ready",
                "; ".join(source_problem),
            )
        )
    elif detail_blockers or detail_decision != "ready_for_detail_gate_review":
        source_status = "warn"
        source_problem = detail_blockers or [f"detail observation decision is {detail_decision}"]
        source_next = "Keep detail-quality observation-only until the blockers clear."
        findings.append(
            _finding(
                "P2",
                "source_quality",
                "Detail-quality promotion is still blocked",
                "; ".join(source_problem),
            )
        )

    source_details = []
    for item in list(source_quality.get("perSource") or []):
        if not isinstance(item, dict):
            continue
        source_details.append(
            "source={source} rows={rows} route={route} stale_citations={stale} legacy={legacy} capability_missing={capability}".format(
                source=item.get("source", ""),
                rows=item.get("rows", 0),
                route=_metric(item.get("routeCorrectness")),
                stale=_metric(item.get("staleCitationRate")),
                legacy=_metric(item.get("legacyRuntimeRate")),
                capability=_metric(item.get("capabilityMissingRate")),
            )
        )
    sections.append(
        _section(
            section_id="source_quality",
            title="Source Quality Gate",
            status=source_status,
            ran=[
                f"latest_run={latest_run.get('path') or 'missing'}",
                f"latest_run_modified_at={latest_run.get('modifiedAt') or 'unknown'}",
                f"freshness={freshness_status}",
                f"freshness_expected_date={freshness.get('expectedLocalDate') or 'not_checked'}",
                f"freshness_latest_date={freshness.get('latestRunLocalDate') or 'unknown'}",
                f"base={base_decision}",
                f"detail={detail_decision}",
            ],
            problem=source_problem,
            next_action=source_next,
            details=source_details,
        )
    )

    answer_summary = dict(answer_loop.get("summary") or {})
    latest_alias = dict(answer_loop.get("latestAlias") or {})
    failure_card_count = int(answer_summary.get("failureCardCount") or 0)
    failure_buckets = dict(answer_summary.get("failureBucketCounts") or {})
    answer_status = _text(answer_summary.get("status") or "missing")
    answer_problem: list[str] = []
    answer_section_status = "ok"
    if answer_status != "ok":
        answer_section_status = "failed" if answer_status == "missing" else "warn"
        answer_problem.append(f"answer-loop summary status is {answer_status}")
    if not latest_alias.get("exists") and answer_status == "missing":
        answer_section_status = "warn" if answer_section_status == "ok" else answer_section_status
        answer_problem.append("answer-loop latest alias is missing")
        findings.append(
            _finding(
                "P2",
                "answer_loop",
                "Latest answer-loop alias is missing",
                "Eval Center must discover the latest answer-loop artifact by mtime instead of a stable alias.",
            )
        )
    if failure_card_count:
        answer_section_status = "warn" if answer_section_status == "ok" else answer_section_status
        answer_problem.append(f"{failure_card_count} answer-loop failure card(s) are present")
        if not failure_bank.get("exists"):
            findings.append(
                _finding(
                    "P2",
                    "answer_loop",
                    "Answer-loop failures are not first-class yet",
                    "Failure cards exist, but they are not promoted into a Failure Bank.",
                )
            )
    sections.append(
        _section(
            section_id="answer_loop",
            title="Answer Loop",
            status=answer_section_status,
            ran=[
                f"latest_run={answer_loop.get('latestRunDir') or 'missing'}",
                f"summary_modified_at={answer_summary.get('modifiedAt') or 'unknown'}",
                f"status={answer_status}",
                f"rows={answer_summary.get('rowCount', 0)}",
            ],
            problem=answer_problem,
            next_action="Promote stable failure cards into Failure Bank v0 before EvalCase registry work.",
            details=[f"{key}={value}" for key, value in sorted(failure_buckets.items())],
        )
    )

    failure_bank_problem: list[str] = []
    failure_bank_status = "ok"
    if not failure_bank.get("exists"):
        failure_bank_status = "warn"
        failure_bank_problem.append("failure_bank.jsonl is missing")
    elif int(failure_bank.get("recordCount") or 0) == 0:
        failure_bank_status = "warn"
        failure_bank_problem.append("failure bank has no records")
    sections.append(
        _section(
            section_id="failure_bank",
            title="Failure Bank",
            status=failure_bank_status,
            ran=[
                f"path={failure_bank.get('path') or 'missing'}",
                f"modified_at={failure_bank.get('modifiedAt') or 'unknown'}",
                f"records={failure_bank.get('recordCount', 0)}",
            ],
            problem=failure_bank_problem,
            next_action="Use `khub labs eval failure-bank sync` after answer-loop summarize to preserve failure cards.",
            details=[
                f"status_counts={failure_bank.get('statusCounts') or {}}",
                f"bucket_counts={failure_bank.get('bucketCounts') or {}}",
            ],
        )
    )

    eval_cases_problem: list[str] = []
    eval_cases_status = "ok"
    eval_case_count = int(eval_cases.get("recordCount") or 0)
    if not eval_cases.get("exists"):
        eval_cases_status = "warn"
        eval_cases_problem.append("eval_cases.jsonl is missing")
    elif eval_case_count <= 0:
        eval_cases_status = "warn"
        eval_cases_problem.append("EvalCase registry has no records")
    sections.append(
        _section(
            section_id="eval_cases",
            title="EvalCase Registry",
            status=eval_cases_status,
            ran=[
                f"path={eval_cases.get('path') or 'missing'}",
                f"modified_at={eval_cases.get('modifiedAt') or 'unknown'}",
                f"records={eval_case_count}",
            ],
            problem=eval_cases_problem,
            next_action="Import canonical CSV rows with `khub labs eval eval-case import-csv` before using optimizer workflows.",
            details=[
                f"lane_counts={eval_cases.get('laneCounts') or {}}",
                f"source_type_counts={eval_cases.get('sourceTypeCounts') or {}}",
                f"status_counts={eval_cases.get('statusCounts') or {}}",
            ],
        )
    )

    parse_warnings = []
    for item in list(query_inventory.get("items") or []):
        if not isinstance(item, dict):
            continue
        for warning in _strings(item.get("parseWarnings")):
            parse_warnings.append(f"{item.get('fileName')}: {warning}")
    if parse_warnings:
        findings.append(
            _finding(
                "P2",
                "query_inventory",
                "Query CSV parse warnings are present",
                "; ".join(parse_warnings[:3]),
            )
        )
    sections.append(
        _section(
            section_id="query_inventory",
            title="Query Inventory",
            status="warn" if parse_warnings else "ok",
            ran=[f"query_sets={query_inventory.get('count', 0)}", f"dir={query_inventory.get('path', '')}"],
            problem=parse_warnings,
            next_action="Fix CSV rows with extra fields before promoting schema-backed EvalCase records.",
        )
    )

    gap_ids = [_text(item.get("id")) for item in gaps if _text(item.get("id"))]
    for gap in gaps:
        gap_id = _text(gap.get("id"))
        if gap_id in {"failure_bank", "eval_cases_store"}:
            findings.append(_finding("P2", "eval_maturity", gap_id, _text(gap.get("summary"))))
    sections.append(
        _section(
            section_id="eval_maturity",
            title="Eval Maturity",
            status="warn" if gap_ids else "ok",
            ran=[f"gaps={len(gap_ids)}", f"warnings={len(warnings)}"],
            problem=gap_ids,
            next_action=recommendations[0] if recommendations else "No action required.",
        )
    )

    priority = "normal"
    if any(item.get("severity") == "P1" for item in findings):
        priority = "quality_triage"
    elif any(item.get("part") == "answer_loop" for item in findings):
        priority = "answer_loop_triage"
    elif parse_warnings:
        priority = "eval_hygiene"
    elif any(item.get("part") == "source_quality" for item in findings):
        priority = "detail_observation"

    return {
        "summary": {
            "priority": priority,
            "findingCount": len(findings),
            "sectionCount": len(sections),
        },
        "sections": sections,
        "findings": findings,
    }
