"""Schema-backed EvalCase registry helpers.

This MVP keeps EvalCase records local, inspectable, and easy to migrate from
existing CSV-backed eval sheets without introducing a database dependency.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EVAL_CASE_SCHEMA = "knowledge-hub.eval-case.v1"
EVAL_CASE_REGISTRY_SCHEMA = "knowledge-hub.eval-case-registry.result.v1"
DEFAULT_EVAL_CASE_REGISTRY_PATH = Path("~/.khub/eval/knowledgeos/eval_cases/eval_cases.jsonl")


def _now_iso(now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc).isoformat()


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _clean_key(value: Any) -> str:
    return " ".join(_clean(value).casefold().split())


def _resolve_path(path: str | Path, *, base: str | Path | None = None) -> Path:
    target = Path(path).expanduser()
    if target.is_absolute():
        return target
    return (Path(base or Path.cwd()).expanduser() / target).resolve()


def _parse_int(value: Any) -> int | None:
    text = _clean(value)
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _parse_bool(value: Any) -> bool | None:
    text = _clean(value).casefold()
    if not text:
        return None
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _deterministic_eval_case_id(*, lane: str, source_type: str, scenario_type: str, query: str) -> str:
    token = "|".join(
        [
            _clean_key(lane),
            _clean_key(source_type),
            _clean_key(scenario_type),
            _clean_key(query),
        ]
    )
    return "ec_" + hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], []
    try:
        raw_lines = path.read_bytes().splitlines()
    except Exception as error:
        return [], [f"{path}: {error}"]
    items: list[dict[str, Any]] = []
    warnings: list[str] = []
    for line_number, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.strip():
            continue
        try:
            token = raw_line.decode("utf-8").strip()
        except UnicodeDecodeError as error:
            warnings.append(f"{path}: line {line_number}: invalid utf-8: {error}")
            continue
        if not token:
            continue
        try:
            payload = json.loads(token)
        except Exception as error:
            warnings.append(f"{path}: line {line_number}: {error}")
            continue
        if isinstance(payload, dict):
            items.append(payload)
        else:
            warnings.append(f"{path}: line {line_number}: expected JSON object")
    return items, warnings


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in rows)
    path.write_text((text + "\n") if text else "", encoding="utf-8")


def _sort_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda item: (
            _clean(item.get("lane")),
            _clean(item.get("sourceType")),
            _clean(item.get("scenarioType")),
            _clean(item.get("evalCaseId")),
        ),
    )


def _count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = _clean(row.get(field)) or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _normalize_tags(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    tags: list[str] = []
    for value in values:
        if isinstance(value, (list, tuple, set)):
            candidates = value
        else:
            candidates = [value]
        for candidate in candidates:
            text = _clean(candidate)
            if not text or text in seen:
                continue
            seen.add(text)
            tags.append(text)
    return tags


def _scenario_type_from_row(row: Mapping[str, Any]) -> str:
    for key in ("scenarioType", "scenario_type", "eval_bucket", "expected_family"):
        value = _clean(row.get(key))
        if value:
            return value
    return "generic"


def _expected_source_scope_from_success_criteria(success_criteria: Mapping[str, Any]) -> str:
    scoped = success_criteria.get("expectedScopeApplied")
    if scoped is True:
        return "scoped"
    if scoped is False:
        return "unscoped"
    expected_match_count = success_criteria.get("expectedMatchCount")
    if isinstance(expected_match_count, int) and expected_match_count > 1:
        return "multi_source"
    if expected_match_count == 1:
        return "single_source"
    return "unspecified"


def _expected_evidence_policy(
    *,
    explicit: str,
    expected_source_scope: str,
    should_abstain: bool,
) -> str:
    if explicit:
        return explicit
    if expected_source_scope == "scoped":
        return "scoped_evidence_required"
    if should_abstain:
        return "abstain_when_evidence_is_weak"
    return "default"


def normalize_eval_case(case: Mapping[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    lane = _clean(case.get("lane"))
    source_type = _clean(case.get("sourceType") or case.get("source") or "unknown")
    scenario_type = _clean(case.get("scenarioType") or case.get("scenario") or "generic")
    query = _clean(case.get("query"))
    expected_family = _clean(case.get("expectedFamily") or case.get("expected_family"))
    expected_answer_mode = _clean(case.get("expectedAnswerMode") or case.get("expected_answer_mode"))
    raw_success = case.get("successCriteria")
    success_criteria = dict(raw_success) if isinstance(raw_success, Mapping) else {}
    if "expectedTop1OrSet" not in success_criteria:
        success_criteria["expectedTop1OrSet"] = _clean(
            case.get("expectedTop1OrSet") or case.get("expected_top1_or_set")
        )
    if "allowedFallback" not in success_criteria:
        success_criteria["allowedFallback"] = _clean(case.get("allowedFallback") or case.get("allowed_fallback"))
    if "expectedMatchCount" not in success_criteria:
        success_criteria["expectedMatchCount"] = _parse_int(
            case.get("expectedMatchCount") or case.get("expected_match_count")
        )
    if "expectedScopeApplied" not in success_criteria:
        success_criteria["expectedScopeApplied"] = _parse_bool(
            case.get("expectedScopeApplied") or case.get("expected_scope_applied")
        )
    for key in ("expectedTop1OrSet", "allowedFallback"):
        success_criteria[key] = _clean(success_criteria.get(key))

    explicit_should_abstain = case.get("shouldAbstain")
    if isinstance(explicit_should_abstain, bool):
        should_abstain = explicit_should_abstain
    else:
        parsed_should_abstain = _parse_bool(explicit_should_abstain or case.get("should_abstain"))
        should_abstain = (
            parsed_should_abstain
            if parsed_should_abstain is not None
            else "abstain" in expected_answer_mode.casefold()
        )

    expected_source_scope = _clean(case.get("expectedSourceScope"))
    if not expected_source_scope:
        expected_source_scope = _expected_source_scope_from_success_criteria(success_criteria)

    expected_evidence_policy = _expected_evidence_policy(
        explicit=_clean(case.get("expectedEvidencePolicy")),
        expected_source_scope=expected_source_scope,
        should_abstain=should_abstain,
    )
    timestamp = _now_iso(now)
    created_at = _clean(case.get("createdAt")) or timestamp
    updated_at = _clean(case.get("updatedAt")) or timestamp
    provenance = dict(case.get("provenance") or {}) if isinstance(case.get("provenance"), Mapping) else {}
    tags = _normalize_tags(
        [
            case.get("tags") or [],
            lane,
            source_type,
            scenario_type,
            expected_family,
        ]
    )
    eval_case_id = _clean(case.get("evalCaseId")) or _deterministic_eval_case_id(
        lane=lane,
        source_type=source_type,
        scenario_type=scenario_type,
        query=query,
    )
    return {
        "schema": EVAL_CASE_SCHEMA,
        "evalCaseId": eval_case_id,
        "lane": lane,
        "sourceType": source_type,
        "scenarioType": scenario_type,
        "query": query,
        "expectedFamily": expected_family,
        "expectedAnswerMode": expected_answer_mode,
        "expectedSourceScope": expected_source_scope,
        "shouldAbstain": should_abstain,
        "expectedEvidencePolicy": expected_evidence_policy,
        "successCriteria": success_criteria,
        "tags": tags,
        "status": _clean(case.get("status")) or "active",
        "provenance": provenance,
        "createdAt": created_at,
        "updatedAt": updated_at,
    }


def create_eval_case(
    *,
    lane: str,
    source_type: str,
    scenario_type: str,
    query: str,
    expected_family: str = "",
    expected_answer_mode: str = "",
    expected_source_scope: str = "",
    should_abstain: bool = False,
    expected_evidence_policy: str = "",
    success_criteria: Mapping[str, Any] | None = None,
    tags: list[str] | None = None,
    status: str = "active",
    provenance: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    return normalize_eval_case(
        {
            "lane": lane,
            "sourceType": source_type,
            "scenarioType": scenario_type,
            "query": query,
            "expectedFamily": expected_family,
            "expectedAnswerMode": expected_answer_mode,
            "expectedSourceScope": expected_source_scope,
            "shouldAbstain": should_abstain,
            "expectedEvidencePolicy": expected_evidence_policy,
            "successCriteria": dict(success_criteria or {}),
            "tags": list(tags or []),
            "status": status,
            "provenance": dict(provenance or {}),
        },
        now=now,
    )


def build_eval_case_from_csv_row(
    row: Mapping[str, Any],
    *,
    lane: str,
    source_path: str | Path = "",
    row_number: int = 0,
    status: str = "active",
    now: datetime | None = None,
) -> dict[str, Any]:
    clean_row = {_clean(key): _clean(value) for key, value in row.items()}
    expected_match_count = _parse_int(clean_row.get("expected_match_count"))
    expected_scope_applied = _parse_bool(clean_row.get("expected_scope_applied"))
    return normalize_eval_case(
        {
            "lane": lane,
            "sourceType": clean_row.get("source") or "unknown",
            "scenarioType": _scenario_type_from_row(clean_row),
            "query": clean_row.get("query"),
            "expectedFamily": clean_row.get("expected_family"),
            "expectedAnswerMode": clean_row.get("expected_answer_mode"),
            "successCriteria": {
                "expectedTop1OrSet": clean_row.get("expected_top1_or_set", ""),
                "allowedFallback": clean_row.get("allowed_fallback", ""),
                "expectedMatchCount": expected_match_count,
                "expectedScopeApplied": expected_scope_applied,
            },
            "status": status,
            "provenance": {
                "sourceKind": "csv_row",
                "sourcePath": str(source_path) if source_path else "",
                "rowNumber": row_number,
                "rawRow": clean_row,
            },
        },
        now=now,
    )


def read_eval_case_registry(
    *,
    registry_path: str | Path = DEFAULT_EVAL_CASE_REGISTRY_PATH,
    repo_root: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[str], Path]:
    target_path = _resolve_path(registry_path, base=repo_root)
    rows, warnings = _read_jsonl(target_path)
    return _sort_cases(rows), warnings, target_path


def write_eval_case_registry(
    rows: list[Mapping[str, Any]],
    *,
    registry_path: str | Path = DEFAULT_EVAL_CASE_REGISTRY_PATH,
    repo_root: str | Path | None = None,
    now: datetime | None = None,
) -> Path:
    target_path = _resolve_path(registry_path, base=repo_root)
    normalized_rows = [normalize_eval_case(row, now=now) for row in rows]
    _write_jsonl(target_path, _sort_cases(normalized_rows))
    return target_path


def list_eval_cases(
    *,
    registry_path: str | Path = DEFAULT_EVAL_CASE_REGISTRY_PATH,
    repo_root: str | Path | None = None,
    lane: str = "",
    source_type: str = "",
    status: str = "",
    limit: int = 50,
) -> dict[str, Any]:
    rows, warnings, target_path = read_eval_case_registry(registry_path=registry_path, repo_root=repo_root)
    wanted_lane = _clean(lane)
    wanted_source_type = _clean(source_type)
    wanted_status = _clean(status)
    filtered = rows
    if wanted_lane:
        filtered = [row for row in filtered if _clean(row.get("lane")) == wanted_lane]
    if wanted_source_type:
        filtered = [row for row in filtered if _clean(row.get("sourceType")) == wanted_source_type]
    if wanted_status:
        filtered = [row for row in filtered if _clean(row.get("status")) == wanted_status]
    limited = filtered[: max(0, int(limit))]
    return {
        "schema": EVAL_CASE_REGISTRY_SCHEMA,
        "action": "list",
        "status": "ok" if target_path.exists() else "missing",
        "registryPath": str(target_path),
        "recordCount": len(filtered),
        "totalRecordCount": len(rows),
        "laneCounts": _count_by(rows, "lane"),
        "sourceTypeCounts": _count_by(rows, "sourceType"),
        "statusCounts": _count_by(rows, "status"),
        "items": limited,
        "filter": {
            "lane": wanted_lane,
            "sourceType": wanted_source_type,
            "status": wanted_status,
            "limit": max(0, int(limit)),
        },
        "warnings": warnings,
    }


def import_eval_cases(
    rows: list[Mapping[str, Any]],
    *,
    lane: str,
    registry_path: str | Path = DEFAULT_EVAL_CASE_REGISTRY_PATH,
    repo_root: str | Path | None = None,
    source_path: str | Path = "",
    status: str = "active",
    now: datetime | None = None,
) -> dict[str, Any]:
    target_path = _resolve_path(registry_path, base=repo_root)
    existing, warnings = _read_jsonl(target_path)
    timestamp = _now_iso(now)
    by_id: dict[str, dict[str, Any]] = {}
    for record in existing:
        normalized = normalize_eval_case(record, now=now)
        by_id[_clean(normalized.get("evalCaseId"))] = normalized

    imported_count = 0
    created_count = 0
    updated_count = 0
    touched_ids: list[str] = []
    imported_items: list[dict[str, Any]] = []
    for offset, row in enumerate(rows, start=1):
        if not _clean(row.get("query")):
            continue
        eval_case = build_eval_case_from_csv_row(
            row,
            lane=lane,
            source_path=source_path,
            row_number=offset + 1,
            status=status,
            now=now,
        )
        eval_case_id = eval_case["evalCaseId"]
        touched_ids.append(eval_case_id)
        imported_items.append(eval_case)
        imported_count += 1
        existing_case = by_id.get(eval_case_id)
        if existing_case is None:
            by_id[eval_case_id] = eval_case
            created_count += 1
            continue
        eval_case["createdAt"] = _clean(existing_case.get("createdAt")) or timestamp
        by_id[eval_case_id] = eval_case
        updated_count += 1

    records = _sort_cases(list(by_id.values()))
    _write_jsonl(target_path, records)
    return {
        "schema": EVAL_CASE_REGISTRY_SCHEMA,
        "action": "import",
        "status": "ok",
        "registryPath": str(target_path),
        "sourcePath": str(source_path) if source_path else "",
        "lane": lane,
        "importedAt": timestamp,
        "importedCount": imported_count,
        "createdCount": created_count,
        "updatedCount": updated_count,
        "recordCount": len(records),
        "touchedEvalCaseIds": touched_ids,
        "laneCounts": _count_by(records, "lane"),
        "sourceTypeCounts": _count_by(records, "sourceType"),
        "statusCounts": _count_by(records, "status"),
        "items": imported_items,
        "warnings": warnings,
    }


def import_eval_cases_from_csv(
    *,
    csv_path: str | Path,
    lane: str,
    registry_path: str | Path = DEFAULT_EVAL_CASE_REGISTRY_PATH,
    repo_root: str | Path | None = None,
    status: str = "active",
    now: datetime | None = None,
) -> dict[str, Any]:
    target_csv_path = _resolve_path(csv_path, base=repo_root)
    with target_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return import_eval_cases(
        rows,
        lane=lane,
        registry_path=registry_path,
        repo_root=repo_root,
        source_path=target_csv_path,
        status=status,
        now=now,
    )
