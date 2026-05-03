"""JSONL-backed Failure Bank v0 utilities."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowledge_hub.application.eval_cases import DEFAULT_EVAL_CASE_REGISTRY_PATH, read_eval_case_registry


FAILURE_BANK_SYNC_SCHEMA = "knowledge-hub.failure-bank.sync.result.v1"
FAILURE_BANK_LIST_SCHEMA = "knowledge-hub.failure-bank.list.result.v1"
FAILURE_BANK_LINK_SCHEMA = "knowledge-hub.failure-bank.link-eval-cases.result.v1"
DEFAULT_FAILURE_BANK_PATH = Path("~/.khub/eval/knowledgeos/failures/failure_bank.jsonl")


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


def _failure_triage_hint(bucket: str) -> tuple[str, str]:
    normalized = _clean(bucket) or "unknown"
    if normalized in {"groundedness_failure", "source_accuracy_failure"}:
        return (
            "Linked answer-loop failure suggests evidence grounding or source attribution drift.",
            "Review retrieval/evidence assembly and citation selection before changing answer generation.",
        )
    if normalized == "abstention_failure":
        return (
            "Linked answer-loop failure suggests abstention or fallback policy mismatch.",
            "Check should-abstain expectations, weak-evidence fallback, and answerability thresholds.",
        )
    if normalized in {"readability_failure", "usefulness_failure"}:
        return (
            "Linked answer-loop failure suggests final-answer presentation quality drift.",
            "Review the answer template only after evidence fit is confirmed.",
        )
    return (
        "Linked answer-loop failure requires manual root-cause review.",
        "Inspect the linked EvalCase, failure bucket, and latest answer-loop artifacts.",
    )


def _failure_id(card: dict[str, Any]) -> str:
    source = "|".join(
        [
            _clean(card.get("bucket")),
            _clean(card.get("packetRef")),
            _clean(card.get("query")),
            _clean(card.get("answerBackend")),
        ]
    )
    return "fb_" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def _latest_existing(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: (path.stat().st_mtime, str(path)))


def find_latest_answer_loop_failure_cards(*, runs_root: str | Path) -> Path | None:
    root = _resolve_path(runs_root)
    answer_root = root / "answer_loop"
    latest = answer_root / "latest" / "answer_loop_failure_cards.jsonl"
    if latest.exists():
        return latest
    return _latest_existing(list(answer_root.glob("**/answer_loop_failure_cards.jsonl")))


def sync_failure_bank_from_answer_loop(
    *,
    failure_cards_path: str | Path,
    bank_path: str | Path = DEFAULT_FAILURE_BANK_PATH,
    repo_root: str | Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    cards_path = _resolve_path(failure_cards_path, base=repo_root)
    target_path = _resolve_path(bank_path, base=repo_root)
    imported_at = _now_iso(now)
    cards, card_warnings = _read_jsonl(cards_path)
    existing, bank_warnings = _read_jsonl(target_path)
    warnings = card_warnings + bank_warnings

    by_id: dict[str, dict[str, Any]] = {}
    for record in existing:
        failure_id = _clean(record.get("failureId"))
        if failure_id:
            by_id[failure_id] = record

    imported = 0
    created = 0
    updated = 0
    touched_ids: list[str] = []
    for card in cards:
        failure_id = _failure_id(card)
        touched_ids.append(failure_id)
        imported += 1
        observation = {
            "sourceType": "answer_loop",
            "sourcePath": str(cards_path),
            "importedAt": imported_at,
        }
        if failure_id not in by_id:
            by_id[failure_id] = {
                "failureId": failure_id,
                "status": "open",
                "sourceType": "answer_loop",
                "bucket": _clean(card.get("bucket")),
                "packetRef": _clean(card.get("packetRef")),
                "query": _clean(card.get("query")),
                "answerBackend": _clean(card.get("answerBackend")),
                "reason": _clean(card.get("reason")),
                "predLabel": _clean(card.get("predLabel")),
                "predGroundedness": _clean(card.get("predGroundedness")),
                "predUsefulness": _clean(card.get("predUsefulness")),
                "predReadability": _clean(card.get("predReadability")),
                "predSourceAccuracy": _clean(card.get("predSourceAccuracy")),
                "predShouldAbstain": _clean(card.get("predShouldAbstain")),
                "rootCauseHypothesis": "",
                "recommendedFix": "",
                "linkedEvalCaseId": "",
                "firstSeenAt": imported_at,
                "lastSeenAt": imported_at,
                "observationCount": 1,
                "observations": [observation],
            }
            created += 1
            continue

        record = by_id[failure_id]
        record["lastSeenAt"] = imported_at
        record["observationCount"] = int(record.get("observationCount") or 0) + 1
        observations = list(record.get("observations") or [])
        observations.append(observation)
        record["observations"] = observations[-20:]
        for field in [
            "reason",
            "predLabel",
            "predGroundedness",
            "predUsefulness",
            "predReadability",
            "predSourceAccuracy",
            "predShouldAbstain",
        ]:
            value = _clean(card.get(field[0].lower() + field[1:])) or _clean(card.get(field))
            if value:
                record[field] = value
        updated += 1

    records = sorted(by_id.values(), key=lambda item: (_clean(item.get("status")), _clean(item.get("bucket")), _clean(item.get("failureId"))))
    _write_jsonl(target_path, records)

    bucket_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for record in records:
        bucket = _clean(record.get("bucket")) or "unknown"
        status = _clean(record.get("status")) or "unknown"
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "schema": FAILURE_BANK_SYNC_SCHEMA,
        "status": "ok",
        "bankPath": str(target_path),
        "sourcePath": str(cards_path),
        "importedAt": imported_at,
        "importedCount": imported,
        "createdCount": created,
        "updatedCount": updated,
        "recordCount": len(records),
        "touchedFailureIds": touched_ids,
        "bucketCounts": bucket_counts,
        "statusCounts": status_counts,
        "warnings": warnings,
    }


def list_failure_bank(
    *,
    bank_path: str | Path = DEFAULT_FAILURE_BANK_PATH,
    repo_root: str | Path | None = None,
    status: str = "",
    bucket: str = "",
    limit: int = 50,
) -> dict[str, Any]:
    target_path = _resolve_path(bank_path, base=repo_root)
    rows, warnings = _read_jsonl(target_path)
    wanted_status = _clean(status)
    wanted_bucket = _clean(bucket)
    if wanted_status:
        rows = [row for row in rows if _clean(row.get("status")) == wanted_status]
    if wanted_bucket:
        rows = [row for row in rows if _clean(row.get("bucket")) == wanted_bucket]
    rows = sorted(rows, key=lambda item: (_clean(item.get("status")), _clean(item.get("bucket")), _clean(item.get("failureId"))))
    limited = rows[: max(0, int(limit))]
    bucket_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for record in rows:
        bucket_name = _clean(record.get("bucket")) or "unknown"
        status_name = _clean(record.get("status")) or "unknown"
        bucket_counts[bucket_name] = bucket_counts.get(bucket_name, 0) + 1
        status_counts[status_name] = status_counts.get(status_name, 0) + 1
    return {
        "schema": FAILURE_BANK_LIST_SCHEMA,
        "status": "ok" if target_path.exists() else "missing",
        "bankPath": str(target_path),
        "recordCount": len(rows),
        "bucketCounts": bucket_counts,
        "statusCounts": status_counts,
        "items": limited,
        "warnings": warnings,
    }


def link_failure_bank_to_eval_cases(
    *,
    bank_path: str | Path = DEFAULT_FAILURE_BANK_PATH,
    registry_path: str | Path = DEFAULT_EVAL_CASE_REGISTRY_PATH,
    repo_root: str | Path | None = None,
    linked_status: str = "linked",
    now: datetime | None = None,
) -> dict[str, Any]:
    target_path = _resolve_path(bank_path, base=repo_root)
    rows, bank_warnings = _read_jsonl(target_path)
    eval_cases, registry_warnings, resolved_registry_path = read_eval_case_registry(
        registry_path=registry_path,
        repo_root=repo_root,
    )
    warnings = bank_warnings + [f"eval case registry: {warning}" for warning in registry_warnings]
    imported_at = _now_iso(now)

    cases_by_query: dict[str, list[dict[str, Any]]] = {}
    for eval_case in eval_cases:
        query_key = _clean_key(eval_case.get("query"))
        if not query_key:
            continue
        cases_by_query.setdefault(query_key, []).append(eval_case)

    linked_count = 0
    already_linked_count = 0
    unmatched_count = 0
    touched_failure_ids: list[str] = []
    unmatched_failure_ids: list[str] = []
    for record in rows:
        failure_id = _clean(record.get("failureId"))
        if _clean(record.get("linkedEvalCaseId")):
            already_linked_count += 1
            continue
        query_key = _clean_key(record.get("query"))
        matches = cases_by_query.get(query_key) or []
        if not matches:
            unmatched_count += 1
            if failure_id:
                unmatched_failure_ids.append(failure_id)
            continue
        chosen = sorted(
            matches,
            key=lambda item: (
                0 if _clean(item.get("lane")) == "user_answer_eval_queries_v1" else 1,
                0 if _clean(item.get("status")) == "active" else 1,
                _clean(item.get("evalCaseId")),
            ),
        )[0]
        record["linkedEvalCaseId"] = _clean(chosen.get("evalCaseId"))
        record["linkedEvalCaseLane"] = _clean(chosen.get("lane"))
        record["linkedEvalCaseSourceType"] = _clean(chosen.get("sourceType"))
        record["lastTriagedAt"] = imported_at
        if _clean(linked_status):
            record["status"] = _clean(linked_status)
        hypothesis, recommended_fix = _failure_triage_hint(_clean(record.get("bucket")))
        if not _clean(record.get("rootCauseHypothesis")):
            record["rootCauseHypothesis"] = hypothesis
        if not _clean(record.get("recommendedFix")):
            record["recommendedFix"] = recommended_fix
        linked_count += 1
        if failure_id:
            touched_failure_ids.append(failure_id)

    if rows:
        records = sorted(rows, key=lambda item: (_clean(item.get("status")), _clean(item.get("bucket")), _clean(item.get("failureId"))))
        _write_jsonl(target_path, records)
    else:
        records = []

    bucket_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for record in records:
        bucket_name = _clean(record.get("bucket")) or "unknown"
        status_name = _clean(record.get("status")) or "unknown"
        bucket_counts[bucket_name] = bucket_counts.get(bucket_name, 0) + 1
        status_counts[status_name] = status_counts.get(status_name, 0) + 1
    return {
        "schema": FAILURE_BANK_LINK_SCHEMA,
        "status": "ok" if target_path.exists() else "missing",
        "bankPath": str(target_path),
        "registryPath": str(resolved_registry_path),
        "linkedAt": imported_at,
        "linkedStatus": _clean(linked_status),
        "linkedCount": linked_count,
        "alreadyLinkedCount": already_linked_count,
        "unmatchedCount": unmatched_count,
        "recordCount": len(records),
        "touchedFailureIds": touched_failure_ids,
        "unmatchedFailureIds": unmatched_failure_ids,
        "bucketCounts": bucket_counts,
        "statusCounts": status_counts,
        "warnings": warnings,
    }
