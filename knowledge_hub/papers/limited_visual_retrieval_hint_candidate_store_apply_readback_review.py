"""Readback review for applied visual retrieval-hint candidate-store records.

The review reads the local JSONL candidate store after an explicit apply and
checks that every expected candidate record is present exactly once, byte-level
equivalent under canonical JSON, and still policy-quarantined as a retrieval
hint only. It does not index, expose, promote, or query the records.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_apply_executor import (
    APPLIED_DECISION as APPLY_EXECUTOR_APPLIED_DECISION,
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_EXECUTOR_SCHEMA_ID,
    _contains_private_path,
    _idempotency_key_from_record,
    _record_hash,
    _record_policy_ok,
    normalize_text,
)


LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_READBACK_REVIEW_SCHEMA_ID = (
    "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-apply-readback-review.v1"
)

READBACK_STATUS_VALIDATED = "readback_validated_candidate_record"
READBACK_STATUS_BLOCKED_MISSING = "blocked_missing_store_record"
READBACK_STATUS_BLOCKED_DUPLICATE = "blocked_duplicate_store_record"
READBACK_STATUS_BLOCKED_HASH_MISMATCH = "blocked_hash_mismatch"
READBACK_STATUS_BLOCKED_POLICY_VIOLATION = "blocked_policy_violation"

READY_DECISION = "ready_for_limited_visual_retrieval_hint_candidate_store_labs_vector_index_dry_run"
BLOCKED_DECISION = "blocked"
NEXT_TRANCHE = "limited_visual_retrieval_hint_candidate_store_labs_vector_index_dry_run"

PLANNED_STORE_REF = "papers_dir/visual_retrieval_hints/visual_retrieval_hint_candidates.v1.jsonl"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sanitized_report_ref(path: Path, *, project_root: Path | None = None) -> str:
    resolved = path.expanduser()
    if project_root is not None:
        try:
            return resolved.resolve().relative_to(project_root.resolve()).as_posix()
        except Exception:
            pass
    return f"input_reports/{resolved.name}"


def _store_path(papers_dir: str | Path) -> Path:
    return Path(str(papers_dir)).expanduser() / "visual_retrieval_hints" / "visual_retrieval_hint_candidates.v1.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _source_apply_report_summary(report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = dict(report.get("counts") or {})
    return {
        "schema": normalize_text(report.get("schema")),
        "status": normalize_text(report.get("status")),
        "decision": normalize_text(report.get("decision")),
        "reportRef": normalize_text(report_ref),
        "plannedApplyRows": int(counts.get("plannedApplyRows") or 0),
        "appliedCandidateRecordRows": int(counts.get("appliedCandidateRecordRows") or 0),
        "candidateStoreWriteRows": int(counts.get("candidateStoreWriteRows") or 0),
        "readbackValidatedRows": int(counts.get("readbackValidatedRows") or 0),
        "blockedRows": int(counts.get("blockedRows") or 0),
        "privatePathLeakRows": int(counts.get("privatePathLeakRows") or 0),
        "schemaViolationCount": int(counts.get("schemaViolationCount") or 0),
    }


def _source_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers: list[str] = []
    if report.get("schema") != LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_EXECUTOR_SCHEMA_ID:
        blockers.append("invalid_apply_executor_schema")
    if report.get("status") != "applied":
        blockers.append("apply_executor_not_applied")
    if report.get("decision") != APPLY_EXECUTOR_APPLIED_DECISION:
        blockers.append("apply_executor_invalid_decision")
    for field_name in ("blockedRows", "privatePathLeakRows", "schemaViolationCount"):
        if int(counts.get(field_name) or 0) != 0:
            blockers.append(f"apply_executor_has_{field_name}")
    if int(counts.get("candidateStoreWriteRows") or 0) <= 0:
        blockers.append("apply_executor_has_no_candidate_store_writes")
    if int(counts.get("readbackValidatedRows") or 0) != int(counts.get("candidateStoreWriteRows") or 0):
        blockers.append("apply_executor_readback_count_mismatch")
    return blockers


def _scope(candidate_rows: int) -> dict[str, Any]:
    return {
        "writes": "report_only",
        "apiCalls": False,
        "modelCalls": False,
        "webModelCalls": False,
        "expectedCandidateRows": int(candidate_rows),
        "candidateStoreWriteRows": 0,
        "vectorIndexing": False,
        "indexEligibleRows": 0,
        "runtimeVisibleRows": 0,
        "strictEvidencePromotionRows": 0,
        "runtimeAnswerVisibleExposureRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "reindexOrReembedRows": 0,
        "vaultScanRows": 0,
        "externalDownloadRows": 0,
        "answerabilityGateBypassRows": 0,
        "operationalSearchIndexQueryRows": 0,
        "answerGenerationRows": 0,
    }


def _policy() -> dict[str, Any]:
    return {
        "readbackReviewOnly": True,
        "candidateStoreWrite": False,
        "sourceSpanCreated": False,
        "strictEvidenceCreated": False,
        "citationGradeEvidenceCreated": False,
        "runtimeEvidenceCreated": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "databaseMutation": False,
        "vaultScan": False,
        "reindexOrReembed": False,
        "allowedUse": "retrieval_hint_only",
        "strictEvidence": False,
        "citationGrade": False,
        "answerableWithoutTextEvidence": False,
        "runtimeVisible": False,
        "indexEligible": False,
    }


def review_limited_visual_retrieval_hint_candidate_store_apply_readback(
    *,
    apply_report: dict[str, Any],
    source_apply_report_ref: str,
    papers_dir: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    expected_records = [dict(row) for row in apply_report.get("candidateRecords") or [] if isinstance(row, dict)]
    store_rows = _read_jsonl(_store_path(papers_dir))
    store_rows_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for store_row in store_rows:
        store_rows_by_key[_idempotency_key_from_record(store_row)].append(store_row)

    schema_violations = _source_blockers(apply_report)
    rows: list[dict[str, Any]] = []
    for index, expected in enumerate(expected_records, start=1):
        key = _idempotency_key_from_record(expected)
        matches = store_rows_by_key.get(key, [])
        stored = matches[0] if matches else {}
        expected_hash = _record_hash(expected)
        stored_hash = _record_hash(stored) if stored else ""
        checks = {
            "storeRecordPresent": bool(matches),
            "singleStoreRecordForIdempotencyKey": len(matches) == 1,
            "recordHashMatches": bool(stored) and expected_hash == stored_hash,
            "recordBytesMatchCanonicalJson": bool(stored) and expected == stored,
            "hintCandidateIdMatches": normalize_text(expected.get("hintCandidateId"))
            == normalize_text(stored.get("hintCandidateId")),
            "sourceCandidateIdMatches": normalize_text(expected.get("sourceCandidateId"))
            == normalize_text(stored.get("sourceCandidateId")),
            "sourceContentHashMatches": normalize_text(expected.get("sourceContentHash"))
            == normalize_text(stored.get("sourceContentHash")),
            "pageMatches": int(expected.get("page") or 0) == int(stored.get("page") or 0),
            "bboxMatches": list(expected.get("bbox") or []) == list(stored.get("bbox") or []),
            "policyRetrievalHintOnly": bool(stored) and _record_policy_ok(stored),
            "recordNotPrivatePathLeaking": bool(stored) and not _contains_private_path(stored),
        }
        blockers = [name for name, passed in checks.items() if not passed]
        if not matches:
            readback_status = READBACK_STATUS_BLOCKED_MISSING
        elif len(matches) != 1:
            readback_status = READBACK_STATUS_BLOCKED_DUPLICATE
        elif not checks["policyRetrievalHintOnly"] or not checks["recordNotPrivatePathLeaking"]:
            readback_status = READBACK_STATUS_BLOCKED_POLICY_VIOLATION
        elif blockers:
            readback_status = READBACK_STATUS_BLOCKED_HASH_MISMATCH
        else:
            readback_status = READBACK_STATUS_VALIDATED
        rows.append(
            {
                "readbackReviewRowId": f"limited-visual-retrieval-hint-candidate-store-apply-readback-review:{index:04d}",
                "hintCandidateId": normalize_text(expected.get("hintCandidateId")),
                "sourceCandidateId": normalize_text(expected.get("sourceCandidateId")),
                "paperId": normalize_text(expected.get("paperId")),
                "paperRef": normalize_text(expected.get("paperRef")),
                "sourceContentHash": normalize_text(expected.get("sourceContentHash")),
                "page": int(expected.get("page") or 0),
                "bbox": list(expected.get("bbox") or []),
                "candidateType": normalize_text(expected.get("candidateType")),
                "idempotencyKey": key,
                "expectedRecordSha256": expected_hash,
                "storedRecordSha256": stored_hash,
                "matchingStoreRecordRows": len(matches),
                "readbackValidated": readback_status == READBACK_STATUS_VALIDATED,
                "indexEligible": False,
                "runtimeVisible": False,
                "strictEvidence": False,
                "citationGrade": False,
                "answerableWithoutTextEvidence": False,
                "readbackStatus": readback_status,
                "readbackBlockers": sorted(set(blockers)),
                "checks": checks,
            }
        )

    expected_keys = {_idempotency_key_from_record(record) for record in expected_records}
    matching_store_rows = sum(1 for row in store_rows if _idempotency_key_from_record(row) in expected_keys)
    private_path_leak_rows = 1 if _contains_private_path(rows) else 0
    if private_path_leak_rows:
        schema_violations.append("private_path_leak")
    schema_violations = sorted(set(schema_violations))
    by_status = Counter(row["readbackStatus"] for row in rows)
    blocked_rows = sum(1 for row in rows if row["readbackStatus"] != READBACK_STATUS_VALIDATED)
    counts = {
        "sourceAppliedRows": int(dict(apply_report.get("counts") or {}).get("appliedCandidateRecordRows") or 0),
        "expectedCandidateRows": len(expected_records),
        "storeRows": len(store_rows),
        "matchingStoreRows": matching_store_rows,
        "unmatchedStoreRows": len(store_rows) - matching_store_rows,
        "readbackValidatedRows": by_status.get(READBACK_STATUS_VALIDATED, 0),
        "missingStoreRows": by_status.get(READBACK_STATUS_BLOCKED_MISSING, 0),
        "duplicateStoreRows": by_status.get(READBACK_STATUS_BLOCKED_DUPLICATE, 0),
        "hashMismatchRows": by_status.get(READBACK_STATUS_BLOCKED_HASH_MISMATCH, 0),
        "policyViolationRows": by_status.get(READBACK_STATUS_BLOCKED_POLICY_VIOLATION, 0),
        "blockedRows": blocked_rows,
        "candidateStoreWriteRows": 0,
        "indexEligibleRows": 0,
        "runtimeVisibleRows": 0,
        "strictEvidenceRows": 0,
        "citationGradeRows": 0,
        "answerableWithoutTextEvidenceRows": 0,
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(schema_violations),
        "byCandidateType": dict(Counter(row["candidateType"] for row in rows)),
        "byReadbackStatus": dict(by_status),
    }
    status = "blocked" if schema_violations or blocked_rows else "ready"
    return {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_READBACK_REVIEW_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE,
        "sourceApplyExecutorReport": _source_apply_report_summary(apply_report, report_ref=source_apply_report_ref),
        "input": {
            "papersDirRef": "papers_dir",
            "candidateStoreRef": PLANNED_STORE_REF,
        },
        "scope": _scope(len(expected_records)),
        "policy": _policy(),
        "counts": counts,
        "gate": {
            "readyForLabsVectorIndexDryRun": status == "ready",
            "candidateStoreWriteAllowed": False,
            "vectorIndexingAllowed": False,
            "runtimeVisibilityAllowed": False,
            "evidencePromotionAllowed": False,
            "schemaViolations": schema_violations,
        },
        "rows": rows,
        "warnings": [],
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Limited Visual Retrieval Hint Candidate Store Apply Readback Review",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- expectedCandidateRows: `{counts.get('expectedCandidateRows')}`",
        f"- storeRows: `{counts.get('storeRows')}`",
        f"- matchingStoreRows: `{counts.get('matchingStoreRows')}`",
        f"- readbackValidatedRows: `{counts.get('readbackValidatedRows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Mutation Guarantees",
        "",
        f"- candidateStoreWriteRows: `{counts.get('candidateStoreWriteRows')}`",
        f"- vectorIndexingAllowed: `{dict(report.get('gate') or {}).get('vectorIndexingAllowed')}`",
        f"- runtimeVisibilityAllowed: `{dict(report.get('gate') or {}).get('runtimeVisibilityAllowed')}`",
        f"- evidencePromotionAllowed: `{dict(report.get('gate') or {}).get('evidencePromotionAllowed')}`",
        f"- indexEligibleRows: `{counts.get('indexEligibleRows')}`",
        f"- runtimeVisibleRows: `{counts.get('runtimeVisibleRows')}`",
        f"- strictEvidenceRows: `{counts.get('strictEvidenceRows')}`",
        f"- citationGradeRows: `{counts.get('citationGradeRows')}`",
        "",
        "## Readback Status",
        "",
    ]
    for status, count in sorted(dict(counts.get("byReadbackStatus") or {}).items()):
        lines.append(f"- `{status}`: `{count}`")
    return "\n".join(lines).rstrip() + "\n"


def write_limited_visual_retrieval_hint_candidate_store_apply_readback_review(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(render_markdown_report(report), encoding="utf-8")
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_READBACK_REVIEW_SCHEMA_ID",
    "READBACK_STATUS_VALIDATED",
    "READBACK_STATUS_BLOCKED_DUPLICATE",
    "READBACK_STATUS_BLOCKED_HASH_MISMATCH",
    "READBACK_STATUS_BLOCKED_MISSING",
    "READBACK_STATUS_BLOCKED_POLICY_VIOLATION",
    "load_json",
    "review_limited_visual_retrieval_hint_candidate_store_apply_readback",
    "sanitized_report_ref",
    "write_limited_visual_retrieval_hint_candidate_store_apply_readback_review",
]
