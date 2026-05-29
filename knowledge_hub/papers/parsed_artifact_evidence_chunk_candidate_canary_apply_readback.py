"""Apply-gated canary executor for parsed-artifact evidence chunk candidates.

The executor consumes the parsed-artifact evidence chunk candidate dry-run
report, selects a small canary set, and writes candidate JSONL records only
when explicitly invoked with ``apply=True`` and a ``papers_dir``. Candidate
records are not SourceSpan, StrictEvidence, citation-grade, runtime evidence,
or answer-visible payloads.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_apply_executor import (
    _contains_private_path,
    normalize_text,
    sanitized_report_ref,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_dry_run import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_DRY_RUN_SCHEMA_ID,
    READY_DECISION as CANDIDATE_DRY_RUN_READY_DECISION,
    ZERO_COUNTER_FIELDS,
    load_json,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_CANARY_APPLY_READBACK_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-candidate-canary-apply-readback.v1"
)
PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RECORD_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-candidate-record.v1"
)

EXECUTOR_STATUS_DRY_RUN_READY = "dry_run_ready_candidate_record"
EXECUTOR_STATUS_APPLIED = "applied_candidate_record"
EXECUTOR_STATUS_ALREADY_CORRECT = "already_correct_candidate_record"
EXECUTOR_STATUS_BLOCKED_NON_READY_INPUT = "blocked_non_ready_input_row"
EXECUTOR_STATUS_BLOCKED_SCHEMA_VIOLATION = "blocked_schema_violation"
EXECUTOR_STATUS_BLOCKED_READBACK_MISMATCH = "blocked_readback_mismatch"
EXECUTOR_STATUS_BLOCKED_IDEMPOTENCY_CONFLICT = "blocked_idempotency_conflict"

PENDING_APPLY_DECISION = "parsed_artifact_evidence_chunk_candidate_canary_apply_readback_pending_apply"
APPLIED_DECISION = "parsed_artifact_evidence_chunk_candidate_canary_apply_readback_ready"
BLOCKED_DECISION = "blocked"

NEXT_TRANCHE_PENDING = "parsed_artifact_evidence_chunk_candidate_canary_apply_readback_apply"
NEXT_TRANCHE_APPLIED = "parsed_artifact_evidence_chunk_candidate_full_apply_readiness_gate"

DEFAULT_CANARY_RECORD_LIMIT = 10
DEFAULT_PREFERRED_CANARY_PAPER_IDS = (
    "2005.11401",
    "1706.03762",
    "1512.03385",
    "1810.04805",
    "2005.14165",
    "alexnet-2012",
    "2010.11929",
    "1406.2661",
    "2404.16130",
    "2410.05779",
)
DEFAULT_STORE_REF_TEMPLATE = "papers_dir/structured_evidence_candidates/evidence_chunk/{paperId}.jsonl"
DEFAULT_RUN_MANIFEST_REF_TEMPLATE = "papers_dir/structured_evidence_candidates/evidence_chunk/runs/{runId}.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _safe_filename(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return text.strip("._") or "unknown"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _short_hash(value: str, *, length: int = 20) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _record_hash(record: dict[str, Any]) -> str:
    return _sha256_text(_canonical_json(record))


def _store_ref(paper_id: str) -> str:
    return DEFAULT_STORE_REF_TEMPLATE.format(paperId=_safe_filename(paper_id))


def _store_path(papers_dir: str | Path, paper_id: str) -> Path:
    return (
        Path(str(papers_dir)).expanduser()
        / "structured_evidence_candidates"
        / "evidence_chunk"
        / f"{_safe_filename(paper_id)}.jsonl"
    )


def _run_manifest_ref(run_id: str) -> str:
    return DEFAULT_RUN_MANIFEST_REF_TEMPLATE.format(runId=_safe_filename(run_id))


def _run_manifest_path(papers_dir: str | Path, run_id: str) -> Path:
    return (
        Path(str(papers_dir)).expanduser()
        / "structured_evidence_candidates"
        / "evidence_chunk"
        / "runs"
        / f"{_safe_filename(run_id)}.json"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _idempotency_key(row: dict[str, Any]) -> str:
    basis = {
        "paperId": normalize_text(row.get("paperId")),
        "artifactType": normalize_text(row.get("artifactType")),
        "sourceContentHash": normalize_text(row.get("sourceContentHash")),
        "spanLocator": normalize_text(row.get("spanLocator")),
        "snippetHash": normalize_text(row.get("snippetHash")),
    }
    return "parsed-artifact-evidence-chunk-candidate:" + _short_hash(_canonical_json(basis), length=24)


def _candidate_record_id(row: dict[str, Any], key: str) -> str:
    return ":".join(
        [
            "parsed-artifact-evidence-chunk-candidate",
            _safe_filename(normalize_text(row.get("paperId"))),
            _safe_filename(normalize_text(row.get("artifactType"))),
            key.rsplit(":", 1)[-1][:16],
        ]
    )


def _candidate_record(row: dict[str, Any], *, run_id: str, source_report_ref: str) -> dict[str, Any]:
    key = _idempotency_key(row)
    record = {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RECORD_SCHEMA_ID,
        "candidateRecordId": _candidate_record_id(row, key),
        "runId": run_id,
        "sourceDryRunReportRef": normalize_text(source_report_ref),
        "sourceCandidateRowId": normalize_text(row.get("candidateRowId")),
        "paperId": normalize_text(row.get("paperId")),
        "sourceType": "paper",
        "artifactType": normalize_text(row.get("artifactType")),
        "sourceRef": normalize_text(row.get("sourceRef")),
        "sourceContentHash": normalize_text(row.get("sourceContentHash")),
        "locator": dict(row.get("locator") or {}),
        "spanLocator": normalize_text(row.get("spanLocator")),
        "excerpt": str(row.get("excerpt") or ""),
        "snippetHash": normalize_text(row.get("snippetHash")),
        "sectionTitle": normalize_text(row.get("sectionTitle")),
        "sectionPath": [str(item) for item in list(row.get("sectionPath") or [])],
        "evidenceKind": "parsed_artifact_evidence_chunk_candidate",
        "idempotencyKey": key,
        "candidateOnly": True,
        "answerEvidenceCandidate": True,
        "answerabilityCandidate": True,
        "strictEvidence": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "answerVisible": False,
        "evidenceTier": "parsed_artifact_evidence_chunk_candidate_only",
        "strictBlockers": [
            "candidate_store_record_not_strict_evidence",
            "source_span_promotion_gate_not_run",
            "runtime_integration_not_allowed",
        ],
        "writePolicy": {
            "candidateStoreWrite": True,
            "sourceSpanCreated": False,
            "strictEvidenceCreated": False,
            "citationGradeEvidenceCreated": False,
            "runtimeEvidenceCreated": False,
            "parserRoutingChanged": False,
            "answerIntegrationChanged": False,
            "databaseMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "canonicalParsedArtifactsWritten": False,
        },
    }
    record["candidateRecordHash"] = _record_hash(record)
    return record


def _source_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers: list[str] = []
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_DRY_RUN_SCHEMA_ID:
        blockers.append("invalid_candidate_dry_run_schema")
    if report.get("status") != "ready":
        blockers.append("candidate_dry_run_not_ready")
    if report.get("decision") != CANDIDATE_DRY_RUN_READY_DECISION:
        blockers.append("candidate_dry_run_invalid_decision")
    if dict(report.get("gate") or {}).get("passed") is not True:
        blockers.append("candidate_dry_run_gate_not_passed")
    if _int(counts.get("selectedCandidateRows")) <= 0:
        blockers.append("candidate_dry_run_has_no_selected_rows")
    for field_name in ("blockedRows", "privatePathLeakRows", "schemaViolationCount", *ZERO_COUNTER_FIELDS):
        if _int(counts.get(field_name)) != 0:
            blockers.append(f"candidate_dry_run_has_{field_name}")
    if _contains_private_path(report):
        blockers.append("candidate_dry_run_has_private_path_leak")
    return sorted(set(blockers))


def _candidate_row_ready(row: dict[str, Any]) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if row.get("candidateOnly") is not True:
        blockers.append("candidate_only_false")
    if row.get("answerEvidenceCandidate") is not True:
        blockers.append("answer_evidence_candidate_false")
    if row.get("answerabilityCandidate") is not True:
        blockers.append("answerability_candidate_false")
    if row.get("strictEvidence") is not False:
        blockers.append("strict_evidence_not_false")
    if row.get("citationGrade") is not False:
        blockers.append("citation_grade_not_false")
    if row.get("runtimeEvidence") is not False:
        blockers.append("runtime_evidence_not_false")
    if row.get("answerVisible") is not False:
        blockers.append("answer_visible_not_false")
    for field_name in ("candidateRowId", "paperId", "artifactType", "sourceRef", "sourceContentHash", "spanLocator", "excerpt", "snippetHash"):
        if not normalize_text(row.get(field_name)):
            blockers.append(f"{field_name}_missing")
    if normalize_text(row.get("artifactType")) not in {"section", "paragraph"}:
        blockers.append("unsupported_canary_artifact_type")
    locator = dict(row.get("locator") or {})
    chars = dict(locator.get("chars") or {})
    if locator.get("kind") != "parsed_document_chars":
        blockers.append("unsupported_locator_kind")
    if _int(chars.get("end")) <= _int(chars.get("start")):
        blockers.append("invalid_char_locator")
    if _contains_private_path(row):
        blockers.append("candidate_row_has_private_path_leak")
    return not blockers, sorted(set(blockers))


def _select_canary_rows(
    rows: list[dict[str, Any]],
    *,
    preferred_paper_ids: tuple[str, ...],
    canary_record_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    remaining = [dict(row) for row in rows if isinstance(row, dict)]
    by_paper: dict[str, list[dict[str, Any]]] = {}
    for row in remaining:
        by_paper.setdefault(normalize_text(row.get("paperId")), []).append(row)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    missing_preferred: list[str] = []
    for paper_id in preferred_paper_ids:
        paper_rows = by_paper.get(paper_id) or []
        if not paper_rows:
            missing_preferred.append(paper_id)
            continue
        selected.append(paper_rows[0])
        selected_ids.add(normalize_text(paper_rows[0].get("candidateRowId")))
        if len(selected) >= canary_record_limit:
            break

    top_up_rows = 0
    if len(selected) < canary_record_limit:
        for row in remaining:
            row_id = normalize_text(row.get("candidateRowId"))
            if row_id in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row_id)
            top_up_rows += 1
            if len(selected) >= canary_record_limit:
                break

    return selected, {
        "preferredCanaryPaperRows": len(selected) - top_up_rows,
        "topUpCanaryRows": top_up_rows,
        "missingPreferredCanaryPaperIds": missing_preferred,
    }


def _write_jsonl_idempotent(
    path: Path,
    records: list[dict[str, Any]],
) -> tuple[int, int, int, list[str], dict[str, str]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = _read_jsonl(path)
    existing_by_key = {normalize_text(row.get("idempotencyKey")): row for row in existing_rows}
    incoming_by_key = {normalize_text(record.get("idempotencyKey")): record for record in records}

    conflicts: list[str] = []
    new_records: list[dict[str, Any]] = []
    already_correct_rows = 0
    path_by_record_id: dict[str, str] = {}
    for key, record in incoming_by_key.items():
        existing = existing_by_key.get(key)
        if existing is None:
            new_records.append(record)
        elif existing == record:
            already_correct_rows += 1
            path_by_record_id[normalize_text(record.get("candidateRecordId"))] = path.as_posix()
        else:
            conflicts.append(f"idempotency_conflict:{normalize_text(record.get('candidateRecordId'))}")

    if conflicts:
        return 0, already_correct_rows, already_correct_rows, sorted(set(conflicts)), path_by_record_id

    output = existing_rows + new_records
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in output),
        encoding="utf-8",
    )
    readback_by_key = {normalize_text(row.get("idempotencyKey")): row for row in _read_jsonl(path)}
    readback_rows = 0
    warnings: list[str] = []
    for record in records:
        stored = readback_by_key.get(normalize_text(record.get("idempotencyKey")))
        if stored == record:
            readback_rows += 1
            path_by_record_id[normalize_text(record.get("candidateRecordId"))] = path.as_posix()
        else:
            warnings.append(f"readback_mismatch:{normalize_text(record.get('candidateRecordId'))}")
    return len(new_records), already_correct_rows, readback_rows, sorted(set(warnings)), path_by_record_id


def _apply_records(
    records: list[dict[str, Any]],
    *,
    papers_dir: str | Path,
) -> tuple[int, int, int, list[str], dict[str, str]]:
    records_by_path: dict[Path, list[dict[str, Any]]] = {}
    for record in records:
        path = _store_path(papers_dir, normalize_text(record.get("paperId")))
        records_by_path.setdefault(path, []).append(record)

    applied_rows = 0
    already_correct_rows = 0
    readback_rows = 0
    warnings: list[str] = []
    path_by_record_id: dict[str, str] = {}
    for path, path_records in sorted(records_by_path.items(), key=lambda item: str(item[0])):
        applied, already_correct, readback, path_warnings, path_refs = _write_jsonl_idempotent(path, path_records)
        applied_rows += applied
        already_correct_rows += already_correct
        readback_rows += readback
        warnings.extend(path_warnings)
        path_by_record_id.update(path_refs)
    return applied_rows, already_correct_rows, readback_rows, sorted(set(warnings)), path_by_record_id


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in (
            "schema",
            "status",
            "generatedAt",
            "decision",
            "nextRecommendedTranche",
            "input",
            "counts",
            "gate",
            "policy",
            "warnings",
            "rows",
            "candidateRecords",
        )
        if key in report
    }


def build_parsed_artifact_evidence_chunk_candidate_canary_apply_readback(
    *,
    candidate_dry_run_report: dict[str, Any],
    source_candidate_dry_run_report_ref: str,
    papers_dir: str | Path | None = None,
    run_id: str | None = None,
    apply: bool = False,
    all_requested: bool = False,
    canary_record_limit: int = DEFAULT_CANARY_RECORD_LIMIT,
    preferred_paper_ids: tuple[str, ...] = DEFAULT_PREFERRED_CANARY_PAPER_IDS,
    generated_at: str | None = None,
) -> dict[str, Any]:
    run_id = normalize_text(run_id) or f"parsed-artifact-evidence-chunk-canary-{utc_now_iso()}"
    source_blockers = _source_blockers(candidate_dry_run_report)
    warnings: list[str] = []
    schema_violations: list[str] = []
    if source_blockers:
        schema_violations.extend(source_blockers)
    if apply and not papers_dir:
        warnings.append("apply_requires_papers_dir")
        schema_violations.append("apply_requires_papers_dir")
    if all_requested:
        warnings.append("all_apply_not_allowed_in_canary_tranche")
        schema_violations.append("all_apply_not_allowed_in_canary_tranche")

    source_rows = [
        dict(row)
        for row in list(candidate_dry_run_report.get("candidateRows") or [])
        if isinstance(row, dict)
    ]
    selected_rows: list[dict[str, Any]] = []
    selection = {
        "preferredCanaryPaperRows": 0,
        "topUpCanaryRows": 0,
        "missingPreferredCanaryPaperIds": list(preferred_paper_ids),
    }
    if not schema_violations:
        selected_rows, selection = _select_canary_rows(
            source_rows,
            preferred_paper_ids=preferred_paper_ids,
            canary_record_limit=max(1, int(canary_record_limit)),
        )
        if not selected_rows:
            warnings.append("selected_canary_rows_missing")
            schema_violations.append("selected_canary_rows_missing")

    candidate_records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, source_row in enumerate(selected_rows, start=1):
        ready, blockers = _candidate_row_ready(source_row)
        record: dict[str, Any] | None = None
        execution_status = EXECUTOR_STATUS_DRY_RUN_READY if ready else EXECUTOR_STATUS_BLOCKED_SCHEMA_VIOLATION
        if ready:
            record = _candidate_record(
                source_row,
                run_id=run_id,
                source_report_ref=source_candidate_dry_run_report_ref,
            )
            if _contains_private_path(record):
                blockers.append("candidate_record_has_private_path_leak")
                execution_status = EXECUTOR_STATUS_BLOCKED_SCHEMA_VIOLATION
                record = None
            else:
                candidate_records.append(record)
        rows.append(
            {
                "executorRowId": f"parsed-artifact-evidence-chunk-candidate-canary-apply-readback:{index:04d}",
                "sourceCandidateRowId": normalize_text(source_row.get("candidateRowId")),
                "candidateRecordId": normalize_text(record.get("candidateRecordId")) if record else "",
                "paperId": normalize_text(source_row.get("paperId")),
                "artifactType": normalize_text(source_row.get("artifactType")),
                "sourceRef": normalize_text(source_row.get("sourceRef")),
                "sourceContentHash": normalize_text(source_row.get("sourceContentHash")),
                "spanLocator": normalize_text(source_row.get("spanLocator")),
                "snippetHash": normalize_text(source_row.get("snippetHash")),
                "candidateStoreRef": _store_ref(normalize_text(source_row.get("paperId"))),
                "idempotencyKey": normalize_text(record.get("idempotencyKey")) if record else "",
                "candidateRecordHash": normalize_text(record.get("candidateRecordHash")) if record else "",
                "wouldWriteCandidateRecord": bool(record) and not apply,
                "appliedCandidateRecord": False,
                "alreadyCorrectCandidateRecord": False,
                "readbackValidated": False,
                "candidateOnly": True,
                "answerEvidenceCandidate": True,
                "answerabilityCandidate": True,
                "strictEvidence": False,
                "citationGrade": False,
                "runtimeEvidence": False,
                "answerVisible": False,
                "executionStatus": execution_status,
                "executionBlockers": sorted(set(blockers)),
            }
        )

    if not candidate_records and not schema_violations:
        warnings.append("candidate_records_missing")
        schema_violations.append("candidate_records_missing")

    applied_rows = 0
    already_correct_rows = 0
    readback_rows = 0
    manifest_write_rows = 0
    if apply and candidate_records and not schema_violations and papers_dir:
        applied_rows, already_correct_rows, readback_rows, apply_warnings, path_by_record_id = _apply_records(
            candidate_records,
            papers_dir=papers_dir,
        )
        warnings.extend(apply_warnings)
        if any(warning.startswith("idempotency_conflict:") for warning in apply_warnings):
            schema_violations.append("apply_idempotency_conflict")
        if readback_rows != len(candidate_records):
            schema_violations.append("apply_readback_incomplete")
        for row in rows:
            record_id = normalize_text(row.get("candidateRecordId"))
            if record_id in path_by_record_id and "apply_readback_incomplete" not in schema_violations:
                row["wouldWriteCandidateRecord"] = False
                row["appliedCandidateRecord"] = True
                row["readbackValidated"] = True
                row["alreadyCorrectCandidateRecord"] = False
                row["executionStatus"] = EXECUTOR_STATUS_APPLIED
        if already_correct_rows:
            already_keys = {
                normalize_text(record.get("candidateRecordId"))
                for record in candidate_records
                if normalize_text(record.get("candidateRecordId")) in path_by_record_id
            }
            for row in rows:
                if normalize_text(row.get("candidateRecordId")) in already_keys and applied_rows == 0:
                    row["alreadyCorrectCandidateRecord"] = True
                    row["executionStatus"] = EXECUTOR_STATUS_ALREADY_CORRECT
        if schema_violations:
            for row in rows:
                if row["executionStatus"] == EXECUTOR_STATUS_DRY_RUN_READY:
                    if "apply_idempotency_conflict" in schema_violations:
                        row["executionStatus"] = EXECUTOR_STATUS_BLOCKED_IDEMPOTENCY_CONFLICT
                    else:
                        row["executionStatus"] = EXECUTOR_STATUS_BLOCKED_READBACK_MISMATCH
                    row["executionBlockers"] = sorted(set([*row["executionBlockers"], *schema_violations]))

    private_path_leak_rows = 1 if _contains_private_path(rows) or _contains_private_path(candidate_records) else 0
    if private_path_leak_rows:
        schema_violations.append("private_path_leak")
    schema_violations = sorted(set(schema_violations))

    blocked_rows = sum(1 for row in rows if str(row.get("executionStatus") or "").startswith("blocked_"))
    counts = {
        "inputRows": len(source_rows),
        "selectedCanaryRows": len(selected_rows),
        "heldRows": max(0, len(source_rows) - len(selected_rows)),
        "preferredCanaryPaperRows": int(selection.get("preferredCanaryPaperRows") or 0),
        "topUpCanaryRows": int(selection.get("topUpCanaryRows") or 0),
        "missingPreferredCanaryPaperRows": len(selection.get("missingPreferredCanaryPaperIds") or []),
        "plannedApplyRows": len(candidate_records),
        "applyRequestedRows": len(candidate_records) if apply else 0,
        "dryRunCandidateRecordRows": 0 if apply else len(candidate_records),
        "appliedCandidateRecordRows": (applied_rows + already_correct_rows) if apply else 0,
        "alreadyCorrectRows": already_correct_rows if apply else 0,
        "candidateStoreWriteRows": applied_rows if apply else 0,
        "readbackRows": readback_rows if apply else 0,
        "readbackValidatedRows": readback_rows if apply else 0,
        "runManifestWriteRows": manifest_write_rows,
        "blockedRows": blocked_rows,
        "blockedNonReadyInputRows": sum(1 for row in rows if row.get("executionStatus") == EXECUTOR_STATUS_BLOCKED_NON_READY_INPUT),
        "blockedSchemaViolationRows": sum(1 for row in rows if row.get("executionStatus") == EXECUTOR_STATUS_BLOCKED_SCHEMA_VIOLATION),
        "blockedReadbackMismatchRows": sum(1 for row in rows if row.get("executionStatus") == EXECUTOR_STATUS_BLOCKED_READBACK_MISMATCH),
        "blockedIdempotencyConflictRows": sum(1 for row in rows if row.get("executionStatus") == EXECUTOR_STATUS_BLOCKED_IDEMPOTENCY_CONFLICT),
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(schema_violations),
        "sourceSpanCreatedRows": 0,
        "strictEvidenceRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "answerVisibleRows": 0,
        "answerGenerationRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "reindexOrReembedRows": 0,
        "parserExecutionRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
        "vaultScanRows": 0,
        "externalDownloadRows": 0,
        "byArtifactType": dict(Counter(row.get("artifactType") for row in rows)),
        "byExecutionStatus": dict(Counter(row.get("executionStatus") for row in rows)),
    }
    status = "blocked" if schema_violations or blocked_rows else ("applied" if apply else "ready")
    decision = BLOCKED_DECISION if status == "blocked" else (APPLIED_DECISION if apply else PENDING_APPLY_DECISION)
    report: dict[str, Any] = {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_CANARY_APPLY_READBACK_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": decision,
        "nextRecommendedTranche": NEXT_TRANCHE_APPLIED if status == "applied" else NEXT_TRANCHE_PENDING,
        "input": {
            "sourceCandidateDryRunReportRef": normalize_text(source_candidate_dry_run_report_ref),
            "papersDirRef": "papers_dir" if papers_dir else "",
            "runId": run_id,
            "apply": bool(apply),
            "allRequested": bool(all_requested),
            "canaryRecordLimit": max(1, int(canary_record_limit)),
            "preferredCanaryPaperIds": list(preferred_paper_ids),
            "missingPreferredCanaryPaperIds": list(selection.get("missingPreferredCanaryPaperIds") or []),
            "runManifestRef": _run_manifest_ref(run_id),
        },
        "sourceCandidateDryRun": {
            "schema": normalize_text(candidate_dry_run_report.get("schema")),
            "status": normalize_text(candidate_dry_run_report.get("status")),
            "decision": normalize_text(candidate_dry_run_report.get("decision")),
            "reportRef": normalize_text(source_candidate_dry_run_report_ref),
            "selectedCandidateRows": _int(dict(candidate_dry_run_report.get("counts") or {}).get("selectedCandidateRows")),
            "blockedRows": _int(dict(candidate_dry_run_report.get("counts") or {}).get("blockedRows")),
            "privatePathLeakRows": _int(dict(candidate_dry_run_report.get("counts") or {}).get("privatePathLeakRows")),
            "schemaViolationCount": _int(dict(candidate_dry_run_report.get("counts") or {}).get("schemaViolationCount")),
        },
        "policy": {
            "dryRunByDefault": True,
            "applyRequiredForCandidateStoreWrites": True,
            "applyMode": bool(apply),
            "candidateStoreWrite": bool(counts["candidateStoreWriteRows"]),
            "sourceSpanCreated": False,
            "strictEvidenceCreated": False,
            "citationGradeEvidenceCreated": False,
            "runtimeEvidenceCreated": False,
            "answerVisibleExposure": False,
            "answerGeneration": False,
            "parserRoutingChanged": False,
            "answerIntegrationChanged": False,
            "databaseMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "parserExecution": False,
            "canonicalParsedArtifactsWritten": False,
            "fullApplyAllowed": False,
        },
        "counts": counts,
        "gate": {
            "readyForApply": status == "ready",
            "applyMode": bool(apply),
            "candidateStoreWriteAllowed": bool(apply and papers_dir and not schema_violations),
            "canaryOnly": True,
            "fullApplyAllowed": False,
            "runtimeVisibilityAllowed": False,
            "evidencePromotionAllowed": False,
            "schemaViolations": schema_violations,
        },
        "rows": rows,
        "candidateRecords": candidate_records,
        "warnings": sorted(set(warnings)),
    }

    if apply and status == "applied" and papers_dir:
        manifest_path = _run_manifest_path(papers_dir, run_id)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        report["counts"]["runManifestWriteRows"] = 1
        manifest_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return report


def render_parsed_artifact_evidence_chunk_candidate_canary_apply_readback_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    by_status = dict(counts.get("byExecutionStatus") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Candidate Canary Apply/Readback",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- apply: `{dict(report.get('input') or {}).get('apply')}`",
        f"- inputRows: `{counts.get('inputRows')}`",
        f"- selectedCanaryRows: `{counts.get('selectedCanaryRows')}`",
        f"- heldRows: `{counts.get('heldRows')}`",
        f"- preferredCanaryPaperRows: `{counts.get('preferredCanaryPaperRows')}`",
        f"- topUpCanaryRows: `{counts.get('topUpCanaryRows')}`",
        f"- missingPreferredCanaryPaperRows: `{counts.get('missingPreferredCanaryPaperRows')}`",
        f"- plannedApplyRows: `{counts.get('plannedApplyRows')}`",
        f"- candidateStoreWriteRows: `{counts.get('candidateStoreWriteRows')}`",
        f"- readbackValidatedRows: `{counts.get('readbackValidatedRows')}`",
        f"- alreadyCorrectRows: `{counts.get('alreadyCorrectRows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Mutation Guarantees",
        "",
        f"- sourceSpanCreatedRows: `{counts.get('sourceSpanCreatedRows')}`",
        f"- strictEvidenceRows: `{counts.get('strictEvidenceRows')}`",
        f"- citationGradeRows: `{counts.get('citationGradeRows')}`",
        f"- runtimeEvidenceRows: `{counts.get('runtimeEvidenceRows')}`",
        f"- answerVisibleRows: `{counts.get('answerVisibleRows')}`",
        f"- databaseMutationRows: `{counts.get('databaseMutationRows')}`",
        f"- indexMutationRows: `{counts.get('indexMutationRows')}`",
        f"- vaultScanRows: `{counts.get('vaultScanRows')}`",
        f"- externalDownloadRows: `{counts.get('externalDownloadRows')}`",
        "",
        "## Execution Status",
        "",
    ]
    for status, count in sorted(by_status.items()):
        lines.append(f"- `{status}`: `{count}`")
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report.get("warnings", []):
            lines.append(f"- `{warning}`")
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_evidence_chunk_candidate_canary_apply_readback(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(render_parsed_artifact_evidence_chunk_candidate_canary_apply_readback_markdown(report), encoding="utf-8")
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "APPLIED_DECISION",
    "EXECUTOR_STATUS_ALREADY_CORRECT",
    "EXECUTOR_STATUS_APPLIED",
    "EXECUTOR_STATUS_BLOCKED_IDEMPOTENCY_CONFLICT",
    "EXECUTOR_STATUS_BLOCKED_READBACK_MISMATCH",
    "EXECUTOR_STATUS_BLOCKED_SCHEMA_VIOLATION",
    "EXECUTOR_STATUS_DRY_RUN_READY",
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_CANARY_APPLY_READBACK_SCHEMA_ID",
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RECORD_SCHEMA_ID",
    "PENDING_APPLY_DECISION",
    "build_parsed_artifact_evidence_chunk_candidate_canary_apply_readback",
    "load_json",
    "sanitized_report_ref",
    "write_parsed_artifact_evidence_chunk_candidate_canary_apply_readback",
]
