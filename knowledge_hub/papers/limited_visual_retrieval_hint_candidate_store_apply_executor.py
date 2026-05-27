"""Apply-gated executor for visual retrieval-hint candidate-store records.

The executor consumes the allowlist apply-executor dry-run report and writes
candidate JSONL records only when explicitly invoked with ``apply=True`` and a
``papers_dir``. Candidate records remain retrieval hints only: they are not
indexed, runtime-visible, strict evidence, or citation-grade evidence.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_allowlist_apply_executor_dry_run import (
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_ALLOWLIST_APPLY_EXECUTOR_DRY_RUN_SCHEMA_ID,
    READY_DECISION as APPLY_EXECUTOR_DRY_RUN_READY_DECISION,
)
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_design import (
    PLANNED_STORE_REF,
    VISUAL_RETRIEVAL_HINT_CANDIDATE_ROW_SCHEMA_ID,
)


LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_EXECUTOR_SCHEMA_ID = (
    "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-apply-executor.v1"
)

EXECUTOR_STATUS_DRY_RUN_READY = "dry_run_ready_candidate_record"
EXECUTOR_STATUS_APPLIED = "applied_candidate_record"
EXECUTOR_STATUS_BLOCKED_NON_READY_INPUT = "blocked_non_ready_input_row"
EXECUTOR_STATUS_BLOCKED_SCHEMA_VIOLATION = "blocked_schema_violation"
EXECUTOR_STATUS_BLOCKED_READBACK_MISMATCH = "blocked_readback_mismatch"

READY_DECISION = "ready_for_limited_visual_retrieval_hint_candidate_store_apply"
APPLIED_DECISION = "applied_limited_visual_retrieval_hint_candidate_store_apply"
BLOCKED_DECISION = "blocked"

NEXT_TRANCHE_DRY_RUN = "limited_visual_retrieval_hint_candidate_store_apply_executor_apply_review"
NEXT_TRANCHE_APPLIED = "limited_visual_retrieval_hint_candidate_store_apply_readback_review"

PRIVATE_PATH_TOKENS = (
    "/" + "Users" + "/",
    "/" + "Volumes" + "/",
    "Mobile " + "Documents",
    "i" + "Cloud",
)
PRIVATE_PATH_RE = re.compile("|".join(re.escape(token) for token in PRIVATE_PATH_TOKENS), re.IGNORECASE)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sanitized_report_ref(path: Path, *, project_root: Path | None = None) -> str:
    resolved = path.expanduser()
    if project_root is not None:
        try:
            rel = resolved.resolve().relative_to(project_root.resolve())
            return rel.as_posix()
        except Exception:
            pass
    return f"input_reports/{resolved.name}"


def _safe_filename(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return text.strip("._") or "unknown"


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _short_hash(value: str, *, length: int = 20) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _record_hash(record: dict[str, Any]) -> str:
    return _sha256_text(_canonical_json(record))


def _contains_private_path(value: Any) -> bool:
    return bool(PRIVATE_PATH_RE.search(json.dumps(value, ensure_ascii=False, sort_keys=True)))


def _store_path(papers_dir: str | Path) -> Path:
    return Path(str(papers_dir)).expanduser() / "visual_retrieval_hints" / "visual_retrieval_hint_candidates.v1.jsonl"


def _run_manifest_path(papers_dir: str | Path, run_id: str) -> Path:
    return Path(str(papers_dir)).expanduser() / "visual_retrieval_hints" / "runs" / f"{_safe_filename(run_id)}.json"


def _run_manifest_ref(run_id: str) -> str:
    return f"papers_dir/visual_retrieval_hints/runs/{_safe_filename(run_id)}.json"


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


def _idempotency_key_from_record(record: dict[str, Any]) -> str:
    basis = "|".join(
        [
            normalize_text(record.get("hintCandidateId")),
            normalize_text(record.get("sourceCandidateId")),
            normalize_text(record.get("sourceContentHash")),
            str(record.get("page") or ""),
            json.dumps(record.get("bbox") or [], ensure_ascii=True, sort_keys=True),
        ]
    )
    return "visual-retrieval-hint-idempotency:" + _short_hash(basis, length=24)


def _record_policy(record: dict[str, Any]) -> dict[str, Any]:
    return dict(record.get("policy") or {})


def _record_policy_ok(record: dict[str, Any]) -> bool:
    policy = _record_policy(record)
    return (
        policy.get("allowedUse") == "retrieval_hint_only"
        and policy.get("strictEvidence") is False
        and policy.get("citationGrade") is False
        and policy.get("answerableWithoutTextEvidence") is False
        and policy.get("runtimeVisible") is False
        and policy.get("indexEligible") is False
        and policy.get("answerabilityGateBypassAllowed") is False
    )


def _source_report_summary(report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = dict(report.get("counts") or {})
    return {
        "schema": normalize_text(report.get("schema")),
        "status": normalize_text(report.get("status")),
        "decision": normalize_text(report.get("decision")),
        "reportRef": normalize_text(report_ref),
        "executorDryRunRows": int(counts.get("executorDryRunRows") or 0),
        "plannedWriteRows": int(counts.get("plannedWriteRows") or 0),
        "candidateStoreWriteRows": int(counts.get("candidateStoreWriteRows") or 0),
        "candidateStoreApplyRows": int(counts.get("candidateStoreApplyRows") or 0),
        "blockedRows": int(counts.get("blockedRows") or 0),
        "excludedHoldoutRows": int(counts.get("excludedHoldoutRows") or 0),
        "privatePathLeakRows": int(counts.get("privatePathLeakRows") or 0),
        "schemaViolationCount": int(counts.get("schemaViolationCount") or 0),
    }


def _scope(*, apply: bool, candidate_rows: int) -> dict[str, Any]:
    return {
        "writes": "candidate_store_jsonl" if apply else "report_only",
        "apiCalls": False,
        "modelCalls": False,
        "webModelCalls": False,
        "plannedApplyRows": int(candidate_rows),
        "candidateStoreWriteRows": 0,
        "candidateStoreApplyRows": 0,
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
        "cropWriteRows": 0,
        "pageImageWriteRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
    }


def _policy(*, apply: bool) -> dict[str, Any]:
    return {
        "dryRunByDefault": True,
        "applyRequiredForCandidateStoreWrites": True,
        "applyMode": bool(apply),
        "plannedStoreRef": PLANNED_STORE_REF,
        "runManifestWriteTarget": "papers_dir/visual_retrieval_hints/runs/{runId}.json",
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
        "canonicalParsedArtifactsWritten": False,
        "allowedUse": "retrieval_hint_only",
        "strictEvidence": False,
        "citationGrade": False,
        "answerableWithoutTextEvidence": False,
        "runtimeVisible": False,
        "indexEligible": False,
    }


def _source_blockers(report: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    counts = dict(report.get("counts") or {})
    if report.get("schema") != LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_ALLOWLIST_APPLY_EXECUTOR_DRY_RUN_SCHEMA_ID:
        blockers.append("invalid_apply_executor_dry_run_schema")
    if report.get("status") != "ready":
        blockers.append("apply_executor_dry_run_not_ready")
    if report.get("decision") != APPLY_EXECUTOR_DRY_RUN_READY_DECISION:
        blockers.append("apply_executor_dry_run_invalid_decision")
    for field_name in (
        "candidateStoreWriteRows",
        "candidateStoreApplyRows",
        "blockedRows",
        "privatePathLeakRows",
        "schemaViolationCount",
    ):
        if int(counts.get(field_name) or 0) != 0:
            blockers.append(f"apply_executor_dry_run_has_{field_name}")
    return blockers


def _record_checks(row: dict[str, Any], record: dict[str, Any]) -> dict[str, bool]:
    result = dict(row.get("executorDryRunResult") or {})
    expected_hash = normalize_text(row.get("plannedJsonlRecordSha256"))
    computed_hash = _record_hash(record) if record else ""
    return {
        "sourceRowReadyForSeparateApply": result.get("wouldWriteOnSeparateExplicitApply") is True,
        "sourceRowDidNotWriteStore": result.get("candidateStoreWrite") is False,
        "sourceRowActualStoreWriteFalse": result.get("actualStoreWrite") is False,
        "sourceRowJsonlSerializable": result.get("jsonlSerializable") is True,
        "sourceRowPolicyCompliant": result.get("policyCompliant") is True,
        "sourceRowNoBlockerReason": not normalize_text(row.get("blockerReason")),
        "recordPresent": bool(record),
        "recordSchemaMatches": normalize_text(record.get("schema")) == VISUAL_RETRIEVAL_HINT_CANDIDATE_ROW_SCHEMA_ID,
        "hintCandidateIdMatches": normalize_text(record.get("hintCandidateId")) == normalize_text(row.get("hintCandidateId")),
        "sourceCandidateIdMatches": normalize_text(record.get("sourceCandidateId")) == normalize_text(row.get("sourceCandidateId")),
        "sourceContentHashMatches": normalize_text(record.get("sourceContentHash")) == normalize_text(row.get("sourceContentHash")),
        "pageMatches": int(record.get("page") or 0) == int(row.get("page") or 0),
        "bboxMatches": list(record.get("bbox") or []) == list(row.get("bbox") or []),
        "candidateTypeMatches": normalize_text(record.get("candidateType")) == normalize_text(row.get("candidateType")),
        "idempotencyKeyMatches": _idempotency_key_from_record(record) == normalize_text(row.get("idempotencyKey")),
        "plannedJsonlRecordHashMatches": computed_hash == expected_hash,
        "recordPolicyRetrievalHintOnly": _record_policy_ok(record),
        "recordNotPrivatePathLeaking": not _contains_private_path(record),
    }


def _candidate_record_from_row(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str, list[str], dict[str, bool]]:
    record = dict(row.get("plannedJsonlRecordPreview") or {})
    checks = _record_checks(row, record)
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        non_ready_blockers = {
            "sourceRowReadyForSeparateApply",
            "sourceRowNoBlockerReason",
        }
        if any(blocker in non_ready_blockers for blocker in blockers):
            return None, EXECUTOR_STATUS_BLOCKED_NON_READY_INPUT, blockers, checks
        return None, EXECUTOR_STATUS_BLOCKED_SCHEMA_VIOLATION, blockers, checks
    return record, EXECUTOR_STATUS_DRY_RUN_READY, [], checks


def _write_jsonl_idempotent(path: Path, records: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming_by_key = {_idempotency_key_from_record(record): record for record in records}
    retained: list[dict[str, Any]] = []
    for existing in _read_jsonl(path):
        key = _idempotency_key_from_record(existing)
        if key and key in incoming_by_key:
            continue
        retained.append(existing)
    output = retained + list(incoming_by_key.values())
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in output),
        encoding="utf-8",
    )
    return len(incoming_by_key)


def _apply_records(
    records: list[dict[str, Any]],
    *,
    papers_dir: str | Path,
) -> tuple[int, int, list[str]]:
    path = _store_path(papers_dir)
    applied_rows = _write_jsonl_idempotent(path, records)
    readback_by_key = {_idempotency_key_from_record(row): row for row in _read_jsonl(path)}
    readback_rows = 0
    warnings: list[str] = []
    for record in records:
        key = _idempotency_key_from_record(record)
        stored = readback_by_key.get(key)
        if stored == record and _record_hash(stored) == _record_hash(record):
            readback_rows += 1
        else:
            warnings.append(f"readback_mismatch:{normalize_text(record.get('hintCandidateId'))}")
    return applied_rows, readback_rows, sorted(set(warnings))


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


def execute_limited_visual_retrieval_hint_candidate_store_apply_executor(
    *,
    apply_executor_dry_run_report: dict[str, Any],
    source_apply_executor_dry_run_report_ref: str,
    papers_dir: str | Path | None = None,
    run_id: str | None = None,
    apply: bool = False,
    generated_at: str | None = None,
) -> dict[str, Any]:
    run_id = normalize_text(run_id) or f"visual-retrieval-hint-candidate-store-apply-{utc_now_iso()}"
    source_blockers = _source_blockers(apply_executor_dry_run_report)
    warnings: list[str] = []
    schema_violations: list[str] = []
    if apply and not papers_dir:
        warnings.append("apply_requires_papers_dir")
        schema_violations.append("apply_requires_papers_dir")

    input_rows = [
        dict(row)
        for row in list(apply_executor_dry_run_report.get("executorDryRunRowsDetail") or [])
        if isinstance(row, dict)
    ]
    candidate_records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, input_row in enumerate(input_rows, start=1):
        record, execution_status, blockers, checks = _candidate_record_from_row(input_row)
        candidate_store_ref = PLANNED_STORE_REF if record else ""
        if record:
            candidate_records.append(record)

        rows.append(
            {
                "executorRowId": f"limited-visual-retrieval-hint-candidate-store-apply-executor:{index:04d}",
                "sourceExecutorDryRunRowId": normalize_text(input_row.get("executorDryRunRowId")),
                "sourceReviewRowId": normalize_text(input_row.get("sourceReviewRowId")),
                "sourceDryRunRowId": normalize_text(input_row.get("sourceDryRunRowId")),
                "hintCandidateId": normalize_text(input_row.get("hintCandidateId")),
                "sourceCandidateId": normalize_text(input_row.get("sourceCandidateId")),
                "paperId": normalize_text(input_row.get("paperId")),
                "paperRef": normalize_text(input_row.get("paperRef")),
                "sourceContentHash": normalize_text(input_row.get("sourceContentHash")),
                "page": int(input_row.get("page") or 0),
                "bbox": list(input_row.get("bbox") or []),
                "candidateType": normalize_text(input_row.get("candidateType")),
                "idempotencyKey": normalize_text(input_row.get("idempotencyKey")),
                "plannedJsonlRecordSha256": normalize_text(input_row.get("plannedJsonlRecordSha256")),
                "candidateRecordSha256": _record_hash(record) if record else "",
                "candidateStoreRef": candidate_store_ref,
                "wouldWriteCandidateRecord": bool(record) and not apply,
                "appliedCandidateRecord": False,
                "readbackValidated": False,
                "indexEligible": False,
                "runtimeVisible": False,
                "strictEvidence": False,
                "citationGrade": False,
                "answerableWithoutTextEvidence": False,
                "executionStatus": execution_status,
                "executionBlockers": sorted(set(blockers)),
                "checks": checks,
            }
        )

    if source_blockers:
        schema_violations.extend(source_blockers)
    if not input_rows:
        warnings.append("apply_executor_dry_run_rows_missing")
        schema_violations.append("apply_executor_dry_run_rows_missing")
    if not candidate_records:
        warnings.append("candidate_records_missing")
        schema_violations.append("candidate_records_missing")

    applied_rows = 0
    readback_rows = 0
    manifest_write_rows = 0
    if apply and candidate_records and not schema_violations and papers_dir:
        applied_rows, readback_rows, readback_warnings = _apply_records(candidate_records, papers_dir=papers_dir)
        warnings.extend(readback_warnings)
        if readback_rows != len(candidate_records):
            schema_violations.append("apply_readback_incomplete")
        for row in rows:
            if row["executionStatus"] == EXECUTOR_STATUS_DRY_RUN_READY:
                matched = not schema_violations and row["hintCandidateId"] not in " ".join(readback_warnings)
                row["wouldWriteCandidateRecord"] = False
                row["appliedCandidateRecord"] = matched
                row["readbackValidated"] = matched
                row["executionStatus"] = EXECUTOR_STATUS_APPLIED if matched else EXECUTOR_STATUS_BLOCKED_READBACK_MISMATCH
                if not matched:
                    row["executionBlockers"] = ["readback_mismatch"]

    private_path_leak_rows = 1 if _contains_private_path(rows) or _contains_private_path(candidate_records) else 0
    if private_path_leak_rows:
        schema_violations.append("private_path_leak")
    schema_violations = sorted(set(schema_violations))
    blocked_schema_rows = sum(1 for row in rows if row["executionStatus"] == EXECUTOR_STATUS_BLOCKED_SCHEMA_VIOLATION)
    blocked_readback_rows = sum(1 for row in rows if row["executionStatus"] == EXECUTOR_STATUS_BLOCKED_READBACK_MISMATCH)
    blocked_rows = blocked_schema_rows + blocked_readback_rows
    counts = {
        "inputRows": len(input_rows),
        "candidateInputRows": len(candidate_records),
        "heldInputRows": len(input_rows) - len(candidate_records),
        "plannedApplyRows": len(candidate_records),
        "dryRunCandidateRecordRows": 0 if apply else len(candidate_records),
        "appliedCandidateRecordRows": applied_rows if apply else 0,
        "candidateStoreWriteRows": applied_rows if apply else 0,
        "readbackValidatedRows": readback_rows if apply else 0,
        "runManifestWriteRows": manifest_write_rows,
        "blockedRows": blocked_rows,
        "blockedNonReadyInputRows": sum(1 for row in rows if row["executionStatus"] == EXECUTOR_STATUS_BLOCKED_NON_READY_INPUT),
        "blockedSchemaViolationRows": blocked_schema_rows,
        "blockedReadbackMismatchRows": blocked_readback_rows,
        "indexEligibleRows": 0,
        "runtimeVisibleRows": 0,
        "strictEvidenceRows": 0,
        "citationGradeRows": 0,
        "answerableWithoutTextEvidenceRows": 0,
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(schema_violations),
        "byCandidateType": dict(Counter(row["candidateType"] for row in rows)),
        "byExecutionStatus": dict(Counter(row["executionStatus"] for row in rows)),
    }
    status = "blocked" if schema_violations or blocked_rows else ("applied" if apply else "ready")
    decision = BLOCKED_DECISION if status == "blocked" else (APPLIED_DECISION if apply else READY_DECISION)
    report: dict[str, Any] = {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_EXECUTOR_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": decision,
        "nextRecommendedTranche": NEXT_TRANCHE_APPLIED if apply and status != "blocked" else NEXT_TRANCHE_DRY_RUN,
        "sourceApplyExecutorDryRunReport": _source_report_summary(
            apply_executor_dry_run_report,
            report_ref=source_apply_executor_dry_run_report_ref,
        ),
        "input": {
            "apply": bool(apply),
            "papersDirRef": "papers_dir" if papers_dir else "",
            "runId": run_id,
            "candidateStoreRef": PLANNED_STORE_REF,
            "runManifestRef": _run_manifest_ref(run_id),
        },
        "scope": _scope(apply=apply, candidate_rows=len(candidate_records)),
        "policy": _policy(apply=apply),
        "counts": counts,
        "gate": {
            "readyForApply": status == "ready",
            "applyMode": bool(apply),
            "candidateStoreWriteAllowed": bool(apply and papers_dir and not schema_violations),
            "vectorIndexingAllowed": False,
            "runtimeVisibilityAllowed": False,
            "evidencePromotionAllowed": False,
            "schemaViolations": schema_violations,
        },
        "rows": rows,
        "candidateRecords": candidate_records,
        "warnings": sorted(set(warnings)),
    }
    report["scope"]["candidateStoreWriteRows"] = counts["candidateStoreWriteRows"]
    report["scope"]["candidateStoreApplyRows"] = counts["appliedCandidateRecordRows"]
    report["policy"]["candidateStoreWrite"] = bool(counts["candidateStoreWriteRows"])

    if apply and status != "blocked" and papers_dir:
        manifest_path = _run_manifest_path(papers_dir, run_id)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        report["counts"]["runManifestWriteRows"] = 1
        manifest_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    by_status = dict(counts.get("byExecutionStatus") or {})
    lines = [
        "# Limited Visual Retrieval Hint Candidate Store Apply Executor",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- apply: `{dict(report.get('input') or {}).get('apply')}`",
        f"- plannedApplyRows: `{counts.get('plannedApplyRows')}`",
        f"- appliedCandidateRecordRows: `{counts.get('appliedCandidateRecordRows')}`",
        f"- candidateStoreWriteRows: `{counts.get('candidateStoreWriteRows')}`",
        f"- readbackValidatedRows: `{counts.get('readbackValidatedRows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Mutation Guarantees",
        "",
        f"- candidateStoreRef: `{dict(report.get('input') or {}).get('candidateStoreRef')}`",
        f"- vectorIndexingAllowed: `{dict(report.get('gate') or {}).get('vectorIndexingAllowed')}`",
        f"- runtimeVisibilityAllowed: `{dict(report.get('gate') or {}).get('runtimeVisibilityAllowed')}`",
        f"- evidencePromotionAllowed: `{dict(report.get('gate') or {}).get('evidencePromotionAllowed')}`",
        f"- indexEligibleRows: `{counts.get('indexEligibleRows')}`",
        f"- runtimeVisibleRows: `{counts.get('runtimeVisibleRows')}`",
        f"- strictEvidenceRows: `{counts.get('strictEvidenceRows')}`",
        f"- citationGradeRows: `{counts.get('citationGradeRows')}`",
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


def write_limited_visual_retrieval_hint_candidate_store_apply_executor(
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
    "EXECUTOR_STATUS_APPLIED",
    "EXECUTOR_STATUS_BLOCKED_NON_READY_INPUT",
    "EXECUTOR_STATUS_BLOCKED_READBACK_MISMATCH",
    "EXECUTOR_STATUS_BLOCKED_SCHEMA_VIOLATION",
    "EXECUTOR_STATUS_DRY_RUN_READY",
    "LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_EXECUTOR_SCHEMA_ID",
    "execute_limited_visual_retrieval_hint_candidate_store_apply_executor",
    "load_json",
    "sanitized_report_ref",
    "write_limited_visual_retrieval_hint_candidate_store_apply_executor",
]
