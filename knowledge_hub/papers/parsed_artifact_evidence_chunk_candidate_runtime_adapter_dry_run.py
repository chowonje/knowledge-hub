"""Dry-run the parsed-artifact evidence chunk runtime adapter contract.

This report simulates the future opt-in adapter without touching runtime code.
It joins the adapter design rows to the applied candidate-store record payloads
and verifies that a resolved-paper query would select bounded, provenance-safe
EvidencePacket item previews.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from knowledge_hub.ai.answer_contracts import parse_span_offsets
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_apply_executor import (
    _contains_private_path,
    normalize_text,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_canary_apply_readback import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RECORD_SCHEMA_ID,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_full_apply_executor_apply_readback import (
    APPLIED_DECISION as FULL_APPLY_APPLIED_DECISION,
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_EXECUTOR_APPLY_READBACK_SCHEMA_ID,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_full_apply_readback_review import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_READBACK_REVIEW_SCHEMA_ID,
    READY_DECISION as FULL_APPLY_READBACK_REVIEW_READY_DECISION,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_runtime_adapter_design import (
    ADAPTER_ID,
    ADAPTER_STATUS_READY_CANDIDATE_ONLY,
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_ADAPTER_DESIGN_SCHEMA_ID,
    READY_DECISION as ADAPTER_DESIGN_READY_DECISION,
    load_json,
    sanitized_report_ref,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_ADAPTER_DRY_RUN_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-candidate-runtime-adapter-dry-run.v1"
)

READY_DECISION = "parsed_artifact_evidence_chunk_candidate_runtime_adapter_dry_run_ready"
BLOCKED_DECISION = "blocked"
NEXT_TRANCHE = "parsed_artifact_evidence_chunk_candidate_runtime_adapter_implementation_opt_in"

DRY_RUN_STATUS_READY = "runtime_adapter_dry_run_ready_candidate_only"
DRY_RUN_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"
DRY_RUN_STATUS_BLOCKED_DESIGN_NOT_READY = "blocked_adapter_design_not_ready"
DRY_RUN_STATUS_BLOCKED_MISSING_CANDIDATE_RECORD = "blocked_missing_candidate_store_record"
DRY_RUN_STATUS_BLOCKED_RECORD_MISMATCH = "blocked_candidate_store_record_mismatch"
DRY_RUN_STATUS_BLOCKED_LOCATOR = "blocked_invalid_locator"
DRY_RUN_STATUS_BLOCKED_POLICY = "blocked_policy_quarantine_violation"
DRY_RUN_STATUS_BLOCKED_PRIVATE_PATH = "blocked_private_path_leak"

MAX_ROWS_PER_RESOLVED_PAPER = 2
MAX_ROWS_TOTAL = 4
ALLOWED_ARTIFACT_TYPES = {"section", "paragraph"}
ZERO_COUNTER_FIELDS = (
    "candidateStoreWriteRows",
    "sourceSpanCreatedRows",
    "strictEvidenceRows",
    "citationGradeRows",
    "runtimeEvidenceRows",
    "answerVisibleRows",
    "answerGenerationRows",
    "databaseMutationRows",
    "indexMutationRows",
    "reindexOrReembedRows",
    "parserExecutionRows",
    "canonicalParsedArtifactWriteRows",
    "vaultScanRows",
    "externalDownloadRows",
)
FULL_APPLY_UNSAFE_COUNTER_FIELDS = tuple(
    field for field in ZERO_COUNTER_FIELDS if field != "candidateStoreWriteRows"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_summary(report: dict[str, Any], *, report_ref: str, count_fields: tuple[str, ...]) -> dict[str, Any]:
    counts = dict(report.get("counts") or {})
    return {
        "schema": normalize_text(report.get("schema")),
        "status": normalize_text(report.get("status")),
        "decision": normalize_text(report.get("decision")),
        "nextRecommendedTranche": normalize_text(report.get("nextRecommendedTranche")),
        "reportRef": normalize_text(report_ref),
        **{field: _int(counts.get(field)) for field in count_fields},
    }


def _adapter_design_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    blockers: list[str] = []
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_ADAPTER_DESIGN_SCHEMA_ID:
        blockers.append("invalid_runtime_adapter_design_schema")
    if report.get("status") != "ready":
        blockers.append("runtime_adapter_design_not_ready")
    if report.get("decision") != ADAPTER_DESIGN_READY_DECISION:
        blockers.append("runtime_adapter_design_invalid_decision")
    if gate.get("readyForRuntimeAdapterDryRun") is not True:
        blockers.append("runtime_adapter_design_gate_not_ready_for_dry_run")
    input_rows = _int(counts.get("inputRows"))
    if input_rows <= 0:
        blockers.append("runtime_adapter_design_has_no_input_rows")
    if _int(counts.get("runtimeAdapterDesignReadyRows")) != input_rows:
        blockers.append("runtime_adapter_design_ready_count_mismatch")
    if _int(counts.get("plannedCandidateStoreReadRows")) != input_rows:
        blockers.append("runtime_adapter_design_candidate_store_read_count_mismatch")
    for field_name in ("answerableRows", "blockedRows", "privatePathLeakRows", "schemaViolationCount", *ZERO_COUNTER_FIELDS):
        if _int(counts.get(field_name)) != 0:
            blockers.append(f"runtime_adapter_design_has_{field_name}")
    if _contains_private_path(report):
        blockers.append("runtime_adapter_design_has_private_path_leak")
    return sorted(set(blockers))


def _full_apply_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    records = [record for record in list(report.get("candidateRecords") or []) if isinstance(record, dict)]
    blockers: list[str] = []
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_EXECUTOR_APPLY_READBACK_SCHEMA_ID:
        blockers.append("invalid_full_apply_executor_schema")
    if report.get("status") != "applied":
        blockers.append("full_apply_executor_not_applied")
    if report.get("decision") != FULL_APPLY_APPLIED_DECISION:
        blockers.append("full_apply_executor_invalid_decision")
    if not records:
        blockers.append("full_apply_executor_candidate_records_missing")
    if _int(counts.get("readbackValidatedRows")) != len(records):
        blockers.append("full_apply_executor_readback_count_mismatch")
    for field_name in ("blockedRows", "privatePathLeakRows", "schemaViolationCount", *FULL_APPLY_UNSAFE_COUNTER_FIELDS):
        if _int(counts.get(field_name)) != 0:
            blockers.append(f"full_apply_executor_has_{field_name}")
    if _contains_private_path(report):
        blockers.append("full_apply_executor_has_private_path_leak")
    return sorted(set(blockers))


def _readback_review_blockers(report: dict[str, Any], *, expected_rows: int) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers: list[str] = []
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_READBACK_REVIEW_SCHEMA_ID:
        blockers.append("invalid_full_apply_readback_review_schema")
    if report.get("status") != "ready":
        blockers.append("full_apply_readback_review_not_ready")
    if report.get("decision") != FULL_APPLY_READBACK_REVIEW_READY_DECISION:
        blockers.append("full_apply_readback_review_invalid_decision")
    if _int(counts.get("readbackValidatedRows")) != expected_rows:
        blockers.append("full_apply_readback_review_validated_count_mismatch")
    if _int(counts.get("matchingStoreRows")) != expected_rows:
        blockers.append("full_apply_readback_review_matching_store_count_mismatch")
    for field_name in ("blockedRows", "privatePathLeakRows", "schemaViolationCount", *ZERO_COUNTER_FIELDS):
        if _int(counts.get(field_name)) != 0:
            blockers.append(f"full_apply_readback_review_has_{field_name}")
    if _contains_private_path(report):
        blockers.append("full_apply_readback_review_has_private_path_leak")
    return sorted(set(blockers))


def _candidate_record_policy_ok(record: dict[str, Any]) -> bool:
    write_policy = dict(record.get("writePolicy") or {})
    return (
        record.get("schema") == PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RECORD_SCHEMA_ID
        and record.get("candidateOnly") is True
        and record.get("answerEvidenceCandidate") is True
        and record.get("answerabilityCandidate") is True
        and record.get("strictEvidence") is False
        and record.get("citationGrade") is False
        and record.get("runtimeEvidence") is False
        and record.get("answerVisible") is False
        and normalize_text(record.get("evidenceTier")) == "parsed_artifact_evidence_chunk_candidate_only"
        and write_policy.get("candidateStoreWrite") is True
        and write_policy.get("sourceSpanCreated") is False
        and write_policy.get("strictEvidenceCreated") is False
        and write_policy.get("citationGradeEvidenceCreated") is False
        and write_policy.get("runtimeEvidenceCreated") is False
        and write_policy.get("answerIntegrationChanged") is False
        and write_policy.get("databaseMutation") is False
        and write_policy.get("vaultScan") is False
        and write_policy.get("reindexOrReembed") is False
        and write_policy.get("canonicalParsedArtifactsWritten") is False
    )


def _record_locator(record: dict[str, Any]) -> tuple[int | None, int | None]:
    locator = dict(record.get("locator") or {})
    chars = dict(locator.get("chars") or {})
    return _int(chars.get("start")), _int(chars.get("end"))


def _row_blockers(row: dict[str, Any], record: dict[str, Any] | None) -> list[str]:
    blockers: list[str] = []
    if normalize_text(row.get("adapterDesignStatus")) != ADAPTER_STATUS_READY_CANDIDATE_ONLY:
        blockers.append("adapter_design_status_not_ready_candidate_only")
    for flag_name in ("adapterDesignReady", "futureAdapterCandidate", "futureCandidateStoreReadRequired"):
        if row.get(flag_name) is not True:
            blockers.append(f"{flag_name}_not_true")
    for flag_name in ("futureRuntimeEvidenceAllowed", "futureAnswerVisibleAllowed", "futureAnswerabilityAllowed"):
        if row.get(flag_name) is not False:
            blockers.append(f"{flag_name}_not_false")
    for field_name in (
        "candidateRecordId",
        "paperId",
        "artifactType",
        "sourceRef",
        "sourceContentHash",
        "spanLocator",
        "snippetHash",
        "candidateStoreRef",
    ):
        if not normalize_text(row.get(field_name)):
            blockers.append(f"{field_name}_missing")
    if normalize_text(row.get("artifactType")) not in ALLOWED_ARTIFACT_TYPES:
        blockers.append("unsupported_artifact_type")
    parsed_start, parsed_end = parse_span_offsets(normalize_text(row.get("spanLocator")), normalize_text(row.get("spanLocator")))
    if parsed_start is None or parsed_end is None:
        blockers.append("chars_locator_missing_or_invalid")
    if _int(row.get("charStart")) != parsed_start:
        blockers.append("char_start_mismatch")
    if _int(row.get("charEnd")) != parsed_end:
        blockers.append("char_end_mismatch")
    if record is None:
        blockers.append("candidate_store_record_missing")
    else:
        for field_name in ("candidateRecordId", "paperId", "artifactType", "sourceRef", "sourceContentHash", "spanLocator", "snippetHash"):
            if normalize_text(record.get(field_name)) != normalize_text(row.get(field_name)):
                blockers.append(f"candidate_record_{field_name}_mismatch")
        record_start, record_end = _record_locator(record)
        if record_start != parsed_start or record_end != parsed_end:
            blockers.append("candidate_record_locator_mismatch")
        excerpt = normalize_text(record.get("excerpt"))
        if not excerpt:
            blockers.append("candidate_record_excerpt_missing")
        elif _sha256_text(excerpt) != normalize_text(row.get("snippetHash")):
            blockers.append("candidate_record_excerpt_hash_mismatch")
        if not _candidate_record_policy_ok(record):
            blockers.append("candidate_record_policy_quarantine_violation")
        if _contains_private_path(record):
            blockers.append("candidate_record_private_path_leak")
    for flag_name in ("strictEvidence", "citationGrade", "runtimeEvidence", "answerVisible", "answerable"):
        if row.get(flag_name) is not False:
            blockers.append(f"{flag_name}_not_false")
    if row.get("candidateOnly") is not True:
        blockers.append("candidateOnly_not_true")
    if _contains_private_path(row):
        blockers.append("private_path_leak")
    return sorted(set(blockers))


def _dry_run_status(blockers: list[str], source_blockers: list[str]) -> str:
    if source_blockers:
        return DRY_RUN_STATUS_BLOCKED_INPUT_SCHEMA
    if not blockers:
        return DRY_RUN_STATUS_READY
    if any(item.startswith("adapter_design") or item.startswith("future") for item in blockers):
        return DRY_RUN_STATUS_BLOCKED_DESIGN_NOT_READY
    if "candidate_store_record_missing" in blockers:
        return DRY_RUN_STATUS_BLOCKED_MISSING_CANDIDATE_RECORD
    if any("locator" in item or item in {"char_start_mismatch", "char_end_mismatch"} for item in blockers):
        return DRY_RUN_STATUS_BLOCKED_LOCATOR
    if any("policy" in item or item.endswith("_not_false") or item.endswith("_not_true") for item in blockers):
        return DRY_RUN_STATUS_BLOCKED_POLICY
    if any("private_path" in item for item in blockers):
        return DRY_RUN_STATUS_BLOCKED_PRIVATE_PATH
    return DRY_RUN_STATUS_BLOCKED_RECORD_MISMATCH


def _first_resolved_paper_ids(rows: list[dict[str, Any]], *, limit: int = 2) -> list[str]:
    counts: Counter[str] = Counter(normalize_text(row.get("paperId")) for row in rows if normalize_text(row.get("paperId")))
    resolved: list[str] = []
    for row in rows:
        paper_id = normalize_text(row.get("paperId"))
        if paper_id and counts[paper_id] >= MAX_ROWS_PER_RESOLVED_PAPER and paper_id not in resolved:
            resolved.append(paper_id)
        if len(resolved) >= limit:
            break
    return resolved


def _selected_ids_for_positive_scenario(rows: list[dict[str, Any]], resolved_paper_ids: list[str]) -> set[str]:
    selected: set[str] = set()
    per_paper: Counter[str] = Counter()
    resolved_order = {paper_id: index for index, paper_id in enumerate(resolved_paper_ids)}
    ready_rows = [
        row
        for row in rows
        if row.get("adapterDryRunReady") is True and normalize_text(row.get("paperId")) in resolved_order
    ]
    ready_rows.sort(key=lambda item: (resolved_order[normalize_text(item.get("paperId"))], _int(item.get("sourceOrder"))))
    for row in ready_rows:
        if len(selected) >= MAX_ROWS_TOTAL:
            break
        paper_id = normalize_text(row.get("paperId"))
        if per_paper[paper_id] >= MAX_ROWS_PER_RESOLVED_PAPER:
            continue
        selected.add(normalize_text(row.get("candidateRecordId")))
        per_paper[paper_id] += 1
    return selected


def _planned_evidence_item(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_type": "paper",
        "sourceType": "paper",
        "source_id": normalize_text(row.get("paperId")),
        "sourceId": normalize_text(row.get("paperId")),
        "source_ref": normalize_text(row.get("sourceRef")),
        "sourceRef": normalize_text(row.get("sourceRef")),
        "source_content_hash": normalize_text(row.get("sourceContentHash")),
        "sourceContentHash": normalize_text(row.get("sourceContentHash")),
        "span_locator": normalize_text(row.get("spanLocator")),
        "spanLocator": normalize_text(row.get("spanLocator")),
        "char_start": _int(row.get("charStart")),
        "charStart": _int(row.get("charStart")),
        "char_end": _int(row.get("charEnd")),
        "charEnd": _int(row.get("charEnd")),
        "excerptSource": "candidate_store_readback",
        "excerptIncludedInThisReport": False,
        "snippet_hash": normalize_text(row.get("snippetHash")),
        "snippetHash": normalize_text(row.get("snippetHash")),
        "content_hash": normalize_text(row.get("snippetHash")),
        "evidence_kind": "parsed_artifact_evidence_chunk",
        "evidenceKind": "parsed_artifact_evidence_chunk",
        "derivative_source": {
            "candidateRecordId": normalize_text(row.get("candidateRecordId")),
            "candidateStoreRef": normalize_text(row.get("candidateStoreRef")),
            "artifactType": normalize_text(row.get("artifactType")),
        },
        "derivativeSource": {
            "candidateRecordId": normalize_text(row.get("candidateRecordId")),
            "candidateStoreRef": normalize_text(row.get("candidateStoreRef")),
            "artifactType": normalize_text(row.get("artifactType")),
        },
        "candidateOnly": True,
        "runtimeEvidence": False,
        "answerVisible": False,
        "answerable": False,
        "strictEvidence": False,
        "citationGrade": False,
    }


def _scenario_results(*, resolved_paper_ids: list[str], selected_rows: list[dict[str, Any]], considered_rows: int) -> list[dict[str, Any]]:
    return [
        {
            "scenarioId": "opt_in_off",
            "enabled": False,
            "status": "skipped",
            "sourceType": "paper",
            "resolvedPaperIds": resolved_paper_ids,
            "candidateRowsConsidered": 0,
            "rowsAdded": 0,
            "blockedRows": 0,
            "skippedReason": "query_plan_opt_in_not_enabled",
        },
        {
            "scenarioId": "opt_in_missing_resolved_paper_ids",
            "enabled": True,
            "status": "skipped",
            "sourceType": "paper",
            "resolvedPaperIds": [],
            "candidateRowsConsidered": 0,
            "rowsAdded": 0,
            "blockedRows": 0,
            "skippedReason": "resolved_paper_ids_required",
        },
        {
            "scenarioId": "source_type_not_paper",
            "enabled": True,
            "status": "skipped",
            "sourceType": "web",
            "resolvedPaperIds": resolved_paper_ids,
            "candidateRowsConsidered": 0,
            "rowsAdded": 0,
            "blockedRows": 0,
            "skippedReason": "source_type_not_paper",
        },
        {
            "scenarioId": "opt_in_paper_resolved",
            "enabled": True,
            "status": "applied_dry_run",
            "sourceType": "paper",
            "resolvedPaperIds": resolved_paper_ids,
            "candidateRowsConsidered": considered_rows,
            "rowsAdded": len(selected_rows),
            "blockedRows": 0,
            "skippedReason": "",
            "selectedCandidateRecordIds": [normalize_text(row.get("candidateRecordId")) for row in selected_rows],
        },
    ]


def build_parsed_artifact_evidence_chunk_candidate_runtime_adapter_dry_run(
    *,
    runtime_adapter_design_report: dict[str, Any],
    full_apply_report: dict[str, Any],
    full_apply_readback_review_report: dict[str, Any],
    source_runtime_adapter_design_report_ref: str,
    source_full_apply_report_ref: str,
    source_full_apply_readback_review_report_ref: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    design_rows = [
        dict(row)
        for row in list(runtime_adapter_design_report.get("rows") or [])
        if isinstance(row, dict)
    ]
    candidate_records = [
        dict(record)
        for record in list(full_apply_report.get("candidateRecords") or [])
        if isinstance(record, dict)
    ]
    records_by_id = {
        normalize_text(record.get("candidateRecordId")): record
        for record in candidate_records
        if normalize_text(record.get("candidateRecordId"))
    }
    source_blockers = sorted(
        set(
            [
                *_adapter_design_blockers(runtime_adapter_design_report),
                *_full_apply_blockers(full_apply_report),
                *_readback_review_blockers(
                    full_apply_readback_review_report,
                    expected_rows=len(design_rows),
                ),
            ]
        )
    )

    rows: list[dict[str, Any]] = []
    for index, source_row in enumerate(design_rows, start=1):
        record = records_by_id.get(normalize_text(source_row.get("candidateRecordId")))
        row_blockers = _row_blockers(source_row, record)
        status = _dry_run_status(row_blockers, source_blockers)
        blockers = sorted(set([*source_blockers, *row_blockers]))
        ready = status == DRY_RUN_STATUS_READY and not blockers
        excerpt = normalize_text(record.get("excerpt")) if record else ""
        rows.append(
            {
                "adapterDryRunRowId": f"parsed-artifact-evidence-chunk-candidate-runtime-adapter-dry-run:{index:04d}",
                "sourceAdapterDesignRowId": normalize_text(source_row.get("adapterDesignRowId")),
                "candidateRecordId": normalize_text(source_row.get("candidateRecordId")),
                "sourceCandidateRowId": normalize_text(source_row.get("sourceCandidateRowId")),
                "paperId": normalize_text(source_row.get("paperId")),
                "artifactType": normalize_text(source_row.get("artifactType")),
                "sourceRef": normalize_text(source_row.get("sourceRef")),
                "sourceContentHash": normalize_text(source_row.get("sourceContentHash")),
                "spanLocator": normalize_text(source_row.get("spanLocator")),
                "charStart": source_row.get("charStart"),
                "charEnd": source_row.get("charEnd"),
                "snippetHash": normalize_text(source_row.get("snippetHash")),
                "candidateStoreRef": normalize_text(source_row.get("candidateStoreRef")),
                "sourceOrder": index,
                "adapterDryRunStatus": status,
                "adapterDryRunBlockers": blockers,
                "adapterDryRunReady": ready,
                "candidateStoreRecordMatched": record is not None and not blockers,
                "excerptReadback": bool(excerpt) and "candidate_record_excerpt_missing" not in blockers,
                "excerptLength": len(excerpt),
                "snippetHashVerified": bool(excerpt) and _sha256_text(excerpt) == normalize_text(source_row.get("snippetHash")),
                "selectedForPositiveScenario": False,
                "selectionReason": "pending_selection" if ready else "blocked",
                "plannedEvidenceItemPreview": _planned_evidence_item(source_row),
                "candidateOnly": True,
                "strictEvidence": False,
                "citationGrade": False,
                "runtimeEvidence": False,
                "answerVisible": False,
                "answerable": False,
                "checks": {
                    "adapterDesignReady": normalize_text(source_row.get("adapterDesignStatus")) == ADAPTER_STATUS_READY_CANDIDATE_ONLY
                    and source_row.get("adapterDesignReady") is True,
                    "candidateStoreRecordPresent": record is not None,
                    "sourceIdentityMatches": not any("sourceRef_mismatch" in item or "paperId_mismatch" in item for item in blockers),
                    "sourceHashMatches": "candidate_record_sourceContentHash_mismatch" not in blockers,
                    "snippetHashMatches": "candidate_record_snippetHash_mismatch" not in blockers
                    and "candidate_record_excerpt_hash_mismatch" not in blockers,
                    "strictCharsLocatorMatches": "candidate_record_locator_mismatch" not in blockers
                    and "chars_locator_missing_or_invalid" not in blockers,
                    "policyQuarantineMaintained": not any("policy" in item or item.endswith("_not_false") for item in blockers),
                    "excerptReadbackPresent": bool(excerpt),
                    "runtimeEvidenceStillFalse": True,
                    "answerVisibleStillFalse": True,
                    "answerableStillFalse": True,
                },
            }
        )

    ready_rows = [row for row in rows if row.get("adapterDryRunReady") is True]
    resolved_paper_ids = _first_resolved_paper_ids(ready_rows)
    selected_ids = _selected_ids_for_positive_scenario(ready_rows, resolved_paper_ids)
    selected_rows: list[dict[str, Any]] = []
    ready_by_paper = Counter(normalize_text(row.get("paperId")) for row in ready_rows)
    for row in rows:
        if normalize_text(row.get("candidateRecordId")) in selected_ids:
            row["selectedForPositiveScenario"] = True
            row["selectionReason"] = "selected_by_resolved_paper_order_and_caps"
            selected_rows.append(row)
        elif row.get("adapterDryRunReady") is True and normalize_text(row.get("paperId")) in resolved_paper_ids:
            row["selectionReason"] = "held_by_per_paper_or_total_cap"
        elif row.get("adapterDryRunReady") is True:
            row["selectionReason"] = "held_outside_resolved_paper_scope"

    by_status = Counter(row["adapterDryRunStatus"] for row in rows)
    by_type = Counter(row["artifactType"] for row in rows)
    blocked_rows = len(rows) - len(ready_rows)
    private_path_leak_rows = sum(1 for row in rows if any("private_path" in item for item in row["adapterDryRunBlockers"]))
    schema_violations = list(source_blockers)
    if private_path_leak_rows:
        schema_violations = sorted(set([*schema_violations, "private_path_leak"]))
    considered_rows = sum(ready_by_paper[paper_id] for paper_id in resolved_paper_ids)
    counts = {
        "inputRows": len(design_rows),
        "adapterDesignReadyInputRows": sum(1 for row in design_rows if row.get("adapterDesignReady") is True),
        "candidateStoreRecordInputRows": len(candidate_records),
        "candidateStoreRecordMatchedRows": sum(1 for row in rows if row.get("candidateStoreRecordMatched") is True),
        "excerptReadbackRows": sum(1 for row in rows if row.get("excerptReadback") is True),
        "snippetHashVerifiedRows": sum(1 for row in rows if row.get("snippetHashVerified") is True),
        "adapterDryRunReadyRows": len(ready_rows) if not source_blockers else 0,
        "resolvedPaperRows": len(resolved_paper_ids) if not source_blockers else 0,
        "positiveScenarioCandidateRowsConsidered": considered_rows if not source_blockers else 0,
        "positiveScenarioSelectedRows": len(selected_rows) if not source_blockers else 0,
        "positiveScenarioHeldRows": max(0, len(ready_rows) - len(selected_rows)) if not source_blockers else 0,
        "dryRunScenarioRows": 4,
        "skippedScenarioRows": 3,
        "plannedEvidenceItemPreviewRows": len(selected_rows) if not source_blockers else 0,
        "plannedDiagnosticsRows": 4,
        "answerableRows": 0,
        "blockedRows": blocked_rows,
        "blockedInputSchemaViolationRows": len(rows) if source_blockers and rows else int(bool(source_blockers)),
        "blockedAdapterDesignNotReadyRows": by_status.get(DRY_RUN_STATUS_BLOCKED_DESIGN_NOT_READY, 0),
        "blockedMissingCandidateRecordRows": by_status.get(DRY_RUN_STATUS_BLOCKED_MISSING_CANDIDATE_RECORD, 0),
        "blockedRecordMismatchRows": by_status.get(DRY_RUN_STATUS_BLOCKED_RECORD_MISMATCH, 0),
        "blockedInvalidLocatorRows": by_status.get(DRY_RUN_STATUS_BLOCKED_LOCATOR, 0),
        "blockedPolicyQuarantineRows": by_status.get(DRY_RUN_STATUS_BLOCKED_POLICY, 0),
        "blockedPrivatePathRows": by_status.get(DRY_RUN_STATUS_BLOCKED_PRIVATE_PATH, 0),
        **{field: 0 for field in ZERO_COUNTER_FIELDS},
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(schema_violations),
        "byArtifactType": dict(by_type),
        "byAdapterDryRunStatus": dict(by_status),
    }
    status = "blocked" if schema_violations or blocked_rows else "ready"
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_ADAPTER_DRY_RUN_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE if status == "ready" else "parsed_artifact_evidence_chunk_candidate_runtime_adapter_dry_run_repair",
        "sourceRuntimeAdapterDesign": _source_summary(
            runtime_adapter_design_report,
            report_ref=source_runtime_adapter_design_report_ref,
            count_fields=(
                "inputRows",
                "runtimeAdapterDesignReadyRows",
                "plannedCandidateStoreReadRows",
                "plannedEvidenceItemShapeRows",
                "answerableRows",
                "blockedRows",
                "privatePathLeakRows",
                "schemaViolationCount",
            ),
        ),
        "sourceFullApply": _source_summary(
            full_apply_report,
            report_ref=source_full_apply_report_ref,
            count_fields=(
                "plannedApplyRows",
                "appliedCandidateRecordRows",
                "alreadyCorrectRows",
                "candidateStoreWriteRows",
                "readbackValidatedRows",
                "blockedRows",
                "privatePathLeakRows",
                "schemaViolationCount",
            ),
        ),
        "sourceFullApplyReadbackReview": _source_summary(
            full_apply_readback_review_report,
            report_ref=source_full_apply_readback_review_report_ref,
            count_fields=(
                "expectedCandidateRows",
                "storeRows",
                "matchingStoreRows",
                "readbackValidatedRows",
                "blockedRows",
                "privatePathLeakRows",
                "schemaViolationCount",
            ),
        ),
        "policy": {
            "reportOnly": True,
            "runtimeAdapterDryRunOnly": True,
            "runtimeCodeChanged": False,
            "evidenceAssemblyChanged": False,
            "candidateStoreReadSimulated": True,
            "candidateStoreWrite": False,
            "sourceSpanCreated": False,
            "strictEvidenceCreated": False,
            "citationGradeEvidenceCreated": False,
            "runtimeEvidenceCreated": False,
            "answerVisibleExposure": False,
            "answerGeneration": False,
            "answerableRowsAllowedInThisTranche": False,
            "parserRoutingChanged": False,
            "answerIntegrationChanged": False,
            "databaseMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "parserExecution": False,
            "canonicalParsedArtifactsWritten": False,
            "futureRuntimeAdapterImplementationRequired": True,
            "nextGateRequiredBeforeRuntimeUse": True,
        },
        "adapterDryRun": {
            "adapterId": ADAPTER_ID,
            "adapterMode": "opt_in_runtime_dry_run",
            "sourceTypeRequired": "paper",
            "queryPlanOptInKeys": [
                "parsed_artifact_evidence_chunk_adapter",
                "parsedArtifactEvidenceChunkAdapter",
            ],
            "queryPlanOptInValue": "runtime_v1",
            "resolvedPaperIds": resolved_paper_ids,
            "maxRowsPerResolvedPaper": MAX_ROWS_PER_RESOLVED_PAPER,
            "maxRowsTotal": MAX_ROWS_TOTAL,
            "fallbackToAllPapersAllowed": False,
            "selectedCandidateRecordIds": [normalize_text(row.get("candidateRecordId")) for row in selected_rows],
        },
        "scenarioResults": _scenario_results(
            resolved_paper_ids=resolved_paper_ids,
            selected_rows=selected_rows,
            considered_rows=considered_rows,
        ),
        "counts": counts,
        "gate": {
            "readyForRuntimeAdapterImplementation": status == "ready",
            "readyForRuntimeApply": False,
            "runtimeCodeChangeAllowed": False,
            "candidateStoreWriteAllowed": False,
            "sourceSpanCreationAllowed": False,
            "runtimeVisibilityAllowed": False,
            "evidencePromotionAllowed": False,
            "answerVisibleAllowed": False,
            "answerableAllowed": False,
            "schemaViolations": schema_violations,
        },
        "rows": rows,
        "warnings": [],
    }


def render_parsed_artifact_evidence_chunk_candidate_runtime_adapter_dry_run_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    dry_run = dict(report.get("adapterDryRun") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Candidate Runtime Adapter Dry-run",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- adapterId: `{dry_run.get('adapterId')}`",
        f"- resolvedPaperIds: `{dry_run.get('resolvedPaperIds')}`",
        f"- inputRows: `{counts.get('inputRows')}`",
        f"- adapterDryRunReadyRows: `{counts.get('adapterDryRunReadyRows')}`",
        f"- candidateStoreRecordMatchedRows: `{counts.get('candidateStoreRecordMatchedRows')}`",
        f"- excerptReadbackRows: `{counts.get('excerptReadbackRows')}`",
        f"- positiveScenarioSelectedRows: `{counts.get('positiveScenarioSelectedRows')}`",
        f"- plannedEvidenceItemPreviewRows: `{counts.get('plannedEvidenceItemPreviewRows')}`",
        f"- answerableRows: `{counts.get('answerableRows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Mutation Guarantees",
        "",
    ]
    for field in ZERO_COUNTER_FIELDS:
        lines.append(f"- {field}: `{counts.get(field)}`")
    lines.extend(["", "## Scenario Results", ""])
    for scenario in list(report.get("scenarioResults") or []):
        lines.append(
            f"- `{scenario.get('scenarioId')}`: status=`{scenario.get('status')}`, "
            f"rowsAdded=`{scenario.get('rowsAdded')}`, skippedReason=`{scenario.get('skippedReason')}`"
        )
    lines.extend(["", "## Adapter Dry-run Status", ""])
    for status, count in sorted(dict(counts.get("byAdapterDryRunStatus") or {}).items()):
        lines.append(f"- `{status}`: `{count}`")
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_evidence_chunk_candidate_runtime_adapter_dry_run(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_parsed_artifact_evidence_chunk_candidate_runtime_adapter_dry_run_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_ADAPTER_DRY_RUN_SCHEMA_ID",
    "DRY_RUN_STATUS_READY",
    "READY_DECISION",
    "build_parsed_artifact_evidence_chunk_candidate_runtime_adapter_dry_run",
    "load_json",
    "sanitized_report_ref",
    "write_parsed_artifact_evidence_chunk_candidate_runtime_adapter_dry_run",
]
