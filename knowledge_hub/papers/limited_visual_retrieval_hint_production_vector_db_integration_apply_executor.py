"""Apply-gated production vector DB executor for visual retrieval hints.

The executor consumes the Phase 12 dry-run and can write the 125 visual
retrieval-hint records into an isolated production candidate-discovery vector
collection. By default it is report-only. Actual Chroma/FTS mutation requires
``apply=True`` and an explicit ``vector_db_path``.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from knowledge_hub.infrastructure.persistence.vector import VectorDatabase
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_apply_executor import (
    _contains_private_path,
    normalize_text,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor_dry_run import (
    load_json,
    sanitized_report_ref,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_production_vector_db_integration_dry_run import (
    LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_DRY_RUN_SCHEMA_ID,
    READY_DECISION as SOURCE_DRY_RUN_READY_DECISION,
)


LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_APPLY_EXECUTOR_SCHEMA_ID = (
    "knowledge-hub.paper.limited-visual-retrieval-hint-production-vector-db-integration-apply-executor.v1"
)
PRODUCTION_VECTOR_INDEX_RECORD_SCHEMA_ID = "knowledge-hub.paper.visual-retrieval-hint-production-vector-index-record.v1"

EXECUTOR_STATUS_DRY_RUN_READY = "dry_run_ready_production_vector_record"
EXECUTOR_STATUS_APPLIED = "applied_production_vector_record"
EXECUTOR_STATUS_BLOCKED_SOURCE_GATE = "blocked_source_dry_run_gate"
EXECUTOR_STATUS_BLOCKED_CONTRACT = "blocked_contract_violation"
EXECUTOR_STATUS_BLOCKED_POLICY = "blocked_policy_violation"
EXECUTOR_STATUS_BLOCKED_READBACK = "blocked_readback_mismatch"

READY_DECISION = "ready_for_limited_visual_retrieval_hint_production_vector_db_integration_apply"
APPLIED_DECISION = "applied_limited_visual_retrieval_hint_production_vector_db_integration"
BLOCKED_DECISION = "blocked"
NEXT_TRANCHE_READY = "limited_visual_retrieval_hint_production_vector_db_search_quality_eval"
NEXT_TRANCHE_HOLD = "limited_visual_retrieval_hint_production_vector_db_integration_apply_review"

EXPECTED_HINT_ROWS = 125
PRODUCTION_NAMESPACE = "production_visual_retrieval_hint_candidates_v1"
DEFAULT_COLLECTION_NAME = "knowledge_hub_visual_retrieval_hints"
LOCAL_EMBEDDING_PROVIDER_REF = "local_hashing_vectorizer_v1"
LOCAL_EMBEDDING_MODEL_REF = "lexical_hashing_256d_v1"
LOCAL_EMBEDDING_DIMENSIONS = 256
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./+-]*")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _record_hash(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


def _bbox(value: Any) -> list[Any]:
    return list(value or [])


def _counts(report: dict[str, Any]) -> dict[str, Any]:
    return dict(report.get("counts") or {})


def _hashing_vector(text: str, *, dimensions: int = LOCAL_EMBEDDING_DIMENSIONS) -> list[float]:
    buckets = [0.0] * dimensions
    for match in TOKEN_RE.finditer(text or ""):
        token = match.group(0).lower()
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        buckets[bucket] += sign
    norm = math.sqrt(sum(value * value for value in buckets))
    if norm <= 0:
        return buckets
    return [round(value / norm, 6) for value in buckets]


def _vector_hash(vector: list[float]) -> str:
    return _sha256_text(json.dumps(vector, ensure_ascii=True, separators=(",", ":")))


def _policy_ok(policy: dict[str, Any]) -> bool:
    return (
        normalize_text(policy.get("allowedUse") or policy.get("allowed_use")) == "retrieval_hint_only"
        and (policy.get("strictEvidence") if "strictEvidence" in policy else policy.get("strict_evidence")) is False
        and (policy.get("citationGrade") if "citationGrade" in policy else policy.get("citation_grade")) is False
        and (
            policy.get("answerableWithoutTextEvidence")
            if "answerableWithoutTextEvidence" in policy
            else policy.get("answerable_without_text_evidence")
        )
        is False
        and (policy.get("runtimeVisible") if "runtimeVisible" in policy else policy.get("runtime_visible")) is False
        and (policy.get("indexEligible") if "indexEligible" in policy else policy.get("index_eligible")) is False
    )


def _source_summary(report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = _counts(report)
    return {
        "schema": normalize_text(report.get("schema")),
        "status": normalize_text(report.get("status")),
        "decision": normalize_text(report.get("decision")),
        "reportRef": normalize_text(report_ref),
        "plannedProductionVectorRecordRows": _int(counts.get("plannedProductionVectorRecordRows")),
        "futureApplyCandidateRows": _int(counts.get("futureApplyCandidateRows")),
        "candidateDiscoveryOnlyRows": _int(counts.get("candidateDiscoveryOnlyRows")),
        "blockedRows": _int(counts.get("blockedRows")),
        "policyViolationRows": _int(counts.get("policyViolationRows")),
        "privatePathLeakRows": _int(counts.get("privatePathLeakRows")),
        "schemaViolationCount": _int(counts.get("schemaViolationCount")),
    }


def _source_blockers(report: dict[str, Any]) -> list[str]:
    counts = _counts(report)
    records = list(report.get("plannedProductionVectorRecords") or [])
    blockers: list[str] = []
    if report.get("schema") != LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_DRY_RUN_SCHEMA_ID:
        blockers.append("invalid_production_vector_db_integration_dry_run_schema")
    if report.get("status") != "ready":
        blockers.append("production_vector_db_integration_dry_run_not_ready")
    if report.get("decision") != SOURCE_DRY_RUN_READY_DECISION:
        blockers.append("production_vector_db_integration_dry_run_invalid_decision")
    if _int(counts.get("plannedProductionVectorRecordRows")) != EXPECTED_HINT_ROWS:
        blockers.append("source_dry_run_planned_rows_not_125")
    if _int(counts.get("futureApplyCandidateRows")) != EXPECTED_HINT_ROWS:
        blockers.append("source_dry_run_future_apply_rows_not_125")
    if len(records) != EXPECTED_HINT_ROWS:
        blockers.append("source_dry_run_records_not_125")
    for field in (
        "blockedRows",
        "policyViolationRows",
        "privatePathLeakRows",
        "schemaViolationCount",
        "candidateStoreWriteRows",
        "embeddingCallRows",
        "embeddingVectorWriteRows",
        "vectorIndexWriteRows",
        "productionVectorIndexWriteRows",
        "databaseMutationRows",
        "indexMutationRows",
        "runtimeVisibleRows",
        "strictEvidenceRows",
        "citationGradeRows",
        "answerableWithoutTextEvidenceRows",
    ):
        if _int(counts.get(field)) != 0:
            blockers.append(f"source_dry_run_has_{field}")
    if _contains_private_path(report):
        blockers.append("source_dry_run_has_private_path_leak")
    return blockers


def _record_contract_blockers(record: dict[str, Any]) -> list[str]:
    metadata = dict(record.get("metadata") or {})
    policy = dict(record.get("policy") or {})
    execution_plan = dict(record.get("executionPlan") or {})
    blockers: list[str] = []
    for field in (
        "vectorDocumentId",
        "idempotencyKey",
        "hintCandidateId",
        "sourceCandidateId",
        "sourceContentHash",
        "paperId",
        "page",
        "bbox",
        "candidateType",
        "documentText",
        "documentTextHash",
        "embeddingText",
        "embeddingTextHash",
        "metadata",
        "policy",
    ):
        if not record.get(field):
            blockers.append(f"missing_{field}")
    if record.get("schema") != "knowledge-hub.paper.visual-retrieval-hint-production-vector-record-preview.v1":
        blockers.append("invalid_source_preview_record_schema")
    if normalize_text(record.get("productionNamespace")) != PRODUCTION_NAMESPACE:
        blockers.append("unexpected_production_namespace")
    if normalize_text(record.get("targetWriteMethod")) != "VectorDatabase.add_documents":
        blockers.append("unexpected_target_write_method")
    if normalize_text(record.get("routingMode")) != "candidate_discovery_only":
        blockers.append("unexpected_routing_mode")
    if not _policy_ok(policy):
        blockers.append("record_policy_not_retrieval_hint_only")
    if policy.get("productionIndexEligible") is not False:
        blockers.append("record_policy_production_index_eligible_not_false")
    if policy.get("candidateDiscoveryOnly") is not True:
        blockers.append("record_policy_candidate_discovery_only_not_true")
    if not _policy_ok(metadata):
        blockers.append("metadata_policy_not_retrieval_hint_only")
    for key in (
        "actualCandidateStoreWrite",
        "actualEmbeddingCall",
        "actualEmbeddingVectorWrite",
        "actualVectorIndexWrite",
        "actualProductionVectorIndexWrite",
        "actualDatabaseMutation",
        "actualRuntimeExposure",
        "actualEvidencePromotion",
    ):
        if execution_plan.get(key) is not False:
            blockers.append(f"source_execution_plan_{key}_not_false")
    if _contains_private_path(record):
        blockers.append("private_path_leak")
    return blockers


def _index_record(record: dict[str, Any], *, collection_name: str) -> dict[str, Any]:
    embedding_text = normalize_text(record.get("embeddingText"))
    vector = _hashing_vector(embedding_text)
    metadata = dict(record.get("metadata") or {})
    metadata.update(
        {
            "retrieval_unit_schema": "visual_retrieval_hint_production_vector_document.v1",
            "namespace": PRODUCTION_NAMESPACE,
            "collection_name": collection_name,
            "source_type": "visual_retrieval_hint",
            "retrieval_unit_kind": "candidate_discovery_signal",
            "allowedUse": "retrieval_hint_only",
            "allowed_use": "retrieval_hint_only",
            "document_id": normalize_text(record.get("vectorDocumentId")),
            "hint_candidate_id": normalize_text(record.get("hintCandidateId")),
            "source_candidate_id": normalize_text(record.get("sourceCandidateId")),
            "paper_id": normalize_text(record.get("paperId")),
            "source_content_hash": normalize_text(record.get("sourceContentHash")),
            "candidate_type": normalize_text(record.get("candidateType")),
            "page": _int(record.get("page")),
            "bbox": _bbox(record.get("bbox")),
            "document_text_hash": normalize_text(record.get("documentTextHash")),
            "embedding_text_hash": normalize_text(record.get("embeddingTextHash")),
            "embedding_provider_ref": LOCAL_EMBEDDING_PROVIDER_REF,
            "embedding_model_ref": LOCAL_EMBEDDING_MODEL_REF,
            "embedding_dimensions": LOCAL_EMBEDDING_DIMENSIONS,
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": False,
            "indexEligible": False,
            "productionIndexEligible": False,
            "candidateDiscoveryOnly": True,
        }
    )
    payload = {
        "schema": PRODUCTION_VECTOR_INDEX_RECORD_SCHEMA_ID,
        "namespace": PRODUCTION_NAMESPACE,
        "collectionName": collection_name,
        "vectorDocumentId": normalize_text(record.get("vectorDocumentId")),
        "idempotencyKey": normalize_text(record.get("idempotencyKey")),
        "hintCandidateId": normalize_text(record.get("hintCandidateId")),
        "sourceCandidateId": normalize_text(record.get("sourceCandidateId")),
        "sourceContentHash": normalize_text(record.get("sourceContentHash")),
        "paperId": normalize_text(record.get("paperId")),
        "paperRef": normalize_text(record.get("paperRef")),
        "page": _int(record.get("page")),
        "bbox": _bbox(record.get("bbox")),
        "candidateType": normalize_text(record.get("candidateType")),
        "documentText": normalize_text(record.get("documentText")),
        "documentTextHash": normalize_text(record.get("documentTextHash")),
        "embeddingText": embedding_text,
        "embeddingTextHash": normalize_text(record.get("embeddingTextHash")),
        "embeddingVector": vector,
        "embeddingVectorSha256": _vector_hash(vector),
        "sourcePreviewRecordSha256": normalize_text(record.get("plannedVectorRecordSha256")),
        "metadata": metadata,
        "policy": {
            "allowedUse": "retrieval_hint_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": False,
            "indexEligible": False,
            "productionIndexEligible": False,
            "candidateDiscoveryOnly": True,
        },
    }
    payload["productionVectorRecordSha256"] = _record_hash({k: v for k, v in payload.items() if k != "embeddingVector"})
    return payload


def _readback_validate(vector_db: VectorDatabase, records: list[dict[str, Any]]) -> tuple[int, list[str]]:
    validated = 0
    blockers: list[str] = []
    for record in records:
        doc_id = normalize_text(record.get("vectorDocumentId"))
        result = vector_db.get_documents(
            filter_dict={"document_id": doc_id},
            limit=1,
            include_ids=True,
            include_documents=True,
            include_metadatas=True,
            include_embeddings=False,
        )
        ids = [normalize_text(value) for value in list(result.get("ids") or [])]
        documents = list(result.get("documents") or [])
        metadatas = [dict(item or {}) for item in list(result.get("metadatas") or [])]
        if doc_id not in ids:
            blockers.append(f"readback_missing:{doc_id}")
            continue
        index = ids.index(doc_id)
        document = normalize_text(documents[index] if index < len(documents) else "")
        metadata = metadatas[index] if index < len(metadatas) else {}
        if document != normalize_text(record.get("documentText")):
            blockers.append(f"readback_document_mismatch:{doc_id}")
            continue
        if normalize_text(metadata.get("source_content_hash")) != normalize_text(record.get("sourceContentHash")):
            blockers.append(f"readback_source_content_hash_mismatch:{doc_id}")
            continue
        if metadata.get("runtimeVisible") is not False or metadata.get("strictEvidence") is not False:
            blockers.append(f"readback_policy_mismatch:{doc_id}")
            continue
        validated += 1
    return validated, blockers


def execute_limited_visual_retrieval_hint_production_vector_db_integration_apply_executor(
    *,
    production_vector_db_integration_dry_run: dict[str, Any],
    source_production_vector_db_integration_dry_run_ref: str,
    vector_db_path: str | Path | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    apply: bool = False,
    run_id: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    run_id = normalize_text(run_id) or f"visual-retrieval-hint-production-vector-apply-{utc_now_iso()}"
    source_blockers = _source_blockers(production_vector_db_integration_dry_run)
    source_records = [dict(row) for row in production_vector_db_integration_dry_run.get("plannedProductionVectorRecords") or []]
    technical_blockers = list(source_blockers)
    warnings: list[str] = []
    if apply and not vector_db_path:
        technical_blockers.append("apply_requires_vector_db_path")
        warnings.append("apply_requires_vector_db_path")

    rows: list[dict[str, Any]] = []
    index_records: list[dict[str, Any]] = []
    row_blocker_names: list[str] = []
    for index, source_record in enumerate(source_records, start=1):
        record_blockers = _record_contract_blockers(source_record)
        row_blocker_names.extend(record_blockers)
        index_record = _index_record(source_record, collection_name=collection_name)
        if source_blockers:
            status = EXECUTOR_STATUS_BLOCKED_SOURCE_GATE
        elif any("policy" in blocker for blocker in record_blockers):
            status = EXECUTOR_STATUS_BLOCKED_POLICY
        elif record_blockers:
            status = EXECUTOR_STATUS_BLOCKED_CONTRACT
        else:
            status = EXECUTOR_STATUS_DRY_RUN_READY
            index_records.append(index_record)
        rows.append(
            {
                "executorRowId": f"limited-visual-retrieval-hint-production-vector-apply-executor:{index:04d}",
                "status": status,
                "vectorDocumentId": normalize_text(index_record.get("vectorDocumentId")),
                "idempotencyKey": normalize_text(index_record.get("idempotencyKey")),
                "productionVectorRecordSha256": normalize_text(index_record.get("productionVectorRecordSha256")),
                "hintCandidateId": normalize_text(index_record.get("hintCandidateId")),
                "sourceCandidateId": normalize_text(index_record.get("sourceCandidateId")),
                "sourceContentHash": normalize_text(index_record.get("sourceContentHash")),
                "paperId": normalize_text(index_record.get("paperId")),
                "page": _int(index_record.get("page")),
                "bbox": _bbox(index_record.get("bbox")),
                "candidateType": normalize_text(index_record.get("candidateType")),
                "collectionName": collection_name,
                "embeddingProviderRef": LOCAL_EMBEDDING_PROVIDER_REF,
                "embeddingModelRef": LOCAL_EMBEDDING_MODEL_REF,
                "embeddingDimensions": LOCAL_EMBEDDING_DIMENSIONS,
                "appliedProductionVectorRecord": False,
                "readbackValidated": False,
                "runtimeVisible": False,
                "strictEvidence": False,
                "citationGrade": False,
                "answerableWithoutTextEvidence": False,
                "blockers": [*source_blockers, *record_blockers],
            }
        )

    readback_validated = 0
    applied_rows = 0
    if apply and index_records and not technical_blockers and vector_db_path:
        vector_db = VectorDatabase(str(Path(vector_db_path).expanduser()), collection_name)
        vector_db.add_documents(
            documents=[normalize_text(record.get("documentText")) for record in index_records],
            embeddings=[list(record.get("embeddingVector") or []) for record in index_records],
            metadatas=[dict(record.get("metadata") or {}) for record in index_records],
            ids=[normalize_text(record.get("vectorDocumentId")) for record in index_records],
        )
        applied_rows = len(index_records)
        readback_validated, readback_blockers = _readback_validate(vector_db, index_records)
        if readback_blockers:
            technical_blockers.extend(readback_blockers)
            warnings.extend(readback_blockers)
        applied_ids = {normalize_text(record.get("vectorDocumentId")) for record in index_records}
        for row in rows:
            if row["vectorDocumentId"] in applied_ids and not readback_blockers:
                row["status"] = EXECUTOR_STATUS_APPLIED
                row["appliedProductionVectorRecord"] = True
                row["readbackValidated"] = True
            elif row["vectorDocumentId"] in applied_ids:
                row["status"] = EXECUTOR_STATUS_BLOCKED_READBACK
                row["blockers"] = sorted(set(row.get("blockers") or []) | set(readback_blockers))

    technical_blockers.extend(row_blocker_names)
    blocked_rows = [row for row in rows if row.get("status") not in {EXECUTOR_STATUS_DRY_RUN_READY, EXECUTOR_STATUS_APPLIED}]
    if _contains_private_path(rows) or _contains_private_path(index_records):
        technical_blockers.append("private_path_leak")
    technical_blockers = sorted(set(technical_blockers))
    ready_count = sum(1 for row in rows if row.get("status") == EXECUTOR_STATUS_DRY_RUN_READY)
    applied_count = sum(1 for row in rows if row.get("status") == EXECUTOR_STATUS_APPLIED)
    status = "blocked" if technical_blockers or blocked_rows else ("applied" if apply else "ready")
    decision = BLOCKED_DECISION if status == "blocked" else (APPLIED_DECISION if status == "applied" else READY_DECISION)
    counts = {
        "sourceDryRunRows": len(source_records),
        "executorRows": len(rows),
        "plannedProductionVectorRecordRows": ready_count if not apply else 0,
        "dryRunReadyProductionVectorRecordRows": ready_count if not apply else 0,
        "appliedProductionVectorRecordRows": applied_count,
        "readbackValidatedRows": readback_validated,
        "candidateDiscoveryOnlyRows": ready_count if not apply else applied_count,
        "blockedRows": len(blocked_rows),
        "policyViolationRows": sum(1 for row in rows if row.get("status") == EXECUTOR_STATUS_BLOCKED_POLICY),
        "privatePathLeakRows": 1 if "private_path_leak" in technical_blockers else 0,
        "externalEmbeddingCallRows": 0,
        "embeddingCallRows": 0,
        "localEmbeddingRows": len(index_records),
        "embeddingVectorWriteRows": applied_rows,
        "vectorIndexWriteRows": applied_rows,
        "productionVectorIndexWriteRows": applied_rows,
        "databaseMutationRows": applied_rows,
        "indexMutationRows": applied_rows,
        "runtimeVisibleRows": 0,
        "strictEvidenceRows": 0,
        "citationGradeRows": 0,
        "answerableWithoutTextEvidenceRows": 0,
        "graphDbWriteRows": 0,
        "ontologyWriteRows": 0,
        "memoryCardWriteRows": 0,
        "clusterWriteRows": 0,
        "schemaViolationCount": len(technical_blockers),
        "byCandidateType": dict(Counter(row.get("candidateType") for row in rows)),
        "byStatus": dict(Counter(row.get("status") for row in rows)),
    }
    gate = {
        "passed": status in {"ready", "applied"} and counts["blockedRows"] == 0,
        "expectedRows": EXPECTED_HINT_ROWS,
        "checks": {
            "sourceDryRunReady": not source_blockers,
            "rowsExactly125": len(rows) == EXPECTED_HINT_ROWS,
            "allRowsReadyOrApplied": ready_count + applied_count == EXPECTED_HINT_ROWS,
            "applyRequiresVectorDbPath": (not apply) or bool(vector_db_path),
            "readbackValidatedWhenApplied": (not apply) or readback_validated == EXPECTED_HINT_ROWS,
            "noRuntimeExposure": counts["runtimeVisibleRows"] == 0,
            "noEvidencePromotion": counts["strictEvidenceRows"] == 0 and counts["citationGradeRows"] == 0,
        },
        "observed": {
            "plannedProductionVectorRecordRows": counts["plannedProductionVectorRecordRows"],
            "appliedProductionVectorRecordRows": counts["appliedProductionVectorRecordRows"],
            "readbackValidatedRows": counts["readbackValidatedRows"],
            "blockedRows": counts["blockedRows"],
        },
    }
    return {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_APPLY_EXECUTOR_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": decision,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status in {"ready", "applied"} else NEXT_TRANCHE_HOLD,
        "sourceProductionVectorDbIntegrationDryRun": _source_summary(
            production_vector_db_integration_dry_run,
            report_ref=source_production_vector_db_integration_dry_run_ref,
        ),
        "input": {
            "apply": bool(apply),
            "runId": run_id,
            "vectorDbPathRef": "explicit_vector_db_path" if vector_db_path else "",
            "collectionName": collection_name,
            "productionNamespace": PRODUCTION_NAMESPACE,
        },
        "policy": {
            "dryRunByDefault": True,
            "applyRequired": True,
            "candidateDiscoveryOnly": True,
            "externalEmbeddingCalls": False,
            "localEmbeddingProviderRef": LOCAL_EMBEDDING_PROVIDER_REF,
            "runtimeVisible": False,
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "indexEligible": False,
            "productionIndexEligible": False,
        },
        "method": {
            "name": "visual_retrieval_hint_production_vector_db_integration_apply_executor_v1",
            "writeBoundary": "knowledge_hub.infrastructure.persistence.vector.VectorDatabase.add_documents",
            "readbackBoundary": "VectorDatabase.get_documents(filter_dict={'document_id': ...})",
            "description": "Apply-gated isolated production candidate-discovery vector upsert for visual retrieval hints.",
        },
        "counts": counts,
        "gate": gate,
        "rows": rows,
        "productionVectorIndexRecordPreviews": [
            {key: value for key, value in record.items() if key != "embeddingVector"} | {
                "embeddingVectorPresent": True,
                "embeddingVectorLength": len(list(record.get("embeddingVector") or [])),
            }
            for record in index_records
        ],
        "sourceBlockers": sorted(set(source_blockers)),
        "technicalBlockers": technical_blockers,
        "warnings": sorted(set(warnings)),
    }


def render_limited_visual_retrieval_hint_production_vector_db_integration_apply_executor_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# Limited Visual Retrieval Hint Production Vector DB Integration Apply Executor 005",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- plannedProductionVectorRecordRows: `{counts.get('plannedProductionVectorRecordRows')}`",
        f"- appliedProductionVectorRecordRows: `{counts.get('appliedProductionVectorRecordRows')}`",
        f"- readbackValidatedRows: `{counts.get('readbackValidatedRows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        f"- productionVectorIndexWriteRows: `{counts.get('productionVectorIndexWriteRows')}`",
        f"- databaseMutationRows: `{counts.get('databaseMutationRows')}`",
        f"- runtimeVisibleRows: `{counts.get('runtimeVisibleRows')}`",
        f"- strictEvidenceRows: `{counts.get('strictEvidenceRows')}`",
        f"- citationGradeRows: `{counts.get('citationGradeRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Gate",
        "",
        f"- passed: `{gate.get('passed')}`",
        f"- allRowsReadyOrApplied: `{dict(gate.get('checks') or {}).get('allRowsReadyOrApplied')}`",
        f"- readbackValidatedWhenApplied: `{dict(gate.get('checks') or {}).get('readbackValidatedWhenApplied')}`",
        "",
        "## Boundary",
        "",
        "- Writes only through `VectorDatabase.add_documents` when `--apply --vector-db-path` is explicit.",
        "- Records remain candidate-discovery-only and are not answer evidence.",
    ]
    blockers = list(report.get("technicalBlockers") or [])
    if blockers:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    return "\n".join(lines) + "\n"


def write_limited_visual_retrieval_hint_production_vector_db_integration_apply_executor(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_limited_visual_retrieval_hint_production_vector_db_integration_apply_executor_markdown(report),
        encoding="utf-8",
    )
    return {"json": str(report_json), "md": str(report_md)}


__all__ = [
    "APPLIED_DECISION",
    "EXECUTOR_STATUS_APPLIED",
    "EXECUTOR_STATUS_DRY_RUN_READY",
    "LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_APPLY_EXECUTOR_SCHEMA_ID",
    "READY_DECISION",
    "execute_limited_visual_retrieval_hint_production_vector_db_integration_apply_executor",
    "load_json",
    "sanitized_report_ref",
    "write_limited_visual_retrieval_hint_production_vector_db_integration_apply_executor",
]
