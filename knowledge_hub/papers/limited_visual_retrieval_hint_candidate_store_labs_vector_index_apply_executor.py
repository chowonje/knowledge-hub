"""Apply-gated executor for labs visual retrieval-hint vector indexing.

The executor consumes the labs vector-index apply review plus its source
apply-executor dry-run. By default it is report-only. With ``apply=True`` and a
``papers_dir`` it writes a local labs-only JSONL vector index and validates
readback. It never writes the production vector DB, promotes evidence, or makes
runtime-answer-visible records.
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

from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_apply_executor import (
    _contains_private_path,
    normalize_text,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor_dry_run import (
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_EXECUTOR_DRY_RUN_SCHEMA_ID,
    PLANNED_LABS_VECTOR_INDEX_REF,
    READY_DECISION as APPLY_EXECUTOR_DRY_RUN_READY_DECISION,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_review import (
    APPLY_REVIEW_STATUS_READY,
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_REVIEW_SCHEMA_ID,
    READY_DECISION as APPLY_REVIEW_READY_DECISION,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_dry_run import (
    LABS_NAMESPACE,
)


LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_EXECUTOR_SCHEMA_ID = (
    "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-labs-vector-index-apply-executor.v1"
)
LABS_VECTOR_INDEX_RECORD_SCHEMA_ID = "knowledge-hub.paper.visual-retrieval-hint-labs-vector-index-record.v1"

EXECUTOR_STATUS_DRY_RUN_READY = "dry_run_ready_labs_vector_record"
EXECUTOR_STATUS_APPLIED = "applied_labs_vector_record"
EXECUTOR_STATUS_BLOCKED_NON_READY_APPLY_REVIEW_ROW = "blocked_non_ready_apply_review_row"
EXECUTOR_STATUS_BLOCKED_MISSING_UPSERT_RECORD = "blocked_missing_upsert_record"
EXECUTOR_STATUS_BLOCKED_VECTOR_RECORD_CONTRACT = "blocked_vector_record_contract"
EXECUTOR_STATUS_BLOCKED_POLICY_VIOLATION = "blocked_policy_violation"
EXECUTOR_STATUS_BLOCKED_READBACK_MISMATCH = "blocked_readback_mismatch"

READY_DECISION = "ready_for_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply"
APPLIED_DECISION = "applied_limited_visual_retrieval_hint_candidate_store_labs_vector_index"
BLOCKED_DECISION = "blocked"

NEXT_TRANCHE_DRY_RUN = "limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply"
NEXT_TRANCHE_APPLIED = "limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_readback_review"

LOCAL_EMBEDDING_PROVIDER_REF = "local_hashing_vectorizer_v1"
LOCAL_EMBEDDING_MODEL_REF = "lexical_hashing_256d_v1"
LOCAL_EMBEDDING_DIMENSIONS = 256
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./+-]*")


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


def _safe_filename(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return text.strip("._") or "unknown"


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _record_hash(record: dict[str, Any]) -> str:
    return _sha256_text(_canonical_json(record))


def _bbox(value: Any) -> list[Any]:
    return list(value or [])


def _index_path(papers_dir: str | Path) -> Path:
    return (
        Path(str(papers_dir)).expanduser()
        / "visual_retrieval_hints"
        / "labs_vector_index"
        / f"{LABS_NAMESPACE}.v1.jsonl"
    )


def _run_manifest_path(papers_dir: str | Path, run_id: str) -> Path:
    return (
        Path(str(papers_dir)).expanduser()
        / "visual_retrieval_hints"
        / "labs_vector_index"
        / "runs"
        / f"{_safe_filename(run_id)}.json"
    )


def _index_ref() -> str:
    return f"papers_dir/visual_retrieval_hints/labs_vector_index/{LABS_NAMESPACE}.v1.jsonl"


def _run_manifest_ref(run_id: str) -> str:
    return f"papers_dir/visual_retrieval_hints/labs_vector_index/runs/{_safe_filename(run_id)}.json"


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


def _vector_record_key(record: dict[str, Any]) -> str:
    return "|".join(
        [
            normalize_text(record.get("namespace")),
            normalize_text(record.get("vectorDocumentId")),
            normalize_text(record.get("embeddingTextHash")),
        ]
    )


def _hashing_vector(text: str, *, dimensions: int = LOCAL_EMBEDDING_DIMENSIONS) -> list[float]:
    buckets = [0.0] * dimensions
    tokens = [match.group(0).lower() for match in TOKEN_RE.finditer(text or "")]
    for token in tokens:
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


def _policy_flags_ok(policy: dict[str, Any]) -> bool:
    return (
        normalize_text(policy.get("allowedUse")) == "retrieval_hint_only"
        and policy.get("strictEvidence") is False
        and policy.get("citationGrade") is False
        and policy.get("answerableWithoutTextEvidence") is False
        and policy.get("runtimeVisible") is False
        and policy.get("indexEligible") is False
        and policy.get("productionIndexEligible") is False
        and policy.get("labsOnly") is True
    )


def _metadata_policy_ok(metadata: dict[str, Any]) -> bool:
    return (
        normalize_text(metadata.get("allowedUse")) == "retrieval_hint_only"
        and metadata.get("strictEvidence") is False
        and metadata.get("citationGrade") is False
        and metadata.get("answerableWithoutTextEvidence") is False
        and metadata.get("runtimeVisible") is False
        and metadata.get("indexEligible") is False
    )


def _source_apply_review_summary(report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = dict(report.get("counts") or {})
    return {
        "schema": normalize_text(report.get("schema")),
        "status": normalize_text(report.get("status")),
        "decision": normalize_text(report.get("decision")),
        "reportRef": normalize_text(report_ref),
        "reviewRows": int(counts.get("reviewRows") or 0),
        "reviewReadyRows": int(counts.get("reviewReadyRows") or 0),
        "labsApplyExecutorCandidateRows": int(counts.get("labsApplyExecutorCandidateRows") or 0),
        "candidateStoreWriteRows": int(counts.get("candidateStoreWriteRows") or 0),
        "embeddingCallRows": int(counts.get("embeddingCallRows") or 0),
        "embeddingVectorWriteRows": int(counts.get("embeddingVectorWriteRows") or 0),
        "vectorIndexWriteRows": int(counts.get("vectorIndexWriteRows") or 0),
        "blockedRows": int(counts.get("blockedRows") or 0),
        "privatePathLeakRows": int(counts.get("privatePathLeakRows") or 0),
        "schemaViolationCount": int(counts.get("schemaViolationCount") or 0),
    }


def _source_apply_executor_dry_run_summary(report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = dict(report.get("counts") or {})
    return {
        "schema": normalize_text(report.get("schema")),
        "status": normalize_text(report.get("status")),
        "decision": normalize_text(report.get("decision")),
        "reportRef": normalize_text(report_ref),
        "executorDryRunRows": int(counts.get("executorDryRunRows") or 0),
        "plannedVectorUpsertRows": int(counts.get("plannedVectorUpsertRows") or 0),
        "plannedLabsNamespaceRows": int(counts.get("plannedLabsNamespaceRows") or 0),
        "embeddingInputRows": int(counts.get("embeddingInputRows") or 0),
        "candidateStoreWriteRows": int(counts.get("candidateStoreWriteRows") or 0),
        "embeddingCallRows": int(counts.get("embeddingCallRows") or 0),
        "embeddingVectorWriteRows": int(counts.get("embeddingVectorWriteRows") or 0),
        "vectorIndexWriteRows": int(counts.get("vectorIndexWriteRows") or 0),
        "blockedRows": int(counts.get("blockedRows") or 0),
        "privatePathLeakRows": int(counts.get("privatePathLeakRows") or 0),
        "schemaViolationCount": int(counts.get("schemaViolationCount") or 0),
    }


def _source_apply_review_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers: list[str] = []
    if report.get("schema") != LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_REVIEW_SCHEMA_ID:
        blockers.append("invalid_labs_vector_index_apply_review_schema")
    if report.get("status") != "ready":
        blockers.append("labs_vector_index_apply_review_not_ready")
    if report.get("decision") != APPLY_REVIEW_READY_DECISION:
        blockers.append("labs_vector_index_apply_review_invalid_decision")
    for field_name in (
        "blockedRows",
        "privatePathLeakRows",
        "schemaViolationCount",
        "candidateStoreWriteRows",
        "embeddingCallRows",
        "embeddingVectorWriteRows",
        "vectorIndexWriteRows",
        "databaseMutationRows",
        "indexMutationRows",
        "reindexOrReembedRows",
        "indexEligibleRows",
        "runtimeVisibleRows",
        "strictEvidenceRows",
        "citationGradeRows",
        "answerableWithoutTextEvidenceRows",
    ):
        if int(counts.get(field_name) or 0) != 0:
            blockers.append(f"labs_vector_index_apply_review_has_{field_name}")
    if int(counts.get("reviewReadyRows") or 0) <= 0:
        blockers.append("labs_vector_index_apply_review_has_no_ready_rows")
    return blockers


def _source_apply_executor_dry_run_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers: list[str] = []
    if report.get("schema") != LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_EXECUTOR_DRY_RUN_SCHEMA_ID:
        blockers.append("invalid_labs_vector_index_apply_executor_dry_run_schema")
    if report.get("status") != "ready":
        blockers.append("labs_vector_index_apply_executor_dry_run_not_ready")
    if report.get("decision") != APPLY_EXECUTOR_DRY_RUN_READY_DECISION:
        blockers.append("labs_vector_index_apply_executor_dry_run_invalid_decision")
    for field_name in (
        "blockedRows",
        "privatePathLeakRows",
        "schemaViolationCount",
        "candidateStoreWriteRows",
        "embeddingCallRows",
        "embeddingVectorWriteRows",
        "vectorIndexWriteRows",
        "databaseMutationRows",
        "indexMutationRows",
        "reindexOrReembedRows",
        "indexEligibleRows",
        "runtimeVisibleRows",
        "strictEvidenceRows",
        "citationGradeRows",
        "answerableWithoutTextEvidenceRows",
    ):
        if int(counts.get(field_name) or 0) != 0:
            blockers.append(f"labs_vector_index_apply_executor_dry_run_has_{field_name}")
    return blockers


def _scope(*, apply: bool, vector_rows: int) -> dict[str, Any]:
    return {
        "writes": "labs_vector_index_jsonl" if apply else "report_only",
        "apiCalls": False,
        "modelCalls": False,
        "webModelCalls": False,
        "plannedVectorRecordRows": int(vector_rows),
        "candidateStoreWriteRows": 0,
        "externalEmbeddingCallRows": 0,
        "localEmbeddingRows": int(vector_rows),
        "embeddingVectorWriteRows": int(vector_rows) if apply else 0,
        "vectorIndexWriteRows": int(vector_rows) if apply else 0,
        "productionVectorIndexWriteRows": 0,
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


def _policy(*, apply: bool) -> dict[str, Any]:
    return {
        "dryRunByDefault": True,
        "applyRequiredForLabsVectorIndexWrites": True,
        "applyMode": bool(apply),
        "labsOnly": True,
        "labsNamespace": LABS_NAMESPACE,
        "plannedVectorIndexRef": PLANNED_LABS_VECTOR_INDEX_REF,
        "labsVectorIndexWrite": False,
        "productionVectorIndexWrite": False,
        "candidateStoreWrite": False,
        "externalEmbeddingCalls": False,
        "localEmbeddingProviderRef": LOCAL_EMBEDDING_PROVIDER_REF,
        "localEmbeddingModelRef": LOCAL_EMBEDDING_MODEL_REF,
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


def _vector_record_from_upsert_record(
    *,
    review_row: dict[str, Any],
    executor_row: dict[str, Any],
    upsert_record: dict[str, Any],
) -> dict[str, Any]:
    embedding_text = str(upsert_record.get("embeddingText") or "")
    vector = _hashing_vector(embedding_text)
    return {
        "schema": LABS_VECTOR_INDEX_RECORD_SCHEMA_ID,
        "namespace": LABS_NAMESPACE,
        "plannedVectorIndexRef": PLANNED_LABS_VECTOR_INDEX_REF,
        "vectorDocumentId": normalize_text(upsert_record.get("vectorDocumentId")),
        "hintCandidateId": normalize_text(upsert_record.get("hintCandidateId")),
        "sourceCandidateId": normalize_text(upsert_record.get("sourceCandidateId")),
        "paperId": normalize_text(upsert_record.get("paperId")),
        "paperRef": normalize_text(upsert_record.get("paperRef")),
        "sourceContentHash": normalize_text(upsert_record.get("sourceContentHash")),
        "page": int(upsert_record.get("page") or 0),
        "bbox": _bbox(upsert_record.get("bbox")),
        "candidateType": normalize_text(upsert_record.get("candidateType")),
        "documentTextHash": normalize_text(upsert_record.get("documentTextHash")),
        "embeddingTextHash": normalize_text(upsert_record.get("embeddingTextHash")),
        "embeddingProviderRef": LOCAL_EMBEDDING_PROVIDER_REF,
        "embeddingModelRef": LOCAL_EMBEDDING_MODEL_REF,
        "embeddingDimensions": LOCAL_EMBEDDING_DIMENSIONS,
        "embeddingVector": vector,
        "embeddingVectorSha256": _vector_hash(vector),
        "sourceUpsertRecordSha256": _record_hash(upsert_record),
        "sourceApplyReviewRowId": normalize_text(review_row.get("applyReviewRowId")),
        "sourceExecutorDryRunRowId": normalize_text(executor_row.get("executorDryRunRowId")),
        "metadata": dict(upsert_record.get("metadata") or {}),
        "policy": dict(upsert_record.get("policy") or {}),
        "execution": {
            "externalEmbeddingCall": False,
            "localEmbeddingComputed": True,
            "labsVectorIndexWrite": False,
            "productionVectorIndexWrite": False,
        },
    }


def _vector_record_preview(record: dict[str, Any]) -> dict[str, Any]:
    vector = list(record.get("embeddingVector") or [])
    return {
        key: value
        for key, value in record.items()
        if key != "embeddingVector"
    } | {
        "embeddingVectorPresent": bool(vector),
        "embeddingVectorLength": len(vector),
    }


def _vector_record_checks(
    *,
    review_row: dict[str, Any],
    executor_row: dict[str, Any] | None,
    upsert_record: dict[str, Any] | None,
    vector_record: dict[str, Any] | None,
) -> dict[str, bool]:
    metadata = dict((upsert_record or {}).get("metadata") or {})
    policy = dict((upsert_record or {}).get("policy") or {})
    return {
        "applyReviewRowReady": normalize_text(review_row.get("applyReviewStatus")) == APPLY_REVIEW_STATUS_READY,
        "applyReviewExecutorCandidate": review_row.get("labsApplyExecutorCandidate") is True,
        "executorRowPresent": bool(executor_row),
        "upsertRecordPresent": bool(upsert_record),
        "upsertRecordHashMatchesReview": bool(upsert_record)
        and _record_hash(upsert_record) == normalize_text(review_row.get("plannedVectorUpsertRecordSha256")),
        "upsertRecordHashMatchesExecutor": bool(upsert_record)
        and bool(executor_row)
        and _record_hash(upsert_record) == normalize_text((executor_row or {}).get("plannedVectorUpsertRecordSha256")),
        "namespaceMatchesLabs": normalize_text((upsert_record or {}).get("namespace")) == LABS_NAMESPACE,
        "plannedVectorIndexRefMatches": normalize_text((upsert_record or {}).get("plannedVectorIndexRef"))
        == PLANNED_LABS_VECTOR_INDEX_REF,
        "vectorDocumentIdMatches": normalize_text((upsert_record or {}).get("vectorDocumentId"))
        == normalize_text(review_row.get("vectorDocumentId")),
        "hintCandidateIdMatches": normalize_text((upsert_record or {}).get("hintCandidateId"))
        == normalize_text(review_row.get("hintCandidateId")),
        "sourceCandidateIdMatches": normalize_text((upsert_record or {}).get("sourceCandidateId"))
        == normalize_text(review_row.get("sourceCandidateId")),
        "sourceContentHashMatches": normalize_text((upsert_record or {}).get("sourceContentHash"))
        == normalize_text(review_row.get("sourceContentHash")),
        "pageMatches": int((upsert_record or {}).get("page") or 0) == int(review_row.get("page") or 0),
        "bboxMatches": _bbox((upsert_record or {}).get("bbox")) == _bbox(review_row.get("bbox")),
        "candidateTypeMatches": normalize_text((upsert_record or {}).get("candidateType"))
        == normalize_text(review_row.get("candidateType")),
        "documentTextHashMatches": normalize_text((upsert_record or {}).get("documentTextHash"))
        == normalize_text(review_row.get("documentTextHash")),
        "embeddingTextHashMatches": normalize_text((upsert_record or {}).get("embeddingTextHash"))
        == normalize_text(review_row.get("embeddingTextHash")),
        "metadataPolicyRetrievalHintOnly": _metadata_policy_ok(metadata),
        "upsertPolicyRetrievalHintOnly": _policy_flags_ok(policy),
        "upsertRecordNotPrivatePathLeaking": bool(upsert_record) and not _contains_private_path(upsert_record),
        "vectorRecordPresent": bool(vector_record),
        "vectorRecordSchemaMatches": normalize_text((vector_record or {}).get("schema")) == LABS_VECTOR_INDEX_RECORD_SCHEMA_ID,
        "vectorRecordHasExpectedDimensions": len(list((vector_record or {}).get("embeddingVector") or []))
        == LOCAL_EMBEDDING_DIMENSIONS,
        "vectorRecordHashMatchesVector": bool(vector_record)
        and normalize_text((vector_record or {}).get("embeddingVectorSha256"))
        == _vector_hash(list((vector_record or {}).get("embeddingVector") or [])),
        "vectorRecordPolicyRetrievalHintOnly": bool(vector_record)
        and _policy_flags_ok(dict((vector_record or {}).get("policy") or {})),
        "vectorRecordNotPrivatePathLeaking": bool(vector_record) and not _contains_private_path(vector_record),
    }


def _row_status(
    *,
    review_row: dict[str, Any],
    upsert_record: dict[str, Any] | None,
    blockers: list[str],
) -> str:
    if normalize_text(review_row.get("applyReviewStatus")) != APPLY_REVIEW_STATUS_READY:
        return EXECUTOR_STATUS_BLOCKED_NON_READY_APPLY_REVIEW_ROW
    if not upsert_record:
        return EXECUTOR_STATUS_BLOCKED_MISSING_UPSERT_RECORD
    policy_blockers = {
        "metadataPolicyRetrievalHintOnly",
        "upsertPolicyRetrievalHintOnly",
        "upsertRecordNotPrivatePathLeaking",
        "vectorRecordPolicyRetrievalHintOnly",
        "vectorRecordNotPrivatePathLeaking",
    }
    if any(blocker in policy_blockers for blocker in blockers):
        return EXECUTOR_STATUS_BLOCKED_POLICY_VIOLATION
    if blockers:
        return EXECUTOR_STATUS_BLOCKED_VECTOR_RECORD_CONTRACT
    return EXECUTOR_STATUS_DRY_RUN_READY


def _write_jsonl_idempotent(path: Path, records: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming_by_key = {_vector_record_key(record): record for record in records}
    retained = [record for record in _read_jsonl(path) if _vector_record_key(record) not in incoming_by_key]
    output = retained + list(incoming_by_key.values())
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in output),
        encoding="utf-8",
    )
    return len(incoming_by_key)


def _apply_records(
    records: list[dict[str, Any]],
    *,
    papers_dir: str | Path,
) -> tuple[int, int, list[str]]:
    path = _index_path(papers_dir)
    applied_rows = _write_jsonl_idempotent(path, records)
    readback_by_key = {_vector_record_key(record): record for record in _read_jsonl(path)}
    readback_rows = 0
    warnings: list[str] = []
    for record in records:
        key = _vector_record_key(record)
        stored = readback_by_key.get(key)
        if stored == record and _record_hash(stored) == _record_hash(record):
            readback_rows += 1
        else:
            warnings.append(f"readback_mismatch:{normalize_text(record.get('vectorDocumentId'))}")
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
        )
        if key in report
    }


def execute_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor(
    *,
    labs_vector_index_apply_review: dict[str, Any],
    labs_vector_index_apply_executor_dry_run: dict[str, Any],
    source_labs_vector_index_apply_review_ref: str,
    source_labs_vector_index_apply_executor_dry_run_ref: str,
    papers_dir: str | Path | None = None,
    run_id: str | None = None,
    apply: bool = False,
    generated_at: str | None = None,
) -> dict[str, Any]:
    run_id = normalize_text(run_id) or f"visual-retrieval-hint-labs-vector-index-apply-{utc_now_iso()}"
    source_blockers = sorted(
        set(
            _source_apply_review_blockers(labs_vector_index_apply_review)
            + _source_apply_executor_dry_run_blockers(labs_vector_index_apply_executor_dry_run)
        )
    )
    warnings: list[str] = []
    schema_violations: list[str] = []
    if apply and not papers_dir:
        warnings.append("apply_requires_papers_dir")
        schema_violations.append("apply_requires_papers_dir")

    review_rows = [
        dict(row) for row in labs_vector_index_apply_review.get("applyReviewRowsDetail") or [] if isinstance(row, dict)
    ]
    executor_rows_by_id = {
        normalize_text(row.get("vectorDocumentId")): dict(row)
        for row in labs_vector_index_apply_executor_dry_run.get("executorDryRunRowsDetail") or []
        if isinstance(row, dict)
    }
    upsert_records_by_id = {
        normalize_text(row.get("vectorDocumentId")): dict(row)
        for row in labs_vector_index_apply_executor_dry_run.get("plannedVectorUpsertRecords") or []
        if isinstance(row, dict)
    }
    rows: list[dict[str, Any]] = []
    vector_records: list[dict[str, Any]] = []
    for index, review_row in enumerate(review_rows, start=1):
        vector_doc_id = normalize_text(review_row.get("vectorDocumentId"))
        executor_row = executor_rows_by_id.get(vector_doc_id)
        upsert_record = upsert_records_by_id.get(vector_doc_id)
        vector_record = (
            _vector_record_from_upsert_record(
                review_row=review_row,
                executor_row=executor_row or {},
                upsert_record=upsert_record,
            )
            if upsert_record
            else None
        )
        checks = _vector_record_checks(
            review_row=review_row,
            executor_row=executor_row,
            upsert_record=upsert_record,
            vector_record=vector_record,
        )
        blockers = [name for name, passed in checks.items() if not passed]
        status = _row_status(review_row=review_row, upsert_record=upsert_record, blockers=blockers)
        if vector_record and status == EXECUTOR_STATUS_DRY_RUN_READY:
            vector_records.append(vector_record)
        rows.append(
            {
                "executorRowId": (
                    "limited-visual-retrieval-hint-candidate-store-labs-vector-index-apply-executor:"
                    f"{index:04d}"
                ),
                "sourceApplyReviewRowId": normalize_text(review_row.get("applyReviewRowId")),
                "sourceExecutorDryRunRowId": normalize_text((executor_row or {}).get("executorDryRunRowId")),
                "vectorDocumentId": vector_doc_id,
                "namespace": LABS_NAMESPACE if upsert_record else "",
                "hintCandidateId": normalize_text(review_row.get("hintCandidateId")),
                "sourceCandidateId": normalize_text(review_row.get("sourceCandidateId")),
                "paperId": normalize_text(review_row.get("paperId")),
                "paperRef": normalize_text(review_row.get("paperRef")),
                "sourceContentHash": normalize_text(review_row.get("sourceContentHash")),
                "page": int(review_row.get("page") or 0),
                "bbox": _bbox(review_row.get("bbox")),
                "candidateType": normalize_text(review_row.get("candidateType")),
                "documentTextHash": normalize_text(review_row.get("documentTextHash")),
                "embeddingTextHash": normalize_text(review_row.get("embeddingTextHash")),
                "sourceUpsertRecordSha256": _record_hash(upsert_record) if upsert_record else "",
                "vectorRecordSha256": _record_hash(vector_record) if vector_record else "",
                "embeddingProviderRef": LOCAL_EMBEDDING_PROVIDER_REF if vector_record else "",
                "embeddingModelRef": LOCAL_EMBEDDING_MODEL_REF if vector_record else "",
                "embeddingDimensions": LOCAL_EMBEDDING_DIMENSIONS if vector_record else 0,
                "labsVectorIndexRef": _index_ref() if vector_record else "",
                "wouldWriteLabsVectorRecord": bool(vector_record) and not apply,
                "appliedLabsVectorRecord": False,
                "readbackValidated": False,
                "indexEligible": False,
                "runtimeVisible": False,
                "strictEvidence": False,
                "citationGrade": False,
                "answerableWithoutTextEvidence": False,
                "executionStatus": status,
                "executionBlockers": sorted(set(blockers)),
                "checks": checks,
            }
        )

    if source_blockers:
        schema_violations.extend(source_blockers)
    if not review_rows:
        warnings.append("apply_review_rows_missing")
        schema_violations.append("apply_review_rows_missing")
    if not vector_records:
        warnings.append("labs_vector_records_missing")
        schema_violations.append("labs_vector_records_missing")
    if int(dict(labs_vector_index_apply_review.get("counts") or {}).get("reviewReadyRows") or 0) != len(vector_records):
        schema_violations.append("review_ready_count_does_not_match_vector_records")

    applied_rows = 0
    readback_rows = 0
    if apply and vector_records and not schema_violations and papers_dir:
        applied_records = []
        for record in vector_records:
            applied = dict(record)
            applied["execution"] = dict(applied.get("execution") or {})
            applied["execution"]["labsVectorIndexWrite"] = True
            applied_records.append(applied)
        applied_rows, readback_rows, readback_warnings = _apply_records(applied_records, papers_dir=papers_dir)
        warnings.extend(readback_warnings)
        if readback_rows != len(applied_records):
            schema_violations.append("apply_readback_incomplete")
        applied_by_id = {normalize_text(record.get("vectorDocumentId")): record for record in applied_records}
        vector_records = applied_records
        for row in rows:
            if row["executionStatus"] == EXECUTOR_STATUS_DRY_RUN_READY:
                matched = not schema_violations and normalize_text(row.get("vectorDocumentId")) in applied_by_id
                row["wouldWriteLabsVectorRecord"] = False
                row["appliedLabsVectorRecord"] = matched
                row["readbackValidated"] = matched
                row["executionStatus"] = EXECUTOR_STATUS_APPLIED if matched else EXECUTOR_STATUS_BLOCKED_READBACK_MISMATCH
                if not matched:
                    row["executionBlockers"] = ["readback_mismatch"]

    private_path_leak_rows = 1 if _contains_private_path(rows) or _contains_private_path(vector_records) else 0
    if private_path_leak_rows:
        schema_violations.append("private_path_leak")
    schema_violations = sorted(set(schema_violations))
    blocked_rows = sum(1 for row in rows if not str(row["executionStatus"]).startswith(("dry_run_ready", "applied")))
    counts = {
        "sourceApplyReviewRows": len(review_rows),
        "sourceReviewReadyRows": int(dict(labs_vector_index_apply_review.get("counts") or {}).get("reviewReadyRows") or 0),
        "sourcePlannedVectorUpsertRows": int(
            dict(labs_vector_index_apply_executor_dry_run.get("counts") or {}).get("plannedVectorUpsertRows") or 0
        ),
        "executorRows": len(rows),
        "plannedVectorRecordRows": len(vector_records),
        "dryRunVectorRecordRows": 0 if apply else len(vector_records),
        "appliedLabsVectorRecordRows": applied_rows if apply else 0,
        "readbackValidatedRows": readback_rows if apply else 0,
        "runManifestWriteRows": 0,
        "candidateStoreWriteRows": 0,
        "externalEmbeddingCallRows": 0,
        "embeddingCallRows": 0,
        "localEmbeddingRows": len(vector_records),
        "embeddingVectorWriteRows": applied_rows if apply else 0,
        "vectorIndexWriteRows": applied_rows if apply else 0,
        "productionVectorIndexWriteRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "reindexOrReembedRows": 0,
        "indexEligibleRows": 0,
        "runtimeVisibleRows": 0,
        "strictEvidenceRows": 0,
        "citationGradeRows": 0,
        "answerableWithoutTextEvidenceRows": 0,
        "blockedRows": blocked_rows,
        "blockedNonReadyApplyReviewRows": sum(
            1 for row in rows if row["executionStatus"] == EXECUTOR_STATUS_BLOCKED_NON_READY_APPLY_REVIEW_ROW
        ),
        "blockedMissingUpsertRecordRows": sum(
            1 for row in rows if row["executionStatus"] == EXECUTOR_STATUS_BLOCKED_MISSING_UPSERT_RECORD
        ),
        "blockedVectorRecordContractRows": sum(
            1 for row in rows if row["executionStatus"] == EXECUTOR_STATUS_BLOCKED_VECTOR_RECORD_CONTRACT
        ),
        "blockedPolicyViolationRows": sum(
            1 for row in rows if row["executionStatus"] == EXECUTOR_STATUS_BLOCKED_POLICY_VIOLATION
        ),
        "blockedReadbackMismatchRows": sum(
            1 for row in rows if row["executionStatus"] == EXECUTOR_STATUS_BLOCKED_READBACK_MISMATCH
        ),
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(schema_violations),
        "byCandidateType": dict(Counter(row["candidateType"] for row in rows)),
        "byExecutionStatus": dict(Counter(row["executionStatus"] for row in rows)),
    }
    status = "blocked" if schema_violations or blocked_rows else ("applied" if apply else "ready")
    decision = BLOCKED_DECISION if status == "blocked" else (APPLIED_DECISION if apply else READY_DECISION)
    report: dict[str, Any] = {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_EXECUTOR_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": decision,
        "nextRecommendedTranche": NEXT_TRANCHE_APPLIED if apply and status != "blocked" else NEXT_TRANCHE_DRY_RUN,
        "sourceLabsVectorIndexApplyReview": _source_apply_review_summary(
            labs_vector_index_apply_review,
            report_ref=source_labs_vector_index_apply_review_ref,
        ),
        "sourceLabsVectorIndexApplyExecutorDryRun": _source_apply_executor_dry_run_summary(
            labs_vector_index_apply_executor_dry_run,
            report_ref=source_labs_vector_index_apply_executor_dry_run_ref,
        ),
        "input": {
            "apply": bool(apply),
            "papersDirRef": "papers_dir" if papers_dir else "",
            "runId": run_id,
            "labsNamespace": LABS_NAMESPACE,
            "labsVectorIndexRef": _index_ref(),
            "runManifestRef": _run_manifest_ref(run_id),
        },
        "scope": _scope(apply=apply, vector_rows=len(vector_records)),
        "policy": _policy(apply=apply),
        "method": {
            "name": "labs_visual_retrieval_hint_vector_index_apply_executor_v1",
            "description": (
                "Builds deterministic local lexical-hashing vectors for reviewed visual retrieval hints and "
                "writes them only to the labs JSONL vector index under explicit apply mode."
            ),
            "limitations": [
                "This executor does not call external embedding providers or model APIs.",
                "The labs JSONL index is not the production vector DB and is not runtime-answer-visible.",
                "Passing this executor should be followed by readback/search review, not automatic production indexing.",
            ],
        },
        "counts": counts,
        "gate": {
            "readyForLabsVectorIndexApply": status == "ready",
            "applyMode": bool(apply),
            "labsVectorIndexWriteAllowed": bool(apply and papers_dir and not schema_violations),
            "candidateStoreWriteAllowed": False,
            "externalEmbeddingCallsAllowed": False,
            "productionVectorIndexingAllowed": False,
            "runtimeVisibilityAllowed": False,
            "evidencePromotionAllowed": False,
            "schemaViolations": schema_violations,
        },
        "rows": rows,
        "labsVectorRecordPreviews": [_vector_record_preview(record) for record in vector_records],
        "warnings": sorted(set(warnings)),
    }
    report["policy"]["labsVectorIndexWrite"] = bool(counts["vectorIndexWriteRows"])

    if apply and status != "blocked" and papers_dir:
        manifest_path = _run_manifest_path(papers_dir, run_id)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        report["counts"]["runManifestWriteRows"] = 1
        manifest_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Limited Visual Retrieval Hint Candidate Store Labs Vector Index Apply Executor",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- apply: `{dict(report.get('input') or {}).get('apply')}`",
        f"- plannedVectorRecordRows: `{counts.get('plannedVectorRecordRows')}`",
        f"- dryRunVectorRecordRows: `{counts.get('dryRunVectorRecordRows')}`",
        f"- appliedLabsVectorRecordRows: `{counts.get('appliedLabsVectorRecordRows')}`",
        f"- readbackValidatedRows: `{counts.get('readbackValidatedRows')}`",
        f"- localEmbeddingRows: `{counts.get('localEmbeddingRows')}`",
        f"- vectorIndexWriteRows: `{counts.get('vectorIndexWriteRows')}`",
        f"- productionVectorIndexWriteRows: `{counts.get('productionVectorIndexWriteRows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Mutation Guarantees",
        "",
        f"- labsVectorIndexRef: `{dict(report.get('input') or {}).get('labsVectorIndexRef')}`",
        f"- candidateStoreWriteRows: `{counts.get('candidateStoreWriteRows')}`",
        f"- externalEmbeddingCallRows: `{counts.get('externalEmbeddingCallRows')}`",
        f"- databaseMutationRows: `{counts.get('databaseMutationRows')}`",
        f"- runtimeVisibleRows: `{counts.get('runtimeVisibleRows')}`",
        f"- strictEvidenceRows: `{counts.get('strictEvidenceRows')}`",
        f"- citationGradeRows: `{counts.get('citationGradeRows')}`",
        "",
        "## Candidate Types",
        "",
    ]
    for candidate_type, count in sorted(dict(counts.get("byCandidateType") or {}).items()):
        lines.append(f"- `{candidate_type}`: `{count}`")
    lines.extend(["", "## Execution Status", ""])
    for execution_status, count in sorted(dict(counts.get("byExecutionStatus") or {}).items()):
        lines.append(f"- `{execution_status}`: `{count}`")
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report.get("warnings", []):
            lines.append(f"- `{warning}`")
    return "\n".join(lines).rstrip() + "\n"


def write_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor(
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
    "EXECUTOR_STATUS_BLOCKED_MISSING_UPSERT_RECORD",
    "EXECUTOR_STATUS_BLOCKED_NON_READY_APPLY_REVIEW_ROW",
    "EXECUTOR_STATUS_BLOCKED_POLICY_VIOLATION",
    "EXECUTOR_STATUS_BLOCKED_READBACK_MISMATCH",
    "EXECUTOR_STATUS_BLOCKED_VECTOR_RECORD_CONTRACT",
    "EXECUTOR_STATUS_DRY_RUN_READY",
    "LABS_VECTOR_INDEX_RECORD_SCHEMA_ID",
    "LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_EXECUTOR_SCHEMA_ID",
    "execute_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor",
    "load_json",
    "sanitized_report_ref",
    "write_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor",
]
