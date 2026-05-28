"""Labs-only search quality eval for applied visual retrieval-hint vectors.

This helper reads the actual labs JSONL vector index, validates it against the
source apply reports, and compares deterministic text-only retrieval with a
hybrid that can also hit the labs vector records. It is report-only: it never
queries or mutates the production vector DB, candidate store, runtime route, or
evidence layer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_apply_executor import (
    _contains_private_path,
    normalize_text,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor import (
    LABS_VECTOR_INDEX_RECORD_SCHEMA_ID,
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_EXECUTOR_SCHEMA_ID,
    LOCAL_EMBEDDING_DIMENSIONS,
    LOCAL_EMBEDDING_MODEL_REF,
    LOCAL_EMBEDDING_PROVIDER_REF,
    _hashing_vector,
    _vector_hash,
    load_json,
    sanitized_report_ref,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor_dry_run import (
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_EXECUTOR_DRY_RUN_SCHEMA_ID,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_dry_run import (
    LABS_NAMESPACE,
)
from knowledge_hub.papers.visual_retrieval_hint_search_eval import (
    VISUAL_LAYOUT_CANDIDATE_LIST_REPORT_SCHEMA_ID,
)
from knowledge_hub.papers.visual_retrieval_hint_usefulness_eval import (
    _rank_documents,
    _text_context,
    _tokens,
    clear_rank_index_cache,
)


LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_SEARCH_QUALITY_EVAL_SCHEMA_ID = (
    "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-labs-vector-index-search-quality-eval.v1"
)
LABS_VECTOR_INDEX_SEARCH_QUALITY_QUERY_ROW_SCHEMA_ID = (
    "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-labs-vector-index-search-quality-query-row.v1"
)

READY_DECISION = "ready_for_limited_visual_retrieval_hint_production_vector_db_integration_design"
KEEP_IN_LABS_DECISION = "keep_in_labs_pending_search_quality_review"
BLOCKED_DECISION = "blocked"
NEXT_TRANCHE_READY = "limited_visual_retrieval_hint_production_vector_db_integration_design"
NEXT_TRANCHE_HOLD = "limited_visual_retrieval_hint_labs_vector_index_search_quality_review"

DEFAULT_MIN_LABS_HIT_AT5_ROWS = 200
DEFAULT_MIN_HYBRID_HIT_AT5_LIFT_ROWS = 25

STOP_TOKENS = {
    "a",
    "about",
    "allowed_use",
    "and",
    "are",
    "as",
    "by",
    "candidate",
    "figure",
    "find",
    "for",
    "from",
    "hint",
    "image",
    "in",
    "is",
    "layout",
    "only",
    "page",
    "paper",
    "region",
    "retrieval",
    "table",
    "the",
    "to",
    "type",
    "visual",
    "with",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _short_hash(value: str, *, length: int = 24) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _index_path(papers_dir: str | Path) -> Path:
    return (
        Path(str(papers_dir)).expanduser()
        / "visual_retrieval_hints"
        / "labs_vector_index"
        / f"{LABS_NAMESPACE}.v1.jsonl"
    )


def _index_ref() -> str:
    return f"papers_dir/visual_retrieval_hints/labs_vector_index/{LABS_NAMESPACE}.v1.jsonl"


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


def _source_apply_summary(report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = dict(report.get("counts") or {})
    return {
        "schema": normalize_text(report.get("schema")),
        "status": normalize_text(report.get("status")),
        "decision": normalize_text(report.get("decision")),
        "reportRef": normalize_text(report_ref),
        "appliedLabsVectorRecordRows": int(counts.get("appliedLabsVectorRecordRows") or 0),
        "readbackValidatedRows": int(counts.get("readbackValidatedRows") or 0),
        "vectorIndexWriteRows": int(counts.get("vectorIndexWriteRows") or 0),
        "productionVectorIndexWriteRows": int(counts.get("productionVectorIndexWriteRows") or 0),
        "candidateStoreWriteRows": int(counts.get("candidateStoreWriteRows") or 0),
        "blockedRows": int(counts.get("blockedRows") or 0),
        "privatePathLeakRows": int(counts.get("privatePathLeakRows") or 0),
        "schemaViolationCount": int(counts.get("schemaViolationCount") or 0),
    }


def _source_dry_run_summary(report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = dict(report.get("counts") or {})
    return {
        "schema": normalize_text(report.get("schema")),
        "status": normalize_text(report.get("status")),
        "decision": normalize_text(report.get("decision")),
        "reportRef": normalize_text(report_ref),
        "executorDryRunRows": int(counts.get("executorDryRunRows") or 0),
        "plannedVectorUpsertRows": int(counts.get("plannedVectorUpsertRows") or 0),
        "embeddingInputRows": int(counts.get("embeddingInputRows") or 0),
        "candidateStoreWriteRows": int(counts.get("candidateStoreWriteRows") or 0),
        "vectorIndexWriteRows": int(counts.get("vectorIndexWriteRows") or 0),
        "blockedRows": int(counts.get("blockedRows") or 0),
        "privatePathLeakRows": int(counts.get("privatePathLeakRows") or 0),
        "schemaViolationCount": int(counts.get("schemaViolationCount") or 0),
    }


def _source_apply_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers: list[str] = []
    if report.get("schema") != LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_EXECUTOR_SCHEMA_ID:
        blockers.append("invalid_labs_vector_index_apply_executor_schema")
    if report.get("status") != "applied":
        blockers.append("labs_vector_index_apply_executor_not_applied")
    if report.get("decision") != "applied_limited_visual_retrieval_hint_candidate_store_labs_vector_index":
        blockers.append("labs_vector_index_apply_executor_invalid_decision")
    applied_rows = int(counts.get("appliedLabsVectorRecordRows") or 0)
    readback_rows = int(counts.get("readbackValidatedRows") or 0)
    vector_write_rows = int(counts.get("vectorIndexWriteRows") or 0)
    if not applied_rows:
        blockers.append("labs_vector_index_apply_executor_has_no_applied_rows")
    if applied_rows != readback_rows or applied_rows != vector_write_rows:
        blockers.append("labs_vector_index_apply_executor_count_mismatch")
    for field_name in (
        "productionVectorIndexWriteRows",
        "candidateStoreWriteRows",
        "externalEmbeddingCallRows",
        "databaseMutationRows",
        "indexMutationRows",
        "reindexOrReembedRows",
        "indexEligibleRows",
        "runtimeVisibleRows",
        "strictEvidenceRows",
        "citationGradeRows",
        "answerableWithoutTextEvidenceRows",
        "blockedRows",
        "privatePathLeakRows",
        "schemaViolationCount",
    ):
        if int(counts.get(field_name) or 0) != 0:
            blockers.append(f"labs_vector_index_apply_executor_has_{field_name}")
    return blockers


def _source_dry_run_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers: list[str] = []
    if report.get("schema") != LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_EXECUTOR_DRY_RUN_SCHEMA_ID:
        blockers.append("invalid_labs_vector_index_apply_executor_dry_run_schema")
    if report.get("status") != "ready":
        blockers.append("labs_vector_index_apply_executor_dry_run_not_ready")
    if not list(report.get("plannedVectorUpsertRecords") or []):
        blockers.append("labs_vector_index_apply_executor_dry_run_missing_upsert_records")
    for field_name in (
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
        "blockedRows",
        "privatePathLeakRows",
        "schemaViolationCount",
    ):
        if int(counts.get(field_name) or 0) != 0:
            blockers.append(f"labs_vector_index_apply_executor_dry_run_has_{field_name}")
    return blockers


def _scope(*, query_rows: int, index_rows: int) -> dict[str, Any]:
    return {
        "writes": "report_only",
        "apiCalls": False,
        "modelCalls": False,
        "webModelCalls": False,
        "actualLabsVectorIndexReadRows": int(index_rows),
        "queryRows": int(query_rows),
        "candidateStoreWriteRows": 0,
        "embeddingCallRows": 0,
        "embeddingVectorWriteRows": 0,
        "vectorIndexWriteRows": 0,
        "productionVectorIndexWriteRows": 0,
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
        "evalOnly": True,
        "labsOnly": True,
        "usesActualLabsJsonlIndex": True,
        "candidateStoreWrite": False,
        "embeddingCalls": False,
        "vectorIndexWrite": False,
        "productionVectorIndexWrite": False,
        "operationalSearchIndexQuery": False,
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


def _policy_flags_ok(value: dict[str, Any]) -> bool:
    return (
        normalize_text(value.get("allowedUse")) == "retrieval_hint_only"
        and value.get("strictEvidence") is False
        and value.get("citationGrade") is False
        and value.get("answerableWithoutTextEvidence") is False
        and value.get("runtimeVisible") is False
        and value.get("indexEligible") is False
    )


def _bbox(value: Any) -> list[Any]:
    return list(value or [])


def _tokens_for_query(text: str, *, limit: int = 12) -> list[str]:
    output: list[str] = []
    for token in _tokens(normalize_text(text)):
        if token in STOP_TOKENS or len(token) < 2:
            continue
        if token not in output:
            output.append(token)
        if len(output) >= limit:
            break
    return output


def _query_specs(upsert_record: dict[str, Any]) -> list[dict[str, str]]:
    document_text = normalize_text(upsert_record.get("documentText"))
    embedding_text = normalize_text(upsert_record.get("embeddingText"))
    tokens = _tokens_for_query(document_text, limit=12)
    if len(tokens) < 4:
        tokens = _tokens_for_query(embedding_text, limit=12)
    natural_terms = ", ".join(tokens[:6])
    paper_id = normalize_text(upsert_record.get("paperId")).replace("-", " ")
    candidate_type = normalize_text(upsert_record.get("candidateType")).replace("_", " ")
    return [
        {"queryKind": "keyword_lookup", "query": " ".join(tokens[:12])},
        {"queryKind": "natural_lookup", "query": f"find {paper_id} {candidate_type} about {natural_terms}"},
    ]


def _text_rank(
    *,
    documents: dict[str, str],
    query: str,
    source_candidate_id: str,
) -> tuple[int | None, float]:
    return _rank_documents(documents=documents, query=query, target_id=source_candidate_id)


def _vector_rank(
    *,
    vector_records: dict[str, dict[str, Any]],
    query: str,
    target_vector_document_id: str,
) -> tuple[int | None, float]:
    query_vector = _hashing_vector(query)
    scored: list[tuple[str, float]] = []
    for vector_doc_id, record in vector_records.items():
        vector = list(record.get("embeddingVector") or [])
        score = sum(float(left) * float(right) for left, right in zip(query_vector, vector))
        scored.append((vector_doc_id, score))
    scored.sort(key=lambda item: (-item[1], item[0]))
    for index, (vector_doc_id, score) in enumerate(scored, start=1):
        if vector_doc_id == target_vector_document_id:
            return index, score
    return None, 0.0


def _rank_value(rank: int | None, *, missing_rank: int) -> int:
    return rank if rank is not None else missing_rank


def _search_result(rank: int | None, score: float) -> dict[str, Any]:
    return {
        "rank": rank,
        "score": round(float(score), 6),
        "hitAt1": bool(rank is not None and rank <= 1),
        "hitAt5": bool(rank is not None and rank <= 5),
        "hitAt10": bool(rank is not None and rank <= 10),
    }


def _mrr(rows: list[dict[str, Any]], *, result_key: str) -> float:
    if not rows:
        return 0.0
    total = 0.0
    for row in rows:
        rank = dict(row.get(result_key) or {}).get("rank")
        if isinstance(rank, int) and rank > 0:
            total += 1.0 / rank
    return round(total / len(rows), 6)


def _lift_bucket(*, text_rank: int | None, hybrid_rank: int | None, rank_delta: int) -> str:
    if hybrid_rank is None:
        return "blocked"
    if hybrid_rank <= 5 and (text_rank is None or text_rank > 10):
        return "strong_lift"
    if rank_delta > 0:
        return "moderate_lift"
    if rank_delta == 0:
        return "no_lift"
    return "regression"


def _record_checks(
    *,
    vector_record: dict[str, Any] | None,
    upsert_record: dict[str, Any],
) -> dict[str, bool]:
    metadata = dict((vector_record or {}).get("metadata") or {})
    policy = dict((vector_record or {}).get("policy") or {})
    execution = dict((vector_record or {}).get("execution") or {})
    expected_vector = _hashing_vector(str(upsert_record.get("embeddingText") or ""))
    return {
        "vectorRecordPresent": bool(vector_record),
        "schemaMatches": normalize_text((vector_record or {}).get("schema")) == LABS_VECTOR_INDEX_RECORD_SCHEMA_ID,
        "namespaceMatchesLabs": normalize_text((vector_record or {}).get("namespace")) == LABS_NAMESPACE,
        "vectorDocumentIdMatches": normalize_text((vector_record or {}).get("vectorDocumentId"))
        == normalize_text(upsert_record.get("vectorDocumentId")),
        "hintCandidateIdMatches": normalize_text((vector_record or {}).get("hintCandidateId"))
        == normalize_text(upsert_record.get("hintCandidateId")),
        "sourceCandidateIdMatches": normalize_text((vector_record or {}).get("sourceCandidateId"))
        == normalize_text(upsert_record.get("sourceCandidateId")),
        "sourceContentHashMatches": normalize_text((vector_record or {}).get("sourceContentHash"))
        == normalize_text(upsert_record.get("sourceContentHash")),
        "pageMatches": int((vector_record or {}).get("page") or 0) == int(upsert_record.get("page") or 0),
        "bboxMatches": _bbox((vector_record or {}).get("bbox")) == _bbox(upsert_record.get("bbox")),
        "candidateTypeMatches": normalize_text((vector_record or {}).get("candidateType"))
        == normalize_text(upsert_record.get("candidateType")),
        "documentTextHashMatches": normalize_text((vector_record or {}).get("documentTextHash"))
        == normalize_text(upsert_record.get("documentTextHash")),
        "embeddingTextHashMatches": normalize_text((vector_record or {}).get("embeddingTextHash"))
        == normalize_text(upsert_record.get("embeddingTextHash")),
        "embeddingProviderMatches": normalize_text((vector_record or {}).get("embeddingProviderRef"))
        == LOCAL_EMBEDDING_PROVIDER_REF,
        "embeddingModelMatches": normalize_text((vector_record or {}).get("embeddingModelRef")) == LOCAL_EMBEDDING_MODEL_REF,
        "embeddingDimensionsMatch": int((vector_record or {}).get("embeddingDimensions") or 0) == LOCAL_EMBEDDING_DIMENSIONS,
        "embeddingVectorMatchesSourceText": list((vector_record or {}).get("embeddingVector") or []) == expected_vector,
        "embeddingVectorHashMatches": bool(vector_record)
        and normalize_text((vector_record or {}).get("embeddingVectorSha256"))
        == _vector_hash(list((vector_record or {}).get("embeddingVector") or [])),
        "metadataPolicyRetrievalHintOnly": _policy_flags_ok(metadata),
        "rowPolicyRetrievalHintOnly": _policy_flags_ok(policy),
        "executionUsesLocalEmbeddingOnly": execution.get("localEmbeddingComputed") is True
        and execution.get("externalEmbeddingCall") is False,
        "executionLabsWriteOnly": execution.get("labsVectorIndexWrite") is True
        and execution.get("productionVectorIndexWrite") is False,
        "notPrivatePathLeaking": bool(vector_record) and not _contains_private_path(vector_record),
    }


def _query_row(
    *,
    upsert_record: dict[str, Any],
    vector_record: dict[str, Any],
    query_kind: str,
    query: str,
    text_documents: dict[str, str],
    vector_records: dict[str, dict[str, Any]],
    missing_rank: int,
) -> dict[str, Any]:
    source_candidate_id = normalize_text(upsert_record.get("sourceCandidateId"))
    vector_document_id = normalize_text(upsert_record.get("vectorDocumentId"))
    text_rank, text_score = _text_rank(
        documents=text_documents,
        query=query,
        source_candidate_id=source_candidate_id,
    )
    vector_rank, vector_score = _vector_rank(
        vector_records=vector_records,
        query=query,
        target_vector_document_id=vector_document_id,
    )
    text_rank_value = _rank_value(text_rank, missing_rank=missing_rank)
    vector_rank_value = _rank_value(vector_rank, missing_rank=missing_rank)
    hybrid_rank_value = min(text_rank_value, vector_rank_value)
    hybrid_rank = hybrid_rank_value if hybrid_rank_value < missing_rank else None
    rank_delta = text_rank_value - hybrid_rank_value
    row_id_basis = "|".join([vector_document_id, source_candidate_id, query_kind, query])
    return {
        "schema": LABS_VECTOR_INDEX_SEARCH_QUALITY_QUERY_ROW_SCHEMA_ID,
        "queryRowId": "limited-visual-retrieval-hint-labs-vector-search-quality-query:"
        + _short_hash(row_id_basis),
        "queryKind": query_kind,
        "query": query,
        "targetVectorDocumentId": vector_document_id,
        "hintCandidateId": normalize_text(upsert_record.get("hintCandidateId")),
        "sourceCandidateId": source_candidate_id,
        "paperId": normalize_text(upsert_record.get("paperId")),
        "paperRef": normalize_text(upsert_record.get("paperRef")),
        "sourceContentHash": normalize_text(upsert_record.get("sourceContentHash")),
        "page": int(upsert_record.get("page") or 0),
        "bbox": _bbox(upsert_record.get("bbox")),
        "candidateType": normalize_text(upsert_record.get("candidateType")),
        "embeddingTextHash": normalize_text(upsert_record.get("embeddingTextHash")),
        "textOnlyResult": _search_result(text_rank, text_score),
        "labsVectorResult": _search_result(vector_rank, vector_score),
        "hybridResult": _search_result(hybrid_rank, vector_score if hybrid_rank == vector_rank else text_score),
        "comparison": {
            "textToHybridRankDelta": rank_delta,
            "improved": rank_delta > 0,
            "regressed": rank_delta < 0,
            "unchanged": rank_delta == 0,
            "liftBucket": _lift_bucket(text_rank=text_rank, hybrid_rank=hybrid_rank, rank_delta=rank_delta),
        },
        "policy": {
            "allowedUse": "retrieval_hint_search_quality_eval_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": False,
            "indexEligible": False,
        },
        "blockerReason": "",
    }


def _quality_gate(
    counts: dict[str, Any],
    *,
    min_labs_hit_at5_rows: int,
    min_hybrid_hit_at5_lift_rows: int,
) -> dict[str, Any]:
    text_hit_at5 = int(counts.get("textOnlyHitAt5Rows") or 0)
    hybrid_hit_at5 = int(counts.get("hybridHitAt5Rows") or 0)
    labs_hit_at5 = int(counts.get("labsVectorHitAt5Rows") or 0)
    text_mrr = float(counts.get("textOnlyMrr") or 0.0)
    hybrid_mrr = float(counts.get("hybridMrr") or 0.0)
    hit_at5_lift = hybrid_hit_at5 - text_hit_at5
    checks = {
        "actualLabsIndexRead": int(counts.get("actualLabsVectorIndexRows") or 0) > 0,
        "allSourceUpsertRowsMatched": int(counts.get("matchedVectorRecordRows") or 0)
        == int(counts.get("sourcePlannedVectorUpsertRows") or 0),
        "allSourceCandidatesCoveredByTextBaseline": int(
            counts.get("sourceCandidateRowsMissingFromTextBaseline") or 0
        )
        == 0,
        "queryRowsPresent": int(counts.get("evaluatedQueryRows") or 0) > 0,
        "labsHitAt5MeetsThreshold": labs_hit_at5 >= int(min_labs_hit_at5_rows),
        "hybridHitAt5ImprovesTextOnly": hit_at5_lift >= int(min_hybrid_hit_at5_lift_rows),
        "hybridMrrImprovesTextOnly": hybrid_mrr > text_mrr,
        "noHybridRankRegressions": int(counts.get("rankRegressedRows") or 0) == 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "minLabsHitAt5Rows": int(min_labs_hit_at5_rows),
            "minHybridHitAt5LiftRows": int(min_hybrid_hit_at5_lift_rows),
        },
        "observed": {
            "sourceCandidateRowsCoveredByTextBaseline": int(
                counts.get("sourceCandidateRowsCoveredByTextBaseline") or 0
            ),
            "sourceCandidateRowsMissingFromTextBaseline": int(
                counts.get("sourceCandidateRowsMissingFromTextBaseline") or 0
            ),
            "textOnlyHitAt5Rows": text_hit_at5,
            "labsVectorHitAt5Rows": labs_hit_at5,
            "hybridHitAt5Rows": hybrid_hit_at5,
            "hybridHitAt5LiftRows": hit_at5_lift,
            "textOnlyMrr": text_mrr,
            "labsVectorMrr": float(counts.get("labsVectorMrr") or 0.0),
            "hybridMrr": hybrid_mrr,
        },
    }


def _type_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[normalize_text(row.get("candidateType"))].append(row)
    output: list[dict[str, Any]] = []
    for candidate_type in sorted(grouped):
        items = grouped[candidate_type]
        output.append(
            {
                "candidateType": candidate_type,
                "queryRows": len(items),
                "textOnlyHitAt5Rows": sum(1 for row in items if row.get("textOnlyResult", {}).get("hitAt5")),
                "labsVectorHitAt5Rows": sum(1 for row in items if row.get("labsVectorResult", {}).get("hitAt5")),
                "hybridHitAt5Rows": sum(1 for row in items if row.get("hybridResult", {}).get("hitAt5")),
                "improvedRows": sum(1 for row in items if row.get("comparison", {}).get("improved")),
                "regressedRows": sum(1 for row in items if row.get("comparison", {}).get("regressed")),
            }
        )
    return output


def build_limited_visual_retrieval_hint_candidate_store_labs_vector_index_search_quality_eval(
    *,
    layout_candidate_report: dict[str, Any],
    labs_vector_index_apply_report: dict[str, Any],
    labs_vector_index_apply_executor_dry_run: dict[str, Any],
    source_layout_candidate_report_ref: str,
    source_labs_vector_index_apply_report_ref: str,
    source_labs_vector_index_apply_executor_dry_run_ref: str,
    papers_dir: str | Path,
    min_labs_hit_at5_rows: int = DEFAULT_MIN_LABS_HIT_AT5_ROWS,
    min_hybrid_hit_at5_lift_rows: int = DEFAULT_MIN_HYBRID_HIT_AT5_LIFT_ROWS,
    generated_at: str | None = None,
) -> dict[str, Any]:
    clear_rank_index_cache()
    source_blockers = _source_apply_blockers(labs_vector_index_apply_report) + _source_dry_run_blockers(
        labs_vector_index_apply_executor_dry_run
    )
    if layout_candidate_report.get("schema") != VISUAL_LAYOUT_CANDIDATE_LIST_REPORT_SCHEMA_ID:
        source_blockers.append("invalid_layout_candidate_report_schema")
    if layout_candidate_report.get("status") != "ready":
        source_blockers.append("layout_candidate_report_not_ready")

    layout_rows = [row for row in layout_candidate_report.get("candidateRowsDetail") or [] if isinstance(row, dict)]
    text_documents = {normalize_text(row.get("candidateId")): _text_context(row) for row in layout_rows}

    upsert_records = [
        dict(row)
        for row in labs_vector_index_apply_executor_dry_run.get("plannedVectorUpsertRecords") or []
        if isinstance(row, dict)
    ]
    upsert_by_id = {normalize_text(row.get("vectorDocumentId")): row for row in upsert_records}
    source_candidate_rows_covered_by_text_baseline = sum(
        1 for row in upsert_records if normalize_text(row.get("sourceCandidateId")) in text_documents
    )
    actual_index_rows = _read_jsonl(_index_path(papers_dir))
    actual_by_id = {
        normalize_text(row.get("vectorDocumentId")): row
        for row in actual_index_rows
        if isinstance(row, dict)
    }

    vector_contract_rows: list[dict[str, Any]] = []
    matched_vector_records: dict[str, dict[str, Any]] = {}
    for index, upsert_record in enumerate(upsert_records, start=1):
        vector_doc_id = normalize_text(upsert_record.get("vectorDocumentId"))
        vector_record = actual_by_id.get(vector_doc_id)
        checks = _record_checks(vector_record=vector_record, upsert_record=upsert_record)
        blockers = [name for name, passed in checks.items() if not passed]
        if vector_record and not blockers:
            matched_vector_records[vector_doc_id] = vector_record
        vector_contract_rows.append(
            {
                "contractRowId": f"limited-visual-retrieval-hint-labs-vector-search-quality-contract:{index:04d}",
                "vectorDocumentId": vector_doc_id,
                "hintCandidateId": normalize_text(upsert_record.get("hintCandidateId")),
                "sourceCandidateId": normalize_text(upsert_record.get("sourceCandidateId")),
                "paperId": normalize_text(upsert_record.get("paperId")),
                "candidateType": normalize_text(upsert_record.get("candidateType")),
                "recordMatched": bool(vector_record) and not blockers,
                "contractBlockers": sorted(set(blockers)),
                "checks": checks,
            }
        )

    missing_rank = max(len(text_documents), len(matched_vector_records)) + 1
    query_rows: list[dict[str, Any]] = []
    for vector_doc_id, upsert_record in upsert_by_id.items():
        vector_record = matched_vector_records.get(vector_doc_id)
        if not vector_record:
            continue
        for spec in _query_specs(upsert_record):
            if not normalize_text(spec["query"]):
                continue
            query_rows.append(
                _query_row(
                    upsert_record=upsert_record,
                    vector_record=vector_record,
                    query_kind=spec["queryKind"],
                    query=spec["query"],
                    text_documents=text_documents,
                    vector_records=matched_vector_records,
                    missing_rank=missing_rank,
                )
            )

    private_path_leak_rows = (
        1
        if _contains_private_path(vector_contract_rows)
        or _contains_private_path(query_rows)
        or _contains_private_path(
            [
                source_layout_candidate_report_ref,
                source_labs_vector_index_apply_report_ref,
                source_labs_vector_index_apply_executor_dry_run_ref,
            ]
        )
        else 0
    )
    vector_contract_violation_rows = sum(1 for row in vector_contract_rows if row.get("contractBlockers"))
    source_blockers = sorted(set(source_blockers))
    technical_blockers = sorted(
        set(
            source_blockers
            + (["private_path_leak"] if private_path_leak_rows else [])
            + (["actual_labs_vector_index_rows_missing"] if not actual_index_rows else [])
            + (["query_rows_missing"] if not query_rows else [])
        )
    )
    counts: dict[str, Any] = {
        "layoutCandidateRows": len(layout_rows),
        "sourcePlannedVectorUpsertRows": len(upsert_records),
        "sourceCandidateRowsCoveredByTextBaseline": source_candidate_rows_covered_by_text_baseline,
        "sourceCandidateRowsMissingFromTextBaseline": len(upsert_records)
        - source_candidate_rows_covered_by_text_baseline,
        "actualLabsVectorIndexRows": len(actual_index_rows),
        "matchedVectorRecordRows": len(matched_vector_records),
        "inputHintRows": len(matched_vector_records),
        "queryRows": len(query_rows),
        "evaluatedQueryRows": len(query_rows),
        "textOnlyHitAt1Rows": sum(1 for row in query_rows if row.get("textOnlyResult", {}).get("hitAt1")),
        "textOnlyHitAt5Rows": sum(1 for row in query_rows if row.get("textOnlyResult", {}).get("hitAt5")),
        "textOnlyHitAt10Rows": sum(1 for row in query_rows if row.get("textOnlyResult", {}).get("hitAt10")),
        "labsVectorHitAt1Rows": sum(1 for row in query_rows if row.get("labsVectorResult", {}).get("hitAt1")),
        "labsVectorHitAt5Rows": sum(1 for row in query_rows if row.get("labsVectorResult", {}).get("hitAt5")),
        "labsVectorHitAt10Rows": sum(1 for row in query_rows if row.get("labsVectorResult", {}).get("hitAt10")),
        "hybridHitAt1Rows": sum(1 for row in query_rows if row.get("hybridResult", {}).get("hitAt1")),
        "hybridHitAt5Rows": sum(1 for row in query_rows if row.get("hybridResult", {}).get("hitAt5")),
        "hybridHitAt10Rows": sum(1 for row in query_rows if row.get("hybridResult", {}).get("hitAt10")),
        "rankImprovedRows": sum(1 for row in query_rows if row.get("comparison", {}).get("improved")),
        "rankRegressedRows": sum(1 for row in query_rows if row.get("comparison", {}).get("regressed")),
        "rankUnchangedRows": sum(1 for row in query_rows if row.get("comparison", {}).get("unchanged")),
        "strongLiftRows": sum(1 for row in query_rows if row.get("comparison", {}).get("liftBucket") == "strong_lift"),
        "textOnlyMrr": _mrr(query_rows, result_key="textOnlyResult"),
        "labsVectorMrr": _mrr(query_rows, result_key="labsVectorResult"),
        "hybridMrr": _mrr(query_rows, result_key="hybridResult"),
        "vectorContractViolationRows": vector_contract_violation_rows,
        "blockedRows": len(technical_blockers) + vector_contract_violation_rows,
        "candidateStoreWriteRows": 0,
        "embeddingCallRows": 0,
        "embeddingVectorWriteRows": 0,
        "vectorIndexWriteRows": 0,
        "productionVectorIndexWriteRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "reindexOrReembedRows": 0,
        "indexEligibleRows": 0,
        "runtimeVisibleRows": 0,
        "strictEvidenceRows": 0,
        "citationGradeRows": 0,
        "answerableWithoutTextEvidenceRows": 0,
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": 0,
        "byCandidateType": dict(Counter(row.get("candidateType") for row in vector_contract_rows)),
    }
    gate = _quality_gate(
        counts,
        min_labs_hit_at5_rows=min_labs_hit_at5_rows,
        min_hybrid_hit_at5_lift_rows=min_hybrid_hit_at5_lift_rows,
    )
    status = "blocked" if technical_blockers or vector_contract_violation_rows else "ready"
    if status == "blocked":
        decision = BLOCKED_DECISION
        next_tranche = NEXT_TRANCHE_HOLD
    elif gate["passed"]:
        decision = READY_DECISION
        next_tranche = NEXT_TRANCHE_READY
    else:
        decision = KEEP_IN_LABS_DECISION
        next_tranche = NEXT_TRANCHE_HOLD

    return {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_SEARCH_QUALITY_EVAL_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": decision,
        "nextRecommendedTranche": next_tranche,
        "sourceLayoutCandidateReport": {
            "schema": normalize_text(layout_candidate_report.get("schema")),
            "status": normalize_text(layout_candidate_report.get("status")),
            "reportRef": normalize_text(source_layout_candidate_report_ref),
            "candidateRows": len(layout_rows),
        },
        "sourceLabsVectorIndexApplyReport": _source_apply_summary(
            labs_vector_index_apply_report,
            report_ref=source_labs_vector_index_apply_report_ref,
        ),
        "sourceLabsVectorIndexApplyExecutorDryRun": _source_dry_run_summary(
            labs_vector_index_apply_executor_dry_run,
            report_ref=source_labs_vector_index_apply_executor_dry_run_ref,
        ),
        "input": {
            "papersDirRef": "papers_dir",
            "labsNamespace": LABS_NAMESPACE,
            "labsVectorIndexRef": _index_ref(),
            "embeddingProviderRef": LOCAL_EMBEDDING_PROVIDER_REF,
            "embeddingModelRef": LOCAL_EMBEDDING_MODEL_REF,
            "embeddingDimensions": LOCAL_EMBEDDING_DIMENSIONS,
        },
        "scope": _scope(query_rows=len(query_rows), index_rows=len(actual_index_rows)),
        "policy": _policy(),
        "method": {
            "name": "actual_labs_vector_index_text_only_vs_hybrid_search_quality_eval_v1",
            "description": (
                "Reads the applied labs JSONL vector index, verifies vector records against planned "
                "upsert records, runs deterministic lookup probes, and compares text-only layout "
                "retrieval with a hybrid that can also retrieve labs vector records."
            ),
            "queryKinds": ["keyword_lookup", "natural_lookup"],
            "limitations": [
                "This does not query the production vector DB or operational search route.",
                "Queries are deterministic probes derived from retrieval-hint text, not live user traffic.",
                "A passing result authorizes only production vector DB integration design, not production indexing.",
            ],
        },
        "counts": counts,
        "qualityGate": gate,
        "productValidation": {
            "productQuestion": "Do the 125 applied visual retrieval hints improve search candidate discovery enough to design production vector DB integration?",
            "userScenario": "A paper search should find visually described figures, tables, equations, and layout regions that text-only layout context often misses.",
            "decision": decision,
            "scenarioResults": [
                {
                    "scenario": "visual_specific_lookup",
                    "status": "pass" if gate["observed"]["hybridHitAt5LiftRows"] > 0 else "fail",
                    "evidence": "hybridHitAt5LiftRows",
                },
                {
                    "scenario": "policy_quarantine",
                    "status": "pass" if counts["runtimeVisibleRows"] == 0 and counts["strictEvidenceRows"] == 0 else "fail",
                    "evidence": "runtime/evidence counters",
                },
                {
                    "scenario": "production_safety",
                    "status": "pass" if counts["productionVectorIndexWriteRows"] == 0 else "fail",
                    "evidence": "productionVectorIndexWriteRows",
                },
            ],
        },
        "typeSummary": _type_summary(query_rows),
        "vectorContractRows": vector_contract_rows,
        "queryRowsDetail": query_rows,
        "sourceBlockers": source_blockers,
        "technicalBlockers": technical_blockers,
        "warnings": [
            "This report evaluates labs vector search quality only; visual hints remain non-evidence.",
            "Production vector DB integration still requires a separate design/apply gate.",
        ],
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("qualityGate") or {})
    observed = dict(gate.get("observed") or {})
    lines = [
        "# Limited Visual Retrieval Hint Labs Vector Index Search Quality Eval",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- sourcePlannedVectorUpsertRows: `{counts.get('sourcePlannedVectorUpsertRows')}`",
        f"- actualLabsVectorIndexRows: `{counts.get('actualLabsVectorIndexRows')}`",
        f"- matchedVectorRecordRows: `{counts.get('matchedVectorRecordRows')}`",
        f"- queryRows: `{counts.get('queryRows')}`",
        f"- textOnlyHitAt5Rows: `{counts.get('textOnlyHitAt5Rows')}`",
        f"- labsVectorHitAt5Rows: `{counts.get('labsVectorHitAt5Rows')}`",
        f"- hybridHitAt5Rows: `{counts.get('hybridHitAt5Rows')}`",
        f"- hybridHitAt5LiftRows: `{observed.get('hybridHitAt5LiftRows')}`",
        f"- textOnlyMrr: `{counts.get('textOnlyMrr')}`",
        f"- labsVectorMrr: `{counts.get('labsVectorMrr')}`",
        f"- hybridMrr: `{counts.get('hybridMrr')}`",
        f"- qualityGatePassed: `{gate.get('passed')}`",
        "",
        "## Mutation Guarantees",
        "",
        f"- candidateStoreWriteRows: `{counts.get('candidateStoreWriteRows')}`",
        f"- embeddingCallRows: `{counts.get('embeddingCallRows')}`",
        f"- vectorIndexWriteRows: `{counts.get('vectorIndexWriteRows')}`",
        f"- productionVectorIndexWriteRows: `{counts.get('productionVectorIndexWriteRows')}`",
        f"- databaseMutationRows: `{counts.get('databaseMutationRows')}`",
        f"- indexMutationRows: `{counts.get('indexMutationRows')}`",
        f"- runtimeVisibleRows: `{counts.get('runtimeVisibleRows')}`",
        f"- strictEvidenceRows: `{counts.get('strictEvidenceRows')}`",
        f"- citationGradeRows: `{counts.get('citationGradeRows')}`",
        "",
        "## Type Summary",
        "",
        "| type | queryRows | textHit@5 | labsHit@5 | hybridHit@5 | improved | regressed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("typeSummary", []):
        lines.append(
            "| {candidateType} | {queryRows} | {textHit} | {labsHit} | {hybridHit} | {improved} | {regressed} |".format(
                candidateType=row.get("candidateType"),
                queryRows=row.get("queryRows"),
                textHit=row.get("textOnlyHitAt5Rows"),
                labsHit=row.get("labsVectorHitAt5Rows"),
                hybridHit=row.get("hybridHitAt5Rows"),
                improved=row.get("improvedRows"),
                regressed=row.get("regressedRows"),
            )
        )
    lines.extend(
        [
            "",
            "## Sample Query Rows",
            "",
            "| # | kind | paperId | type | textRank | labsRank | hybridRank | delta | sourceCandidateId |",
            "|---:|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    sorted_rows = sorted(
        list(report.get("queryRowsDetail") or []),
        key=lambda row: (
            -(int(row.get("comparison", {}).get("textToHybridRankDelta") or 0)),
            row.get("sourceCandidateId"),
            row.get("queryKind"),
        ),
    )
    for index, row in enumerate(sorted_rows[:24], start=1):
        lines.append(
            "| {index} | {kind} | {paperId} | {candidateType} | {textRank} | {labsRank} | {hybridRank} | {delta} | `{sourceId}` |".format(
                index=index,
                kind=row.get("queryKind"),
                paperId=row.get("paperId"),
                candidateType=row.get("candidateType"),
                textRank=row.get("textOnlyResult", {}).get("rank"),
                labsRank=row.get("labsVectorResult", {}).get("rank"),
                hybridRank=row.get("hybridResult", {}).get("rank"),
                delta=row.get("comparison", {}).get("textToHybridRankDelta"),
                sourceId=row.get("sourceCandidateId"),
            )
        )
    if report.get("technicalBlockers"):
        lines.extend(["", "## Technical Blockers", ""])
        for blocker in report.get("technicalBlockers", []):
            lines.append(f"- `{blocker}`")
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report.get("warnings", []):
            lines.append(f"- `{warning}`")
    return "\n".join(lines).rstrip() + "\n"


def write_limited_visual_retrieval_hint_candidate_store_labs_vector_index_search_quality_eval(
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
    "KEEP_IN_LABS_DECISION",
    "LABS_VECTOR_INDEX_SEARCH_QUALITY_QUERY_ROW_SCHEMA_ID",
    "LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_SEARCH_QUALITY_EVAL_SCHEMA_ID",
    "READY_DECISION",
    "build_limited_visual_retrieval_hint_candidate_store_labs_vector_index_search_quality_eval",
    "load_json",
    "sanitized_report_ref",
    "write_limited_visual_retrieval_hint_candidate_store_labs_vector_index_search_quality_eval",
]
