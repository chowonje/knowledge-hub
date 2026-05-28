"""Search-quality eval for production visual retrieval-hint vector records.

This report-only helper evaluates the production vector record shape from the
apply executor against a local visual-layout baseline. It does not query or
mutate the operational runtime route; it only measures whether the isolated
candidate-discovery records preserve the labs retrieval lift.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_apply_executor import (
    _contains_private_path,
    normalize_text,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor_dry_run import (
    load_json,
    sanitized_report_ref,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_production_vector_db_integration_apply_executor import (
    APPLIED_DECISION as APPLY_APPLIED_DECISION,
    LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_APPLY_EXECUTOR_SCHEMA_ID,
    PRODUCTION_VECTOR_INDEX_RECORD_SCHEMA_ID,
    READY_DECISION as APPLY_READY_DECISION,
    _hashing_vector,
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


LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_SEARCH_QUALITY_EVAL_SCHEMA_ID = (
    "knowledge-hub.paper.limited-visual-retrieval-hint-production-vector-db-search-quality-eval.v1"
)
PRODUCTION_VECTOR_SEARCH_QUALITY_QUERY_ROW_SCHEMA_ID = (
    "knowledge-hub.paper.limited-visual-retrieval-hint-production-vector-db-search-quality-query-row.v1"
)

READY_DECISION = "ready_for_limited_visual_retrieval_hint_runtime_candidate_discovery_route_design"
KEEP_IN_PRODUCTION_DRY_RUN_DECISION = "keep_in_production_vector_dry_run_pending_search_quality_review"
BLOCKED_DECISION = "blocked"
NEXT_TRANCHE_READY = "limited_visual_retrieval_hint_runtime_candidate_discovery_route_design"
NEXT_TRANCHE_HOLD = "limited_visual_retrieval_hint_production_vector_db_search_quality_review"

EXPECTED_HINT_ROWS = 125
DEFAULT_MIN_PRODUCTION_HIT_AT5_ROWS = 200
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


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _short_hash(value: str, *, length: int = 24) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _counts(report: dict[str, Any]) -> dict[str, Any]:
    return dict(report.get("counts") or {})


def _source_summary(report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = _counts(report)
    return {
        "schema": normalize_text(report.get("schema")),
        "status": normalize_text(report.get("status")),
        "decision": normalize_text(report.get("decision")),
        "reportRef": normalize_text(report_ref),
        "plannedProductionVectorRecordRows": _int(counts.get("plannedProductionVectorRecordRows")),
        "appliedProductionVectorRecordRows": _int(counts.get("appliedProductionVectorRecordRows")),
        "readbackValidatedRows": _int(counts.get("readbackValidatedRows")),
        "productionVectorIndexWriteRows": _int(counts.get("productionVectorIndexWriteRows")),
        "databaseMutationRows": _int(counts.get("databaseMutationRows")),
        "blockedRows": _int(counts.get("blockedRows")),
        "policyViolationRows": _int(counts.get("policyViolationRows")),
        "privatePathLeakRows": _int(counts.get("privatePathLeakRows")),
        "schemaViolationCount": _int(counts.get("schemaViolationCount")),
    }


def _layout_summary(report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = _counts(report)
    rows = list(report.get("candidateRowsDetail") or [])
    return {
        "schema": normalize_text(report.get("schema")),
        "status": normalize_text(report.get("status")),
        "reportRef": normalize_text(report_ref),
        "candidateRows": _int(counts.get("candidateRows") or len(rows)),
        "loaded": True,
    }


def _source_blockers(report: dict[str, Any]) -> list[str]:
    counts = _counts(report)
    records = list(report.get("productionVectorIndexRecordPreviews") or [])
    blockers: list[str] = []
    if report.get("schema") != LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_APPLY_EXECUTOR_SCHEMA_ID:
        blockers.append("invalid_production_vector_apply_executor_schema")
    if report.get("status") not in {"ready", "applied"}:
        blockers.append("production_vector_apply_executor_not_ready_or_applied")
    if report.get("decision") not in {APPLY_READY_DECISION, APPLY_APPLIED_DECISION}:
        blockers.append("production_vector_apply_executor_invalid_decision")
    if len(records) != EXPECTED_HINT_ROWS:
        blockers.append("production_vector_record_previews_not_125")
    ready_or_applied = _int(counts.get("plannedProductionVectorRecordRows")) + _int(
        counts.get("appliedProductionVectorRecordRows")
    )
    if ready_or_applied != EXPECTED_HINT_ROWS:
        blockers.append("production_vector_ready_or_applied_rows_not_125")
    for field in (
        "blockedRows",
        "policyViolationRows",
        "privatePathLeakRows",
        "schemaViolationCount",
        "externalEmbeddingCallRows",
        "embeddingCallRows",
        "runtimeVisibleRows",
        "strictEvidenceRows",
        "citationGradeRows",
        "answerableWithoutTextEvidenceRows",
    ):
        if _int(counts.get(field)) != 0:
            blockers.append(f"production_vector_apply_executor_has_{field}")
    if _contains_private_path(report):
        blockers.append("production_vector_apply_executor_has_private_path_leak")
    return blockers


def _layout_blockers(report: dict[str, Any]) -> list[str]:
    rows = list(report.get("candidateRowsDetail") or [])
    counts = _counts(report)
    blockers: list[str] = []
    if report.get("schema") != VISUAL_LAYOUT_CANDIDATE_LIST_REPORT_SCHEMA_ID:
        blockers.append("invalid_layout_candidate_report_schema")
    if not rows:
        blockers.append("layout_candidate_report_rows_missing")
    if _int(counts.get("candidateRows") or len(rows)) < EXPECTED_HINT_ROWS:
        blockers.append("layout_candidate_report_too_small")
    return blockers


def _policy_ok(record: dict[str, Any]) -> bool:
    policy = dict(record.get("policy") or {})
    metadata = dict(record.get("metadata") or {})
    return (
        normalize_text(policy.get("allowedUse")) == "retrieval_hint_only"
        and policy.get("strictEvidence") is False
        and policy.get("citationGrade") is False
        and policy.get("answerableWithoutTextEvidence") is False
        and policy.get("runtimeVisible") is False
        and policy.get("indexEligible") is False
        and policy.get("productionIndexEligible") is False
        and metadata.get("runtimeVisible") is False
        and metadata.get("strictEvidence") is False
    )


def _record_blockers(record: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if record.get("schema") != PRODUCTION_VECTOR_INDEX_RECORD_SCHEMA_ID:
        blockers.append("invalid_production_vector_index_record_schema")
    for field in (
        "vectorDocumentId",
        "hintCandidateId",
        "sourceCandidateId",
        "sourceContentHash",
        "paperId",
        "page",
        "bbox",
        "candidateType",
        "documentText",
        "embeddingText",
        "embeddingTextHash",
        "metadata",
        "policy",
    ):
        if not record.get(field):
            blockers.append(f"missing_{field}")
    if not _policy_ok(record):
        blockers.append("policy_not_retrieval_hint_only")
    if _contains_private_path(record):
        blockers.append("private_path_leak")
    return blockers


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


def _query_specs(record: dict[str, Any]) -> list[dict[str, str]]:
    document = normalize_text(record.get("documentText"))
    embedding = normalize_text(record.get("embeddingText"))
    tokens = _tokens_for_query(document, limit=12) or _tokens_for_query(embedding, limit=12)
    paper_id = normalize_text(record.get("paperId")).replace("-", " ")
    candidate_type = normalize_text(record.get("candidateType")).replace("_", " ")
    natural_terms = ", ".join(tokens[:6])
    return [
        {"queryKind": "keyword_lookup", "query": " ".join(tokens[:12])},
        {"queryKind": "natural_lookup", "query": f"find {paper_id} {candidate_type} about {natural_terms}"},
    ]


def _layout_documents(layout_report: dict[str, Any]) -> dict[str, str]:
    documents: dict[str, str] = {}
    for row in list(layout_report.get("candidateRowsDetail") or []):
        source_id = normalize_text(row.get("candidateId"))
        if not source_id:
            continue
        documents[source_id] = _text_context(row)
    return documents


def _vector_rank(
    *,
    vector_records: dict[str, dict[str, Any]],
    query: str,
    target_vector_document_id: str,
) -> tuple[int | None, float]:
    query_vector = _hashing_vector(query)
    scored: list[tuple[str, float]] = []
    for doc_id, record in vector_records.items():
        vector = _hashing_vector(normalize_text(record.get("embeddingText")))
        score = sum(float(left) * float(right) for left, right in zip(query_vector, vector))
        scored.append((doc_id, score))
    scored.sort(key=lambda item: (-item[1], item[0]))
    for index, (doc_id, score) in enumerate(scored, start=1):
        if doc_id == target_vector_document_id:
            return index, score
    return None, 0.0


def _result(rank: int | None, score: float) -> dict[str, Any]:
    return {
        "rank": rank,
        "score": round(float(score), 6),
        "hitAt1": bool(rank is not None and rank <= 1),
        "hitAt5": bool(rank is not None and rank <= 5),
        "hitAt10": bool(rank is not None and rank <= 10),
    }


def _mrr(rows: list[dict[str, Any]], *, key: str) -> float:
    if not rows:
        return 0.0
    total = 0.0
    for row in rows:
        rank = dict(row.get(key) or {}).get("rank")
        if isinstance(rank, int) and rank > 0:
            total += 1.0 / rank
    return round(total / len(rows), 6)


def _query_rows(
    *,
    records: list[dict[str, Any]],
    layout_documents: dict[str, str],
) -> list[dict[str, Any]]:
    vector_by_id = {normalize_text(record.get("vectorDocumentId")): record for record in records}
    rows: list[dict[str, Any]] = []
    missing_rank = max(len(layout_documents), len(vector_by_id)) + 1
    clear_rank_index_cache()
    for record in records:
        for spec in _query_specs(record):
            source_candidate_id = normalize_text(record.get("sourceCandidateId"))
            vector_document_id = normalize_text(record.get("vectorDocumentId"))
            text_rank, text_score = _rank_documents(
                documents=layout_documents,
                query=spec["query"],
                target_id=source_candidate_id,
            )
            vector_rank, vector_score = _vector_rank(
                vector_records=vector_by_id,
                query=spec["query"],
                target_vector_document_id=vector_document_id,
            )
            text_rank_value = text_rank if text_rank is not None else missing_rank
            vector_rank_value = vector_rank if vector_rank is not None else missing_rank
            hybrid_rank = min(text_rank_value, vector_rank_value)
            hybrid_rank = hybrid_rank if hybrid_rank <= missing_rank else None
            rank_delta = text_rank_value - hybrid_rank if hybrid_rank is not None else 0
            rows.append(
                {
                    "schema": PRODUCTION_VECTOR_SEARCH_QUALITY_QUERY_ROW_SCHEMA_ID,
                    "rowId": "limited-visual-retrieval-hint-production-vector-search-quality-query:"
                    + _short_hash("|".join([vector_document_id, spec["queryKind"], spec["query"]])),
                    "queryKind": spec["queryKind"],
                    "query": spec["query"],
                    "hintCandidateId": normalize_text(record.get("hintCandidateId")),
                    "sourceCandidateId": source_candidate_id,
                    "vectorDocumentId": vector_document_id,
                    "paperId": normalize_text(record.get("paperId")),
                    "candidateType": normalize_text(record.get("candidateType")),
                    "textOnly": _result(text_rank, text_score),
                    "productionVector": _result(vector_rank, vector_score),
                    "hybrid": _result(hybrid_rank, max(text_score, vector_score)),
                    "rankDelta": rank_delta,
                    "blocked": False,
                }
            )
    return rows


def _counts_for(rows: list[dict[str, Any]], *, records: list[dict[str, Any]], layout_count: int) -> dict[str, Any]:
    return {
        "layoutCandidateRows": layout_count,
        "sourceProductionVectorRecordRows": len(records),
        "matchedProductionVectorRecordRows": len(records),
        "queryRows": len(rows),
        "textOnlyHitAt5Rows": sum(1 for row in rows if dict(row.get("textOnly") or {}).get("hitAt5")),
        "productionVectorHitAt5Rows": sum(1 for row in rows if dict(row.get("productionVector") or {}).get("hitAt5")),
        "hybridHitAt5Rows": sum(1 for row in rows if dict(row.get("hybrid") or {}).get("hitAt5")),
        "hybridHitAt5LiftRows": sum(
            1
            for row in rows
            if dict(row.get("hybrid") or {}).get("hitAt5")
            and not dict(row.get("textOnly") or {}).get("hitAt5")
        ),
        "rankRegressedRows": sum(1 for row in rows if _int(row.get("rankDelta")) < 0),
        "textOnlyMrr": _mrr(rows, key="textOnly"),
        "productionVectorMrr": _mrr(rows, key="productionVector"),
        "hybridMrr": _mrr(rows, key="hybrid"),
        "candidateStoreWriteRows": 0,
        "embeddingCallRows": 0,
        "embeddingVectorWriteRows": 0,
        "vectorIndexWriteRows": 0,
        "productionVectorIndexWriteRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "operationalSearchIndexQueryRows": 0,
        "runtimeVisibleRows": 0,
        "strictEvidenceRows": 0,
        "citationGradeRows": 0,
        "answerableWithoutTextEvidenceRows": 0,
        "blockedRows": sum(1 for row in rows if row.get("blocked")),
        "policyViolationRows": 0,
        "privatePathLeakRows": 0,
        "schemaViolationCount": 0,
        "byCandidateType": dict(Counter(record.get("candidateType") for record in records)),
    }


def build_limited_visual_retrieval_hint_production_vector_db_search_quality_eval(
    *,
    production_vector_apply_executor_report: dict[str, Any],
    layout_candidate_report: dict[str, Any],
    source_production_vector_apply_executor_report_ref: str,
    source_layout_candidate_report_ref: str,
    min_production_hit_at5_rows: int = DEFAULT_MIN_PRODUCTION_HIT_AT5_ROWS,
    min_hybrid_hit_at5_lift_rows: int = DEFAULT_MIN_HYBRID_HIT_AT5_LIFT_ROWS,
    generated_at: str | None = None,
) -> dict[str, Any]:
    source_blockers = [*_source_blockers(production_vector_apply_executor_report), *_layout_blockers(layout_candidate_report)]
    raw_records = [dict(row) for row in production_vector_apply_executor_report.get("productionVectorIndexRecordPreviews") or []]
    row_blockers = sorted({blocker for record in raw_records for blocker in _record_blockers(record)})
    records = [record for record in raw_records if not _record_blockers(record) and not source_blockers]
    docs = _layout_documents(layout_candidate_report)
    rows = _query_rows(records=records, layout_documents=docs) if records and docs else []
    counts = _counts_for(rows, records=records, layout_count=len(docs))
    counts["policyViolationRows"] = 1 if any("policy" in blocker for blocker in row_blockers) else 0
    counts["privatePathLeakRows"] = 1 if _contains_private_path(rows) or "private_path_leak" in row_blockers else 0
    counts["schemaViolationCount"] = len(set(source_blockers + row_blockers))
    quality_gate = {
        "passed": (
            not source_blockers
            and not row_blockers
            and counts["sourceProductionVectorRecordRows"] == EXPECTED_HINT_ROWS
            and counts["productionVectorHitAt5Rows"] >= int(min_production_hit_at5_rows)
            and counts["hybridHitAt5LiftRows"] >= int(min_hybrid_hit_at5_lift_rows)
            and counts["rankRegressedRows"] == 0
            and counts["blockedRows"] == 0
        ),
        "thresholds": {
            "minProductionVectorHitAt5Rows": int(min_production_hit_at5_rows),
            "minHybridHitAt5LiftRows": int(min_hybrid_hit_at5_lift_rows),
            "maxRankRegressedRows": 0,
        },
        "observed": {
            "productionVectorHitAt5Rows": counts["productionVectorHitAt5Rows"],
            "hybridHitAt5LiftRows": counts["hybridHitAt5LiftRows"],
            "rankRegressedRows": counts["rankRegressedRows"],
        },
    }
    status = "ready" if quality_gate["passed"] else ("blocked" if source_blockers or row_blockers else "needs_review")
    decision = READY_DECISION if status == "ready" else (BLOCKED_DECISION if status == "blocked" else KEEP_IN_PRODUCTION_DRY_RUN_DECISION)
    return {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_SEARCH_QUALITY_EVAL_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": decision,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_HOLD,
        "sourceProductionVectorApplyExecutorReport": _source_summary(
            production_vector_apply_executor_report,
            report_ref=source_production_vector_apply_executor_report_ref,
        ),
        "sourceLayoutCandidateReport": _layout_summary(
            layout_candidate_report,
            report_ref=source_layout_candidate_report_ref,
        ),
        "input": {
            "expectedHintRows": EXPECTED_HINT_ROWS,
            "sourceProductionVectorApplyExecutorReportRef": normalize_text(
                source_production_vector_apply_executor_report_ref
            ),
            "sourceLayoutCandidateReportRef": normalize_text(source_layout_candidate_report_ref),
        },
        "policy": {
            "evalOnly": True,
            "candidateDiscoveryOnly": True,
            "candidateStoreWrite": False,
            "embeddingCalls": False,
            "vectorIndexWrite": False,
            "productionVectorIndexWrite": False,
            "operationalSearchIndexQuery": False,
            "runtimeVisible": False,
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
        },
        "method": {
            "name": "visual_retrieval_hint_production_vector_db_search_quality_eval_v1",
            "description": "Compares text-only visual layout retrieval with production visual-hint candidate records.",
            "limitations": [
                "The eval is local and deterministic.",
                "It does not query the operational runtime search route.",
                "It does not promote visual hints to answer evidence.",
            ],
        },
        "counts": counts,
        "qualityGate": quality_gate,
        "queryRowsDetail": rows,
        "sourceBlockers": sorted(set(source_blockers)),
        "technicalBlockers": sorted(set(source_blockers + row_blockers)),
        "warnings": [],
    }


def render_limited_visual_retrieval_hint_production_vector_db_search_quality_eval_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("qualityGate") or {})
    lines = [
        "# Limited Visual Retrieval Hint Production Vector DB Search Quality Eval 005",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- sourceProductionVectorRecordRows: `{counts.get('sourceProductionVectorRecordRows')}`",
        f"- queryRows: `{counts.get('queryRows')}`",
        f"- textOnlyHitAt5Rows: `{counts.get('textOnlyHitAt5Rows')}`",
        f"- productionVectorHitAt5Rows: `{counts.get('productionVectorHitAt5Rows')}`",
        f"- hybridHitAt5Rows: `{counts.get('hybridHitAt5Rows')}`",
        f"- hybridHitAt5LiftRows: `{counts.get('hybridHitAt5LiftRows')}`",
        f"- rankRegressedRows: `{counts.get('rankRegressedRows')}`",
        f"- productionVectorIndexWriteRows: `{counts.get('productionVectorIndexWriteRows')}`",
        f"- runtimeVisibleRows: `{counts.get('runtimeVisibleRows')}`",
        f"- strictEvidenceRows: `{counts.get('strictEvidenceRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Quality Gate",
        "",
        f"- passed: `{gate.get('passed')}`",
        f"- thresholds: `{gate.get('thresholds')}`",
        f"- observed: `{gate.get('observed')}`",
    ]
    blockers = list(report.get("technicalBlockers") or [])
    if blockers:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    return "\n".join(lines) + "\n"


def write_limited_visual_retrieval_hint_production_vector_db_search_quality_eval(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_limited_visual_retrieval_hint_production_vector_db_search_quality_eval_markdown(report),
        encoding="utf-8",
    )
    return {"json": str(report_json), "md": str(report_md)}


__all__ = [
    "LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_SEARCH_QUALITY_EVAL_SCHEMA_ID",
    "READY_DECISION",
    "build_limited_visual_retrieval_hint_production_vector_db_search_quality_eval",
    "load_json",
    "sanitized_report_ref",
    "write_limited_visual_retrieval_hint_production_vector_db_search_quality_eval",
]
