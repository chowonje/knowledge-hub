"""Labs-only vector index dry-run for visual retrieval-hint candidates.

This module reads the applied candidate-store JSONL and produces a schema-backed
plan for a future labs vector namespace. It does not call an embedder, write a
vector DB, mutate an index, expose hints at runtime, or promote evidence.
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
    _idempotency_key_from_record,
    _record_hash,
    _record_policy_ok,
    normalize_text,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_apply_readback_review import (
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_READBACK_REVIEW_SCHEMA_ID,
    READY_DECISION as READBACK_READY_DECISION,
)
from knowledge_hub.papers.visual_retrieval_hint_usefulness_eval import (
    _rank_documents,
    _tokens,
    clear_rank_index_cache,
)


LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_DRY_RUN_SCHEMA_ID = (
    "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-labs-vector-index-dry-run.v1"
)

VECTOR_DOC_STATUS_PLANNED = "planned_labs_vector_document"
VECTOR_DOC_STATUS_BLOCKED_MISSING_STORE_RECORD = "blocked_missing_store_record"
VECTOR_DOC_STATUS_BLOCKED_HASH_MISMATCH = "blocked_hash_mismatch"
VECTOR_DOC_STATUS_BLOCKED_POLICY_VIOLATION = "blocked_policy_violation"

READY_DECISION = "ready_for_limited_visual_retrieval_hint_candidate_store_labs_vector_index_review"
BLOCKED_DECISION = "blocked"
NEXT_RECOMMENDED_TRANCHE = "limited_visual_retrieval_hint_candidate_store_labs_vector_index_review"

LABS_NAMESPACE = "labs_visual_retrieval_hint_candidates_v1"
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


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _short_hash(value: str, *, length: int = 24) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


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


def _record_text(record: dict[str, Any]) -> str:
    return normalize_text(
        " ".join(
            [
                normalize_text(record.get("derivedTextForRetrieval")),
                normalize_text(record.get("visibleText")),
                " ".join(normalize_text(item) for item in list(record.get("retrievalKeywords") or [])),
            ]
        )
    )


def _embedding_text(record: dict[str, Any]) -> str:
    keywords = ", ".join(normalize_text(item) for item in list(record.get("retrievalKeywords") or []))
    return normalize_text(
        " | ".join(
            [
                "allowed_use=retrieval_hint_only",
                f"paper={normalize_text(record.get('paperId'))}",
                f"type={normalize_text(record.get('candidateType'))}",
                f"page={int(record.get('page') or 0)}",
                f"keywords={keywords}",
                _record_text(record),
            ]
        )
    )


def _vector_document_id(record: dict[str, Any]) -> str:
    basis = "|".join(
        [
            normalize_text(record.get("hintCandidateId")),
            normalize_text(record.get("sourceCandidateId")),
            normalize_text(record.get("sourceContentHash")),
            json.dumps(record.get("bbox") or [], ensure_ascii=True, sort_keys=True),
        ]
    )
    return "visual-retrieval-hint-vector-doc:" + _short_hash(basis)


def _query_specs(record: dict[str, Any]) -> list[dict[str, str]]:
    tokens: list[str] = []
    for keyword in list(record.get("retrievalKeywords") or []):
        for token in _tokens(normalize_text(keyword)):
            if token not in tokens:
                tokens.append(token)
    if len(tokens) < 4:
        for token in _tokens(_record_text(record)):
            if token not in tokens:
                tokens.append(token)
    natural = normalize_text(
        f"find {record.get('paperId')} {record.get('candidateType')} about {', '.join(list(record.get('retrievalKeywords') or [])[:4])}"
    )
    return [
        {"queryKind": "keyword_lookup", "query": " ".join(tokens[:12])},
        {"queryKind": "natural_lookup", "query": natural},
    ]


def _source_readback_summary(report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = dict(report.get("counts") or {})
    return {
        "schema": normalize_text(report.get("schema")),
        "status": normalize_text(report.get("status")),
        "decision": normalize_text(report.get("decision")),
        "reportRef": normalize_text(report_ref),
        "expectedCandidateRows": int(counts.get("expectedCandidateRows") or 0),
        "storeRows": int(counts.get("storeRows") or 0),
        "readbackValidatedRows": int(counts.get("readbackValidatedRows") or 0),
        "blockedRows": int(counts.get("blockedRows") or 0),
        "privatePathLeakRows": int(counts.get("privatePathLeakRows") or 0),
        "schemaViolationCount": int(counts.get("schemaViolationCount") or 0),
    }


def _source_blockers(readback_review: dict[str, Any]) -> list[str]:
    counts = dict(readback_review.get("counts") or {})
    blockers: list[str] = []
    if readback_review.get("schema") != LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_READBACK_REVIEW_SCHEMA_ID:
        blockers.append("invalid_readback_review_schema")
    if readback_review.get("status") != "ready":
        blockers.append("readback_review_not_ready")
    if readback_review.get("decision") != READBACK_READY_DECISION:
        blockers.append("readback_review_invalid_decision")
    for field_name in ("blockedRows", "privatePathLeakRows", "schemaViolationCount"):
        if int(counts.get(field_name) or 0) != 0:
            blockers.append(f"readback_review_has_{field_name}")
    if int(counts.get("readbackValidatedRows") or 0) != int(counts.get("expectedCandidateRows") or 0):
        blockers.append("readback_review_count_mismatch")
    return blockers


def _scope(planned_rows: int, query_rows: int) -> dict[str, Any]:
    return {
        "writes": "report_only",
        "apiCalls": False,
        "modelCalls": False,
        "webModelCalls": False,
        "plannedVectorDocumentRows": int(planned_rows),
        "labsQueryRows": int(query_rows),
        "candidateStoreWriteRows": 0,
        "embeddingCallRows": 0,
        "vectorIndexWriteRows": 0,
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
        "dryRunOnly": True,
        "labsOnly": True,
        "candidateStoreWrite": False,
        "embeddingCalls": False,
        "vectorIndexWrite": False,
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


def _planned_vector_document(record: dict[str, Any]) -> dict[str, Any]:
    embedding_text = _embedding_text(record)
    document_text = normalize_text(record.get("derivedTextForRetrieval"))
    return {
        "vectorDocumentId": _vector_document_id(record),
        "namespace": LABS_NAMESPACE,
        "hintCandidateId": normalize_text(record.get("hintCandidateId")),
        "sourceCandidateId": normalize_text(record.get("sourceCandidateId")),
        "paperId": normalize_text(record.get("paperId")),
        "paperRef": normalize_text(record.get("paperRef")),
        "sourceContentHash": normalize_text(record.get("sourceContentHash")),
        "page": int(record.get("page") or 0),
        "bbox": list(record.get("bbox") or []),
        "candidateType": normalize_text(record.get("candidateType")),
        "idempotencyKey": _idempotency_key_from_record(record),
        "sourceRecordSha256": _record_hash(record),
        "documentText": document_text,
        "documentTextHash": _sha256_text(document_text),
        "embeddingText": embedding_text,
        "embeddingTextHash": _sha256_text(embedding_text),
        "metadata": {
            "retrieval_unit_schema": "visual_retrieval_hint_vector_document.v1",
            "namespace": LABS_NAMESPACE,
            "allowedUse": "retrieval_hint_only",
            "hintCandidateId": normalize_text(record.get("hintCandidateId")),
            "sourceCandidateId": normalize_text(record.get("sourceCandidateId")),
            "paperId": normalize_text(record.get("paperId")),
            "paperRef": normalize_text(record.get("paperRef")),
            "sourceContentHash": normalize_text(record.get("sourceContentHash")),
            "page": int(record.get("page") or 0),
            "bbox": list(record.get("bbox") or []),
            "candidateType": normalize_text(record.get("candidateType")),
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": False,
            "indexEligible": False,
        },
        "policy": {
            "allowedUse": "retrieval_hint_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": False,
            "indexEligible": False,
        },
    }


def _candidate_status(
    *,
    readback_row: dict[str, Any],
    store_matches: list[dict[str, Any]],
) -> tuple[str, list[str], dict[str, bool], dict[str, Any] | None]:
    expected_hash = normalize_text(readback_row.get("storedRecordSha256") or readback_row.get("expectedRecordSha256"))
    stored = store_matches[0] if store_matches else None
    checks = {
        "readbackValidated": readback_row.get("readbackValidated") is True,
        "singleStoreRecordForIdempotencyKey": len(store_matches) == 1,
        "storeRecordPresent": bool(stored),
        "sourceRecordHashMatchesReadback": bool(stored) and _record_hash(stored) == expected_hash,
        "policyRetrievalHintOnly": bool(stored) and _record_policy_ok(stored),
        "recordNotPrivatePathLeaking": bool(stored) and not _contains_private_path(stored),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if not stored:
        return VECTOR_DOC_STATUS_BLOCKED_MISSING_STORE_RECORD, blockers, checks, None
    if len(store_matches) != 1 or not checks["sourceRecordHashMatchesReadback"]:
        return VECTOR_DOC_STATUS_BLOCKED_HASH_MISMATCH, blockers, checks, None
    if not checks["policyRetrievalHintOnly"] or not checks["recordNotPrivatePathLeaking"]:
        return VECTOR_DOC_STATUS_BLOCKED_POLICY_VIOLATION, blockers, checks, None
    return VECTOR_DOC_STATUS_PLANNED, [], checks, stored


def _labs_query_rows(planned_documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents = {doc["vectorDocumentId"]: normalize_text(doc.get("embeddingText")) for doc in planned_documents}
    missing_rank = len(documents) + 1
    rows: list[dict[str, Any]] = []
    source_records = {doc["vectorDocumentId"]: doc for doc in planned_documents}
    for vector_doc_id, doc in source_records.items():
        record_like = {
            "paperId": doc.get("paperId"),
            "candidateType": doc.get("candidateType"),
            "retrievalKeywords": [
                part.strip() for part in normalize_text(doc.get("metadata", {}).get("candidateType")).split(",") if part.strip()
            ],
            "derivedTextForRetrieval": doc.get("documentText"),
        }
        record_like["retrievalKeywords"] = _tokens(doc.get("embeddingText", ""))[:8]
        for spec in _query_specs(record_like):
            rank, score = _rank_documents(
                documents=documents,
                query=spec["query"],
                target_id=vector_doc_id,
            )
            rank_value = rank if rank is not None else missing_rank
            rows.append(
                {
                    "queryRowId": "visual-retrieval-hint-labs-vector-query:"
                    + _short_hash("|".join([vector_doc_id, spec["queryKind"], spec["query"]])),
                    "queryKind": spec["queryKind"],
                    "query": spec["query"],
                    "targetVectorDocumentId": vector_doc_id,
                    "hintCandidateId": normalize_text(doc.get("hintCandidateId")),
                    "sourceCandidateId": normalize_text(doc.get("sourceCandidateId")),
                    "rank": rank,
                    "score": round(float(score), 6),
                    "hitAt1": bool(rank is not None and rank <= 1),
                    "hitAt5": bool(rank is not None and rank <= 5),
                    "hitAt10": bool(rank is not None and rank <= 10),
                    "missingRankValue": rank_value,
                    "policy": {
                        "allowedUse": "retrieval_hint_only",
                        "strictEvidence": False,
                        "citationGrade": False,
                        "answerableWithoutTextEvidence": False,
                        "runtimeVisible": False,
                        "indexEligible": False,
                    },
                }
            )
    return rows


def build_limited_visual_retrieval_hint_candidate_store_labs_vector_index_dry_run(
    *,
    readback_review: dict[str, Any],
    source_readback_review_ref: str,
    papers_dir: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    clear_rank_index_cache()
    source_blockers = _source_blockers(readback_review)
    store_rows = _read_jsonl(_store_path(papers_dir))
    store_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in store_rows:
        store_by_key[_idempotency_key_from_record(row)].append(row)

    plan_rows: list[dict[str, Any]] = []
    planned_documents: list[dict[str, Any]] = []
    for index, readback_row in enumerate(list(readback_review.get("rows") or []), start=1):
        key = normalize_text(readback_row.get("idempotencyKey"))
        status, blockers, checks, stored = _candidate_status(
            readback_row=dict(readback_row),
            store_matches=store_by_key.get(key, []),
        )
        planned_document = _planned_vector_document(stored) if stored and status == VECTOR_DOC_STATUS_PLANNED else None
        if planned_document:
            planned_documents.append(planned_document)
        plan_rows.append(
            {
                "planRowId": f"limited-visual-retrieval-hint-candidate-store-labs-vector-index-dry-run:{index:04d}",
                "sourceReadbackReviewRowId": normalize_text(readback_row.get("readbackReviewRowId")),
                "hintCandidateId": normalize_text(readback_row.get("hintCandidateId")),
                "sourceCandidateId": normalize_text(readback_row.get("sourceCandidateId")),
                "paperId": normalize_text(readback_row.get("paperId")),
                "paperRef": normalize_text(readback_row.get("paperRef")),
                "sourceContentHash": normalize_text(readback_row.get("sourceContentHash")),
                "page": int(readback_row.get("page") or 0),
                "bbox": list(readback_row.get("bbox") or []),
                "candidateType": normalize_text(readback_row.get("candidateType")),
                "idempotencyKey": key,
                "sourceRecordSha256": normalize_text(readback_row.get("storedRecordSha256")),
                "vectorDocumentId": normalize_text(planned_document.get("vectorDocumentId")) if planned_document else "",
                "namespace": LABS_NAMESPACE if planned_document else "",
                "embeddingTextHash": normalize_text(planned_document.get("embeddingTextHash")) if planned_document else "",
                "documentTextHash": normalize_text(planned_document.get("documentTextHash")) if planned_document else "",
                "wouldPlanVectorDocument": bool(planned_document),
                "wouldCallEmbedder": False,
                "wouldWriteVectorIndex": False,
                "indexEligible": False,
                "runtimeVisible": False,
                "strictEvidence": False,
                "citationGrade": False,
                "answerableWithoutTextEvidence": False,
                "planStatus": status,
                "planBlockers": sorted(set(blockers)),
                "checks": checks,
            }
        )

    query_rows = _labs_query_rows(planned_documents)
    private_path_leak_rows = 1 if _contains_private_path(plan_rows) or _contains_private_path(planned_documents) else 0
    blocked_rows = sum(1 for row in plan_rows if row["planStatus"] != VECTOR_DOC_STATUS_PLANNED)
    schema_violations = sorted(set(source_blockers + (["private_path_leak"] if private_path_leak_rows else [])))
    by_status = Counter(row["planStatus"] for row in plan_rows)
    by_type = Counter(row["candidateType"] for row in plan_rows)
    counts = {
        "sourceReadbackRows": len(list(readback_review.get("rows") or [])),
        "storeRows": len(store_rows),
        "plannedVectorDocumentRows": len(planned_documents),
        "plannedNamespaceRows": 1 if planned_documents else 0,
        "labsQueryRows": len(query_rows),
        "labsHitAt1Rows": sum(1 for row in query_rows if row.get("hitAt1")),
        "labsHitAt5Rows": sum(1 for row in query_rows if row.get("hitAt5")),
        "labsHitAt10Rows": sum(1 for row in query_rows if row.get("hitAt10")),
        "candidateStoreWriteRows": 0,
        "embeddingCallRows": 0,
        "vectorIndexWriteRows": 0,
        "indexEligibleRows": 0,
        "runtimeVisibleRows": 0,
        "strictEvidenceRows": 0,
        "citationGradeRows": 0,
        "answerableWithoutTextEvidenceRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "reindexOrReembedRows": 0,
        "blockedRows": blocked_rows,
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(schema_violations),
        "byCandidateType": dict(by_type),
        "byPlanStatus": dict(by_status),
    }
    status = "blocked" if schema_violations or blocked_rows or not planned_documents else "ready"
    return {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_DRY_RUN_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_RECOMMENDED_TRANCHE,
        "sourceReadbackReview": _source_readback_summary(readback_review, report_ref=source_readback_review_ref),
        "input": {
            "papersDirRef": "papers_dir",
            "candidateStoreRef": PLANNED_STORE_REF,
            "labsNamespace": LABS_NAMESPACE,
        },
        "scope": _scope(len(planned_documents), len(query_rows)),
        "policy": _policy(),
        "method": {
            "name": "labs_visual_retrieval_hint_vector_index_dry_run_v1",
            "description": (
                "Plans labs-only vector documents from readback-validated retrieval hints and runs "
                "an in-memory lexical proxy against the planned embedding text."
            ),
            "limitations": [
                "No embedding model is called in this dry-run.",
                "No vector DB, lexical DB, runtime route, or evidence store is mutated.",
                "Lexical proxy hit rates are a readiness signal for a later labs vector index review, not production search proof.",
            ],
        },
        "counts": counts,
        "gate": {
            "readyForLabsVectorIndexReview": status == "ready",
            "candidateStoreWriteAllowed": False,
            "embeddingCallsAllowed": False,
            "vectorIndexingAllowed": False,
            "runtimeVisibilityAllowed": False,
            "evidencePromotionAllowed": False,
            "schemaViolations": schema_violations,
        },
        "planRows": plan_rows,
        "plannedVectorDocuments": planned_documents,
        "labsQueryRowsDetail": query_rows,
        "warnings": [
            "This report is labs-only and does not write vectors or change operational search.",
            "Visual retrieval hints remain non-evidence and not answer-visible.",
        ],
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Limited Visual Retrieval Hint Candidate Store Labs Vector Index Dry-Run",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- plannedVectorDocumentRows: `{counts.get('plannedVectorDocumentRows')}`",
        f"- plannedNamespaceRows: `{counts.get('plannedNamespaceRows')}`",
        f"- labsQueryRows: `{counts.get('labsQueryRows')}`",
        f"- labsHitAt1Rows: `{counts.get('labsHitAt1Rows')}`",
        f"- labsHitAt5Rows: `{counts.get('labsHitAt5Rows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Mutation Guarantees",
        "",
        f"- candidateStoreWriteRows: `{counts.get('candidateStoreWriteRows')}`",
        f"- embeddingCallRows: `{counts.get('embeddingCallRows')}`",
        f"- vectorIndexWriteRows: `{counts.get('vectorIndexWriteRows')}`",
        f"- indexMutationRows: `{counts.get('indexMutationRows')}`",
        f"- runtimeVisibleRows: `{counts.get('runtimeVisibleRows')}`",
        f"- strictEvidenceRows: `{counts.get('strictEvidenceRows')}`",
        f"- citationGradeRows: `{counts.get('citationGradeRows')}`",
        "",
        "## Candidate Types",
        "",
    ]
    for candidate_type, count in sorted(dict(counts.get("byCandidateType") or {}).items()):
        lines.append(f"- `{candidate_type}`: `{count}`")
    lines.extend(["", "## Plan Status", ""])
    for status, count in sorted(dict(counts.get("byPlanStatus") or {}).items()):
        lines.append(f"- `{status}`: `{count}`")
    return "\n".join(lines).rstrip() + "\n"


def write_limited_visual_retrieval_hint_candidate_store_labs_vector_index_dry_run(
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
    "LABS_NAMESPACE",
    "LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_DRY_RUN_SCHEMA_ID",
    "VECTOR_DOC_STATUS_PLANNED",
    "build_limited_visual_retrieval_hint_candidate_store_labs_vector_index_dry_run",
    "load_json",
    "sanitized_report_ref",
    "write_limited_visual_retrieval_hint_candidate_store_labs_vector_index_dry_run",
]
