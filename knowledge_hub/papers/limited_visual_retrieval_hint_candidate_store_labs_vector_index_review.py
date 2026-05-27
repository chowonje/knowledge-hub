"""Review gate for labs visual retrieval-hint vector-index dry-runs.

The review validates planned labs vector documents before any apply executor is
allowed to exist. It does not call an embedder, write a vector DB, expose runtime
answers, or promote visual hints into evidence.
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
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_dry_run import (
    LABS_NAMESPACE,
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_DRY_RUN_SCHEMA_ID,
    READY_DECISION as LABS_VECTOR_INDEX_DRY_RUN_READY_DECISION,
)


LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_REVIEW_SCHEMA_ID = (
    "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-labs-vector-index-review.v1"
)

REVIEW_STATUS_READY = "review_ready_labs_vector_document"
REVIEW_STATUS_BLOCKED_MISSING_PLAN_ROW = "blocked_missing_plan_row"
REVIEW_STATUS_BLOCKED_CONTRACT = "blocked_vector_document_contract"
REVIEW_STATUS_BLOCKED_POLICY = "blocked_policy_violation"
REVIEW_STATUS_BLOCKED_QUALITY_GATE = "blocked_quality_gate"

READY_DECISION = "ready_for_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor_dry_run"
BLOCKED_DECISION = "blocked"
NEXT_RECOMMENDED_TRANCHE = "limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor_dry_run"

DEFAULT_MIN_LABS_HIT_AT5_ROWS = 200
DEFAULT_MIN_LABS_HIT_AT10_ROWS = 240


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


def _source_dry_run_summary(report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = dict(report.get("counts") or {})
    return {
        "schema": normalize_text(report.get("schema")),
        "status": normalize_text(report.get("status")),
        "decision": normalize_text(report.get("decision")),
        "reportRef": normalize_text(report_ref),
        "sourceReadbackRows": int(counts.get("sourceReadbackRows") or 0),
        "storeRows": int(counts.get("storeRows") or 0),
        "plannedVectorDocumentRows": int(counts.get("plannedVectorDocumentRows") or 0),
        "plannedNamespaceRows": int(counts.get("plannedNamespaceRows") or 0),
        "labsQueryRows": int(counts.get("labsQueryRows") or 0),
        "labsHitAt1Rows": int(counts.get("labsHitAt1Rows") or 0),
        "labsHitAt5Rows": int(counts.get("labsHitAt5Rows") or 0),
        "labsHitAt10Rows": int(counts.get("labsHitAt10Rows") or 0),
        "blockedRows": int(counts.get("blockedRows") or 0),
        "privatePathLeakRows": int(counts.get("privatePathLeakRows") or 0),
        "schemaViolationCount": int(counts.get("schemaViolationCount") or 0),
    }


def _source_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers: list[str] = []
    if report.get("schema") != LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_DRY_RUN_SCHEMA_ID:
        blockers.append("invalid_labs_vector_index_dry_run_schema")
    if report.get("status") != "ready":
        blockers.append("labs_vector_index_dry_run_not_ready")
    if report.get("decision") != LABS_VECTOR_INDEX_DRY_RUN_READY_DECISION:
        blockers.append("labs_vector_index_dry_run_invalid_decision")
    for field_name in (
        "blockedRows",
        "privatePathLeakRows",
        "schemaViolationCount",
        "candidateStoreWriteRows",
        "embeddingCallRows",
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
            blockers.append(f"labs_vector_index_dry_run_has_{field_name}")
    planned_count = int(counts.get("plannedVectorDocumentRows") or 0)
    if planned_count <= 0:
        blockers.append("labs_vector_index_dry_run_has_no_planned_documents")
    if int(counts.get("plannedNamespaceRows") or 0) != (1 if planned_count else 0):
        blockers.append("labs_vector_index_dry_run_namespace_count_mismatch")
    return sorted(set(blockers))


def _policy_flags_ok(container: dict[str, Any]) -> bool:
    return (
        normalize_text(container.get("allowedUse")) == "retrieval_hint_only"
        and container.get("strictEvidence") is False
        and container.get("citationGrade") is False
        and container.get("answerableWithoutTextEvidence") is False
        and container.get("runtimeVisible") is False
        and container.get("indexEligible") is False
    )


def _bbox(value: Any) -> list[Any]:
    return list(value or [])


def _quality_gate(
    counts: dict[str, Any],
    *,
    min_labs_hit_at5_rows: int,
    min_labs_hit_at10_rows: int,
) -> dict[str, Any]:
    labs_query_rows = int(counts.get("labsQueryRows") or 0)
    labs_hit_at5_rows = int(counts.get("labsHitAt5Rows") or 0)
    labs_hit_at10_rows = int(counts.get("labsHitAt10Rows") or 0)
    planned_rows = int(counts.get("plannedVectorDocumentRows") or 0)
    checks = {
        "labsQueryRowsPresent": labs_query_rows > 0,
        "labsHitAt5MeetsThreshold": labs_hit_at5_rows >= int(min_labs_hit_at5_rows),
        "labsHitAt10MeetsThreshold": labs_hit_at10_rows >= int(min_labs_hit_at10_rows),
        "labsQueryRowsCoverPlannedDocuments": labs_query_rows >= planned_rows,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "minLabsHitAt5Rows": int(min_labs_hit_at5_rows),
            "minLabsHitAt10Rows": int(min_labs_hit_at10_rows),
        },
        "observed": {
            "plannedVectorDocumentRows": planned_rows,
            "labsQueryRows": labs_query_rows,
            "labsHitAt5Rows": labs_hit_at5_rows,
            "labsHitAt10Rows": labs_hit_at10_rows,
        },
    }


def _scope(review_rows: int) -> dict[str, Any]:
    return {
        "writes": "report_only",
        "apiCalls": False,
        "modelCalls": False,
        "webModelCalls": False,
        "reviewRows": int(review_rows),
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
        "reviewOnly": True,
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


def _review_document(
    *,
    index: int,
    document: dict[str, Any],
    plan_rows_by_vector_doc_id: dict[str, list[dict[str, Any]]],
    quality_gate_passed: bool,
) -> dict[str, Any]:
    vector_doc_id = normalize_text(document.get("vectorDocumentId"))
    matches = plan_rows_by_vector_doc_id.get(vector_doc_id, [])
    plan_row = matches[0] if matches else {}
    metadata = dict(document.get("metadata") or {})
    policy = dict(document.get("policy") or {})
    document_text = str(document.get("documentText") or "")
    embedding_text = str(document.get("embeddingText") or "")
    checks = {
        "sourcePlanRowPresent": bool(matches),
        "singleSourcePlanRow": len(matches) == 1,
        "namespaceMatchesLabs": normalize_text(document.get("namespace")) == LABS_NAMESPACE,
        "vectorDocumentIdPresent": bool(vector_doc_id),
        "hintCandidateIdPresent": bool(normalize_text(document.get("hintCandidateId"))),
        "sourceCandidateIdPresent": bool(normalize_text(document.get("sourceCandidateId"))),
        "sourceContentHashPresent": normalize_text(document.get("sourceContentHash")).startswith("sha256:"),
        "pagePresent": int(document.get("page") or 0) >= 0,
        "bboxPresent": bool(_bbox(document.get("bbox"))),
        "candidateTypePresent": bool(normalize_text(document.get("candidateType"))),
        "documentTextPresent": bool(normalize_text(document_text)),
        "embeddingTextPresent": bool(normalize_text(embedding_text)),
        "documentTextHashMatches": _sha256_text(document_text) == normalize_text(document.get("documentTextHash")),
        "embeddingTextHashMatches": _sha256_text(embedding_text) == normalize_text(document.get("embeddingTextHash")),
        "metadataSchemaMatches": normalize_text(metadata.get("retrieval_unit_schema"))
        == "visual_retrieval_hint_vector_document.v1",
        "metadataNamespaceMatches": normalize_text(metadata.get("namespace")) == LABS_NAMESPACE,
        "metadataPolicyRetrievalHintOnly": _policy_flags_ok(metadata),
        "rowPolicyRetrievalHintOnly": _policy_flags_ok(policy),
        "metadataHintCandidateIdMatches": normalize_text(metadata.get("hintCandidateId"))
        == normalize_text(document.get("hintCandidateId")),
        "metadataSourceCandidateIdMatches": normalize_text(metadata.get("sourceCandidateId"))
        == normalize_text(document.get("sourceCandidateId")),
        "metadataSourceContentHashMatches": normalize_text(metadata.get("sourceContentHash"))
        == normalize_text(document.get("sourceContentHash")),
        "metadataPageMatches": int(metadata.get("page") or 0) == int(document.get("page") or 0),
        "metadataBboxMatches": _bbox(metadata.get("bbox")) == _bbox(document.get("bbox")),
        "metadataCandidateTypeMatches": normalize_text(metadata.get("candidateType"))
        == normalize_text(document.get("candidateType")),
        "planHintCandidateIdMatches": bool(plan_row)
        and normalize_text(plan_row.get("hintCandidateId")) == normalize_text(document.get("hintCandidateId")),
        "planSourceCandidateIdMatches": bool(plan_row)
        and normalize_text(plan_row.get("sourceCandidateId")) == normalize_text(document.get("sourceCandidateId")),
        "planSourceContentHashMatches": bool(plan_row)
        and normalize_text(plan_row.get("sourceContentHash")) == normalize_text(document.get("sourceContentHash")),
        "planPageMatches": bool(plan_row) and int(plan_row.get("page") or 0) == int(document.get("page") or 0),
        "planBboxMatches": bool(plan_row) and _bbox(plan_row.get("bbox")) == _bbox(document.get("bbox")),
        "planCandidateTypeMatches": bool(plan_row)
        and normalize_text(plan_row.get("candidateType")) == normalize_text(document.get("candidateType")),
        "planEmbeddingTextHashMatches": bool(plan_row)
        and normalize_text(plan_row.get("embeddingTextHash")) == normalize_text(document.get("embeddingTextHash")),
        "planDocumentTextHashMatches": bool(plan_row)
        and normalize_text(plan_row.get("documentTextHash")) == normalize_text(document.get("documentTextHash")),
        "sourcePlanDoesNotCallEmbedder": bool(plan_row) and plan_row.get("wouldCallEmbedder") is False,
        "sourcePlanDoesNotWriteVectorIndex": bool(plan_row) and plan_row.get("wouldWriteVectorIndex") is False,
        "sourcePlanNotRuntimeVisible": bool(plan_row)
        and plan_row.get("runtimeVisible") is False
        and plan_row.get("indexEligible") is False,
        "documentNotPrivatePathLeaking": not _contains_private_path(document),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    policy_failed = any(
        not checks[name]
        for name in (
            "metadataPolicyRetrievalHintOnly",
            "rowPolicyRetrievalHintOnly",
            "sourcePlanNotRuntimeVisible",
            "documentNotPrivatePathLeaking",
        )
    )
    if not matches:
        review_status = REVIEW_STATUS_BLOCKED_MISSING_PLAN_ROW
    elif policy_failed:
        review_status = REVIEW_STATUS_BLOCKED_POLICY
    elif blockers:
        review_status = REVIEW_STATUS_BLOCKED_CONTRACT
    elif not quality_gate_passed:
        review_status = REVIEW_STATUS_BLOCKED_QUALITY_GATE
        blockers.append("quality_gate_not_passed")
    else:
        review_status = REVIEW_STATUS_READY
    return {
        "reviewRowId": f"limited-visual-retrieval-hint-candidate-store-labs-vector-index-review:{index:04d}",
        "sourcePlanRowId": normalize_text(plan_row.get("planRowId")),
        "vectorDocumentId": vector_doc_id,
        "namespace": normalize_text(document.get("namespace")),
        "hintCandidateId": normalize_text(document.get("hintCandidateId")),
        "sourceCandidateId": normalize_text(document.get("sourceCandidateId")),
        "paperId": normalize_text(document.get("paperId")),
        "paperRef": normalize_text(document.get("paperRef")),
        "sourceContentHash": normalize_text(document.get("sourceContentHash")),
        "page": int(document.get("page") or 0),
        "bbox": _bbox(document.get("bbox")),
        "candidateType": normalize_text(document.get("candidateType")),
        "documentTextHash": normalize_text(document.get("documentTextHash")),
        "embeddingTextHash": normalize_text(document.get("embeddingTextHash")),
        "qualityGatePassed": bool(quality_gate_passed),
        "applyExecutorDryRunCandidate": review_status == REVIEW_STATUS_READY,
        "indexEligible": False,
        "runtimeVisible": False,
        "strictEvidence": False,
        "citationGrade": False,
        "answerableWithoutTextEvidence": False,
        "reviewStatus": review_status,
        "reviewBlockers": sorted(set(blockers)),
        "checks": checks,
    }


def review_limited_visual_retrieval_hint_candidate_store_labs_vector_index(
    *,
    labs_vector_index_dry_run: dict[str, Any],
    source_labs_vector_index_dry_run_ref: str,
    generated_at: str | None = None,
    min_labs_hit_at5_rows: int = DEFAULT_MIN_LABS_HIT_AT5_ROWS,
    min_labs_hit_at10_rows: int = DEFAULT_MIN_LABS_HIT_AT10_ROWS,
) -> dict[str, Any]:
    counts_in = dict(labs_vector_index_dry_run.get("counts") or {})
    source_blockers = _source_blockers(labs_vector_index_dry_run)
    planned_documents = [
        dict(row) for row in labs_vector_index_dry_run.get("plannedVectorDocuments") or [] if isinstance(row, dict)
    ]
    plan_rows = [dict(row) for row in labs_vector_index_dry_run.get("planRows") or [] if isinstance(row, dict)]
    plan_rows_by_vector_doc_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for plan_row in plan_rows:
        plan_rows_by_vector_doc_id[normalize_text(plan_row.get("vectorDocumentId"))].append(plan_row)

    if int(counts_in.get("plannedVectorDocumentRows") or 0) != len(planned_documents):
        source_blockers.append("planned_vector_document_count_mismatch")
    if int(counts_in.get("plannedVectorDocumentRows") or 0) != len(plan_rows):
        source_blockers.append("plan_row_count_mismatch")
    if len({normalize_text(doc.get("namespace")) for doc in planned_documents}) != (1 if planned_documents else 0):
        source_blockers.append("planned_document_namespace_count_mismatch")

    quality_gate = _quality_gate(
        counts_in,
        min_labs_hit_at5_rows=min_labs_hit_at5_rows,
        min_labs_hit_at10_rows=min_labs_hit_at10_rows,
    )
    quality_gate_passed = bool(quality_gate["passed"]) and not source_blockers
    review_rows = [
        _review_document(
            index=index,
            document=document,
            plan_rows_by_vector_doc_id=plan_rows_by_vector_doc_id,
            quality_gate_passed=quality_gate_passed,
        )
        for index, document in enumerate(planned_documents, start=1)
    ]
    private_path_leak_rows = 1 if _contains_private_path(review_rows) else 0
    if private_path_leak_rows:
        source_blockers.append("private_path_leak")
    source_blockers = sorted(set(source_blockers))
    by_status = Counter(row["reviewStatus"] for row in review_rows)
    by_type = Counter(row["candidateType"] for row in review_rows)
    blocked_rows = sum(1 for row in review_rows if row["reviewStatus"] != REVIEW_STATUS_READY)
    counts = {
        "sourcePlannedVectorDocumentRows": int(counts_in.get("plannedVectorDocumentRows") or 0),
        "sourcePlanRows": len(plan_rows),
        "sourceLabsQueryRows": int(counts_in.get("labsQueryRows") or 0),
        "reviewRows": len(review_rows),
        "reviewReadyRows": by_status.get(REVIEW_STATUS_READY, 0),
        "applyExecutorDryRunCandidateRows": by_status.get(REVIEW_STATUS_READY, 0),
        "qualityGatePassedRows": by_status.get(REVIEW_STATUS_READY, 0) if quality_gate_passed else 0,
        "plannedNamespaceRows": int(counts_in.get("plannedNamespaceRows") or 0),
        "labsQueryRows": int(counts_in.get("labsQueryRows") or 0),
        "labsHitAt1Rows": int(counts_in.get("labsHitAt1Rows") or 0),
        "labsHitAt5Rows": int(counts_in.get("labsHitAt5Rows") or 0),
        "labsHitAt10Rows": int(counts_in.get("labsHitAt10Rows") or 0),
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
        "schemaViolationCount": len(source_blockers),
        "byCandidateType": dict(by_type),
        "byReviewStatus": dict(by_status),
    }
    status = "ready" if not source_blockers and not blocked_rows and review_rows else "blocked"
    return {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_REVIEW_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_RECOMMENDED_TRANCHE,
        "sourceLabsVectorIndexDryRun": _source_dry_run_summary(
            labs_vector_index_dry_run,
            report_ref=source_labs_vector_index_dry_run_ref,
        ),
        "input": {
            "sourceReportRef": normalize_text(source_labs_vector_index_dry_run_ref),
            "labsNamespace": LABS_NAMESPACE,
        },
        "scope": _scope(len(review_rows)),
        "policy": _policy(),
        "method": {
            "name": "labs_visual_retrieval_hint_vector_index_review_v1",
            "description": (
                "Reviews planned labs vector documents for policy quarantine, source-plan alignment, "
                "hash integrity, and lexical dry-run quality before a later apply-executor dry-run."
            ),
            "qualityGate": quality_gate,
            "limitations": [
                "No embedding model is called by this review.",
                "No vector DB, lexical DB, runtime route, or evidence store is mutated.",
                "Passing this review permits only a later labs vector-index apply-executor dry-run, not production indexing.",
            ],
        },
        "counts": counts,
        "gate": {
            "readyForLabsVectorIndexApplyExecutorDryRun": status == "ready",
            "qualityGatePassed": quality_gate_passed,
            "candidateStoreWriteAllowed": False,
            "embeddingCallsAllowed": False,
            "vectorIndexingAllowed": False,
            "runtimeVisibilityAllowed": False,
            "evidencePromotionAllowed": False,
            "schemaViolations": source_blockers,
        },
        "reviewRowsDetail": review_rows,
        "warnings": [
            "This review is labs-only and does not write vectors or change operational search.",
            "Visual retrieval hints remain non-evidence and not answer-visible.",
        ],
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    quality_gate = dict(dict(report.get("method") or {}).get("qualityGate") or {})
    observed = dict(quality_gate.get("observed") or {})
    thresholds = dict(quality_gate.get("thresholds") or {})
    lines = [
        "# Limited Visual Retrieval Hint Candidate Store Labs Vector Index Review",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- reviewRows: `{counts.get('reviewRows')}`",
        f"- reviewReadyRows: `{counts.get('reviewReadyRows')}`",
        f"- applyExecutorDryRunCandidateRows: `{counts.get('applyExecutorDryRunCandidateRows')}`",
        f"- labsQueryRows: `{counts.get('labsQueryRows')}`",
        f"- labsHitAt5Rows: `{counts.get('labsHitAt5Rows')}`",
        f"- labsHitAt10Rows: `{counts.get('labsHitAt10Rows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Quality Gate",
        "",
        f"- qualityGatePassed: `{quality_gate.get('passed')}`",
        f"- minLabsHitAt5Rows: `{thresholds.get('minLabsHitAt5Rows')}`",
        f"- observedLabsHitAt5Rows: `{observed.get('labsHitAt5Rows')}`",
        f"- minLabsHitAt10Rows: `{thresholds.get('minLabsHitAt10Rows')}`",
        f"- observedLabsHitAt10Rows: `{observed.get('labsHitAt10Rows')}`",
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
    lines.extend(["", "## Review Status", ""])
    for review_status, count in sorted(dict(counts.get("byReviewStatus") or {}).items()):
        lines.append(f"- `{review_status}`: `{count}`")
    return "\n".join(lines).rstrip() + "\n"


def write_limited_visual_retrieval_hint_candidate_store_labs_vector_index_review(
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
    "LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_REVIEW_SCHEMA_ID",
    "REVIEW_STATUS_READY",
    "REVIEW_STATUS_BLOCKED_CONTRACT",
    "REVIEW_STATUS_BLOCKED_MISSING_PLAN_ROW",
    "REVIEW_STATUS_BLOCKED_POLICY",
    "REVIEW_STATUS_BLOCKED_QUALITY_GATE",
    "build_limited_visual_retrieval_hint_candidate_store_labs_vector_index_review",
    "load_json",
    "review_limited_visual_retrieval_hint_candidate_store_labs_vector_index",
    "sanitized_report_ref",
    "write_limited_visual_retrieval_hint_candidate_store_labs_vector_index_review",
]


build_limited_visual_retrieval_hint_candidate_store_labs_vector_index_review = (
    review_limited_visual_retrieval_hint_candidate_store_labs_vector_index
)
