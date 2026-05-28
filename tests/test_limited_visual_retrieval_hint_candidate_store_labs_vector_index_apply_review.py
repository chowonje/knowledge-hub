from __future__ import annotations

import hashlib
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor_dry_run import (
    EXECUTOR_DRY_RUN_STATUS_READY,
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_EXECUTOR_DRY_RUN_SCHEMA_ID,
    PLANNED_LABS_VECTOR_INDEX_REF,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_review import (
    APPLY_REVIEW_STATUS_BLOCKED_MISSING_UPSERT_RECORD,
    APPLY_REVIEW_STATUS_BLOCKED_NON_READY_EXECUTOR_ROW,
    APPLY_REVIEW_STATUS_BLOCKED_POLICY,
    APPLY_REVIEW_STATUS_BLOCKED_UPSERT_CONTRACT,
    APPLY_REVIEW_STATUS_READY,
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_REVIEW_SCHEMA_ID,
    build_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_review,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_dry_run import (
    LABS_NAMESPACE,
)


SOURCE_REF = "eval/knowledgeos/reports/labs_vector_apply_executor_fixture.v1.json"


def _hash_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash() -> str:
    return "sha256:" + "a" * 64


def _record_hash(record: dict[str, object]) -> str:
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _hash_text(canonical)


def _upsert_record(index: int, *, runtime_visible: bool = False) -> dict[str, object]:
    document_text = f"Retrieval hint only: sample visual region {index} about ReLU CIFAR-10 training error."
    embedding_text = f"allowed_use=retrieval_hint_only | paper=sample-paper | keywords=ReLU, CIFAR-10 | {document_text}"
    hint_candidate_id = f"visual-retrieval-hint:sample-paper:figure_caption_region:{index}:abcdefabcdef{index:04d}"
    source_candidate_id = f"visual-layout:sample-paper:figure_caption_region:{index}:aaaaaaaaaaaa{index:04d}"
    bbox = [10.0, 20.0, 120.0, 180.0]
    metadata = {
        "retrieval_unit_schema": "visual_retrieval_hint_vector_document.v1",
        "namespace": LABS_NAMESPACE,
        "allowedUse": "retrieval_hint_only",
        "hintCandidateId": hint_candidate_id,
        "sourceCandidateId": source_candidate_id,
        "paperId": "sample-paper",
        "paperRef": "papers_dir/sample.pdf",
        "sourceContentHash": _hash(),
        "page": index,
        "bbox": bbox,
        "candidateType": "figure_caption_region",
        "strictEvidence": False,
        "citationGrade": False,
        "answerableWithoutTextEvidence": False,
        "runtimeVisible": False,
        "indexEligible": False,
    }
    return {
        "schema": "knowledge-hub.paper.visual-retrieval-hint-labs-vector-upsert-record.v1",
        "namespace": LABS_NAMESPACE,
        "plannedVectorIndexRef": PLANNED_LABS_VECTOR_INDEX_REF,
        "vectorDocumentId": f"visual-retrieval-hint-vector-doc:{index:04d}",
        "hintCandidateId": hint_candidate_id,
        "sourceCandidateId": source_candidate_id,
        "paperId": "sample-paper",
        "paperRef": "papers_dir/sample.pdf",
        "sourceContentHash": _hash(),
        "page": index,
        "bbox": bbox,
        "candidateType": "figure_caption_region",
        "sourceRecordSha256": _hash(),
        "documentText": document_text,
        "documentTextHash": _hash_text(document_text),
        "embeddingText": embedding_text,
        "embeddingTextHash": _hash_text(embedding_text),
        "metadata": metadata,
        "policy": {
            "allowedUse": "retrieval_hint_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": runtime_visible,
            "indexEligible": False,
            "productionIndexEligible": False,
            "labsOnly": True,
        },
        "executionPlan": {
            "embeddingProviderRef": "deferred_to_separate_explicit_labs_apply",
            "embeddingVectorPresent": False,
            "wouldCallEmbedderOnSeparateExplicitApply": True,
            "wouldWriteVectorIndexOnSeparateExplicitApply": True,
            "actualEmbeddingCall": False,
            "actualEmbeddingVectorWrite": False,
            "actualVectorIndexWrite": False,
        },
    }


def _executor_row(record: dict[str, object], index: int, *, status: str = EXECUTOR_DRY_RUN_STATUS_READY) -> dict[str, object]:
    return {
        "executorDryRunRowId": (
            "limited-visual-retrieval-hint-candidate-store-labs-vector-index-apply-executor-dry-run:"
            f"{index:04d}"
        ),
        "sourceReviewRowId": f"limited-visual-retrieval-hint-candidate-store-labs-vector-index-review:{index:04d}",
        "sourcePlanRowId": f"plan:{record['vectorDocumentId']}",
        "vectorDocumentId": record["vectorDocumentId"],
        "namespace": record["namespace"],
        "hintCandidateId": record["hintCandidateId"],
        "sourceCandidateId": record["sourceCandidateId"],
        "paperId": record["paperId"],
        "paperRef": record["paperRef"],
        "sourceContentHash": record["sourceContentHash"],
        "page": record["page"],
        "bbox": record["bbox"],
        "candidateType": record["candidateType"],
        "documentTextHash": record["documentTextHash"],
        "embeddingTextHash": record["embeddingTextHash"],
        "plannedVectorUpsertRecordSha256": _record_hash(record),
        "plannedVectorIndexRef": PLANNED_LABS_VECTOR_INDEX_REF,
        "wouldCallEmbedderOnSeparateExplicitApply": True,
        "wouldWriteVectorIndexOnSeparateExplicitApply": True,
        "actualEmbeddingCall": False,
        "actualEmbeddingVectorWrite": False,
        "actualVectorIndexWrite": False,
        "indexEligible": False,
        "runtimeVisible": False,
        "strictEvidence": False,
        "citationGrade": False,
        "answerableWithoutTextEvidence": False,
        "executionStatus": status,
        "executionBlockers": [] if status == EXECUTOR_DRY_RUN_STATUS_READY else ["fixture_non_ready"],
        "checks": {},
    }


def _source_report(
    records: list[dict[str, object]],
    *,
    status: str = "ready",
    rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    executor_rows = rows or [_executor_row(record, index + 1) for index, record in enumerate(records)]
    return {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_EXECUTOR_DRY_RUN_SCHEMA_ID,
        "status": status,
        "decision": (
            "ready_for_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_review"
            if status == "ready"
            else "blocked"
        ),
        "counts": {
            "executorDryRunRows": len(executor_rows),
            "plannedVectorUpsertRows": len(records),
            "plannedLabsNamespaceRows": 1 if records else 0,
            "embeddingInputRows": len(records),
            "candidateStoreWriteRows": 0,
            "embeddingCallRows": 0,
            "embeddingVectorWriteRows": 0,
            "vectorIndexWriteRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "indexEligibleRows": 0,
            "runtimeVisibleRows": 0,
            "strictEvidenceRows": 0,
            "citationGradeRows": 0,
            "answerableWithoutTextEvidenceRows": 0,
            "blockedRows": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
        },
        "executorDryRunRowsDetail": executor_rows,
        "plannedVectorUpsertRecords": records,
    }


def _build(report: dict[str, object]) -> dict[str, object]:
    return build_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_review(
        labs_vector_index_apply_executor_dry_run=report,
        source_labs_vector_index_apply_executor_dry_run_ref=SOURCE_REF,
        generated_at="2026-05-27T00:00:00Z",
    )


def test_labs_vector_apply_review_accepts_ready_upserts_without_writes() -> None:
    source = _source_report([_upsert_record(1), _upsert_record(2)])

    report = _build(source)

    assert report["status"] == "ready"
    assert report["decision"] == "ready_for_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor"
    assert report["counts"]["reviewReadyRows"] == 2
    assert report["counts"]["labsApplyExecutorCandidateRows"] == 2
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert report["counts"]["embeddingCallRows"] == 0
    assert report["counts"]["embeddingVectorWriteRows"] == 0
    assert report["counts"]["vectorIndexWriteRows"] == 0
    assert {row["applyReviewStatus"] for row in report["applyReviewRowsDetail"]} == {APPLY_REVIEW_STATUS_READY}
    assert validate_payload(
        report,
        LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_labs_vector_apply_review_blocks_non_ready_source_report() -> None:
    source = _source_report([_upsert_record(1)], status="blocked")

    report = _build(source)

    assert report["status"] == "blocked"
    assert report["counts"]["schemaViolationCount"] > 0
    assert report["gate"]["readyForLabsVectorIndexApplyExecutor"] is False


def test_labs_vector_apply_review_blocks_non_ready_executor_row() -> None:
    record = _upsert_record(1)
    source = _source_report(
        [record],
        rows=[_executor_row(record, 1, status="blocked_vector_document_contract")],
    )

    report = _build(source)

    assert report["status"] == "blocked"
    assert report["applyReviewRowsDetail"][0]["applyReviewStatus"] == APPLY_REVIEW_STATUS_BLOCKED_NON_READY_EXECUTOR_ROW


def test_labs_vector_apply_review_blocks_missing_upsert_record() -> None:
    record = _upsert_record(1)
    source = _source_report([], rows=[_executor_row(record, 1)])

    report = _build(source)

    assert report["status"] == "blocked"
    assert report["applyReviewRowsDetail"][0]["applyReviewStatus"] == APPLY_REVIEW_STATUS_BLOCKED_MISSING_UPSERT_RECORD


def test_labs_vector_apply_review_blocks_hash_mismatch() -> None:
    record = _upsert_record(1)
    row = _executor_row(record, 1)
    row["plannedVectorUpsertRecordSha256"] = _hash()
    source = _source_report([record], rows=[row])

    report = _build(source)

    assert report["status"] == "blocked"
    assert report["applyReviewRowsDetail"][0]["applyReviewStatus"] == APPLY_REVIEW_STATUS_BLOCKED_UPSERT_CONTRACT
    assert "plannedUpsertRecordHashMatches" in report["applyReviewRowsDetail"][0]["applyReviewBlockers"]


def test_labs_vector_apply_review_blocks_policy_violation() -> None:
    record = _upsert_record(1, runtime_visible=True)
    source = _source_report([record])

    report = _build(source)

    assert report["status"] == "blocked"
    assert report["applyReviewRowsDetail"][0]["applyReviewStatus"] == APPLY_REVIEW_STATUS_BLOCKED_POLICY
    assert report["counts"]["candidateStoreWriteRows"] == 0
