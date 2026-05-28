from __future__ import annotations

import hashlib
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor_dry_run import (
    EXECUTOR_DRY_RUN_STATUS_BLOCKED_MISSING_VECTOR_DOCUMENT,
    EXECUTOR_DRY_RUN_STATUS_BLOCKED_POLICY_VIOLATION,
    EXECUTOR_DRY_RUN_STATUS_BLOCKED_VECTOR_DOCUMENT_CONTRACT,
    EXECUTOR_DRY_RUN_STATUS_READY,
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_EXECUTOR_DRY_RUN_SCHEMA_ID,
    build_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor_dry_run,
)


SOURCE_REVIEW_REF = "eval/knowledgeos/reports/labs_vector_review_fixture.v1.json"
SOURCE_DRY_RUN_REF = "eval/knowledgeos/reports/labs_vector_dry_run_fixture.v1.json"


def _hash_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash() -> str:
    return "sha256:" + "a" * 64


def _vector_doc(index: int, *, runtime_visible: bool = False) -> dict[str, object]:
    document_text = f"Retrieval hint only: sample visual region {index} about ReLU CIFAR-10 training error."
    embedding_text = f"allowed_use=retrieval_hint_only | paper=sample-paper | keywords=ReLU, CIFAR-10 | {document_text}"
    vector_doc_id = f"visual-retrieval-hint-vector-doc:{index:04d}"
    hint_candidate_id = f"visual-retrieval-hint:sample-paper:figure_caption_region:{index}:abcdefabcdef{index:04d}"
    source_candidate_id = f"visual-layout:sample-paper:figure_caption_region:{index}:aaaaaaaaaaaa{index:04d}"
    bbox = [10.0, 20.0, 120.0, 180.0]
    metadata = {
        "retrieval_unit_schema": "visual_retrieval_hint_vector_document.v1",
        "namespace": "labs_visual_retrieval_hint_candidates_v1",
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
        "runtimeVisible": runtime_visible,
        "indexEligible": False,
    }
    return {
        "vectorDocumentId": vector_doc_id,
        "namespace": "labs_visual_retrieval_hint_candidates_v1",
        "hintCandidateId": hint_candidate_id,
        "sourceCandidateId": source_candidate_id,
        "paperId": "sample-paper",
        "paperRef": "papers_dir/sample.pdf",
        "sourceContentHash": _hash(),
        "page": index,
        "bbox": bbox,
        "candidateType": "figure_caption_region",
        "idempotencyKey": f"visual-retrieval-hint-idempotency:{index:04d}",
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
        },
    }


def _plan_row(document: dict[str, object]) -> dict[str, object]:
    return {
        "planRowId": f"plan:{document['vectorDocumentId']}",
        "vectorDocumentId": document["vectorDocumentId"],
        "hintCandidateId": document["hintCandidateId"],
        "sourceCandidateId": document["sourceCandidateId"],
        "paperId": document["paperId"],
        "paperRef": document["paperRef"],
        "sourceContentHash": document["sourceContentHash"],
        "page": document["page"],
        "bbox": document["bbox"],
        "candidateType": document["candidateType"],
        "embeddingTextHash": document["embeddingTextHash"],
        "documentTextHash": document["documentTextHash"],
        "wouldCallEmbedder": False,
        "wouldWriteVectorIndex": False,
        "indexEligible": False,
        "runtimeVisible": False,
    }


def _review_row(document: dict[str, object], index: int, *, ready: bool = True) -> dict[str, object]:
    return {
        "reviewRowId": f"limited-visual-retrieval-hint-candidate-store-labs-vector-index-review:{index:04d}",
        "sourcePlanRowId": f"plan:{document['vectorDocumentId']}",
        "vectorDocumentId": document["vectorDocumentId"],
        "namespace": document["namespace"],
        "hintCandidateId": document["hintCandidateId"],
        "sourceCandidateId": document["sourceCandidateId"],
        "paperId": document["paperId"],
        "paperRef": document["paperRef"],
        "sourceContentHash": document["sourceContentHash"],
        "page": document["page"],
        "bbox": document["bbox"],
        "candidateType": document["candidateType"],
        "documentTextHash": document["documentTextHash"],
        "embeddingTextHash": document["embeddingTextHash"],
        "qualityGatePassed": ready,
        "applyExecutorDryRunCandidate": ready,
        "indexEligible": False,
        "runtimeVisible": False,
        "strictEvidence": False,
        "citationGrade": False,
        "answerableWithoutTextEvidence": False,
        "reviewStatus": "review_ready_labs_vector_document" if ready else "blocked_quality_gate",
        "reviewBlockers": [] if ready else ["quality_gate_not_passed"],
        "checks": {},
    }


def _dry_run(documents: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema": "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-labs-vector-index-dry-run.v1",
        "status": "ready",
        "decision": "ready_for_limited_visual_retrieval_hint_candidate_store_labs_vector_index_review",
        "counts": {
            "plannedVectorDocumentRows": len(documents),
            "plannedNamespaceRows": 1 if documents else 0,
            "labsQueryRows": len(documents) * 2,
            "labsHitAt5Rows": len(documents) * 2,
            "labsHitAt10Rows": len(documents) * 2,
            "candidateStoreWriteRows": 0,
            "embeddingCallRows": 0,
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
        "planRows": [_plan_row(document) for document in documents],
        "plannedVectorDocuments": documents,
    }


def _review(documents: list[dict[str, object]], *, ready: bool = True) -> dict[str, object]:
    ready_rows = len(documents) if ready else 0
    return {
        "schema": "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-labs-vector-index-review.v1",
        "status": "ready" if ready else "blocked",
        "decision": (
            "ready_for_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor_dry_run"
            if ready
            else "blocked"
        ),
        "counts": {
            "reviewRows": len(documents),
            "reviewReadyRows": ready_rows,
            "applyExecutorDryRunCandidateRows": ready_rows,
            "qualityGatePassedRows": ready_rows,
            "candidateStoreWriteRows": 0,
            "embeddingCallRows": 0,
            "vectorIndexWriteRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "indexEligibleRows": 0,
            "runtimeVisibleRows": 0,
            "strictEvidenceRows": 0,
            "citationGradeRows": 0,
            "answerableWithoutTextEvidenceRows": 0,
            "blockedRows": 0 if ready else len(documents),
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
        },
        "reviewRowsDetail": [_review_row(document, index + 1, ready=ready) for index, document in enumerate(documents)],
    }


def _build(
    review: dict[str, object],
    dry_run: dict[str, object],
) -> dict[str, object]:
    return build_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor_dry_run(
        labs_vector_index_review=review,
        labs_vector_index_dry_run=dry_run,
        source_labs_vector_index_review_ref=SOURCE_REVIEW_REF,
        source_labs_vector_index_dry_run_ref=SOURCE_DRY_RUN_REF,
        generated_at="2026-05-27T00:00:00Z",
    )


def test_labs_vector_apply_executor_dry_run_plans_upserts_without_writes() -> None:
    documents = [_vector_doc(1), _vector_doc(2)]

    report = _build(_review(documents), _dry_run(documents))

    assert report["status"] == "ready"
    assert report["counts"]["plannedVectorUpsertRows"] == 2
    assert report["counts"]["embeddingInputRows"] == 2
    assert report["counts"]["embeddingCallRows"] == 0
    assert report["counts"]["vectorIndexWriteRows"] == 0
    assert report["counts"]["blockedRows"] == 0
    assert {row["executionStatus"] for row in report["executorDryRunRowsDetail"]} == {
        EXECUTOR_DRY_RUN_STATUS_READY
    }
    assert {row["executionPlan"]["actualVectorIndexWrite"] for row in report["plannedVectorUpsertRecords"]} == {False}
    assert validate_payload(
        report,
        LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok


def test_labs_vector_apply_executor_dry_run_blocks_non_ready_review() -> None:
    documents = [_vector_doc(1)]

    report = _build(_review(documents, ready=False), _dry_run(documents))

    assert report["status"] == "blocked"
    assert report["counts"]["plannedVectorUpsertRows"] == 0
    assert report["counts"]["schemaViolationCount"] > 0


def test_labs_vector_apply_executor_dry_run_blocks_missing_vector_document() -> None:
    documents = [_vector_doc(1)]

    report = _build(_review(documents), _dry_run([]))

    assert report["status"] == "blocked"
    assert report["executorDryRunRowsDetail"][0]["executionStatus"] == EXECUTOR_DRY_RUN_STATUS_BLOCKED_MISSING_VECTOR_DOCUMENT


def test_labs_vector_apply_executor_dry_run_blocks_policy_violation() -> None:
    documents = [_vector_doc(1, runtime_visible=True)]

    report = _build(_review(documents), _dry_run(documents))

    assert report["status"] == "blocked"
    assert report["executorDryRunRowsDetail"][0]["executionStatus"] == EXECUTOR_DRY_RUN_STATUS_BLOCKED_POLICY_VIOLATION


def test_labs_vector_apply_executor_dry_run_blocks_hash_mismatch() -> None:
    documents = [_vector_doc(1)]
    review = _review(documents)
    review["reviewRowsDetail"][0]["embeddingTextHash"] = "sha256:" + "f" * 64

    report = _build(review, _dry_run(documents))

    assert report["status"] == "blocked"
    assert report["executorDryRunRowsDetail"][0]["executionStatus"] == EXECUTOR_DRY_RUN_STATUS_BLOCKED_VECTOR_DOCUMENT_CONTRACT
    assert "embeddingTextHashMatches" in report["executorDryRunRowsDetail"][0]["executionBlockers"]
