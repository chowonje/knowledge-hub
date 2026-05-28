from __future__ import annotations

import hashlib
import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers import limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor as executor_module
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor import (
    EXECUTOR_STATUS_APPLIED,
    EXECUTOR_STATUS_BLOCKED_POLICY_VIOLATION,
    EXECUTOR_STATUS_BLOCKED_READBACK_MISMATCH,
    EXECUTOR_STATUS_DRY_RUN_READY,
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_EXECUTOR_SCHEMA_ID,
    execute_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor_dry_run import (
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_EXECUTOR_DRY_RUN_SCHEMA_ID,
    PLANNED_LABS_VECTOR_INDEX_REF,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_review import (
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_REVIEW_SCHEMA_ID,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_dry_run import (
    LABS_NAMESPACE,
)


SOURCE_REVIEW_REF = "eval/knowledgeos/reports/labs_vector_apply_review_fixture.v1.json"
SOURCE_DRY_RUN_REF = "eval/knowledgeos/reports/labs_vector_apply_executor_dry_run_fixture.v1.json"


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


def _executor_row(record: dict[str, object], index: int) -> dict[str, object]:
    return {
        "executorDryRunRowId": f"executor-dry-run:{index:04d}",
        "sourceReviewRowId": f"review:{index:04d}",
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
        "executionStatus": "dry_run_ready_labs_vector_upsert",
        "executionBlockers": [],
        "checks": {},
    }


def _review_row(record: dict[str, object], index: int) -> dict[str, object]:
    return {
        "applyReviewRowId": f"apply-review:{index:04d}",
        "sourceExecutorDryRunRowId": f"executor-dry-run:{index:04d}",
        "sourceReviewRowId": f"review:{index:04d}",
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
        "labsApplyExecutorCandidate": True,
        "indexEligible": False,
        "runtimeVisible": False,
        "strictEvidence": False,
        "citationGrade": False,
        "answerableWithoutTextEvidence": False,
        "applyReviewStatus": "review_ready_labs_vector_upsert",
        "applyReviewBlockers": [],
        "checks": {},
    }


def _source_reports(records: list[dict[str, object]]) -> tuple[dict[str, object], dict[str, object]]:
    review = {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_REVIEW_SCHEMA_ID,
        "status": "ready",
        "decision": "ready_for_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor",
        "counts": {
            "reviewRows": len(records),
            "reviewReadyRows": len(records),
            "labsApplyExecutorCandidateRows": len(records),
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
        "applyReviewRowsDetail": [_review_row(record, index + 1) for index, record in enumerate(records)],
    }
    dry_run = {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_EXECUTOR_DRY_RUN_SCHEMA_ID,
        "status": "ready",
        "decision": "ready_for_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_review",
        "counts": {
            "executorDryRunRows": len(records),
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
        "executorDryRunRowsDetail": [_executor_row(record, index + 1) for index, record in enumerate(records)],
        "plannedVectorUpsertRecords": records,
    }
    return review, dry_run


def _build(
    review: dict[str, object],
    dry_run: dict[str, object],
    *,
    apply: bool = False,
    papers_dir: Path | None = None,
) -> dict[str, object]:
    return execute_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor(
        labs_vector_index_apply_review=review,
        labs_vector_index_apply_executor_dry_run=dry_run,
        source_labs_vector_index_apply_review_ref=SOURCE_REVIEW_REF,
        source_labs_vector_index_apply_executor_dry_run_ref=SOURCE_DRY_RUN_REF,
        apply=apply,
        papers_dir=papers_dir,
        run_id="fixture-run",
        generated_at="2026-05-27T00:00:00Z",
    )


def test_labs_vector_apply_executor_dry_run_builds_local_vectors_without_writes(tmp_path: Path) -> None:
    review, dry_run = _source_reports([_upsert_record(1), _upsert_record(2)])

    report = _build(review, dry_run)

    assert report["status"] == "ready"
    assert report["counts"]["plannedVectorRecordRows"] == 2
    assert report["counts"]["dryRunVectorRecordRows"] == 2
    assert report["counts"]["localEmbeddingRows"] == 2
    assert report["counts"]["vectorIndexWriteRows"] == 0
    assert {row["executionStatus"] for row in report["rows"]} == {EXECUTOR_STATUS_DRY_RUN_READY}
    assert {record["embeddingVectorLength"] for record in report["labsVectorRecordPreviews"]} == {256}
    assert not (tmp_path / "visual_retrieval_hints").exists()
    assert validate_payload(
        report,
        LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_APPLY_EXECUTOR_SCHEMA_ID,
        strict=True,
    ).ok


def test_labs_vector_apply_executor_requires_papers_dir_for_apply() -> None:
    review, dry_run = _source_reports([_upsert_record(1)])

    report = _build(review, dry_run, apply=True)

    assert report["status"] == "blocked"
    assert "apply_requires_papers_dir" in report["gate"]["schemaViolations"]
    assert report["counts"]["vectorIndexWriteRows"] == 0


def test_labs_vector_apply_executor_writes_idempotently(tmp_path: Path) -> None:
    review, dry_run = _source_reports([_upsert_record(1), _upsert_record(2)])

    first = _build(review, dry_run, apply=True, papers_dir=tmp_path)
    second = _build(review, dry_run, apply=True, papers_dir=tmp_path)

    assert first["status"] == "applied"
    assert first["counts"]["appliedLabsVectorRecordRows"] == 2
    assert first["counts"]["readbackValidatedRows"] == 2
    assert {row["executionStatus"] for row in first["rows"]} == {EXECUTOR_STATUS_APPLIED}
    assert second["status"] == "applied"
    index_path = tmp_path / "visual_retrieval_hints/labs_vector_index/labs_visual_retrieval_hint_candidates_v1.v1.jsonl"
    assert len(index_path.read_text().splitlines()) == 2


def test_labs_vector_apply_executor_blocks_policy_violation() -> None:
    review, dry_run = _source_reports([_upsert_record(1, runtime_visible=True)])

    report = _build(review, dry_run)

    assert report["status"] == "blocked"
    assert report["rows"][0]["executionStatus"] == EXECUTOR_STATUS_BLOCKED_POLICY_VIOLATION
    assert report["counts"]["vectorIndexWriteRows"] == 0


def test_labs_vector_apply_executor_blocks_readback_mismatch(monkeypatch, tmp_path: Path) -> None:
    review, dry_run = _source_reports([_upsert_record(1)])

    def fake_apply_records(records, *, papers_dir):
        return len(records), 0, ["readback_mismatch:fixture"]

    monkeypatch.setattr(executor_module, "_apply_records", fake_apply_records)
    report = _build(review, dry_run, apply=True, papers_dir=tmp_path)

    assert report["status"] == "blocked"
    assert report["rows"][0]["executionStatus"] == EXECUTOR_STATUS_BLOCKED_READBACK_MISMATCH
    assert report["counts"]["blockedReadbackMismatchRows"] == 1
