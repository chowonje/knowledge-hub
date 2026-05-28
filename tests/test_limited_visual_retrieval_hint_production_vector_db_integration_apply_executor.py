from __future__ import annotations

import hashlib

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.limited_visual_retrieval_hint_production_vector_db_integration_apply_executor import (
    APPLIED_DECISION,
    EXECUTOR_STATUS_APPLIED,
    EXECUTOR_STATUS_DRY_RUN_READY,
    LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_APPLY_EXECUTOR_SCHEMA_ID,
    READY_DECISION,
    execute_limited_visual_retrieval_hint_production_vector_db_integration_apply_executor,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_production_vector_db_integration_dry_run import (
    LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_DRY_RUN_SCHEMA_ID,
    READY_DECISION as SOURCE_READY_DECISION,
)


SOURCE_REF = "eval/knowledgeos/reports/production_vector_dry_run_fixture.v1.json"


def _hash_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash() -> str:
    return "sha256:" + "f" * 64


def _preview_record(index: int, *, runtime_visible: bool = False) -> dict[str, object]:
    document = f"Retrieval hint only: CLIP Figure {index} prompt engineering and zero-shot transfer."
    embedding = f"allowed_use=retrieval_hint_only | paper=clip | keywords=CLIP, zero-shot | {document}"
    vector_doc_id = f"visual-retrieval-hint-production-vector-doc:{index:04d}"
    hint_id = f"visual-retrieval-hint:clip:figure_caption_region:{index}:bbbbbbbbbbbb{index:04d}"
    source_id = f"visual-layout:clip:figure_caption_region:{index}:aaaaaaaaaaaa{index:04d}"
    metadata = {
        "allowedUse": "retrieval_hint_only",
        "strictEvidence": False,
        "citationGrade": False,
        "answerableWithoutTextEvidence": False,
        "runtimeVisible": False,
        "indexEligible": False,
        "document_id": vector_doc_id,
        "source_type": "visual_retrieval_hint",
        "source_content_hash": _hash(),
    }
    return {
        "schema": "knowledge-hub.paper.visual-retrieval-hint-production-vector-record-preview.v1",
        "productionNamespace": "production_visual_retrieval_hint_candidates_v1",
        "targetVectorDatabaseClass": "knowledge_hub.infrastructure.persistence.vector.VectorDatabase",
        "targetWriteMethod": "VectorDatabase.add_documents",
        "targetVectorStoreRef": "config.vector_db_path/visual_retrieval_hints",
        "targetCollectionName": "knowledge_hub_visual_retrieval_hints",
        "routingMode": "candidate_discovery_only",
        "vectorDocumentId": vector_doc_id,
        "idempotencyKey": f"production-visual-retrieval-hint-vector:{index:04d}",
        "hintCandidateId": hint_id,
        "sourceCandidateId": source_id,
        "sourceContentHash": _hash(),
        "paperId": "clip",
        "paperRef": "papers_dir/clip.pdf",
        "page": index,
        "bbox": [1.0, 2.0, 3.0, 4.0],
        "candidateType": "figure_caption_region",
        "derivedTextForRetrieval": document,
        "retrievalKeywords": ["CLIP", "zero-shot"],
        "documentText": document,
        "documentTextHash": _hash_text(document),
        "embeddingText": embedding,
        "embeddingTextHash": _hash_text(embedding),
        "metadata": metadata,
        "sourceRefs": {},
        "policy": {
            "allowedUse": "retrieval_hint_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": runtime_visible,
            "indexEligible": False,
            "productionIndexEligible": False,
            "candidateDiscoveryOnly": True,
        },
        "executionPlan": {
            "dryRunOnly": True,
            "futureApplyGateRequired": True,
            "actualCandidateStoreWrite": False,
            "actualEmbeddingCall": False,
            "actualEmbeddingVectorWrite": False,
            "actualVectorIndexWrite": False,
            "actualProductionVectorIndexWrite": False,
            "actualDatabaseMutation": False,
            "actualRuntimeExposure": False,
            "actualEvidencePromotion": False,
        },
        "plannedVectorRecordSha256": _hash_text(f"preview-{index}"),
    }


def _records(count: int = 125) -> list[dict[str, object]]:
    return [_preview_record(index) for index in range(1, count + 1)]


def _source_report(records: list[dict[str, object]] | None = None, *, status: str = "ready") -> dict[str, object]:
    records = _records() if records is None else records
    return {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_DRY_RUN_SCHEMA_ID,
        "status": status,
        "decision": SOURCE_READY_DECISION if status == "ready" else "blocked",
        "counts": {
            "plannedProductionVectorRecordRows": len(records),
            "futureApplyCandidateRows": len(records),
            "candidateDiscoveryOnlyRows": len(records),
            "blockedRows": 0,
            "policyViolationRows": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
            "candidateStoreWriteRows": 0,
            "embeddingCallRows": 0,
            "embeddingVectorWriteRows": 0,
            "vectorIndexWriteRows": 0,
            "productionVectorIndexWriteRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "runtimeVisibleRows": 0,
            "strictEvidenceRows": 0,
            "citationGradeRows": 0,
            "answerableWithoutTextEvidenceRows": 0,
        },
        "plannedProductionVectorRecords": records,
    }


def _build(source: dict[str, object] | None = None, **kwargs: object) -> dict[str, object]:
    return execute_limited_visual_retrieval_hint_production_vector_db_integration_apply_executor(
        production_vector_db_integration_dry_run=source or _source_report(),
        source_production_vector_db_integration_dry_run_ref=SOURCE_REF,
        run_id="fixture-run",
        generated_at="2026-05-28T00:00:00Z",
        **kwargs,
    )


def test_production_vector_apply_executor_dry_run_writes_nothing() -> None:
    report = _build()

    assert report["schema"] == LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_APPLY_EXECUTOR_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["counts"]["plannedProductionVectorRecordRows"] == 125
    assert report["counts"]["productionVectorIndexWriteRows"] == 0
    assert report["counts"]["databaseMutationRows"] == 0
    assert report["counts"]["runtimeVisibleRows"] == 0
    assert all(row["status"] == EXECUTOR_STATUS_DRY_RUN_READY for row in report["rows"])

    validation = validate_payload(
        report,
        LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_APPLY_EXECUTOR_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors


def test_production_vector_apply_executor_requires_vector_db_path() -> None:
    report = _build(apply=True)

    assert report["status"] == "blocked"
    assert "apply_requires_vector_db_path" in report["technicalBlockers"]


def test_production_vector_apply_executor_applies_and_readbacks(tmp_path) -> None:
    report = _build(apply=True, vector_db_path=tmp_path / "vector")

    assert report["status"] == "applied"
    assert report["decision"] == APPLIED_DECISION
    assert report["counts"]["appliedProductionVectorRecordRows"] == 125
    assert report["counts"]["readbackValidatedRows"] == 125
    assert report["counts"]["productionVectorIndexWriteRows"] == 125
    assert report["counts"]["runtimeVisibleRows"] == 0
    assert all(row["status"] == EXECUTOR_STATUS_APPLIED for row in report["rows"])


def test_production_vector_apply_executor_blocks_unsafe_policy() -> None:
    records = _records()
    records[0] = _preview_record(1, runtime_visible=True)
    report = _build(_source_report(records))

    assert report["status"] == "blocked"
    assert report["counts"]["plannedProductionVectorRecordRows"] == 124
    assert report["counts"]["policyViolationRows"] == 1
    assert "record_policy_not_retrieval_hint_only" in report["technicalBlockers"]
