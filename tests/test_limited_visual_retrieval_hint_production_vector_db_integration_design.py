from __future__ import annotations

import hashlib

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.limited_visual_retrieval_hint_production_vector_db_integration_design import (
    LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_DESIGN_SCHEMA_ID,
    PLANNED_STATUS,
    READY_DECISION,
    build_limited_visual_retrieval_hint_production_vector_db_integration_design,
)


QUALITY_REF = "eval/knowledgeos/reports/search_quality_fixture.v1.json"
CANDIDATE_APPLY_REF = "eval/knowledgeos/reports/candidate_apply_fixture.v1.json"
LABS_APPLY_REF = "eval/knowledgeos/reports/labs_apply_fixture.v1.json"
LABS_DRY_RUN_REF = "eval/knowledgeos/reports/labs_dry_run_fixture.v1.json"


def _hash_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash() -> str:
    return "sha256:" + "d" * 64


def _record(index: int) -> dict[str, object]:
    document_text = f"Retrieval hint only: sample paper Figure {index} about ReLU CIFAR-10 training error."
    embedding_text = (
        "allowed_use=retrieval_hint_only | "
        f"paper=sample-paper | type=figure_caption_region | page={index} | "
        "keywords=ReLU, CIFAR-10, training error, visual hint | "
        f"{document_text}"
    )
    source_candidate_id = f"visual-layout:sample-paper:figure_caption_region:{index}:aaaaaaaaaaaa{index:04d}"
    hint_candidate_id = f"visual-retrieval-hint:sample-paper:figure_caption_region:{index}:bbbbbbbbbbbb{index:04d}"
    return {
        "schema": "knowledge-hub.paper.visual-retrieval-hint-labs-vector-upsert-record.v1",
        "namespace": "labs_visual_retrieval_hint_candidates_v1",
        "plannedVectorIndexRef": "labs_vector_index/labs_visual_retrieval_hint_candidates_v1",
        "vectorDocumentId": f"visual-retrieval-hint-vector-doc:{index:04d}",
        "hintCandidateId": hint_candidate_id,
        "sourceCandidateId": source_candidate_id,
        "paperId": "sample-paper",
        "paperRef": "papers_dir/sample.pdf",
        "sourceContentHash": _hash(),
        "page": index,
        "bbox": [10.0, 20.0, 120.0, 180.0],
        "candidateType": "figure_caption_region",
        "sourceRecordSha256": _hash(),
        "documentText": document_text,
        "documentTextHash": _hash_text(document_text),
        "embeddingText": embedding_text,
        "embeddingTextHash": _hash_text(embedding_text),
        "metadata": {
            "retrieval_unit_schema": "visual_retrieval_hint_vector_document.v1",
            "namespace": "labs_visual_retrieval_hint_candidates_v1",
            "allowedUse": "retrieval_hint_only",
            "hintCandidateId": hint_candidate_id,
            "sourceCandidateId": source_candidate_id,
            "paperId": "sample-paper",
            "paperRef": "papers_dir/sample.pdf",
            "sourceContentHash": _hash(),
            "page": index,
            "bbox": [10.0, 20.0, 120.0, 180.0],
            "candidateType": "figure_caption_region",
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
            "productionIndexEligible": False,
            "labsOnly": True,
        },
        "executionPlan": {},
    }


def _records(count: int = 125) -> list[dict[str, object]]:
    return [_record(index) for index in range(1, count + 1)]


def _quality_eval_report(*, passed: bool = True, missing_baseline: int = 0, mutation_count: int = 0) -> dict[str, object]:
    return {
        "schema": "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-labs-vector-index-search-quality-eval.v1",
        "status": "ready" if passed else "blocked",
        "decision": (
            "ready_for_limited_visual_retrieval_hint_production_vector_db_integration_design"
            if passed
            else "keep_in_labs_pending_search_quality_review"
        ),
        "qualityGate": {"passed": passed},
        "counts": {
            "sourcePlannedVectorUpsertRows": 125,
            "sourceCandidateRowsCoveredByTextBaseline": 125 - missing_baseline,
            "sourceCandidateRowsMissingFromTextBaseline": missing_baseline,
            "actualLabsVectorIndexRows": 125,
            "matchedVectorRecordRows": 125,
            "queryRows": 250,
            "textOnlyHitAt5Rows": 133,
            "labsVectorHitAt5Rows": 234,
            "hybridHitAt5Rows": 241,
            "hybridHitAt5LiftRows": 108,
            "rankRegressedRows": 0,
            "textOnlyMrr": 0.445518,
            "labsVectorMrr": 0.788217,
            "hybridMrr": 0.843291,
            "candidateStoreWriteRows": 0,
            "embeddingCallRows": 0,
            "embeddingVectorWriteRows": 0,
            "vectorIndexWriteRows": 0,
            "productionVectorIndexWriteRows": mutation_count,
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
    }


def _candidate_apply_report() -> dict[str, object]:
    return {
        "schema": "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-apply-executor.v1",
        "status": "applied",
        "decision": "applied_limited_visual_retrieval_hint_candidate_store_apply",
        "counts": {
            "appliedCandidateRecordRows": 125,
            "candidateStoreWriteRows": 125,
            "readbackValidatedRows": 125,
            "blockedRows": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
            "productionVectorIndexWriteRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "runtimeVisibleRows": 0,
            "strictEvidenceRows": 0,
            "citationGradeRows": 0,
            "answerableWithoutTextEvidenceRows": 0,
        },
    }


def _labs_apply_report() -> dict[str, object]:
    return {
        "schema": "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-labs-vector-index-apply-executor.v1",
        "status": "applied",
        "decision": "applied_limited_visual_retrieval_hint_candidate_store_labs_vector_index",
        "counts": {
            "appliedLabsVectorRecordRows": 125,
            "readbackValidatedRows": 125,
            "vectorIndexWriteRows": 125,
            "productionVectorIndexWriteRows": 0,
            "candidateStoreWriteRows": 0,
            "externalEmbeddingCallRows": 0,
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
    }


def _dry_run_report(records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema": "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-labs-vector-index-apply-executor-dry-run.v1",
        "status": "ready",
        "decision": "ready_for_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_review",
        "counts": {
            "executorDryRunRows": len(records),
            "plannedVectorUpsertRows": len(records),
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
        "plannedVectorUpsertRecords": records,
    }


def _build(
    records: list[dict[str, object]] | None = None,
    *,
    quality_eval_report: dict[str, object] | None = None,
) -> dict[str, object]:
    records = _records() if records is None else records
    return build_limited_visual_retrieval_hint_production_vector_db_integration_design(
        search_quality_eval_report=quality_eval_report or _quality_eval_report(),
        candidate_store_apply_report=_candidate_apply_report(),
        labs_vector_index_apply_report=_labs_apply_report(),
        labs_vector_index_apply_executor_dry_run=_dry_run_report(records),
        source_search_quality_eval_report_ref=QUALITY_REF,
        source_candidate_store_apply_report_ref=CANDIDATE_APPLY_REF,
        source_labs_vector_index_apply_report_ref=LABS_APPLY_REF,
        source_labs_vector_index_apply_executor_dry_run_ref=LABS_DRY_RUN_REF,
        generated_at="2026-05-28T00:00:00Z",
    )


def test_production_vector_db_integration_design_projects_125_ready_rows() -> None:
    report = _build()

    assert report["schema"] == LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_DESIGN_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["counts"]["productionVectorIntegrationCandidateRows"] == 125
    assert report["counts"]["plannedProductionVectorRecordRows"] == 125
    assert report["counts"]["blockedRows"] == 0
    assert report["counts"]["productionVectorIndexWriteRows"] == 0
    assert report["counts"]["runtimeVisibleRows"] == 0
    assert report["gate"]["passed"] is True
    assert all(row["status"] == PLANNED_STATUS for row in report["rows"])
    assert report["rows"][0]["plannedMetadata"]["source_type"] == "visual_retrieval_hint"
    assert report["rows"][0]["policy"]["allowedUse"] == "retrieval_hint_only"

    validation = validate_payload(
        report,
        LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_DESIGN_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors


def test_production_vector_db_integration_design_blocks_failed_quality_gate() -> None:
    report = _build(quality_eval_report=_quality_eval_report(passed=False))

    assert report["status"] == "blocked"
    assert report["counts"]["plannedProductionVectorRecordRows"] == 0
    assert report["counts"]["blockedRows"] == 125
    assert "labs_vector_index_search_quality_gate_not_passed" in report["technicalBlockers"]


def test_production_vector_db_integration_design_blocks_missing_baseline_coverage() -> None:
    report = _build(quality_eval_report=_quality_eval_report(missing_baseline=1))

    assert report["status"] == "blocked"
    assert report["counts"]["sourceCandidateRowsMissingFromTextBaseline"] == 1
    assert report["counts"]["blockedRows"] == 125
    assert "source_baseline_coverage_incomplete" in report["technicalBlockers"]


def test_production_vector_db_integration_design_blocks_unsafe_policy_row() -> None:
    records = _records()
    records[0]["policy"] = dict(records[0]["policy"])
    records[0]["policy"]["runtimeVisible"] = True

    report = _build(records)

    assert report["status"] == "blocked"
    assert report["counts"]["plannedProductionVectorRecordRows"] == 124
    assert report["counts"]["blockedRows"] == 1
    assert report["counts"]["policyViolationRows"] == 1
    assert "row_policy_not_retrieval_hint_only" in report["technicalBlockers"]


def test_production_vector_db_integration_design_blocks_source_mutation_counter() -> None:
    report = _build(quality_eval_report=_quality_eval_report(mutation_count=1))

    assert report["status"] == "blocked"
    assert report["counts"]["blockedRows"] == 125
    assert "labs_vector_index_search_quality_eval_has_productionVectorIndexWriteRows" in report["technicalBlockers"]
