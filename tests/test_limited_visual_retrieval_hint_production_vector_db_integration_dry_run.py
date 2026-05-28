from __future__ import annotations

import hashlib

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.limited_visual_retrieval_hint_production_vector_db_integration_design import (
    LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_DESIGN_SCHEMA_ID,
    PLANNED_STATUS as DESIGN_PLANNED_STATUS,
    PRODUCTION_NAMESPACE,
    READY_DECISION as DESIGN_READY_DECISION,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_production_vector_db_integration_dry_run import (
    DRY_RUN_READY_STATUS,
    LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_DRY_RUN_SCHEMA_ID,
    READY_DECISION,
    build_limited_visual_retrieval_hint_production_vector_db_integration_dry_run,
)


SOURCE_DESIGN_REF = "eval/knowledgeos/reports/source_design_fixture.v1.json"


def _hash_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash() -> str:
    return "sha256:" + "e" * 64


def _design_row(index: int) -> dict[str, object]:
    document_text = f"Retrieval hint only: sample paper Figure {index} about prompt engineering."
    embedding_text = (
        "allowed_use=retrieval_hint_only | "
        f"paper=sample-paper | type=figure_caption_region | page={index} | "
        "keywords=CLIP, prompt engineering, zero-shot, visual hint | "
        f"{document_text}"
    )
    source_candidate_id = f"visual-layout:sample-paper:figure_caption_region:{index}:aaaaaaaaaaaa{index:04d}"
    hint_candidate_id = f"visual-retrieval-hint:sample-paper:figure_caption_region:{index}:bbbbbbbbbbbb{index:04d}"
    document_id = f"visual-retrieval-hint-production-vector-doc:{index:04d}"
    return {
        "schema": "knowledge-hub.paper.visual-retrieval-hint-production-vector-integration-design-row.v1",
        "rowId": f"limited-visual-retrieval-hint-production-vector-db-integration-design:{index:04d}",
        "status": DESIGN_PLANNED_STATUS,
        "productionNamespace": PRODUCTION_NAMESPACE,
        "productionVectorDocumentId": document_id,
        "hintCandidateId": hint_candidate_id,
        "sourceCandidateId": source_candidate_id,
        "sourceContentHash": _hash(),
        "paperId": "sample-paper",
        "paperRef": "papers_dir/sample.pdf",
        "page": index,
        "bbox": [10.0, 20.0, 120.0, 180.0],
        "candidateType": "figure_caption_region",
        "derivedTextForRetrieval": document_text,
        "retrievalKeywords": ["CLIP", "prompt engineering", "zero-shot", "visual hint"],
        "documentTextHash": _hash_text(document_text),
        "embeddingTextHash": _hash_text(embedding_text),
        "plannedDocumentText": document_text,
        "plannedEmbeddingText": embedding_text,
        "plannedMetadata": {
            "retrieval_unit_schema": "visual_retrieval_hint_production_vector_document_design.v1",
            "namespace": PRODUCTION_NAMESPACE,
            "source_type": "visual_retrieval_hint",
            "retrieval_unit_kind": "candidate_discovery_signal",
            "allowed_use": "retrieval_hint_only",
            "allowedUse": "retrieval_hint_only",
            "document_id": document_id,
            "hint_candidate_id": hint_candidate_id,
            "source_candidate_id": source_candidate_id,
            "paper_id": "sample-paper",
            "paper_ref": "papers_dir/sample.pdf",
            "source_content_hash": _hash(),
            "page": index,
            "bbox": [10.0, 20.0, 120.0, 180.0],
            "candidate_type": "figure_caption_region",
            "strict_evidence": False,
            "strictEvidence": False,
            "citation_grade": False,
            "citationGrade": False,
            "answerable_without_text_evidence": False,
            "answerableWithoutTextEvidence": False,
            "runtime_visible": False,
            "runtimeVisible": False,
            "index_eligible": False,
            "indexEligible": False,
            "production_index_eligible": False,
            "candidate_discovery_only": True,
        },
        "sourceRefs": {
            "sourceQualityEvalReportRef": "eval/knowledgeos/reports/search_quality_fixture.v1.json",
            "sourceCandidateStoreApplyReportRef": "eval/knowledgeos/reports/candidate_apply_fixture.v1.json",
            "sourceLabsVectorIndexApplyReportRef": "eval/knowledgeos/reports/labs_apply_fixture.v1.json",
            "sourceLabsVectorIndexApplyExecutorDryRunRef": "eval/knowledgeos/reports/labs_dry_run_fixture.v1.json",
            "sourceLayoutCandidateReportRef": "eval/knowledgeos/reports/layout_fixture.v1.json",
        },
        "routeDesign": {
            "targetVectorDatabaseClass": "knowledge_hub.infrastructure.persistence.vector.VectorDatabase",
            "writeMethod": "VectorDatabase.add_documents",
            "proposedCollectionName": "knowledge_hub_visual_retrieval_hints",
            "proposedStoreRef": "config.vector_db_path/visual_retrieval_hints",
            "routingMode": "candidate_discovery_only",
            "mergePolicy": "visual_hint_candidates_may_expand_candidates_but_must_not_be_citation_evidence",
            "answerEvidenceGate": "strict_text_evidence_required_after_candidate_discovery",
            "defaultRuntimeExposure": False,
        },
        "policy": {
            "allowedUse": "retrieval_hint_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": False,
            "indexEligible": False,
            "productionIndexEligible": False,
            "candidateDiscoveryOnly": True,
        },
        "executionPlan": {
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
        "blockers": [],
    }


def _design_rows(count: int = 125) -> list[dict[str, object]]:
    return [_design_row(index) for index in range(1, count + 1)]


def _design_report(
    rows: list[dict[str, object]] | None = None,
    *,
    status: str = "ready",
    decision: str = DESIGN_READY_DECISION,
    mutation_count: int = 0,
) -> dict[str, object]:
    rows = _design_rows() if rows is None else rows
    return {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_DESIGN_SCHEMA_ID,
        "status": status,
        "decision": decision,
        "counts": {
            "productionVectorIntegrationCandidateRows": len(rows),
            "plannedProductionVectorRecordRows": len(rows),
            "candidateDiscoveryOnlyRows": len(rows),
            "blockedRows": 0,
            "policyViolationRows": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
            "qualityEvalHybridHitAt5Rows": 241,
            "qualityEvalHybridHitAt5LiftRows": 108,
            "candidateStoreWriteRows": 0,
            "embeddingCallRows": 0,
            "embeddingVectorWriteRows": 0,
            "vectorIndexWriteRows": 0,
            "productionVectorIndexWriteRows": mutation_count,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "operationalSearchIndexQueryRows": 0,
            "runtimeVisibleRows": 0,
            "strictEvidenceRows": 0,
            "citationGradeRows": 0,
            "answerableWithoutTextEvidenceRows": 0,
            "graphDbWriteRows": 0,
            "ontologyWriteRows": 0,
            "memoryCardWriteRows": 0,
            "clusterWriteRows": 0,
        },
        "rows": rows,
        "productionVectorRecordPreviews": rows,
    }


def _build(design_report: dict[str, object] | None = None) -> dict[str, object]:
    return build_limited_visual_retrieval_hint_production_vector_db_integration_dry_run(
        production_vector_db_integration_design=design_report or _design_report(),
        source_production_vector_db_integration_design_ref=SOURCE_DESIGN_REF,
        generated_at="2026-05-28T00:00:00Z",
    )


def test_production_vector_db_integration_dry_run_projects_125_ready_records() -> None:
    report = _build()

    assert report["schema"] == LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_DRY_RUN_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["counts"]["plannedProductionVectorRecordRows"] == 125
    assert report["counts"]["futureApplyCandidateRows"] == 125
    assert report["counts"]["blockedRows"] == 0
    assert report["counts"]["productionVectorIndexWriteRows"] == 0
    assert report["counts"]["databaseMutationRows"] == 0
    assert report["counts"]["runtimeVisibleRows"] == 0
    assert report["gate"]["passed"] is True
    assert len(report["plannedProductionVectorRecords"]) == 125
    assert all(row["status"] == DRY_RUN_READY_STATUS for row in report["dryRunRowsDetail"])
    assert report["plannedProductionVectorRecords"][0]["targetWriteMethod"] == "VectorDatabase.add_documents"
    assert report["plannedProductionVectorRecords"][0]["policy"]["allowedUse"] == "retrieval_hint_only"
    assert report["plannedProductionVectorRecords"][0]["executionPlan"]["actualProductionVectorIndexWrite"] is False

    validation = validate_payload(
        report,
        LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_DRY_RUN_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors


def test_production_vector_db_integration_dry_run_blocks_failed_source_design_gate() -> None:
    report = _build(_design_report(status="blocked", decision="blocked"))

    assert report["status"] == "blocked"
    assert report["counts"]["plannedProductionVectorRecordRows"] == 0
    assert report["counts"]["blockedRows"] == 125
    assert "production_vector_db_integration_design_not_ready" in report["technicalBlockers"]


def test_production_vector_db_integration_dry_run_blocks_unsafe_policy_row() -> None:
    rows = _design_rows()
    rows[0]["policy"] = dict(rows[0]["policy"])
    rows[0]["policy"]["runtimeVisible"] = True

    report = _build(_design_report(rows))

    assert report["status"] == "blocked"
    assert report["counts"]["plannedProductionVectorRecordRows"] == 124
    assert report["counts"]["blockedRows"] == 1
    assert report["counts"]["policyViolationRows"] == 1
    assert "row_policy_not_retrieval_hint_only" in report["technicalBlockers"]


def test_production_vector_db_integration_dry_run_blocks_missing_required_contract_field() -> None:
    rows = _design_rows()
    rows[0] = dict(rows[0])
    rows[0]["sourceContentHash"] = ""

    report = _build(_design_report(rows))

    assert report["status"] == "blocked"
    assert report["counts"]["plannedProductionVectorRecordRows"] == 124
    assert report["counts"]["contractViolationRows"] == 1
    assert "missing_sourceContentHash" in report["technicalBlockers"]


def test_production_vector_db_integration_dry_run_blocks_source_mutation_counter() -> None:
    report = _build(_design_report(mutation_count=1))

    assert report["status"] == "blocked"
    assert report["counts"]["plannedProductionVectorRecordRows"] == 0
    assert report["counts"]["blockedRows"] == 125
    assert "source_design_has_productionVectorIndexWriteRows" in report["technicalBlockers"]
