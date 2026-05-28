from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_apply_executor import (
    _idempotency_key_from_record,
    _record_hash,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_dry_run import (
    LABS_NAMESPACE,
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_DRY_RUN_SCHEMA_ID,
    VECTOR_DOC_STATUS_PLANNED,
    build_limited_visual_retrieval_hint_candidate_store_labs_vector_index_dry_run,
)


SOURCE_READBACK_REF = "eval/knowledgeos/reports/readback_fixture.v1.json"


def _hash() -> str:
    return "sha256:" + "d" * 64


def _candidate_record(index: int, *, runtime_visible: bool = False) -> dict[str, object]:
    return {
        "schema": "knowledge-hub.paper.visual-retrieval-hint-candidate-row.v1",
        "hintCandidateId": f"visual-retrieval-hint:sample-paper:figure_caption_region:{index}:abcdefabcdef{index:04d}",
        "sourceCandidateId": f"visual-layout:sample-paper:figure_caption_region:{index}:aaaaaaaaaaaa{index:04d}",
        "paperId": "sample-paper",
        "paperRef": "papers_dir/sample.pdf",
        "sourceContentHash": _hash(),
        "page": index,
        "bbox": [10.0, 20.0, 120.0, 180.0],
        "candidateType": "figure_caption_region",
        "sourceAttachmentRef": "eval/knowledgeos/reports/pack/assets/01.png",
        "derivedTextForRetrieval": f"Retrieval hint only: sample visual region {index} about ReLU CIFAR-10 training error.",
        "visibleText": "Visible fragments include: ReLU, CIFAR-10, training error.",
        "retrievalKeywords": ["ReLU", "CIFAR-10", f"training error {index}"],
        "uncertainty": "Low.",
        "limitations": "Retrieval hint only, not evidence.",
        "policy": {
            "allowedUse": "retrieval_hint_only",
            "targetDerivedTextField": "derivedTextForRetrieval",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": runtime_visible,
            "indexEligible": False,
            "requiresFutureIndexEligibilityGate": True,
            "answerabilityGateBypassAllowed": False,
        },
        "provenance": {
            "sourceCandidateId": f"visual-layout:sample-paper:figure_caption_region:{index}:aaaaaaaaaaaa{index:04d}",
            "sourceContentHash": _hash(),
            "page": index,
            "bbox": [10.0, 20.0, 120.0, 180.0],
        },
    }


def _write_store(papers_dir: Path, records: list[dict[str, object]]) -> None:
    store = papers_dir / "visual_retrieval_hints" / "visual_retrieval_hint_candidates.v1.jsonl"
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _readback_row(record: dict[str, object], index: int) -> dict[str, object]:
    return {
        "readbackReviewRowId": f"limited-visual-retrieval-hint-candidate-store-apply-readback-review:{index:04d}",
        "hintCandidateId": record["hintCandidateId"],
        "sourceCandidateId": record["sourceCandidateId"],
        "paperId": record["paperId"],
        "paperRef": record["paperRef"],
        "sourceContentHash": record["sourceContentHash"],
        "page": record["page"],
        "bbox": record["bbox"],
        "candidateType": record["candidateType"],
        "idempotencyKey": _idempotency_key_from_record(record),
        "expectedRecordSha256": _record_hash(record),
        "storedRecordSha256": _record_hash(record),
        "matchingStoreRecordRows": 1,
        "readbackValidated": True,
        "indexEligible": False,
        "runtimeVisible": False,
        "strictEvidence": False,
        "citationGrade": False,
        "answerableWithoutTextEvidence": False,
        "readbackStatus": "readback_validated_candidate_record",
        "readbackBlockers": [],
        "checks": {},
    }


def _readback_review(records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema": "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-apply-readback-review.v1",
        "status": "ready",
        "decision": "ready_for_limited_visual_retrieval_hint_candidate_store_labs_vector_index_dry_run",
        "counts": {
            "expectedCandidateRows": len(records),
            "storeRows": len(records),
            "readbackValidatedRows": len(records),
            "blockedRows": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
        },
        "rows": [_readback_row(record, index + 1) for index, record in enumerate(records)],
    }


def test_labs_vector_index_dry_run_plans_documents_without_indexing(tmp_path: Path) -> None:
    records = [_candidate_record(1), _candidate_record(2)]
    _write_store(tmp_path / "papers", records)

    report = build_limited_visual_retrieval_hint_candidate_store_labs_vector_index_dry_run(
        readback_review=_readback_review(records),
        source_readback_review_ref=SOURCE_READBACK_REF,
        papers_dir=tmp_path / "papers",
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "ready"
    assert report["counts"]["plannedVectorDocumentRows"] == 2
    assert report["counts"]["labsQueryRows"] == 4
    assert report["counts"]["vectorIndexWriteRows"] == 0
    assert report["counts"]["embeddingCallRows"] == 0
    assert report["counts"]["blockedRows"] == 0
    assert {row["planStatus"] for row in report["planRows"]} == {VECTOR_DOC_STATUS_PLANNED}
    assert {doc["namespace"] for doc in report["plannedVectorDocuments"]} == {LABS_NAMESPACE}
    assert validate_payload(
        report,
        LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok


def test_labs_vector_index_dry_run_blocks_missing_store_record(tmp_path: Path) -> None:
    expected = [_candidate_record(1), _candidate_record(2)]
    _write_store(tmp_path / "papers", expected[:1])

    report = build_limited_visual_retrieval_hint_candidate_store_labs_vector_index_dry_run(
        readback_review=_readback_review(expected),
        source_readback_review_ref=SOURCE_READBACK_REF,
        papers_dir=tmp_path / "papers",
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["plannedVectorDocumentRows"] == 1
    assert report["counts"]["blockedRows"] == 1


def test_labs_vector_index_dry_run_blocks_policy_violation(tmp_path: Path) -> None:
    expected = [_candidate_record(1)]
    _write_store(tmp_path / "papers", [_candidate_record(1, runtime_visible=True)])

    report = build_limited_visual_retrieval_hint_candidate_store_labs_vector_index_dry_run(
        readback_review=_readback_review(expected),
        source_readback_review_ref=SOURCE_READBACK_REF,
        papers_dir=tmp_path / "papers",
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["plannedVectorDocumentRows"] == 0
    assert report["counts"]["blockedRows"] == 1
    assert report["counts"]["vectorIndexWriteRows"] == 0
