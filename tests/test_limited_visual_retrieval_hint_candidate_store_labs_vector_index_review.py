from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_apply_executor import (
    _idempotency_key_from_record,
    _record_hash,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_dry_run import (
    build_limited_visual_retrieval_hint_candidate_store_labs_vector_index_dry_run,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_review import (
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_REVIEW_SCHEMA_ID,
    REVIEW_STATUS_BLOCKED_CONTRACT,
    REVIEW_STATUS_BLOCKED_MISSING_PLAN_ROW,
    REVIEW_STATUS_BLOCKED_POLICY,
    REVIEW_STATUS_BLOCKED_QUALITY_GATE,
    REVIEW_STATUS_READY,
    review_limited_visual_retrieval_hint_candidate_store_labs_vector_index,
)


SOURCE_READBACK_REF = "eval/knowledgeos/reports/readback_fixture.v1.json"
SOURCE_DRY_RUN_REF = "eval/knowledgeos/reports/labs_vector_dry_run_fixture.v1.json"


def _hash() -> str:
    return "sha256:" + "e" * 64


def _candidate_record(index: int) -> dict[str, object]:
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
            "runtimeVisible": False,
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


def _dry_run_report(tmp_path: Path, records: list[dict[str, object]]) -> dict[str, object]:
    papers_dir = tmp_path / "papers"
    _write_store(papers_dir, records)
    return build_limited_visual_retrieval_hint_candidate_store_labs_vector_index_dry_run(
        readback_review=_readback_review(records),
        source_readback_review_ref=SOURCE_READBACK_REF,
        papers_dir=papers_dir,
        generated_at="2026-05-27T00:00:00Z",
    )


def _review(report: dict[str, object], *, min_hit5: int = 1, min_hit10: int = 1) -> dict[str, object]:
    return review_limited_visual_retrieval_hint_candidate_store_labs_vector_index(
        labs_vector_index_dry_run=report,
        source_labs_vector_index_dry_run_ref=SOURCE_DRY_RUN_REF,
        generated_at="2026-05-27T00:00:00Z",
        min_labs_hit_at5_rows=min_hit5,
        min_labs_hit_at10_rows=min_hit10,
    )


def test_labs_vector_index_review_accepts_ready_dry_run(tmp_path: Path) -> None:
    dry_run = _dry_run_report(tmp_path, [_candidate_record(1), _candidate_record(2)])

    report = _review(dry_run)

    assert report["status"] == "ready"
    assert report["counts"]["reviewRows"] == 2
    assert report["counts"]["reviewReadyRows"] == 2
    assert report["counts"]["applyExecutorDryRunCandidateRows"] == 2
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert report["counts"]["embeddingCallRows"] == 0
    assert report["counts"]["vectorIndexWriteRows"] == 0
    assert {row["reviewStatus"] for row in report["reviewRowsDetail"]} == {REVIEW_STATUS_READY}
    assert validate_payload(
        report,
        LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_labs_vector_index_review_blocks_non_ready_source_dry_run(tmp_path: Path) -> None:
    dry_run = _dry_run_report(tmp_path, [_candidate_record(1)])
    dry_run["status"] = "blocked"

    report = _review(dry_run)

    assert report["status"] == "blocked"
    assert report["counts"]["schemaViolationCount"] > 0
    assert report["gate"]["readyForLabsVectorIndexApplyExecutorDryRun"] is False


def test_labs_vector_index_review_blocks_policy_violation(tmp_path: Path) -> None:
    dry_run = _dry_run_report(tmp_path, [_candidate_record(1)])
    dry_run["plannedVectorDocuments"][0]["policy"]["runtimeVisible"] = True

    report = _review(dry_run)

    assert report["status"] == "blocked"
    assert report["counts"]["blockedRows"] == 1
    assert report["reviewRowsDetail"][0]["reviewStatus"] == REVIEW_STATUS_BLOCKED_POLICY


def test_labs_vector_index_review_blocks_embedding_hash_mismatch(tmp_path: Path) -> None:
    dry_run = _dry_run_report(tmp_path, [_candidate_record(1)])
    dry_run["plannedVectorDocuments"][0]["embeddingTextHash"] = "sha256:" + "f" * 64

    report = _review(dry_run)

    assert report["status"] == "blocked"
    assert report["reviewRowsDetail"][0]["reviewStatus"] == REVIEW_STATUS_BLOCKED_CONTRACT
    assert "embeddingTextHashMatches" in report["reviewRowsDetail"][0]["reviewBlockers"]


def test_labs_vector_index_review_blocks_missing_plan_row(tmp_path: Path) -> None:
    dry_run = _dry_run_report(tmp_path, [_candidate_record(1)])
    dry_run["planRows"] = []

    report = _review(dry_run)

    assert report["status"] == "blocked"
    assert report["reviewRowsDetail"][0]["reviewStatus"] == REVIEW_STATUS_BLOCKED_MISSING_PLAN_ROW


def test_labs_vector_index_review_blocks_quality_gate_failure(tmp_path: Path) -> None:
    dry_run = _dry_run_report(tmp_path, [_candidate_record(1), _candidate_record(2)])
    too_high_hit5 = int(dry_run["counts"]["labsHitAt5Rows"]) + 1
    too_high_hit10 = int(dry_run["counts"]["labsHitAt10Rows"]) + 1

    report = _review(dry_run, min_hit5=too_high_hit5, min_hit10=too_high_hit10)

    assert report["status"] == "blocked"
    assert report["gate"]["qualityGatePassed"] is False
    assert report["counts"]["reviewReadyRows"] == 0
    assert {row["reviewStatus"] for row in report["reviewRowsDetail"]} == {REVIEW_STATUS_BLOCKED_QUALITY_GATE}
