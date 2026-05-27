from __future__ import annotations

import hashlib
import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_allowlist_apply_executor_dry_run import (
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_ALLOWLIST_APPLY_EXECUTOR_DRY_RUN_SCHEMA_ID,
    build_limited_visual_retrieval_hint_candidate_store_allowlist_apply_executor_dry_run,
    write_limited_visual_retrieval_hint_candidate_store_allowlist_apply_executor_dry_run,
)
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_expansion_dry_run import (
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DRY_RUN_SCHEMA_ID,
)


DRY_RUN_REF = "eval/knowledgeos/reports/dry_run_fixture.v1.json"
PLANNED_STORE_REF = "papers_dir/visual_retrieval_hints/visual_retrieval_hint_candidates.v1.jsonl"


def _hash() -> str:
    return "sha256:" + "a" * 64


def _hint_id(index: int) -> str:
    return f"visual-retrieval-hint:sample-paper:figure_caption_region:{index}:abcdefabcdefab{index:02d}"


def _source_id(index: int) -> str:
    return f"visual-layout:sample-paper:figure_caption_region:{index}:aaaaaaaaaaaaaa{index:02d}"


def _canonical_hash(record: dict[str, object]) -> str:
    line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(line.encode("utf-8")).hexdigest()


def _candidate_record(index: int) -> dict[str, object]:
    return {
        "schema": "knowledge-hub.paper.visual-retrieval-hint-candidate-row.v1",
        "hintCandidateId": _hint_id(index),
        "sourceCandidateId": _source_id(index),
        "paperId": "sample-paper",
        "paperRef": "papers_dir/sample.pdf",
        "sourceContentHash": _hash(),
        "page": index,
        "bbox": [10.0, 20.0, 120.0, 180.0],
        "candidateType": "figure_caption_region",
        "sourceAttachmentRef": "eval/knowledgeos/reports/pack/assets/01.png",
        "derivedTextForRetrieval": "Retrieval hint only: a ReLU tanh CIFAR-10 training error plot.",
        "visibleText": "Visible fragments include: ReLU, tanh, CIFAR-10, training error.",
        "retrievalKeywords": ["ReLU", "tanh", "CIFAR-10", "training error"],
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
            "sourceCandidateId": _source_id(index),
            "sourceContentHash": _hash(),
            "page": index,
            "bbox": [10.0, 20.0, 120.0, 180.0],
        },
    }


def _dry_row(index: int) -> dict[str, object]:
    record = _candidate_record(index)
    return {
        "schema": "knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-dry-run-row.v1",
        "dryRunRowId": f"visual-retrieval-hint-dry-run:fixture{index}",
        "hintCandidateId": _hint_id(index),
        "sourceCandidateId": _source_id(index),
        "paperId": "sample-paper",
        "paperRef": "papers_dir/sample.pdf",
        "sourceContentHash": _hash(),
        "page": index,
        "bbox": [10.0, 20.0, 120.0, 180.0],
        "candidateType": "figure_caption_region",
        "plannedStoreRef": PLANNED_STORE_REF,
        "idempotencyKey": f"visual-retrieval-hint-idempotency:fixture{index}",
        "plannedJsonlRecordSha256": _canonical_hash(record),
        "plannedJsonlRecordPreview": record,
        "dryRunResult": {
            "wouldWriteOnApply": True,
            "actualStoreWrite": False,
            "jsonlSerializable": True,
            "policyCompliant": True,
            "indexEligible": False,
            "runtimeVisible": False,
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
        },
        "provenance": {},
        "blockerReason": "",
    }


def _review_row(index: int, *, allowlisted: bool) -> dict[str, object]:
    dry = _dry_row(index)
    status = "allowlisted_for_apply_executor_dry_run" if allowlisted else "holdout_pending_search_gate_review"
    return {
        "schema": "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-apply-allowlist-review-row.v1",
        "reviewRowId": f"limited-visual-retrieval-hint-apply-allowlist-review:fixture{index}",
        "sourceApplyDesignRowId": f"limited-visual-retrieval-hint-apply-design:fixture{index}",
        "hintCandidateId": _hint_id(index),
        "sourceCandidateId": _source_id(index),
        "paperId": "sample-paper",
        "paperRef": "papers_dir/sample.pdf",
        "sourceContentHash": _hash(),
        "page": index,
        "bbox": [10.0, 20.0, 120.0, 180.0],
        "candidateType": "figure_caption_region",
        "plannedStoreRef": PLANNED_STORE_REF,
        "sourceDryRunReportRef": DRY_RUN_REF,
        "sourceDryRunRowId": dry["dryRunRowId"],
        "idempotencyKey": dry["idempotencyKey"],
        "plannedJsonlRecordSha256": dry["plannedJsonlRecordSha256"],
        "reviewStatus": status,
        "checks": {},
        "searchEvalSummary": {
            "sourceSearchEvalQueryRowIds": [
                f"visual-retrieval-hint-search-query:{index:020d}",
                f"visual-retrieval-hint-search-query:{index + 100:020d}",
            ],
            "queryRows": 2,
            "textOnlyBestRank": 20,
            "visualHintAugmentedBestRank": 1,
            "augmentedHitAt5Rows": 2,
            "improvedQueryRows": 2,
            "regressedQueryRows": 0,
            "strongLiftQueryRows": 1,
            "blockedQueryRows": 0,
            "passesLimitedApplySearchGate": allowlisted,
        },
        "reviewPlan": {
            "applyExecutorDryRunCandidate": allowlisted,
            "holdoutFromApplyExecutorDryRun": not allowlisted,
            "candidateStoreWrite": False,
            "applyAllowedByThisReport": False,
            "requiresSeparateApplyExecutor": True,
            "requiresSeparateIndexingGate": True,
        },
        "policy": {
            "allowedUse": "retrieval_hint_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": False,
            "indexEligible": False,
        },
        "blockerReason": "" if allowlisted else "search_eval_gate_not_passed",
    }


def _allowlist_review_report() -> dict[str, object]:
    return {
        "schema": "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-apply-allowlist-review.v1",
        "status": "ready",
        "decision": "ready_for_limited_visual_retrieval_hint_candidate_store_allowlist_apply_executor_dry_run",
        "counts": {
            "reviewRows": 2,
            "allowlistRows": 1,
            "holdoutRows": 1,
            "applyExecutorDryRunCandidateRows": 1,
            "blockedRows": 0,
            "candidateStoreWriteRows": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
        },
        "reviewRowsDetail": [
            _review_row(1, allowlisted=True),
            _review_row(2, allowlisted=False),
        ],
    }


def _dry_report(*, store_writes: int = 0) -> dict[str, object]:
    return {
        "schema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DRY_RUN_SCHEMA_ID,
        "status": "ready",
        "decision": "ready_for_visual_retrieval_hint_candidate_store_expansion_review",
        "counts": {
            "dryRunRows": 2,
            "plannedWriteRows": 2,
            "blockedRows": 0,
            "candidateStoreWriteRows": store_writes,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
        },
        "dryRunRowsDetail": [_dry_row(1), _dry_row(2)],
    }


def test_allowlist_apply_executor_dry_run_projects_allowlist_only_without_writes() -> None:
    report = build_limited_visual_retrieval_hint_candidate_store_allowlist_apply_executor_dry_run(
        _allowlist_review_report(),
        [(DRY_RUN_REF, _dry_report())],
        source_allowlist_review_report_ref="eval/knowledgeos/reports/allowlist_fixture.v1.json",
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["schema"] == LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_ALLOWLIST_APPLY_EXECUTOR_DRY_RUN_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == "ready_for_limited_visual_retrieval_hint_candidate_store_apply_executor_review"
    assert report["counts"]["sourceAllowlistRows"] == 1
    assert report["counts"]["sourceHoldoutRows"] == 1
    assert report["counts"]["executorDryRunRows"] == 1
    assert report["counts"]["plannedWriteRows"] == 1
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert report["counts"]["candidateStoreApplyRows"] == 0
    assert report["scope"]["vectorIndexing"] is False

    row = report["executorDryRunRowsDetail"][0]
    assert row["hintCandidateId"] == _hint_id(1)
    assert row["plannedJsonlRecordPreview"]["hintCandidateId"] == _hint_id(1)
    assert row["plannedJsonlRecordSha256"] == row["recomputedPlannedJsonlRecordSha256"]
    assert row["executorDryRunResult"]["wouldWriteOnSeparateExplicitApply"] is True
    assert row["executorDryRunResult"]["candidateStoreWrite"] is False
    assert all(row["checks"].values())

    validation = validate_payload(
        report,
        LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_ALLOWLIST_APPLY_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors


def test_allowlist_apply_executor_dry_run_blocks_missing_source_dry_run_row() -> None:
    dry_report = _dry_report()
    dry_report["dryRunRowsDetail"] = []
    dry_report["counts"]["dryRunRows"] = 0
    dry_report["counts"]["plannedWriteRows"] = 0

    report = build_limited_visual_retrieval_hint_candidate_store_allowlist_apply_executor_dry_run(
        _allowlist_review_report(),
        [(DRY_RUN_REF, dry_report)],
        source_allowlist_review_report_ref="eval/knowledgeos/reports/allowlist_fixture.v1.json",
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["plannedWriteRows"] == 0
    assert "sourceDryRunMatched" in report["executorDryRunRowsDetail"][0]["blockerReason"]


def test_allowlist_apply_executor_dry_run_blocks_hash_mismatch() -> None:
    allowlist = _allowlist_review_report()
    allowlist["reviewRowsDetail"][0]["plannedJsonlRecordSha256"] = "sha256:" + "f" * 64

    report = build_limited_visual_retrieval_hint_candidate_store_allowlist_apply_executor_dry_run(
        allowlist,
        [(DRY_RUN_REF, _dry_report())],
        source_allowlist_review_report_ref="eval/knowledgeos/reports/allowlist_fixture.v1.json",
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["plannedWriteRows"] == 0
    assert "plannedJsonlRecordHashMatches" in report["executorDryRunRowsDetail"][0]["blockerReason"]
    assert "recomputedJsonlRecordHashMatches" in report["executorDryRunRowsDetail"][0]["blockerReason"]


def test_allowlist_apply_executor_dry_run_blocks_source_report_with_store_writes() -> None:
    report = build_limited_visual_retrieval_hint_candidate_store_allowlist_apply_executor_dry_run(
        _allowlist_review_report(),
        [(DRY_RUN_REF, _dry_report(store_writes=1))],
        source_allowlist_review_report_ref="eval/knowledgeos/reports/allowlist_fixture.v1.json",
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert report["sourceBlockers"] == [f"dry_run_has_store_writes:{DRY_RUN_REF}"]


def test_allowlist_apply_executor_dry_run_writer_uses_sanitized_refs(tmp_path: Path) -> None:
    report = build_limited_visual_retrieval_hint_candidate_store_allowlist_apply_executor_dry_run(
        _allowlist_review_report(),
        [(DRY_RUN_REF, _dry_report())],
        source_allowlist_review_report_ref="eval/knowledgeos/reports/allowlist_fixture.v1.json",
        generated_at="2026-05-27T00:00:00Z",
    )
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    write_limited_visual_retrieval_hint_candidate_store_allowlist_apply_executor_dry_run(
        report,
        report_json=json_path,
        report_md=md_path,
    )

    combined = json_path.read_text(encoding="utf-8") + md_path.read_text(encoding="utf-8")
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["sourceAllowlistReviewReport"]["reportRef"].startswith("eval/knowledgeos/reports/")
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "i" + "Cloud" not in combined
