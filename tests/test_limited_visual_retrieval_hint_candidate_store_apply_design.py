from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_apply_design import (
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_DESIGN_SCHEMA_ID,
    build_limited_visual_retrieval_hint_candidate_store_apply_design,
    write_limited_visual_retrieval_hint_candidate_store_apply_design,
)
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_expansion_dry_run import (
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DRY_RUN_SCHEMA_ID,
)


def _hash() -> str:
    return "sha256:" + "6" * 64


def _hint_id() -> str:
    return "visual-retrieval-hint:sample-paper:figure_caption_region:2:abcdefabcdefabcd"


def _source_candidate_id() -> str:
    return "visual-layout:sample-paper:figure_caption_region:2:aaaaaaaaaaaaaaaa"


def _query_row(index: int, *, regressed: bool = False) -> dict[str, object]:
    return {
        "schema": "knowledge-hub.paper.visual-retrieval-hint-search-eval-query-row.v1",
        "queryRowId": f"visual-retrieval-hint-search-query:{index:020d}",
        "queryKind": "keyword_lookup" if index == 1 else "natural_lookup",
        "query": "relu tanh cifar training error",
        "hintCandidateId": _hint_id(),
        "sourceCandidateId": _source_candidate_id(),
        "paperId": "sample-paper",
        "paperRef": "papers_dir/sample.pdf",
        "sourceContentHash": _hash(),
        "page": 2,
        "bbox": [10.0, 20.0, 120.0, 180.0],
        "candidateType": "figure_caption_region",
        "sourceDryRunReportRef": "eval/knowledgeos/reports/dry_run_fixture.v1.json",
        "expectedTarget": {
            "sourceCandidateId": _source_candidate_id(),
            "hintCandidateId": _hint_id(),
        },
        "textOnlyResult": {
            "rank": 20,
            "score": 1.0,
            "hitAt1": False,
            "hitAt5": False,
            "hitAt10": False,
        },
        "visualHintAugmentedResult": {
            "rank": 1 if not regressed else 30,
            "score": 5.0,
            "hitAt1": not regressed,
            "hitAt5": not regressed,
            "hitAt10": not regressed,
        },
        "comparison": {
            "rankDelta": 19 if not regressed else -10,
            "improved": not regressed,
            "regressed": regressed,
            "unchanged": False,
            "liftBucket": "moderate_lift" if not regressed else "regression",
        },
        "policy": {
            "allowedUse": "retrieval_hint_search_utility_estimation_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": False,
            "indexEligible": False,
        },
        "blockerReason": "",
    }


def _search_eval_report(*, regressed: bool = False) -> dict[str, object]:
    return {
        "schema": "knowledge-hub.paper.visual-retrieval-hint-search-eval.v1",
        "status": "ready",
        "decision": "ready_for_limited_visual_retrieval_hint_candidate_store_apply_design",
        "counts": {
            "inputHintRows": 1,
            "queryRows": 2,
            "augmentedHitAt5Rows": 1 if regressed else 2,
            "rankRegressedRows": 1 if regressed else 0,
            "blockedRows": 0,
            "candidateStoreWriteRows": 0,
        },
        "queryRowsDetail": [
            _query_row(1, regressed=regressed),
            _query_row(2, regressed=False),
        ],
    }


def _dry_row() -> dict[str, object]:
    return {
        "schema": "knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-dry-run-row.v1",
        "dryRunRowId": "visual-retrieval-hint-dry-run:fixture",
        "hintCandidateId": _hint_id(),
        "sourceCandidateId": _source_candidate_id(),
        "paperId": "sample-paper",
        "paperRef": "papers_dir/sample.pdf",
        "sourceContentHash": _hash(),
        "page": 2,
        "bbox": [10.0, 20.0, 120.0, 180.0],
        "candidateType": "figure_caption_region",
        "plannedStoreRef": "papers_dir/visual_retrieval_hints/visual_retrieval_hint_candidates.v1.jsonl",
        "idempotencyKey": "visual-retrieval-hint-idempotency:fixture",
        "plannedJsonlRecordSha256": "sha256:" + "7" * 64,
        "plannedJsonlRecordPreview": {
            "schema": "knowledge-hub.paper.visual-retrieval-hint-candidate-row.v1",
            "hintCandidateId": _hint_id(),
            "sourceCandidateId": _source_candidate_id(),
            "paperId": "sample-paper",
            "paperRef": "papers_dir/sample.pdf",
            "sourceContentHash": _hash(),
            "page": 2,
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
                "strictEvidence": False,
                "citationGrade": False,
                "answerableWithoutTextEvidence": False,
                "runtimeVisible": False,
                "indexEligible": False,
                "answerabilityGateBypassAllowed": False,
            },
            "provenance": {},
        },
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


def _dry_report(*, store_writes: int = 0) -> dict[str, object]:
    return {
        "schema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DRY_RUN_SCHEMA_ID,
        "status": "ready",
        "counts": {
            "dryRunRows": 1,
            "plannedWriteRows": 1,
            "blockedRows": 0,
            "candidateStoreWriteRows": store_writes,
        },
        "dryRunRowsDetail": [_dry_row()],
    }


def test_limited_apply_design_projects_search_ready_rows_without_writes() -> None:
    report = build_limited_visual_retrieval_hint_candidate_store_apply_design(
        _search_eval_report(),
        [("eval/knowledgeos/reports/dry_run_fixture.v1.json", _dry_report())],
        source_search_eval_report_ref="eval/knowledgeos/reports/search_eval_fixture.v1.json",
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["schema"] == LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_DESIGN_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == "ready_for_limited_visual_retrieval_hint_candidate_store_apply_review"
    assert report["counts"]["inputHintRows"] == 1
    assert report["counts"]["limitedApplyDesignCandidateRows"] == 1
    assert report["counts"]["plannedSeparateApplyWriteRows"] == 1
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert report["scope"]["candidateStoreWriteRows"] == 0
    assert report["scope"]["vectorIndexing"] is False
    assert report["scope"]["candidateStoreApplyExecutorRows"] == 0

    row = report["applyDesignRowsDetail"][0]
    assert row["applyPlan"]["limitedApplyDesignCandidate"] is True
    assert row["applyPlan"]["applyAllowedByThisReport"] is False
    assert row["applyPlan"]["candidateStoreWrite"] is False
    assert row["policy"]["strictEvidence"] is False

    validation = validate_payload(
        report,
        LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_DESIGN_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors


def test_limited_apply_design_blocks_search_regression() -> None:
    report = build_limited_visual_retrieval_hint_candidate_store_apply_design(
        _search_eval_report(regressed=True),
        [("eval/knowledgeos/reports/dry_run_fixture.v1.json", _dry_report())],
        source_search_eval_report_ref="eval/knowledgeos/reports/search_eval_fixture.v1.json",
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["decision"] == "blocked"
    assert report["counts"]["limitedApplyDesignCandidateRows"] == 0
    assert "search_eval_has_regressions" in report["sourceBlockers"]


def test_limited_apply_design_blocks_dry_run_store_write() -> None:
    report = build_limited_visual_retrieval_hint_candidate_store_apply_design(
        _search_eval_report(),
        [("eval/knowledgeos/reports/unsafe.v1.json", _dry_report(store_writes=1))],
        source_search_eval_report_ref="eval/knowledgeos/reports/search_eval_fixture.v1.json",
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert report["sourceBlockers"] == ["dry_run_has_store_writes:eval/knowledgeos/reports/unsafe.v1.json"]


def test_limited_apply_design_writer_uses_sanitized_refs(tmp_path: Path) -> None:
    report = build_limited_visual_retrieval_hint_candidate_store_apply_design(
        _search_eval_report(),
        [("eval/knowledgeos/reports/dry_run_fixture.v1.json", _dry_report())],
        source_search_eval_report_ref="eval/knowledgeos/reports/search_eval_fixture.v1.json",
        generated_at="2026-05-27T00:00:00Z",
    )
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    write_limited_visual_retrieval_hint_candidate_store_apply_design(
        report,
        report_json=json_path,
        report_md=md_path,
    )

    combined = json_path.read_text(encoding="utf-8") + md_path.read_text(encoding="utf-8")
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["sourceSearchEvalReport"]["reportRef"].startswith("eval/knowledgeos/reports/")
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "i" + "Cloud" not in combined
