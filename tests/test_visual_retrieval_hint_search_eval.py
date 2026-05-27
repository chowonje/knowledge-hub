from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_expansion_dry_run import (
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DRY_RUN_SCHEMA_ID,
)
from knowledge_hub.papers.visual_retrieval_hint_search_eval import (
    VISUAL_RETRIEVAL_HINT_SEARCH_EVAL_SCHEMA_ID,
    build_visual_retrieval_hint_search_eval,
    write_visual_retrieval_hint_search_eval,
)
from knowledge_hub.papers.visual_retrieval_hint_usefulness_eval import (
    VISUAL_RETRIEVAL_HINT_USEFULNESS_EVAL_SCHEMA_ID,
)


def _hash() -> str:
    return "sha256:" + "4" * 64


def _source_candidate_id() -> str:
    return "visual-layout:sample-paper:figure_caption_region:2:aaaaaaaaaaaaaaaa"


def _layout_row(candidate_id: str, *, nearby_text: str, candidate_type: str = "figure_caption_region") -> dict[str, object]:
    return {
        "schema": "knowledge-hub.paper.visual-layout-candidate-row.v1",
        "candidateId": candidate_id,
        "paperId": "sample-paper",
        "paperRef": "papers_dir/sample.pdf",
        "sourceContentHash": _hash(),
        "page": 2,
        "bbox": [10.0, 20.0, 120.0, 180.0],
        "candidateType": candidate_type,
        "textContext": {
            "nearbyText": nearby_text,
            "captionText": "",
            "headingPath": ["Experiments"],
        },
        "visualContext": {
            "cropRef": "papers_dir/visual_layout_planned_crops/sample/page-2/sample.png",
            "imageHash": "",
            "pageImageRequired": True,
        },
        "retrievalHintPlan": {
            "targetDerivedTextField": "derivedTextForRetrieval",
            "allowedUse": "retrieval_hint_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
        },
        "provenance": {
            "sourceContentHash": _hash(),
            "page": 2,
            "bbox": [10.0, 20.0, 120.0, 180.0],
            "extractionMethod": "fixture",
        },
        "blockerReason": "",
    }


def _layout_report() -> dict[str, object]:
    return {
        "schema": "knowledge-hub.paper.visual-layout-candidate-list-report.v1",
        "status": "ready",
        "candidateRowsDetail": [
            _layout_row(
                _source_candidate_id(),
                nearby_text="A nearby paragraph discusses implementation notes and setup.",
            ),
            _layout_row(
                "visual-layout:sample-paper:table_region:3:bbbbbbbbbbbbbbbb",
                nearby_text="A table paragraph discusses validation accuracy and model size.",
                candidate_type="table_region",
            ),
        ],
    }


def _dry_row(source_candidate_id: str) -> dict[str, object]:
    return {
        "schema": "knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-dry-run-row.v1",
        "dryRunRowId": "visual-retrieval-hint-dry-run:fixture",
        "hintCandidateId": "visual-retrieval-hint:sample-paper:figure_caption_region:2:abcdefabcdefabcd",
        "sourceCandidateId": source_candidate_id,
        "paperId": "sample-paper",
        "paperRef": "papers_dir/sample.pdf",
        "sourceContentHash": _hash(),
        "page": 2,
        "bbox": [10.0, 20.0, 120.0, 180.0],
        "candidateType": "figure_caption_region",
        "plannedStoreRef": "papers_dir/visual_retrieval_hints/visual_retrieval_hint_candidates.v1.jsonl",
        "idempotencyKey": "visual-retrieval-hint-idempotency:fixture",
        "plannedJsonlRecordSha256": "sha256:" + "5" * 64,
        "plannedJsonlRecordPreview": {
            "schema": "knowledge-hub.paper.visual-retrieval-hint-candidate-row.v1",
            "hintCandidateId": "visual-retrieval-hint:sample-paper:figure_caption_region:2:abcdefabcdefabcd",
            "sourceCandidateId": source_candidate_id,
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


def _dry_report(source_candidate_id: str) -> dict[str, object]:
    return {
        "schema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DRY_RUN_SCHEMA_ID,
        "status": "ready",
        "counts": {
            "dryRunRows": 1,
            "blockedRows": 0,
            "candidateStoreWriteRows": 0,
        },
        "dryRunRowsDetail": [_dry_row(source_candidate_id)],
    }


def _usefulness_report(*, status: str = "ready", blocked_rows: int = 0) -> dict[str, object]:
    return {
        "schema": VISUAL_RETRIEVAL_HINT_USEFULNESS_EVAL_SCHEMA_ID,
        "status": status,
        "_sourceReportRef": "eval/knowledgeos/reports/visual_retrieval_hint_usefulness_eval_fixture.v1.json",
        "counts": {
            "inputHintRows": 1,
            "highUsefulnessRows": 1,
            "mediumUsefulnessRows": 0,
            "blockedRows": blocked_rows,
        },
        "evalRowsDetail": [
            {
                "schema": "knowledge-hub.paper.visual-retrieval-hint-usefulness-eval-row.v1",
                "sourceCandidateId": _source_candidate_id(),
                "hintCandidateId": "visual-retrieval-hint:sample-paper:figure_caption_region:2:abcdefabcdefabcd",
                "usefulness": {"tier": "high"},
                "blockerReason": "",
            }
        ],
    }


def test_search_eval_compares_text_only_and_augmented_target_ranks() -> None:
    report = build_visual_retrieval_hint_search_eval(
        _layout_report(),
        _usefulness_report(),
        [
            (
                "eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_dry_run_fixture.v1.json",
                _dry_report(_source_candidate_id()),
            )
        ],
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["schema"] == VISUAL_RETRIEVAL_HINT_SEARCH_EVAL_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == "ready_for_limited_visual_retrieval_hint_candidate_store_apply_design"
    assert report["nextRecommendedTranche"] == "limited_visual_retrieval_hint_candidate_store_apply_design"
    assert report["counts"]["inputHintRows"] == 1
    assert report["counts"]["queryRows"] == 2
    assert report["counts"]["rankImprovedRows"] == 2
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert report["scope"]["candidateStoreWriteRows"] == 0
    assert report["scope"]["vectorIndexing"] is False
    assert report["scope"]["operationalSearchIndexQueryRows"] == 0
    assert report["scope"]["answerGenerationRows"] == 0

    row = report["queryRowsDetail"][0]
    assert row["textOnlyResult"]["hitAt5"] is False
    assert row["visualHintAugmentedResult"]["hitAt5"] is True
    assert row["policy"]["strictEvidence"] is False
    assert row["policy"]["citationGrade"] is False
    assert row["policy"]["answerableWithoutTextEvidence"] is False

    validation = validate_payload(report, VISUAL_RETRIEVAL_HINT_SEARCH_EVAL_SCHEMA_ID, strict=True)
    assert validation.ok, validation.errors


def test_search_eval_blocks_usefulness_report_with_blockers() -> None:
    report = build_visual_retrieval_hint_search_eval(
        _layout_report(),
        _usefulness_report(blocked_rows=1),
        [
            (
                "eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_dry_run_fixture.v1.json",
                _dry_report(_source_candidate_id()),
            )
        ],
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["decision"] == "blocked"
    assert report["counts"]["candidateStoreWriteRows"] == 0


def test_search_eval_blocks_source_report_with_store_writes() -> None:
    dry = _dry_report(_source_candidate_id())
    dry["counts"]["candidateStoreWriteRows"] = 1

    report = build_visual_retrieval_hint_search_eval(
        _layout_report(),
        _usefulness_report(),
        [("eval/knowledgeos/reports/unsafe.v1.json", dry)],
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["sourceBlockers"] == ["dry_run_has_store_writes:eval/knowledgeos/reports/unsafe.v1.json"]
    assert report["counts"]["candidateStoreWriteRows"] == 0


def test_search_eval_writer_uses_sanitized_refs(tmp_path: Path) -> None:
    report = build_visual_retrieval_hint_search_eval(
        _layout_report(),
        _usefulness_report(),
        [
            (
                "eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_dry_run_fixture.v1.json",
                _dry_report(_source_candidate_id()),
            )
        ],
        generated_at="2026-05-27T00:00:00Z",
    )
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    write_visual_retrieval_hint_search_eval(report, report_json=json_path, report_md=md_path)

    combined = json_path.read_text(encoding="utf-8") + md_path.read_text(encoding="utf-8")
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["sourceDryRunReports"][0]["reportRef"].startswith("eval/knowledgeos/reports/")
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "i" + "Cloud" not in combined
