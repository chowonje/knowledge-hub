from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_expansion_dry_run import (
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DRY_RUN_SCHEMA_ID,
)
from knowledge_hub.papers.visual_retrieval_hint_usefulness_eval import (
    VISUAL_RETRIEVAL_HINT_USEFULNESS_EVAL_SCHEMA_ID,
    build_visual_retrieval_hint_usefulness_eval,
    write_visual_retrieval_hint_usefulness_eval,
)


def _hash() -> str:
    return "sha256:" + "2" * 64


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
                "visual-layout:sample-paper:figure_caption_region:2:aaaaaaaaaaaaaaaa",
                nearby_text="A nearby paragraph discusses training curves and classification.",
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
        "plannedJsonlRecordSha256": "sha256:" + "3" * 64,
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


def test_usefulness_eval_compares_text_only_and_augmented_proxy() -> None:
    report = build_visual_retrieval_hint_usefulness_eval(
        _layout_report(),
        [
            (
                "eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_dry_run_fixture.v1.json",
                _dry_report("visual-layout:sample-paper:figure_caption_region:2:aaaaaaaaaaaaaaaa"),
            )
        ],
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["schema"] == VISUAL_RETRIEVAL_HINT_USEFULNESS_EVAL_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == "ready_for_targeted_visual_retrieval_hint_search_eval"
    assert report["nextRecommendedTranche"] == "targeted_visual_retrieval_hint_search_eval"
    assert report["counts"]["inputHintRows"] == 1
    assert report["counts"]["evaluatedRows"] == 1
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert report["counts"]["indexEligibleRows"] == 0
    assert report["scope"]["searchIndexQueryRows"] == 0
    assert report["scope"]["answerGenerationRows"] == 0

    row = report["evalRowsDetail"][0]
    assert row["visualHint"]["novelRetrievalKeywordCount"] >= 3
    assert row["proxyProbe"]["augmentedTop5Hit"] is True
    assert row["proxyProbe"]["rankDelta"] > 0
    assert row["policy"]["strictEvidence"] is False
    assert row["policy"]["citationGrade"] is False

    validation = validate_payload(report, VISUAL_RETRIEVAL_HINT_USEFULNESS_EVAL_SCHEMA_ID, strict=True)
    assert validation.ok, validation.errors


def test_usefulness_eval_blocks_missing_layout_candidate() -> None:
    report = build_visual_retrieval_hint_usefulness_eval(
        _layout_report(),
        [
            (
                "eval/knowledgeos/reports/missing.v1.json",
                _dry_report("visual-layout:sample-paper:figure_caption_region:9:missingmissing"),
            )
        ],
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["decision"] == "blocked"
    assert report["counts"]["blockedRows"] == 1
    assert report["evalRowsDetail"][0]["blockerReason"] == "missing_layout_candidate_context"


def test_usefulness_eval_blocks_source_report_with_store_writes() -> None:
    dry = _dry_report("visual-layout:sample-paper:figure_caption_region:2:aaaaaaaaaaaaaaaa")
    dry["counts"]["candidateStoreWriteRows"] = 1

    report = build_visual_retrieval_hint_usefulness_eval(
        _layout_report(),
        [("eval/knowledgeos/reports/unsafe.v1.json", dry)],
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["sourceBlockers"] == [
        "dry_run_has_store_writes:eval/knowledgeos/reports/unsafe.v1.json"
    ]
    assert report["counts"]["candidateStoreWriteRows"] == 0


def test_usefulness_eval_writer_uses_sanitized_refs(tmp_path: Path) -> None:
    report = build_visual_retrieval_hint_usefulness_eval(
        _layout_report(),
        [
            (
                "eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_dry_run_fixture.v1.json",
                _dry_report("visual-layout:sample-paper:figure_caption_region:2:aaaaaaaaaaaaaaaa"),
            )
        ],
        generated_at="2026-05-27T00:00:00Z",
    )
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    write_visual_retrieval_hint_usefulness_eval(report, report_json=json_path, report_md=md_path)

    combined = json_path.read_text(encoding="utf-8") + md_path.read_text(encoding="utf-8")
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["sourceDryRunReports"][0]["reportRef"].startswith("eval/knowledgeos/reports/")
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "i" + "Cloud" not in combined
