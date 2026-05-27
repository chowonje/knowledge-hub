from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack import (
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_RECOMMENDATION_OUTPUT_SCHEMA_ID,
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_REVIEW_PACK_SCHEMA_ID,
)
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_expansion_gpt_recommendation_output_capture import (
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_RECOMMENDATION_OUTPUT_VALIDATION_SCHEMA_ID,
    build_visual_retrieval_hint_candidate_store_expansion_gpt_recommendation_output_validation,
    write_visual_retrieval_hint_candidate_store_expansion_gpt_recommendation_output_validation,
)


def _pack_row(index: int) -> dict[str, object]:
    return {
        "rowNumber": index,
        "decisionRowId": f"visual-retrieval-hint-expansion-decision:{index:020d}",
        "sourceReviewRowId": f"visual-retrieval-hint-expansion-review:{index:020d}",
        "hintCandidateId": f"visual-retrieval-hint:sample:image_region:{index}:aaaaaaaaaaaaaaaa",
        "sourceCandidateId": f"visual-layout:sample-paper:image_region:{index}:aaaaaaaaaaaaaaaa",
        "paperId": "sample-paper",
        "paperRef": "papers_dir/sample.pdf",
        "sourceContentHash": "sha256:" + "6" * 64,
        "page": index,
        "bbox": [10.0, 20.0, 200.0, 260.0],
        "candidateType": "image_region",
        "currentDecision": "hold_pending_human_product_review",
        "derivedTextForRetrievalSnippet": f"Retrieval hint only: sample image {index}.",
        "visibleTextSnippet": f"Visible fragments include sample image {index}.",
        "retrievalKeywordCount": 3,
        "allowedSuggestedDecisions": [
            "approve_store_candidate_only",
            "hold_pending_more_context",
            "reject_visual_hint_candidate",
            "request_recrop_or_reannotation",
        ],
        "gptOutputPolicy": {
            "finalHumanDecision": False,
            "applyAllowed": False,
            "candidateStoreWrite": False,
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": False,
            "indexEligible": False,
        },
    }


def _review_pack(row_count: int = 4) -> dict[str, object]:
    rows = [_pack_row(index) for index in range(1, row_count + 1)]
    return {
        "schema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_REVIEW_PACK_SCHEMA_ID,
        "status": "ready",
        "generatedAt": "2026-05-27T00:00:00Z",
        "decision": "ready_for_manual_gpt_decision_recommendation_run",
        "nextRecommendedTranche": "visual_retrieval_hint_candidate_store_expansion_gpt_recommendation_output_capture",
        "batchBundles": [
            {
                "batchId": "sample_batch_01",
                "batchNumber": 1,
                "rowCount": len(rows),
                "promptRef": (
                    "eval/knowledgeos/reports/"
                    "visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack_001/"
                    "batch_01_prompt.md"
                ),
                "recommendationTemplateRef": (
                    "eval/knowledgeos/reports/"
                    "visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack_001/"
                    "batch_01_recommendation_template.v1.json"
                ),
                "rows": rows,
            }
        ],
    }


def _output(row_count: int = 4) -> dict[str, object]:
    rows = []
    decisions = [
        "approve_store_candidate_only",
        "hold_pending_more_context",
        "request_recrop_or_reannotation",
        "reject_visual_hint_candidate",
    ]
    for index in range(1, row_count + 1):
        rows.append(
            {
                "sourceDecisionRowId": f"visual-retrieval-hint-expansion-decision:{index:020d}",
                "hintCandidateId": f"visual-retrieval-hint:sample:image_region:{index}:aaaaaaaaaaaaaaaa",
                "sourceCandidateId": f"visual-layout:sample-paper:image_region:{index}:aaaaaaaaaaaaaaaa",
                "suggestedDecision": decisions[index - 1],
                "recommendationRationale": f"Sample rationale {index}.",
                "risk": "low: sample risk.",
                "needsHumanCheck": True,
                "finalHumanDecision": False,
                "applyAllowed": False,
                "candidateStoreWrite": False,
                "strictEvidence": False,
                "citationGrade": False,
                "answerableWithoutTextEvidence": False,
                "runtimeVisible": False,
                "indexEligible": False,
            }
        )
    return {
        "schema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_RECOMMENDATION_OUTPUT_SCHEMA_ID,
        "rows": rows,
    }


def test_gpt_recommendation_output_validation_ready_and_advisory_only() -> None:
    report = build_visual_retrieval_hint_candidate_store_expansion_gpt_recommendation_output_validation(
        _output(),
        _review_pack(),
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["schema"] == (
        VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_RECOMMENDATION_OUTPUT_VALIDATION_SCHEMA_ID
    )
    assert report["status"] == "ready"
    assert report["decision"] == "ready_for_project_side_human_decision_synthesis"
    assert report["nextRecommendedTranche"] == "visual_annotation_expansion_pack_design_003"
    assert report["counts"]["sourcePackRows"] == 4
    assert report["counts"]["outputRows"] == 4
    assert report["counts"]["matchedRows"] == 4
    assert report["counts"]["approvedRecommendationRows"] == 1
    assert report["counts"]["holdRecommendationRows"] == 1
    assert report["counts"]["rejectRecommendationRows"] == 1
    assert report["counts"]["recropRecommendationRows"] == 1
    assert report["counts"]["finalHumanDecisionRows"] == 0
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert report["counts"]["strictEvidenceRows"] == 0
    assert report["counts"]["citationGradeRows"] == 0
    assert report["scope"]["writes"] == "report_only"
    assert report["scope"]["modelCalls"] is False
    assert report["scope"]["webModelCalls"] is False
    assert report["scope"]["manualOperatorWebModelRunCompletedExternally"] is True
    assert report["policy"]["projectSideGateOwnsFinalDecision"] is True

    validation = validate_payload(
        report,
        VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_RECOMMENDATION_OUTPUT_VALIDATION_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors


def test_gpt_recommendation_output_validation_blocks_mutation_signal() -> None:
    output = _output()
    output["rows"][0]["candidateStoreWrite"] = True

    report = build_visual_retrieval_hint_candidate_store_expansion_gpt_recommendation_output_validation(
        output,
        _review_pack(),
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["decision"] == "blocked"
    assert report["counts"]["candidateStoreWriteRows"] == 1
    assert report["counts"]["policyViolationRows"] == 1
    assert any(v["kind"] == "candidateStoreWrite_not_false" for v in report["violations"])


def test_gpt_recommendation_output_validation_detects_missing_extra_and_private_path() -> None:
    output = _output()
    output["rows"] = [output["rows"][0], output["rows"][1]]
    output["rows"].append(
        {
            **output["rows"][1],
            "hintCandidateId": "visual-retrieval-hint:extra:image_region:1:bbbbbbbbbbbbbbbb",
            "sourceCandidateId": "visual-layout:extra:image_region:1:bbbbbbbbbbbbbbbb",
            "recommendationRationale": "Private /" + "Users" + "/won/path leak.",
        }
    )

    report = build_visual_retrieval_hint_candidate_store_expansion_gpt_recommendation_output_validation(
        output,
        _review_pack(),
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["missingRows"] == 2
    assert report["counts"]["extraRows"] == 1
    assert report["counts"]["privatePathLeakRows"] == 1
    assert any(v["kind"] == "private_path_leak" for v in report["violations"])


def test_gpt_recommendation_output_validation_writer_uses_sanitized_refs(tmp_path: Path) -> None:
    report = build_visual_retrieval_hint_candidate_store_expansion_gpt_recommendation_output_validation(
        _output(row_count=2),
        _review_pack(row_count=2),
        output_ref=(
            "eval/knowledgeos/reports/"
            "visual_retrieval_hint_candidate_store_expansion_gpt_recommendation_output_001.manual.json"
        ),
        source_gpt_review_pack_ref=(
            "eval/knowledgeos/reports/"
            "visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack.v1.json"
        ),
        generated_at="2026-05-27T00:00:00Z",
    )
    report_json = tmp_path / "validation.json"
    report_md = tmp_path / "validation.md"

    paths = write_visual_retrieval_hint_candidate_store_expansion_gpt_recommendation_output_validation(
        report,
        report_json=report_json,
        report_md=report_md,
    )

    assert Path(paths["json"]).exists()
    parsed = json.loads(report_json.read_text(encoding="utf-8"))
    combined = report_json.read_text(encoding="utf-8") + report_md.read_text(encoding="utf-8")
    assert parsed["sourceRecommendationOutput"]["reportRef"].startswith("eval/knowledgeos/reports/")
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "i" + "Cloud" not in combined
