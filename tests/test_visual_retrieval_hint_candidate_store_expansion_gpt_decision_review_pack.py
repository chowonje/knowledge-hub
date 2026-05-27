from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack import (
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_RECOMMENDATION_BATCH_TEMPLATE_SCHEMA_ID,
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_RECOMMENDATION_OUTPUT_SCHEMA_ID,
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_REVIEW_PACK_SCHEMA_ID,
    build_gpt_recommendation_batch_template,
    build_visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack,
    render_gpt_review_batch_prompt,
    write_visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack,
)
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_expansion_human_product_decision_record import (
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DECISION_RECORD_SCHEMA_ID,
)
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_expansion_review import (
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_REVIEW_SCHEMA_ID,
)


def _hash() -> str:
    return "sha256:" + "4" * 64


def _decision_row(index: int) -> dict[str, object]:
    return {
        "decisionRowId": f"visual-retrieval-hint-expansion-decision:{index:020d}",
        "sourceReviewRowId": f"visual-retrieval-hint-expansion-review:{index:020d}",
        "hintCandidateId": f"visual-retrieval-hint:sample:image-region:{index}:aaaaaaaaaaaaaaaa",
        "sourceCandidateId": f"visual-layout:sample-paper:image_region:{index}:aaaaaaaaaaaaaaaa",
        "paperId": "sample-paper",
        "paperRef": "papers_dir/sample.pdf",
        "sourceContentHash": _hash(),
        "page": index,
        "bbox": [10.0, 20.0, 200.0, 260.0],
        "candidateType": "image_region",
        "decision": "hold_pending_human_product_review",
    }


def _review_row(index: int) -> dict[str, object]:
    return {
        "reviewRowId": f"visual-retrieval-hint-expansion-review:{index:020d}",
        "hintCandidateId": f"visual-retrieval-hint:sample:image-region:{index}:aaaaaaaaaaaaaaaa",
        "derivedTextForRetrievalSnippet": f"Retrieval hint only: sample row {index} image region.",
        "visibleTextSnippet": f"Visible fragments include sample row {index}.",
        "retrievalKeywordCount": 3,
    }


def _decision_record(row_count: int = 2) -> dict[str, object]:
    rows = [_decision_row(index) for index in range(1, row_count + 1)]
    return {
        "schema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DECISION_RECORD_SCHEMA_ID,
        "status": "decision_record_template_ready",
        "counts": {
            "humanDecisionRows": 0,
            "candidateStoreWriteRows": 0,
        },
        "decisionRowsDetail": rows,
    }


def _review_report(row_count: int = 2) -> dict[str, object]:
    rows = [_review_row(index) for index in range(1, row_count + 1)]
    return {
        "schema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_REVIEW_SCHEMA_ID,
        "status": "ready",
        "counts": {
            "blockedRows": 0,
        },
        "reviewRowsDetail": rows,
    }


def test_gpt_decision_review_pack_batches_rows_without_final_decisions() -> None:
    report = build_visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack(
        _decision_record(row_count=9),
        _review_report(row_count=9),
        batch_size=4,
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["schema"] == VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_REVIEW_PACK_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == "ready_for_manual_gpt_decision_recommendation_run"
    assert report["counts"]["gptReviewRows"] == 9
    assert report["counts"]["batchRows"] == 3
    assert report["counts"]["operatorPromptRows"] == 3
    assert report["counts"]["recommendationTemplateRows"] == 3
    assert report["counts"]["completedGptRecommendationRows"] == 0
    assert report["counts"]["finalHumanDecisionRows"] == 0
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert report["scope"]["modelCalls"] is False
    assert report["scope"]["webModelCalls"] is False
    assert report["scope"]["manualOperatorWebModelRunRequired"] is True

    row = report["batchBundles"][0]["rows"][0]
    assert row["currentDecision"] == "hold_pending_human_product_review"
    assert row["gptOutputPolicy"]["finalHumanDecision"] is False
    assert row["gptOutputPolicy"]["applyAllowed"] is False
    assert row["gptOutputPolicy"]["strictEvidence"] is False
    assert row["derivedTextForRetrievalSnippet"].startswith("Retrieval hint only:")

    validation = validate_payload(
        report,
        VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_REVIEW_PACK_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors


def test_gpt_batch_prompt_and_template_preserve_recommendation_only_policy() -> None:
    report = build_visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack(
        _decision_record(),
        _review_report(),
        generated_at="2026-05-27T00:00:00Z",
    )
    batch = report["batchBundles"][0]

    prompt = render_gpt_review_batch_prompt(batch)
    template = build_gpt_recommendation_batch_template(batch)

    assert "Do not attach images or PDFs" not in prompt
    assert "Do not use outside sources, web search, PDFs, screenshots, full pages, or images." in prompt
    assert "finalHumanDecision=false" in prompt
    assert template["schema"] == (
        VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_RECOMMENDATION_BATCH_TEMPLATE_SCHEMA_ID
    )
    assert template["targetOutputSchema"] == (
        VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_RECOMMENDATION_OUTPUT_SCHEMA_ID
    )
    assert template["rows"][0]["finalHumanDecision"] is False
    assert template["rows"][0]["candidateStoreWrite"] is False
    assert template["rows"][0]["strictEvidence"] is False


def test_gpt_review_pack_blocks_candidate_store_write_signal() -> None:
    decision_record = _decision_record()
    decision_record["counts"]["candidateStoreWriteRows"] = 1

    report = build_visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack(
        decision_record,
        _review_report(),
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["decision"] == "blocked"


def test_gpt_review_pack_detects_private_path_leaks() -> None:
    review_report = _review_report()
    review_report["reviewRowsDetail"][0]["derivedTextForRetrievalSnippet"] = (
        "Retrieval hint only: /" + "Users" + "/won/private.pdf"
    )

    report = build_visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack(
        _decision_record(),
        review_report,
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] == 1


def test_gpt_review_pack_writer_creates_batch_files_with_sanitized_refs(tmp_path: Path) -> None:
    report = build_visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack(
        _decision_record(row_count=3),
        _review_report(row_count=3),
        source_decision_record_ref=(
            "eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_human_product_decision_record.v1.json"
        ),
        source_review_report_ref="eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_review.v1.json",
        pack_dir_ref="eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack_001",
        batch_size=2,
        generated_at="2026-05-27T00:00:00Z",
    )
    report_json = tmp_path / "pack.json"
    report_md = tmp_path / "pack.md"
    pack_dir = tmp_path / "pack"

    paths = write_visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack(
        report,
        report_json=report_json,
        report_md=report_md,
        pack_dir=pack_dir,
    )

    assert len(paths["batchFiles"]) == 2
    assert (pack_dir / "batch_01_prompt.md").exists()
    assert (pack_dir / "batch_01_recommendation_template.v1.json").exists()
    combined = (
        report_json.read_text(encoding="utf-8")
        + report_md.read_text(encoding="utf-8")
        + (pack_dir / "batch_01_prompt.md").read_text(encoding="utf-8")
        + (pack_dir / "batch_01_recommendation_template.v1.json").read_text(encoding="utf-8")
    )
    parsed = json.loads(report_json.read_text(encoding="utf-8"))
    assert parsed["sourceDecisionRecord"]["reportRef"].startswith("eval/knowledgeos/reports/")
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "i" + "Cloud" not in combined
