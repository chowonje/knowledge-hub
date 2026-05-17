from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.sectionspan_candidate_audit import (
    SECTIONSPAN_CANDIDATE_REPORT_SCHEMA_ID,
    build_sectionspan_candidate_report,
    write_sectionspan_candidate_reports,
)


def _write_decision_review_fixture(root: Path, *, cards: list[dict] | None = None) -> Path:
    payload = {
        "summary": {
            "schema": "knowledge-hub.paper.mineru-sectionspan-decision-review.v1.local",
            "status": "review_ready",
            "promoteCandidateCount": 3,
            "holdCount": 2,
            "policy": {
                "strictEvidenceCreated": False,
                "runtimePromotionAllowed": False,
            },
        },
        "cards": cards if cards is not None else _decision_cards(),
    }
    path = root / "sectionspan-decision-review.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _decision_cards() -> list[dict]:
    return [
        {
            "cardId": "mineru-review:0001",
            "candidateId": "paper-1:section:0001",
            "paperId": "paper-1",
            "candidateText": "1. Introduction",
            "page": 1,
            "charsStart": 42,
            "charsEnd": 57,
            "alignmentMethod": "exact",
            "sourceContentHash": "hash-source",
            "reviewClass": "numbered_section",
            "recommendedDecision": "promote_as_sectionspan_candidate",
            "strictEligibleNow": False,
            "citationGradeNow": False,
            "strictBlockers": [
                "runtime_promotion_disabled_for_tranche",
                "markdown_offsets_are_generated_not_original_pdf_offsets",
            ],
            "humanReviewQuestions": ["Is this a real section heading?"],
        },
        {
            "cardId": "mineru-review:0002",
            "candidateId": "paper-1:section:0002",
            "paperId": "paper-1",
            "candidateText": "Abstract",
            "page": 1,
            "charsStart": 10,
            "charsEnd": 18,
            "alignmentMethod": "exact",
            "sourceContentHash": "hash-source",
            "reviewClass": "abstract",
            "recommendedDecision": "promote_as_frontmatter_section_candidate",
            "strictEligibleNow": False,
            "citationGradeNow": False,
            "strictBlockers": ["runtime_promotion_disabled_for_tranche"],
            "humanReviewQuestions": [],
        },
        {
            "cardId": "mineru-review:0003",
            "candidateId": "paper-1:section:0003",
            "paperId": "paper-1",
            "candidateText": "References",
            "page": 9,
            "charsStart": 900,
            "charsEnd": 910,
            "alignmentMethod": "exact",
            "sourceContentHash": "hash-source",
            "reviewClass": "backmatter",
            "recommendedDecision": "promote_as_backmatter_section_candidate_with_lower_priority",
            "strictEligibleNow": False,
            "citationGradeNow": False,
            "strictBlockers": ["runtime_promotion_disabled_for_tranche"],
            "humanReviewQuestions": [],
        },
        {
            "cardId": "mineru-review:0004",
            "candidateId": "paper-1:section:0004",
            "paperId": "paper-1",
            "candidateText": "Paper Title",
            "page": 1,
            "charsStart": 0,
            "charsEnd": 10,
            "alignmentMethod": "exact",
            "sourceContentHash": "hash-source",
            "reviewClass": "paper_title",
            "recommendedDecision": "hold_as_document_title_not_section",
            "strictEligibleNow": False,
            "citationGradeNow": False,
            "strictBlockers": ["runtime_promotion_disabled_for_tranche"],
            "humanReviewQuestions": [],
        },
        {
            "cardId": "mineru-review:0005",
            "candidateId": "paper-1:section:0005",
            "paperId": "paper-1",
            "candidateText": "Contents",
            "page": 2,
            "charsStart": 100,
            "charsEnd": 108,
            "alignmentMethod": "exact",
            "sourceContentHash": "hash-source",
            "reviewClass": "toc",
            "recommendedDecision": "hold_toc_navigation_not_content_section",
            "strictEligibleNow": False,
            "citationGradeNow": False,
            "strictBlockers": ["runtime_promotion_disabled_for_tranche"],
            "humanReviewQuestions": [],
        },
    ]


def test_sectionspan_candidate_report_emits_approved_types_and_holds_title_toc(tmp_path: Path) -> None:
    decision_path = _write_decision_review_fixture(tmp_path)

    payload = build_sectionspan_candidate_report(decision_path)

    assert payload["schema"] == SECTIONSPAN_CANDIDATE_REPORT_SCHEMA_ID
    assert validate_payload(payload, SECTIONSPAN_CANDIDATE_REPORT_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["sectionSpanCandidates"] == 3
    assert payload["counts"]["heldOutCandidates"] == 2
    assert payload["counts"]["bySectionType"] == {
        "numbered_section": 1,
        "abstract": 1,
        "backmatter": 1,
    }
    assert payload["counts"]["heldOutByReason"] == {
        "held_out_paper_title": 1,
        "held_out_toc": 1,
    }


def test_sectionspan_candidates_remain_non_strict_despite_hash_page_and_chars(tmp_path: Path) -> None:
    decision_path = _write_decision_review_fixture(tmp_path)

    payload = build_sectionspan_candidate_report(decision_path)

    for item in payload["candidates"]:
        assert item["sourceContentHash"] == "hash-source"
        assert item["chars_start"] is not None
        assert item["chars_end"] is not None
        assert item["page"] is not None
        assert item["evidence_tier"] == "sectionspan_candidate_only"
        assert item["strict_eligible"] is False
        assert item["citation_grade"] is False
        assert "sectionspan_candidate_layer_not_runtime_evidence" in item["strict_blockers"]
    assert payload["policy"]["strictEvidenceCreated"] is False
    assert payload["policy"]["runtimePromotionAllowed"] is False
    assert payload["policy"]["parserRoutingChanged"] is False


def test_missing_page_hash_or_span_is_held_out(tmp_path: Path) -> None:
    broken = [
        {
            **_decision_cards()[0],
            "candidateId": "paper-1:section:missing-page",
            "page": None,
        },
        {
            **_decision_cards()[0],
            "candidateId": "paper-1:section:missing-hash",
            "sourceContentHash": "",
        },
        {
            **_decision_cards()[0],
            "candidateId": "paper-1:section:missing-span",
            "charsStart": None,
        },
    ]
    decision_path = _write_decision_review_fixture(tmp_path, cards=broken)

    payload = build_sectionspan_candidate_report(decision_path)

    assert payload["status"] == "empty"
    assert payload["candidates"] == []
    assert payload["counts"]["heldOutCandidates"] == 3
    assert payload["counts"]["heldOutByReason"] == {
        "missing_page": 1,
        "missing_source_content_hash": 1,
        "missing_chars_start_end": 1,
    }
    assert validate_payload(payload, SECTIONSPAN_CANDIDATE_REPORT_SCHEMA_ID, strict=True).ok


def test_section_number_and_level_are_derived_for_numbered_sections(tmp_path: Path) -> None:
    cards = [
        {
            **_decision_cards()[0],
            "candidateText": "3.2.1 Scaled Dot-Product Attention",
            "candidateId": "paper-1:section:attention",
        }
    ]
    decision_path = _write_decision_review_fixture(tmp_path, cards=cards)

    payload = build_sectionspan_candidate_report(decision_path)

    [candidate] = payload["candidates"]
    assert candidate["section_label"] == "3.2.1"
    assert candidate["section_title"] == "Scaled Dot-Product Attention"
    assert candidate["section_level"] == 3


def test_sectionspan_report_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    decision_path = _write_decision_review_fixture(tmp_path / "input")
    payload = build_sectionspan_candidate_report(decision_path)

    paths = write_sectionspan_candidate_reports(payload, tmp_path / "reports")

    assert set(paths) == {"candidates", "summary", "markdown"}
    candidates = json.loads(Path(paths["candidates"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(candidates, SECTIONSPAN_CANDIDATE_REPORT_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, SECTIONSPAN_CANDIDATE_REPORT_SCHEMA_ID, strict=True).ok
    assert "They are not strict evidence" in markdown
