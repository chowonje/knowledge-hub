from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.mineru_potential_review_pack import (
    MINERU_POTENTIAL_REVIEW_PACK_SCHEMA_ID,
    build_mineru_potential_review_pack,
    write_mineru_potential_review_pack_reports,
)


def _write_source_alignment_fixture(root: Path, *, candidates: list[dict] | None = None) -> Path:
    parsed = root / "parsed" / "paper-1"
    parsed.mkdir(parents=True)
    markdown = """# paper-1

## Page 1

Intro Title

Repeated caption.

## Page 2

Another Section
"""
    (parsed / "document.md").write_text(markdown, encoding="utf-8")
    payload = {
        "schema": "knowledge-hub.paper.mineru-source-alignment-audit.v1",
        "status": "ok",
        "policy": {
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
        },
        "counts": {
            "totalCandidates": 4,
            "strictRequirementCompleteCandidates": 2,
            "strictEligibleCandidates": 0,
        },
        "papers": [
            {
                "paperId": "paper-1",
                "input": {"pymupdfDocumentMarkdownPath": str(parsed / "document.md")},
                "source": {"sourceContentHashAvailable": True},
                "counts": {},
                "strictBlockerSummary": {},
                "status": "ok",
            }
        ],
        "candidates": candidates if candidates is not None else _source_alignment_candidates(),
    }
    report_path = root / "mineru-source-alignment-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    return report_path


def _source_alignment_candidates() -> list[dict]:
    return [
        {
            "candidate_id": "paper-1:section:0001",
            "candidate_type": "section_candidate",
            "paper_id": "paper-1",
            "candidate_text": "Intro Title",
            "alignment_status": "aligned",
            "alignment_method": "exact",
            "alignment_reason": "single_exact_text_match",
            "chars_start": 22,
            "chars_end": 33,
            "page": 1,
            "sourceContentHash": "hash-source",
            "sourceContentHashSource": "sourceContentHash",
            "confidence": 0.99,
            "classification": "potential_strict_candidate",
            "strict_requirements_met": True,
            "strict_eligible": False,
            "citation_grade": False,
            "evidence_tier": "source_alignment_candidate_only",
            "strict_blockers": [
                "runtime_promotion_disabled_for_tranche",
                "markdown_offsets_are_generated_not_original_pdf_offsets",
            ],
            "source_span_locator": {
                "path": "document.md",
                "locatorKind": "canonical_generated_markdown",
                "chars": {"start": 22, "end": 33},
            },
            "mineruCandidate": {
                "layout_element_ids": ["mineru:1"],
                "bbox": [1, 2, 3, 4],
            },
        },
        {
            "candidate_id": "paper-1:table:0001",
            "candidate_type": "table_candidate",
            "paper_id": "paper-1",
            "candidate_text": "Table 1: Results",
            "classification": "page_recovered_non_strict",
            "alignment_status": "aligned",
            "alignment_method": "exact",
            "strict_requirements_met": False,
            "strict_eligible": False,
            "citation_grade": False,
            "evidence_tier": "source_alignment_candidate_only",
            "strict_blockers": ["table_cell_provenance_missing"],
        },
        {
            "candidate_id": "paper-1:section:0002",
            "candidate_type": "section_candidate",
            "paper_id": "paper-1",
            "candidate_text": "Repeated caption.",
            "classification": "blocked",
            "alignment_status": "ambiguous",
            "alignment_method": "exact",
            "strict_requirements_met": False,
            "strict_eligible": False,
            "citation_grade": False,
            "evidence_tier": "source_alignment_candidate_only",
            "strict_blockers": ["ambiguous_match"],
        },
        {
            "candidate_id": "paper-1:section:9999",
            "candidate_type": "section_candidate",
            "paper_id": "paper-1",
            "candidate_text": "Another Section",
            "classification": "potential_strict_candidate",
            "alignment_status": "aligned",
            "alignment_method": "normalized",
            "chars_start": 65,
            "chars_end": 80,
            "page": 2,
            "sourceContentHash": "hash-source",
            "strict_requirements_met": True,
            "strict_eligible": False,
            "citation_grade": False,
            "evidence_tier": "source_alignment_candidate_only",
            "strict_blockers": ["runtime_promotion_disabled_for_tranche"],
        },
    ]


def test_review_pack_emits_only_exact_potential_candidates_and_keeps_non_strict(tmp_path: Path) -> None:
    report_path = _write_source_alignment_fixture(tmp_path)

    payload = build_mineru_potential_review_pack(report_path)

    assert payload["schema"] == MINERU_POTENTIAL_REVIEW_PACK_SCHEMA_ID
    assert validate_payload(payload, MINERU_POTENTIAL_REVIEW_PACK_SCHEMA_ID, strict=True).ok
    assert payload["policy"]["strictEvidenceCreated"] is False
    assert payload["policy"]["runtimePromotionAllowed"] is False
    assert payload["counts"]["inputCandidateCount"] == 4
    assert payload["counts"]["emittedCardCount"] == 1
    assert payload["counts"]["skippedCount"] == 3
    assert payload["counts"]["strictEligibleCards"] == 0
    assert payload["counts"]["citationGradeCards"] == 0

    [card] = payload["cards"]
    assert card["candidateId"] == "paper-1:section:0001"
    assert card["candidateType"] == "section_candidate"
    assert card["recommendedAction"] == "review_for_section_span_schema"
    assert card["formalArtifactCandidate"] == "SectionSpan"
    assert card["evidenceTier"] == "human_review_candidate_only"
    assert card["strict"] is False
    assert card["citationGrade"] is False
    assert card["canonicalSpan"]["matchedText"] == "Intro Title"
    assert "runtime_promotion_disabled_for_tranche" in card["promotionAssessment"]["strictBlockers"]


def test_review_pack_returns_empty_pack_when_no_candidate_is_reviewable(tmp_path: Path) -> None:
    report_path = _write_source_alignment_fixture(
        tmp_path,
        candidates=[
            {
                "candidate_id": "paper-1:section:0001",
                "candidate_type": "section_candidate",
                "paper_id": "paper-1",
                "candidate_text": "Intro Title",
                "classification": "blocked",
                "alignment_status": "ambiguous",
                "strict_eligible": False,
                "citation_grade": False,
                "evidence_tier": "source_alignment_candidate_only",
                "strict_blockers": ["ambiguous_match"],
            }
        ],
    )

    payload = build_mineru_potential_review_pack(report_path)

    assert payload["status"] == "empty"
    assert payload["counts"]["inputCandidateCount"] == 1
    assert payload["counts"]["emittedCardCount"] == 0
    assert payload["counts"]["skippedCount"] == 1
    assert payload["cards"] == []
    assert validate_payload(payload, MINERU_POTENTIAL_REVIEW_PACK_SCHEMA_ID, strict=True).ok


def test_review_pack_output_order_is_deterministic(tmp_path: Path) -> None:
    candidates = _source_alignment_candidates()
    candidates.append(
        {
            **candidates[0],
            "candidate_id": "paper-1:section:0000",
            "candidate_text": "Another Section",
            "chars_start": 65,
            "chars_end": 80,
            "page": 2,
        }
    )
    report_path = _write_source_alignment_fixture(tmp_path, candidates=list(reversed(candidates)))

    payload = build_mineru_potential_review_pack(report_path)

    assert [card["candidateId"] for card in payload["cards"]] == [
        "paper-1:section:0001",
        "paper-1:section:0000",
    ]


def test_review_pack_writes_schema_valid_reports(tmp_path: Path) -> None:
    report_path = _write_source_alignment_fixture(tmp_path / "input")
    payload = build_mineru_potential_review_pack(report_path)

    paths = write_mineru_potential_review_pack_reports(payload, tmp_path / "reports")

    assert set(paths) == {"cards", "summary", "markdown"}
    cards = json.loads(Path(paths["cards"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(cards, MINERU_POTENTIAL_REVIEW_PACK_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, MINERU_POTENTIAL_REVIEW_PACK_SCHEMA_ID, strict=True).ok
    assert "does not create strict evidence" in markdown
