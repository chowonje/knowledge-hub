from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.sectionspan_contract_review_pack import (
    SECTIONSPAN_CONTRACT_REVIEW_PACK_SCHEMA_ID,
    build_sectionspan_contract_review_pack,
    write_sectionspan_contract_review_pack_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _contract_review(*, status: str = "contract_review_ready", unsafe: bool = False) -> dict:
    return {
        "schema": "knowledge-hub.paper.sectionspan-contract-review.v1",
        "status": status,
        "counts": {
            "inputSectionSpanCandidates": 3,
            "contractCandidateCount": 3,
            "contractReadyCandidates": 3,
            "blockedContractCandidates": 0,
            "heldOutCandidates": 2,
            "strictEligibleCandidates": 1 if unsafe else 0,
            "citationGradeCandidates": 0,
            "runtimeEvidenceCandidates": 0,
        },
        "gate": {
            "sectionspanContractReviewReady": status == "contract_review_ready",
            "candidateFormalizationReady": status == "contract_review_ready",
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "allCandidatesNonStrict": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "contractCandidates": [
            _candidate("sectionspan-contract:paper-1:0001", "paper-1", "1. Introduction", "numbered_section", 1, 42),
            _candidate("sectionspan-contract:paper-1:0002", "paper-1", "Abstract", "abstract", 1, 10),
            _candidate("sectionspan-contract:paper-2:0001", "paper-2", "References", "backmatter", 9, 900),
        ],
        "heldOut": [
            {
                "sourceCandidateId": "paper-1:title:0001",
                "paperId": "paper-1",
                "candidateText": "Paper Title",
                "reviewClass": "paper_title",
                "reason": "held_out_paper_title",
                "contractStatus": "held_out",
                "strictEligible": False,
                "citationGrade": False,
                "runtimeEvidence": False,
            },
            {
                "sourceCandidateId": "paper-2:toc:0001",
                "paperId": "paper-2",
                "candidateText": "Contents",
                "reviewClass": "toc",
                "reason": "held_out_toc",
                "contractStatus": "held_out",
                "strictEligible": False,
                "citationGrade": False,
                "runtimeEvidence": False,
            },
        ],
    }


def _candidate(
    candidate_id: str,
    paper_id: str,
    text: str,
    section_type: str,
    page: int,
    chars_start: int,
) -> dict:
    chars_end = chars_start + len(text)
    return {
        "contract_candidate_id": candidate_id,
        "source_candidate_id": candidate_id.replace("sectionspan-contract", "sectionspan"),
        "candidate_type": "sectionspan_contract_candidate",
        "paper_id": paper_id,
        "source_parser": "mineru+pymupdf_alignment",
        "candidate_text": text,
        "section_label": "1" if section_type == "numbered_section" else "",
        "section_title": text.replace("1. ", ""),
        "section_type": section_type,
        "section_level": 1 if section_type == "numbered_section" else 0,
        "canonical_alignment_status": "aligned",
        "alignment_method": "exact",
        "chars_start": chars_start,
        "chars_end": chars_end,
        "page": page,
        "sourceContentHash": "hash-source",
        "confidence": 0.99,
        "contract_status": "contract_ready_non_strict",
        "contract_ready": True,
        "contract_blockers": [],
        "contract_requirements": ["exact_alignment_to_canonical_generated_text"],
        "allowed_next_actions": ["human_operator_review"],
        "disallowed_actions": ["strict_evidence_promotion"],
        "evidence_tier": "sectionspan_contract_candidate_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": [
            "runtime_promotion_disabled_for_tranche",
            "sectionspan_contract_review_only",
        ],
        "non_strict_reason": ["contract_review_is_report_only"],
    }


def _report_path(root: Path, payload: dict | None = None) -> Path:
    return _write(root, "sectionspan-contract-review.json", payload or _contract_review())


def test_review_pack_emits_human_review_cards_and_validates_schema(tmp_path: Path) -> None:
    payload = build_sectionspan_contract_review_pack(
        sectionspan_contract_review_report=_report_path(tmp_path)
    )

    assert payload["schema"] == SECTIONSPAN_CONTRACT_REVIEW_PACK_SCHEMA_ID
    assert validate_payload(payload, SECTIONSPAN_CONTRACT_REVIEW_PACK_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "review_pack_ready"
    assert payload["counts"]["reviewCardCount"] == 3
    assert payload["counts"]["heldOutCandidates"] == 2
    assert payload["counts"]["bySectionType"] == {
        "numbered_section": 1,
        "abstract": 1,
        "backmatter": 1,
    }
    assert payload["counts"]["byRecommendedAction"] == {
        "approve_candidate_contract": 1,
        "approve_frontmatter_candidate_contract": 1,
        "approve_low_priority_backmatter_candidate_contract": 1,
    }


def test_review_pack_remains_non_strict_despite_hash_page_and_chars(tmp_path: Path) -> None:
    payload = build_sectionspan_contract_review_pack(
        sectionspan_contract_review_report=_report_path(tmp_path)
    )

    assert payload["policy"]["strictEvidenceCreated"] is False
    assert payload["policy"]["runtimePromotionAllowed"] is False
    assert payload["policy"]["parserRoutingChanged"] is False
    assert payload["policy"]["databaseMutation"] is False
    for card in payload["reviewCards"]:
        assert card["canonical_span"]["sourceContentHash"] == "hash-source"
        assert card["canonical_span"]["page"] > 0
        assert card["canonical_span"]["chars_start"] >= 0
        assert card["evidence_tier"] == "sectionspan_contract_review_card_only"
        assert card["strict_eligible"] is False
        assert card["citation_grade"] is False
        assert card["runtime_evidence"] is False
        assert "strict_promotion_requires_explicit_later_tranche" in card["strict_blockers"]


def test_review_pack_blocks_wrong_schema_or_unsafe_upstream_counts(tmp_path: Path) -> None:
    payload_data = _contract_review(unsafe=True)
    payload_data["schema"] = "example.wrong.schema"
    payload = build_sectionspan_contract_review_pack(
        sectionspan_contract_review_report=_report_path(tmp_path, payload_data)
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert payload["gate"]["schemaViolations"] == ["sectionspan_contract_review_schema_mismatch"]
    assert "strictEligibleCandidates_nonzero" in payload["gate"]["unsafeUpstreamFlags"]
    assert validate_payload(payload, SECTIONSPAN_CONTRACT_REVIEW_PACK_SCHEMA_ID, strict=True).ok


def test_review_pack_blocks_when_contract_review_is_not_ready(tmp_path: Path) -> None:
    payload = build_sectionspan_contract_review_pack(
        sectionspan_contract_review_report=_report_path(tmp_path, _contract_review(status="blocked"))
    )

    assert payload["status"] == "blocked"
    assert "sectionspan_contract_review_not_ready" in payload["gate"]["unsafeUpstreamFlags"]
    assert payload["counts"]["reviewCardCount"] == 3


def test_review_pack_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    payload = build_sectionspan_contract_review_pack(
        sectionspan_contract_review_report=_report_path(tmp_path / "input")
    )

    paths = write_sectionspan_contract_review_pack_reports(payload, tmp_path / "reports")

    assert set(paths) == {"cards", "summary", "markdown"}
    cards = json.loads(Path(paths["cards"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(cards, SECTIONSPAN_CONTRACT_REVIEW_PACK_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, SECTIONSPAN_CONTRACT_REVIEW_PACK_SCHEMA_ID, strict=True).ok
    assert "does not create strict evidence" in markdown
