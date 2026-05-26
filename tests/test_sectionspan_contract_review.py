from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.sectionspan_contract_review import (
    SECTIONSPAN_CONTRACT_REVIEW_SCHEMA_ID,
    build_sectionspan_contract_review,
    write_sectionspan_contract_review_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _candidate_report(*, candidate: dict | None = None, strict_nonzero: bool = False) -> dict:
    strict_count = 1 if strict_nonzero else 0
    candidate = candidate or _section_candidate()
    return {
        "schema": "knowledge-hub.paper.sectionspan-candidate-report.v1",
        "status": "ok",
        "counts": {
            "inputReviewCards": 3,
            "sectionSpanCandidates": 1,
            "heldOutCandidates": 2,
            "strictEligibleCandidates": strict_count,
            "citationGradeCandidates": 0,
            "bySectionType": {"numbered_section": 1},
            "byPaper": {"paper-1": 1},
            "heldOutByReason": {"held_out_paper_title": 1, "held_out_toc": 1},
        },
        "policy": {
            "allCandidatesNonStrict": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
        },
        "candidates": [candidate],
        "heldOut": [
            {
                "sourceCandidateId": "paper-1:title:0001",
                "paperId": "paper-1",
                "candidateText": "Paper Title",
                "reviewClass": "paper_title",
                "reason": "held_out_paper_title",
                "strictEligible": False,
                "citationGrade": False,
            },
            {
                "sourceCandidateId": "paper-1:toc:0001",
                "paperId": "paper-1",
                "candidateText": "Contents",
                "reviewClass": "toc",
                "reason": "held_out_toc",
                "strictEligible": False,
                "citationGrade": False,
            },
        ],
    }


def _section_candidate(**overrides: object) -> dict:
    payload = {
        "candidate_id": "sectionspan:paper-1:0001",
        "candidate_type": "section_span_candidate",
        "source_candidate_id": "paper-1:section:0001",
        "paper_id": "paper-1",
        "source_parser": "mineru+pymupdf_alignment",
        "candidate_text": "1. Introduction",
        "section_label": "1",
        "section_title": "Introduction",
        "section_type": "numbered_section",
        "section_level": 1,
        "canonical_alignment_status": "aligned",
        "alignment_method": "exact",
        "chars_start": 42,
        "chars_end": 57,
        "page": 1,
        "sourceContentHash": "hash-source",
        "confidence": 0.99,
        "source_span_locator": {
            "path": "document.md",
            "locatorKind": "canonical_generated_markdown",
            "chars": {"start": 42, "end": 57},
        },
        "evidence_tier": "sectionspan_candidate_only",
        "strict_eligible": False,
        "citation_grade": False,
        "strict_blockers": [
            "runtime_promotion_disabled_for_tranche",
            "sectionspan_candidate_layer_not_runtime_evidence",
            "markdown_offsets_are_generated_not_original_pdf_offsets",
        ],
        "non_strict_reason": [
            "runtime_promotion_disabled_for_tranche",
            "sectionspan_candidate_layer_not_runtime_evidence",
        ],
    }
    payload.update(overrides)
    return payload


def _promotion_report(
    *,
    status: str = "draft_ready",
    strict_ready: bool = False,
    wrong_track: bool = False,
) -> dict:
    return {
        "schema": "knowledge-hub.paper.candidate-layer-promotion-policy-draft.v1",
        "status": status,
        "counts": {
            "totalCandidates": 86,
            "promotionTrackCount": 4,
            "candidateOnlyPromotionTracks": 4,
            "candidateFormalizationReadyTracks": 1,
            "strictPromotionReadyTracks": 0,
            "parserRoutingReadyTracks": 0,
            "answerIntegrationReadyTracks": 0,
            "runtimePromotionAllowedTracks": 0,
            "strictEligibleCandidates": 0,
            "citationGradeCandidates": 0,
            "runtimeEvidenceCandidates": 0,
            "currentRuntimeAnswerableQuestions": 0,
        },
        "gate": {
            "promotionPolicyDraftReady": status == "draft_ready",
            "candidateLayerReviewReady": True,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
        },
        "policy": {
            "draftOnly": True,
            "runtimePolicyChanged": False,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "promotionTracks": [
            {
                "track_id": "candidate-layer-promotion:sectionspan",
                "layer": "sectionspan",
                "candidate_count": 1,
                "canonical_source_span_candidate_count": 1,
                "promotion_readiness": (
                    "blocked_for_test" if wrong_track else "candidate_formalization_ready_non_strict"
                ),
                "candidate_layer_formalization_ready": not wrong_track,
                "strict_promotion_ready": strict_ready,
                "parser_routing_ready": False,
                "answer_integration_ready": False,
                "runtime_evidence": False,
            }
        ],
    }


def _reports(root: Path, *, candidate_payload: dict | None = None, promotion_payload: dict | None = None) -> dict[str, Path]:
    return {
        "sectionspan": _write(root, "sectionspan-candidates.json", candidate_payload or _candidate_report()),
        "promotion": _write(root, "candidate-layer-promotion-policy-draft.json", promotion_payload or _promotion_report()),
    }


def _build(paths: dict[str, Path]) -> dict:
    return build_sectionspan_contract_review(
        sectionspan_candidate_report=paths["sectionspan"],
        candidate_layer_promotion_policy_draft_report=paths["promotion"],
    )


def test_sectionspan_contract_review_accepts_candidate_layer_as_non_strict_contract(tmp_path: Path) -> None:
    payload = _build(_reports(tmp_path))

    assert payload["schema"] == SECTIONSPAN_CONTRACT_REVIEW_SCHEMA_ID
    assert validate_payload(payload, SECTIONSPAN_CONTRACT_REVIEW_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "contract_review_ready"
    assert payload["counts"]["contractCandidateCount"] == 1
    assert payload["counts"]["contractReadyCandidates"] == 1
    assert payload["counts"]["heldOutCandidates"] == 2
    assert payload["gate"]["sectionspanContractReviewReady"] is True
    [row] = payload["contractCandidates"]
    assert row["contract_status"] == "contract_ready_non_strict"
    assert row["evidence_tier"] == "sectionspan_contract_candidate_only"


def test_sectionspan_contract_review_never_promotes_runtime_or_strict_evidence(tmp_path: Path) -> None:
    payload = _build(_reports(tmp_path))

    assert payload["policy"]["reportOnly"] is True
    assert payload["policy"]["strictEvidenceCreated"] is False
    assert payload["policy"]["runtimePromotionAllowed"] is False
    assert payload["policy"]["parserRoutingChanged"] is False
    assert payload["policy"]["databaseMutation"] is False
    assert payload["policy"]["reindexOrReembed"] is False
    [row] = payload["contractCandidates"]
    assert row["sourceContentHash"] == "hash-source"
    assert row["chars_start"] == 42
    assert row["chars_end"] == 57
    assert row["page"] == 1
    assert row["strict_eligible"] is False
    assert row["citation_grade"] is False
    assert row["runtime_evidence"] is False
    assert "runtime_answer_citation" in row["disallowed_actions"]
    assert "source_hash_page_and_chars_do_not_create_strict_evidence" in payload["contractPrinciples"]


def test_sectionspan_contract_review_blocks_missing_required_candidate_fields(tmp_path: Path) -> None:
    broken = _section_candidate(page=0, sourceContentHash="", alignment_method="normalized")
    candidate_payload = _candidate_report(candidate=broken)
    payload = _build(_reports(tmp_path, candidate_payload=candidate_payload))

    assert payload["status"] == "blocked"
    assert payload["counts"]["blockedContractCandidates"] == 1
    [row] = payload["contractCandidates"]
    assert row["contract_status"] == "blocked"
    assert set(row["contract_blockers"]) >= {
        "alignment_method_not_exact",
        "invalid_or_missing_page",
        "missing_source_content_hash",
    }
    assert validate_payload(payload, SECTIONSPAN_CONTRACT_REVIEW_SCHEMA_ID, strict=True).ok


def test_sectionspan_contract_review_blocks_unsafe_upstream_status(tmp_path: Path) -> None:
    candidate_payload = _candidate_report(strict_nonzero=True)
    promotion_payload = _promotion_report(status="blocked", strict_ready=True)
    payload = _build(
        _reports(tmp_path, candidate_payload=candidate_payload, promotion_payload=promotion_payload)
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert "sectionspan_strictEligibleCandidates_nonzero" in payload["gate"]["unsafeUpstreamFlags"]
    assert "promotion_policy_draft_not_ready" in payload["gate"]["unsafeUpstreamFlags"]
    assert "sectionspan_track_strict_promotion_ready_true" in payload["gate"]["unsafeUpstreamFlags"]


def test_sectionspan_contract_review_blocks_schema_or_track_mismatch(tmp_path: Path) -> None:
    candidate_payload = _candidate_report()
    candidate_payload["schema"] = "example.wrong.sectionspan"
    promotion_payload = _promotion_report(wrong_track=True)
    payload = _build(
        _reports(tmp_path, candidate_payload=candidate_payload, promotion_payload=promotion_payload)
    )

    assert payload["status"] == "blocked"
    assert "sectionspan_candidate_schema_mismatch" in payload["gate"]["schemaViolations"]
    assert "sectionspan_track_not_formalization_ready" in payload["gate"]["unsafeUpstreamFlags"]


def test_sectionspan_contract_review_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    payload = _build(_reports(tmp_path / "input"))

    paths = write_sectionspan_contract_review_reports(payload, tmp_path / "reports")

    assert set(paths) == {"review", "summary", "markdown"}
    review = json.loads(Path(paths["review"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(review, SECTIONSPAN_CONTRACT_REVIEW_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, SECTIONSPAN_CONTRACT_REVIEW_SCHEMA_ID, strict=True).ok
    assert "does not create strict evidence" in markdown
