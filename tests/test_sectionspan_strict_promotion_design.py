from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.sectionspan_strict_promotion_design import (
    SECTIONSPAN_STRICT_PROMOTION_DESIGN_SCHEMA_ID,
    build_sectionspan_strict_promotion_design,
    write_sectionspan_strict_promotion_design_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _review_pack(*, unsafe: bool = False) -> dict:
    return {
        "schema": "knowledge-hub.paper.sectionspan-contract-review-pack.v1",
        "status": "review_pack_ready",
        "counts": {
            "reviewCardCount": 2,
            "heldOutCandidates": 1,
            "strictEligibleCards": 1 if unsafe else 0,
            "citationGradeCards": 0,
            "runtimeEvidenceCards": 0,
        },
        "gate": {
            "reviewPackReady": True,
            "candidateFormalizationReady": True,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "allCardsNonStrict": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "reviewCards": [
            _card("sectionspan:paper-1:0001", "paper-1", "1. Introduction", "numbered_section", 1, 42),
            _card("sectionspan:paper-2:0001", "paper-2", "Abstract", "abstract", 1, 10),
        ],
        "heldOut": [
            {
                "sourceCandidateId": "paper-1:title:0001",
                "paperId": "paper-1",
                "candidateText": "Paper Title",
                "reviewClass": "paper_title",
                "reason": "held_out_paper_title",
                "reviewStatus": "held_out",
                "strictEligible": False,
                "citationGrade": False,
                "runtimeEvidence": False,
            }
        ],
    }


def _card(candidate_id: str, paper_id: str, text: str, section_type: str, page: int, chars_start: int) -> dict:
    return {
        "card_id": f"card:{candidate_id}",
        "review_status": "needs_human_review",
        "source_contract_candidate_id": candidate_id.replace("sectionspan:", "sectionspan-contract:"),
        "source_sectionspan_candidate_id": candidate_id,
        "paper_id": paper_id,
        "candidate_text": text,
        "section_label": "1" if section_type == "numbered_section" else "",
        "section_title": text.replace("1. ", ""),
        "section_type": section_type,
        "section_level": 1 if section_type == "numbered_section" else 0,
        "canonical_span": {
            "chars_start": chars_start,
            "chars_end": chars_start + len(text),
            "page": page,
            "sourceContentHash": "hash-source",
            "alignmentMethod": "exact",
            "alignmentStatus": "aligned",
            "locatorKind": "canonical_generated_markdown",
        },
    }


def _source_audit(*, missing_second: bool = False, strict_nonzero: bool = False) -> dict:
    rows = [
        _authority_row("sectionspan:paper-1:0001", "paper-1", "1. Introduction", 1, 42),
    ]
    if not missing_second:
        rows.append(_authority_row("sectionspan:paper-2:0001", "paper-2", "Abstract", 1, 10))
    return {
        "schema": "knowledge-hub.paper.source-span-offset-authority-audit.v1",
        "status": "ok",
        "counts": {
            "strictEligibleCandidates": 1 if strict_nonzero else 0,
            "citationGradeCandidates": 0,
            "runtimeEvidenceCandidates": 0,
            "originalPdfOffsetCandidates": 0,
        },
        "gate": {
            "canonicalParsedTextSpanCandidateLayerReady": True,
            "originalPdfOffsetReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
        },
        "policy": {
            "auditOnly": True,
            "allRowsNonStrict": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "rows": rows,
    }


def _authority_row(candidate_id: str, paper_id: str, text: str, page: int, chars_start: int) -> dict:
    return {
        "audit_id": f"audit:{candidate_id}",
        "candidate_id": candidate_id,
        "candidate_type": "section_span_candidate",
        "layer": "sectionspan",
        "paper_id": paper_id,
        "candidate_text": text,
        "canonical_alignment_status": "aligned",
        "alignment_method": "exact",
        "chars_start": chars_start,
        "chars_end": chars_start + len(text),
        "page": page,
        "sourceContentHash": "hash-source",
        "locatorKind": "canonical_generated_markdown",
        "source_span_authority_status": "canonical_generated_markdown_span_non_strict",
        "canonical_parsed_text_span_available": True,
        "original_pdf_offset_available": False,
        "layout_or_bbox_only": False,
        "markdown_offset_only": False,
        "strict_blockers": ["runtime_promotion_disabled_for_tranche"],
    }


def _reports(root: Path, *, review_pack: dict | None = None, source_audit: dict | None = None) -> dict[str, Path]:
    return {
        "review": _write(root, "sectionspan-contract-review-cards.json", review_pack or _review_pack()),
        "source": _write(root, "source-span-offset-authority-audit.json", source_audit or _source_audit()),
    }


def _build(paths: dict[str, Path]) -> dict:
    return build_sectionspan_strict_promotion_design(
        sectionspan_contract_review_pack_report=paths["review"],
        source_span_offset_authority_audit_report=paths["source"],
    )


def test_strict_promotion_design_records_blocked_runtime_status_and_validates_schema(tmp_path: Path) -> None:
    payload = _build(_reports(tmp_path))

    assert payload["schema"] == SECTIONSPAN_STRICT_PROMOTION_DESIGN_SCHEMA_ID
    assert validate_payload(payload, SECTIONSPAN_STRICT_PROMOTION_DESIGN_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "design_ready"
    assert payload["counts"]["designRowCount"] == 2
    assert payload["counts"]["candidateFormalizationReadyRows"] == 2
    assert payload["counts"]["strictPromotionReadyRows"] == 0
    assert payload["counts"]["runtimePromotionAllowedRows"] == 0
    assert payload["gate"]["strictPromotionDesignReady"] is True
    assert payload["gate"]["strictEvidenceReady"] is False


def test_strict_promotion_design_keeps_source_hash_page_chars_non_runtime(tmp_path: Path) -> None:
    payload = _build(_reports(tmp_path))

    assert payload["policy"]["reportOnly"] is True
    assert payload["policy"]["strictPromotionImplemented"] is False
    assert payload["policy"]["strictEvidenceCreated"] is False
    assert payload["policy"]["runtimePromotionAllowed"] is False
    assert payload["policy"]["parserRoutingChanged"] is False
    for row in payload["designRows"]:
        assert row["canonical_span"]["sourceContentHash"] == "hash-source"
        assert row["canonical_span"]["page"] > 0
        assert row["canonical_span"]["chars_start"] >= 0
        assert row["source_span_authority"]["canonicalParsedTextSpanAvailable"] is True
        assert row["source_span_authority"]["originalPdfOffsetAvailable"] is False
        assert row["promotion_design_status"] == "blocked_original_pdf_offset_or_authority_decision_required"
        assert row["strict_eligible"] is False
        assert row["runtime_evidence"] is False
        assert "runtime_answer_citation" in row["disallowed_actions"]
        assert "original_pdf_offset_not_available" in row["strict_blockers"]


def test_strict_promotion_design_reports_missing_source_authority_rows(tmp_path: Path) -> None:
    payload = _build(_reports(tmp_path, source_audit=_source_audit(missing_second=True)))

    assert payload["status"] == "design_ready"
    by_status = payload["counts"]["byPromotionDesignStatus"]
    assert by_status["blocked_source_authority_row_missing"] == 1
    missing = [row for row in payload["designRows"] if row["promotion_design_status"] == "blocked_source_authority_row_missing"]
    assert missing
    assert "source_span_authority_row_missing" in missing[0]["strict_blockers"]


def test_strict_promotion_design_blocks_unsafe_upstream_inputs(tmp_path: Path) -> None:
    review = _review_pack(unsafe=True)
    review["schema"] = "example.wrong.review"
    payload = _build(_reports(tmp_path, review_pack=review, source_audit=_source_audit(strict_nonzero=True)))

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert "sectionspan_contract_review_pack_schema_mismatch" in payload["gate"]["schemaViolations"]
    assert "reviewPack_strictEligibleCards_nonzero" in payload["gate"]["unsafeUpstreamFlags"]
    assert "sourceAuthority_strictEligibleCandidates_nonzero" in payload["gate"]["unsafeUpstreamFlags"]


def test_strict_promotion_design_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    payload = _build(_reports(tmp_path / "input"))

    paths = write_sectionspan_strict_promotion_design_reports(payload, tmp_path / "reports")

    assert set(paths) == {"design", "summary", "markdown"}
    design = json.loads(Path(paths["design"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(design, SECTIONSPAN_STRICT_PROMOTION_DESIGN_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, SECTIONSPAN_STRICT_PROMOTION_DESIGN_SCHEMA_ID, strict=True).ok
    assert "does not create strict evidence" in markdown
