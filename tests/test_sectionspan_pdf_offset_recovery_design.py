from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.sectionspan_pdf_offset_recovery_design import (
    SECTIONSPAN_PDF_OFFSET_RECOVERY_DESIGN_SCHEMA_ID,
    build_sectionspan_pdf_offset_recovery_design,
    write_sectionspan_pdf_offset_recovery_design_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _options(*, missing_recovery_option: bool = False, unsafe: bool = False) -> dict:
    options = [
        {"option_id": "keep_candidate_layer_only"},
        {"option_id": "explicitly_authorize_canonical_generated_markdown_offsets"},
    ]
    if not missing_recovery_option:
        options.insert(1, {"option_id": "recover_original_pdf_offsets_first"})
    return {
        "schema": "knowledge-hub.paper.sectionspan-source-authority-options.v1",
        "status": "options_ready",
        "counts": {
            "authorityDecisionMadeOptions": 1 if unsafe else 0,
            "strictPromotionReadyOptions": 0,
            "runtimePromotionAllowedOptions": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "authorityOptionsReady": True,
            "authorityDecisionMade": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "authorityDecisionMade": False,
            "strictPromotionImplemented": False,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "options": options,
    }


def _design(*, unsafe: bool = False, status: str = "design_ready") -> dict:
    return {
        "schema": "knowledge-hub.paper.sectionspan-strict-promotion-design.v1",
        "status": status,
        "counts": {
            "strictPromotionReadyRows": 0,
            "runtimePromotionAllowedRows": 0,
            "strictEligibleRows": 1 if unsafe else 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "strictPromotionDesignReady": status == "design_ready",
            "candidateFormalizationReady": status == "design_ready",
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "strictPromotionImplemented": False,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "designRows": [
            _design_row("sectionspan:paper-1:0001", "paper-1", "1. Introduction", "numbered_section", 1, 42),
            _design_row("sectionspan:paper-2:0001", "paper-2", "Abstract", "abstract", 1, 10),
        ],
    }


def _design_row(candidate_id: str, paper_id: str, text: str, section_type: str, page: int, chars_start: int) -> dict:
    return {
        "design_id": f"design:{candidate_id}",
        "source_review_card_id": f"card:{candidate_id}",
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
        "source_span_authority": {
            "authorityStatus": "canonical_generated_markdown_span_non_strict",
            "locatorKind": "canonical_generated_markdown",
            "canonicalParsedTextSpanAvailable": True,
            "originalPdfOffsetAvailable": False,
        },
    }


def _reports(root: Path, *, options: dict | None = None, design: dict | None = None) -> dict[str, Path]:
    return {
        "options": _write(root, "sectionspan-source-authority-options.json", options or _options()),
        "design": _write(root, "sectionspan-strict-promotion-design.json", design or _design()),
    }


def _build(paths: dict[str, Path]) -> dict:
    return build_sectionspan_pdf_offset_recovery_design(
        sectionspan_source_authority_options_report=paths["options"],
        sectionspan_strict_promotion_design_report=paths["design"],
    )


def test_pdf_offset_recovery_design_plans_rows_without_executing_recovery(tmp_path: Path) -> None:
    payload = _build(_reports(tmp_path))

    assert payload["schema"] == SECTIONSPAN_PDF_OFFSET_RECOVERY_DESIGN_SCHEMA_ID
    assert validate_payload(payload, SECTIONSPAN_PDF_OFFSET_RECOVERY_DESIGN_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "design_ready"
    assert payload["counts"]["recoveryPlanRows"] == 2
    assert payload["counts"]["plannedRows"] == 2
    assert payload["counts"]["executedRows"] == 0
    assert payload["counts"]["originalPdfOffsetRecoveredRows"] == 0
    assert payload["gate"]["pdfOffsetRecoveryImplemented"] is False
    assert payload["gate"]["strictEvidenceReady"] is False


def test_pdf_offset_recovery_design_keeps_every_row_non_strict(tmp_path: Path) -> None:
    payload = _build(_reports(tmp_path))

    assert payload["policy"]["reportOnly"] is True
    assert payload["policy"]["authorityDecisionMade"] is False
    assert payload["policy"]["pdfOffsetRecoveryImplemented"] is False
    assert payload["policy"]["originalPdfOffsetRecovered"] is False
    assert payload["policy"]["strictEvidenceCreated"] is False
    for row in payload["recoveryPlanRows"]:
        assert row["recovery_status"] == "planned_not_executed"
        assert row["original_pdf_offset_recovered"] is False
        assert row["strict_promotion_ready"] is False
        assert row["runtime_promotion_allowed"] is False
        assert row["strict_eligible"] is False
        assert row["planned_output_contract"]["originalPdfCharsStart"] is None
        assert "match_ambiguous" in row["stop_conditions"]
        assert "summary_or_paraphrase_matches_must_be_rejected" in row["matching_requirements"]


def test_pdf_offset_recovery_design_blocks_when_recommended_option_missing(tmp_path: Path) -> None:
    payload = _build(_reports(tmp_path, options=_options(missing_recovery_option=True)))

    assert payload["status"] == "blocked"
    assert payload["counts"]["recommendedRecoveryOptionPresent"] == 0
    assert "recommended_recovery_option_missing" in payload["gate"]["unsafeUpstreamFlags"]


def test_pdf_offset_recovery_design_blocks_unsafe_or_wrong_schema_inputs(tmp_path: Path) -> None:
    options = _options(unsafe=True)
    options["schema"] = "example.wrong.options"
    payload = _build(_reports(tmp_path, options=options, design=_design(unsafe=True)))

    assert payload["status"] == "blocked"
    assert "sectionspan_source_authority_options_schema_mismatch" in payload["gate"]["schemaViolations"]
    assert "sourceAuthorityOptions_authorityDecisionMadeOptions_nonzero" in payload["gate"]["unsafeUpstreamFlags"]
    assert "strictPromotionDesign_strictEligibleRows_nonzero" in payload["gate"]["unsafeUpstreamFlags"]
    assert validate_payload(payload, SECTIONSPAN_PDF_OFFSET_RECOVERY_DESIGN_SCHEMA_ID, strict=True).ok


def test_pdf_offset_recovery_design_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    payload = _build(_reports(tmp_path / "input"))

    paths = write_sectionspan_pdf_offset_recovery_design_reports(payload, tmp_path / "reports")

    assert set(paths) == {"design", "summary", "markdown"}
    design = json.loads(Path(paths["design"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(design, SECTIONSPAN_PDF_OFFSET_RECOVERY_DESIGN_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, SECTIONSPAN_PDF_OFFSET_RECOVERY_DESIGN_SCHEMA_ID, strict=True).ok
    assert "does not execute PDF text extraction" in markdown
