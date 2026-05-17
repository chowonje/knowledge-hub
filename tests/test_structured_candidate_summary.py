from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.structured_candidate_summary import (
    STRUCTURED_CANDIDATE_SUMMARY_SCHEMA_ID,
    build_structured_candidate_summary,
    write_structured_candidate_summary_reports,
)


def _write_report(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _reports(root: Path) -> dict[str, Path]:
    return {
        "sectionspan": _write_report(
            root,
            "sectionspan.json",
            {
                "schema": "knowledge-hub.paper.sectionspan-candidate-report.v1",
                "status": "ok",
                "counts": {
                    "sectionSpanCandidates": 3,
                    "heldOutCandidates": 1,
                    "strictEligibleCandidates": 0,
                    "citationGradeCandidates": 0,
                    "byPaper": {"paper-1": 3},
                    "heldOutByReason": {"held_out_toc": 1},
                },
            },
        ),
        "figure_caption": _write_report(
            root,
            "figure.json",
            {
                "schema": "knowledge-hub.paper.figure-caption-candidate-report.v1",
                "status": "ok",
                "counts": {
                    "figureCaptionCandidates": 2,
                    "alignedCaptionSpanCandidates": 1,
                    "strictEligibleCandidates": 0,
                    "citationGradeCandidates": 0,
                    "byPaper": {"paper-1": 2},
                    "byReadiness": {
                        "caption_span_aligned_region_candidate_non_strict": 1,
                        "blocked_alignment_incomplete": 1,
                    },
                    "strictBlockerSummary": {"figure_region_link_incomplete": 2},
                },
            },
        ),
        "equation_quote": _write_report(
            root,
            "equation.json",
            {
                "schema": "knowledge-hub.paper.equation-quote-candidate-report.v1",
                "status": "ok",
                "counts": {
                    "equationQuoteCandidates": 1,
                    "alignedEquationQuoteCandidates": 0,
                    "strictEligibleCandidates": 0,
                    "citationGradeCandidates": 0,
                    "byPaper": {"paper-1": 1},
                    "byReadiness": {"blocked_alignment_incomplete": 1},
                    "strictBlockerSummary": {"equation_alignment_missing": 1},
                },
            },
        ),
        "table_region": _write_report(
            root,
            "table.json",
            {
                "schema": "knowledge-hub.paper.table-region-candidate-report.v1",
                "status": "ok",
                "counts": {
                    "tableRegionCandidates": 1,
                    "alignedTableCaptionCandidates": 1,
                    "strictEligibleCandidates": 0,
                    "citationGradeCandidates": 0,
                    "byPaper": {"paper-1": 1},
                    "byReadiness": {"caption_span_aligned_region_candidate_cell_blocked": 1},
                    "strictBlockerSummary": {"table_cell_provenance_missing": 1},
                },
            },
        ),
    }


def _sectionspan_offset_review_pack(root: Path) -> Path:
    return _write_report(
        root,
        "sectionspan-offset-review.json",
        {
            "schema": "knowledge-hub.paper.sectionspan-pdf-offset-recovery-review-pack.v1",
            "status": "review_pack_ready",
            "counts": {
                "reviewCardRows": 3,
                "readyForHumanReviewRows": 3,
                "heldOutRows": 0,
                "pageAgreementRows": 3,
                "sourceHashAgreementRows": 3,
                "strictEligibleRows": 0,
                "citationGradeRows": 0,
                "runtimeEvidenceRows": 0,
                "unsafeUpstreamFlagCount": 0,
                "byMatchMethod": {"exact": 2, "normalized_whitespace_case": 1},
                "byReviewStatus": {"ready_for_human_review": 3},
            },
        },
    )


def _figure_caption_pdf_offset_feasibility(root: Path) -> Path:
    return _write_report(
        root,
        "figure-caption-pdf-offset-feasibility.json",
        {
            "schema": "knowledge-hub.paper.figure-caption-pdf-offset-feasibility.v1",
            "status": "feasibility_complete",
            "counts": {
                "feasibilityRows": 2,
                "originalPdfOffsetRecoveredRows": 1,
                "blockedRows": 1,
                "exactRecoveredRows": 1,
                "normalizedRecoveredRows": 0,
                "pageAgreementRows": 1,
                "sourceHashAgreementRows": 1,
                "strictEligibleRows": 0,
                "citationGradeRows": 0,
                "runtimeEvidenceRows": 0,
                "schemaViolationCount": 0,
                "byFeasibilityStatus": {"recovered_exact": 1, "blocked_no_match": 1},
            },
        },
    )


def test_structured_candidate_summary_aggregates_four_layers_and_validates_schema(tmp_path: Path) -> None:
    paths = _reports(tmp_path)
    sectionspan_offset_path = _sectionspan_offset_review_pack(tmp_path)
    figure_offset_path = _figure_caption_pdf_offset_feasibility(tmp_path)

    payload = build_structured_candidate_summary(
        sectionspan_report=paths["sectionspan"],
        figure_caption_report=paths["figure_caption"],
        equation_quote_report=paths["equation_quote"],
        table_region_report=paths["table_region"],
        sectionspan_pdf_offset_review_pack_report=sectionspan_offset_path,
        figure_caption_pdf_offset_feasibility_report=figure_offset_path,
    )

    assert payload["schema"] == STRUCTURED_CANDIDATE_SUMMARY_SCHEMA_ID
    assert validate_payload(payload, STRUCTURED_CANDIDATE_SUMMARY_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["layerCount"] == 4
    assert payload["counts"]["totalCandidates"] == 7
    assert payload["counts"]["byLayer"] == {
        "sectionspan": 3,
        "figure_caption": 2,
        "equation_quote": 1,
        "table_region": 1,
    }
    assert payload["counts"]["alignedByLayer"] == {
        "sectionspan": 3,
        "figure_caption": 1,
        "equation_quote": 0,
        "table_region": 1,
    }
    assert payload["counts"]["blockedOrHeldOutByLayer"] == {
        "sectionspan": 1,
        "figure_caption": 1,
        "equation_quote": 1,
        "table_region": 0,
    }
    assert payload["counts"]["sourceAlignmentSupplementCount"] == 2
    assert payload["counts"]["sectionspanOriginalPdfOffsetReviewCards"] == 3
    assert payload["counts"]["sectionspanOriginalPdfOffsetReadyForReviewRows"] == 3
    assert payload["counts"]["sectionspanOriginalPdfOffsetHeldOutRows"] == 0
    assert payload["counts"]["figureCaptionOriginalPdfOffsetFeasibilityRows"] == 2
    assert payload["counts"]["figureCaptionOriginalPdfOffsetRecoveredRows"] == 1
    assert payload["counts"]["figureCaptionOriginalPdfOffsetBlockedRows"] == 1
    assert payload["sourceAlignmentSupplements"][0]["readyForReview"] is True
    assert payload["sourceAlignmentSupplements"][1]["supplement"] == "figure_caption_pdf_offset_feasibility"
    assert payload["sourceAlignmentSupplements"][1]["readyForRegionReviewRows"] == 1


def test_structured_candidate_summary_keeps_runtime_promotion_closed(tmp_path: Path) -> None:
    paths = _reports(tmp_path)
    sectionspan_offset_path = _sectionspan_offset_review_pack(tmp_path)
    figure_offset_path = _figure_caption_pdf_offset_feasibility(tmp_path)

    payload = build_structured_candidate_summary(
        sectionspan_report=paths["sectionspan"],
        figure_caption_report=paths["figure_caption"],
        equation_quote_report=paths["equation_quote"],
        table_region_report=paths["table_region"],
        sectionspan_pdf_offset_review_pack_report=sectionspan_offset_path,
        figure_caption_pdf_offset_feasibility_report=figure_offset_path,
    )

    assert payload["policy"]["strictEvidenceCreated"] is False
    assert payload["policy"]["runtimePromotionAllowed"] is False
    assert payload["policy"]["parserRoutingChanged"] is False
    assert payload["policy"]["answerIntegrationChanged"] is False
    assert payload["counts"]["strictEligibleCandidates"] == 0
    assert payload["counts"]["citationGradeCandidates"] == 0
    assert payload["counts"]["runtimeEvidenceCandidates"] == 0
    assert payload["releaseCandidateAssessment"]["strictEvidenceReady"] is False
    assert payload["releaseCandidateAssessment"]["parserRoutingReady"] is False
    assert "sectionspan_pdf_offsets_require_human_review_before_strict_promotion" in payload["releaseCandidateAssessment"]["mainBlockers"]
    assert "figure_caption_pdf_offsets_require_region_link_review" in payload["releaseCandidateAssessment"]["mainBlockers"]
    assert "generated_markdown_offsets_are_not_original_pdf_offsets" not in payload["releaseCandidateAssessment"]["mainBlockers"]
    assert payload["sourceAlignmentSupplements"][0]["strictEligibleRows"] == 0
    assert payload["sourceAlignmentSupplements"][0]["runtimeEvidenceRows"] == 0
    assert payload["sourceAlignmentSupplements"][1]["strictEligibleRows"] == 0
    assert payload["sourceAlignmentSupplements"][1]["runtimeEvidenceRows"] == 0


def test_structured_candidate_summary_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    paths = _reports(tmp_path / "input")
    sectionspan_offset_path = _sectionspan_offset_review_pack(tmp_path / "input")
    figure_offset_path = _figure_caption_pdf_offset_feasibility(tmp_path / "input")
    payload = build_structured_candidate_summary(
        sectionspan_report=paths["sectionspan"],
        figure_caption_report=paths["figure_caption"],
        equation_quote_report=paths["equation_quote"],
        table_region_report=paths["table_region"],
        sectionspan_pdf_offset_review_pack_report=sectionspan_offset_path,
        figure_caption_pdf_offset_feasibility_report=figure_offset_path,
    )

    report_paths = write_structured_candidate_summary_reports(payload, tmp_path / "reports")

    assert set(report_paths) == {"summary", "markdown"}
    summary = json.loads(Path(report_paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(report_paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(summary, STRUCTURED_CANDIDATE_SUMMARY_SCHEMA_ID, strict=True).ok
    assert "All layer outputs remain non-strict candidates" in markdown
    assert "candidate_layer_review_gate_refresh" in markdown
    assert "SectionSpan original-PDF-offset review cards" in markdown
    assert "FigureCaption original-PDF-offset recovered" in markdown


def test_structured_candidate_summary_without_offset_review_keeps_original_offset_blocker(tmp_path: Path) -> None:
    paths = _reports(tmp_path)

    payload = build_structured_candidate_summary(
        sectionspan_report=paths["sectionspan"],
        figure_caption_report=paths["figure_caption"],
        equation_quote_report=paths["equation_quote"],
        table_region_report=paths["table_region"],
    )

    assert validate_payload(payload, STRUCTURED_CANDIDATE_SUMMARY_SCHEMA_ID, strict=True).ok
    assert payload["sourceAlignmentSupplements"][0]["status"] == "not_provided"
    assert payload["sourceAlignmentSupplements"][0]["readyForReview"] is False
    assert payload["sourceAlignmentSupplements"][1]["status"] == "not_provided"
    assert payload["sourceAlignmentSupplements"][1]["readyForReview"] is False
    assert "generated_markdown_offsets_are_not_original_pdf_offsets" in payload["releaseCandidateAssessment"]["mainBlockers"]
