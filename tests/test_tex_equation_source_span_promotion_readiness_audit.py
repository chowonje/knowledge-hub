from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.tex_equation_label_number_pdf_region_disambiguation_design import (
    TEX_EQUATION_LABEL_NUMBER_PDF_REGION_DISAMBIGUATION_DESIGN_SCHEMA_ID,
)
from knowledge_hub.papers.tex_equation_pdf_region_anchor_audit import (
    TEX_EQUATION_PDF_REGION_ANCHOR_AUDIT_SCHEMA_ID,
)
from knowledge_hub.papers.tex_equation_rendered_macro_term_profile_design import (
    TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_SCHEMA_ID,
)
from knowledge_hub.papers.tex_equation_segmented_multiline_matching_design import (
    TEX_EQUATION_SEGMENTED_MULTILINE_MATCHING_DESIGN_SCHEMA_ID,
)
from knowledge_hub.papers.tex_equation_source_span_promotion_readiness_audit import (
    TEX_EQUATION_SOURCE_SPAN_PROMOTION_READINESS_AUDIT_SCHEMA_ID,
    build_tex_equation_source_span_promotion_readiness_audit,
    write_tex_equation_source_span_promotion_readiness_audit_reports,
)


def _write_json(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _rendered_row(source_candidate_id: str, paper_id: str = "1512.03385", action: str = "reviewed") -> dict:
    return {
        "design_id": f"rendered:{source_candidate_id}",
        "source_diagnostic_id": f"diagnostic:{source_candidate_id}",
        "source_pdf_region_anchor_id": f"pdf-region:{source_candidate_id}",
        "source_line_local_anchor_id": f"line-local:{source_candidate_id}",
        "source_design_id": f"design:{source_candidate_id}",
        "source_candidate_id": source_candidate_id,
        "paper_id": paper_id,
        "source_parser": "arxiv_tex+pymupdf_pdf_blocks",
        "source_file": "main.tex",
        "equation_environment": "equation",
        "candidate_text": f"y = W x + b ({source_candidate_id})",
        "recommended_action": action,
        "source_span_created": False,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "parser_routing_changed": False,
        "answer_integration_changed": False,
    }


def _label_row(
    source_candidate_id: str,
    paper_id: str,
    status: str,
    *,
    source_content_hash: str,
) -> dict:
    return {
        "source_rendered_macro_design_id": f"design:{source_candidate_id}",
        "source_diagnostic_id": f"diagnostic:{source_candidate_id}",
        "source_pdf_region_anchor_id": f"pdf-region:{source_candidate_id}",
        "source_line_local_anchor_id": f"line-local:{source_candidate_id}",
        "source_design_id": f"design:{source_candidate_id}",
        "source_candidate_id": source_candidate_id,
        "paper_id": paper_id,
        "source_parser": "arxiv_tex+pymupdf_pdf_blocks",
        "source_file": "main.tex",
        "equation_environment": "equation",
        "candidate_text": "y = W x + b",
        "sourceContentHash": source_content_hash,
        "source_pdf_path": "paper.pdf",
        "source_manifest_path": "manifest.json",
        "source_label_number_hint": {
            "sourceStructureRowId": f"source-row:{source_candidate_id}",
            "texEnvironment": "equation",
            "latexLabels": ["eq:ready"],
            "inferredEquationNumbers": ["1"],
            "method": "test",
            "status": "ok",
        },
        "selected_pdf_region": {
            "page": 1,
            "bbox": [10.0, 10.0, 100.0, 20.0],
            "blockIndexes": [1],
            "matchedTerms": ["W", "x", "b"],
            "coverage": 1.0,
            "formulaScore": 0.9,
            "textPreview": "y = W x + b",
        },
        "disambiguation_status": status,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "source_span_created": False,
        "parser_routing_changed": False,
        "answer_integration_changed": False,
        "strict_blockers": [],
        "non_strict_reason": [],
    }


def _segment_row(
    source_candidate_id: str,
    paper_id: str,
    *,
    segments: list[dict],
    source_content_hash: str,
) -> dict:
    return {
        "source_rendered_macro_design_id": f"design:{source_candidate_id}",
        "source_diagnostic_id": f"diagnostic:{source_candidate_id}",
        "source_pdf_region_anchor_id": f"pdf-region:{source_candidate_id}",
        "source_line_local_anchor_id": f"line-local:{source_candidate_id}",
        "source_design_id": f"design:{source_candidate_id}",
        "source_candidate_id": source_candidate_id,
        "paper_id": paper_id,
        "source_parser": "arxiv_tex+pymupdf_pdf_blocks",
        "source_file": "main.tex",
        "equation_environment": "align",
        "candidate_text": "x = y + z",
        "sourceContentHash": source_content_hash,
        "segment_count": len(segments),
        "candidate_ready_segment_count": sum(1 for item in segments if bool(item.get("segment_candidate_ready"))),
        "segments": segments,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "source_span_created": False,
        "parser_routing_changed": False,
        "answer_integration_changed": False,
        "strict_blockers": [],
        "non_strict_reason": [],
    }


def _pdf_anchor_row(
    source_candidate_id: str,
    paper_id: str,
    *,
    status: str,
    source_content_hash: str,
) -> dict:
    return {
        "source_candidate_id": source_candidate_id,
        "source_rendered_macro_design_id": f"design:{source_candidate_id}",
        "source_diagnostic_id": f"diagnostic:{source_candidate_id}",
        "source_pdf_region_anchor_id": f"pdf-region:{source_candidate_id}",
        "source_line_local_anchor_id": f"line-local:{source_candidate_id}",
        "source_design_id": f"design:{source_candidate_id}",
        "paper_id": paper_id,
        "source_parser": "arxiv_tex+pymupdf_pdf_blocks",
        "source_file": "main.tex",
        "equation_environment": "equation",
        "candidate_text": "y = W x + b",
        "sourceContentHash": source_content_hash,
        "pdf_region_candidate_count": 1,
        "pdf_region_anchor_status": status,
        "selected_pdf_region": {
            "page": 2,
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "blockIndexes": [10],
            "matchedTerms": ["y", "x", "b"],
            "coverage": 0.93,
            "formulaScore": 1.0,
            "textPreview": "y = W x + b",
        },
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "source_span_created": False,
        "parser_routing_changed": False,
        "answer_integration_changed": False,
        "strict_blockers": [],
        "non_strict_reason": [],
    }


def _report(path: Path, name: str, rows: list[dict], schema: str | None = None) -> Path:
    return _write_json(
        path,
        name,
        {
            "schema": schema or TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_SCHEMA_ID,
            "status": "ok",
            "generatedAt": "2026-05-19T00:00:00Z",
            "rows": rows,
        },
    )


def test_build_readiness_audit_classifies_all_readiness_categories(tmp_path: Path) -> None:
    rendered_rows = [
        _rendered_row("candidate:ready", action="requires_equation_number_or_label_to_pdf_region_disambiguation_design"),
    ]
    rendered_report = _report(tmp_path, "rendered.json", rendered_rows)

    label_rows = [
        _label_row("candidate:ready", "1512.03385", "unique_label_number_pdf_region_candidate_only", source_content_hash="hash-ready"),
        _label_row(
            "candidate:missing-hash",
            "1512.03385",
            "unique_label_number_pdf_region_candidate_only",
            source_content_hash="",
        ),
        _label_row("candidate:ambiguous", "1512.03385", "ambiguous_label_number_pdf_region_candidate_only", source_content_hash="hash-amb"),
        _label_row(
            "candidate:missing-label",
            "1512.03385",
            "blocked_no_label_number_hint",
            source_content_hash="hash-label-missing",
        ),
    ]
    label_report = _report(
        tmp_path,
        "label.json",
        label_rows,
        schema=TEX_EQUATION_LABEL_NUMBER_PDF_REGION_DISAMBIGUATION_DESIGN_SCHEMA_ID,
    )

    segmented_rows = [
        _segment_row(
            "candidate:manual",
            "1512.03385",
            source_content_hash="hash-manual",
            segments=[{"segment_index": 0, "segment_text": "x", "segment_candidate_ready": False}],
        )
    ]
    segmented_report = _report(
        tmp_path,
        "segmented.json",
        segmented_rows,
        schema=TEX_EQUATION_SEGMENTED_MULTILINE_MATCHING_DESIGN_SCHEMA_ID,
    )

    pdf_rows = [
        _pdf_anchor_row("candidate:pdf-only", "1512.03385", status="unique_pdf_region_anchor_candidate_only", source_content_hash="hash-pdf-only")
    ]
    pdf_report = _report(
        tmp_path,
        "pdf-anchor.json",
        pdf_rows,
        schema=TEX_EQUATION_PDF_REGION_ANCHOR_AUDIT_SCHEMA_ID,
    )

    payload = build_tex_equation_source_span_promotion_readiness_audit(
        rendered_macro_term_profile_design_report=rendered_report,
        label_number_pdf_region_disambiguation_design_report=label_report,
        segmented_multiline_matching_design_report=segmented_report,
        pdf_region_anchor_audit_report=pdf_report,
    )

    assert payload["schema"] == TEX_EQUATION_SOURCE_SPAN_PROMOTION_READINESS_AUDIT_SCHEMA_ID
    assert validate_payload(payload, TEX_EQUATION_SOURCE_SPAN_PROMOTION_READINESS_AUDIT_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "ok"
    counts = payload["counts"]
    assert counts["targetRows"] == 6
    assert counts["promotionReviewReadyCandidateOnlyRows"] == 1
    assert counts["blockedMissingSourceHashRows"] == 1
    assert counts["blockedAmbiguousPdfRegionRows"] == 1
    assert counts["blockedMissingLabelNumberRows"] == 1
    assert counts["blockedPdfRegionOnlyRows"] == 1
    assert counts["blockedManualOrLaterExtractorRows"] == 1
    assert counts["sourceSpanCreatedRows"] == 0

    by_category = {row["readiness_category"] for row in payload["rows"]}
    assert by_category == {
        "promotion_review_ready_candidate_only",
        "blocked_missing_source_hash",
        "blocked_ambiguous_pdf_region",
        "blocked_missing_label_number_disambiguation",
        "blocked_pdf_region_only_not_source_span",
        "blocked_requires_manual_or_later_extractor_review",
    }
    for row in payload["rows"]:
        assert row["source_span_created"] is False
        assert row["strict_eligible"] is False
        assert row["citation_grade"] is False
        assert row["runtime_evidence"] is False
        assert row["parser_routing_changed"] is False
        assert row["answer_integration_changed"] is False


def test_readiness_audit_blocks_schema_mismatch(tmp_path: Path) -> None:
    rendered_report = _report(
        tmp_path,
        "rendered.json",
        [],
        schema="example.wrong",
    )
    payload = build_tex_equation_source_span_promotion_readiness_audit(
        rendered_macro_term_profile_design_report=rendered_report,
        label_number_pdf_region_disambiguation_design_report=_report(
            tmp_path,
            "label.json",
            [],
            schema=TEX_EQUATION_LABEL_NUMBER_PDF_REGION_DISAMBIGUATION_DESIGN_SCHEMA_ID,
        ),
        segmented_multiline_matching_design_report=_report(
            tmp_path,
            "segmented.json",
            [],
            schema=TEX_EQUATION_SEGMENTED_MULTILINE_MATCHING_DESIGN_SCHEMA_ID,
        ),
        pdf_region_anchor_audit_report=_report(
            tmp_path,
            "pdf-anchor.json",
            [],
            schema=TEX_EQUATION_PDF_REGION_ANCHOR_AUDIT_SCHEMA_ID,
        ),
    )

    assert payload["status"] == "blocked"
    assert payload["counts"]["schemaViolationCount"] == 1
    assert payload["counts"]["targetRows"] == 0
    assert payload["gate"]["schemaViolations"] == [
        "tex_equation_rendered_macro_term_profile_design_schema_mismatch",
    ]
    assert payload["rows"] == []


def test_readiness_audit_report_outputs_validate(tmp_path: Path) -> None:
    rendered_report = _report(
        tmp_path,
        "rendered.json",
        [_rendered_row("candidate:ready", action="requires_equation_number_or_label_to_pdf_region_disambiguation_design")],
    )
    label_report = _report(
        tmp_path,
        "label.json",
        [
            _label_row(
                "candidate:ready",
                "1512.03385",
                "unique_label_number_pdf_region_candidate_only",
                source_content_hash="hash-ready",
            )
        ],
        schema=TEX_EQUATION_LABEL_NUMBER_PDF_REGION_DISAMBIGUATION_DESIGN_SCHEMA_ID,
    )
    segmented_report = _report(
        tmp_path,
        "segmented.json",
        [],
        schema=TEX_EQUATION_SEGMENTED_MULTILINE_MATCHING_DESIGN_SCHEMA_ID,
    )
    pdf_report = _report(
        tmp_path,
        "pdf-anchor.json",
        [],
        schema=TEX_EQUATION_PDF_REGION_ANCHOR_AUDIT_SCHEMA_ID,
    )

    payload = build_tex_equation_source_span_promotion_readiness_audit(
        rendered_macro_term_profile_design_report=rendered_report,
        label_number_pdf_region_disambiguation_design_report=label_report,
        segmented_multiline_matching_design_report=segmented_report,
        pdf_region_anchor_audit_report=pdf_report,
    )
    paths = write_tex_equation_source_span_promotion_readiness_audit_reports(payload, tmp_path / "reports")

    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert set(paths) == {"report", "summary", "markdown"}
    assert validate_payload(report, TEX_EQUATION_SOURCE_SPAN_PROMOTION_READINESS_AUDIT_SCHEMA_ID, strict=True).ok
    assert summary["status"] == "ok"
    assert summary["counts"]["promotionReviewReadyCandidateOnlyRows"] == 1
    assert "TeX Equation Source-Span Promotion Readiness Audit" in markdown
