from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.tex_equation_pdf_region_anchor_audit import (
    TEX_EQUATION_PDF_REGION_ANCHOR_AUDIT_SCHEMA_ID,
)
from knowledge_hub.papers.tex_equation_quote_candidate_v2_design import (
    TEX_EQUATION_QUOTE_CANDIDATE_V2_DESIGN_SCHEMA_ID,
    build_tex_equation_quote_candidate_v2_design,
    write_tex_equation_quote_candidate_v2_design_reports,
)


def _write_json(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _pdf_region_row(
    row_id: str,
    text: str,
    *,
    paper_id: str = "paper-1",
    status: str = "unique_pdf_region_anchor_candidate_only",
    unique: bool = True,
    source_hash: str = "source-hash",
    bbox: list[float] | None = None,
    page: int | None = 2,
    resolved_ambiguous: bool = False,
) -> dict:
    return {
        "pdf_region_anchor_id": row_id,
        "source_line_local_anchor_id": row_id.replace("pdf", "line"),
        "source_design_id": row_id.replace("pdf", "design"),
        "source_candidate_id": row_id.replace("pdf", "candidate"),
        "paper_id": paper_id,
        "candidate_type": "tex_equation_pdf_region_anchor_candidate",
        "source_parser": "arxiv_tex+pymupdf_pdf_blocks",
        "source_file": "main.tex",
        "equation_environment": "equation",
        "candidate_text": text,
        "normalized_terms": ["Pr", "Class", "Object"],
        "normalized_window_count": 1,
        "line_local_anchor_status": (
            "ambiguous_same_line_local_anchor_candidate_only"
            if resolved_ambiguous
            else "unique_line_local_anchor_candidate_only"
        ),
        "line_local_anchor_method": "formula_like_pdf_block_window",
        "canonical_page_markers": [page] if page else [],
        "source_pdf_path": "/tmp/source.pdf",
        "source_manifest_path": "/tmp/manifest.json",
        "sourceContentHash": source_hash,
        "input_sourceContentHash": source_hash,
        "source_hash_agrees_with_input": bool(source_hash),
        "pdf_region_anchor_status": status,
        "pdf_region_anchor_method": "formula_like_pdf_block_window",
        "pdf_region_candidate_count": 1 if unique else 2,
        "pdf_region_candidates": [],
        "selected_pdf_region": {
            "page": page,
            "bbox": bbox if bbox is not None else [10.0, 20.0, 200.0, 40.0],
            "block_indexes": [3],
            "matched_terms": ["Pr", "Class", "Object"],
            "coverage": 1.0,
            "formula_score": 1.55,
            "equation_numbers": ["1"],
            "text_preview": text,
        },
        "pdf_region_anchor_unique": unique,
        "line_local_ambiguity_resolved_by_pdf_region": resolved_ambiguous,
        "feasibility_failure_reason": "" if unique else "pdf_region_anchor_ambiguous",
        "chars_start": None,
        "chars_end": None,
        "page": None,
        "bbox": bbox if bbox is not None else [10.0, 20.0, 200.0, 40.0],
        "source_span_created": False,
        "equation_semantics_interpreted": False,
        "equation_region_verified": False,
        "evidence_tier": "tex_equation_pdf_region_anchor_candidate_only",
        "confidence": 0.7,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": ["pdf_region_bbox_is_not_source_span"],
        "non_strict_reason": ["pdf_region_anchor_rows_are_not_evidence"],
    }


def _pdf_region_report(root: Path, rows: list[dict], *, schema: str | None = None) -> Path:
    return _write_json(
        root,
        "pdf-region.json",
        {
            "schema": schema or TEX_EQUATION_PDF_REGION_ANCHOR_AUDIT_SCHEMA_ID,
            "status": "ok",
            "generatedAt": "2026-05-18T00:00:00Z",
            "input": {},
            "counts": {},
            "gate": {},
            "policy": {},
            "warnings": [],
            "rows": rows,
        },
    )


def test_equation_quote_v2_design_emits_unique_pdf_region_rows_non_strict(tmp_path: Path) -> None:
    report_path = _pdf_region_report(
        tmp_path,
        [
            _pdf_region_row(
                "pdf:0001",
                r"\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^T/\sqrt{d_k})V",
                resolved_ambiguous=True,
            )
        ],
    )

    payload = build_tex_equation_quote_candidate_v2_design(report_path)

    assert payload["schema"] == TEX_EQUATION_QUOTE_CANDIDATE_V2_DESIGN_SCHEMA_ID
    assert validate_payload(payload, TEX_EQUATION_QUOTE_CANDIDATE_V2_DESIGN_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["v2DesignCandidates"] == 1
    assert payload["counts"]["pdfRegionResolvedAmbiguousRows"] == 1
    candidate = payload["candidates"][0]
    assert candidate["candidate_type"] == "equation_quote_candidate_v2_design"
    assert candidate["pdf_region"]["page"] == 2
    assert candidate["pdf_region"]["bbox"] == [10.0, 20.0, 200.0, 40.0]
    assert candidate["equation_label"] == "pdf-equation-number:1"
    assert candidate["source_span_created"] is False
    assert candidate["strict_eligible"] is False
    assert candidate["runtime_evidence"] is False
    assert "pdf_region_bbox_is_not_source_span" in candidate["strict_blockers"]


def test_equation_quote_v2_design_holds_out_missing_or_nonunique_region_rows(tmp_path: Path) -> None:
    report_path = _pdf_region_report(
        tmp_path,
        [
            _pdf_region_row("pdf:0001", "x = y", unique=False, status="ambiguous_pdf_region_anchor_candidate_only"),
            _pdf_region_row("pdf:0002", "x = y", source_hash=""),
            _pdf_region_row("pdf:0003", "x = y", bbox=[]),
            _pdf_region_row("pdf:0004", "", status="empty_equation_text", unique=False),
        ],
    )

    payload = build_tex_equation_quote_candidate_v2_design(report_path)

    assert payload["counts"]["v2DesignCandidates"] == 0
    assert payload["counts"]["heldOutRows"] == 4
    assert payload["counts"]["heldOutByReason"] == {
        "empty_equation_text": 1,
        "missing_pdf_region_bbox": 1,
        "missing_source_content_hash": 1,
        "pdf_region_anchor_not_unique": 1,
    }
    assert payload["counts"]["strictEligibleCandidates"] == 0


def test_equation_quote_v2_design_blocks_wrong_parent_schema(tmp_path: Path) -> None:
    report_path = _pdf_region_report(
        tmp_path,
        [_pdf_region_row("pdf:0001", "x = y")],
        schema="example.wrong",
    )

    payload = build_tex_equation_quote_candidate_v2_design(report_path)

    assert validate_payload(payload, TEX_EQUATION_QUOTE_CANDIDATE_V2_DESIGN_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "blocked"
    assert payload["counts"]["inputRows"] == 0
    assert payload["gate"]["schemaViolations"] == [
        "tex_equation_pdf_region_anchor_audit_schema_mismatch"
    ]


def test_equation_quote_v2_design_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    report_path = _pdf_region_report(tmp_path, [_pdf_region_row("pdf:0001", "x = y")])
    payload = build_tex_equation_quote_candidate_v2_design(report_path)

    paths = write_tex_equation_quote_candidate_v2_design_reports(payload, tmp_path / "reports")

    assert set(paths) == {"report", "summary", "markdown"}
    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(report, TEX_EQUATION_QUOTE_CANDIDATE_V2_DESIGN_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["sourceSpanCreatedRows"] == 0
    assert "v2 design report only" in markdown
