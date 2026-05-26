from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.tex_equation_line_local_anchor_audit import (
    TEX_EQUATION_LINE_LOCAL_ANCHOR_AUDIT_SCHEMA_ID,
)
from knowledge_hub.papers.tex_equation_pdf_region_anchor_audit import (
    TEX_EQUATION_PDF_REGION_ANCHOR_AUDIT_SCHEMA_ID,
    build_tex_equation_pdf_region_anchor_audit,
    write_tex_equation_pdf_region_anchor_audit_reports,
)


def _write_json(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _parsed_root(root: Path) -> Path:
    parsed = root / "parsed"
    paper_root = parsed / "paper-1"
    paper_root.mkdir(parents=True)
    source_pdf = root / "source.pdf"
    source_pdf.write_bytes(b"not a real pdf; fake block loader is used")
    _write_json(
        paper_root,
        "manifest.json",
        {
            "paper_id": "paper-1",
            "parser_meta": {
                "parser": "pymupdf",
                "source_pdf": str(source_pdf),
                "extracted_from": str(source_pdf),
            },
        },
    )
    return parsed


def _line_row(
    anchor_id: str,
    text: str,
    terms: list[str],
    *,
    status: str = "unique_line_local_anchor_candidate_only",
    normalized_window_count: int = 1,
    page: int = 1,
) -> dict:
    return {
        "anchor_id": anchor_id,
        "source_design_id": anchor_id.replace("anchor", "design"),
        "source_diagnostic_id": anchor_id.replace("anchor", "diagnostic"),
        "source_candidate_id": anchor_id.replace("anchor", "candidate"),
        "paper_id": "paper-1",
        "candidate_type": "tex_equation_line_local_anchor_candidate",
        "source_parser": "arxiv_tex+pymupdf_alignment",
        "source_file": "main.tex",
        "equation_environment": "equation",
        "candidate_text": text,
        "recommended_profile": "canonical_math_compaction_v1",
        "normalized_terms": terms,
        "normalized_term_count": len(terms),
        "canonical_document_path": "",
        "canonical_document_available": True,
        "normalized_window_count": normalized_window_count,
        "distinct_line_count": 1 if normalized_window_count else 0,
        "distinct_page_marker_count": 1 if normalized_window_count else 0,
        "equation_number_candidates": ["1"],
        "window_details": [
            {
                "window_index": index,
                "line_number": 10,
                "page_marker": page,
                "line_equation_numbers": ["1"],
                "context_equation_numbers": ["1"],
                "line_preview": text,
                "context_preview": text,
            }
            for index in range(1, normalized_window_count + 1)
        ],
        "line_local_anchor_status": status,
        "line_local_anchor_method": "same_line_or_equation_number",
        "recommended_action": "test",
        "sourceContentHash": "",
        "chars_start": None,
        "chars_end": None,
        "page": None,
        "source_span_created": False,
        "equation_semantics_interpreted": False,
        "equation_region_verified": False,
        "evidence_tier": "tex_equation_line_local_anchor_candidate_only",
        "confidence": 0.3,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": ["line_local_windows_are_diagnostic_not_provenance"],
        "non_strict_reason": ["line_local_anchor_rows_are_not_evidence"],
    }


def _line_report(root: Path, rows: list[dict], *, schema: str | None = None) -> Path:
    return _write_json(
        root,
        "line-local.json",
        {
            "schema": schema or TEX_EQUATION_LINE_LOCAL_ANCHOR_AUDIT_SCHEMA_ID,
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


def _fake_pdf_blocks(_: str | Path) -> list[dict]:
    return [
        {
            "page": 1,
            "blocks": [
                {
                    "block_index": 0,
                    "bbox": [50.0, 50.0, 280.0, 420.0],
                    "text": (
                        "A long prose block mentions Pr Class Object IOU pred truth repeatedly, "
                        "but it is explanatory text without an equality formula. " * 8
                    ),
                },
                {
                    "block_index": 1,
                    "bbox": [310.0, 120.0, 540.0, 140.0],
                    "text": "Pr(Classi|Object) * Pr(Object) * IOUtruth pred = Pr(Classi) * IOUtruth pred\n(1)",
                },
                {
                    "block_index": 5,
                    "bbox": [230.0, 330.0, 390.0, 350.0],
                    "text": "PE(pos,2i) = sin(pos/100002i/dmodel)",
                },
                {
                    "block_index": 6,
                    "bbox": [225.0, 353.0, 390.0, 367.0],
                    "text": "PE(pos,2i+1) = cos(pos/100002i/dmodel)",
                },
                {
                    "block_index": 7,
                    "bbox": [112.0, 677.0, 142.0, 688.0],
                    "text": "φ(x) =",
                },
                {
                    "block_index": 8,
                    "bbox": [145.0, 667.0, 286.0, 696.0],
                    "text": "( x, if x > 0\n0.1x, otherwise\n(2)",
                },
                {
                    "block_index": 9,
                    "bbox": [235.0, 460.0, 504.0, 472.0],
                    "text": "h = W0x + ∆Wx = W0x + BAx (3)",
                },
            ],
        }
    ]


def test_pdf_region_anchor_resolves_ambiguous_line_local_formula_region(tmp_path: Path) -> None:
    parsed = _parsed_root(tmp_path)
    report_path = _line_report(
        tmp_path,
        [
            _line_row(
                "anchor:0001",
                "Pr(Class_i|Object) * Pr(Object) * IOU_pred_truth = Pr(Class_i) * IOU_pred_truth",
                ["Pr", "Class", "Object", "IOU", "pred", "truth"],
                status="ambiguous_same_line_local_anchor_candidate_only",
                normalized_window_count=2,
            )
        ],
    )

    payload = build_tex_equation_pdf_region_anchor_audit(
        line_local_anchor_report=report_path,
        parsed_root=parsed,
        pdf_block_loader=_fake_pdf_blocks,
    )

    assert payload["schema"] == TEX_EQUATION_PDF_REGION_ANCHOR_AUDIT_SCHEMA_ID
    assert validate_payload(payload, TEX_EQUATION_PDF_REGION_ANCHOR_AUDIT_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["pdfRegionResolvedAmbiguousRows"] == 1
    row = payload["rows"][0]
    assert row["pdf_region_anchor_status"] == "pdf_region_resolves_line_local_ambiguity_candidate_only"
    assert row["selected_pdf_region"]["block_indexes"] == [1]
    assert row["line_local_ambiguity_resolved_by_pdf_region"] is True
    assert row["strict_eligible"] is False
    assert row["runtime_evidence"] is False


def test_pdf_region_anchor_uses_adjacent_block_window_for_split_equation(tmp_path: Path) -> None:
    parsed = _parsed_root(tmp_path)
    report_path = _line_report(
        tmp_path,
        [
            _line_row(
                "anchor:0002",
                "PE(pos,2i) = sin(pos / 10000^{2i/dmodel}); PE(pos,2i+1) = cos(pos / 10000^{2i/dmodel})",
                ["PE", "pos", "sin", "10000", "dmodel", "cos"],
            )
        ],
    )

    payload = build_tex_equation_pdf_region_anchor_audit(
        line_local_anchor_report=report_path,
        parsed_root=parsed,
        pdf_block_loader=_fake_pdf_blocks,
    )

    row = payload["rows"][0]
    assert row["pdf_region_anchor_status"] == "unique_pdf_region_anchor_candidate_only"
    assert row["selected_pdf_region"]["block_indexes"] == [5, 6]
    assert row["selected_pdf_region"]["coverage"] == 1.0
    assert row["source_span_created"] is False
    assert row["bbox"]


def test_pdf_region_anchor_bridges_unicode_and_compact_pdf_tokens(tmp_path: Path) -> None:
    parsed = _parsed_root(tmp_path)
    report_path = _line_report(
        tmp_path,
        [
            _line_row(
                "anchor:0003",
                r"\phi(x) = \begin{cases} x, & \text{if } x > 0\\ 0.1x, & \text{otherwise} \end{cases}",
                ["phi", "if", "01", "otherwise"],
            ),
            _line_row(
                "anchor:0004",
                r"h = W_0 x + \Delta W x = W_0 x + BA x",
                ["W0", "Delta", "BA"],
            ),
        ],
    )

    payload = build_tex_equation_pdf_region_anchor_audit(
        line_local_anchor_report=report_path,
        parsed_root=parsed,
        pdf_block_loader=_fake_pdf_blocks,
    )

    by_anchor = {row["source_line_local_anchor_id"]: row for row in payload["rows"]}
    phi_row = by_anchor["anchor:0003"]
    assert phi_row["pdf_region_anchor_status"] == "unique_pdf_region_anchor_candidate_only"
    assert phi_row["selected_pdf_region"]["block_indexes"] == [7, 8]
    assert phi_row["selected_pdf_region"]["matched_terms"] == ["phi", "if", "01", "otherwise"]
    assert phi_row["strict_eligible"] is False

    lora_row = by_anchor["anchor:0004"]
    assert lora_row["pdf_region_anchor_status"] == "unique_pdf_region_anchor_candidate_only"
    assert lora_row["selected_pdf_region"]["block_indexes"] == [9]
    assert lora_row["selected_pdf_region"]["matched_terms"] == ["W0", "Delta", "BA"]
    assert lora_row["runtime_evidence"] is False


def test_pdf_region_anchor_blocks_rows_without_line_local_window(tmp_path: Path) -> None:
    parsed = _parsed_root(tmp_path)
    report_path = _line_report(
        tmp_path,
        [_line_row("anchor:0005", "missing", ["Missing", "Pair"], normalized_window_count=0)],
    )

    payload = build_tex_equation_pdf_region_anchor_audit(
        line_local_anchor_report=report_path,
        parsed_root=parsed,
        pdf_block_loader=_fake_pdf_blocks,
    )

    row = payload["rows"][0]
    assert row["pdf_region_anchor_status"] == "blocked_no_line_local_normalized_window"
    assert row["pdf_region_candidate_count"] == 0
    assert row["selected_pdf_region"]["bbox"] == []


def test_pdf_region_anchor_can_recover_pdf_only_region_without_line_local_window(tmp_path: Path) -> None:
    parsed = _parsed_root(tmp_path)
    report_path = _line_report(
        tmp_path,
        [
            _line_row(
                "anchor:0006",
                "h = W0x + Delta W x = W0x + BAx",
                ["W0", "Delta", "BA"],
                normalized_window_count=0,
            )
        ],
    )

    payload = build_tex_equation_pdf_region_anchor_audit(
        line_local_anchor_report=report_path,
        parsed_root=parsed,
        pdf_block_loader=_fake_pdf_blocks,
    )

    row = payload["rows"][0]
    assert row["pdf_region_anchor_status"] == "unique_pdf_region_without_line_local_window_candidate_only"
    assert row["pdf_region_anchor_method"] == "formula_like_pdf_block_window_without_line_local_window"
    assert row["selected_pdf_region"]["block_indexes"] == [9]
    assert row["normalized_window_count"] == 0
    assert row["pdf_region_anchor_unique"] is True
    assert row["strict_eligible"] is False
    assert row["runtime_evidence"] is False
    assert "canonical_line_local_anchor_missing_for_pdf_only_region_candidate" in row["strict_blockers"]
    assert "pdf_region_recovered_without_canonical_line_local_window" in row["non_strict_reason"]
    assert payload["counts"]["pdfRegionWithoutLineLocalWindowRows"] == 1
    assert payload["counts"]["strictEligibleRows"] == 0


def test_pdf_region_anchor_blocks_wrong_parent_schema(tmp_path: Path) -> None:
    parsed = _parsed_root(tmp_path)
    report_path = _line_report(
        tmp_path,
        [_line_row("anchor:0001", "x = y", ["x", "y"])],
        schema="example.wrong",
    )

    payload = build_tex_equation_pdf_region_anchor_audit(
        line_local_anchor_report=report_path,
        parsed_root=parsed,
        pdf_block_loader=_fake_pdf_blocks,
    )

    assert validate_payload(payload, TEX_EQUATION_PDF_REGION_ANCHOR_AUDIT_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "blocked"
    assert payload["counts"]["lineLocalAnchorRows"] == 0
    assert payload["gate"]["schemaViolations"] == ["tex_equation_line_local_anchor_audit_schema_mismatch"]


def test_pdf_region_anchor_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    parsed = _parsed_root(tmp_path)
    report_path = _line_report(
        tmp_path,
        [
            _line_row(
                "anchor:0001",
                "Pr(Class_i|Object) * Pr(Object) * IOU_pred_truth = Pr(Class_i) * IOU_pred_truth",
                ["Pr", "Class", "Object", "IOU", "pred", "truth"],
                status="ambiguous_same_line_local_anchor_candidate_only",
                normalized_window_count=2,
            )
        ],
    )
    payload = build_tex_equation_pdf_region_anchor_audit(
        line_local_anchor_report=report_path,
        parsed_root=parsed,
        pdf_block_loader=_fake_pdf_blocks,
    )

    paths = write_tex_equation_pdf_region_anchor_audit_reports(payload, tmp_path / "reports")

    assert set(paths) == {"report", "summary", "markdown"}
    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(report, TEX_EQUATION_PDF_REGION_ANCHOR_AUDIT_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["sourceSpanCreatedRows"] == 0
    assert "diagnostic candidates, not source spans or evidence" in markdown
