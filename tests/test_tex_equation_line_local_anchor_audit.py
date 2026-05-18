from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.tex_equation_canonical_text_normalizer_design import (
    TEX_EQUATION_CANONICAL_TEXT_NORMALIZER_DESIGN_SCHEMA_ID,
)
from knowledge_hub.papers.tex_equation_line_local_anchor_audit import (
    TEX_EQUATION_LINE_LOCAL_ANCHOR_AUDIT_SCHEMA_ID,
    build_tex_equation_line_local_anchor_audit,
    write_tex_equation_line_local_anchor_audit_reports,
)


def _write_json(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _design_row(
    design_id: str,
    text: str,
    terms: list[str],
    *,
    paper_id: str = "paper-1",
    recommended_profile: str = "canonical_math_compaction_v1",
) -> dict:
    return {
        "design_id": design_id,
        "source_diagnostic_id": design_id.replace("design", "diagnostic"),
        "source_candidate_id": design_id.replace("design", "source"),
        "paper_id": paper_id,
        "candidate_type": "tex_equation_canonical_text_normalizer_design",
        "source_parser": "arxiv_tex+pymupdf_alignment",
        "source_file": "main.tex",
        "equation_environment": "equation",
        "candidate_text": text,
        "canonical_document_path": "",
        "canonical_document_available": True,
        "profile_results": [
            {
                "profile_name": recommended_profile,
                "description": "test profile",
                "proposed_rules": ["ordered_anchor_token_window"],
                "normalized_terms": terms,
                "normalized_term_count": len(terms),
                "window_count": 0,
                "status": "normalized_window_not_found",
                "confidence": 0.1,
            }
        ],
        "recommended_profile": recommended_profile,
        "recommended_status": "normalized_window_not_found",
        "recommended_action": "test",
        "sourceContentHash": "hash-source",
        "chars_start": None,
        "chars_end": None,
        "page": None,
        "source_span_created": False,
        "equation_semantics_interpreted": False,
        "equation_region_verified": False,
        "evidence_tier": "tex_equation_canonical_text_normalizer_design_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": ["normalized_windows_are_diagnostic_not_provenance"],
        "non_strict_reason": ["normalizer_design_rows_are_not_evidence"],
    }


def _design_report(root: Path, rows: list[dict], *, schema: str | None = None) -> Path:
    return _write_json(
        root,
        "design.json",
        {
            "schema": schema or TEX_EQUATION_CANONICAL_TEXT_NORMALIZER_DESIGN_SCHEMA_ID,
            "status": "ok",
            "generatedAt": "2026-05-18T00:00:00Z",
            "input": {},
            "counts": {},
            "gate": {},
            "policy": {},
            "profiles": [],
            "warnings": [],
            "rows": rows,
        },
    )


def _parsed_root(root: Path) -> Path:
    parsed = root / "parsed"
    (parsed / "paper-1").mkdir(parents=True)
    (parsed / "paper-1" / "document.md").write_text(
        "## Page 1\n"
        "Unicode bridge chars before the target: × × × √ − —\n"
        "The unique formula says FFN max xW1 b1 W2 b2 in the same line.\n"
        "A repeated phrase Alpha Beta Gamma is prose only.\n"
        "The numbered equation is Alpha Beta Gamma (1) in this line.\n"
        "Another repeated value Delta Epsilon Zeta appears here.\n"
        "Again Delta Epsilon Zeta appears there.\n",
        encoding="utf-8",
    )
    return parsed


def test_tex_equation_line_local_anchor_audit_classifies_unique_numbered_and_ambiguous_rows(
    tmp_path: Path,
) -> None:
    parsed = _parsed_root(tmp_path)
    report_path = _design_report(
        tmp_path,
        [
            _design_row("design:0001", "ffn", ["FFN", "max", "xW1", "b1", "W2", "b2"]),
            _design_row("design:0002", "alpha", ["Alpha", "Beta", "Gamma"]),
            _design_row("design:0003", "delta", ["Delta", "Epsilon", "Zeta"]),
            _design_row("design:0004", "missing", ["Missing", "Pair"]),
            _design_row("design:0005", "", []),
        ],
    )

    payload = build_tex_equation_line_local_anchor_audit(
        normalizer_design_report=report_path,
        parsed_root=parsed,
    )

    assert payload["schema"] == TEX_EQUATION_LINE_LOCAL_ANCHOR_AUDIT_SCHEMA_ID
    assert validate_payload(payload, TEX_EQUATION_LINE_LOCAL_ANCHOR_AUDIT_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "ok"
    assert payload["counts"]["normalizerDesignRows"] == 5
    assert payload["counts"]["uniqueLineLocalAnchorRows"] == 1
    assert payload["counts"]["uniqueEquationNumberAnchorRows"] == 1
    assert payload["counts"]["ambiguousAnchorRows"] == 1
    assert payload["counts"]["failedAnchorRows"] == 1
    assert payload["counts"]["strictEligibleRows"] == 0

    by_id = {row["source_design_id"]: row for row in payload["rows"]}
    assert by_id["design:0001"]["line_local_anchor_status"] == "unique_line_local_anchor_candidate_only"
    assert "The unique formula says" in by_id["design:0001"]["window_details"][0]["line_preview"]
    assert by_id["design:0002"]["line_local_anchor_status"] == "unique_equation_number_anchor_candidate_only"
    assert by_id["design:0002"]["equation_number_candidates"] == ["1"]
    assert by_id["design:0003"]["line_local_anchor_status"] == "ambiguous_line_local_anchor_candidate_only"
    assert by_id["design:0004"]["line_local_anchor_status"] == "no_normalized_windows"
    assert by_id["design:0005"]["line_local_anchor_status"] == "empty_equation_text"


def test_tex_equation_line_local_anchor_audit_never_creates_source_span_or_runtime_evidence(
    tmp_path: Path,
) -> None:
    parsed = _parsed_root(tmp_path)
    report_path = _design_report(tmp_path, [_design_row("design:0001", "ffn", ["FFN", "max", "xW1"])])

    payload = build_tex_equation_line_local_anchor_audit(
        normalizer_design_report=report_path,
        parsed_root=parsed,
    )

    assert payload["policy"]["reportOnly"] is True
    assert payload["policy"]["lineLocalAnchorAuditOnly"] is True
    assert payload["policy"]["strictEvidenceCreated"] is False
    assert payload["policy"]["parserRoutingChanged"] is False
    assert payload["policy"]["databaseMutation"] is False
    row = payload["rows"][0]
    assert row["chars_start"] is None
    assert row["chars_end"] is None
    assert row["page"] is None
    assert row["source_span_created"] is False
    assert row["equation_semantics_interpreted"] is False
    assert row["equation_region_verified"] is False
    assert row["strict_eligible"] is False
    assert row["runtime_evidence"] is False
    assert "line_local_windows_are_diagnostic_not_provenance" in row["strict_blockers"]


def test_tex_equation_line_local_anchor_audit_blocks_wrong_parent_schema(tmp_path: Path) -> None:
    parsed = _parsed_root(tmp_path)
    report_path = _design_report(
        tmp_path,
        [_design_row("design:0001", "ffn", ["FFN", "max", "xW1"])],
        schema="example.wrong",
    )

    payload = build_tex_equation_line_local_anchor_audit(
        normalizer_design_report=report_path,
        parsed_root=parsed,
    )

    assert validate_payload(payload, TEX_EQUATION_LINE_LOCAL_ANCHOR_AUDIT_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "blocked"
    assert payload["counts"]["normalizerDesignRows"] == 0
    assert payload["gate"]["schemaViolations"] == [
        "tex_equation_canonical_text_normalizer_design_schema_mismatch"
    ]


def test_tex_equation_line_local_anchor_audit_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    parsed = _parsed_root(tmp_path)
    report_path = _design_report(tmp_path, [_design_row("design:0001", "ffn", ["FFN", "max", "xW1"])])
    payload = build_tex_equation_line_local_anchor_audit(
        normalizer_design_report=report_path,
        parsed_root=parsed,
    )

    paths = write_tex_equation_line_local_anchor_audit_reports(payload, tmp_path / "reports")

    assert set(paths) == {"report", "summary", "markdown"}
    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(report, TEX_EQUATION_LINE_LOCAL_ANCHOR_AUDIT_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["sourceSpanCreatedRows"] == 0
    assert "not source spans or evidence" in markdown
