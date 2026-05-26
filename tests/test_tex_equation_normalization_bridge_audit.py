from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.tex_equation_canonical_alignment_diagnostic_audit import (
    TEX_EQUATION_CANONICAL_ALIGNMENT_DIAGNOSTIC_AUDIT_SCHEMA_ID,
)
from knowledge_hub.papers.tex_equation_normalization_bridge_audit import (
    TEX_EQUATION_NORMALIZATION_BRIDGE_AUDIT_SCHEMA_ID,
    build_tex_equation_normalization_bridge_audit,
    write_tex_equation_normalization_bridge_audit_reports,
)


def _write_json(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _diagnostic_row(diagnostic_id: str, text: str, *, paper_id: str = "paper-1") -> dict:
    return {
        "diagnostic_id": diagnostic_id,
        "source_candidate_id": diagnostic_id.replace("diagnostic", "source"),
        "paper_id": paper_id,
        "candidate_type": "tex_equation_canonical_alignment_diagnostic",
        "source_parser": "arxiv_tex+pymupdf_alignment",
        "source_file": "main.tex",
        "equation_environment": "equation",
        "candidate_text": text,
        "plain_text_candidate": text,
        "canonical_document_path": "",
        "canonical_document_available": True,
        "existing_alignment_status": "blocked",
        "existing_alignment_method": "none",
        "existing_alignment_reason": "structure_type_has_no_text_span",
        "tex_has_macros": "\\" in text,
        "raw_tex_match_count": 0,
        "compact_tex_match_count": 0,
        "plain_text_match_count": 0,
        "diagnostic_terms": [],
        "diagnostic_term_matches": [],
        "diagnostic_term_coverage": 0.0,
        "diagnosis": "tex_to_canonical_normalization_gap_candidate_only",
        "confidence": 0.62,
        "sourceContentHash": "hash-source",
        "source_span_created": False,
        "equation_semantics_interpreted": False,
        "equation_region_verified": False,
        "evidence_tier": "tex_equation_alignment_diagnostic_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": ["diagnostic_matches_do_not_create_source_spans"],
        "non_strict_reason": ["diagnostic_rows_are_not_evidence"],
    }


def _diagnostic_report(root: Path, rows: list[dict], *, schema: str | None = None) -> Path:
    return _write_json(
        root,
        "diagnostic.json",
        {
            "schema": schema or TEX_EQUATION_CANONICAL_ALIGNMENT_DIAGNOSTIC_AUDIT_SCHEMA_ID,
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


def _parsed_root(root: Path) -> Path:
    parsed = root / "parsed"
    (parsed / "paper-1").mkdir(parents=True)
    (parsed / "paper-1" / "document.md").write_text(
        "The rendered form says Attention Q K V softmax QK T sqrt d k V in one equation.\n"
        "The first repeated bridge has Alpha Beta Gamma.\n"
        "The second repeated bridge has Alpha Beta Gamma.\n",
        encoding="utf-8",
    )
    return parsed


def test_tex_equation_normalization_bridge_finds_unique_and_ambiguous_windows(tmp_path: Path) -> None:
    parsed = _parsed_root(tmp_path)
    report_path = _diagnostic_report(
        tmp_path,
        [
            _diagnostic_row(
                "diagnostic:0001",
                r"\mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V",
            ),
            _diagnostic_row("diagnostic:0002", "Alpha Beta Gamma"),
            _diagnostic_row("diagnostic:0003", "Missing Token Pair"),
            _diagnostic_row("diagnostic:0004", ""),
        ],
    )

    payload = build_tex_equation_normalization_bridge_audit(
        diagnostic_report=report_path,
        parsed_root=parsed,
    )

    assert payload["schema"] == TEX_EQUATION_NORMALIZATION_BRIDGE_AUDIT_SCHEMA_ID
    assert validate_payload(payload, TEX_EQUATION_NORMALIZATION_BRIDGE_AUDIT_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "ok"
    assert payload["counts"]["equationDiagnosticRows"] == 4
    assert payload["counts"]["textBearingEquationRows"] == 3
    assert payload["counts"]["bridgeWindowRows"] == 2
    assert payload["counts"]["uniqueBridgeWindowRows"] == 1
    assert payload["counts"]["ambiguousBridgeWindowRows"] == 1
    assert payload["counts"]["failedBridgeWindowRows"] == 1
    assert payload["counts"]["strictEligibleRows"] == 0

    statuses = {row["source_diagnostic_id"]: row["bridge_status"] for row in payload["rows"]}
    assert statuses["diagnostic:0001"] == "unique_ordered_token_window_candidate_only"
    assert statuses["diagnostic:0002"] == "ambiguous_ordered_token_window_candidate_only"
    assert statuses["diagnostic:0003"] == "ordered_token_window_not_found"
    assert statuses["diagnostic:0004"] == "empty_equation_text"


def test_tex_equation_normalization_bridge_never_creates_source_span_or_runtime_evidence(tmp_path: Path) -> None:
    parsed = _parsed_root(tmp_path)
    report_path = _diagnostic_report(tmp_path, [_diagnostic_row("diagnostic:0001", "Alpha Beta Gamma")])

    payload = build_tex_equation_normalization_bridge_audit(
        diagnostic_report=report_path,
        parsed_root=parsed,
    )

    assert payload["policy"]["reportOnly"] is True
    assert payload["policy"]["bridgeAuditOnly"] is True
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
    assert "ordered_token_windows_are_diagnostic_not_provenance" in row["strict_blockers"]


def test_tex_equation_normalization_bridge_blocks_wrong_parent_schema(tmp_path: Path) -> None:
    parsed = _parsed_root(tmp_path)
    report_path = _diagnostic_report(
        tmp_path,
        [_diagnostic_row("diagnostic:0001", "Alpha Beta Gamma")],
        schema="example.wrong",
    )

    payload = build_tex_equation_normalization_bridge_audit(
        diagnostic_report=report_path,
        parsed_root=parsed,
    )

    assert validate_payload(payload, TEX_EQUATION_NORMALIZATION_BRIDGE_AUDIT_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "blocked"
    assert payload["counts"]["equationDiagnosticRows"] == 0
    assert payload["gate"]["schemaViolations"] == [
        "tex_equation_canonical_alignment_diagnostic_report_schema_mismatch"
    ]


def test_tex_equation_normalization_bridge_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    parsed = _parsed_root(tmp_path)
    report_path = _diagnostic_report(tmp_path, [_diagnostic_row("diagnostic:0001", "Alpha Beta Gamma")])
    payload = build_tex_equation_normalization_bridge_audit(
        diagnostic_report=report_path,
        parsed_root=parsed,
    )

    paths = write_tex_equation_normalization_bridge_audit_reports(payload, tmp_path / "reports")

    assert set(paths) == {"report", "summary", "markdown"}
    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(report, TEX_EQUATION_NORMALIZATION_BRIDGE_AUDIT_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["sourceSpanCreatedRows"] == 0
    assert "not source spans or evidence" in markdown
