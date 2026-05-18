from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.tex_equation_canonical_alignment_diagnostic_audit import (
    TEX_EQUATION_CANONICAL_ALIGNMENT_DIAGNOSTIC_AUDIT_SCHEMA_ID,
    build_tex_equation_canonical_alignment_diagnostic_audit,
    write_tex_equation_canonical_alignment_diagnostic_audit_reports,
)


def _write_json(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _row(candidate_id: str, paper_id: str, text: str, *, structure_type: str = "equation_environment") -> dict:
    return {
        "candidate_id": candidate_id,
        "paper_id": paper_id,
        "structure_type": structure_type,
        "source_file": "main.tex",
        "tex_environment": "equation",
        "candidate_text": text,
        "alignment_status": "blocked",
        "alignment_method": "none",
        "alignment_reason": "structure_type_has_no_text_span",
        "sourceContentHash": "hash-source",
        "strict_blockers": ["equation_text_or_semantics_not_citation_grade"],
    }


def _alignment_report(root: Path, rows: list[dict], *, schema: str | None = None) -> Path:
    return _write_json(
        root,
        "alignment.json",
        {
            "schema": schema or "knowledge-hub.paper.tex-structure-candidate-alignment-audit.v1",
            "candidates": rows,
        },
    )


def _parsed_root(root: Path) -> Path:
    parsed = root / "parsed"
    (parsed / "paper-1").mkdir(parents=True)
    (parsed / "paper-1" / "document.md").write_text(
        "ExactEquationText\n"
        "The compact text appears as AlphaBeta=Gamma in generated markdown.\n"
        "Attention and softmax are present in prose near the rendered formula.\n",
        encoding="utf-8",
    )
    return parsed


def test_tex_equation_alignment_diagnostic_classifies_match_modes_and_validates_schema(tmp_path: Path) -> None:
    parsed = _parsed_root(tmp_path)
    report_path = _alignment_report(
        tmp_path,
        [
            _row("tex:eq:0001", "paper-1", "ExactEquationText"),
            _row("tex:eq:0002", "paper-1", "Alpha Beta = Gamma"),
            _row("tex:eq:0003", "paper-1", r"\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^T)"),
            _row("tex:eq:0004", "paper-1", "NotPresentAnywhere"),
            _row("tex:eq:0005", "paper-1", ""),
            _row("tex:eq:0006", "paper-1", "Introduction", structure_type="section"),
        ],
    )

    payload = build_tex_equation_canonical_alignment_diagnostic_audit(
        alignment_report=report_path,
        parsed_root=parsed,
    )

    assert payload["schema"] == TEX_EQUATION_CANONICAL_ALIGNMENT_DIAGNOSTIC_AUDIT_SCHEMA_ID
    assert validate_payload(payload, TEX_EQUATION_CANONICAL_ALIGNMENT_DIAGNOSTIC_AUDIT_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "ok"
    assert payload["counts"]["equationEnvironmentRows"] == 5
    assert payload["counts"]["textBearingEquationEnvironmentRows"] == 4
    assert payload["counts"]["emptyEquationTextRows"] == 1
    assert payload["counts"]["rawTexMatchRows"] == 1
    assert payload["counts"]["compactTexMatchRows"] == 2
    assert payload["counts"]["normalizationGapRows"] == 1
    assert payload["counts"]["strictEligibleRows"] == 0

    diagnoses = payload["counts"]["byDiagnosis"]
    assert diagnoses["raw_tex_unique_match_candidate_only"] == 1
    assert diagnoses["compact_tex_unique_match_candidate_only"] == 1
    assert diagnoses["tex_to_canonical_normalization_gap_candidate_only"] == 1
    assert diagnoses["likely_canonical_equation_text_missing"] == 1
    assert diagnoses["empty_equation_text"] == 1


def test_tex_equation_alignment_diagnostic_never_creates_evidence_or_runtime_outputs(tmp_path: Path) -> None:
    parsed = _parsed_root(tmp_path)
    report_path = _alignment_report(tmp_path, [_row("tex:eq:0001", "paper-1", "ExactEquationText")])

    payload = build_tex_equation_canonical_alignment_diagnostic_audit(
        alignment_report=report_path,
        parsed_root=parsed,
    )

    assert payload["policy"]["reportOnly"] is True
    assert payload["policy"]["diagnosticOnly"] is True
    assert payload["policy"]["strictEvidenceCreated"] is False
    assert payload["policy"]["parserRoutingChanged"] is False
    assert payload["policy"]["databaseMutation"] is False
    assert payload["policy"]["answerIntegrationChanged"] is False
    row = payload["rows"][0]
    assert row["source_span_created"] is False
    assert row["equation_semantics_interpreted"] is False
    assert row["equation_region_verified"] is False
    assert row["strict_eligible"] is False
    assert row["runtime_evidence"] is False
    assert "diagnostic_matches_do_not_create_source_spans" in row["strict_blockers"]


def test_tex_equation_alignment_diagnostic_blocks_wrong_parent_schema(tmp_path: Path) -> None:
    parsed = _parsed_root(tmp_path)
    report_path = _alignment_report(
        tmp_path,
        [_row("tex:eq:0001", "paper-1", "ExactEquationText")],
        schema="example.wrong",
    )

    payload = build_tex_equation_canonical_alignment_diagnostic_audit(
        alignment_report=report_path,
        parsed_root=parsed,
    )

    assert validate_payload(payload, TEX_EQUATION_CANONICAL_ALIGNMENT_DIAGNOSTIC_AUDIT_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "blocked"
    assert payload["counts"]["equationEnvironmentRows"] == 0
    assert payload["gate"]["schemaViolations"] == ["tex_structure_candidate_alignment_report_schema_mismatch"]


def test_tex_equation_alignment_diagnostic_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    parsed = _parsed_root(tmp_path)
    report_path = _alignment_report(tmp_path, [_row("tex:eq:0001", "paper-1", "ExactEquationText")])
    payload = build_tex_equation_canonical_alignment_diagnostic_audit(
        alignment_report=report_path,
        parsed_root=parsed,
    )

    paths = write_tex_equation_canonical_alignment_diagnostic_audit_reports(payload, tmp_path / "reports")

    assert set(paths) == {"report", "summary", "markdown"}
    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(report, TEX_EQUATION_CANONICAL_ALIGNMENT_DIAGNOSTIC_AUDIT_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["sourceSpanCreatedRows"] == 0
    assert "does not interpret equations" in markdown
