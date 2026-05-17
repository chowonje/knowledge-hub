from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.equation_alignment_feasibility_audit import (
    EQUATION_ALIGNMENT_FEASIBILITY_AUDIT_SCHEMA_ID,
    build_equation_alignment_feasibility_audit,
    write_equation_alignment_feasibility_audit_reports,
)


def _write(root: Path, name: str, payload: dict | str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _reports(root: Path, *, wrong_schema: bool = False) -> tuple[Path, Path]:
    doc_path = _write(
        root,
        "paper-1/document.md",
        "Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V\nThe feed forward network uses FFN and max.",
    )
    equation_schema = "knowledge-hub.paper.equation-quote-candidate-report.v1"
    source_schema = "knowledge-hub.paper.mineru-source-alignment-audit.v1"
    if wrong_schema:
        equation_schema = "example.wrong.equation"
        source_schema = "example.wrong.source"
    equation_path = _write(
        root,
        "equations.json",
        {
            "schema": equation_schema,
            "candidates": [
                {
                    "candidate_id": "equationquote:paper-1:0001",
                    "candidate_type": "equation_quote_candidate",
                    "paper_id": "paper-1",
                    "candidate_text": "Attention ( Q , K , V ) = softmax ( Q K ^ T / sqrt ( d_k ) ) V",
                    "canonical_alignment_status": "failed",
                    "alignment_method": "none",
                    "layout_element_ids": ["mineru:1"],
                    "bbox": [1.0, 2.0, 3.0, 4.0],
                    "sourceContentHash": "hash-source",
                    "strict_blockers": ["equation_alignment_missing"],
                },
                {
                    "candidate_id": "equationquote:paper-1:0002",
                    "candidate_type": "equation_quote_candidate",
                    "paper_id": "paper-1",
                    "candidate_text": "\\mathrm { F F N } ( x ) = \\operatorname* { m a x } ( 0 , x )",
                    "canonical_alignment_status": "failed",
                    "alignment_method": "none",
                    "layout_element_ids": ["mineru:2"],
                    "bbox": [1.0, 2.0, 3.0, 4.0],
                    "sourceContentHash": "hash-source",
                    "strict_blockers": ["equation_alignment_missing"],
                },
            ],
        },
    )
    source_path = _write(
        root,
        "source-alignment.json",
        {
            "schema": source_schema,
            "papers": [
                {
                    "paperId": "paper-1",
                    "input": {"pymupdfDocumentMarkdownPath": str(doc_path)},
                }
            ],
        },
    )
    return equation_path, source_path


def test_equation_alignment_feasibility_reports_diagnostic_matches_and_validates_schema(tmp_path: Path) -> None:
    equation_path, source_path = _reports(tmp_path)

    payload = build_equation_alignment_feasibility_audit(
        equation_quote_report=equation_path,
        mineru_source_alignment_report=source_path,
    )

    assert payload["schema"] == EQUATION_ALIGNMENT_FEASIBILITY_AUDIT_SCHEMA_ID
    assert validate_payload(payload, EQUATION_ALIGNMENT_FEASIBILITY_AUDIT_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "ok"
    assert payload["counts"]["auditedEquationQuoteCandidates"] == 2
    assert payload["counts"]["compactUniqueMatchCandidates"] == 1
    assert payload["counts"]["diagnosticTermContextCandidates"] == 1
    assert payload["counts"]["canonicalSourceSpanCreatedCandidates"] == 0
    assert payload["rows"][0]["feasibility_status"] == "compact_text_unique_match_candidate_only"
    assert payload["rows"][1]["feasibility_status"] == "diagnostic_term_context_candidate_only"


def test_equation_alignment_feasibility_never_creates_source_spans_or_interprets_equations(tmp_path: Path) -> None:
    equation_path, source_path = _reports(tmp_path)

    payload = build_equation_alignment_feasibility_audit(
        equation_quote_report=equation_path,
        mineru_source_alignment_report=source_path,
    )

    assert payload["policy"]["auditOnly"] is True
    assert payload["policy"]["quoteOnly"] is True
    assert payload["policy"]["equationSemanticsInterpreted"] is False
    assert payload["policy"]["strictEvidenceCreated"] is False
    assert payload["policy"]["runtimePromotionAllowed"] is False
    assert payload["policy"]["parserRoutingChanged"] is False
    assert payload["policy"]["canonicalParsedArtifactsWritten"] is False
    assert payload["policy"]["databaseMutation"] is False
    assert payload["policy"]["reindexOrReembed"] is False
    assert payload["policy"]["answerIntegrationChanged"] is False
    for row in payload["rows"]:
        assert row["source_span_created"] is False
        assert row["equation_semantics_interpreted"] is False
        assert row["strict_eligible"] is False
        assert row["evidence_tier"] == "equation_alignment_feasibility_candidate_only"
        assert "diagnostic_matches_do_not_create_source_spans" in row["non_strict_reason"]


def test_equation_alignment_feasibility_blocks_wrong_input_schema_ids(tmp_path: Path) -> None:
    equation_path, source_path = _reports(tmp_path, wrong_schema=True)

    payload = build_equation_alignment_feasibility_audit(
        equation_quote_report=equation_path,
        mineru_source_alignment_report=source_path,
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert set(payload["gate"]["schemaViolations"]) == {
        "equation_quote_candidate_report_schema_mismatch",
        "mineru_source_alignment_report_schema_mismatch",
    }


def test_equation_alignment_feasibility_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    equation_path, source_path = _reports(tmp_path / "input")
    payload = build_equation_alignment_feasibility_audit(
        equation_quote_report=equation_path,
        mineru_source_alignment_report=source_path,
    )

    paths = write_equation_alignment_feasibility_audit_reports(payload, tmp_path / "reports")

    assert set(paths) == {"audit", "markdown"}
    audit = json.loads(Path(paths["audit"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(audit, EQUATION_ALIGNMENT_FEASIBILITY_AUDIT_SCHEMA_ID, strict=True).ok
    assert "does not interpret equations" in markdown
    assert "Canonical source spans created" in markdown
