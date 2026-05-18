from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.tex_equation_quote_candidate_audit import (
    TEX_EQUATION_QUOTE_CANDIDATE_REPORT_SCHEMA_ID,
    build_tex_equation_quote_candidate_report,
    write_tex_equation_quote_candidate_reports,
)


def _write_json(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _row(
    candidate_id: str,
    *,
    paper_id: str = "1234.5678",
    structure_type: str = "equation_environment",
    text: str = "Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V",
    status: str = "aligned",
    method: str = "exact",
    page: int | None = 1,
    source_hash: str | None = "hash-source",
    source_ready: bool | None = None,
) -> dict:
    ready = status == "aligned" and page is not None and bool(source_hash) if source_ready is None else source_ready
    return {
        "candidate_id": candidate_id,
        "candidate_type": "tex_structure_candidate",
        "paper_id": paper_id,
        "source_parser": "arxiv_tex",
        "source_file": "main.tex",
        "structure_type": structure_type,
        "tex_command": "\\begin",
        "tex_environment": "equation" if structure_type == "equation_environment" else "",
        "candidate_text": text,
        "tex_locator": {"source_file": "main.tex", "chars": {"start": 0, "end": 10}},
        "alignment_status": status,
        "alignment_method": method,
        "alignment_reason": "test",
        "chars_start": 20 if status == "aligned" else None,
        "chars_end": 20 + len(text) if status == "aligned" else None,
        "page": page,
        "sourceContentHash": source_hash,
        "sourceContentHashSource": "manifest",
        "confidence": 0.99 if method == "exact" else 0.82,
        "source_span_locator": {"path": "document.md", "chars": {"start": 20, "end": 20 + len(text)}} if status == "aligned" else {},
        "source_span_candidate_ready": ready,
        "classification": "source_span_candidate_only" if text else "raw_tex_environment_only",
        "mineru_layout_link_status": "failed",
        "mineru_layout_link_method": "none",
        "mineru_candidate_ids": [],
        "mineru_bbox_link_count": 0,
        "evidence_tier": "source_structure_candidate_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": ["runtime_promotion_disabled_for_tranche"],
        "non_strict_reason": ["runtime_promotion_disabled_for_tranche"],
    }


def _alignment_report(root: Path) -> Path:
    return _write_json(
        root,
        "tex-structure-candidate-alignment-report.json",
        {
            "schema": "knowledge-hub.paper.tex-structure-candidate-alignment-audit.v1",
            "status": "ok",
            "candidates": [
                _row("tex:0001", text="Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V"),
                _row("tex:0002", text=""),
                _row("tex:0003", text="Normalized equation.", method="normalized"),
                _row("tex:0004", text="Repeated equation.", status="ambiguous"),
                _row("tex:0005", text="Failed equation.", status="failed"),
                _row("tex:0006", text="No page.", page=None),
                _row("tex:0007", text="No hash.", source_hash=""),
                _row("tex:0008", structure_type="section", text="Introduction"),
            ],
        },
    )


def test_tex_equation_quote_candidate_report_emits_exact_equation_candidates_only(tmp_path: Path) -> None:
    report_path = _alignment_report(tmp_path)

    payload = build_tex_equation_quote_candidate_report(report_path)

    assert payload["schema"] == TEX_EQUATION_QUOTE_CANDIDATE_REPORT_SCHEMA_ID
    assert validate_payload(payload, TEX_EQUATION_QUOTE_CANDIDATE_REPORT_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "ok"
    assert payload["counts"]["inputRows"] == 8
    assert payload["counts"]["equationEnvironmentRows"] == 7
    assert payload["counts"]["textBearingEquationEnvironmentRows"] == 6
    assert payload["counts"]["equationQuoteCandidates"] == 1
    assert payload["counts"]["heldOutCandidates"] == 6
    assert payload["counts"]["strictEligibleCandidates"] == 0
    assert payload["counts"]["runtimeEvidenceCandidates"] == 0
    assert payload["counts"]["equationRegionVerifiedCandidates"] == 0
    assert payload["counts"]["equationSemanticsInterpretedCandidates"] == 0
    assert payload["policy"]["answerIntegrationChanged"] is False
    assert payload["policy"]["equationInterpretationAllowed"] is False

    candidate = payload["candidates"][0]
    assert candidate["candidate_type"] == "equation_quote_candidate"
    assert candidate["source_parser"] == "arxiv_tex+pymupdf_alignment"
    assert candidate["alignment_method"] == "exact"
    assert candidate["evidence_tier"] == "equation_quote_candidate_only"
    assert candidate["equation_region_verified"] is False
    assert candidate["equation_semantics_interpreted"] is False
    assert candidate["strict_eligible"] is False
    assert candidate["runtime_evidence"] is False
    assert "equation_semantics_not_interpreted" in candidate["strict_blockers"]

    reasons = payload["counts"]["heldOutByReason"]
    assert reasons["empty_equation_text"] == 1
    assert reasons["non_exact_alignment"] == 1
    assert reasons["ambiguous_canonical_match"] == 1
    assert reasons["canonical_alignment_not_available"] == 1
    assert reasons["missing_page"] == 1
    assert reasons["missing_source_content_hash"] == 1


def test_tex_equation_quote_candidate_report_filters_paper_id(tmp_path: Path) -> None:
    report_path = _alignment_report(tmp_path)

    payload = build_tex_equation_quote_candidate_report(report_path, paper_ids=["missing-paper"])

    assert validate_payload(payload, TEX_EQUATION_QUOTE_CANDIDATE_REPORT_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "empty"
    assert payload["counts"]["inputRows"] == 0
    assert payload["counts"]["textBearingEquationEnvironmentRows"] == 0
    assert payload["counts"]["equationQuoteCandidates"] == 0
    assert payload["policy"]["runtimePromotionAllowed"] is False


def test_tex_equation_quote_candidate_report_fails_closed_on_parent_schema_mismatch(tmp_path: Path) -> None:
    report_path = _write_json(
        tmp_path,
        "wrong-report.json",
        {
            "schema": "wrong.schema",
            "status": "ok",
            "candidates": [_row("tex:0001", text="Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V")],
        },
    )

    payload = build_tex_equation_quote_candidate_report(report_path)

    assert validate_payload(payload, TEX_EQUATION_QUOTE_CANDIDATE_REPORT_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "empty"
    assert payload["counts"]["inputRows"] == 0
    assert payload["counts"]["equationQuoteCandidates"] == 0
    assert "alignment_report_schema_mismatch" in payload["warnings"]


def test_tex_equation_quote_candidate_report_requires_parent_source_span_ready(tmp_path: Path) -> None:
    report_path = _write_json(
        tmp_path,
        "tex-structure-candidate-alignment-report.json",
        {
            "schema": "knowledge-hub.paper.tex-structure-candidate-alignment-audit.v1",
            "status": "ok",
            "candidates": [
                _row(
                    "tex:0001",
                    text="Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V",
                    status="aligned",
                    method="exact",
                    page=1,
                    source_hash="hash-source",
                    source_ready=False,
                )
            ],
        },
    )

    payload = build_tex_equation_quote_candidate_report(report_path)

    assert validate_payload(payload, TEX_EQUATION_QUOTE_CANDIDATE_REPORT_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "empty"
    assert payload["counts"]["equationQuoteCandidates"] == 0
    assert payload["counts"]["heldOutByReason"]["source_span_candidate_not_ready"] == 1


def test_tex_equation_quote_candidate_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    report_path = _alignment_report(tmp_path)
    payload = build_tex_equation_quote_candidate_report(report_path)

    paths = write_tex_equation_quote_candidate_reports(payload, tmp_path / "reports")

    assert set(paths) == {"candidates", "summary", "markdown"}
    report = json.loads(Path(paths["candidates"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(report, TEX_EQUATION_QUOTE_CANDIDATE_REPORT_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["citationGradeCandidates"] == 0
    assert summary["counts"]["equationRegionVerifiedCandidates"] == 0
    assert "not strict evidence" in markdown
