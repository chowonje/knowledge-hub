from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.equation_authority_readiness_audit import (
    EQUATION_AUTHORITY_READINESS_AUDIT_SCHEMA_ID,
    STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH,
    STATUS_BLOCKED_INPUT_REPORT_MISSING,
    STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION,
    STATUS_BLOCKED_MISSING_EQUATION_IDENTITY,
    STATUS_BLOCKED_MISSING_PDF_REGION,
    STATUS_BLOCKED_MISSING_SOURCE_HASH,
    STATUS_BLOCKED_MISSING_TEX_OR_MATHML_HASH,
    STATUS_BLOCKED_RASTER_ONLY_EQUATION,
    STATUS_CANDIDATE_ONLY,
    build_equation_authority_readiness_audit,
    write_equation_authority_readiness_audit_reports,
)
from knowledge_hub.papers.tex_equation_canonical_alignment_diagnostic_audit import (
    TEX_EQUATION_CANONICAL_ALIGNMENT_DIAGNOSTIC_AUDIT_SCHEMA_ID,
)
from knowledge_hub.papers.tex_equation_label_number_pdf_region_disambiguation_design import (
    TEX_EQUATION_LABEL_NUMBER_PDF_REGION_DISAMBIGUATION_DESIGN_SCHEMA_ID,
)
from knowledge_hub.papers.tex_equation_pdf_region_anchor_audit import (
    TEX_EQUATION_PDF_REGION_ANCHOR_AUDIT_SCHEMA_ID,
)
from knowledge_hub.papers.tex_equation_rendered_macro_term_profile_design import (
    TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_SCHEMA_ID,
)


def _write_report(path: Path, schema: str, rows: list[dict]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": schema,
                "status": "ok",
                "generatedAt": "2026-05-20T00:00:00Z",
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )
    return path


def _selected_region(*, page: int | None = 1, bbox: list[float] | None = None) -> dict:
    return {
        "page": page,
        "bbox": bbox if bbox is not None else [1.0, 2.0, 3.0, 4.0],
        "blockIndexes": [7],
        "block_indexes": [7],
        "matchedTerms": ["x", "y"],
        "matched_terms": ["x", "y"],
        "coverage": 1.0,
        "formulaScore": 1.4,
        "formula_score": 1.4,
        "equationNumbers": ["1"],
        "equation_numbers": ["1"],
        "textPreview": "x = y (1)",
        "text_preview": "x = y (1)",
    }


def _rendered_row(
    source_id: str,
    *,
    source_hash: str = "hash-source",
    tex_hash: str = "",
    ambiguous: bool = False,
    raster: bool = False,
    include_identity: bool = True,
    page: int | None = 1,
    bbox: list[float] | None = None,
) -> dict:
    status = "ambiguous_rendered_macro_profile_candidate_only" if ambiguous else "unique_rendered_macro_profile_candidate_only"
    row = {
        "paper_id": "paper-1",
        "source_file": "main.tex",
        "equation_environment": "equation",
        "candidate_text": "x = y",
        "rendered_alias_text": "x = y",
        "latex_labels": ["eq:test"] if include_identity else [],
        "sourceContentHash": source_hash,
        "equationTeXHash": tex_hash,
        "recommended_profile": "rendered_macro_alias_terms_v1",
        "recommended_status": status,
        "profile_results": [
            {
                "profile_name": "rendered_macro_alias_terms_v1",
                "normalized_terms": ["x", "y"],
                "canonical_match_status": "unique_rendered_macro_canonical_window_candidate_only",
                "pdf_region_match_status": status.replace("profile", "pdf_region"),
                "pdf_region_candidate_count": 1 if not ambiguous else 2,
                "selected_pdf_region": _selected_region(page=page, bbox=bbox),
                "profile_status": status,
            }
        ],
        "strict_blockers": ["raster_only_equation"] if raster else [],
    }
    if include_identity:
        row["source_candidate_id"] = source_id
    return row


def _canonical_row(
    source_id: str,
    *,
    source_hash: str = "hash-source",
    ambiguous: bool = False,
    tex_hash: str = "",
) -> dict:
    return {
        "diagnostic_id": f"diag:{source_id}",
        "source_candidate_id": source_id,
        "paper_id": "paper-1",
        "source_file": "main.tex",
        "equation_environment": "equation",
        "candidate_text": "x = y",
        "raw_tex_match_count": 2 if ambiguous else 1,
        "compact_tex_match_count": 0,
        "plain_text_match_count": 0,
        "diagnostic_term_coverage": 1.0,
        "canonical_document_available": True,
        "diagnostic_terms": ["x", "y"],
        "diagnostic_term_matches": ["x", "y"],
        "diagnosis": "raw_tex_ambiguous_match_candidate_only" if ambiguous else "raw_tex_unique_match_candidate_only",
        "sourceContentHash": source_hash,
        "equationTeXHash": tex_hash,
    }


def _pdf_row(
    source_id: str,
    *,
    source_hash: str = "hash-source",
    tex_hash: str = "",
    page: int | None = 1,
    bbox: list[float] | None = None,
    ambiguous: bool = False,
) -> dict:
    return {
        "pdf_region_anchor_id": f"pdf:{source_id}",
        "source_candidate_id": source_id,
        "paper_id": "paper-1",
        "source_file": "main.tex",
        "equation_environment": "equation",
        "candidate_text": "x = y",
        "sourceContentHash": source_hash,
        "equationTeXHash": tex_hash,
        "pdf_region_anchor_status": "ambiguous_pdf_region_anchor_candidate_only" if ambiguous else "unique_pdf_region_anchor_candidate_only",
        "pdf_region_candidate_count": 2 if ambiguous else 1,
        "pdf_region_anchor_unique": not ambiguous,
        "selected_pdf_region": _selected_region(page=page, bbox=bbox),
    }


def _label_row(
    source_id: str,
    *,
    source_hash: str = "hash-source",
    tex_hash: str = "",
    page: int | None = 1,
    bbox: list[float] | None = None,
    ambiguous: bool = False,
) -> dict:
    return {
        "design_id": f"label:{source_id}",
        "source_candidate_id": source_id,
        "paper_id": "paper-1",
        "source_file": "main.tex",
        "equation_environment": "equation",
        "candidate_text": "x = y",
        "latex_labels": ["eq:test"],
        "sourceContentHash": source_hash,
        "equationTeXHash": tex_hash,
        "source_label_number_hint": {
            "sourceStructureRowId": source_id,
            "latexLabels": ["eq:test"],
            "inferredEquationNumbers": ["1"],
        },
        "disambiguation_status": "ambiguous_label_number_pdf_region_candidate_only"
        if ambiguous
        else "unique_label_number_pdf_region_candidate_only",
        "pdf_region_candidate_count": 2 if ambiguous else 1,
        "label_number_matching_candidate_count": 2 if ambiguous else 1,
        "selected_pdf_region": _selected_region(page=page, bbox=bbox),
    }


def _write_inputs(tmp_path: Path, *, wrong_schema: bool = False) -> dict[str, Path]:
    rows = {
        "candidate": {"hash": "hash-candidate", "tex_hash": "tex-hash-candidate"},
        "missing-tex-hash": {"hash": "hash-missing-tex", "tex_hash": ""},
        "missing-pdf": {"hash": "hash-missing-pdf", "tex_hash": "tex-hash-pdf"},
        "missing-source-hash": {"hash": "", "tex_hash": "tex-hash-source"},
        "ambiguous": {"hash": "hash-ambiguous", "tex_hash": "tex-hash-ambiguous", "ambiguous": True},
        "raster": {"hash": "hash-raster", "tex_hash": "tex-hash-raster", "raster": True},
    }
    rendered_rows = [
        _rendered_row(
            key,
            source_hash=value["hash"],
            tex_hash=value["tex_hash"],
            ambiguous=value.get("ambiguous", False),
            raster=value.get("raster", False),
            page=None if key == "missing-pdf" else 1,
            bbox=[] if key == "missing-pdf" else None,
        )
        for key, value in rows.items()
    ]
    rendered_rows.append(_rendered_row("missing-identity", include_identity=False, tex_hash="tex-hash-identity"))
    canonical_rows = [
        _canonical_row(key, source_hash=value["hash"], tex_hash=value["tex_hash"], ambiguous=value.get("ambiguous", False))
        for key, value in rows.items()
    ]
    pdf_rows = [
        _pdf_row(
            key,
            source_hash=value["hash"],
            tex_hash=value["tex_hash"],
            page=None if key == "missing-pdf" else 1,
            bbox=[] if key == "missing-pdf" else None,
            ambiguous=value.get("ambiguous", False),
        )
        for key, value in rows.items()
    ]
    label_rows = [
        _label_row(
            key,
            source_hash=value["hash"],
            tex_hash=value["tex_hash"],
            page=None if key == "missing-pdf" else 1,
            bbox=[] if key == "missing-pdf" else None,
            ambiguous=value.get("ambiguous", False),
        )
        for key, value in rows.items()
    ]
    return {
        "rendered": _write_report(
            tmp_path / "rendered.json",
            "wrong.schema" if wrong_schema else TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_SCHEMA_ID,
            rendered_rows,
        ),
        "canonical": _write_report(
            tmp_path / "canonical.json",
            TEX_EQUATION_CANONICAL_ALIGNMENT_DIAGNOSTIC_AUDIT_SCHEMA_ID,
            canonical_rows,
        ),
        "pdf": _write_report(
            tmp_path / "pdf.json",
            TEX_EQUATION_PDF_REGION_ANCHOR_AUDIT_SCHEMA_ID,
            pdf_rows,
        ),
        "label": _write_report(
            tmp_path / "label.json",
            TEX_EQUATION_LABEL_NUMBER_PDF_REGION_DISAMBIGUATION_DESIGN_SCHEMA_ID,
            label_rows,
        ),
    }


def _build(paths: dict[str, Path]) -> dict:
    return build_equation_authority_readiness_audit(
        rendered_macro_term_profile_design_report=paths["rendered"],
        canonical_alignment_diagnostic_report=paths["canonical"],
        pdf_region_anchor_audit_report=paths["pdf"],
        label_number_pdf_region_disambiguation_design_report=paths["label"],
    )


def test_equation_authority_readiness_audit_classifies_all_statuses(tmp_path: Path) -> None:
    payload = _build(_write_inputs(tmp_path))

    assert payload["schema"] == EQUATION_AUTHORITY_READINESS_AUDIT_SCHEMA_ID
    assert validate_payload(payload, EQUATION_AUTHORITY_READINESS_AUDIT_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "ok"
    counts = payload["counts"]
    assert counts["targetRows"] == 7
    assert counts["equationAuthorityCandidateOnlyRows"] == 1
    assert counts["blockedMissingEquationIdentityRows"] == 1
    assert counts["blockedMissingTexOrMathmlHashRows"] == 1
    assert counts["blockedMissingPdfRegionRows"] == 1
    assert counts["blockedMissingSourceHashRows"] == 1
    assert counts["blockedAmbiguousEquationMatchRows"] == 1
    assert counts["blockedRasterOnlyEquationRows"] == 1
    assert counts["equationArtifactCreatedRows"] == 0
    assert counts["strictEvidenceCreatedRows"] == 0
    assert counts["sourceSpanMutatedRows"] == 0
    assert {row["readiness_status"] for row in payload["rows"]} == {
        STATUS_CANDIDATE_ONLY,
        STATUS_BLOCKED_MISSING_EQUATION_IDENTITY,
        STATUS_BLOCKED_MISSING_TEX_OR_MATHML_HASH,
        STATUS_BLOCKED_MISSING_PDF_REGION,
        STATUS_BLOCKED_MISSING_SOURCE_HASH,
        STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH,
        STATUS_BLOCKED_RASTER_ONLY_EQUATION,
    }
    for row in payload["rows"]:
        assert row["equationArtifactCreated"] is False
        assert row["strictEvidenceCreated"] is False
        assert row["sourceSpanMutated"] is False
        assert row["databaseMutation"] is False
        assert row["vaultScan"] is False
        assert row["parserRoutingChanged"] is False
        assert row["answerIntegrationChanged"] is False


def test_equation_authority_readiness_audit_blocks_missing_input_report(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    paths["label"] = tmp_path / "missing-label.json"

    payload = _build(paths)

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == STATUS_BLOCKED_INPUT_REPORT_MISSING
    assert payload["counts"]["blockedInputReportMissingRows"] == 1
    assert payload["rows"] == []
    assert payload["inputBlockers"][0]["readiness_status"] == STATUS_BLOCKED_INPUT_REPORT_MISSING
    assert validate_payload(payload, EQUATION_AUTHORITY_READINESS_AUDIT_SCHEMA_ID, strict=True).ok


def test_equation_authority_readiness_audit_blocks_input_schema_violation(tmp_path: Path) -> None:
    payload = _build(_write_inputs(tmp_path, wrong_schema=True))

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION
    assert payload["counts"]["blockedInputSchemaViolationRows"] == 1
    assert payload["rows"] == []
    assert "schema mismatch" in payload["inputBlockers"][0]["detail"]
    assert validate_payload(payload, EQUATION_AUTHORITY_READINESS_AUDIT_SCHEMA_ID, strict=True).ok


def test_equation_authority_readiness_audit_writer_outputs_valid_report(tmp_path: Path) -> None:
    payload = _build(_write_inputs(tmp_path))
    paths = write_equation_authority_readiness_audit_reports(payload, tmp_path / "reports")

    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert set(paths) == {"report", "summary", "markdown"}
    assert validate_payload(report, EQUATION_AUTHORITY_READINESS_AUDIT_SCHEMA_ID, strict=True).ok
    assert summary["status"] == "ok"
    assert summary["counts"]["targetRows"] == 7
    assert "Equation Authority Readiness Audit" in markdown
