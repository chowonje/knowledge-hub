from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.equation_authority_hash_identity_design import (
    EQUATION_AUTHORITY_HASH_IDENTITY_DESIGN_SCHEMA_ID,
    STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH,
    STATUS_BLOCKED_INPUT_REPORT_MISSING,
    STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION,
    STATUS_BLOCKED_MISSING_EQUATION_IDENTITY_SIGNAL,
    STATUS_BLOCKED_MISSING_SOURCE_HASH,
    STATUS_BLOCKED_MISSING_TEX_OR_MATHML_TEXT,
    STATUS_BLOCKED_RASTER_ONLY_EQUATION,
    STATUS_HASH_IDENTITY_DESIGN_CANDIDATE_ONLY,
    build_equation_authority_hash_identity_design,
    write_equation_authority_hash_identity_design_reports,
)
from knowledge_hub.papers.equation_authority_readiness_audit import (
    EQUATION_AUTHORITY_READINESS_AUDIT_SCHEMA_ID,
    STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH as READINESS_STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH,
    STATUS_BLOCKED_MISSING_EQUATION_IDENTITY as READINESS_STATUS_BLOCKED_MISSING_EQUATION_IDENTITY,
    STATUS_BLOCKED_MISSING_PDF_REGION as READINESS_STATUS_BLOCKED_MISSING_PDF_REGION,
    STATUS_BLOCKED_MISSING_SOURCE_HASH as READINESS_STATUS_BLOCKED_MISSING_SOURCE_HASH,
    STATUS_BLOCKED_MISSING_TEX_OR_MATHML_HASH as READINESS_STATUS_BLOCKED_MISSING_TEX_OR_MATHML_HASH,
    STATUS_BLOCKED_RASTER_ONLY_EQUATION as READINESS_STATUS_BLOCKED_RASTER_ONLY_EQUATION,
    STATUS_CANDIDATE_ONLY as READINESS_STATUS_CANDIDATE_ONLY,
)


def _pdf_region(*, available: bool = True) -> dict:
    return {
        "page": 1 if available else None,
        "bbox": [1.0, 2.0, 3.0, 4.0] if available else [],
        "blockIndexes": [7],
        "matchedTerms": ["x", "y"],
        "coverage": 1.0,
        "formulaScore": 1.4,
        "equationNumbers": ["1"],
        "textPreview": "x = y (1)",
        "available": available,
    }


def _readiness_row(
    key: str,
    *,
    readiness_status: str = READINESS_STATUS_CANDIDATE_ONLY,
    identity_available: bool = True,
    candidate_text: str = "x = y",
    tex_hash: str = "input-tex-hash",
    source_hash: str = "source-hash",
    pdf_available: bool = True,
    ambiguous: bool = False,
    raster: bool = False,
) -> dict:
    return {
        "audit_row_id": f"equation-authority-readiness-audit:paper-1:{key}",
        "candidate_type": "equation_authority_readiness_audit",
        "paper_id": "paper-1",
        "source_candidate_id": key if identity_available else "",
        "source_file": "main.tex",
        "equation_environment": "equation",
        "source_diagnostic_ids": {
            "canonicalAlignmentDiagnosticId": f"diag:{key}",
            "renderedMacroDesignId": f"rendered:{key}",
            "pdfRegionAnchorId": f"pdf:{key}",
            "labelNumberDisambiguationDesignId": f"label:{key}",
        },
        "equation_identity": {
            "equationId": "",
            "sourceCandidateId": key if identity_available else "",
            "sourceTexRowId": key if identity_available else "",
            "latexLabels": ["eq:test"] if identity_available else [],
            "equationNumbers": ["1"] if identity_available else [],
            "identityBasis": "source_tex_row_id" if identity_available else "missing",
            "available": identity_available,
        },
        "tex_mathml_hash_availability": {
            "texAvailable": bool(candidate_text),
            "mathmlAvailable": False,
            "equationTeXHash": tex_hash,
            "mathmlHash": "",
            "hashAvailable": bool(tex_hash),
            "candidateText": candidate_text,
            "renderedAliasText": candidate_text,
        },
        "pdf_region": _pdf_region(available=pdf_available),
        "canonical_text_alignment": {
            "status": "raw_tex_ambiguous_match_candidate_only" if ambiguous else "raw_tex_unique_match_candidate_only",
            "renderedProfileCanonicalStatus": "",
            "rawTexMatchCount": 2 if ambiguous else 1,
            "compactTexMatchCount": 0,
            "plainTextMatchCount": 0,
            "diagnosticTermCoverage": 1.0,
            "canonicalDocumentAvailable": True,
            "matched": True,
            "nonUnique": ambiguous,
        },
        "sourceContentHash": source_hash,
        "source_hash_available": bool(source_hash),
        "surrounding_context": {
            "canonicalDiagnosticTerms": ["x", "y"],
            "canonicalDiagnosticTermMatches": ["x", "y"],
            "renderedProfileTerms": ["x", "y"],
            "pdfTextPreview": "x = y (1)",
            "renderedAliasText": candidate_text,
        },
        "ambiguity": {
            "nonUniqueMatch": ambiguous,
            "reasons": ["canonical_alignment:raw_tex_ambiguous_match_candidate_only"] if ambiguous else [],
            "pdfRegionCandidateCount": 2 if ambiguous else 1,
            "labelNumberMatchingCandidateCount": 0,
        },
        "raster_image_only_blocker": raster,
        "readiness_status": readiness_status,
        "blockers": [readiness_status],
        "recommended_action": "test_action",
        "equationArtifactCreated": False,
        "strictEvidenceCreated": False,
        "sourceSpanMutated": False,
        "databaseMutation": False,
        "indexMutation": False,
        "reindexOrReembed": False,
        "vaultScan": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
    }


def _readiness_report(rows: list[dict], *, schema: str = EQUATION_AUTHORITY_READINESS_AUDIT_SCHEMA_ID) -> dict:
    by_status = Counter(row["readiness_status"] for row in rows)
    return {
        "schema": schema,
        "status": "ok",
        "generatedAt": "2026-05-20T00:00:00Z",
        "input": {
            "paperIds": [],
            "renderedMacroTermProfileDesignReportPath": "rendered.json",
            "renderedMacroTermProfileDesignReportSchema": "knowledge-hub.paper.tex-equation-rendered-macro-term-profile-design.v1",
            "renderedMacroTermProfileDesignReportExists": True,
            "canonicalAlignmentDiagnosticReportPath": "canonical.json",
            "canonicalAlignmentDiagnosticReportSchema": "knowledge-hub.paper.tex-equation-canonical-alignment-diagnostic-audit.v1",
            "canonicalAlignmentDiagnosticReportExists": True,
            "pdfRegionAnchorAuditReportPath": "pdf.json",
            "pdfRegionAnchorAuditReportSchema": "knowledge-hub.paper.tex-equation-pdf-region-anchor-audit.v1",
            "pdfRegionAnchorAuditReportExists": True,
            "labelNumberPdfRegionDisambiguationDesignReportPath": "label.json",
            "labelNumberPdfRegionDisambiguationDesignReportSchema": "knowledge-hub.paper.tex-equation-label-number-pdf-region-disambiguation-design.v1",
            "labelNumberPdfRegionDisambiguationDesignReportExists": True,
        },
        "counts": {
            "inputRows": len(rows),
            "targetRows": len(rows),
            "equationAuthorityCandidateOnlyRows": by_status[READINESS_STATUS_CANDIDATE_ONLY],
            "blockedMissingEquationIdentityRows": by_status[READINESS_STATUS_BLOCKED_MISSING_EQUATION_IDENTITY],
            "blockedMissingTexOrMathmlHashRows": by_status[READINESS_STATUS_BLOCKED_MISSING_TEX_OR_MATHML_HASH],
            "blockedMissingPdfRegionRows": by_status[READINESS_STATUS_BLOCKED_MISSING_PDF_REGION],
            "blockedMissingSourceHashRows": by_status[READINESS_STATUS_BLOCKED_MISSING_SOURCE_HASH],
            "blockedAmbiguousEquationMatchRows": by_status[READINESS_STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH],
            "blockedRasterOnlyEquationRows": by_status[READINESS_STATUS_BLOCKED_RASTER_ONLY_EQUATION],
            "blockedInputReportMissingRows": 0,
            "blockedInputSchemaViolationRows": 0,
            "equationArtifactCreatedRows": 0,
            "strictEvidenceCreatedRows": 0,
            "sourceSpanMutatedRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "vaultScanRows": 0,
            "parserRoutingChangedRows": 0,
            "answerIntegrationChangedRows": 0,
            "inputBlockerCount": 0,
            "byReadinessStatus": dict(by_status),
            "byPaper": {"paper-1": len(rows)},
            "byEnvironment": {"equation": len(rows)},
        },
        "gate": {
            "readinessAuditRows": bool(rows),
            "equationAuthorityPromotionReady": False,
            "equationArtifactCreationReady": False,
            "strictEvidenceReady": False,
            "sourceSpanMutationReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "equation_authority_readiness_audit_ready",
            "recommendedNextTranche": "equation_authority_hash_and_identity_design",
            "inputBlockers": [],
        },
        "policy": {
            "reportOnly": True,
            "classificationOnly": True,
            "equationArtifactCreated": False,
            "strictEvidenceCreated": False,
            "sourceSpanMutation": False,
            "databaseMutation": False,
            "indexMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "parserRoutingChanged": False,
            "answerIntegrationChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "equationInterpretationAllowed": False,
        },
        "warnings": [],
        "inputBlockers": [],
        "rows": rows,
    }


def _write_readiness_report(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "equation-authority-readiness-audit-report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_equation_authority_hash_identity_design_classifies_rows(tmp_path: Path) -> None:
    readiness_payload = _readiness_report(
        [
            _readiness_row("candidate"),
            _readiness_row(
                "derive-hash",
                readiness_status=READINESS_STATUS_BLOCKED_MISSING_TEX_OR_MATHML_HASH,
                tex_hash="",
            ),
            _readiness_row(
                "missing-pdf",
                readiness_status=READINESS_STATUS_BLOCKED_MISSING_PDF_REGION,
                pdf_available=False,
            ),
            _readiness_row(
                "missing-identity",
                readiness_status=READINESS_STATUS_BLOCKED_MISSING_EQUATION_IDENTITY,
                identity_available=False,
            ),
            _readiness_row(
                "missing-text",
                readiness_status=READINESS_STATUS_BLOCKED_MISSING_TEX_OR_MATHML_HASH,
                candidate_text="",
                tex_hash="",
            ),
            _readiness_row(
                "missing-source",
                readiness_status=READINESS_STATUS_BLOCKED_MISSING_SOURCE_HASH,
                source_hash="",
            ),
            _readiness_row(
                "ambiguous",
                readiness_status=READINESS_STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH,
                ambiguous=True,
            ),
            _readiness_row(
                "raster",
                readiness_status=READINESS_STATUS_BLOCKED_RASTER_ONLY_EQUATION,
                raster=True,
            ),
        ]
    )
    assert validate_payload(readiness_payload, EQUATION_AUTHORITY_READINESS_AUDIT_SCHEMA_ID, strict=True).ok

    payload = build_equation_authority_hash_identity_design(_write_readiness_report(tmp_path, readiness_payload))

    assert payload["schema"] == EQUATION_AUTHORITY_HASH_IDENTITY_DESIGN_SCHEMA_ID
    assert validate_payload(payload, EQUATION_AUTHORITY_HASH_IDENTITY_DESIGN_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "ok"
    counts = payload["counts"]
    assert counts["targetRows"] == 8
    assert counts["hashIdentityDesignCandidateOnlyRows"] == 3
    assert counts["blockedMissingEquationIdentitySignalRows"] == 1
    assert counts["blockedMissingTexOrMathmlTextRows"] == 1
    assert counts["blockedMissingSourceHashRows"] == 1
    assert counts["blockedAmbiguousEquationMatchRows"] == 1
    assert counts["blockedRasterOnlyEquationRows"] == 1
    assert counts["equationArtifactCreatedRows"] == 0
    assert counts["strictEvidenceCreatedRows"] == 0
    assert counts["sourceSpanMutatedRows"] == 0
    assert counts["authorityPolicyCreatedRows"] == 0
    assert {row["design_status"] for row in payload["rows"]} == {
        STATUS_HASH_IDENTITY_DESIGN_CANDIDATE_ONLY,
        STATUS_BLOCKED_MISSING_EQUATION_IDENTITY_SIGNAL,
        STATUS_BLOCKED_MISSING_TEX_OR_MATHML_TEXT,
        STATUS_BLOCKED_MISSING_SOURCE_HASH,
        STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH,
        STATUS_BLOCKED_RASTER_ONLY_EQUATION,
    }
    derive_hash = next(row for row in payload["rows"] if row["source_candidate_id"] == "derive-hash")
    assert derive_hash["design_status"] == STATUS_HASH_IDENTITY_DESIGN_CANDIDATE_ONLY
    assert derive_hash["proposed_hash_design"]["hashSource"] == "normalized_candidate_text_sha256"
    missing_pdf = next(row for row in payload["rows"] if row["source_candidate_id"] == "missing-pdf")
    assert missing_pdf["design_status"] == STATUS_HASH_IDENTITY_DESIGN_CANDIDATE_ONLY
    assert "input_readiness_status=blocked_missing_pdf_region" in missing_pdf["downstream_strict_evidence_blockers"]
    for row in payload["rows"]:
        assert row["equationArtifactCreated"] is False
        assert row["strictEvidenceCreated"] is False
        assert row["sourceSpanMutated"] is False
        assert row["databaseMutation"] is False
        assert row["vaultScan"] is False
        assert row["parserRoutingChanged"] is False
        assert row["answerIntegrationChanged"] is False
        assert row["authorityPolicyCreated"] is False


def test_equation_authority_hash_identity_design_blocks_missing_input_report(tmp_path: Path) -> None:
    payload = build_equation_authority_hash_identity_design(tmp_path / "missing-readiness-report.json")

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == STATUS_BLOCKED_INPUT_REPORT_MISSING
    assert payload["counts"]["blockedInputReportMissingRows"] == 1
    assert payload["rows"] == []
    assert payload["inputBlockers"][0]["design_status"] == STATUS_BLOCKED_INPUT_REPORT_MISSING
    assert validate_payload(payload, EQUATION_AUTHORITY_HASH_IDENTITY_DESIGN_SCHEMA_ID, strict=True).ok


def test_equation_authority_hash_identity_design_blocks_input_schema_violation(tmp_path: Path) -> None:
    payload = build_equation_authority_hash_identity_design(
        _write_readiness_report(tmp_path, _readiness_report([_readiness_row("candidate")], schema="wrong.schema"))
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION
    assert payload["counts"]["blockedInputSchemaViolationRows"] == 1
    assert payload["rows"] == []
    assert "schema mismatch" in payload["inputBlockers"][0]["detail"]
    assert validate_payload(payload, EQUATION_AUTHORITY_HASH_IDENTITY_DESIGN_SCHEMA_ID, strict=True).ok


def test_equation_authority_hash_identity_design_writer_outputs_valid_report(tmp_path: Path) -> None:
    readiness_payload = _readiness_report([_readiness_row("candidate")])
    payload = build_equation_authority_hash_identity_design(_write_readiness_report(tmp_path, readiness_payload))
    paths = write_equation_authority_hash_identity_design_reports(payload, tmp_path / "reports")

    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert set(paths) == {"report", "summary", "markdown"}
    assert validate_payload(report, EQUATION_AUTHORITY_HASH_IDENTITY_DESIGN_SCHEMA_ID, strict=True).ok
    assert summary["status"] == "ok"
    assert summary["counts"]["targetRows"] == 1
    assert "Equation Authority Hash/Identity Design" in markdown
