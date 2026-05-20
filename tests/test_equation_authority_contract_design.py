from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.equation_authority_contract_design import (
    EQUATION_AUTHORITY_CONTRACT_DESIGN_SCHEMA_ID,
    FUTURE_EQUATION_AUTHORITY_RECORD_SCHEMA_ID,
    STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH,
    STATUS_BLOCKED_INPUT_REPORT_MISSING,
    STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION,
    STATUS_BLOCKED_MISSING_DESIGN_HASH,
    STATUS_BLOCKED_MISSING_DESIGN_IDENTITY,
    STATUS_BLOCKED_MISSING_SOURCE_HASH,
    STATUS_BLOCKED_RASTER_ONLY_EQUATION,
    STATUS_EQUATION_AUTHORITY_CONTRACT_CANDIDATE_ONLY,
    build_equation_authority_contract_design,
    write_equation_authority_contract_design_reports,
)
from knowledge_hub.papers.equation_authority_hash_identity_design import (
    EQUATION_AUTHORITY_HASH_IDENTITY_DESIGN_SCHEMA_ID,
    STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH as HASH_STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH,
    STATUS_BLOCKED_MISSING_EQUATION_IDENTITY_SIGNAL as HASH_STATUS_BLOCKED_MISSING_EQUATION_IDENTITY_SIGNAL,
    STATUS_BLOCKED_MISSING_SOURCE_HASH as HASH_STATUS_BLOCKED_MISSING_SOURCE_HASH,
    STATUS_BLOCKED_MISSING_TEX_OR_MATHML_TEXT as HASH_STATUS_BLOCKED_MISSING_TEX_OR_MATHML_TEXT,
    STATUS_BLOCKED_RASTER_ONLY_EQUATION as HASH_STATUS_BLOCKED_RASTER_ONLY_EQUATION,
    STATUS_HASH_IDENTITY_DESIGN_CANDIDATE_ONLY,
)


def _hash_row(
    key: str,
    *,
    design_status: str = STATUS_HASH_IDENTITY_DESIGN_CANDIDATE_ONLY,
    identity_available: bool = True,
    identity_digest: str = "identity-digest",
    selected_hash: str = "equation-hash",
    source_hash: str = "source-hash",
    ambiguous: bool = False,
    raster: bool = False,
) -> dict:
    return {
        "design_row_id": f"equation-authority-hash-identity-design:paper-1:{key}",
        "candidate_type": "equation_authority_hash_identity_design",
        "readiness_audit_row_id": f"equation-authority-readiness-audit:paper-1:{key}",
        "paper_id": "paper-1",
        "source_candidate_id": key if identity_available else "",
        "source_file": "main.tex",
        "equation_environment": "equation",
        "readiness_status": "blocked_missing_pdf_region"
        if design_status == STATUS_HASH_IDENTITY_DESIGN_CANDIDATE_ONLY
        else design_status,
        "equation_identity": {
            "equationId": "",
            "sourceCandidateId": key if identity_available else "",
            "sourceTexRowId": key if identity_available else "",
            "latexLabels": ["eq:test"] if identity_available else [],
            "equationNumbers": ["1"] if identity_available else [],
            "identityBasis": "source_tex_row_id" if identity_available else "missing",
            "available": identity_available,
        },
        "tex_mathml_hash_signal": {
            "candidateText": "x = y" if selected_hash else "",
            "normalizedCandidateText": "x = y" if selected_hash else "",
            "normalizedCandidateTextSha256": selected_hash,
            "equationTeXHash": "",
            "mathmlHash": "",
            "texAvailable": bool(selected_hash),
            "mathmlAvailable": False,
            "hashAvailable": False,
            "designHashSource": "normalized_candidate_text_sha256" if selected_hash else "none",
            "selectedDesignHash": selected_hash,
            "designHashAvailable": bool(selected_hash),
        },
        "pdf_region_signal": {"available": True, "page": 1, "bbox": [1.0, 2.0, 3.0, 4.0]},
        "canonical_text_alignment": {
            "status": "raw_tex_ambiguous_match_candidate_only" if ambiguous else "raw_tex_unique_match_candidate_only"
        },
        "sourceContentHash": source_hash,
        "source_hash_available": bool(source_hash),
        "surrounding_context": {"pdfTextPreview": "x = y (1)"},
        "ambiguity": {
            "nonUniqueMatch": ambiguous,
            "reasons": ["canonical_alignment:raw_tex_ambiguous_match_candidate_only"] if ambiguous else [],
            "pdfRegionCandidateCount": 2 if ambiguous else 1,
            "labelNumberMatchingCandidateCount": 0,
        },
        "raster_image_only_blocker": raster,
        "proposed_identity_design": {
            "designOnly": True,
            "identityDigestAlgorithm": "equation_authority_identity_design_digest_v1",
            "identityComponents": {
                "paperId": "paper-1",
                "sourceContentHash": source_hash,
                "equationId": "",
                "sourceCandidateId": key if identity_available else "",
                "sourceTexRowId": key if identity_available else "",
                "latexLabels": ["eq:test"] if identity_available else [],
                "equationNumbers": ["1"] if identity_available else [],
                "identityBasis": "source_tex_row_id" if identity_available else "missing",
                "selectedDesignHash": selected_hash,
                "readinessAuditRowId": f"equation-authority-readiness-audit:paper-1:{key}",
            },
            "identityDigestSha256": identity_digest,
            "proposedEquationAuthorityDesignId": f"eqauth-design:paper-1:{key}",
            "equationIdentityPromoted": False,
            "promotionAllowed": False,
        },
        "proposed_hash_design": {
            "designOnly": True,
            "textNormalization": "equation_text_nfkc_whitespace_v1",
            "hashSource": "normalized_candidate_text_sha256" if selected_hash else "none",
            "normalizedCandidateTextSha256": selected_hash,
            "inputEquationTeXHash": "",
            "inputMathMLHash": "",
            "selectedDesignHash": selected_hash,
            "equationHashPromoted": False,
            "promotionAllowed": False,
        },
        "downstream_strict_evidence_blockers": [design_status],
        "design_status": design_status,
        "blockers": [design_status],
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
        "authorityPolicyCreated": False,
    }


def _hash_identity_report(rows: list[dict], *, schema: str = EQUATION_AUTHORITY_HASH_IDENTITY_DESIGN_SCHEMA_ID) -> dict:
    by_status = Counter(row["design_status"] for row in rows)
    return {
        "schema": schema,
        "status": "ok",
        "generatedAt": "2026-05-20T00:00:00Z",
        "input": {
            "paperIds": [],
            "equationAuthorityReadinessAuditReportPath": "readiness.json",
            "equationAuthorityReadinessAuditReportSchema": "knowledge-hub.paper.equation-authority-readiness-audit.v1",
            "equationAuthorityReadinessAuditReportExists": True,
            "equationAuthorityReadinessAuditReportStrictSchemaValid": True,
        },
        "counts": {
            "inputRows": len(rows),
            "targetRows": len(rows),
            "hashIdentityDesignCandidateOnlyRows": by_status[STATUS_HASH_IDENTITY_DESIGN_CANDIDATE_ONLY],
            "blockedMissingEquationIdentitySignalRows": by_status[
                HASH_STATUS_BLOCKED_MISSING_EQUATION_IDENTITY_SIGNAL
            ],
            "blockedMissingTexOrMathmlTextRows": by_status[HASH_STATUS_BLOCKED_MISSING_TEX_OR_MATHML_TEXT],
            "blockedMissingSourceHashRows": by_status[HASH_STATUS_BLOCKED_MISSING_SOURCE_HASH],
            "blockedAmbiguousEquationMatchRows": by_status[HASH_STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH],
            "blockedRasterOnlyEquationRows": by_status[HASH_STATUS_BLOCKED_RASTER_ONLY_EQUATION],
            "heldOutNonHashIdentityBlockerRows": 0,
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
            "authorityPolicyCreatedRows": 0,
            "inputBlockerCount": 0,
            "byDesignStatus": dict(by_status),
            "byReadinessStatus": {"blocked_missing_pdf_region": len(rows)},
            "byPaper": {"paper-1": len(rows)},
            "byEnvironment": {"equation": len(rows)},
        },
        "gate": {
            "hashIdentityDesignRows": bool(rows),
            "equationAuthorityPromotionReady": False,
            "equationArtifactCreationReady": False,
            "strictEvidenceReady": False,
            "sourceSpanMutationReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "equation_authority_hash_identity_design_ready",
            "recommendedNextTranche": "equation_authority_contract_design",
            "inputBlockers": [],
        },
        "policy": {
            "reportOnly": True,
            "designOnly": True,
            "classificationOnly": True,
            "authorityPolicyCreated": False,
            "equationIdentityPromoted": False,
            "equationHashPromoted": False,
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


def _write_hash_report(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "equation-authority-hash-identity-design-report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_equation_authority_contract_design_classifies_rows(tmp_path: Path) -> None:
    hash_payload = _hash_identity_report(
        [
            _hash_row("candidate"),
            _hash_row(
                "missing-identity",
                design_status=HASH_STATUS_BLOCKED_MISSING_EQUATION_IDENTITY_SIGNAL,
                identity_available=False,
                identity_digest="",
            ),
            _hash_row(
                "missing-hash",
                design_status=HASH_STATUS_BLOCKED_MISSING_TEX_OR_MATHML_TEXT,
                selected_hash="",
            ),
            _hash_row(
                "missing-source",
                design_status=HASH_STATUS_BLOCKED_MISSING_SOURCE_HASH,
                source_hash="",
            ),
            _hash_row(
                "ambiguous",
                design_status=HASH_STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH,
                ambiguous=True,
            ),
            _hash_row(
                "raster",
                design_status=HASH_STATUS_BLOCKED_RASTER_ONLY_EQUATION,
                raster=True,
            ),
        ]
    )
    assert validate_payload(hash_payload, EQUATION_AUTHORITY_HASH_IDENTITY_DESIGN_SCHEMA_ID, strict=True).ok

    payload = build_equation_authority_contract_design(_write_hash_report(tmp_path, hash_payload))

    assert payload["schema"] == EQUATION_AUTHORITY_CONTRACT_DESIGN_SCHEMA_ID
    assert validate_payload(payload, EQUATION_AUTHORITY_CONTRACT_DESIGN_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "ok"
    counts = payload["counts"]
    assert counts["targetRows"] == 6
    assert counts["equationAuthorityContractCandidateOnlyRows"] == 1
    assert counts["blockedMissingDesignIdentityRows"] == 1
    assert counts["blockedMissingDesignHashRows"] == 1
    assert counts["blockedMissingSourceHashRows"] == 1
    assert counts["blockedAmbiguousEquationMatchRows"] == 1
    assert counts["blockedRasterOnlyEquationRows"] == 1
    assert counts["authorityPolicyCreatedRows"] == 0
    assert counts["equationAuthorityRecordWrittenRows"] == 0
    assert counts["equationArtifactCreatedRows"] == 0
    assert counts["strictEvidenceCreatedRows"] == 0
    assert {row["contract_status"] for row in payload["rows"]} == {
        STATUS_EQUATION_AUTHORITY_CONTRACT_CANDIDATE_ONLY,
        STATUS_BLOCKED_MISSING_DESIGN_IDENTITY,
        STATUS_BLOCKED_MISSING_DESIGN_HASH,
        STATUS_BLOCKED_MISSING_SOURCE_HASH,
        STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH,
        STATUS_BLOCKED_RASTER_ONLY_EQUATION,
    }
    candidate = next(row for row in payload["rows"] if row["source_candidate_id"] == "candidate")
    assert candidate["contract_record_preview"]["schema"] == FUTURE_EQUATION_AUTHORITY_RECORD_SCHEMA_ID
    assert candidate["contract_record_preview"]["equationAuthorityRecordId"].startswith("eqauth-contract:")
    assert candidate["planned_record_contract"]["plannedWriteTarget"] == "equation_authority_record_candidate_store"
    for row in payload["rows"]:
        assert row["authorityPolicyCreated"] is False
        assert row["equationAuthorityRecordWritten"] is False
        assert row["equationArtifactCreated"] is False
        assert row["strictEvidenceCreated"] is False
        assert row["sourceSpanMutated"] is False
        assert row["databaseMutation"] is False
        assert row["vaultScan"] is False
        assert row["parserRoutingChanged"] is False
        assert row["answerIntegrationChanged"] is False


def test_equation_authority_contract_design_blocks_missing_input_report(tmp_path: Path) -> None:
    payload = build_equation_authority_contract_design(tmp_path / "missing-hash-identity-report.json")

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == STATUS_BLOCKED_INPUT_REPORT_MISSING
    assert payload["counts"]["blockedInputReportMissingRows"] == 1
    assert payload["rows"] == []
    assert payload["inputBlockers"][0]["contract_status"] == STATUS_BLOCKED_INPUT_REPORT_MISSING
    assert validate_payload(payload, EQUATION_AUTHORITY_CONTRACT_DESIGN_SCHEMA_ID, strict=True).ok


def test_equation_authority_contract_design_blocks_input_schema_violation(tmp_path: Path) -> None:
    payload = build_equation_authority_contract_design(
        _write_hash_report(tmp_path, _hash_identity_report([_hash_row("candidate")], schema="wrong.schema"))
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION
    assert payload["counts"]["blockedInputSchemaViolationRows"] == 1
    assert payload["rows"] == []
    assert "schema mismatch" in payload["inputBlockers"][0]["detail"]
    assert validate_payload(payload, EQUATION_AUTHORITY_CONTRACT_DESIGN_SCHEMA_ID, strict=True).ok


def test_equation_authority_contract_design_writer_outputs_valid_report(tmp_path: Path) -> None:
    hash_payload = _hash_identity_report([_hash_row("candidate")])
    payload = build_equation_authority_contract_design(_write_hash_report(tmp_path, hash_payload))
    paths = write_equation_authority_contract_design_reports(payload, tmp_path / "reports")

    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert set(paths) == {"report", "summary", "markdown"}
    assert validate_payload(report, EQUATION_AUTHORITY_CONTRACT_DESIGN_SCHEMA_ID, strict=True).ok
    assert summary["status"] == "ok"
    assert summary["counts"]["targetRows"] == 1
    assert "Equation Authority Contract Design" in markdown
