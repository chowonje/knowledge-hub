from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.equation_authority_contract_design import (
    build_equation_authority_contract_design,
    write_equation_authority_contract_design_reports,
)
from knowledge_hub.papers.equation_authority_record_contract import (
    EQUATION_AUTHORITY_RECORD_CONTRACT_SCHEMA_ID,
    EQUATION_AUTHORITY_RECORD_SCHEMA_ID,
    EQUATION_AUTHORITY_RECORD_STORE,
    EQUATION_AUTHORITY_RECORD_STORE_CONTRACT,
    KNOWN_WRITE_TARGET_CONTRACTS,
    STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH,
    STATUS_BLOCKED_INPUT_REPORT_MISSING,
    STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION,
    STATUS_BLOCKED_MISSING_DESIGN_HASH,
    STATUS_BLOCKED_MISSING_DESIGN_IDENTITY,
    STATUS_BLOCKED_MISSING_SOURCE_HASH,
    STATUS_BLOCKED_RASTER_ONLY_EQUATION,
    STATUS_EQUATION_AUTHORITY_RECORD_CONTRACT_CANDIDATE_ONLY,
    build_equation_authority_record_contract,
    build_sample_equation_authority_record_from_contract_design_row,
    validate_equation_authority_record_semantics,
    write_equation_authority_record_contract_reports,
)
from knowledge_hub.papers.equation_authority_hash_identity_design import (
    STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH as HASH_STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH,
    STATUS_BLOCKED_MISSING_EQUATION_IDENTITY_SIGNAL as HASH_STATUS_BLOCKED_MISSING_EQUATION_IDENTITY_SIGNAL,
    STATUS_BLOCKED_MISSING_SOURCE_HASH as HASH_STATUS_BLOCKED_MISSING_SOURCE_HASH,
    STATUS_BLOCKED_MISSING_TEX_OR_MATHML_TEXT as HASH_STATUS_BLOCKED_MISSING_TEX_OR_MATHML_TEXT,
    STATUS_BLOCKED_RASTER_ONLY_EQUATION as HASH_STATUS_BLOCKED_RASTER_ONLY_EQUATION,
    STATUS_HASH_IDENTITY_DESIGN_CANDIDATE_ONLY,
)
from tests.test_equation_authority_contract_design import (
    _hash_identity_report,
    _hash_row,
    _write_hash_report,
)


def _contract_design_report_path(tmp_path: Path, rows: list[dict], *, schema: str | None = None) -> Path:
    hash_payload = _hash_identity_report(rows)
    hash_path = _write_hash_report(tmp_path, hash_payload)
    payload = build_equation_authority_contract_design(hash_path)
    if schema is not None:
        payload["schema"] = schema
    output_dir = tmp_path / "contract-design"
    write_equation_authority_contract_design_reports(payload, output_dir)
    return output_dir / "equation-authority-contract-design-report.json"


def test_equation_authority_record_contract_classifies_contract_design_rows(tmp_path: Path) -> None:
    contract_design_path = _contract_design_report_path(
        tmp_path,
        [
            _hash_row("candidate", design_status=STATUS_HASH_IDENTITY_DESIGN_CANDIDATE_ONLY),
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
        ],
    )

    payload = build_equation_authority_record_contract(contract_design_path)

    assert payload["schema"] == EQUATION_AUTHORITY_RECORD_CONTRACT_SCHEMA_ID
    assert payload["status"] == "ok"
    assert validate_payload(payload, EQUATION_AUTHORITY_RECORD_CONTRACT_SCHEMA_ID, strict=True).ok
    counts = payload["counts"]
    assert counts["targetRows"] == 6
    assert counts["equationAuthorityRecordContracts"] == 1
    assert counts["equationAuthorityRecordSchemas"] == 1
    assert counts["plannedEquationAuthorityRecordRows"] == 1
    assert counts["equationAuthorityRecordContractCandidateOnlyRows"] == 1
    assert counts["blockedMissingDesignIdentityRows"] == 1
    assert counts["blockedMissingDesignHashRows"] == 1
    assert counts["blockedMissingSourceHashRows"] == 1
    assert counts["blockedAmbiguousEquationMatchRows"] == 1
    assert counts["blockedRasterOnlyEquationRows"] == 1
    assert counts["sampleRecordSchemaValidRows"] == 1
    assert counts["sampleRecordSemanticViolationRows"] == 0
    assert counts["equationAuthorityRecordWrittenRows"] == 0
    assert counts["equationArtifactCreatedRows"] == 0
    assert counts["strictEvidenceCreatedRows"] == 0
    assert counts["sourceSpanMutatedRows"] == 0
    assert {row["record_contract_status"] for row in payload["rows"]} == {
        STATUS_EQUATION_AUTHORITY_RECORD_CONTRACT_CANDIDATE_ONLY,
        STATUS_BLOCKED_MISSING_DESIGN_IDENTITY,
        STATUS_BLOCKED_MISSING_DESIGN_HASH,
        STATUS_BLOCKED_MISSING_SOURCE_HASH,
        STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH,
        STATUS_BLOCKED_RASTER_ONLY_EQUATION,
    }
    candidate = next(row for row in payload["rows"] if row["source_candidate_id"] == "candidate")
    record_preview = candidate["equation_authority_record_preview"]
    assert record_preview["schema"] == EQUATION_AUTHORITY_RECORD_SCHEMA_ID
    assert record_preview["plannedWriteTarget"] == EQUATION_AUTHORITY_RECORD_STORE
    assert record_preview["runtimeVisible"] is False
    assert record_preview["answerIntegrationVisible"] is False
    assert validate_payload(record_preview, EQUATION_AUTHORITY_RECORD_SCHEMA_ID, strict=True).ok
    assert validate_equation_authority_record_semantics(record_preview) == []
    assert payload["writeTargets"][0] == EQUATION_AUTHORITY_RECORD_STORE_CONTRACT
    assert KNOWN_WRITE_TARGET_CONTRACTS[EQUATION_AUTHORITY_RECORD_STORE] == (
        EQUATION_AUTHORITY_RECORD_CONTRACT_SCHEMA_ID
    )
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


def test_equation_authority_record_schema_rejects_runtime_or_artifact_mutation(tmp_path: Path) -> None:
    contract_design_path = _contract_design_report_path(
        tmp_path,
        [_hash_row("candidate", design_status=STATUS_HASH_IDENTITY_DESIGN_CANDIDATE_ONLY)],
    )
    contract_design = json.loads(contract_design_path.read_text(encoding="utf-8"))
    record = build_sample_equation_authority_record_from_contract_design_row(contract_design["rows"][0])

    assert validate_payload(record, EQUATION_AUTHORITY_RECORD_SCHEMA_ID, strict=True).ok
    assert validate_equation_authority_record_semantics(record) == []

    record["runtimeVisible"] = True
    record["writePolicy"]["equationArtifactCreated"] = True
    assert validate_payload(record, EQUATION_AUTHORITY_RECORD_SCHEMA_ID, strict=True).ok is False
    errors = validate_equation_authority_record_semantics(record)
    assert "runtimeVisible_must_be_false" in errors
    assert "writePolicy.equationArtifactCreated_must_be_false" in errors


def test_equation_authority_record_contract_blocks_missing_input_report(tmp_path: Path) -> None:
    payload = build_equation_authority_record_contract(tmp_path / "missing-contract-design-report.json")

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == STATUS_BLOCKED_INPUT_REPORT_MISSING
    assert payload["counts"]["blockedInputReportMissingRows"] == 1
    assert payload["rows"] == []
    assert payload["inputBlockers"][0]["record_contract_status"] == STATUS_BLOCKED_INPUT_REPORT_MISSING
    assert validate_payload(payload, EQUATION_AUTHORITY_RECORD_CONTRACT_SCHEMA_ID, strict=True).ok


def test_equation_authority_record_contract_blocks_input_schema_violation(tmp_path: Path) -> None:
    contract_design_path = _contract_design_report_path(
        tmp_path,
        [_hash_row("candidate", design_status=STATUS_HASH_IDENTITY_DESIGN_CANDIDATE_ONLY)],
        schema="wrong.schema",
    )

    payload = build_equation_authority_record_contract(contract_design_path)

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION
    assert payload["counts"]["blockedInputSchemaViolationRows"] == 1
    assert payload["rows"] == []
    assert "schema mismatch" in payload["inputBlockers"][0]["detail"]
    assert validate_payload(payload, EQUATION_AUTHORITY_RECORD_CONTRACT_SCHEMA_ID, strict=True).ok


def test_equation_authority_record_contract_writer_outputs_valid_report(tmp_path: Path) -> None:
    contract_design_path = _contract_design_report_path(
        tmp_path,
        [_hash_row("candidate", design_status=STATUS_HASH_IDENTITY_DESIGN_CANDIDATE_ONLY)],
    )
    payload = build_equation_authority_record_contract(contract_design_path)
    paths = write_equation_authority_record_contract_reports(payload, tmp_path / "reports")

    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert set(paths) == {"report", "summary", "markdown"}
    assert report["schema"] == EQUATION_AUTHORITY_RECORD_CONTRACT_SCHEMA_ID
    assert validate_payload(report, EQUATION_AUTHORITY_RECORD_CONTRACT_SCHEMA_ID, strict=True).ok
    assert summary["status"] == "ok"
    assert summary["counts"]["targetRows"] == 1
    assert "Equation Authority Record Contract" in markdown
    assert validate_payload(report["sampleEquationAuthorityRecord"], EQUATION_AUTHORITY_RECORD_SCHEMA_ID, strict=True).ok
