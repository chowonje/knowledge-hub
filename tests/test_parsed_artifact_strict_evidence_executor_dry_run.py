from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_original_source_offset_authority_design import (
    CHARS_BASIS,
    CHARS_NORMALIZATION,
)
from knowledge_hub.papers.parsed_artifact_source_span_strict_evidence_design_packet_review import (
    PACKET_REVIEW_STATUS_READY,
    PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_PACKET_REVIEW_SCHEMA_ID,
)
from knowledge_hub.papers.parsed_artifact_strict_evidence_record_contract import (
    PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_CONTRACT_SCHEMA_ID,
    build_parsed_artifact_strict_evidence_record_contract,
)
from knowledge_hub.papers.parsed_artifact_strict_evidence_executor_dry_run import (
    DRY_RUN_STATUS_BLOCKED_NORM_MISMATCH,
    DRY_RUN_STATUS_READY,
    PARSED_ARTIFACT_STRICT_EVIDENCE_EXECUTOR_DRY_RUN_SCHEMA_ID,
    build_parsed_artifact_strict_evidence_executor_dry_run,
    compute_contract_substring_sha256,
    compute_raw_utf8_slice_sha256,
    write_parsed_artifact_strict_evidence_executor_dry_run_reports,
)
from knowledge_hub.papers.source_text import source_hash_for_path


def _packet_row(
    *,
    index: int = 1,
    text_surface: str = "Introduction",
    start: int = 0,
    end: int = 12,
    expected_hash: str = "",
) -> dict:
    return {
        "packet_review_row_id": f"packet:{index:04d}",
        "reconciliation_row_id": f"reconciliation:{index:04d}",
        "source": "original_design_review",
        "review_row_id": f"review:{index:04d}",
        "design_row_id": f"offset:{index:04d}",
        "sourceSpanId": f"source-span:paper-1:section:{index}",
        "candidateRecordId": f"source-span-candidate:paper-1:section:{index}",
        "paper_id": "paper-1",
        "artifact_type": "section",
        "sourceContentHash": "hash-paper-1",
        "source_file": "",
        "text_surface": text_surface,
        "proposed_chars": {
            "start": start,
            "end": end,
            "basis": CHARS_BASIS,
            "normalization": CHARS_NORMALIZATION,
            "expectedSubstringSha256": expected_hash,
            "sourceContentHash": "hash-paper-1",
        },
        "packet_review_status": PACKET_REVIEW_STATUS_READY,
        "packet_review_blockers": [],
        "recommended_action": "queue_for_strict_evidence_record_contract",
        "designPacketReviewReady": True,
        "strictEligible": False,
        "strictEvidenceCreated": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "sourceSpanUpdatedRows": 0,
    }


def _packet_review_report(*rows: dict) -> dict:
    row_list = list(rows)
    return {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_PACKET_REVIEW_SCHEMA_ID,
        "status": "ok",
        "generatedAt": "2026-05-19T00:00:00Z",
        "input": {
            "reconciliationReportPath": "/tmp/reconciliation.json",
            "reconciliationSchema": "knowledge-hub.paper.parsed-artifact-source-span-strict-evidence-design-review-reconciliation.v1",
        },
        "counts": {
            "inputRows": 102,
            "packetCandidateRows": len(row_list),
            "designPacketReviewReadyRows": len(row_list),
            "excludedManualOrExtractorRows": 0,
            "blockedMissingRecordIdentityRows": 0,
            "blockedMissingProposedCharsRows": 0,
            "blockedInvalidCharsBasisRows": 0,
            "blockedMissingExpectedSubstringHashRows": 0,
            "blockedRuntimeOrStrictFlagViolationRows": 0,
            "blockedUnexpectedManualRowInPacketRows": 0,
            "blockedInputSchemaViolationRows": 0,
            "sourceSpanUpdatedRows": 0,
            "strictEvidenceCreatedRows": 0,
            "citationGradeEvidenceCreatedRows": 0,
            "runtimeEvidenceCreatedRows": 0,
            "parserRoutingChangedRows": 0,
            "answerIntegrationChangedRows": 0,
            "databaseMutationRows": 0,
            "canonicalParsedArtifactWriteRows": 0,
            "schemaViolationCount": 0,
            "byPaperId": {},
            "byArtifactType": {},
            "byReviewStatus": {},
            "byRecommendedAction": {},
            "byPacketSource": {},
        },
        "gate": {
            "designPacketReviewReady": True,
            "strictEvidenceCreated": False,
            "strictEvidenceReady": False,
            "runtimeEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": [],
            "decision": "parsed_artifact_source_span_strict_evidence_design_packet_review_ready",
            "recommendedNextTranche": "parsed_artifact_strict_evidence_record_contract",
        },
        "policy": {
            "reportOnly": True,
            "sourceSpanStoreWrite": False,
            "strictEvidenceCreated": False,
            "strictEligibleMutation": False,
            "citationGradeEvidenceCreated": False,
            "runtimeEvidenceCreated": False,
            "parserRoutingChanged": False,
            "answerIntegrationChanged": False,
            "databaseMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "canonicalParsedArtifactsWritten": False,
        },
        "warnings": [],
        "packetRows": row_list,
        "excludedManualOrExtractorRows": [],
        "rows": row_list,
    }


def _paper_fixture(tmp_path: Path, *, pdf_text: str) -> str:
    papers_dir = tmp_path / "papers"
    pdf_path = papers_dir / "paper-1.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(pdf_text.encode("utf-8"))
    manifest_dir = papers_dir / "parsed" / "paper-1"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.joinpath("manifest.json").write_text(
        json.dumps({"paper_id": "paper-1", "parser_meta": {"source_pdf": str(pdf_path)}}),
        encoding="utf-8",
    )
    return source_hash_for_path(str(pdf_path))


def test_hash_functions_distinguish_raw_and_contract_slices() -> None:
    canonical = "prefix Introduction suffix"
    start = canonical.index("Introduction")
    end = start + len("Introduction")
    raw_hash = compute_raw_utf8_slice_sha256(canonical, start, end)
    contract_hash = compute_contract_substring_sha256(
        canonical,
        start,
        end,
        normalization=CHARS_NORMALIZATION,
    )
    assert raw_hash != contract_hash  # casefold changes Introduction -> introduction


def test_dry_run_blocks_normalization_hash_contract_mismatch(tmp_path: Path) -> None:
    canonical2 = "A  B"
    start2 = 1
    end2 = 4
    raw_hash2 = compute_raw_utf8_slice_sha256(canonical2, start2, end2)
    contract_hash2 = compute_contract_substring_sha256(
        canonical2, start2, end2,
        normalization=CHARS_NORMALIZATION,
    )
    assert raw_hash2 != contract_hash2

    source_hash2 = _paper_fixture(tmp_path, pdf_text=canonical2)
    packet_path2 = tmp_path / "packet2.json"
    packet_path2.write_text(
        json.dumps(
            _packet_review_report(
                _packet_row(
                    text_surface="  B",
                    start=start2,
                    end=end2,
                    expected_hash=raw_hash2,
                )
            )
        ),
        encoding="utf-8",
    )
    review2 = json.loads(packet_path2.read_text())
    review2["packetRows"][0]["sourceContentHash"] = source_hash2
    review2["packetRows"][0]["proposed_chars"]["sourceContentHash"] = source_hash2
    packet_path2.write_text(json.dumps(review2), encoding="utf-8")

    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(build_parsed_artifact_strict_evidence_record_contract()),
        encoding="utf-8",
    )

    report2 = build_parsed_artifact_strict_evidence_executor_dry_run(
        design_packet_review_report_path=packet_path2,
        contract_report_path=contract_path,
        papers_dir=tmp_path / "papers",
        parsed_root=tmp_path / "papers" / "parsed",
        page_loader=lambda _path: [{"page": 1, "text": canonical2}],
    )

    assert report2["counts"]["blockedNormalizationHashContractMismatchRows"] == 1
    assert report2["rows"][0]["dry_run_status"] == DRY_RUN_STATUS_BLOCKED_NORM_MISMATCH


def test_dry_run_ready_when_contract_hash_matches(tmp_path: Path) -> None:
    canonical = "prefix Introduction suffix"
    start = canonical.index("Introduction")
    end = start + len("Introduction")
    contract_hash = compute_contract_substring_sha256(
        canonical,
        start,
        end,
        normalization=CHARS_NORMALIZATION,
    )

    source_hash = _paper_fixture(tmp_path, pdf_text=canonical)
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(
        json.dumps(
            _packet_review_report(
                _packet_row(
                    text_surface="Introduction",
                    start=start,
                    end=end,
                    expected_hash=contract_hash,
                )
            )
        ),
        encoding="utf-8",
    )
    review = json.loads(packet_path.read_text())
    review["packetRows"][0]["sourceContentHash"] = source_hash
    review["packetRows"][0]["proposed_chars"]["sourceContentHash"] = source_hash
    packet_path.write_text(json.dumps(review), encoding="utf-8")

    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(build_parsed_artifact_strict_evidence_record_contract()),
        encoding="utf-8",
    )

    report = build_parsed_artifact_strict_evidence_executor_dry_run(
        design_packet_review_report_path=packet_path,
        contract_report_path=contract_path,
        papers_dir=tmp_path / "papers",
        parsed_root=tmp_path / "papers" / "parsed",
        page_loader=lambda _path: [{"page": 1, "text": canonical}],
    )

    assert report["counts"]["dryRunReadyStrictEvidenceRecordOnlyRows"] == 1
    assert report["rows"][0]["dry_run_status"] == DRY_RUN_STATUS_READY
    assert report["counts"]["strictEvidenceWriteRows"] == 0
    assert validate_payload(
        report,
        PARSED_ARTIFACT_STRICT_EVIDENCE_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok


def test_dry_run_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(build_parsed_artifact_strict_evidence_record_contract()),
        encoding="utf-8",
    )
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(_packet_review_report()), encoding="utf-8")

    report = build_parsed_artifact_strict_evidence_executor_dry_run(
        design_packet_review_report_path=packet_path,
        contract_report_path=contract_path,
        papers_dir=tmp_path / "papers",
        parsed_root=tmp_path / "papers" / "parsed",
        page_loader=lambda _path: [{"page": 1, "text": ""}],
    )
    paths = write_parsed_artifact_strict_evidence_executor_dry_run_reports(
        report,
        tmp_path / "reports",
    )
    written = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    assert written["schema"] == PARSED_ARTIFACT_STRICT_EVIDENCE_EXECUTOR_DRY_RUN_SCHEMA_ID
    assert validate_payload(
        written,
        PARSED_ARTIFACT_STRICT_EVIDENCE_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok
    assert validate_payload(
        json.loads(Path(paths["summary"]).read_text(encoding="utf-8")),
        PARSED_ARTIFACT_STRICT_EVIDENCE_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok
