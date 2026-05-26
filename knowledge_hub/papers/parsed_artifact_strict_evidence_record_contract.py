"""Contract-only write target definition for parsed-artifact StrictEvidence records.

Defines the StrictEvidence JSONL store contract and record schema after design
packet review. Does not create StrictEvidence records, write JSONL, mutate
SourceSpan rows, or enable citation/runtime/answer integration.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_strict_evidence_design_packet_review import (
    PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_PACKET_REVIEW_SCHEMA_ID,
)


PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_CONTRACT_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-strict-evidence-record-contract.v1"
)
PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-strict-evidence-record.v1"
)

PARSED_ARTIFACT_STRICT_EVIDENCE_STORE = "parsed_artifact_strict_evidence_store"

PROMOTION_GATE_ID = "parsed_artifact_source_span_strict_evidence_design_packet_review"
CHARS_NORMALIZATION_LABEL = "nfkc_whitespace_casefold_v1"
CHARS_BASIS = "sourceContentHash"
AUTHORITY_TYPE_TEXT_OFFSET = "text_offset"
EXECUTOR_BLOCKER_NORMALIZATION_MISMATCH = "blocked_normalization_hash_contract_mismatch"

DEFAULT_DESIGN_PACKET_REVIEW_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-source-span-strict-evidence-design-packet-review"
    / "01-parsed-artifact-source-span-strict-evidence-design-packet-review"
    / "parsed-artifact-source-span-strict-evidence-design-packet-review.json"
)

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-strict-evidence-record-contract"
    / "01-parsed-artifact-strict-evidence-record-contract"
)

NO_MUTATION_POLICY = {
    "contractOnly": True,
    "executorImplemented": False,
    "sourceSpanStoreWrite": False,
    "strictEvidenceCreated": False,
    "citationGradeEvidenceCreated": False,
    "runtimeEvidenceCreated": False,
    "parserRoutingChanged": False,
    "answerIntegrationChanged": False,
    "databaseMutation": False,
    "vaultScan": False,
    "reindexOrReembed": False,
    "canonicalParsedArtifactsWritten": False,
}

REQUIRED_RECORD_FIELDS = [
    "schema",
    "strictEvidenceId",
    "runId",
    "plannedWriteTarget",
    "paperId",
    "artifactType",
    "claimSurface",
    "sourceSpanIds",
    "candidateRecordIds",
    "sourceContentHash",
    "sourceFile",
    "verbatimText",
    "verbatimSubstringSha256",
    "authority",
    "provenanceTrace",
    "designPacketReviewRowId",
    "promotionGateId",
    "idempotencyKey",
    "evidenceTier",
    "strictEligible",
    "citationGrade",
    "runtimeEvidence",
    "writePolicy",
]

IDEMPOTENCY_KEY_FIELDS = [
    "plannedWriteTarget",
    "paperId",
    "artifactType",
    "sourceSpanIds",
    "candidateRecordIds",
    "sourceContentHash",
    "sourceFile",
    "authority",
    "verbatimSubstringSha256",
    "idempotencyKey",
]

NORMALIZATION_HASH_CONTRACT: dict[str, Any] = {
    "normalizationLabel": CHARS_NORMALIZATION_LABEL,
    "basis": CHARS_BASIS,
    "authorityType": AUTHORITY_TYPE_TEXT_OFFSET,
    "designPacketHashField": "authority.chars.expectedSubstringSha256",
    "strictEvidenceCommitField": "verbatimSubstringSha256",
    "equalityRule": "verbatimSubstringSha256_must_equal_authority_chars_expectedSubstringSha256",
    "designPacketImplementationNote": (
        "Offset authority design may have produced expectedSubstringSha256 from raw UTF-8 "
        "canonical_text[start:end] slices while the normalization label is "
        "nfkc_whitespace_casefold_v1."
    ),
    "executorRequirement": (
        "Executor dry-run and readback must compute verbatimSubstringSha256 using the exact "
        "same function applied to authority.chars.expectedSubstringSha256. If the label and "
        "computation disagree, block with blocked_normalization_hash_contract_mismatch."
    ),
    "silentMismatchAllowed": False,
    "executorBlockerOnMismatch": EXECUTOR_BLOCKER_NORMALIZATION_MISMATCH,
}

STRICT_EVIDENCE_STORE_CONTRACT: dict[str, Any] = {
    "plannedWriteTarget": PARSED_ARTIFACT_STRICT_EVIDENCE_STORE,
    "contractReference": PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_CONTRACT_SCHEMA_ID,
    "strictEvidenceRecordSchema": PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_SCHEMA_ID,
    "storeKind": "local_papers_dir_jsonl_strict_evidence_store",
    "storeRootTemplate": "{papers_dir}/structured_evidence/strict_evidence",
    "recordPathTemplate": "{papers_dir}/structured_evidence/strict_evidence/{paper_id}.jsonl",
    "runManifestPathTemplate": "{papers_dir}/structured_evidence/runs/{run_id}.json",
    "allowedArtifactTypes": ["section", "figure"],
    "requiredRecordFields": REQUIRED_RECORD_FIELDS,
    "idempotencyKeyFields": IDEMPOTENCY_KEY_FIELDS,
    "writeSemantics": "explicit_apply_executor_appends_or_replaces_same_idempotency_key",
    "readbackChecks": [
        "record_schema_validates",
        "idempotency_key_stable",
        "every_sourceSpanId_resolves_to_existing_source_span_jsonl",
        "candidateRecordIds_preserved",
        "sourceContentHash_preserved",
        "chars_start_end_basis_normalization_hash_preserved",
        "verbatimSubstringSha256_equals_authority_chars_expectedSubstringSha256",
        "record_remains_non_citation_and_non_runtime",
        "source_span_row_remains_unchanged",
    ],
    "rollbackStrategy": (
        "delete records written by the explicit run_id only while no downstream "
        "citation or runtime binding references them"
    ),
    "rollbackImplemented": False,
    "normalizationHashContract": dict(NORMALIZATION_HASH_CONTRACT),
    "executorImplemented": False,
    "runtimeUseAllowed": False,
    "citationUseAllowed": False,
    "parserRoutingAllowed": False,
    "answerIntegrationAllowed": False,
    "databaseMutationAllowed": False,
    "sourceSpanMutationAllowed": False,
}

KNOWN_WRITE_TARGET_CONTRACTS: dict[str, str] = {
    PARSED_ARTIFACT_STRICT_EVIDENCE_STORE: PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_CONTRACT_SCHEMA_ID,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _write_policy() -> dict[str, Any]:
    return {
        "executorRequired": True,
        "sourceSpanStoreWrite": False,
        "databaseMutation": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "reindexOrReembed": False,
        "canonicalParsedArtifactsWritten": False,
    }


def validate_strict_evidence_record_semantics(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    authority = record.get("authority") if isinstance(record.get("authority"), dict) else {}
    chars = authority.get("chars") if isinstance(authority.get("chars"), dict) else {}
    expected_hash = _safe_text(chars.get("expectedSubstringSha256"))
    verbatim_hash = _safe_text(record.get("verbatimSubstringSha256"))
    if expected_hash and verbatim_hash and expected_hash != verbatim_hash:
        errors.append("verbatimSubstringSha256_must_equal_authority_chars_expectedSubstringSha256")
    source_span_ids = record.get("sourceSpanIds")
    if not isinstance(source_span_ids, list) or not source_span_ids:
        errors.append("sourceSpanIds_must_be_non_empty")
    candidate_ids = record.get("candidateRecordIds")
    if not isinstance(candidate_ids, list) or not candidate_ids:
        errors.append("candidateRecordIds_must_be_non_empty")
    start = chars.get("start")
    end = chars.get("end")
    try:
        if start is not None and end is not None and int(end) <= int(start):
            errors.append("authority_chars_end_must_be_greater_than_start")
    except Exception:
        errors.append("authority_chars_start_or_end_invalid")
    return errors


def build_sample_strict_evidence_record_from_packet_row(
    packet_row: dict[str, Any],
    *,
    run_id: str = "contract-sample-run",
    strict_evidence_id: str | None = None,
    design_packet_review_report_path: str = "",
) -> dict[str, Any]:
    proposed = packet_row.get("proposed_chars")
    proposed = proposed if isinstance(proposed, dict) else {}
    paper_id = _safe_text(packet_row.get("paper_id"))
    source_span_id = _safe_text(packet_row.get("sourceSpanId"))
    candidate_record_id = _safe_text(packet_row.get("candidateRecordId"))
    artifact_type = _safe_text(packet_row.get("artifact_type"))
    expected_hash = _safe_text(proposed.get("expectedSubstringSha256"))
    start = proposed.get("start")
    end = proposed.get("end")

    record_id = strict_evidence_id or (
        f"strict-evidence:{paper_id}:{artifact_type}:{source_span_id.split(':')[-1]}"
    )
    idempotency_key = (
        f"strict:{source_span_id}:{CHARS_NORMALIZATION_LABEL}:{start}:{end}:{expected_hash}"
    )

    return {
        "schema": PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_SCHEMA_ID,
        "strictEvidenceId": record_id,
        "runId": run_id,
        "plannedWriteTarget": PARSED_ARTIFACT_STRICT_EVIDENCE_STORE,
        "paperId": paper_id,
        "artifactType": artifact_type,
        "claimSurface": _safe_text(packet_row.get("text_surface")),
        "sourceSpanIds": [source_span_id],
        "candidateRecordIds": [candidate_record_id],
        "sourceContentHash": _safe_text(packet_row.get("sourceContentHash")),
        "sourceFile": _safe_text(packet_row.get("source_file")),
        "verbatimText": _safe_text(packet_row.get("text_surface")),
        "verbatimSubstringSha256": expected_hash,
        "authority": {
            "type": AUTHORITY_TYPE_TEXT_OFFSET,
            "chars": {
                "start": start,
                "end": end,
                "basis": CHARS_BASIS,
                "normalization": CHARS_NORMALIZATION_LABEL,
                "expectedSubstringSha256": expected_hash,
            },
        },
        "provenanceTrace": {
            "designPacketReviewReportPath": design_packet_review_report_path,
            "designPacketReviewRowId": _safe_text(packet_row.get("packet_review_row_id")),
            "reconciliationRowId": _safe_text(packet_row.get("reconciliation_row_id")),
            "packetSource": _safe_text(packet_row.get("source")),
            "reviewRowId": _safe_text(packet_row.get("review_row_id")),
            "designRowId": _safe_text(packet_row.get("design_row_id")),
        },
        "designPacketReviewRowId": _safe_text(packet_row.get("packet_review_row_id")),
        "promotionGateId": PROMOTION_GATE_ID,
        "idempotencyKey": idempotency_key,
        "evidenceTier": "parsed_artifact_strict_evidence",
        "strictEligible": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "writePolicy": _write_policy(),
    }


def build_parsed_artifact_strict_evidence_record_contract(
    *,
    design_packet_review_report_path: str | Path = DEFAULT_DESIGN_PACKET_REVIEW_REPORT_PATH,
) -> dict[str, Any]:
    report_path = Path(str(design_packet_review_report_path)).expanduser()
    warnings: list[str] = []
    input_notes: list[str] = []
    packet_ready_rows = 0

    packet_report: dict[str, Any] = {}
    if report_path.is_file():
        try:
            packet_report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            warnings.append("design_packet_review_report_unreadable")
    else:
        warnings.append("design_packet_review_report_missing")

    if packet_report:
        validation = validate_payload(
            packet_report,
            PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_PACKET_REVIEW_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            warnings.append("design_packet_review_report_schema_invalid")
            input_notes.extend(str(error) for error in validation.errors[:5])
        packet_ready_rows = int((packet_report.get("counts") or {}).get("designPacketReviewReadyRows") or 0)

    write_target = dict(STRICT_EVIDENCE_STORE_CONTRACT)
    return {
        "schema": PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_CONTRACT_SCHEMA_ID,
        "status": "ok",
        "generatedAt": _now_iso(),
        "input": {
            "designPacketReviewReportPath": str(report_path),
            "designPacketReviewSchema": _safe_text(packet_report.get("schema")),
            "designPacketReviewReadyRows": packet_ready_rows,
        },
        "counts": {
            "writeTargetContracts": 1,
            "strictEvidenceStoreContracts": 1,
            "strictEvidenceRecordSchemas": 1,
            "executorImplementedRows": 0,
            "strictEvidenceCreatedRows": 0,
            "citationGradeEvidenceCreatedRows": 0,
            "runtimeEvidenceCreatedRows": 0,
            "parserRoutingChangedRows": 0,
            "answerIntegrationChangedRows": 0,
            "databaseMutationRows": 0,
            "canonicalParsedArtifactWriteRows": 0,
        },
        "gate": {
            "writeTargetContractsDefined": True,
            "strictEvidenceStoreContractDefined": True,
            "executorReady": False,
            "runtimeMutationAllowed": False,
            "strictEvidenceReady": False,
            "citationReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "decision": "parsed_artifact_strict_evidence_record_contract_ready",
            "recommendedNextTranche": "parsed_artifact_strict_evidence_executor_dry_run",
        },
        "policy": dict(NO_MUTATION_POLICY),
        "contractPrinciples": [
            "strict_evidence_records_are_append_only_and_separate_from_source_span_records",
            "strict_evidence_executor_must_not_mutate_source_span_jsonl",
            "verbatimSubstringSha256_must_match_authority_chars_expectedSubstringSha256",
            "executor_must_verify_normalization_hash_semantics_before_create",
            "blocked_normalization_hash_contract_mismatch_when_label_and_computation_disagree",
            "citation_runtime_and_answer_integration_remain_disabled_for_this_tranche",
            "rollback_is_contract_only_until_explicit_rollback_executor_tranche",
        ],
        "normalizationHashContract": dict(NORMALIZATION_HASH_CONTRACT),
        "writeTargets": [write_target],
        "warnings": warnings,
        "inputNotes": input_notes,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in (
            "schema",
            "status",
            "generatedAt",
            "input",
            "counts",
            "gate",
            "policy",
            "contractPrinciples",
            "normalizationHashContract",
            "writeTargets",
            "warnings",
            "inputNotes",
        )
        if key in report
    }


def render_parsed_artifact_strict_evidence_record_contract_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    norm = dict(report.get("normalizationHashContract") or {})
    lines = [
        "# Parsed Artifact StrictEvidence Record Contract",
        "",
        f"- status: {report.get('status', '')}",
        f"- contract-only: {json.dumps(report.get('policy', {}).get('contractOnly'))}",
        f"- write target contracts: {int(counts.get('writeTargetContracts') or 0)}",
        f"- strict evidence created: {int(counts.get('strictEvidenceCreatedRows') or 0)}",
        f"- design packet review ready rows (input): {int((report.get('input') or {}).get('designPacketReviewReadyRows') or 0)}",
        "",
        "## Normalization / hash contract",
        f"- normalization label: {norm.get('normalizationLabel', '')}",
        f"- equality rule: {norm.get('equalityRule', '')}",
        f"- executor blocker on mismatch: {norm.get('executorBlockerOnMismatch', '')}",
        f"- design note: {norm.get('designPacketImplementationNote', '')}",
        "",
        "## Write Targets",
    ]
    for target in list(report.get("writeTargets") or []):
        lines.extend(
            [
                f"- {target.get('plannedWriteTarget', '')}",
                f"  - record schema: {target.get('strictEvidenceRecordSchema', '')}",
                f"  - store root: {target.get('storeRootTemplate', '')}",
                f"  - record path: {target.get('recordPathTemplate', '')}",
                f"  - rollback implemented: {json.dumps(target.get('rollbackImplemented'))}",
            ]
        )
    return "\n".join(lines)


def write_parsed_artifact_strict_evidence_record_contract_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-strict-evidence-record-contract.json"
    summary_path = root / "parsed-artifact-strict-evidence-record-contract-summary.json"
    markdown_path = root / "parsed-artifact-strict-evidence-record-contract.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_parsed_artifact_strict_evidence_record_contract_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description="Emit the contract-only parsed-artifact StrictEvidence store payload."
    )
    parser.add_argument(
        "--design-packet-review-report",
        default=str(DEFAULT_DESIGN_PACKET_REVIEW_REPORT_PATH),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = build_parsed_artifact_strict_evidence_record_contract(
        design_packet_review_report_path=args.design_packet_review_report,
    )
    if args.output_dir:
        paths = write_parsed_artifact_strict_evidence_record_contract_reports(
            payload,
            args.output_dir,
        )
        print(f"wrote report: {paths['report']}")
        print(f"wrote summary: {paths['summary']}")
        print(f"wrote markdown: {paths['markdown']}")

    if args.json or not args.output_dir:
        print(json.dumps(_summary_payload(payload), ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_TYPE_TEXT_OFFSET",
    "CHARS_BASIS",
    "CHARS_NORMALIZATION_LABEL",
    "EXECUTOR_BLOCKER_NORMALIZATION_MISMATCH",
    "KNOWN_WRITE_TARGET_CONTRACTS",
    "NORMALIZATION_HASH_CONTRACT",
    "NO_MUTATION_POLICY",
    "PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_CONTRACT_SCHEMA_ID",
    "PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_SCHEMA_ID",
    "PARSED_ARTIFACT_STRICT_EVIDENCE_STORE",
    "PROMOTION_GATE_ID",
    "STRICT_EVIDENCE_STORE_CONTRACT",
    "build_parsed_artifact_strict_evidence_record_contract",
    "build_sample_strict_evidence_record_from_packet_row",
    "validate_strict_evidence_record_semantics",
    "render_parsed_artifact_strict_evidence_record_contract_markdown",
    "write_parsed_artifact_strict_evidence_record_contract_reports",
]
