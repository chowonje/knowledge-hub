"""Contract-only equation authority record definition.

This helper consumes the report-only Equation Authority contract design report
and defines the future append-only equation authority record/store contract.
It is still classification and contract planning only: no equation authority
records, EquationArtifact, StrictEvidence, SourceSpan, DB/index state, vault
content, parser routing, answer integration, or authority policy is created.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.equation_authority_contract_design import (
    EQUATION_AUTHORITY_CONTRACT_DESIGN_SCHEMA_ID,
    FUTURE_EQUATION_AUTHORITY_RECORD_SCHEMA_ID,
    FUTURE_EQUATION_AUTHORITY_STORE,
    STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH as CONTRACT_STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH,
    STATUS_BLOCKED_MISSING_DESIGN_HASH as CONTRACT_STATUS_BLOCKED_MISSING_DESIGN_HASH,
    STATUS_BLOCKED_MISSING_DESIGN_IDENTITY as CONTRACT_STATUS_BLOCKED_MISSING_DESIGN_IDENTITY,
    STATUS_BLOCKED_MISSING_SOURCE_HASH as CONTRACT_STATUS_BLOCKED_MISSING_SOURCE_HASH,
    STATUS_BLOCKED_RASTER_ONLY_EQUATION as CONTRACT_STATUS_BLOCKED_RASTER_ONLY_EQUATION,
    STATUS_EQUATION_AUTHORITY_CONTRACT_CANDIDATE_ONLY,
)


EQUATION_AUTHORITY_RECORD_CONTRACT_SCHEMA_ID = (
    "knowledge-hub.paper.equation-authority-record-contract.v1"
)
EQUATION_AUTHORITY_RECORD_SCHEMA_ID = FUTURE_EQUATION_AUTHORITY_RECORD_SCHEMA_ID
EQUATION_AUTHORITY_RECORD_STORE = FUTURE_EQUATION_AUTHORITY_STORE
EQUATION_AUTHORITY_RECORD_CONTRACT_VERSION = "equation_authority_record_contract_v1"
EQUATION_AUTHORITY_RECORD_DECISION = "equation_authority_record_contract_candidate_only"
EQUATION_AUTHORITY_RECORD_STATE = "record_contract_candidate_only"

DEFAULT_EQUATION_AUTHORITY_CONTRACT_DESIGN_REPORT = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "equation-authority-contract-design"
    / "equation-authority-contract-design-report.json"
)

DEFAULT_EQUATION_AUTHORITY_RECORD_CONTRACT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "equation-authority-record-contract"
)

STATUS_EQUATION_AUTHORITY_RECORD_CONTRACT_CANDIDATE_ONLY = (
    "equation_authority_record_contract_candidate_only"
)
STATUS_BLOCKED_MISSING_CONTRACT_PREVIEW = "blocked_missing_contract_preview"
STATUS_BLOCKED_MISSING_DESIGN_IDENTITY = "blocked_missing_design_identity"
STATUS_BLOCKED_MISSING_DESIGN_HASH = "blocked_missing_design_hash"
STATUS_BLOCKED_MISSING_SOURCE_HASH = "blocked_missing_source_hash"
STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH = "blocked_ambiguous_equation_match"
STATUS_BLOCKED_RASTER_ONLY_EQUATION = "blocked_raster_only_equation"
STATUS_HELD_OUT_NON_RECORD_CONTRACT_BLOCKER = "held_out_non_record_contract_blocker"
STATUS_BLOCKED_INPUT_REPORT_MISSING = "blocked_input_report_missing"
STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION = "blocked_input_schema_violation"

REQUIRED_RECORD_FIELDS = [
    "schema",
    "equationAuthorityRecordId",
    "runId",
    "plannedWriteTarget",
    "contractVersion",
    "paperId",
    "sourceCandidateId",
    "sourceTexRowId",
    "sourceFile",
    "equationEnvironment",
    "sourceContentHash",
    "equationIdentityDigestSha256",
    "equationHashSha256",
    "equationTextNormalization",
    "hashSource",
    "readinessAuditRowId",
    "hashIdentityDesignRowId",
    "equationAuthorityContractDesignRowId",
    "pdfRegionSnapshot",
    "canonicalAlignmentSnapshot",
    "surroundingContextSnapshot",
    "ambiguitySnapshot",
    "authorityDecision",
    "authorityState",
    "runtimeVisible",
    "answerIntegrationVisible",
    "equationArtifactCreated",
    "strictEvidenceCreated",
    "sourceSpanMutationAllowed",
    "idempotencyKey",
    "provenanceTrace",
    "writePolicy",
]

IDEMPOTENCY_KEY_FIELDS = [
    "plannedWriteTarget",
    "contractVersion",
    "paperId",
    "sourceContentHash",
    "equationIdentityDigestSha256",
    "equationHashSha256",
    "readinessAuditRowId",
    "hashIdentityDesignRowId",
    "equationAuthorityContractDesignRowId",
]

NO_MUTATION_POLICY = {
    "reportOnly": True,
    "contractOnly": True,
    "classificationOnly": True,
    "executorImplemented": False,
    "authorityPolicyCreated": False,
    "equationAuthorityRecordWrite": False,
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
    "manifestWrite": False,
    "equationInterpretationAllowed": False,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_bool(value: Any) -> bool:
    return bool(value)


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_safe_text(value)] if _safe_text(value) else []
    if not isinstance(value, (list, tuple)):
        return []
    return [_safe_text(item) for item in value if _safe_text(item)]


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        return {}, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, "unreadable"
    if not isinstance(payload, dict):
        return {}, "unreadable"
    return payload, ""


def _input_blockers(report_path: Path, report: dict[str, Any], load_error: str) -> list[dict[str, str]]:
    if load_error == "missing":
        return [
            {
                "inputName": "equationAuthorityContractDesign",
                "inputLabel": "Equation Authority contract design report",
                "path": str(report_path),
                "record_contract_status": STATUS_BLOCKED_INPUT_REPORT_MISSING,
                "detail": "expected input report is missing",
            }
        ]
    if load_error:
        return [
            {
                "inputName": "equationAuthorityContractDesign",
                "inputLabel": "Equation Authority contract design report",
                "path": str(report_path),
                "record_contract_status": STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION,
                "detail": "expected input report is unreadable or not a JSON object",
            }
        ]
    schema = _safe_text(report.get("schema"))
    if schema != EQUATION_AUTHORITY_CONTRACT_DESIGN_SCHEMA_ID:
        return [
            {
                "inputName": "equationAuthorityContractDesign",
                "inputLabel": "Equation Authority contract design report",
                "path": str(report_path),
                "record_contract_status": STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION,
                "detail": (
                    "schema mismatch: expected "
                    f"{EQUATION_AUTHORITY_CONTRACT_DESIGN_SCHEMA_ID}, got {schema or 'missing'}"
                ),
            }
        ]
    validation = validate_payload(report, EQUATION_AUTHORITY_CONTRACT_DESIGN_SCHEMA_ID, strict=True)
    if validation.ok:
        return []
    return [
        {
            "inputName": "equationAuthorityContractDesign",
            "inputLabel": "Equation Authority contract design report",
            "path": str(report_path),
            "record_contract_status": STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION,
            "detail": f"schema validation failed: {error}",
        }
        for error in validation.errors
    ]


def _write_policy() -> dict[str, Any]:
    return {
        "executorRequired": True,
        "equationAuthorityRecordWrite": False,
        "authorityPolicyCreated": False,
        "equationIdentityPromoted": False,
        "equationHashPromoted": False,
        "equationArtifactCreated": False,
        "strictEvidenceCreated": False,
        "sourceSpanMutation": False,
        "runtimeVisible": False,
        "answerIntegrationVisible": False,
        "databaseMutation": False,
        "indexMutation": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "reindexOrReembed": False,
        "canonicalParsedArtifactsWritten": False,
        "manifestWrite": False,
        "vaultScan": False,
    }


def _record_schema_contract() -> dict[str, Any]:
    return {
        "recordSchema": EQUATION_AUTHORITY_RECORD_SCHEMA_ID,
        "contractReference": EQUATION_AUTHORITY_RECORD_CONTRACT_SCHEMA_ID,
        "recordState": EQUATION_AUTHORITY_RECORD_STATE,
        "recordDecision": EQUATION_AUTHORITY_RECORD_DECISION,
        "contractVersion": EQUATION_AUTHORITY_RECORD_CONTRACT_VERSION,
        "requiredRecordFields": REQUIRED_RECORD_FIELDS,
        "idempotencyKeyFields": IDEMPOTENCY_KEY_FIELDS,
        "semanticInvariants": [
            "sourceContentHash_must_be_preserved",
            "equation_identity_digest_must_be_non_empty",
            "equation_hash_must_be_non_empty",
            "contract_design_row_must_resolve",
            "hash_identity_design_row_must_resolve",
            "readiness_audit_row_must_resolve",
            "runtimeVisible_must_be_false",
            "answerIntegrationVisible_must_be_false",
            "equationArtifactCreated_must_be_false",
            "strictEvidenceCreated_must_be_false",
            "sourceSpanMutationAllowed_must_be_false",
        ],
        "executorImplemented": False,
        "recordWriteAllowed": False,
    }


EQUATION_AUTHORITY_RECORD_STORE_CONTRACT: dict[str, Any] = {
    "plannedWriteTarget": EQUATION_AUTHORITY_RECORD_STORE,
    "contractReference": EQUATION_AUTHORITY_RECORD_CONTRACT_SCHEMA_ID,
    "equationAuthorityRecordSchema": EQUATION_AUTHORITY_RECORD_SCHEMA_ID,
    "storeKind": "future_local_papers_dir_jsonl_equation_authority_store",
    "storeRootTemplate": "{papers_dir}/structured_evidence/equation_authority",
    "recordPathTemplate": "{papers_dir}/structured_evidence/equation_authority/{paper_id}.jsonl",
    "runManifestPathTemplate": "{papers_dir}/structured_evidence/runs/{run_id}.json",
    "requiredRecordFields": REQUIRED_RECORD_FIELDS,
    "idempotencyKeyFields": IDEMPOTENCY_KEY_FIELDS,
    "writeSemantics": "explicit_apply_executor_appends_or_replaces_same_idempotency_key",
    "readbackChecks": [
        "equation_authority_record_schema_validates",
        "idempotency_key_stable",
        "sourceContentHash_preserved",
        "equation_identity_digest_preserved",
        "equation_hash_preserved",
        "contract_design_row_resolves",
        "hash_identity_design_row_resolves",
        "readiness_audit_row_resolves",
        "no_equation_artifact_created",
        "no_strict_evidence_created",
        "source_span_remains_unmutated",
        "runtime_visible_remains_false_until_later_gate",
        "answer_integration_visible_remains_false_until_later_gate",
    ],
    "rollbackStrategy": (
        "delete or invalidate equation authority records written by the explicit run_id "
        "before any later runtime or answer-integration gate references them"
    ),
    "rollbackImplemented": False,
    "executorImplemented": False,
    "recordWriteAllowed": False,
    "authorityPolicyMutationAllowed": False,
    "equationArtifactCreationAllowed": False,
    "strictEvidenceCreationAllowed": False,
    "sourceSpanMutationAllowed": False,
    "runtimeUseAllowed": False,
    "parserRoutingAllowed": False,
    "answerIntegrationAllowed": False,
    "databaseMutationAllowed": False,
    "indexMutationAllowed": False,
    "vaultScanAllowed": False,
}

KNOWN_WRITE_TARGET_CONTRACTS: dict[str, str] = {
    EQUATION_AUTHORITY_RECORD_STORE: EQUATION_AUTHORITY_RECORD_CONTRACT_SCHEMA_ID,
}


def _preview(row: dict[str, Any]) -> dict[str, Any]:
    preview = row.get("contract_record_preview")
    return dict(preview) if isinstance(preview, dict) else {}


def _preview_text(row: dict[str, Any], field: str) -> str:
    return _safe_text(_preview(row).get(field))


def _classify(row: dict[str, Any]) -> tuple[str, list[str], str]:
    input_status = _safe_text(row.get("contract_status"))
    preview = _preview(row)
    ambiguity = dict(row.get("ambiguity") or {})
    if not preview:
        return (
            STATUS_BLOCKED_MISSING_CONTRACT_PREVIEW,
            ["contract_record_preview_missing"],
            "recover_equation_authority_contract_design_preview_before_record_contract",
        )
    if not _preview_text(row, "sourceContentHash") and not _safe_text(row.get("sourceContentHash")):
        return (
            STATUS_BLOCKED_MISSING_SOURCE_HASH,
            ["sourceContentHash_missing"],
            "recover_source_content_hash_before_equation_authority_record_contract",
        )
    if _safe_bool(row.get("raster_image_only_blocker")) or input_status == CONTRACT_STATUS_BLOCKED_RASTER_ONLY_EQUATION:
        return (
            STATUS_BLOCKED_RASTER_ONLY_EQUATION,
            ["raster_or_image_only_equation_signal"],
            "route_to_image_or_ocr_equation_extractor_before_record_contract",
        )
    if _safe_bool(ambiguity.get("nonUniqueMatch")) or input_status == CONTRACT_STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH:
        return (
            STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH,
            _string_list(ambiguity.get("reasons")) or ["equation_match_non_unique"],
            "resolve_non_unique_equation_match_before_record_contract",
        )
    if not _preview_text(row, "equationIdentityDigestSha256"):
        return (
            STATUS_BLOCKED_MISSING_DESIGN_IDENTITY,
            ["equation_identity_design_digest_missing"],
            "recover_contract_design_identity_before_record_contract",
        )
    if not _preview_text(row, "equationHashSha256"):
        return (
            STATUS_BLOCKED_MISSING_DESIGN_HASH,
            ["equation_hash_design_missing"],
            "recover_contract_design_hash_before_record_contract",
        )
    if input_status == CONTRACT_STATUS_BLOCKED_MISSING_DESIGN_IDENTITY:
        return (
            STATUS_BLOCKED_MISSING_DESIGN_IDENTITY,
            ["input_contract_status=blocked_missing_design_identity"],
            "recover_contract_design_identity_before_record_contract",
        )
    if input_status == CONTRACT_STATUS_BLOCKED_MISSING_DESIGN_HASH:
        return (
            STATUS_BLOCKED_MISSING_DESIGN_HASH,
            ["input_contract_status=blocked_missing_design_hash"],
            "recover_contract_design_hash_before_record_contract",
        )
    if input_status == CONTRACT_STATUS_BLOCKED_MISSING_SOURCE_HASH:
        return (
            STATUS_BLOCKED_MISSING_SOURCE_HASH,
            ["input_contract_status=blocked_missing_source_hash"],
            "recover_source_content_hash_before_equation_authority_record_contract",
        )
    if input_status != STATUS_EQUATION_AUTHORITY_CONTRACT_CANDIDATE_ONLY:
        return (
            STATUS_HELD_OUT_NON_RECORD_CONTRACT_BLOCKER,
            [f"input_contract_status={input_status or 'missing'}"],
            "resolve_contract_design_blocker_before_record_contract",
        )
    return (
        STATUS_EQUATION_AUTHORITY_RECORD_CONTRACT_CANDIDATE_ONLY,
        ["equation_authority_record_contract_only"],
        "queue_for_later_explicit_equation_authority_record_executor_dry_run",
    )


def _record_id(row: dict[str, Any]) -> str:
    preview_id = _preview_text(row, "equationAuthorityRecordId")
    if preview_id:
        return preview_id
    digest = _sha256_json(
        {
            "paperId": _safe_text(row.get("paper_id")),
            "sourceContentHash": _safe_text(row.get("sourceContentHash")),
            "hashIdentityDesignRowId": _safe_text(row.get("hash_identity_design_row_id")),
            "contractDesignRowId": _safe_text(row.get("contract_row_id")),
            "contractVersion": EQUATION_AUTHORITY_RECORD_CONTRACT_VERSION,
        }
    )
    return f"eqauth-record:{_safe_text(row.get('paper_id'))}:{digest[:20]}"


def _idempotency_key(record: dict[str, Any]) -> str:
    return _sha256_json({field: record.get(field, "") for field in IDEMPOTENCY_KEY_FIELDS})


def build_sample_equation_authority_record_from_contract_design_row(
    contract_design_row: dict[str, Any],
    *,
    run_id: str = "equation-authority-record-contract-sample-run",
    equation_authority_record_id: str | None = None,
) -> dict[str, Any]:
    preview = _preview(contract_design_row)
    identity = dict(contract_design_row.get("equation_identity") or {})
    hash_design = dict(contract_design_row.get("proposed_hash_design") or {})
    record = {
        "schema": EQUATION_AUTHORITY_RECORD_SCHEMA_ID,
        "equationAuthorityRecordId": equation_authority_record_id
        or _safe_text(preview.get("equationAuthorityRecordId"))
        or _record_id(contract_design_row),
        "runId": run_id,
        "plannedWriteTarget": EQUATION_AUTHORITY_RECORD_STORE,
        "contractVersion": EQUATION_AUTHORITY_RECORD_CONTRACT_VERSION,
        "paperId": _safe_text(contract_design_row.get("paper_id") or preview.get("paperId")),
        "sourceCandidateId": _safe_text(
            contract_design_row.get("source_candidate_id") or preview.get("sourceCandidateId")
        ),
        "sourceTexRowId": _safe_text(preview.get("sourceTexRowId") or identity.get("sourceTexRowId")),
        "sourceFile": _safe_text(contract_design_row.get("source_file")),
        "equationEnvironment": _safe_text(contract_design_row.get("equation_environment")),
        "sourceContentHash": _safe_text(
            preview.get("sourceContentHash") or contract_design_row.get("sourceContentHash")
        ),
        "equationIdentityDigestSha256": _safe_text(preview.get("equationIdentityDigestSha256")),
        "equationHashSha256": _safe_text(preview.get("equationHashSha256")),
        "equationTextNormalization": _safe_text(
            preview.get("equationTextNormalization") or hash_design.get("textNormalization")
        ),
        "hashSource": _safe_text(preview.get("hashSource") or hash_design.get("hashSource")),
        "readinessAuditRowId": _safe_text(
            preview.get("readinessAuditRowId") or contract_design_row.get("readiness_audit_row_id")
        ),
        "hashIdentityDesignRowId": _safe_text(
            preview.get("hashIdentityDesignRowId") or contract_design_row.get("hash_identity_design_row_id")
        ),
        "equationAuthorityContractDesignRowId": _safe_text(contract_design_row.get("contract_row_id")),
        "pdfRegionSnapshot": dict(
            preview.get("pdfRegionSnapshot") or contract_design_row.get("pdf_region_signal") or {}
        ),
        "canonicalAlignmentSnapshot": dict(
            preview.get("canonicalAlignmentSnapshot")
            or contract_design_row.get("canonical_text_alignment")
            or {}
        ),
        "surroundingContextSnapshot": dict(contract_design_row.get("surrounding_context") or {}),
        "ambiguitySnapshot": dict(
            preview.get("ambiguitySnapshot") or contract_design_row.get("ambiguity") or {}
        ),
        "authorityDecision": EQUATION_AUTHORITY_RECORD_DECISION,
        "authorityState": EQUATION_AUTHORITY_RECORD_STATE,
        "runtimeVisible": False,
        "answerIntegrationVisible": False,
        "equationArtifactCreated": False,
        "strictEvidenceCreated": False,
        "sourceSpanMutationAllowed": False,
        "idempotencyKey": "",
        "provenanceTrace": {
            "equationAuthorityContractDesignRowId": _safe_text(contract_design_row.get("contract_row_id")),
            "hashIdentityDesignRowId": _safe_text(
                preview.get("hashIdentityDesignRowId") or contract_design_row.get("hash_identity_design_row_id")
            ),
            "readinessAuditRowId": _safe_text(
                preview.get("readinessAuditRowId") or contract_design_row.get("readiness_audit_row_id")
            ),
            "sourceCandidateId": _safe_text(
                contract_design_row.get("source_candidate_id") or preview.get("sourceCandidateId")
            ),
            "sourceContentHash": _safe_text(
                preview.get("sourceContentHash") or contract_design_row.get("sourceContentHash")
            ),
            "inputContractStatus": _safe_text(contract_design_row.get("contract_status")),
            "inputDesignStatus": _safe_text(contract_design_row.get("input_design_status")),
            "readinessStatus": _safe_text(contract_design_row.get("readiness_status")),
        },
        "writePolicy": _write_policy(),
    }
    record["idempotencyKey"] = _idempotency_key(record)
    return record


def validate_equation_authority_record_semantics(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if _safe_text(record.get("plannedWriteTarget")) != EQUATION_AUTHORITY_RECORD_STORE:
        errors.append("plannedWriteTarget_must_be_equation_authority_record_candidate_store")
    if _safe_text(record.get("contractVersion")) != EQUATION_AUTHORITY_RECORD_CONTRACT_VERSION:
        errors.append("contractVersion_mismatch")
    if _safe_text(record.get("authorityDecision")) != EQUATION_AUTHORITY_RECORD_DECISION:
        errors.append("authorityDecision_mismatch")
    if _safe_text(record.get("authorityState")) != EQUATION_AUTHORITY_RECORD_STATE:
        errors.append("authorityState_mismatch")
    for field_name in (
        "paperId",
        "sourceContentHash",
        "equationIdentityDigestSha256",
        "equationHashSha256",
        "readinessAuditRowId",
        "hashIdentityDesignRowId",
        "equationAuthorityContractDesignRowId",
        "idempotencyKey",
    ):
        if not _safe_text(record.get(field_name)):
            errors.append(f"{field_name}_must_be_non_empty")
    for field_name in (
        "runtimeVisible",
        "answerIntegrationVisible",
        "equationArtifactCreated",
        "strictEvidenceCreated",
        "sourceSpanMutationAllowed",
    ):
        if _safe_bool(record.get(field_name)):
            errors.append(f"{field_name}_must_be_false")
    write_policy = dict(record.get("writePolicy") or {})
    for field_name in (
        "equationAuthorityRecordWrite",
        "authorityPolicyCreated",
        "equationArtifactCreated",
        "strictEvidenceCreated",
        "sourceSpanMutation",
        "databaseMutation",
        "indexMutation",
        "parserRoutingChanged",
        "answerIntegrationChanged",
        "vaultScan",
    ):
        if _safe_bool(write_policy.get(field_name)):
            errors.append(f"writePolicy.{field_name}_must_be_false")
    return errors


def _row(index: int, contract_design_row: dict[str, Any]) -> dict[str, Any]:
    status, blockers, recommended_action = _classify(contract_design_row)
    input_status = _safe_text(contract_design_row.get("contract_status"))
    input_marker = (
        [f"input_contract_status={input_status}"]
        if input_status != STATUS_EQUATION_AUTHORITY_CONTRACT_CANDIDATE_ONLY
        else []
    )
    sample_record = build_sample_equation_authority_record_from_contract_design_row(contract_design_row)
    return {
        "record_contract_row_id": (
            "equation-authority-record-contract:"
            f"{_safe_text(contract_design_row.get('paper_id'))}:{index:04d}"
        ),
        "candidate_type": "equation_authority_record_contract",
        "contract_design_row_id": _safe_text(contract_design_row.get("contract_row_id")),
        "hash_identity_design_row_id": _safe_text(contract_design_row.get("hash_identity_design_row_id")),
        "readiness_audit_row_id": _safe_text(contract_design_row.get("readiness_audit_row_id")),
        "paper_id": _safe_text(contract_design_row.get("paper_id")),
        "source_candidate_id": _safe_text(contract_design_row.get("source_candidate_id")),
        "source_file": _safe_text(contract_design_row.get("source_file")),
        "equation_environment": _safe_text(contract_design_row.get("equation_environment")),
        "input_contract_status": input_status,
        "input_design_status": _safe_text(contract_design_row.get("input_design_status")),
        "readiness_status": _safe_text(contract_design_row.get("readiness_status")),
        "sourceContentHash": _safe_text(contract_design_row.get("sourceContentHash")),
        "equation_identity": dict(contract_design_row.get("equation_identity") or {}),
        "proposed_identity_design": dict(contract_design_row.get("proposed_identity_design") or {}),
        "proposed_hash_design": dict(contract_design_row.get("proposed_hash_design") or {}),
        "pdf_region_signal": dict(contract_design_row.get("pdf_region_signal") or {}),
        "canonical_text_alignment": dict(contract_design_row.get("canonical_text_alignment") or {}),
        "surrounding_context": dict(contract_design_row.get("surrounding_context") or {}),
        "ambiguity": dict(contract_design_row.get("ambiguity") or {}),
        "raster_image_only_blocker": _safe_bool(contract_design_row.get("raster_image_only_blocker")),
        "planned_record_contract": dict(contract_design_row.get("planned_record_contract") or {}),
        "record_schema_id": EQUATION_AUTHORITY_RECORD_SCHEMA_ID,
        "planned_write_target": EQUATION_AUTHORITY_RECORD_STORE,
        "contract_record_preview": _preview(contract_design_row),
        "equation_authority_record_preview": sample_record,
        "record_contract_status": status,
        "blockers": _dedupe([*blockers, *input_marker, "equation_authority_record_contract_only"]),
        "recommended_action": recommended_action,
        "authorityPolicyCreated": False,
        "equationAuthorityRecordWritten": False,
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


def _count_status(rows: list[dict[str, Any]], status: str) -> int:
    return sum(1 for row in rows if _safe_text(row.get("record_contract_status")) == status)


def _counts(
    rows: list[dict[str, Any]],
    input_blockers: list[dict[str, str]],
    input_rows: int,
    sample_schema_valid: bool,
    sample_semantic_errors: list[str],
) -> dict[str, Any]:
    by_status = Counter(_safe_text(row.get("record_contract_status")) for row in rows)
    input_statuses = Counter(_safe_text(item.get("record_contract_status")) for item in input_blockers)
    for status, count in input_statuses.items():
        by_status[status] += count
    candidate_rows = _count_status(rows, STATUS_EQUATION_AUTHORITY_RECORD_CONTRACT_CANDIDATE_ONLY)
    return {
        "inputRows": input_rows,
        "targetRows": len(rows),
        "equationAuthorityRecordContracts": 1,
        "equationAuthorityRecordSchemas": 1,
        "plannedEquationAuthorityRecordRows": candidate_rows,
        "equationAuthorityRecordContractCandidateOnlyRows": candidate_rows,
        "blockedMissingContractPreviewRows": _count_status(rows, STATUS_BLOCKED_MISSING_CONTRACT_PREVIEW),
        "blockedMissingDesignIdentityRows": _count_status(rows, STATUS_BLOCKED_MISSING_DESIGN_IDENTITY),
        "blockedMissingDesignHashRows": _count_status(rows, STATUS_BLOCKED_MISSING_DESIGN_HASH),
        "blockedMissingSourceHashRows": _count_status(rows, STATUS_BLOCKED_MISSING_SOURCE_HASH),
        "blockedAmbiguousEquationMatchRows": _count_status(rows, STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH),
        "blockedRasterOnlyEquationRows": _count_status(rows, STATUS_BLOCKED_RASTER_ONLY_EQUATION),
        "heldOutNonRecordContractBlockerRows": _count_status(
            rows, STATUS_HELD_OUT_NON_RECORD_CONTRACT_BLOCKER
        ),
        "blockedInputReportMissingRows": int(input_statuses.get(STATUS_BLOCKED_INPUT_REPORT_MISSING, 0)),
        "blockedInputSchemaViolationRows": int(input_statuses.get(STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION, 0)),
        "executorImplementedRows": 0,
        "authorityPolicyCreatedRows": 0,
        "equationAuthorityRecordWrittenRows": 0,
        "equationArtifactCreatedRows": 0,
        "strictEvidenceCreatedRows": 0,
        "sourceSpanMutatedRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "reindexOrReembedRows": 0,
        "vaultScanRows": 0,
        "parserRoutingChangedRows": 0,
        "answerIntegrationChangedRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
        "manifestWriteRows": 0,
        "inputBlockerCount": len(input_blockers),
        "schemaViolationCount": len(input_blockers),
        "sampleRecordSchemaValidRows": 1 if sample_schema_valid else 0,
        "sampleRecordSemanticViolationRows": len(sample_semantic_errors),
        "byRecordContractStatus": dict(by_status),
        "byInputContractStatus": dict(Counter(_safe_text(row.get("input_contract_status")) for row in rows)),
        "byInputDesignStatus": dict(Counter(_safe_text(row.get("input_design_status")) for row in rows)),
        "byReadinessStatus": dict(Counter(_safe_text(row.get("readiness_status")) for row in rows)),
        "byPaper": dict(Counter(_safe_text(row.get("paper_id")) for row in rows)),
        "byEnvironment": dict(Counter(_safe_text(row.get("equation_environment")) for row in rows)),
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
            "recordSchemaContract",
            "writeTargets",
            "warnings",
            "inputBlockers",
        )
        if key in report
    }


def render_equation_authority_record_contract_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    write_target = (report.get("writeTargets") or [{}])[0]
    lines = [
        "# Equation Authority Record Contract",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Target rows: `{int(counts.get('targetRows') or 0)}`",
        f"- Planned record rows: `{int(counts.get('plannedEquationAuthorityRecordRows') or 0)}`",
        f"- Record contract candidate-only rows: `{int(counts.get('equationAuthorityRecordContractCandidateOnlyRows') or 0)}`",
        f"- Blocked missing contract preview rows: `{int(counts.get('blockedMissingContractPreviewRows') or 0)}`",
        f"- Blocked missing design hash rows: `{int(counts.get('blockedMissingDesignHashRows') or 0)}`",
        f"- Blocked ambiguous equation match rows: `{int(counts.get('blockedAmbiguousEquationMatchRows') or 0)}`",
        f"- Input blockers: `{int(counts.get('inputBlockerCount') or 0)}`",
        f"- Planned write target: `{write_target.get('plannedWriteTarget', '')}`",
        f"- Record writes: `{int(counts.get('equationAuthorityRecordWrittenRows') or 0)}`",
        f"- EquationArtifact created rows: `{int(counts.get('equationArtifactCreatedRows') or 0)}`",
        f"- StrictEvidence created rows: `{int(counts.get('strictEvidenceCreatedRows') or 0)}`",
        "",
        "## Contract principles",
        "",
    ]
    lines.extend(f"- {item}" for item in list(report.get("contractPrinciples") or []))
    lines.extend(["", "## Rows", ""])
    for row in list(report.get("rows") or []):
        preview = dict(row.get("equation_authority_record_preview") or {})
        lines.append(
            f"- `{row.get('paper_id')}` `{row.get('record_contract_status')}` "
            f"`{row.get('input_contract_status')}` `{preview.get('equationAuthorityRecordId', '')}`"
        )
    if report.get("inputBlockers"):
        lines.extend(["", "## Input blockers", ""])
        for blocker in list(report.get("inputBlockers") or []):
            lines.append(
                f"- `{blocker.get('record_contract_status')}` "
                f"`{blocker.get('inputName')}` {blocker.get('detail')}"
            )
    return "\n".join(lines)


def write_equation_authority_record_contract_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "equation-authority-record-contract-report.json"
    summary_path = root / "equation-authority-record-contract-summary.json"
    markdown_path = root / "equation-authority-record-contract.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_equation_authority_record_contract_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def build_equation_authority_record_contract(
    equation_authority_contract_design_report: str | Path = DEFAULT_EQUATION_AUTHORITY_CONTRACT_DESIGN_REPORT,
    *,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    report_path = Path(str(equation_authority_contract_design_report)).expanduser()
    requested = [str(item).strip() for item in (paper_ids or []) if str(item).strip()]
    allowed = set(requested)
    report_exists = report_path.is_file()
    contract_report, load_error = _load_json(report_path)
    input_blockers = _input_blockers(report_path, contract_report, load_error)
    input_rows = len([row for row in list(contract_report.get("rows") or []) if isinstance(row, dict)])
    rows: list[dict[str, Any]] = []
    if not input_blockers:
        source_rows = [
            dict(row)
            for row in list(contract_report.get("rows") or [])
            if isinstance(row, dict) and (not allowed or _safe_text(row.get("paper_id")) in allowed)
        ]
        rows = [_row(index, row) for index, row in enumerate(source_rows, start=1)]

    sample_record = next(
        (
            dict(row.get("equation_authority_record_preview") or {})
            for row in rows
            if row.get("record_contract_status") == STATUS_EQUATION_AUTHORITY_RECORD_CONTRACT_CANDIDATE_ONLY
        ),
        {},
    )
    sample_semantic_errors = (
        validate_equation_authority_record_semantics(sample_record) if sample_record else []
    )
    sample_schema_valid = False
    if sample_record:
        sample_schema_valid = validate_payload(
            sample_record,
            EQUATION_AUTHORITY_RECORD_SCHEMA_ID,
            strict=True,
        ).ok

    counts = _counts(rows, input_blockers, input_rows, sample_schema_valid, sample_semantic_errors)
    status = "ok" if rows and not input_blockers else "blocked"
    decision = (
        "equation_authority_record_contract_ready"
        if status == "ok"
        else STATUS_BLOCKED_INPUT_REPORT_MISSING
        if any(item.get("record_contract_status") == STATUS_BLOCKED_INPUT_REPORT_MISSING for item in input_blockers)
        else STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION
        if input_blockers
        else "blocked_no_equation_authority_contract_design_rows"
    )
    warnings = _dedupe(
        [
            "equation authority record contract is report-only and contract-only",
            "sample equation authority record is a schema preview, not a written record",
            "no authority policy, EquationArtifact, StrictEvidence, SourceSpan, DB/index, vault, parser routing, or answer integration mutation occurs",
            *[item.get("detail", "") for item in input_blockers],
            *sample_semantic_errors,
        ]
    )
    return {
        "schema": EQUATION_AUTHORITY_RECORD_CONTRACT_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "input": {
            "paperIds": requested,
            "equationAuthorityContractDesignReportPath": str(report_path),
            "equationAuthorityContractDesignReportSchema": _safe_text(contract_report.get("schema")),
            "equationAuthorityContractDesignReportExists": report_exists,
            "equationAuthorityContractDesignReportStrictSchemaValid": not input_blockers,
        },
        "counts": counts,
        "gate": {
            "contractDesignRows": bool(rows),
            "recordContractRows": bool(rows),
            "writeTargetContractsDefined": True,
            "equationAuthorityRecordSchemaDefined": True,
            "executorReady": False,
            "equationAuthorityRecordWriteReady": False,
            "equationAuthorityRecordWriteAllowed": False,
            "equationAuthorityPromotionReady": False,
            "equationArtifactCreationReady": False,
            "strictEvidenceReady": False,
            "sourceSpanMutationReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": decision,
            "schemaViolations": [item.get("detail", "") for item in input_blockers],
            "sampleRecordSemanticViolations": sample_semantic_errors,
            "recommendedNextTranche": (
                "equation_authority_record_executor_dry_run"
                if status == "ok"
                else "equation_authority_record_contract_input_repair"
            ),
            "inputBlockers": input_blockers,
        },
        "policy": dict(NO_MUTATION_POLICY),
        "contractPrinciples": [
            "equation authority records are future append-only candidate metadata",
            "contract rows preserve identity, hash, sourceContentHash, PDF region, and ambiguity provenance",
            "EquationArtifact and StrictEvidence remain separate later gates",
            "SourceSpan rows remain immutable under this contract",
            "runtime and answer integration stay false until a later explicit gate",
        ],
        "recordSchemaContract": _record_schema_contract(),
        "writeTargets": [dict(EQUATION_AUTHORITY_RECORD_STORE_CONTRACT)],
        "sampleEquationAuthorityRecord": sample_record,
        "warnings": warnings,
        "inputBlockers": input_blockers,
        "rows": rows,
    }


def _parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--equation-authority-contract-design-report",
        default=str(DEFAULT_EQUATION_AUTHORITY_CONTRACT_DESIGN_REPORT),
        help="Path to an existing equation authority contract design report.",
    )
    parser.add_argument(
        "--paper-id",
        dest="paper_ids",
        action="append",
        default=[],
        help="Optional paper id filter. May be supplied multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_EQUATION_AUTHORITY_RECORD_CONTRACT_OUTPUT_DIR),
        help="Directory for report, summary, and Markdown outputs.",
    )
    parser.add_argument("--json", action="store_true", help="Print output paths as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_equation_authority_record_contract(
        equation_authority_contract_design_report=args.equation_authority_contract_design_report,
        paper_ids=args.paper_ids,
    )
    paths = write_equation_authority_record_contract_reports(report, args.output_dir)
    if args.json:
        print(json.dumps({"status": report["status"], "paths": paths}, ensure_ascii=False, indent=2))
    else:
        print(paths["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_EQUATION_AUTHORITY_CONTRACT_DESIGN_REPORT",
    "DEFAULT_EQUATION_AUTHORITY_RECORD_CONTRACT_OUTPUT_DIR",
    "EQUATION_AUTHORITY_RECORD_CONTRACT_SCHEMA_ID",
    "EQUATION_AUTHORITY_RECORD_CONTRACT_VERSION",
    "EQUATION_AUTHORITY_RECORD_DECISION",
    "EQUATION_AUTHORITY_RECORD_SCHEMA_ID",
    "EQUATION_AUTHORITY_RECORD_STATE",
    "EQUATION_AUTHORITY_RECORD_STORE",
    "EQUATION_AUTHORITY_RECORD_STORE_CONTRACT",
    "KNOWN_WRITE_TARGET_CONTRACTS",
    "STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH",
    "STATUS_BLOCKED_INPUT_REPORT_MISSING",
    "STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION",
    "STATUS_BLOCKED_MISSING_CONTRACT_PREVIEW",
    "STATUS_BLOCKED_MISSING_DESIGN_HASH",
    "STATUS_BLOCKED_MISSING_DESIGN_IDENTITY",
    "STATUS_BLOCKED_MISSING_SOURCE_HASH",
    "STATUS_BLOCKED_RASTER_ONLY_EQUATION",
    "STATUS_EQUATION_AUTHORITY_RECORD_CONTRACT_CANDIDATE_ONLY",
    "STATUS_HELD_OUT_NON_RECORD_CONTRACT_BLOCKER",
    "build_equation_authority_record_contract",
    "build_sample_equation_authority_record_from_contract_design_row",
    "render_equation_authority_record_contract_markdown",
    "validate_equation_authority_record_semantics",
    "write_equation_authority_record_contract_reports",
]
