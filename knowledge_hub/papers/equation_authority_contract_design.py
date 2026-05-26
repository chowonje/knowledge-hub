"""Report-only equation authority contract design.

This helper consumes the Equation Authority hash/identity design report and
defines future contract fields for equation authority records. It is contract
design only: no EquationArtifact, StrictEvidence, SourceSpan, DB/index state,
vault content, parser routing, answer integration, or authority policy is
created.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.equation_authority_hash_identity_design import (
    EQUATION_AUTHORITY_HASH_IDENTITY_DESIGN_SCHEMA_ID,
    STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH as HASH_STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH,
    STATUS_BLOCKED_MISSING_EQUATION_IDENTITY_SIGNAL as HASH_STATUS_BLOCKED_MISSING_EQUATION_IDENTITY_SIGNAL,
    STATUS_BLOCKED_MISSING_SOURCE_HASH as HASH_STATUS_BLOCKED_MISSING_SOURCE_HASH,
    STATUS_BLOCKED_MISSING_TEX_OR_MATHML_TEXT as HASH_STATUS_BLOCKED_MISSING_TEX_OR_MATHML_TEXT,
    STATUS_BLOCKED_RASTER_ONLY_EQUATION as HASH_STATUS_BLOCKED_RASTER_ONLY_EQUATION,
    STATUS_HASH_IDENTITY_DESIGN_CANDIDATE_ONLY as HASH_STATUS_CANDIDATE_ONLY,
)


EQUATION_AUTHORITY_CONTRACT_DESIGN_SCHEMA_ID = (
    "knowledge-hub.paper.equation-authority-contract-design.v1"
)
FUTURE_EQUATION_AUTHORITY_RECORD_SCHEMA_ID = "knowledge-hub.paper.equation-authority-record.v1"
FUTURE_EQUATION_AUTHORITY_STORE = "equation_authority_record_candidate_store"
CONTRACT_VERSION = "equation_authority_contract_design_v1"

DEFAULT_EQUATION_AUTHORITY_HASH_IDENTITY_DESIGN_REPORT = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "equation-authority-hash-identity-design"
    / "equation-authority-hash-identity-design-report.json"
)

DEFAULT_EQUATION_AUTHORITY_CONTRACT_DESIGN_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "equation-authority-contract-design"
)

STATUS_EQUATION_AUTHORITY_CONTRACT_CANDIDATE_ONLY = "equation_authority_contract_candidate_only"
STATUS_BLOCKED_MISSING_DESIGN_IDENTITY = "blocked_missing_design_identity"
STATUS_BLOCKED_MISSING_DESIGN_HASH = "blocked_missing_design_hash"
STATUS_BLOCKED_MISSING_SOURCE_HASH = "blocked_missing_source_hash"
STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH = "blocked_ambiguous_equation_match"
STATUS_BLOCKED_RASTER_ONLY_EQUATION = "blocked_raster_only_equation"
STATUS_HELD_OUT_NON_CONTRACT_DESIGN_BLOCKER = "held_out_non_contract_design_blocker"
STATUS_BLOCKED_INPUT_REPORT_MISSING = "blocked_input_report_missing"
STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION = "blocked_input_schema_violation"

REQUIRED_RECORD_FIELDS = [
    "schema",
    "equationAuthorityRecordId",
    "paperId",
    "sourceCandidateId",
    "sourceTexRowId",
    "sourceContentHash",
    "equationIdentityDigestSha256",
    "equationHashSha256",
    "equationTextNormalization",
    "hashSource",
    "readinessAuditRowId",
    "hashIdentityDesignRowId",
    "pdfRegionSnapshot",
    "canonicalAlignmentSnapshot",
    "ambiguitySnapshot",
    "provenanceTrace",
    "authorityState",
    "runtimeVisible",
    "answerIntegrationVisible",
    "equationArtifactCreated",
    "strictEvidenceCreated",
    "idempotencyKey",
    "writePolicy",
]

IDEMPOTENCY_KEY_FIELDS = [
    "paperId",
    "sourceContentHash",
    "equationIdentityDigestSha256",
    "equationHashSha256",
    "readinessAuditRowId",
    "hashIdentityDesignRowId",
    "contractVersion",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_bool(value: Any) -> bool:
    return bool(value)


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


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return slug.strip("-") or "unknown-paper"


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
                "inputName": "equationAuthorityHashIdentityDesign",
                "inputLabel": "Equation Authority hash/identity design report",
                "path": str(report_path),
                "contract_status": STATUS_BLOCKED_INPUT_REPORT_MISSING,
                "detail": "expected input report is missing",
            }
        ]
    if load_error:
        return [
            {
                "inputName": "equationAuthorityHashIdentityDesign",
                "inputLabel": "Equation Authority hash/identity design report",
                "path": str(report_path),
                "contract_status": STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION,
                "detail": "expected input report is unreadable or not a JSON object",
            }
        ]
    schema = _safe_text(report.get("schema"))
    if schema != EQUATION_AUTHORITY_HASH_IDENTITY_DESIGN_SCHEMA_ID:
        return [
            {
                "inputName": "equationAuthorityHashIdentityDesign",
                "inputLabel": "Equation Authority hash/identity design report",
                "path": str(report_path),
                "contract_status": STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION,
                "detail": (
                    "schema mismatch: expected "
                    f"{EQUATION_AUTHORITY_HASH_IDENTITY_DESIGN_SCHEMA_ID}, got {schema or 'missing'}"
                ),
            }
        ]
    validation = validate_payload(report, EQUATION_AUTHORITY_HASH_IDENTITY_DESIGN_SCHEMA_ID, strict=True)
    if validation.ok:
        return []
    return [
        {
            "inputName": "equationAuthorityHashIdentityDesign",
            "inputLabel": "Equation Authority hash/identity design report",
            "path": str(report_path),
            "contract_status": STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION,
            "detail": f"schema validation failed: {error}",
        }
        for error in validation.errors
    ]


def _identity_digest(row: dict[str, Any]) -> str:
    identity = dict(row.get("proposed_identity_design") or {})
    return _safe_text(identity.get("identityDigestSha256"))


def _selected_hash(row: dict[str, Any]) -> str:
    hash_design = dict(row.get("proposed_hash_design") or {})
    return _safe_text(hash_design.get("selectedDesignHash"))


def _contract_id(row: dict[str, Any]) -> str:
    paper_id = _safe_text(row.get("paper_id"))
    source_hash = _safe_text(row.get("sourceContentHash"))
    digest = _sha256_json(
        {
            "paperId": paper_id,
            "sourceContentHash": source_hash,
            "identityDigest": _identity_digest(row),
            "equationHash": _selected_hash(row),
            "hashIdentityDesignRowId": _safe_text(row.get("design_row_id")),
            "contractVersion": CONTRACT_VERSION,
        }
    )
    return f"eqauth-contract:{_slug(paper_id)}:{digest[:20]}"


def _idempotency_key(row: dict[str, Any], contract_id: str) -> str:
    return _sha256_json(
        {
            "contractId": contract_id,
            "paperId": _safe_text(row.get("paper_id")),
            "sourceContentHash": _safe_text(row.get("sourceContentHash")),
            "identityDigest": _identity_digest(row),
            "equationHash": _selected_hash(row),
            "hashIdentityDesignRowId": _safe_text(row.get("design_row_id")),
            "contractVersion": CONTRACT_VERSION,
        }
    )


def _classify(row: dict[str, Any]) -> tuple[str, list[str], str]:
    design_status = _safe_text(row.get("design_status"))
    identity_digest = _identity_digest(row)
    selected_hash = _selected_hash(row)
    source_hash = _safe_text(row.get("sourceContentHash"))
    identity = dict(row.get("equation_identity") or {})
    if not _safe_bool(identity.get("available")) or not identity_digest:
        return (
            STATUS_BLOCKED_MISSING_DESIGN_IDENTITY,
            ["equation_identity_design_digest_missing"],
            "recover_hash_identity_design_identity_before_contract_design",
        )
    if not source_hash:
        return (
            STATUS_BLOCKED_MISSING_SOURCE_HASH,
            ["sourceContentHash_missing"],
            "recover_source_content_hash_before_equation_authority_contract_design",
        )
    if _safe_bool(row.get("raster_image_only_blocker")) or design_status == HASH_STATUS_BLOCKED_RASTER_ONLY_EQUATION:
        return (
            STATUS_BLOCKED_RASTER_ONLY_EQUATION,
            ["raster_or_image_only_equation_signal"],
            "route_to_image_or_ocr_equation_extractor_before_contract_design",
        )
    if design_status == HASH_STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH:
        return (
            STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH,
            _string_list(dict(row.get("ambiguity") or {}).get("reasons")) or ["equation_match_non_unique"],
            "resolve_non_unique_equation_match_before_contract_design",
        )
    if not selected_hash:
        return (
            STATUS_BLOCKED_MISSING_DESIGN_HASH,
            ["equation_hash_design_missing"],
            "recover_tex_or_mathml_hash_design_before_contract_design",
        )
    if design_status == HASH_STATUS_BLOCKED_MISSING_EQUATION_IDENTITY_SIGNAL:
        return (
            STATUS_BLOCKED_MISSING_DESIGN_IDENTITY,
            ["input_design_status=blocked_missing_equation_identity_signal"],
            "recover_hash_identity_design_identity_before_contract_design",
        )
    if design_status == HASH_STATUS_BLOCKED_MISSING_SOURCE_HASH:
        return (
            STATUS_BLOCKED_MISSING_SOURCE_HASH,
            ["input_design_status=blocked_missing_source_hash"],
            "recover_source_content_hash_before_equation_authority_contract_design",
        )
    if design_status == HASH_STATUS_BLOCKED_MISSING_TEX_OR_MATHML_TEXT:
        return (
            STATUS_BLOCKED_MISSING_DESIGN_HASH,
            ["input_design_status=blocked_missing_tex_or_mathml_text"],
            "recover_tex_or_mathml_hash_design_before_contract_design",
        )
    if design_status != HASH_STATUS_CANDIDATE_ONLY:
        return (
            STATUS_HELD_OUT_NON_CONTRACT_DESIGN_BLOCKER,
            [f"input_design_status={design_status or 'missing'}"],
            "resolve_hash_identity_design_blocker_before_contract_design",
        )
    return (
        STATUS_EQUATION_AUTHORITY_CONTRACT_CANDIDATE_ONLY,
        ["equation_authority_contract_design_only"],
        "queue_for_later_explicit_equation_authority_record_contract_or_executor",
    )


def _planned_contract() -> dict[str, Any]:
    return {
        "contractOnly": True,
        "futureRecordSchema": FUTURE_EQUATION_AUTHORITY_RECORD_SCHEMA_ID,
        "plannedWriteTarget": FUTURE_EQUATION_AUTHORITY_STORE,
        "storeKind": "future_local_papers_dir_jsonl_equation_authority_store",
        "storeRootTemplate": "{papers_dir}/structured_evidence/equation_authority",
        "recordPathTemplate": "{papers_dir}/structured_evidence/equation_authority/{paper_id}.jsonl",
        "requiredRecordFields": REQUIRED_RECORD_FIELDS,
        "idempotencyKeyFields": IDEMPOTENCY_KEY_FIELDS,
        "readbackChecks": [
            "record_schema_validates",
            "idempotency_key_stable",
            "sourceContentHash_preserved",
            "equation_identity_digest_preserved",
            "equation_hash_preserved",
            "hash_identity_design_row_resolves",
            "readiness_audit_row_resolves",
            "no_equation_artifact_created",
            "no_strict_evidence_created",
            "runtime_visible_false_until_later_gate",
            "answer_integration_visible_false_until_later_gate",
        ],
        "executorImplemented": False,
        "rollbackImplemented": False,
        "runtimeUseAllowed": False,
        "parserRoutingAllowed": False,
        "answerIntegrationAllowed": False,
        "databaseMutationAllowed": False,
        "vaultScanAllowed": False,
    }


def _record_preview(row: dict[str, Any], contract_id: str) -> dict[str, Any]:
    identity = dict(row.get("equation_identity") or {})
    hash_design = dict(row.get("proposed_hash_design") or {})
    return {
        "schema": FUTURE_EQUATION_AUTHORITY_RECORD_SCHEMA_ID,
        "equationAuthorityRecordId": contract_id,
        "paperId": _safe_text(row.get("paper_id")),
        "sourceCandidateId": _safe_text(row.get("source_candidate_id")),
        "sourceTexRowId": _safe_text(identity.get("sourceTexRowId")),
        "sourceContentHash": _safe_text(row.get("sourceContentHash")),
        "equationIdentityDigestSha256": _identity_digest(row),
        "equationHashSha256": _selected_hash(row),
        "equationTextNormalization": _safe_text(hash_design.get("textNormalization")),
        "hashSource": _safe_text(hash_design.get("hashSource")),
        "readinessAuditRowId": _safe_text(row.get("readiness_audit_row_id")),
        "hashIdentityDesignRowId": _safe_text(row.get("design_row_id")),
        "pdfRegionSnapshot": dict(row.get("pdf_region_signal") or {}),
        "canonicalAlignmentSnapshot": dict(row.get("canonical_text_alignment") or {}),
        "ambiguitySnapshot": dict(row.get("ambiguity") or {}),
        "provenanceTrace": {
            "hashIdentityDesignRowId": _safe_text(row.get("design_row_id")),
            "readinessAuditRowId": _safe_text(row.get("readiness_audit_row_id")),
            "sourceCandidateId": _safe_text(row.get("source_candidate_id")),
        },
        "authorityState": "contract_design_candidate_only",
        "runtimeVisible": False,
        "answerIntegrationVisible": False,
        "equationArtifactCreated": False,
        "strictEvidenceCreated": False,
        "idempotencyKey": _idempotency_key(row, contract_id),
        "writePolicy": {
            "recordWrite": False,
            "equationArtifactCreation": False,
            "strictEvidenceCreation": False,
            "sourceSpanMutation": False,
            "runtimeVisibility": False,
            "answerIntegrationVisibility": False,
        },
    }


def _row(index: int, row: dict[str, Any]) -> dict[str, Any]:
    contract_status, blockers, recommended_action = _classify(row)
    contract_id = _contract_id(row)
    input_design_status = _safe_text(row.get("design_status"))
    input_marker = [f"input_design_status={input_design_status}"] if input_design_status != HASH_STATUS_CANDIDATE_ONLY else []
    return {
        "contract_row_id": f"equation-authority-contract-design:{_safe_text(row.get('paper_id'))}:{index:04d}",
        "candidate_type": "equation_authority_contract_design",
        "hash_identity_design_row_id": _safe_text(row.get("design_row_id")),
        "readiness_audit_row_id": _safe_text(row.get("readiness_audit_row_id")),
        "paper_id": _safe_text(row.get("paper_id")),
        "source_candidate_id": _safe_text(row.get("source_candidate_id")),
        "source_file": _safe_text(row.get("source_file")),
        "equation_environment": _safe_text(row.get("equation_environment")),
        "input_design_status": input_design_status,
        "readiness_status": _safe_text(row.get("readiness_status")),
        "sourceContentHash": _safe_text(row.get("sourceContentHash")),
        "equation_identity": dict(row.get("equation_identity") or {}),
        "proposed_identity_design": dict(row.get("proposed_identity_design") or {}),
        "proposed_hash_design": dict(row.get("proposed_hash_design") or {}),
        "pdf_region_signal": dict(row.get("pdf_region_signal") or {}),
        "canonical_text_alignment": dict(row.get("canonical_text_alignment") or {}),
        "surrounding_context": dict(row.get("surrounding_context") or {}),
        "ambiguity": dict(row.get("ambiguity") or {}),
        "raster_image_only_blocker": _safe_bool(row.get("raster_image_only_blocker")),
        "planned_record_contract": _planned_contract(),
        "contract_record_preview": _record_preview(row, contract_id),
        "contract_status": contract_status,
        "blockers": _dedupe([*blockers, *input_marker, "equation_authority_contract_design_only"]),
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
    return sum(1 for row in rows if _safe_text(row.get("contract_status")) == status)


def _counts(rows: list[dict[str, Any]], input_blockers: list[dict[str, str]], input_rows: int) -> dict[str, Any]:
    by_status = Counter(_safe_text(row.get("contract_status")) for row in rows)
    input_statuses = Counter(_safe_text(item.get("contract_status")) for item in input_blockers)
    for status, count in input_statuses.items():
        by_status[status] += count
    return {
        "inputRows": input_rows,
        "targetRows": len(rows),
        "equationAuthorityContractCandidateOnlyRows": _count_status(
            rows, STATUS_EQUATION_AUTHORITY_CONTRACT_CANDIDATE_ONLY
        ),
        "blockedMissingDesignIdentityRows": _count_status(rows, STATUS_BLOCKED_MISSING_DESIGN_IDENTITY),
        "blockedMissingDesignHashRows": _count_status(rows, STATUS_BLOCKED_MISSING_DESIGN_HASH),
        "blockedMissingSourceHashRows": _count_status(rows, STATUS_BLOCKED_MISSING_SOURCE_HASH),
        "blockedAmbiguousEquationMatchRows": _count_status(rows, STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH),
        "blockedRasterOnlyEquationRows": _count_status(rows, STATUS_BLOCKED_RASTER_ONLY_EQUATION),
        "heldOutNonContractDesignBlockerRows": _count_status(rows, STATUS_HELD_OUT_NON_CONTRACT_DESIGN_BLOCKER),
        "blockedInputReportMissingRows": int(input_statuses.get(STATUS_BLOCKED_INPUT_REPORT_MISSING, 0)),
        "blockedInputSchemaViolationRows": int(input_statuses.get(STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION, 0)),
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
        "inputBlockerCount": len(input_blockers),
        "byContractStatus": dict(by_status),
        "byInputDesignStatus": dict(Counter(_safe_text(row.get("input_design_status")) for row in rows)),
        "byReadinessStatus": dict(Counter(_safe_text(row.get("readiness_status")) for row in rows)),
        "byPaper": dict(Counter(_safe_text(row.get("paper_id")) for row in rows)),
        "byEnvironment": dict(Counter(_safe_text(row.get("equation_environment")) for row in rows)),
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "input", "counts", "gate", "policy", "warnings", "inputBlockers")
        if key in report
    }


def render_equation_authority_contract_design_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Equation Authority Contract Design",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Target rows: `{int(counts.get('targetRows') or 0)}`",
        f"- Contract candidate-only rows: `{int(counts.get('equationAuthorityContractCandidateOnlyRows') or 0)}`",
        f"- Blocked missing design identity rows: `{int(counts.get('blockedMissingDesignIdentityRows') or 0)}`",
        f"- Blocked missing design hash rows: `{int(counts.get('blockedMissingDesignHashRows') or 0)}`",
        f"- Blocked missing source hash rows: `{int(counts.get('blockedMissingSourceHashRows') or 0)}`",
        f"- Blocked ambiguous equation match rows: `{int(counts.get('blockedAmbiguousEquationMatchRows') or 0)}`",
        f"- Blocked raster/image-only rows: `{int(counts.get('blockedRasterOnlyEquationRows') or 0)}`",
        f"- Input blockers: `{int(counts.get('inputBlockerCount') or 0)}`",
        f"- Equation authority record writes: `{int(counts.get('equationAuthorityRecordWrittenRows') or 0)}`",
        f"- EquationArtifact created rows: `{int(counts.get('equationArtifactCreatedRows') or 0)}`",
        f"- StrictEvidence created rows: `{int(counts.get('strictEvidenceCreatedRows') or 0)}`",
        "",
        "## Rows",
        "",
    ]
    for row in list(report.get("rows") or []):
        preview = dict(row.get("contract_record_preview") or {})
        lines.append(
            f"- `{row.get('paper_id')}` `{row.get('contract_status')}` "
            f"`{row.get('input_design_status')}` `{preview.get('equationAuthorityRecordId', '')}`"
        )
    if report.get("inputBlockers"):
        lines.extend(["", "## Input blockers", ""])
        for blocker in list(report.get("inputBlockers") or []):
            lines.append(
                f"- `{blocker.get('contract_status')}` `{blocker.get('inputName')}` {blocker.get('detail')}"
            )
    return "\n".join(lines)


def write_equation_authority_contract_design_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "equation-authority-contract-design-report.json"
    summary_path = root / "equation-authority-contract-design-summary.json"
    markdown_path = root / "equation-authority-contract-design.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_equation_authority_contract_design_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def build_equation_authority_contract_design(
    equation_authority_hash_identity_design_report: str | Path = DEFAULT_EQUATION_AUTHORITY_HASH_IDENTITY_DESIGN_REPORT,
    *,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    report_path = Path(str(equation_authority_hash_identity_design_report)).expanduser()
    requested = [str(item).strip() for item in (paper_ids or []) if str(item).strip()]
    allowed = set(requested)
    report_exists = report_path.is_file()
    hash_report, load_error = _load_json(report_path)
    input_blockers = _input_blockers(report_path, hash_report, load_error)
    input_rows = len([row for row in list(hash_report.get("rows") or []) if isinstance(row, dict)])
    rows: list[dict[str, Any]] = []
    if not input_blockers:
        source_rows = [
            dict(row)
            for row in list(hash_report.get("rows") or [])
            if isinstance(row, dict) and (not allowed or _safe_text(row.get("paper_id")) in allowed)
        ]
        rows = [_row(index, row) for index, row in enumerate(source_rows, start=1)]
    counts = _counts(rows, input_blockers, input_rows)
    status = "ok" if rows and not input_blockers else "blocked"
    decision = (
        "equation_authority_contract_design_ready"
        if status == "ok"
        else STATUS_BLOCKED_INPUT_REPORT_MISSING
        if any(item.get("contract_status") == STATUS_BLOCKED_INPUT_REPORT_MISSING for item in input_blockers)
        else STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION
        if input_blockers
        else "blocked_no_hash_identity_design_rows"
    )
    warnings = _dedupe(
        [
            "equation authority contract design is report-only and contract-only",
            "planned record fields are a future contract preview, not a written record",
            "no authority policy, EquationArtifact, StrictEvidence, SourceSpan, DB/index, vault, parser routing, or answer integration mutation occurs",
            *[item.get("detail", "") for item in input_blockers],
        ]
    )
    return {
        "schema": EQUATION_AUTHORITY_CONTRACT_DESIGN_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "input": {
            "paperIds": requested,
            "equationAuthorityHashIdentityDesignReportPath": str(report_path),
            "equationAuthorityHashIdentityDesignReportSchema": _safe_text(hash_report.get("schema")),
            "equationAuthorityHashIdentityDesignReportExists": report_exists,
            "equationAuthorityHashIdentityDesignReportStrictSchemaValid": not input_blockers,
        },
        "counts": counts,
        "gate": {
            "contractDesignRows": bool(rows),
            "equationAuthorityRecordContractReady": False,
            "equationAuthorityRecordWriteReady": False,
            "equationAuthorityPromotionReady": False,
            "equationArtifactCreationReady": False,
            "strictEvidenceReady": False,
            "sourceSpanMutationReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": decision,
            "recommendedNextTranche": "equation_authority_record_contract",
            "inputBlockers": input_blockers,
        },
        "policy": {
            "reportOnly": True,
            "contractOnly": True,
            "classificationOnly": True,
            "authorityPolicyCreated": False,
            "equationAuthorityRecordWritten": False,
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
        "warnings": warnings,
        "inputBlockers": input_blockers,
        "rows": rows,
    }


def _parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--equation-authority-hash-identity-design-report",
        default=str(DEFAULT_EQUATION_AUTHORITY_HASH_IDENTITY_DESIGN_REPORT),
        help="Path to an existing equation authority hash/identity design report.",
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
        default=str(DEFAULT_EQUATION_AUTHORITY_CONTRACT_DESIGN_OUTPUT_DIR),
        help="Directory for report, summary, and Markdown outputs.",
    )
    parser.add_argument("--json", action="store_true", help="Print output paths as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_equation_authority_contract_design(
        equation_authority_hash_identity_design_report=args.equation_authority_hash_identity_design_report,
        paper_ids=args.paper_ids,
    )
    paths = write_equation_authority_contract_design_reports(report, args.output_dir)
    if args.json:
        print(json.dumps({"status": report["status"], "paths": paths}, ensure_ascii=False, indent=2))
    else:
        print(paths["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
