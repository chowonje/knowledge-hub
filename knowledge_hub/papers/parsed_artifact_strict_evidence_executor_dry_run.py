"""Dry-run planner for parsed-artifact StrictEvidence promotion.

Consumes design packet review rows and the StrictEvidence record contract,
plans in-memory StrictEvidence records, and validates schema plus hash/normalization
semantics with zero filesystem writes.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Callable

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_original_source_offset_authority_design import (
    CHARS_BASIS as OFFSET_CHARS_BASIS,
    CHARS_NORMALIZATION as OFFSET_CHARS_NORMALIZATION,
    DEFAULT_PAPERS_DIR,
    _PaperSourceContext,
    _resolve_paper_source_context,
)
from knowledge_hub.papers.parsed_artifact_source_span_strict_evidence_design_packet_review import (
    PACKET_REVIEW_STATUS_READY,
    PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_PACKET_REVIEW_SCHEMA_ID,
)
from knowledge_hub.papers.parsed_artifact_strict_evidence_record_contract import (
    EXECUTOR_BLOCKER_NORMALIZATION_MISMATCH,
    PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_CONTRACT_SCHEMA_ID,
    PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_SCHEMA_ID,
    PARSED_ARTIFACT_STRICT_EVIDENCE_STORE,
    build_sample_strict_evidence_record_from_packet_row,
    validate_strict_evidence_record_semantics,
)
from knowledge_hub.papers.sectionspan_pdf_offset_recovery_dry_run import _extract_pdf_pages


PARSED_ARTIFACT_STRICT_EVIDENCE_EXECUTOR_DRY_RUN_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-strict-evidence-executor-dry-run.v1"
)

DRY_RUN_STATUS_READY = "dry_run_ready_strict_evidence_record_only"
DRY_RUN_STATUS_BLOCKED_SOURCE_TEXT = "blocked_source_text_unavailable"
DRY_RUN_STATUS_BLOCKED_NORM_MISMATCH = "blocked_normalization_hash_contract_mismatch"
DRY_RUN_STATUS_BLOCKED_SCHEMA = "blocked_planned_record_schema_violation"
DRY_RUN_STATUS_BLOCKED_SEMANTIC = "blocked_planned_record_semantic_violation"
DRY_RUN_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"

RECOMMENDED_ACTION_READY = "queue_for_strict_evidence_executor_apply"
RECOMMENDED_ACTION_BLOCKED = "hold_strict_evidence_row_until_blockers_resolved"
RECOMMENDED_ACTION_REPAIR = "repair_input_reports_before_strict_evidence_executor_dry_run"

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

DEFAULT_CONTRACT_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-strict-evidence-record-contract"
    / "01-parsed-artifact-strict-evidence-record-contract"
    / "parsed-artifact-strict-evidence-record-contract.json"
)

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-strict-evidence-executor-dry-run"
    / "01-parsed-artifact-strict-evidence-executor-dry-run"
)

DEFAULT_REPAIRED_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-strict-evidence-executor-dry-run"
    / "02-parsed-artifact-strict-evidence-executor-dry-run-repaired-design-packet"
)

DEFAULT_NORMALIZATION_HASH_REPAIR_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-strict-evidence-normalization-hash-repair"
    / "01-parsed-artifact-strict-evidence-normalization-hash-repair"
    / "parsed-artifact-strict-evidence-normalization-hash-repair.json"
)

PARSED_ARTIFACT_STRICT_EVIDENCE_NORMALIZATION_HASH_REPAIR_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-strict-evidence-normalization-hash-repair.v1"
)

REPAIR_STATUS_CANDIDATE = "normalization_hash_repair_candidate_only"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = _safe_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    payload_path = Path(str(path)).expanduser()
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _apply_normalization_hash_repair_to_packet_report(
    packet_report: dict[str, Any],
    repair_report: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    repair_by_packet_id = {
        _safe_text(row.get("packet_review_row_id")): row
        for row in repair_report.get("rows", [])
        if isinstance(row, dict)
        and row.get("repair_status") == REPAIR_STATUS_CANDIDATE
        and isinstance(row.get("repairedPacketRow"), dict)
    }
    repaired = deepcopy(packet_report)
    repaired_rows: list[dict[str, Any]] = []
    applied = 0
    for row in repaired.get("packetRows", []):
        if not isinstance(row, dict):
            continue
        packet_review_row_id = _safe_text(row.get("packet_review_row_id"))
        repair_row = repair_by_packet_id.get(packet_review_row_id)
        if repair_row:
            repaired_rows.append(deepcopy(repair_row["repairedPacketRow"]))
            applied += 1
        else:
            repaired_rows.append(deepcopy(row))
    repaired["packetRows"] = repaired_rows
    repaired["rows"] = repaired_rows
    repaired["normalizationHashRepairApplied"] = applied > 0
    repaired["normalizationHashRepairAppliedRows"] = applied
    return repaired, applied


def normalize_nfkc_whitespace_casefold_v1(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    collapsed = re.sub(r"\s+", " ", normalized).strip()
    return collapsed.casefold()


def compute_raw_utf8_slice_sha256(canonical_text: str, start: int, end: int) -> str:
    return hashlib.sha256(canonical_text[start:end].encode("utf-8")).hexdigest()


def compute_contract_substring_sha256(
    canonical_text: str,
    start: int,
    end: int,
    *,
    normalization: str,
) -> str:
    slice_text = canonical_text[start:end]
    if normalization == OFFSET_CHARS_NORMALIZATION:
        slice_text = normalize_nfkc_whitespace_casefold_v1(slice_text)
    else:
        raise ValueError(f"unsupported_normalization:{normalization}")
    return hashlib.sha256(slice_text.encode("utf-8")).hexdigest()


def _write_matrix() -> dict[str, Any]:
    return {
        "plannedWriteTarget": PARSED_ARTIFACT_STRICT_EVIDENCE_STORE,
        "writeEnabled": False,
        "strictEvidenceStoreWrite": False,
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


def _hash_verification(
    *,
    canonical_text: str,
    start: int,
    end: int,
    normalization: str,
    expected_hash: str,
) -> tuple[str, dict[str, Any]]:
    details: dict[str, Any] = {
        "expectedSubstringSha256": expected_hash,
        "normalization": normalization,
    }
    try:
        contract_hash = compute_contract_substring_sha256(
            canonical_text,
            start,
            end,
            normalization=normalization,
        )
        raw_hash = compute_raw_utf8_slice_sha256(canonical_text, start, end)
    except Exception as exc:
        details["hashError"] = str(exc)
        return DRY_RUN_STATUS_BLOCKED_SOURCE_TEXT, details

    details["contractSubstringSha256"] = contract_hash
    details["rawUtf8SliceSubstringSha256"] = raw_hash

    if contract_hash == expected_hash:
        details["hashMatchMethod"] = "contract_normalization"
        return DRY_RUN_STATUS_READY, details

    if raw_hash == expected_hash:
        details["hashMatchMethod"] = "legacy_raw_utf8_slice_only"
        details["normalizationContractMismatch"] = True
        return DRY_RUN_STATUS_BLOCKED_NORM_MISMATCH, details

    details["hashMatchMethod"] = "none"
    return DRY_RUN_STATUS_BLOCKED_NORM_MISMATCH, details


def _classify_packet_row(
    packet_row: dict[str, Any],
    *,
    paper_contexts: dict[str, _PaperSourceContext],
    parsed_root: Path,
    papers_dir: Path,
    page_loader: Callable[[str | Path], list[dict[str, Any]]],
    design_packet_review_report_path: str,
    run_id: str,
) -> dict[str, Any]:
    blockers: list[str] = ["strict_evidence_executor_dry_run_only"]
    proposed = packet_row.get("proposed_chars")
    proposed = proposed if isinstance(proposed, dict) else {}

    paper_id = _safe_text(packet_row.get("paper_id"))
    start = _safe_int(proposed.get("start"))
    end = _safe_int(proposed.get("end"))
    basis = _safe_text(proposed.get("basis"))
    normalization = _safe_text(proposed.get("normalization"))
    expected_hash = _safe_text(proposed.get("expectedSubstringSha256"))
    source_content_hash = _safe_text(packet_row.get("sourceContentHash"))

    dry_run_status = DRY_RUN_STATUS_READY
    recommended_action = RECOMMENDED_ACTION_READY
    hash_details: dict[str, Any] = {}
    planned_record: dict[str, Any] = {}

    if _safe_text(packet_row.get("packet_review_status")) != PACKET_REVIEW_STATUS_READY:
        dry_run_status = DRY_RUN_STATUS_BLOCKED_INPUT_SCHEMA
        recommended_action = RECOMMENDED_ACTION_BLOCKED
        blockers.append("packet_row_not_design_packet_review_ready")

    if start is None or end is None or end <= start:
        dry_run_status = DRY_RUN_STATUS_BLOCKED_SCHEMA
        recommended_action = RECOMMENDED_ACTION_BLOCKED
        blockers.append("chars_start_or_end_missing_or_invalid")

    if basis != OFFSET_CHARS_BASIS:
        dry_run_status = DRY_RUN_STATUS_BLOCKED_SCHEMA
        recommended_action = RECOMMENDED_ACTION_BLOCKED
        blockers.append(f"chars_basis_invalid:{basis or 'missing'}")

    if normalization != OFFSET_CHARS_NORMALIZATION:
        dry_run_status = DRY_RUN_STATUS_BLOCKED_SCHEMA
        recommended_action = RECOMMENDED_ACTION_BLOCKED
        blockers.append(f"chars_normalization_invalid:{normalization or 'missing'}")

    if not expected_hash:
        dry_run_status = DRY_RUN_STATUS_BLOCKED_SCHEMA
        recommended_action = RECOMMENDED_ACTION_BLOCKED
        blockers.append("chars_expectedSubstringSha256_missing")

    if dry_run_status not in {
        DRY_RUN_STATUS_BLOCKED_INPUT_SCHEMA,
        DRY_RUN_STATUS_BLOCKED_SCHEMA,
    }:
        if paper_id not in paper_contexts:
            paper_contexts[paper_id] = _resolve_paper_source_context(
                paper_id=paper_id,
                parsed_root=parsed_root,
                papers_dir=papers_dir,
                page_loader=page_loader,
                expected_hash=source_content_hash,
            )
        context = paper_contexts[paper_id]
        if not context.canonical_text:
            dry_run_status = DRY_RUN_STATUS_BLOCKED_SOURCE_TEXT
            recommended_action = RECOMMENDED_ACTION_BLOCKED
            blockers.append(f"source_text_unavailable:{context.status}")
        elif (
            source_content_hash
            and context.source_content_hash
            and source_content_hash != context.source_content_hash
        ):
            dry_run_status = DRY_RUN_STATUS_BLOCKED_SOURCE_TEXT
            recommended_action = RECOMMENDED_ACTION_BLOCKED
            blockers.append("sourceContentHash_mismatch_with_canonical_pdf")
        else:
            hash_status, hash_details = _hash_verification(
                canonical_text=context.canonical_text,
                start=start or 0,
                end=end or 0,
                normalization=normalization,
                expected_hash=expected_hash,
            )
            if hash_status != DRY_RUN_STATUS_READY:
                dry_run_status = hash_status
                recommended_action = RECOMMENDED_ACTION_BLOCKED
                if hash_status == DRY_RUN_STATUS_BLOCKED_NORM_MISMATCH:
                    blockers.append(EXECUTOR_BLOCKER_NORMALIZATION_MISMATCH)
                    if hash_details.get("normalizationContractMismatch"):
                        blockers.append("design_packet_used_raw_utf8_slice_hash")
                    else:
                        blockers.append("computed_hash_does_not_match_expected")
                else:
                    blockers.append("source_text_unavailable_for_hash_verification")

    verified_hash = _safe_text(hash_details.get("contractSubstringSha256")) or expected_hash

    if dry_run_status == DRY_RUN_STATUS_READY:
        planned_record = build_sample_strict_evidence_record_from_packet_row(
            packet_row,
            run_id=run_id,
            design_packet_review_report_path=design_packet_review_report_path,
        )
        planned_record["verbatimSubstringSha256"] = verified_hash
        planned_record["authority"]["chars"]["expectedSubstringSha256"] = verified_hash

        schema_validation = validate_payload(
            planned_record,
            PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_SCHEMA_ID,
            strict=True,
        )
        if not schema_validation.ok:
            dry_run_status = DRY_RUN_STATUS_BLOCKED_SCHEMA
            recommended_action = RECOMMENDED_ACTION_BLOCKED
            blockers.extend(str(error) for error in schema_validation.errors)

        semantic_errors = validate_strict_evidence_record_semantics(planned_record)
        if semantic_errors:
            dry_run_status = DRY_RUN_STATUS_BLOCKED_SEMANTIC
            recommended_action = RECOMMENDED_ACTION_BLOCKED
            blockers.extend(semantic_errors)

    return {
        "dry_run_row_id": "",
        "packet_review_row_id": _safe_text(packet_row.get("packet_review_row_id")),
        "paper_id": paper_id,
        "artifact_type": _safe_text(packet_row.get("artifact_type")),
        "sourceSpanId": _safe_text(packet_row.get("sourceSpanId")),
        "candidateRecordId": _safe_text(packet_row.get("candidateRecordId")),
        "dry_run_status": dry_run_status,
        "dry_run_blockers": _dedupe(blockers),
        "recommended_action": recommended_action,
        "hashVerification": hash_details,
        "plannedStrictEvidenceRecord": planned_record if planned_record else {},
        "writeMatrix": _write_matrix(),
        "strictEvidenceWriteRows": 0,
        "strictEvidenceCreated": False,
        "runtimeEvidenceCreated": False,
        "sourceSpanUpdatedRows": 0,
    }


def _count_rows(
    *,
    rows: list[dict[str, Any]],
    input_rows: int,
    packet_ready_rows: int,
    input_schema_violations: list[str],
) -> dict[str, Any]:
    return {
        "inputRows": input_rows,
        "packetReadyRows": packet_ready_rows,
        "plannedStrictEvidenceRows": len(rows),
        "dryRunReadyStrictEvidenceRecordOnlyRows": sum(
            1 for row in rows if row.get("dry_run_status") == DRY_RUN_STATUS_READY
        ),
        "blockedSourceTextUnavailableRows": sum(
            1 for row in rows if row.get("dry_run_status") == DRY_RUN_STATUS_BLOCKED_SOURCE_TEXT
        ),
        "blockedNormalizationHashContractMismatchRows": sum(
            1 for row in rows if row.get("dry_run_status") == DRY_RUN_STATUS_BLOCKED_NORM_MISMATCH
        ),
        "blockedPlannedRecordSchemaViolationRows": sum(
            1 for row in rows if row.get("dry_run_status") == DRY_RUN_STATUS_BLOCKED_SCHEMA
        ),
        "blockedPlannedRecordSemanticViolationRows": sum(
            1 for row in rows if row.get("dry_run_status") == DRY_RUN_STATUS_BLOCKED_SEMANTIC
        ),
        "blockedInputSchemaViolationRows": int(bool(input_schema_violations)),
        "strictEvidenceWriteRows": 0,
        "strictEvidenceCreatedRows": 0,
        "citationGradeEvidenceCreatedRows": 0,
        "runtimeEvidenceCreatedRows": 0,
        "parserRoutingChangedRows": 0,
        "answerIntegrationChangedRows": 0,
        "databaseMutationRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
        "schemaViolationCount": len(input_schema_violations),
        "byPaperId": dict(Counter(str(row.get("paper_id") or "") for row in rows)),
        "byArtifactType": dict(Counter(str(row.get("artifact_type") or "") for row in rows)),
        "byDryRunStatus": dict(Counter(str(row.get("dry_run_status") or "") for row in rows)),
        "byRecommendedAction": dict(Counter(str(row.get("recommended_action") or "") for row in rows)),
    }


def build_parsed_artifact_strict_evidence_executor_dry_run(
    *,
    design_packet_review_report_path: str | Path = DEFAULT_DESIGN_PACKET_REVIEW_REPORT_PATH,
    contract_report_path: str | Path = DEFAULT_CONTRACT_REPORT_PATH,
    normalization_hash_repair_report_path: str | Path | None = None,
    papers_dir: str | Path = DEFAULT_PAPERS_DIR,
    parsed_root: str | Path | None = None,
    run_id: str = "parsed-artifact-strict-evidence-executor-dry-run",
    paper_ids: list[str] | None = None,
    page_loader: Callable[[str | Path], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    packet_path = Path(str(design_packet_review_report_path)).expanduser()
    contract_path = Path(str(contract_report_path)).expanduser()
    repair_path = (
        Path(str(normalization_hash_repair_report_path)).expanduser()
        if normalization_hash_repair_report_path
        else None
    )
    papers_path = Path(str(papers_dir)).expanduser()
    parsed_path = Path(str(parsed_root or (papers_path / "parsed"))).expanduser()
    loader = page_loader or _extract_pdf_pages

    warnings: list[str] = []
    input_schema_violations: list[str] = []
    normalization_repair_applied_rows = 0

    packet_report = _read_json(packet_path)
    contract_report = _read_json(contract_path)
    repair_report = _read_json(repair_path) if repair_path else {}

    if not packet_report:
        warnings.append("design_packet_review_report_missing_or_unreadable")
    if not contract_report:
        warnings.append("strict_evidence_contract_report_missing_or_unreadable")
    if repair_path and not repair_report:
        warnings.append("normalization_hash_repair_report_missing_or_unreadable")

    packet_validation = validate_payload(
        packet_report,
        PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_PACKET_REVIEW_SCHEMA_ID,
        strict=True,
    )
    if not packet_validation.ok:
        input_schema_violations.extend(str(error) for error in packet_validation.errors)

    contract_validation = validate_payload(
        contract_report,
        PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_CONTRACT_SCHEMA_ID,
        strict=True,
    )
    if not contract_validation.ok:
        input_schema_violations.extend(str(error) for error in contract_validation.errors)

    if repair_path:
        repair_validation = validate_payload(
            repair_report,
            PARSED_ARTIFACT_STRICT_EVIDENCE_NORMALIZATION_HASH_REPAIR_SCHEMA_ID,
            strict=True,
        )
        if not repair_validation.ok:
            input_schema_violations.extend(str(error) for error in repair_validation.errors)
        elif not input_schema_violations:
            packet_report, normalization_repair_applied_rows = (
                _apply_normalization_hash_repair_to_packet_report(packet_report, repair_report)
            )
            repaired_packet_validation = validate_payload(
                packet_report,
                PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_PACKET_REVIEW_SCHEMA_ID,
                strict=True,
            )
            if not repaired_packet_validation.ok:
                input_schema_violations.extend(
                    str(error) for error in repaired_packet_validation.errors
                )

    packet_rows = [
        row
        for row in packet_report.get("packetRows", [])
        if isinstance(row, dict)
        and _safe_text(row.get("packet_review_status")) == PACKET_REVIEW_STATUS_READY
    ] if isinstance(packet_report, dict) else []

    input_rows = int((packet_report.get("counts") or {}).get("inputRows") or 0) if packet_report else 0
    packet_ready_rows = len(packet_rows)

    requested_papers = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}
    if requested_papers:
        packet_rows = [row for row in packet_rows if _safe_text(row.get("paper_id")) in requested_papers]

    if input_schema_violations:
        for row in packet_rows:
            row  # keep variable used
        rows = [
            {
                "dry_run_row_id": f"parsed-artifact-strict-evidence-executor-dry-run:{index:04d}",
                "packet_review_row_id": _safe_text(packet_row.get("packet_review_row_id")),
                "paper_id": _safe_text(packet_row.get("paper_id")),
                "artifact_type": _safe_text(packet_row.get("artifact_type")),
                "sourceSpanId": _safe_text(packet_row.get("sourceSpanId")),
                "candidateRecordId": _safe_text(packet_row.get("candidateRecordId")),
                "dry_run_status": DRY_RUN_STATUS_BLOCKED_INPUT_SCHEMA,
                "dry_run_blockers": _dedupe(input_schema_violations),
                "recommended_action": RECOMMENDED_ACTION_REPAIR,
                "hashVerification": {},
                "plannedStrictEvidenceRecord": {},
                "writeMatrix": _write_matrix(),
                "strictEvidenceWriteRows": 0,
                "strictEvidenceCreated": False,
                "runtimeEvidenceCreated": False,
                "sourceSpanUpdatedRows": 0,
            }
            for index, packet_row in enumerate(packet_rows)
        ]
    else:
        paper_contexts: dict[str, _PaperSourceContext] = {}
        rows = [
            _classify_packet_row(
                packet_row,
                paper_contexts=paper_contexts,
                parsed_root=parsed_path,
                papers_dir=papers_path,
                page_loader=loader,
                design_packet_review_report_path=str(packet_path),
                run_id=run_id,
            )
            for packet_row in packet_rows
        ]
        for index, row in enumerate(rows):
            row["dry_run_row_id"] = f"parsed-artifact-strict-evidence-executor-dry-run:{index:04d}"

    counts = _count_rows(
        rows=rows,
        input_rows=input_rows,
        packet_ready_rows=packet_ready_rows,
        input_schema_violations=_dedupe(input_schema_violations),
    )

    ready_count = int(counts.get("dryRunReadyStrictEvidenceRecordOnlyRows") or 0)
    norm_blocked = int(counts.get("blockedNormalizationHashContractMismatchRows") or 0)

    status = "ok"
    if input_schema_violations:
        status = "blocked"

    if ready_count > 0 and not input_schema_violations:
        next_tranche = "parsed_artifact_strict_evidence_executor_apply"
    elif norm_blocked > 0:
        next_tranche = "parsed_artifact_strict_evidence_normalization_hash_repair"
    else:
        next_tranche = "parsed_artifact_strict_evidence_executor_dry_run_repair"

    return {
        "schema": PARSED_ARTIFACT_STRICT_EVIDENCE_EXECUTOR_DRY_RUN_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "designPacketReviewReportPath": str(packet_path),
            "designPacketReviewSchema": _safe_text(packet_report.get("schema")),
            "contractReportPath": str(contract_path),
            "contractSchema": _safe_text(contract_report.get("schema")),
            "normalizationHashRepairReportPath": str(repair_path or ""),
            "normalizationHashRepairSchema": _safe_text(repair_report.get("schema")),
            "normalizationHashRepairApplied": bool(normalization_repair_applied_rows),
            "normalizationHashRepairAppliedRows": normalization_repair_applied_rows,
            "papersDir": str(papers_path),
            "parsedRoot": str(parsed_path),
            "runId": run_id,
            "requestedPaperIds": sorted(requested_papers),
        },
        "counts": counts,
        "gate": {
            "readyForStrictEvidenceExecutorApply": ready_count > 0 and not input_schema_violations,
            "normalizationHashRepairApplied": bool(normalization_repair_applied_rows),
            "normalizationHashRepairAppliedRows": normalization_repair_applied_rows,
            "strictEvidenceCreated": False,
            "strictEvidenceReady": False,
            "citationReady": False,
            "runtimeEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": _dedupe(input_schema_violations),
            "decision": (
                "parsed_artifact_strict_evidence_executor_dry_run_ready"
                if ready_count > 0 and not input_schema_violations
                else "blocked"
            ),
            "recommendedNextTranche": next_tranche,
        },
        "policy": {
            "reportOnly": True,
            "executorImplemented": False,
            "normalizationHashRepairReportOnly": True,
            "strictEvidenceStoreWrite": False,
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
        },
        "warnings": _dedupe(warnings),
        "rows": rows,
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
            "warnings",
            "rows",
        )
        if key in report
    }


def render_parsed_artifact_strict_evidence_executor_dry_run_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byDryRunStatus") or {})).items())
    ]
    return "\n".join(
        [
            "# Parsed Artifact StrictEvidence Executor Dry Run",
            "",
            f"- status: {report.get('status', '')}",
            f"- packet ready rows: {int(counts.get('packetReadyRows') or 0)}",
            f"- dry-run ready: {int(counts.get('dryRunReadyStrictEvidenceRecordOnlyRows') or 0)}",
            f"- normalization/hash blocked: {int(counts.get('blockedNormalizationHashContractMismatchRows') or 0)}",
            f"- source text blocked: {int(counts.get('blockedSourceTextUnavailableRows') or 0)}",
            f"- strict evidence writes: {int(counts.get('strictEvidenceWriteRows') or 0)}",
            "",
            "## Dry-run status breakdown",
            *[f"- {item}" for item in by_status],
        ]
    )


def write_parsed_artifact_strict_evidence_executor_dry_run_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-strict-evidence-executor-dry-run.json"
    summary_path = root / "parsed-artifact-strict-evidence-executor-dry-run-summary.json"
    markdown_path = root / "parsed-artifact-strict-evidence-executor-dry-run.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_parsed_artifact_strict_evidence_executor_dry_run_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description="Dry-run planner for StrictEvidence records with hash verification and zero writes."
    )
    parser.add_argument(
        "--design-packet-review-report",
        default=str(DEFAULT_DESIGN_PACKET_REVIEW_REPORT_PATH),
    )
    parser.add_argument("--contract-report", default=str(DEFAULT_CONTRACT_REPORT_PATH))
    parser.add_argument(
        "--normalization-hash-repair-report",
        default="",
        help="Optional normalization/hash repair report to apply in-memory before dry-run planning.",
    )
    parser.add_argument("--papers-dir", default=str(DEFAULT_PAPERS_DIR))
    parser.add_argument("--parsed-root", default="")
    parser.add_argument("--run-id", default="parsed-artifact-strict-evidence-executor-dry-run")
    parser.add_argument("--paper-id", action="append", default=[])
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    output_dir = args.output_dir or str(
        DEFAULT_REPAIRED_OUTPUT_DIR
        if args.normalization_hash_repair_report
        else DEFAULT_OUTPUT_DIR
    )

    report = build_parsed_artifact_strict_evidence_executor_dry_run(
        design_packet_review_report_path=args.design_packet_review_report,
        contract_report_path=args.contract_report,
        normalization_hash_repair_report_path=args.normalization_hash_repair_report or None,
        papers_dir=args.papers_dir,
        parsed_root=args.parsed_root or None,
        run_id=args.run_id,
        paper_ids=args.paper_id or None,
    )

    if output_dir:
        paths = write_parsed_artifact_strict_evidence_executor_dry_run_reports(
            report,
            output_dir,
        )
        print(f"wrote report: {paths['report']}")
        print(f"wrote summary: {paths['summary']}")
        print(f"wrote markdown: {paths['markdown']}")

    if args.json or not output_dir:
        print(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PARSED_ARTIFACT_STRICT_EVIDENCE_EXECUTOR_DRY_RUN_SCHEMA_ID",
    "DRY_RUN_STATUS_READY",
    "DRY_RUN_STATUS_BLOCKED_NORM_MISMATCH",
    "compute_contract_substring_sha256",
    "compute_raw_utf8_slice_sha256",
    "normalize_nfkc_whitespace_casefold_v1",
    "build_parsed_artifact_strict_evidence_executor_dry_run",
    "write_parsed_artifact_strict_evidence_executor_dry_run_reports",
    "DEFAULT_NORMALIZATION_HASH_REPAIR_REPORT_PATH",
    "DEFAULT_REPAIRED_OUTPUT_DIR",
]
