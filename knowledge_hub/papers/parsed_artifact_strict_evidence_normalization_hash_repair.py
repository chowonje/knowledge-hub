"""Report-only normalization/hash repair for strict-evidence design packet rows.

Consumes executor dry-run rows blocked by normalization/hash mismatch and recomputes
expectedSubstringSha256 using the contract normalization function. Does not mutate
the original design packet report, SourceSpan JSONL, or strict_evidence stores.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
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
from knowledge_hub.papers.parsed_artifact_strict_evidence_executor_dry_run import (
    DRY_RUN_STATUS_BLOCKED_NORM_MISMATCH,
    PARSED_ARTIFACT_STRICT_EVIDENCE_EXECUTOR_DRY_RUN_SCHEMA_ID,
    compute_contract_substring_sha256,
    compute_raw_utf8_slice_sha256,
    normalize_nfkc_whitespace_casefold_v1,
)
from knowledge_hub.papers.sectionspan_pdf_offset_recovery_dry_run import _extract_pdf_pages


PARSED_ARTIFACT_STRICT_EVIDENCE_NORMALIZATION_HASH_REPAIR_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-strict-evidence-normalization-hash-repair.v1"
)

REPAIR_STATUS_CANDIDATE = "normalization_hash_repair_candidate_only"
REPAIR_STATUS_BLOCKED_SOURCE_TEXT = "blocked_source_text_unavailable"
REPAIR_STATUS_BLOCKED_MISSING_CHARS = "blocked_missing_chars"
REPAIR_STATUS_BLOCKED_INVALID_BASIS = "blocked_invalid_chars_basis"
REPAIR_STATUS_BLOCKED_UNSUPPORTED_NORM = "blocked_unsupported_normalization"
REPAIR_STATUS_BLOCKED_NOT_NEEDED = "blocked_hash_repair_not_needed"
REPAIR_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"

RECOMMENDED_ACTION_CANDIDATE = "queue_for_strict_evidence_executor_dry_run_with_repaired_packet"
RECOMMENDED_ACTION_BLOCKED = "hold_hash_repair_row_until_blockers_resolved"
RECOMMENDED_ACTION_REPAIR_INPUT = "repair_input_reports_before_normalization_hash_repair"

DEFAULT_EXECUTOR_DRY_RUN_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-strict-evidence-executor-dry-run"
    / "01-parsed-artifact-strict-evidence-executor-dry-run"
    / "parsed-artifact-strict-evidence-executor-dry-run.json"
)

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
    / "parsed-artifact-strict-evidence-normalization-hash-repair"
    / "01-parsed-artifact-strict-evidence-normalization-hash-repair"
)


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


def _proposed_chars_dict(row: dict[str, Any]) -> dict[str, Any]:
    proposed = row.get("proposed_chars")
    return proposed if isinstance(proposed, dict) else {}


def _write_matrix() -> dict[str, Any]:
    return {
        "writeEnabled": False,
        "designPacketReviewReportWrite": False,
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


def _index_packet_rows(packet_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in packet_report.get("packetRows", []):
        if not isinstance(row, dict):
            continue
        key = _safe_text(row.get("packet_review_row_id"))
        if key:
            index[key] = row
    return index


def _build_repaired_packet_row(packet_row: dict[str, Any], *, repaired_hash: str) -> dict[str, Any]:
    repaired = deepcopy(packet_row)
    proposed = _proposed_chars_dict(repaired)
    proposed["expectedSubstringSha256"] = repaired_hash
    proposed["basis"] = OFFSET_CHARS_BASIS
    proposed["normalization"] = OFFSET_CHARS_NORMALIZATION
    repaired["proposed_chars"] = proposed
    repaired["normalizationHashRepairCandidateOnly"] = True
    repaired["priorExpectedSubstringSha256"] = _safe_text(
        _proposed_chars_dict(packet_row).get("expectedSubstringSha256")
    )
    return repaired


def apply_hash_repair_to_design_packet_review_report(
    packet_report: dict[str, Any],
    repair_report: dict[str, Any],
) -> dict[str, Any]:
    """Return a new design packet review report with repaired hashes (does not write files)."""
    repaired = deepcopy(packet_report)
    repair_by_id = {
        _safe_text(row.get("packet_review_row_id")): row
        for row in repair_report.get("rows", [])
        if isinstance(row, dict)
        and row.get("repair_status") == REPAIR_STATUS_CANDIDATE
    }
    packet_rows = []
    for row in repaired.get("packetRows", []):
        if not isinstance(row, dict):
            continue
        key = _safe_text(row.get("packet_review_row_id"))
        repair_row = repair_by_id.get(key)
        if repair_row and isinstance(repair_row.get("repairedPacketRow"), dict):
            packet_rows.append(deepcopy(repair_row["repairedPacketRow"]))
        else:
            packet_rows.append(deepcopy(row))
    repaired["packetRows"] = packet_rows
    repaired["rows"] = packet_rows
    repaired["normalizationHashRepairApplied"] = True
    repaired["normalizationHashRepairReportPath"] = _safe_text(
        repair_report.get("input", {}).get("executorDryRunReportPath")
        if isinstance(repair_report.get("input"), dict)
        else ""
    )
    return repaired


def _classify_repair_row(
    *,
    dry_run_row: dict[str, Any],
    packet_row: dict[str, Any] | None,
    paper_contexts: dict[str, _PaperSourceContext],
    parsed_root: Path,
    papers_dir: Path,
    page_loader: Callable[[str | Path], list[dict[str, Any]]],
) -> dict[str, Any]:
    blockers: list[str] = ["strict_evidence_normalization_hash_repair_only"]
    dry_run_status = _safe_text(dry_run_row.get("dry_run_status"))

    if dry_run_status != DRY_RUN_STATUS_BLOCKED_NORM_MISMATCH:
        return {
            "repair_status": REPAIR_STATUS_BLOCKED_NOT_NEEDED,
            "repair_blockers": _dedupe([*blockers, f"dry_run_status_not_mismatch:{dry_run_status or 'missing'}"]),
            "recommended_action": RECOMMENDED_ACTION_BLOCKED,
            "hashRepair": {},
            "repairedPacketRow": {},
        }

    if not packet_row:
        return {
            "repair_status": REPAIR_STATUS_BLOCKED_INPUT_SCHEMA,
            "repair_blockers": _dedupe([*blockers, "packet_row_missing_for_dry_run_row"]),
            "recommended_action": RECOMMENDED_ACTION_REPAIR_INPUT,
            "hashRepair": {},
            "repairedPacketRow": {},
        }

    proposed = _proposed_chars_dict(packet_row)
    start = _safe_int(proposed.get("start"))
    end = _safe_int(proposed.get("end"))
    basis = _safe_text(proposed.get("basis"))
    normalization = _safe_text(proposed.get("normalization"))
    prior_hash = _safe_text(proposed.get("expectedSubstringSha256"))
    source_content_hash = _safe_text(packet_row.get("sourceContentHash"))
    paper_id = _safe_text(packet_row.get("paper_id"))

    if start is None or end is None or end <= start:
        return {
            "repair_status": REPAIR_STATUS_BLOCKED_MISSING_CHARS,
            "repair_blockers": _dedupe([*blockers, "chars_start_or_end_missing_or_invalid"]),
            "recommended_action": RECOMMENDED_ACTION_BLOCKED,
            "hashRepair": {"priorExpectedSubstringSha256": prior_hash},
            "repairedPacketRow": {},
        }

    if basis != OFFSET_CHARS_BASIS:
        return {
            "repair_status": REPAIR_STATUS_BLOCKED_INVALID_BASIS,
            "repair_blockers": _dedupe([*blockers, f"chars_basis_invalid:{basis or 'missing'}"]),
            "recommended_action": RECOMMENDED_ACTION_BLOCKED,
            "hashRepair": {"priorExpectedSubstringSha256": prior_hash},
            "repairedPacketRow": {},
        }

    if normalization != OFFSET_CHARS_NORMALIZATION:
        return {
            "repair_status": REPAIR_STATUS_BLOCKED_UNSUPPORTED_NORM,
            "repair_blockers": _dedupe([*blockers, f"chars_normalization_unsupported:{normalization or 'missing'}"]),
            "recommended_action": RECOMMENDED_ACTION_BLOCKED,
            "hashRepair": {"priorExpectedSubstringSha256": prior_hash},
            "repairedPacketRow": {},
        }

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
        return {
            "repair_status": REPAIR_STATUS_BLOCKED_SOURCE_TEXT,
            "repair_blockers": _dedupe([*blockers, f"source_text_unavailable:{context.status}"]),
            "recommended_action": RECOMMENDED_ACTION_BLOCKED,
            "hashRepair": {"priorExpectedSubstringSha256": prior_hash},
            "repairedPacketRow": {},
        }

    if (
        source_content_hash
        and context.source_content_hash
        and source_content_hash != context.source_content_hash
    ):
        return {
            "repair_status": REPAIR_STATUS_BLOCKED_SOURCE_TEXT,
            "repair_blockers": _dedupe([*blockers, "sourceContentHash_mismatch_with_canonical_pdf"]),
            "recommended_action": RECOMMENDED_ACTION_BLOCKED,
            "hashRepair": {"priorExpectedSubstringSha256": prior_hash},
            "repairedPacketRow": {},
        }

    canonical_text = context.canonical_text
    raw_slice = canonical_text[start:end]
    normalized_slice = normalize_nfkc_whitespace_casefold_v1(raw_slice)
    try:
        repaired_hash = compute_contract_substring_sha256(
            canonical_text,
            start,
            end,
            normalization=normalization,
        )
        raw_hash = compute_raw_utf8_slice_sha256(canonical_text, start, end)
    except Exception as exc:
        return {
            "repair_status": REPAIR_STATUS_BLOCKED_SOURCE_TEXT,
            "repair_blockers": _dedupe([*blockers, f"hash_recomputation_failed:{exc}"]),
            "recommended_action": RECOMMENDED_ACTION_BLOCKED,
            "hashRepair": {"priorExpectedSubstringSha256": prior_hash},
            "repairedPacketRow": {},
        }

    hash_repair = {
        "priorExpectedSubstringSha256": prior_hash,
        "repairedExpectedSubstringSha256": repaired_hash,
        "contractSubstringSha256": repaired_hash,
        "rawUtf8SliceSubstringSha256": raw_hash,
        "normalization": normalization,
        "canonicalSubstringLength": len(raw_slice),
        "normalizedSubstringLength": len(normalized_slice),
    }

    if repaired_hash == prior_hash:
        return {
            "repair_status": REPAIR_STATUS_BLOCKED_NOT_NEEDED,
            "repair_blockers": _dedupe([*blockers, "prior_hash_already_matches_contract_normalization"]),
            "recommended_action": RECOMMENDED_ACTION_BLOCKED,
            "hashRepair": hash_repair,
            "repairedPacketRow": {},
        }

    repaired_packet_row = _build_repaired_packet_row(packet_row, repaired_hash=repaired_hash)
    return {
        "repair_status": REPAIR_STATUS_CANDIDATE,
        "repair_blockers": _dedupe(
            [
                *blockers,
                "design_packet_review_report_not_mutated",
                "strict_evidence_creation_disabled_for_tranche",
            ]
        ),
        "recommended_action": RECOMMENDED_ACTION_CANDIDATE,
        "hashRepair": hash_repair,
        "repairedPacketRow": repaired_packet_row,
    }


def _count_rows(
    *,
    rows: list[dict[str, Any]],
    target_mismatch_rows: int,
    input_schema_violations: list[str],
) -> dict[str, Any]:
    return {
        "inputRows": target_mismatch_rows,
        "targetMismatchRows": target_mismatch_rows,
        "normalizationHashRepairCandidateOnlyRows": sum(
            1 for row in rows if row.get("repair_status") == REPAIR_STATUS_CANDIDATE
        ),
        "blockedSourceTextUnavailableRows": sum(
            1 for row in rows if row.get("repair_status") == REPAIR_STATUS_BLOCKED_SOURCE_TEXT
        ),
        "blockedMissingCharsRows": sum(
            1 for row in rows if row.get("repair_status") == REPAIR_STATUS_BLOCKED_MISSING_CHARS
        ),
        "blockedInvalidCharsBasisRows": sum(
            1 for row in rows if row.get("repair_status") == REPAIR_STATUS_BLOCKED_INVALID_BASIS
        ),
        "blockedUnsupportedNormalizationRows": sum(
            1 for row in rows if row.get("repair_status") == REPAIR_STATUS_BLOCKED_UNSUPPORTED_NORM
        ),
        "blockedHashRepairNotNeededRows": sum(
            1 for row in rows if row.get("repair_status") == REPAIR_STATUS_BLOCKED_NOT_NEEDED
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
        "designPacketReviewReportWriteRows": 0,
        "sourceSpanUpdatedRows": 0,
        "schemaViolationCount": len(input_schema_violations),
        "byPaperId": dict(Counter(str(row.get("paper_id") or "") for row in rows)),
        "byArtifactType": dict(Counter(str(row.get("artifact_type") or "") for row in rows)),
        "byRepairStatus": dict(Counter(str(row.get("repair_status") or "") for row in rows)),
        "byRecommendedAction": dict(Counter(str(row.get("recommended_action") or "") for row in rows)),
    }


def build_parsed_artifact_strict_evidence_normalization_hash_repair(
    *,
    executor_dry_run_report_path: str | Path = DEFAULT_EXECUTOR_DRY_RUN_REPORT_PATH,
    design_packet_review_report_path: str | Path = DEFAULT_DESIGN_PACKET_REVIEW_REPORT_PATH,
    papers_dir: str | Path = DEFAULT_PAPERS_DIR,
    parsed_root: str | Path | None = None,
    run_id: str = "parsed-artifact-strict-evidence-normalization-hash-repair",
    paper_ids: list[str] | None = None,
    page_loader: Callable[[str | Path], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    dry_run_path = Path(str(executor_dry_run_report_path)).expanduser()
    packet_path = Path(str(design_packet_review_report_path)).expanduser()
    papers_path = Path(str(papers_dir)).expanduser()
    parsed_path = Path(str(parsed_root or (papers_path / "parsed"))).expanduser()
    loader = page_loader or _extract_pdf_pages

    warnings: list[str] = []
    input_schema_violations: list[str] = []

    dry_run_report = _read_json(dry_run_path)
    packet_report = _read_json(packet_path)

    if not dry_run_report:
        warnings.append("executor_dry_run_report_missing_or_unreadable")
    if not packet_report:
        warnings.append("design_packet_review_report_missing_or_unreadable")

    dry_run_validation = validate_payload(
        dry_run_report,
        PARSED_ARTIFACT_STRICT_EVIDENCE_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    )
    if not dry_run_validation.ok:
        input_schema_violations.extend(str(error) for error in dry_run_validation.errors)

    packet_validation = validate_payload(
        packet_report,
        PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_PACKET_REVIEW_SCHEMA_ID,
        strict=True,
    )
    if not packet_validation.ok:
        input_schema_violations.extend(str(error) for error in packet_validation.errors)

    packet_index = _index_packet_rows(packet_report) if packet_report else {}

    target_dry_run_rows = [
        row
        for row in dry_run_report.get("rows", [])
        if isinstance(row, dict)
        and _safe_text(row.get("dry_run_status")) == DRY_RUN_STATUS_BLOCKED_NORM_MISMATCH
    ] if isinstance(dry_run_report, dict) else []

    requested_papers = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}
    if requested_papers:
        target_dry_run_rows = [
            row for row in target_dry_run_rows if _safe_text(row.get("paper_id")) in requested_papers
        ]

    target_mismatch_rows = len(target_dry_run_rows)

    if input_schema_violations:
        rows = [
            {
                "repair_row_id": f"{run_id}:{index:04d}",
                "dry_run_row_id": _safe_text(dry_run_row.get("dry_run_row_id")),
                "packet_review_row_id": _safe_text(dry_run_row.get("packet_review_row_id")),
                "paper_id": _safe_text(dry_run_row.get("paper_id")),
                "artifact_type": _safe_text(dry_run_row.get("artifact_type")),
                "sourceSpanId": _safe_text(dry_run_row.get("sourceSpanId")),
                "candidateRecordId": _safe_text(dry_run_row.get("candidateRecordId")),
                "repair_status": REPAIR_STATUS_BLOCKED_INPUT_SCHEMA,
                "repair_blockers": _dedupe(input_schema_violations),
                "recommended_action": RECOMMENDED_ACTION_REPAIR_INPUT,
                "hashRepair": {},
                "repairedPacketRow": {},
                "writeMatrix": _write_matrix(),
            }
            for index, dry_run_row in enumerate(target_dry_run_rows)
        ]
    else:
        paper_contexts: dict[str, _PaperSourceContext] = {}
        rows = []
        for index, dry_run_row in enumerate(target_dry_run_rows):
            packet_review_row_id = _safe_text(dry_run_row.get("packet_review_row_id"))
            packet_row = packet_index.get(packet_review_row_id)
            classified = _classify_repair_row(
                dry_run_row=dry_run_row,
                packet_row=packet_row,
                paper_contexts=paper_contexts,
                parsed_root=parsed_path,
                papers_dir=papers_path,
                page_loader=loader,
            )
            rows.append(
                {
                    "repair_row_id": f"{run_id}:{index:04d}",
                    "dry_run_row_id": _safe_text(dry_run_row.get("dry_run_row_id")),
                    "packet_review_row_id": packet_review_row_id,
                    "paper_id": _safe_text(dry_run_row.get("paper_id")),
                    "artifact_type": _safe_text(dry_run_row.get("artifact_type")),
                    "sourceSpanId": _safe_text(dry_run_row.get("sourceSpanId")),
                    "candidateRecordId": _safe_text(dry_run_row.get("candidateRecordId")),
                    "dry_run_status": _safe_text(dry_run_row.get("dry_run_status")),
                    "writeMatrix": _write_matrix(),
                    **classified,
                }
            )

    repaired_packet_rows = [
        deepcopy(row["repairedPacketRow"])
        for row in rows
        if row.get("repair_status") == REPAIR_STATUS_CANDIDATE
        and isinstance(row.get("repairedPacketRow"), dict)
        and row["repairedPacketRow"]
    ]

    counts = _count_rows(
        rows=rows,
        target_mismatch_rows=target_mismatch_rows,
        input_schema_violations=_dedupe(input_schema_violations),
    )

    candidate_count = int(counts.get("normalizationHashRepairCandidateOnlyRows") or 0)
    status = "ok"
    if input_schema_violations:
        status = "blocked"

    executor_consumes_repaired = False
    consumption_note = (
        "parsed_artifact_strict_evidence_executor_dry_run does not yet accept a "
        "normalization-hash-repair report directly. Use "
        "apply_hash_repair_to_design_packet_review_report() to build a new in-memory "
        "design packet review payload (or write a separate repaired packet review file) "
        "before re-running the executor dry-run."
    )

    if candidate_count > 0 and not input_schema_violations:
        next_tranche = "parsed_artifact_strict_evidence_executor_dry_run_with_repaired_design_packet"
    elif input_schema_violations:
        next_tranche = "repair_input_reports_before_normalization_hash_repair"
    else:
        next_tranche = "parsed_artifact_strict_evidence_normalization_hash_repair_follow_up"

    return {
        "schema": PARSED_ARTIFACT_STRICT_EVIDENCE_NORMALIZATION_HASH_REPAIR_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "executorDryRunReportPath": str(dry_run_path),
            "executorDryRunSchema": _safe_text(dry_run_report.get("schema")),
            "designPacketReviewReportPath": str(packet_path),
            "designPacketReviewSchema": _safe_text(packet_report.get("schema")),
            "papersDir": str(papers_path),
            "parsedRoot": str(parsed_path),
            "runId": run_id,
            "requestedPaperIds": sorted(requested_papers),
        },
        "counts": counts,
        "gate": {
            "normalizationHashRepairCandidateReady": candidate_count > 0 and not input_schema_violations,
            "executorDryRunConsumesRepairedHashesDirectly": executor_consumes_repaired,
            "executorDryRunConsumptionNote": consumption_note,
            "strictEvidenceCreated": False,
            "strictEvidenceReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": _dedupe(input_schema_violations),
            "decision": (
                "parsed_artifact_strict_evidence_normalization_hash_repair_ready"
                if candidate_count > 0 and not input_schema_violations
                else "blocked"
            ),
            "recommendedNextTranche": next_tranche,
        },
        "policy": {
            "reportOnly": True,
            "designPacketReviewReportMutated": False,
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
        "repairedPacketRows": repaired_packet_rows,
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
            "repairedPacketRows",
        )
        if key in report
    }


def render_parsed_artifact_strict_evidence_normalization_hash_repair_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byRepairStatus") or {})).items())
    ]
    return "\n".join(
        [
            "# Parsed Artifact StrictEvidence Normalization Hash Repair",
            "",
            f"- status: {report.get('status', '')}",
            f"- target mismatch rows: {int(counts.get('targetMismatchRows') or 0)}",
            f"- repair candidates: {int(counts.get('normalizationHashRepairCandidateOnlyRows') or 0)}",
            f"- design packet writes: {int(counts.get('designPacketReviewReportWriteRows') or 0)}",
            "",
            f"- executor consumes repaired hashes directly: {gate.get('executorDryRunConsumesRepairedHashesDirectly')}",
            f"- consumption note: {gate.get('executorDryRunConsumptionNote', '')}",
            "",
            "## Repair status breakdown",
            *[f"- {item}" for item in by_status],
        ]
    )


def write_parsed_artifact_strict_evidence_normalization_hash_repair_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-strict-evidence-normalization-hash-repair.json"
    summary_path = root / "parsed-artifact-strict-evidence-normalization-hash-repair-summary.json"
    markdown_path = root / "parsed-artifact-strict-evidence-normalization-hash-repair.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_parsed_artifact_strict_evidence_normalization_hash_repair_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description="Report-only normalization/hash repair for strict-evidence design packet rows."
    )
    parser.add_argument("--executor-dry-run-report", default=str(DEFAULT_EXECUTOR_DRY_RUN_REPORT_PATH))
    parser.add_argument(
        "--design-packet-review-report",
        default=str(DEFAULT_DESIGN_PACKET_REVIEW_REPORT_PATH),
    )
    parser.add_argument("--papers-dir", default=str(DEFAULT_PAPERS_DIR))
    parser.add_argument("--parsed-root", default="")
    parser.add_argument("--run-id", default="parsed-artifact-strict-evidence-normalization-hash-repair")
    parser.add_argument("--paper-id", action="append", default=[])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = build_parsed_artifact_strict_evidence_normalization_hash_repair(
        executor_dry_run_report_path=args.executor_dry_run_report,
        design_packet_review_report_path=args.design_packet_review_report,
        papers_dir=args.papers_dir,
        parsed_root=args.parsed_root or None,
        run_id=args.run_id,
        paper_ids=args.paper_id or None,
    )

    if args.output_dir:
        paths = write_parsed_artifact_strict_evidence_normalization_hash_repair_reports(
            report,
            args.output_dir,
        )
        print(f"wrote report: {paths['report']}")
        print(f"wrote summary: {paths['summary']}")
        print(f"wrote markdown: {paths['markdown']}")

    if args.json or not args.output_dir:
        print(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PARSED_ARTIFACT_STRICT_EVIDENCE_NORMALIZATION_HASH_REPAIR_SCHEMA_ID",
    "REPAIR_STATUS_CANDIDATE",
    "apply_hash_repair_to_design_packet_review_report",
    "build_parsed_artifact_strict_evidence_normalization_hash_repair",
    "write_parsed_artifact_strict_evidence_normalization_hash_repair_reports",
]
