"""Dry-run planner for the StrictEvidence figure-caption pilot.

Consumes the figure-caption pilot report, validates figure caption candidate rows, and emits
deterministic validate-only executor plans without mutating records or integration surfaces.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_strict_evidence_record_contract import (
    CHARS_BASIS,
    CHARS_NORMALIZATION_LABEL,
)
from knowledge_hub.papers.parsed_artifact_strict_evidence_promotion_tranche_plan import (
    TRANCHE_FIGURE_CAPTION_PILOT,
    TRANCHE_TEXT_SECTION_PILOT,
)
from knowledge_hub.papers.strict_evidence_figure_caption_pilot import (
    PILOT_STATUS_CANDIDATE_ONLY,
    PILOT_STATUS_HELD_OUT_SECTION,
    STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_SCHEMA_ID,
)


STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_EXECUTOR_DRY_RUN_SCHEMA_ID = (
    "knowledge-hub.paper.strict-evidence-figure-caption-pilot-executor-dry-run.v1"
)

DRY_RUN_STATUS_READY = "dry_run_ready_figure_caption_pilot_executor_only"
DRY_RUN_STATUS_HELD_OUT = "held_out_section_tranche"
DRY_RUN_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"
DRY_RUN_STATUS_BLOCKED_PILOT_NOT_CANDIDATE = "blocked_figure_caption_pilot_not_candidate"
DRY_RUN_STATUS_BLOCKED_MISSING_IDENTITY = "blocked_missing_record_identity"
DRY_RUN_STATUS_BLOCKED_INVALID_ARTIFACT = "blocked_invalid_artifact_type"
DRY_RUN_STATUS_BLOCKED_MISSING_AUTHORITY = "blocked_missing_authority_chars"
DRY_RUN_STATUS_BLOCKED_HASH_CONTRACT = "blocked_hash_contract_violation"
DRY_RUN_STATUS_BLOCKED_RUNTIME_OR_CITATION = "blocked_runtime_or_citation_flag_violation"

PLANNED_ACTION_VALIDATE_ONLY = "figure_caption_strict_evidence_pilot_validate_only"
PLANNED_WRITE_TARGET_NONE = "none"
PLANNED_RUNTIME_EFFECT_NONE = "none"
PLANNED_ANSWER_EFFECT_NONE = "none"

DEFAULT_FIGURE_CAPTION_PILOT_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "strict-evidence-figure-caption-pilot"
    / "01-strict-evidence-figure-caption-pilot"
    / "strict-evidence-figure-caption-pilot.json"
)

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "strict-evidence-figure-caption-pilot-executor-dry-run"
    / "01-strict-evidence-figure-caption-pilot-executor-dry-run"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_bool(value: Any) -> bool:
    return bool(value)


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


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


def _load_strict_evidence_record_at_store_ref(
    store_path: str,
    store_line: int,
) -> dict[str, Any]:
    path = Path(store_path).expanduser()
    if not path.is_file() or store_line <= 0:
        return {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line_number != store_line:
            continue
        text = line.strip()
        if not text:
            return {}
        try:
            payload = json.loads(text)
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def _no_mutation_policy_matrix() -> dict[str, Any]:
    return {
        "plannedAction": PLANNED_ACTION_VALIDATE_ONLY,
        "plannedWriteTarget": PLANNED_WRITE_TARGET_NONE,
        "plannedRuntimeEffect": PLANNED_RUNTIME_EFFECT_NONE,
        "plannedAnswerEffect": PLANNED_ANSWER_EFFECT_NONE,
        "writeEnabled": False,
        "strictEvidenceStoreWrite": False,
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
        "figureCaptionTextOnlyNotFigureRegion": True,
    }


def _mutation_flag_violation(
    *,
    pilot_row: dict[str, Any],
    strict_evidence_record: dict[str, Any],
) -> list[str]:
    violations: list[str] = []
    for field_name in (
        "strictEvidenceCreated",
        "citationGrade",
        "runtimeEvidence",
        "parserRoutingChanged",
        "answerIntegrationChanged",
        "databaseMutation",
        "canonicalParsedArtifactsWritten",
    ):
        if _safe_bool(pilot_row.get(field_name)):
            violations.append(f"pilot_row.{field_name}_true")
    if _safe_bool(pilot_row.get("strictEligible")):
        violations.append("pilot_row.strictEligible_true")
    if _safe_int(pilot_row.get("strictEvidenceWriteRows")):
        violations.append("pilot_row.strictEvidenceWriteRows_nonzero")
    if _safe_int(pilot_row.get("sourceSpanUpdatedRows")):
        violations.append("pilot_row.sourceSpanUpdatedRows_nonzero")
    if _safe_bool(strict_evidence_record.get("citationGrade")):
        violations.append("strict_evidence_record.citationGrade_true")
    if _safe_bool(strict_evidence_record.get("runtimeEvidence")):
        violations.append("strict_evidence_record.runtimeEvidence_true")
    if _safe_bool(strict_evidence_record.get("strictEligible")):
        violations.append("strict_evidence_record.strictEligible_true")
    write_policy = (
        strict_evidence_record.get("writePolicy")
        if isinstance(strict_evidence_record.get("writePolicy"), dict)
        else {}
    )
    for field_name in (
        "databaseMutation",
        "parserRoutingChanged",
        "answerIntegrationChanged",
        "reindexOrReembed",
        "canonicalParsedArtifactsWritten",
    ):
        if _safe_bool(write_policy.get(field_name)):
            violations.append(f"strict_evidence_record.writePolicy.{field_name}_true")
    return violations


def _authority_chars_from_sources(
    pilot_row: dict[str, Any],
    strict_evidence_record: dict[str, Any],
) -> dict[str, Any]:
    row_chars = pilot_row.get("authority_chars")
    if isinstance(row_chars, dict) and row_chars:
        return dict(row_chars)
    authority = (
        strict_evidence_record.get("authority")
        if isinstance(strict_evidence_record.get("authority"), dict)
        else {}
    )
    chars = authority.get("chars") if isinstance(authority.get("chars"), dict) else {}
    return dict(chars) if chars else {}


def _classify_dry_run_row(
    pilot_row: dict[str, Any],
    *,
    strict_evidence_record: dict[str, Any],
) -> tuple[str, list[str]]:
    pilot_status = _safe_text(pilot_row.get("pilot_status"))
    artifact_type = _safe_text(pilot_row.get("artifact_type"))
    planned_tranche = _safe_text(pilot_row.get("planned_tranche"))

    if pilot_status == PILOT_STATUS_HELD_OUT_SECTION:
        return DRY_RUN_STATUS_HELD_OUT, []
    if artifact_type == "section" or planned_tranche == TRANCHE_TEXT_SECTION_PILOT:
        return DRY_RUN_STATUS_HELD_OUT, []

    if pilot_status != PILOT_STATUS_CANDIDATE_ONLY:
        blockers = [_safe_text(item) for item in (pilot_row.get("pilot_blockers") or [])]
        blockers.append(f"pilot_status={pilot_status or 'unknown'}")
        if not _safe_bool(pilot_row.get("figureCaptionPilotCandidateOnly")):
            blockers.append("figureCaptionPilotCandidateOnly_false")
        if not _safe_bool(pilot_row.get("figureCaptionTextOnly")):
            blockers.append("figureCaptionTextOnly_false")
        return DRY_RUN_STATUS_BLOCKED_PILOT_NOT_CANDIDATE, _dedupe(blockers)

    if planned_tranche != TRANCHE_FIGURE_CAPTION_PILOT:
        return DRY_RUN_STATUS_BLOCKED_PILOT_NOT_CANDIDATE, [
            f"planned_tranche={planned_tranche or 'missing'}"
        ]

    if artifact_type != "figure":
        return DRY_RUN_STATUS_BLOCKED_INVALID_ARTIFACT, [
            f"artifact_type={artifact_type or 'missing'}"
        ]

    for field_name in ("strictEvidenceId", "sourceSpanId", "candidateRecordId", "sourceContentHash"):
        if not _safe_text(pilot_row.get(field_name)):
            return DRY_RUN_STATUS_BLOCKED_MISSING_IDENTITY, [f"{field_name}_missing"]

    flag_violations = _mutation_flag_violation(
        pilot_row=pilot_row,
        strict_evidence_record=strict_evidence_record,
    )
    if flag_violations:
        return DRY_RUN_STATUS_BLOCKED_RUNTIME_OR_CITATION, flag_violations

    chars = _authority_chars_from_sources(pilot_row, strict_evidence_record)
    start = chars.get("start")
    end = chars.get("end")
    if start is None or end is None:
        return DRY_RUN_STATUS_BLOCKED_MISSING_AUTHORITY, ["authority_chars_start_or_end_missing"]

    basis = _safe_text(chars.get("basis"))
    if basis != CHARS_BASIS:
        return DRY_RUN_STATUS_BLOCKED_MISSING_AUTHORITY, [
            f"authority_chars_basis_invalid:{basis or 'missing'}"
        ]

    normalization = _safe_text(chars.get("normalization"))
    if normalization != CHARS_NORMALIZATION_LABEL:
        return DRY_RUN_STATUS_BLOCKED_MISSING_AUTHORITY, [
            f"authority_chars_normalization_invalid:{normalization or 'missing'}"
        ]

    verbatim_hash = _safe_text(
        pilot_row.get("verbatimSubstringSha256")
        or strict_evidence_record.get("verbatimSubstringSha256")
    )
    expected_hash = _safe_text(chars.get("expectedSubstringSha256"))
    if not verbatim_hash or not expected_hash:
        return DRY_RUN_STATUS_BLOCKED_HASH_CONTRACT, ["verbatim_or_expected_hash_missing"]
    if verbatim_hash != expected_hash:
        return DRY_RUN_STATUS_BLOCKED_HASH_CONTRACT, [
            "verbatimSubstringSha256_must_equal_authority_chars_expectedSubstringSha256"
        ]

    return DRY_RUN_STATUS_READY, []


def _planned_executor_key(pilot_row: dict[str, Any]) -> str:
    strict_evidence_id = _safe_text(pilot_row.get("strictEvidenceId"))
    idempotency_key = _safe_text(pilot_row.get("idempotencyKey"))
    if strict_evidence_id:
        return f"figure-caption-pilot-executor:{strict_evidence_id}"
    return f"figure-caption-pilot-executor:{idempotency_key or 'unknown'}"


def _dry_run_rows(pilot_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matrix = _no_mutation_policy_matrix()
    rows: list[dict[str, Any]] = []
    for index, pilot_row in enumerate(pilot_rows):
        source_row = dict(pilot_row or {})
        strict_evidence_record = _load_strict_evidence_record_at_store_ref(
            _safe_text(source_row.get("strict_evidence_store_path")),
            _safe_int(source_row.get("strict_evidence_store_line")),
        )
        dry_run_status, blockers = _classify_dry_run_row(
            source_row,
            strict_evidence_record=strict_evidence_record,
        )
        ready = dry_run_status == DRY_RUN_STATUS_READY
        chars = _authority_chars_from_sources(source_row, strict_evidence_record)
        rows.append(
            {
                "dry_run_row_id": f"strict-evidence-figure-caption-pilot-executor-dry-run:{index:04d}",
                "pilot_row_id": _safe_text(source_row.get("pilot_row_id")),
                "plan_row_id": _safe_text(source_row.get("plan_row_id")),
                "strictEvidenceId": _safe_text(source_row.get("strictEvidenceId")),
                "sourceSpanId": _safe_text(source_row.get("sourceSpanId")),
                "candidateRecordId": _safe_text(source_row.get("candidateRecordId")),
                "runId": _safe_text(source_row.get("runId")),
                "paper_id": _safe_text(source_row.get("paper_id")),
                "artifact_type": _safe_text(source_row.get("artifact_type")),
                "sourceContentHash": _safe_text(source_row.get("sourceContentHash")),
                "idempotencyKey": _safe_text(source_row.get("idempotencyKey")),
                "strict_evidence_store_path": _safe_text(source_row.get("strict_evidence_store_path")),
                "strict_evidence_store_line": _safe_int(source_row.get("strict_evidence_store_line")),
                "planned_tranche": _safe_text(source_row.get("planned_tranche")),
                "planned_tranche_scope": _safe_text(source_row.get("planned_tranche_scope")),
                "pilot_status": _safe_text(source_row.get("pilot_status")),
                "figureCaptionPilotCandidateOnly": _safe_bool(
                    source_row.get("figureCaptionPilotCandidateOnly")
                ),
                "figureCaptionTextOnly": _safe_bool(source_row.get("figureCaptionTextOnly")),
                "verbatimSubstringSha256": _safe_text(
                    source_row.get("verbatimSubstringSha256")
                    or strict_evidence_record.get("verbatimSubstringSha256")
                ),
                "authority_chars": {
                    "start": chars.get("start"),
                    "end": chars.get("end"),
                    "basis": _safe_text(chars.get("basis")),
                    "normalization": _safe_text(chars.get("normalization")),
                    "expectedSubstringSha256": _safe_text(chars.get("expectedSubstringSha256")),
                },
                "planned_executor_key": _planned_executor_key(source_row),
                "plannedAction": PLANNED_ACTION_VALIDATE_ONLY if ready else "",
                "plannedWriteTarget": PLANNED_WRITE_TARGET_NONE,
                "plannedRuntimeEffect": PLANNED_RUNTIME_EFFECT_NONE,
                "plannedAnswerEffect": PLANNED_ANSWER_EFFECT_NONE,
                "dry_run_status": dry_run_status,
                "dry_run_blockers": _dedupe(blockers),
                "dryRunReadyFigureCaptionPilotExecutorOnly": ready,
                "writeMatrix": dict(matrix),
                "strictEligible": False,
                "strictEvidenceWriteRows": 0,
                "strictEvidenceCreated": False,
                "citationGrade": False,
                "runtimeEvidence": False,
                "parserRoutingChanged": False,
                "answerIntegrationChanged": False,
                "databaseMutation": False,
                "canonicalParsedArtifactsWritten": False,
                "sourceSpanUpdatedRows": 0,
                "recommended_action": (
                    "queue_for_strict_evidence_figure_caption_pilot_executor_apply"
                    if ready
                    else (
                        "held_for_section_tranche_diagnostics"
                        if dry_run_status == DRY_RUN_STATUS_HELD_OUT
                        else "repair_figure_caption_pilot_row_before_executor_dry_run"
                    )
                ),
            }
        )
    return rows


def _held_out_section_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    held_rows = [row for row in rows if row.get("dry_run_status") == DRY_RUN_STATUS_HELD_OUT]
    return {
        "heldOutSectionRows": len(held_rows),
        "byPlannedTranche": dict(Counter(_safe_text(row.get("planned_tranche")) for row in held_rows)),
        "byArtifactType": dict(Counter(_safe_text(row.get("artifact_type")) for row in held_rows)),
        "diagnosticOnly": True,
        "activeExecutorProcessing": False,
    }


def _count_rows(
    *,
    rows: list[dict[str, Any]],
    input_schema_violations: list[str],
    figure_caption_pilot_candidate_rows: int,
) -> dict[str, Any]:
    return {
        "inputRows": len(rows),
        "figureCaptionPilotCandidateRows": figure_caption_pilot_candidate_rows,
        "dryRunReadyFigureCaptionPilotExecutorOnlyRows": sum(
            1 for row in rows if row.get("dry_run_status") == DRY_RUN_STATUS_READY
        ),
        "heldOutSectionRows": sum(
            1 for row in rows if row.get("dry_run_status") == DRY_RUN_STATUS_HELD_OUT
        ),
        "blockedInputReportSchemaViolationRows": (
            len(rows) if input_schema_violations and rows else int(bool(input_schema_violations))
        ),
        "blockedFigureCaptionPilotNotCandidateRows": sum(
            1
            for row in rows
            if row.get("dry_run_status") == DRY_RUN_STATUS_BLOCKED_PILOT_NOT_CANDIDATE
        ),
        "blockedMissingRecordIdentityRows": sum(
            1 for row in rows if row.get("dry_run_status") == DRY_RUN_STATUS_BLOCKED_MISSING_IDENTITY
        ),
        "blockedInvalidArtifactTypeRows": sum(
            1 for row in rows if row.get("dry_run_status") == DRY_RUN_STATUS_BLOCKED_INVALID_ARTIFACT
        ),
        "blockedMissingAuthorityCharsRows": sum(
            1 for row in rows if row.get("dry_run_status") == DRY_RUN_STATUS_BLOCKED_MISSING_AUTHORITY
        ),
        "blockedHashContractViolationRows": sum(
            1 for row in rows if row.get("dry_run_status") == DRY_RUN_STATUS_BLOCKED_HASH_CONTRACT
        ),
        "blockedRuntimeOrCitationFlagViolationRows": sum(
            1 for row in rows if row.get("dry_run_status") == DRY_RUN_STATUS_BLOCKED_RUNTIME_OR_CITATION
        ),
        "strictEvidenceWriteRows": 0,
        "strictEvidenceCreatedRows": 0,
        "citationGradeEvidenceCreatedRows": 0,
        "runtimeEvidenceCreatedRows": 0,
        "parserRoutingChangedRows": 0,
        "answerIntegrationChangedRows": 0,
        "databaseMutationRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
        "sourceSpanUpdatedRows": 0,
        "schemaViolationCount": len(input_schema_violations),
        "byPaperId": dict(
            Counter(
                _safe_text(row.get("paper_id"))
                for row in rows
                if row.get("dry_run_status") == DRY_RUN_STATUS_READY
            )
        ),
        "byDryRunStatus": dict(Counter(_safe_text(row.get("dry_run_status")) for row in rows)),
        "byRecommendedAction": dict(Counter(_safe_text(row.get("recommended_action")) for row in rows)),
    }


def build_strict_evidence_figure_caption_pilot_executor_dry_run(
    *,
    figure_caption_pilot_report_path: str | Path = DEFAULT_FIGURE_CAPTION_PILOT_REPORT_PATH,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    report_path = Path(str(figure_caption_pilot_report_path)).expanduser()
    warnings: list[str] = []
    input_schema_violations: list[str] = []
    requested_papers = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}

    figure_caption_pilot_report = _read_json(report_path)
    if not figure_caption_pilot_report:
        warnings.append("figure_caption_pilot_report_missing_or_unreadable")

    validation = validate_payload(
        figure_caption_pilot_report,
        STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        input_schema_violations = [str(error) for error in validation.errors]
        if not figure_caption_pilot_report:
            input_schema_violations.append("figure_caption_pilot_report_missing_or_unreadable")

    if figure_caption_pilot_report and _safe_text(figure_caption_pilot_report.get("status")) != "ok":
        input_schema_violations.append(
            f"figure_caption_pilot_report_status={_safe_text(figure_caption_pilot_report.get('status')) or 'unknown'}"
        )

    all_pilot_rows = [
        row for row in figure_caption_pilot_report.get("rows", []) if isinstance(row, dict)
    ] if isinstance(figure_caption_pilot_report, dict) else []

    figure_caption_pilot_candidate_rows = int(
        (figure_caption_pilot_report.get("counts") or {}).get("figureCaptionPilotCandidateOnlyRows")
        or sum(
            1
            for row in all_pilot_rows
            if _safe_bool(row.get("figureCaptionPilotCandidateOnly"))
            and _safe_text(row.get("pilot_status")) == PILOT_STATUS_CANDIDATE_ONLY
        )
    ) if figure_caption_pilot_report else 0

    if requested_papers:
        found_papers = {_safe_text(row.get("paper_id")) for row in all_pilot_rows if _safe_text(row.get("paper_id"))}
        if requested_papers - found_papers:
            warnings.append("requested_paper_ids_not_found")
        all_pilot_rows = [row for row in all_pilot_rows if _safe_text(row.get("paper_id")) in requested_papers]

    if not all_pilot_rows and not input_schema_violations:
        warnings.append("figure_caption_pilot_rows_missing")

    rows = _dry_run_rows(all_pilot_rows)
    if input_schema_violations:
        for row in rows:
            if row.get("dry_run_status") == DRY_RUN_STATUS_HELD_OUT:
                continue
            row["dry_run_status"] = DRY_RUN_STATUS_BLOCKED_INPUT_SCHEMA
            row["dry_run_blockers"] = _dedupe([*row.get("dry_run_blockers", []), *input_schema_violations])
            row["dryRunReadyFigureCaptionPilotExecutorOnly"] = False
            row["plannedAction"] = ""
            row["recommended_action"] = "repair_figure_caption_pilot_report_schema_before_executor_dry_run"

    counts = _count_rows(
        rows=rows,
        input_schema_violations=_dedupe(input_schema_violations),
        figure_caption_pilot_candidate_rows=figure_caption_pilot_candidate_rows,
    )
    ready_rows = int(counts.get("dryRunReadyFigureCaptionPilotExecutorOnlyRows") or 0)
    held_out_rows = int(counts.get("heldOutSectionRows") or 0)
    status = "ok"
    if (
        input_schema_violations
        or not rows
        or ready_rows != figure_caption_pilot_candidate_rows
        or ready_rows + held_out_rows != len(rows)
    ):
        status = "blocked"

    policy_matrix = _no_mutation_policy_matrix()
    held_out_summary = _held_out_section_summary(rows)

    return {
        "schema": STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_EXECUTOR_DRY_RUN_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "figureCaptionPilotReportPath": str(report_path),
            "figureCaptionPilotSchema": _safe_text(figure_caption_pilot_report.get("schema"))
            if figure_caption_pilot_report
            else "",
            "figureCaptionPilotReportStatus": _safe_text(figure_caption_pilot_report.get("status"))
            if figure_caption_pilot_report
            else "",
            "requestedPaperIds": sorted(requested_papers),
            "targetTranche": TRANCHE_FIGURE_CAPTION_PILOT,
        },
        "counts": counts,
        "heldOutSection": held_out_summary,
        "noMutationPolicyMatrix": policy_matrix,
        "gate": {
            "dryRunReadyForFigureCaptionPilotExecutor": (
                bool(ready_rows)
                and ready_rows == figure_caption_pilot_candidate_rows
                and not input_schema_violations
            ),
            "strictEligibleMutationAllowed": False,
            "strictEvidenceStoreWriteAllowed": False,
            "applyMode": False,
            "strictEvidenceCreated": False,
            "citationReady": False,
            "runtimeEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "sectionRowsProcessedAsActiveExecutor": False,
            "schemaViolations": _dedupe(input_schema_violations),
            "decision": (
                "strict_evidence_figure_caption_pilot_executor_dry_run_ready"
                if status == "ok"
                else "blocked"
            ),
            "recommendedNextTranche": (
                "strict_evidence_figure_caption_pilot_executor_apply"
                if status == "ok"
                else "strict_evidence_figure_caption_pilot_repair"
            ),
        },
        "policy": {
            "reportOnly": True,
            "dryRunOnly": True,
            "figureCaptionTextOnlyNotFigureRegion": True,
            "strictEvidenceStoreWrite": False,
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
            "heldOutSection",
            "noMutationPolicyMatrix",
            "gate",
            "policy",
            "warnings",
            "rows",
        )
        if key in report
    }


def render_strict_evidence_figure_caption_pilot_executor_dry_run_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    held_out = dict(report.get("heldOutSection") or {})
    matrix = dict(report.get("noMutationPolicyMatrix") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byDryRunStatus") or {})).items())
    ]
    return "\n".join(
        [
            "# Strict Evidence Figure Caption Pilot Executor Dry Run",
            "",
            f"- status: {report.get('status', '')}",
            f"- report-only: {json.dumps(report.get('policy', {}).get('reportOnly'))}",
            f"- figure caption text only: {json.dumps(report.get('policy', {}).get('figureCaptionTextOnlyNotFigureRegion'))}",
            f"- input rows: {int(counts.get('inputRows') or 0)}",
            f"- dry-run ready figure caption rows: {int(counts.get('dryRunReadyFigureCaptionPilotExecutorOnlyRows') or 0)}",
            f"- held-out section rows: {int(counts.get('heldOutSectionRows') or 0)}",
            f"- strict evidence writes: {int(counts.get('strictEvidenceWriteRows') or 0)}",
            "",
            "## No-mutation policy matrix",
            f"- planned action: {matrix.get('plannedAction', '')}",
            f"- planned write target: {matrix.get('plannedWriteTarget', '')}",
            f"- planned runtime effect: {matrix.get('plannedRuntimeEffect', '')}",
            f"- planned answer effect: {matrix.get('plannedAnswerEffect', '')}",
            "",
            "## Held-out section summary",
            f"- held-out rows: {int(held_out.get('heldOutSectionRows') or 0)}",
            f"- by planned tranche: {json.dumps(held_out.get('byPlannedTranche') or {})}",
            "",
            "## Dry-run status breakdown",
            *[f"- {item}" for item in by_status],
            "",
            f"- recommended next tranche: {report.get('gate', {}).get('recommendedNextTranche', '')}",
        ]
    )


def write_strict_evidence_figure_caption_pilot_executor_dry_run_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "strict-evidence-figure-caption-pilot-executor-dry-run.json"
    summary_path = root / "strict-evidence-figure-caption-pilot-executor-dry-run-summary.json"
    markdown_path = root / "strict-evidence-figure-caption-pilot-executor-dry-run.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_strict_evidence_figure_caption_pilot_executor_dry_run_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Plan validate-only figure-caption pilot executor actions from figure-caption-pilot "
            "rows without mutating records or integration surfaces."
        )
    )
    parser.add_argument(
        "--figure-caption-pilot-report",
        default=str(DEFAULT_FIGURE_CAPTION_PILOT_REPORT_PATH),
        help="Figure-caption pilot JSON report.",
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_strict_evidence_figure_caption_pilot_executor_dry_run(
        figure_caption_pilot_report_path=args.figure_caption_pilot_report,
        paper_ids=args.paper_id or None,
    )

    if args.output_dir:
        paths = write_strict_evidence_figure_caption_pilot_executor_dry_run_reports(
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
    "STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_EXECUTOR_DRY_RUN_SCHEMA_ID",
    "DRY_RUN_STATUS_READY",
    "DRY_RUN_STATUS_HELD_OUT",
    "PLANNED_ACTION_VALIDATE_ONLY",
    "build_strict_evidence_figure_caption_pilot_executor_dry_run",
    "render_strict_evidence_figure_caption_pilot_executor_dry_run_markdown",
    "write_strict_evidence_figure_caption_pilot_executor_dry_run_reports",
]
