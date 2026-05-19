"""Apply-gated manifest writer for the StrictEvidence text-section pilot executor.

Consumes the text-section pilot executor dry-run report, validates section dry-run-ready
rows, and records manifest-only pilot apply outcomes. StrictEvidence and SourceSpan
stores remain read-only unless future tranches explicitly widen scope.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import re

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_strict_evidence_promotion_tranche_plan import (
    TRANCHE_FIGURE_CAPTION_PILOT,
    TRANCHE_TEXT_SECTION_PILOT,
)
from knowledge_hub.papers.strict_evidence_text_section_pilot_executor_dry_run import (
    DRY_RUN_STATUS_HELD_OUT,
    DRY_RUN_STATUS_READY,
    PLANNED_ACTION_VALIDATE_ONLY,
    PLANNED_ANSWER_EFFECT_NONE,
    PLANNED_RUNTIME_EFFECT_NONE,
    PLANNED_WRITE_TARGET_NONE,
    STRICT_EVIDENCE_TEXT_SECTION_PILOT_EXECUTOR_DRY_RUN_SCHEMA_ID,
)


STRICT_EVIDENCE_TEXT_SECTION_PILOT_EXECUTOR_APPLY_SCHEMA_ID = (
    "knowledge-hub.paper.strict-evidence-text-section-pilot-executor-apply.v1"
)

APPLY_STATUS_READY = "apply_ready_text_section_pilot_validate_only"
APPLY_STATUS_APPLIED = "applied_text_section_pilot_manifest_only"
APPLY_STATUS_HELD_OUT = "held_out_non_section_tranche"
APPLY_STATUS_BLOCKED_DRY_RUN_NOT_READY = "blocked_dry_run_not_ready"
APPLY_STATUS_BLOCKED_MISSING_IDENTITY = "blocked_missing_record_identity"
APPLY_STATUS_BLOCKED_INVALID_ARTIFACT = "blocked_invalid_artifact_type"
APPLY_STATUS_BLOCKED_RUNTIME_OR_CITATION = "blocked_runtime_or_citation_flag_violation"
APPLY_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"

MANIFEST_WRITE_TARGET = "structured_evidence_runs_manifest"

DEFAULT_EXECUTOR_DRY_RUN_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "strict-evidence-text-section-pilot-executor-dry-run"
    / "01-strict-evidence-text-section-pilot-executor-dry-run"
    / "strict-evidence-text-section-pilot-executor-dry-run.json"
)

DEFAULT_REPORT_ROOT = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "strict-evidence-text-section-pilot-executor-apply"
)

DEFAULT_DRY_RUN_OUTPUT_DIR = (
    DEFAULT_REPORT_ROOT / "01-strict-evidence-text-section-pilot-executor-apply-dry-run"
)
DEFAULT_APPLY_OUTPUT_DIR = (
    DEFAULT_REPORT_ROOT / "02-strict-evidence-text-section-pilot-executor-apply"
)

DEFAULT_OUTPUT_DIR = DEFAULT_DRY_RUN_OUTPUT_DIR


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


def _safe_filename(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return text.strip("._") or "unknown-run"


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    payload_path = Path(str(path)).expanduser()
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _run_manifest_path(papers_dir: str | Path, run_id: str) -> Path:
    return (
        Path(str(papers_dir)).expanduser()
        / "structured_evidence"
        / "runs"
        / f"{_safe_filename(run_id)}.json"
    )


def default_output_dir(*, apply: bool) -> Path:
    return DEFAULT_APPLY_OUTPUT_DIR if apply else DEFAULT_DRY_RUN_OUTPUT_DIR


def resolve_output_dir(output_dir: str | Path | None, *, apply: bool) -> Path:
    if output_dir:
        return Path(str(output_dir)).expanduser()
    return default_output_dir(apply=apply)


def _manifest_only_policy_matrix() -> dict[str, Any]:
    return {
        "plannedAction": PLANNED_ACTION_VALIDATE_ONLY,
        "plannedWriteTarget": MANIFEST_WRITE_TARGET,
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
        "manifestOnlyApply": True,
    }


def _mutation_flag_violation(dry_row: dict[str, Any]) -> list[str]:
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
        if _safe_bool(dry_row.get(field_name)):
            violations.append(f"dry_run_row.{field_name}_true")
    if _safe_bool(dry_row.get("strictEligible")):
        violations.append("dry_run_row.strictEligible_true")
    if _safe_int(dry_row.get("strictEvidenceWriteRows")):
        violations.append("dry_run_row.strictEvidenceWriteRows_nonzero")
    if _safe_int(dry_row.get("sourceSpanUpdatedRows")):
        violations.append("dry_run_row.sourceSpanUpdatedRows_nonzero")
    write_matrix = dry_row.get("writeMatrix") if isinstance(dry_row.get("writeMatrix"), dict) else {}
    for field_name in (
        "strictEvidenceStoreWrite",
        "sourceSpanStoreWrite",
        "strictEvidenceCreated",
        "strictEligibleMutation",
        "citationGradeEvidenceCreated",
        "runtimeEvidenceCreated",
        "parserRoutingChanged",
        "answerIntegrationChanged",
        "databaseMutation",
        "canonicalParsedArtifactsWritten",
    ):
        if _safe_bool(write_matrix.get(field_name)):
            violations.append(f"writeMatrix.{field_name}_true")
    return violations


def _classify_apply_row(
    dry_row: dict[str, Any],
    *,
    input_schema_violations: list[str],
) -> tuple[str, list[str], bool]:
    dry_run_status = _safe_text(dry_row.get("dry_run_status"))
    artifact_type = _safe_text(dry_row.get("artifact_type"))
    planned_tranche = _safe_text(dry_row.get("planned_tranche"))

    if input_schema_violations:
        if dry_run_status == DRY_RUN_STATUS_HELD_OUT:
            return APPLY_STATUS_HELD_OUT, [], False
        return APPLY_STATUS_BLOCKED_INPUT_SCHEMA, list(input_schema_violations), False

    if dry_run_status == DRY_RUN_STATUS_HELD_OUT:
        return APPLY_STATUS_HELD_OUT, [], False
    if artifact_type == "figure" or planned_tranche == TRANCHE_FIGURE_CAPTION_PILOT:
        return APPLY_STATUS_HELD_OUT, [], False

    if dry_run_status != DRY_RUN_STATUS_READY:
        blockers = [_safe_text(item) for item in (dry_row.get("dry_run_blockers") or [])]
        blockers.append(f"dry_run_status={dry_run_status or 'unknown'}")
        if not _safe_bool(dry_row.get("dryRunReadyTextSectionPilotExecutorOnly")):
            blockers.append("dryRunReadyTextSectionPilotExecutorOnly_false")
        return APPLY_STATUS_BLOCKED_DRY_RUN_NOT_READY, _dedupe(blockers), False

    if planned_tranche != TRANCHE_TEXT_SECTION_PILOT:
        return APPLY_STATUS_BLOCKED_DRY_RUN_NOT_READY, [
            f"planned_tranche={planned_tranche or 'missing'}"
        ], False

    if artifact_type != "section":
        return APPLY_STATUS_BLOCKED_INVALID_ARTIFACT, [
            f"artifact_type={artifact_type or 'missing'}"
        ], False

    for field_name in ("strictEvidenceId", "sourceSpanId", "candidateRecordId", "sourceContentHash"):
        if not _safe_text(dry_row.get(field_name)):
            return APPLY_STATUS_BLOCKED_MISSING_IDENTITY, [f"{field_name}_missing"], False

    flag_violations = _mutation_flag_violation(dry_row)
    if flag_violations:
        return APPLY_STATUS_BLOCKED_RUNTIME_OR_CITATION, flag_violations, False

    return APPLY_STATUS_READY, [], True


def _apply_rows(
    dry_rows: list[dict[str, Any]],
    *,
    input_schema_violations: list[str],
    apply_mode: bool,
) -> list[dict[str, Any]]:
    matrix = _manifest_only_policy_matrix()
    rows: list[dict[str, Any]] = []
    for index, dry_row in enumerate(dry_rows):
        source_row = dict(dry_row or {})
        apply_status, blockers, ready = _classify_apply_row(
            source_row,
            input_schema_violations=input_schema_violations,
        )
        applied = apply_mode and ready and apply_status == APPLY_STATUS_READY
        if applied:
            apply_status = APPLY_STATUS_APPLIED

        rows.append(
            {
                "apply_row_id": f"strict-evidence-text-section-pilot-executor-apply:{index:04d}",
                "dry_run_row_id": _safe_text(source_row.get("dry_run_row_id")),
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
                "dry_run_status": _safe_text(source_row.get("dry_run_status")),
                "planned_executor_key": _safe_text(source_row.get("planned_executor_key")),
                "plannedAction": PLANNED_ACTION_VALIDATE_ONLY if ready or applied else "",
                "plannedWriteTarget": MANIFEST_WRITE_TARGET if applied else PLANNED_WRITE_TARGET_NONE,
                "plannedRuntimeEffect": PLANNED_RUNTIME_EFFECT_NONE,
                "plannedAnswerEffect": PLANNED_ANSWER_EFFECT_NONE,
                "apply_status": apply_status,
                "apply_blockers": _dedupe(blockers),
                "sectionDryRunReady": ready or applied,
                "appliedManifestOnly": applied,
                "wouldWriteRunManifest": ready and not apply_mode,
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
                "manifestWriteRows": 1 if applied else 0,
                "recommended_action": (
                    "run_manifest_only_apply_with_explicit_apply_flag"
                    if ready and not apply_mode
                    else (
                        "manifest_only_apply_recorded"
                        if applied
                        else (
                            "held_for_figure_caption_pilot_tranche"
                            if apply_status == APPLY_STATUS_HELD_OUT
                            else "repair_dry_run_row_before_section_pilot_executor_apply"
                        )
                    )
                ),
            }
        )
    return rows


def _held_out_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    held_rows = [row for row in rows if row.get("apply_status") == APPLY_STATUS_HELD_OUT]
    return {
        "heldOutNonSectionRows": len(held_rows),
        "byPlannedTranche": dict(Counter(_safe_text(row.get("planned_tranche")) for row in held_rows)),
        "byArtifactType": dict(Counter(_safe_text(row.get("artifact_type")) for row in held_rows)),
        "diagnosticOnly": True,
        "activeApplyProcessing": False,
    }


def _count_rows(
    *,
    rows: list[dict[str, Any]],
    input_schema_violations: list[str],
    section_dry_run_ready_rows: int,
    apply_mode: bool,
    manifest_write_rows: int,
) -> dict[str, Any]:
    section_ready = sum(1 for row in rows if row.get("sectionDryRunReady"))
    return {
        "inputRows": len(rows),
        "sectionDryRunReadyRows": section_ready,
        "plannedApplyRows": sum(
            1 for row in rows if _safe_text(row.get("apply_status")) == APPLY_STATUS_READY
        ),
        "appliedManifestOnlyRows": sum(
            1 for row in rows if _safe_text(row.get("apply_status")) == APPLY_STATUS_APPLIED
        ),
        "heldOutNonSectionRows": sum(
            1 for row in rows if _safe_text(row.get("apply_status")) == APPLY_STATUS_HELD_OUT
        ),
        "blockedDryRunNotReadyRows": sum(
            1
            for row in rows
            if _safe_text(row.get("apply_status")) == APPLY_STATUS_BLOCKED_DRY_RUN_NOT_READY
        ),
        "blockedMissingRecordIdentityRows": sum(
            1
            for row in rows
            if _safe_text(row.get("apply_status")) == APPLY_STATUS_BLOCKED_MISSING_IDENTITY
        ),
        "blockedInvalidArtifactTypeRows": sum(
            1
            for row in rows
            if _safe_text(row.get("apply_status")) == APPLY_STATUS_BLOCKED_INVALID_ARTIFACT
        ),
        "blockedRuntimeOrCitationFlagViolationRows": sum(
            1
            for row in rows
            if _safe_text(row.get("apply_status")) == APPLY_STATUS_BLOCKED_RUNTIME_OR_CITATION
        ),
        "blockedInputSchemaViolationRows": sum(
            1
            for row in rows
            if _safe_text(row.get("apply_status")) == APPLY_STATUS_BLOCKED_INPUT_SCHEMA
        ),
        "manifestWriteRows": manifest_write_rows,
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
        "sectionPilotExecutorDryRunReadyRows": section_dry_run_ready_rows,
        "byPaperId": dict(
            Counter(
                _safe_text(row.get("paper_id"))
                for row in rows
                if row.get("sectionDryRunReady")
            )
        ),
        "byApplyStatus": dict(Counter(_safe_text(row.get("apply_status")) for row in rows)),
        "byRecommendedAction": dict(
            Counter(_safe_text(row.get("recommended_action")) for row in rows)
        ),
    }


def build_strict_evidence_text_section_pilot_executor_apply(
    *,
    executor_dry_run_report_path: str | Path = DEFAULT_EXECUTOR_DRY_RUN_REPORT_PATH,
    papers_dir: str | Path | None = None,
    run_id: str | None = None,
    apply: bool = False,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    dry_run_path = Path(str(executor_dry_run_report_path)).expanduser()
    warnings: list[str] = []
    input_schema_violations: list[str] = []
    requested_papers = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}

    dry_run_report = _read_json(dry_run_path)
    if not dry_run_report:
        warnings.append("executor_dry_run_report_missing_or_unreadable")
        input_schema_violations.append("executor_dry_run_report_missing_or_unreadable")

    if dry_run_report:
        validation = validate_payload(
            dry_run_report,
            STRICT_EVIDENCE_TEXT_SECTION_PILOT_EXECUTOR_DRY_RUN_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            input_schema_violations.extend(str(error) for error in validation.errors)
        if _safe_text(dry_run_report.get("status")) != "ok":
            input_schema_violations.append(
                f"executor_dry_run_report_status={_safe_text(dry_run_report.get('status')) or 'unknown'}"
            )
        gate = dry_run_report.get("gate") if isinstance(dry_run_report.get("gate"), dict) else {}
        if not _safe_bool(gate.get("dryRunReadyForTextSectionPilotExecutor")):
            input_schema_violations.append("executor_dry_run_not_ready_for_apply")

    if apply and not papers_dir:
        input_schema_violations.append("apply_requires_papers_dir")

    section_dry_run_ready_rows = int(
        (dry_run_report.get("counts") or {}).get("dryRunReadyTextSectionPilotExecutorOnlyRows")
        or 0
    ) if dry_run_report else 0

    all_dry_rows = [
        row for row in dry_run_report.get("rows", []) if isinstance(row, dict)
    ] if isinstance(dry_run_report, dict) else []

    if requested_papers:
        found_papers = {_safe_text(row.get("paper_id")) for row in all_dry_rows if _safe_text(row.get("paper_id"))}
        if requested_papers - found_papers:
            warnings.append("requested_paper_ids_not_found")
        all_dry_rows = [row for row in all_dry_rows if _safe_text(row.get("paper_id")) in requested_papers]

    if not all_dry_rows and not input_schema_violations:
        warnings.append("executor_dry_run_rows_missing")

    run_id = _safe_text(run_id) or (
        f"strict-evidence-text-section-pilot-executor-apply-{_now_iso()}"
    )

    input_schema_violations = _dedupe(input_schema_violations)
    rows = _apply_rows(
        all_dry_rows,
        input_schema_violations=input_schema_violations,
        apply_mode=bool(apply),
    )

    manifest_write_rows = 0
    manifest_path = ""
    counts = _count_rows(
        rows=rows,
        input_schema_violations=input_schema_violations,
        section_dry_run_ready_rows=section_dry_run_ready_rows,
        apply_mode=bool(apply),
        manifest_write_rows=manifest_write_rows,
    )

    planned_apply_rows = int(counts.get("plannedApplyRows") or 0)
    applied_manifest_rows = int(counts.get("appliedManifestOnlyRows") or 0)
    held_out_rows = int(counts.get("heldOutNonSectionRows") or 0)
    section_ready_rows = int(counts.get("sectionDryRunReadyRows") or 0)

    status = "ok"
    if (
        input_schema_violations
        or not rows
        or section_ready_rows != section_dry_run_ready_rows
        or section_ready_rows + held_out_rows != len(rows)
        or (bool(apply) and applied_manifest_rows != section_dry_run_ready_rows)
        or (not bool(apply) and planned_apply_rows != section_dry_run_ready_rows)
    ):
        status = "blocked"

    policy_matrix = _manifest_only_policy_matrix()
    held_out_summary = _held_out_summary(rows)

    report = {
        "schema": STRICT_EVIDENCE_TEXT_SECTION_PILOT_EXECUTOR_APPLY_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "executorDryRunReportPath": str(dry_run_path),
            "executorDryRunSchema": _safe_text(dry_run_report.get("schema")) if dry_run_report else "",
            "executorDryRunReportStatus": _safe_text(dry_run_report.get("status")) if dry_run_report else "",
            "requestedPaperIds": sorted(requested_papers),
            "papersDir": str(Path(str(papers_dir)).expanduser()) if papers_dir else "",
            "runId": run_id,
            "apply": bool(apply),
            "targetTranche": TRANCHE_TEXT_SECTION_PILOT,
        },
        "counts": counts,
        "heldOutNonSection": held_out_summary,
        "manifestOnlyPolicyMatrix": policy_matrix,
        "gate": {
            "readyForDryRunApplyPlanning": (
                status == "ok"
                and not apply
                and planned_apply_rows == section_dry_run_ready_rows
                and not input_schema_violations
            ),
            "readyForManifestOnlyApply": (
                status == "ok"
                and bool(section_dry_run_ready_rows)
                and not input_schema_violations
            ),
            "applyMode": bool(apply),
            "strictEligibleMutationAllowed": False,
            "strictEvidenceStoreWriteAllowed": False,
            "sourceSpanStoreWriteAllowed": False,
            "runManifestWriteAllowed": bool(apply and papers_dir and status == "ok"),
            "strictEvidenceCreated": False,
            "citationReady": False,
            "runtimeEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "figureRowsProcessedAsActiveApply": False,
            "schemaViolations": input_schema_violations,
            "decision": (
                "strict_evidence_text_section_pilot_executor_apply_ready"
                if status == "ok" and not apply
                else (
                    "strict_evidence_text_section_pilot_executor_applied_manifest_only"
                    if status == "ok" and apply
                    else "blocked"
                )
            ),
            "recommendedNextTranche": (
                "strict_evidence_figure_caption_pilot"
                if status == "ok" and apply
                else "strict_evidence_text_section_pilot_executor_apply_review"
            ),
        },
        "policy": {
            "reportOnly": not apply,
            "dryRunByDefault": True,
            "manifestOnlyApply": True,
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
            "runManifestWrite": bool(manifest_write_rows),
        },
        "warnings": _dedupe(warnings),
        "rows": rows,
    }

    if apply and status == "ok" and papers_dir:
        manifest_path_obj = _run_manifest_path(papers_dir, run_id)
        manifest_path_obj.parent.mkdir(parents=True, exist_ok=True)
        manifest_write_rows = 1
        manifest_path = str(manifest_path_obj)
        manifest = _summary_payload(report)
        manifest_path_obj.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report["counts"]["manifestWriteRows"] = manifest_write_rows
        report["policy"]["runManifestWrite"] = True
        report["gate"]["runManifestPath"] = manifest_path

    return report


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in (
            "schema",
            "status",
            "generatedAt",
            "input",
            "counts",
            "heldOutNonSection",
            "manifestOnlyPolicyMatrix",
            "gate",
            "policy",
            "warnings",
            "rows",
        )
        if key in report
    }


def render_strict_evidence_text_section_pilot_executor_apply_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    held_out = dict(report.get("heldOutNonSection") or {})
    matrix = dict(report.get("manifestOnlyPolicyMatrix") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byApplyStatus") or {})).items())
    ]
    return "\n".join(
        [
            "# Strict Evidence Text Section Pilot Executor Apply",
            "",
            f"- status: {report.get('status', '')}",
            f"- apply mode: {json.dumps(report.get('input', {}).get('apply'))}",
            f"- input rows: {int(counts.get('inputRows') or 0)}",
            f"- section dry-run ready rows: {int(counts.get('sectionDryRunReadyRows') or 0)}",
            f"- planned apply rows: {int(counts.get('plannedApplyRows') or 0)}",
            f"- applied manifest-only rows: {int(counts.get('appliedManifestOnlyRows') or 0)}",
            f"- manifest writes: {int(counts.get('manifestWriteRows') or 0)}",
            f"- strict evidence writes: {int(counts.get('strictEvidenceWriteRows') or 0)}",
            f"- held-out non-section rows: {int(counts.get('heldOutNonSectionRows') or 0)}",
            "",
            "## Manifest-only policy matrix",
            f"- planned action: {matrix.get('plannedAction', '')}",
            f"- planned write target: {matrix.get('plannedWriteTarget', '')}",
            f"- manifest-only apply: {json.dumps(matrix.get('manifestOnlyApply'))}",
            "",
            "## Held-out non-section summary",
            f"- held-out rows: {int(held_out.get('heldOutNonSectionRows') or 0)}",
            f"- by planned tranche: {json.dumps(held_out.get('byPlannedTranche') or {})}",
            "",
            "## Apply status breakdown",
            *[f"- {item}" for item in by_status],
            "",
            f"- recommended next tranche: {report.get('gate', {}).get('recommendedNextTranche', '')}",
        ]
    )


def write_strict_evidence_text_section_pilot_executor_apply_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "strict-evidence-text-section-pilot-executor-apply.json"
    summary_path = root / "strict-evidence-text-section-pilot-executor-apply-summary.json"
    markdown_path = root / "strict-evidence-text-section-pilot-executor-apply.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_strict_evidence_text_section_pilot_executor_apply_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Dry-run or explicitly apply manifest-only text-section pilot executor outcomes "
            "from an executor dry-run report without mutating StrictEvidence or SourceSpan stores."
        )
    )
    parser.add_argument(
        "--executor-dry-run-report",
        default=str(DEFAULT_EXECUTOR_DRY_RUN_REPORT_PATH),
        help="Path to text-section pilot executor dry-run JSON report.",
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable.")
    parser.add_argument(
        "--papers-dir",
        default="",
        help="Local papers_dir root. Required with --apply for run manifest writes.",
    )
    parser.add_argument("--run-id", default="", help="Run id recorded in the run manifest filename.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write a run manifest under structured_evidence/runs/ only.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Report output directory. Defaults to a dry-run-specific directory without --apply, "
            "or an apply-specific directory with --apply."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    apply_mode = bool(args.apply)
    report = build_strict_evidence_text_section_pilot_executor_apply(
        executor_dry_run_report_path=args.executor_dry_run_report,
        papers_dir=args.papers_dir or None,
        run_id=args.run_id or None,
        apply=apply_mode,
        paper_ids=args.paper_id or None,
    )

    output_dir = resolve_output_dir(args.output_dir or None, apply=apply_mode)
    paths = write_strict_evidence_text_section_pilot_executor_apply_reports(report, output_dir)
    print(f"wrote report: {paths['report']}")
    print(f"wrote summary: {paths['summary']}")
    print(f"wrote markdown: {paths['markdown']}")

    if args.json:
        print(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_APPLY_OUTPUT_DIR",
    "DEFAULT_DRY_RUN_OUTPUT_DIR",
    "DEFAULT_EXECUTOR_DRY_RUN_REPORT_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_REPORT_ROOT",
    "APPLY_STATUS_APPLIED",
    "APPLY_STATUS_READY",
    "STRICT_EVIDENCE_TEXT_SECTION_PILOT_EXECUTOR_APPLY_SCHEMA_ID",
    "build_strict_evidence_text_section_pilot_executor_apply",
    "default_output_dir",
    "render_strict_evidence_text_section_pilot_executor_apply_markdown",
    "resolve_output_dir",
    "write_strict_evidence_text_section_pilot_executor_apply_reports",
]
