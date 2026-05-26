"""Report-only figure-caption pilot for parsed-artifact StrictEvidence promotion.

Consumes the promotion tranche-plan report, validates figure-caption-assigned rows against
stored StrictEvidence authority contracts, and holds out section tranches without mutating
records or enabling integration surfaces.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_strict_evidence_promotion_tranche_plan import (
    PARSED_ARTIFACT_STRICT_EVIDENCE_PROMOTION_TRANCHE_PLAN_SCHEMA_ID,
    PLAN_STATUS_CANDIDATE_ONLY,
    TRANCHE_FIGURE_CAPTION_PILOT,
    TRANCHE_TEXT_SECTION_PILOT,
)
from knowledge_hub.papers.parsed_artifact_strict_evidence_record_contract import (
    CHARS_BASIS,
    CHARS_NORMALIZATION_LABEL,
)


STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_SCHEMA_ID = (
    "knowledge-hub.paper.strict-evidence-figure-caption-pilot.v1"
)

PILOT_STATUS_CANDIDATE_ONLY = "figure_caption_pilot_candidate_only"
PILOT_STATUS_HELD_OUT_SECTION = "held_out_section_tranche"
PILOT_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"
PILOT_STATUS_BLOCKED_TRANCHE_NOT_ASSIGNED = "blocked_tranche_not_assigned"
PILOT_STATUS_BLOCKED_MISSING_IDENTITY = "blocked_missing_record_identity"
PILOT_STATUS_BLOCKED_INVALID_ARTIFACT = "blocked_invalid_artifact_type"
PILOT_STATUS_BLOCKED_MISSING_AUTHORITY = "blocked_missing_authority_chars"
PILOT_STATUS_BLOCKED_HASH_CONTRACT = "blocked_hash_contract_violation"
PILOT_STATUS_BLOCKED_RUNTIME_OR_CITATION = "blocked_runtime_or_citation_flag_violation"

DEFAULT_TRANCHE_PLAN_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-strict-evidence-promotion-tranche-plan"
    / "01-parsed-artifact-strict-evidence-promotion-tranche-plan"
    / "parsed-artifact-strict-evidence-promotion-tranche-plan.json"
)

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "strict-evidence-figure-caption-pilot"
    / "01-strict-evidence-figure-caption-pilot"
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


def _runtime_or_integration_violation(
    *,
    plan_row: dict[str, Any],
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
        if _safe_bool(plan_row.get(field_name)):
            violations.append(f"plan_row.{field_name}_true")
    if _safe_int(plan_row.get("sourceSpanUpdatedRows")):
        violations.append("plan_row.sourceSpanUpdatedRows_nonzero")
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


def _classify_figure_caption_pilot_row(
    plan_row: dict[str, Any],
    *,
    strict_evidence_record: dict[str, Any],
) -> tuple[str, list[str]]:
    planned_tranche = _safe_text(plan_row.get("planned_tranche"))
    artifact_type = _safe_text(plan_row.get("artifact_type"))

    if planned_tranche != TRANCHE_FIGURE_CAPTION_PILOT:
        if artifact_type == "section" or planned_tranche == TRANCHE_TEXT_SECTION_PILOT:
            return PILOT_STATUS_HELD_OUT_SECTION, []
        return PILOT_STATUS_BLOCKED_TRANCHE_NOT_ASSIGNED, [
            f"planned_tranche={planned_tranche or 'missing'}"
        ]

    if artifact_type != "figure":
        return PILOT_STATUS_BLOCKED_INVALID_ARTIFACT, [
            f"artifact_type={artifact_type or 'missing'}"
        ]

    blockers: list[str] = []
    if _safe_text(plan_row.get("plan_status")) != PLAN_STATUS_CANDIDATE_ONLY:
        blockers.append(f"plan_status={_safe_text(plan_row.get('plan_status')) or 'unknown'}")
    if not _safe_bool(plan_row.get("promotionTranchePlanCandidateOnly")):
        blockers.append("promotionTranchePlanCandidateOnly_false")
    if blockers:
        return PILOT_STATUS_BLOCKED_TRANCHE_NOT_ASSIGNED, _dedupe(blockers)

    for field_name in ("strictEvidenceId", "sourceSpanId", "candidateRecordId", "sourceContentHash"):
        if not _safe_text(plan_row.get(field_name)):
            blockers.append(f"{field_name}_missing")
    if blockers:
        return PILOT_STATUS_BLOCKED_MISSING_IDENTITY, _dedupe(blockers)

    flag_violations = _runtime_or_integration_violation(
        plan_row=plan_row,
        strict_evidence_record=strict_evidence_record,
    )
    if flag_violations:
        return PILOT_STATUS_BLOCKED_RUNTIME_OR_CITATION, flag_violations

    authority = (
        strict_evidence_record.get("authority")
        if isinstance(strict_evidence_record.get("authority"), dict)
        else {}
    )
    chars = authority.get("chars") if isinstance(authority.get("chars"), dict) else {}
    start = chars.get("start")
    end = chars.get("end")
    if start is None or end is None:
        return PILOT_STATUS_BLOCKED_MISSING_AUTHORITY, ["authority_chars_start_or_end_missing"]

    basis = _safe_text(chars.get("basis"))
    if basis != CHARS_BASIS:
        return PILOT_STATUS_BLOCKED_MISSING_AUTHORITY, [
            f"authority_chars_basis_invalid:{basis or 'missing'}"
        ]

    normalization = _safe_text(chars.get("normalization"))
    if normalization != CHARS_NORMALIZATION_LABEL:
        return PILOT_STATUS_BLOCKED_MISSING_AUTHORITY, [
            f"authority_chars_normalization_invalid:{normalization or 'missing'}"
        ]

    verbatim_hash = _safe_text(strict_evidence_record.get("verbatimSubstringSha256"))
    expected_hash = _safe_text(chars.get("expectedSubstringSha256"))
    if not verbatim_hash or not expected_hash:
        return PILOT_STATUS_BLOCKED_HASH_CONTRACT, ["verbatim_or_expected_hash_missing"]
    if verbatim_hash != expected_hash:
        return PILOT_STATUS_BLOCKED_HASH_CONTRACT, [
            "verbatimSubstringSha256_must_equal_authority_chars_expectedSubstringSha256"
        ]

    return PILOT_STATUS_CANDIDATE_ONLY, []


def _pilot_rows(plan_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, plan_row in enumerate(plan_rows):
        source_row = dict(plan_row or {})
        strict_evidence_record = _load_strict_evidence_record_at_store_ref(
            _safe_text(source_row.get("strict_evidence_store_path")),
            _safe_int(source_row.get("strict_evidence_store_line")),
        )
        pilot_status, blockers = _classify_figure_caption_pilot_row(
            source_row,
            strict_evidence_record=strict_evidence_record,
        )
        ready = pilot_status == PILOT_STATUS_CANDIDATE_ONLY
        authority = (
            strict_evidence_record.get("authority")
            if isinstance(strict_evidence_record.get("authority"), dict)
            else {}
        )
        chars = authority.get("chars") if isinstance(authority.get("chars"), dict) else {}
        rows.append(
            {
                "pilot_row_id": f"strict-evidence-figure-caption-pilot:{index:04d}",
                "plan_row_id": _safe_text(source_row.get("plan_row_id")),
                "policy_gate_row_id": _safe_text(source_row.get("policy_gate_row_id")),
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
                "plan_status": _safe_text(source_row.get("plan_status")),
                "promotionTranchePlanCandidateOnly": _safe_bool(
                    source_row.get("promotionTranchePlanCandidateOnly")
                ),
                "verbatimSubstringSha256": _safe_text(
                    strict_evidence_record.get("verbatimSubstringSha256")
                ),
                "authority_chars": {
                    "start": chars.get("start"),
                    "end": chars.get("end"),
                    "basis": _safe_text(chars.get("basis")),
                    "normalization": _safe_text(chars.get("normalization")),
                    "expectedSubstringSha256": _safe_text(chars.get("expectedSubstringSha256")),
                },
                "pilot_status": pilot_status,
                "pilot_blockers": _dedupe(blockers),
                "figureCaptionPilotCandidateOnly": ready,
                "figureCaptionTextOnly": ready,
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
                    "queue_for_strict_evidence_figure_caption_pilot_executor_dry_run"
                    if ready
                    else (
                        "held_for_section_tranche_diagnostics"
                        if pilot_status == PILOT_STATUS_HELD_OUT_SECTION
                        else "repair_tranche_plan_or_strict_evidence_before_figure_caption_pilot"
                    )
                ),
            }
        )
    return rows


def _held_out_section_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    held_rows = [row for row in rows if row.get("pilot_status") == PILOT_STATUS_HELD_OUT_SECTION]
    return {
        "heldOutSectionRows": len(held_rows),
        "byPlannedTranche": dict(Counter(_safe_text(row.get("planned_tranche")) for row in held_rows)),
        "byArtifactType": dict(Counter(_safe_text(row.get("artifact_type")) for row in held_rows)),
        "diagnosticOnly": True,
        "activePilotProcessing": False,
    }


def _count_rows(
    *,
    rows: list[dict[str, Any]],
    input_schema_violations: list[str],
    figure_caption_pilot_input_rows: int,
) -> dict[str, Any]:
    figure_rows = [
        row for row in rows if _safe_text(row.get("planned_tranche")) == TRANCHE_FIGURE_CAPTION_PILOT
    ]
    return {
        "inputRows": len(rows),
        "figureCaptionPilotInputRows": len(figure_rows),
        "figureCaptionPilotCandidateOnlyRows": sum(
            1 for row in rows if row.get("pilot_status") == PILOT_STATUS_CANDIDATE_ONLY
        ),
        "heldOutSectionRows": sum(
            1 for row in rows if row.get("pilot_status") == PILOT_STATUS_HELD_OUT_SECTION
        ),
        "blockedInputReportSchemaViolationRows": (
            len(rows) if input_schema_violations and rows else int(bool(input_schema_violations))
        ),
        "blockedTrancheNotAssignedRows": sum(
            1 for row in rows if row.get("pilot_status") == PILOT_STATUS_BLOCKED_TRANCHE_NOT_ASSIGNED
        ),
        "blockedMissingRecordIdentityRows": sum(
            1 for row in rows if row.get("pilot_status") == PILOT_STATUS_BLOCKED_MISSING_IDENTITY
        ),
        "blockedInvalidArtifactTypeRows": sum(
            1 for row in rows if row.get("pilot_status") == PILOT_STATUS_BLOCKED_INVALID_ARTIFACT
        ),
        "blockedMissingAuthorityCharsRows": sum(
            1 for row in rows if row.get("pilot_status") == PILOT_STATUS_BLOCKED_MISSING_AUTHORITY
        ),
        "blockedHashContractViolationRows": sum(
            1 for row in rows if row.get("pilot_status") == PILOT_STATUS_BLOCKED_HASH_CONTRACT
        ),
        "blockedRuntimeOrCitationFlagViolationRows": sum(
            1 for row in rows if row.get("pilot_status") == PILOT_STATUS_BLOCKED_RUNTIME_OR_CITATION
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
                if row.get("pilot_status") == PILOT_STATUS_CANDIDATE_ONLY
            )
        ),
        "byArtifactType": dict(Counter(_safe_text(row.get("artifact_type")) for row in rows)),
        "byPilotStatus": dict(Counter(_safe_text(row.get("pilot_status")) for row in rows)),
        "byRecommendedAction": dict(Counter(_safe_text(row.get("recommended_action")) for row in rows)),
        "figureCaptionPilotInputRowsFromTranchePlan": figure_caption_pilot_input_rows,
    }


def build_strict_evidence_figure_caption_pilot(
    *,
    tranche_plan_report_path: str | Path = DEFAULT_TRANCHE_PLAN_REPORT_PATH,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    report_path = Path(str(tranche_plan_report_path)).expanduser()
    warnings: list[str] = []
    input_schema_violations: list[str] = []
    requested_papers = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}

    tranche_plan_report = _read_json(report_path)
    if not tranche_plan_report:
        warnings.append("tranche_plan_report_missing_or_unreadable")

    validation = validate_payload(
        tranche_plan_report,
        PARSED_ARTIFACT_STRICT_EVIDENCE_PROMOTION_TRANCHE_PLAN_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        input_schema_violations = [str(error) for error in validation.errors]
        if not tranche_plan_report:
            input_schema_violations.append("tranche_plan_report_missing_or_unreadable")

    if tranche_plan_report and _safe_text(tranche_plan_report.get("status")) != "ok":
        input_schema_violations.append(
            f"tranche_plan_report_status={_safe_text(tranche_plan_report.get('status')) or 'unknown'}"
        )

    all_plan_rows = [
        row for row in tranche_plan_report.get("rows", []) if isinstance(row, dict)
    ] if isinstance(tranche_plan_report, dict) else []

    figure_caption_pilot_input_rows = int(
        (tranche_plan_report.get("counts") or {}).get("figureCaptionPilotRows")
        or sum(
            1
            for row in all_plan_rows
            if _safe_text(row.get("planned_tranche")) == TRANCHE_FIGURE_CAPTION_PILOT
        )
    ) if tranche_plan_report else 0

    if requested_papers:
        found_papers = {_safe_text(row.get("paper_id")) for row in all_plan_rows if _safe_text(row.get("paper_id"))}
        if requested_papers - found_papers:
            warnings.append("requested_paper_ids_not_found")
        all_plan_rows = [row for row in all_plan_rows if _safe_text(row.get("paper_id")) in requested_papers]

    if not all_plan_rows and not input_schema_violations:
        warnings.append("tranche_plan_rows_missing")

    rows = _pilot_rows(all_plan_rows)
    if input_schema_violations:
        for row in rows:
            if row.get("pilot_status") == PILOT_STATUS_HELD_OUT_SECTION:
                continue
            row["pilot_status"] = PILOT_STATUS_BLOCKED_INPUT_SCHEMA
            row["pilot_blockers"] = _dedupe([*row.get("pilot_blockers", []), *input_schema_violations])
            row["figureCaptionPilotCandidateOnly"] = False
            row["figureCaptionTextOnly"] = False
            row["recommended_action"] = "repair_tranche_plan_report_schema_before_figure_caption_pilot"

    counts = _count_rows(
        rows=rows,
        input_schema_violations=_dedupe(input_schema_violations),
        figure_caption_pilot_input_rows=figure_caption_pilot_input_rows,
    )
    candidate_rows = int(counts.get("figureCaptionPilotCandidateOnlyRows") or 0)
    held_out_rows = int(counts.get("heldOutSectionRows") or 0)
    status = "ok"
    if (
        input_schema_violations
        or not rows
        or candidate_rows != figure_caption_pilot_input_rows
        or candidate_rows + held_out_rows != len(rows)
    ):
        status = "blocked"

    held_out_summary = _held_out_section_summary(rows)

    return {
        "schema": STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "tranchePlanReportPath": str(report_path),
            "tranchePlanSchema": _safe_text(tranche_plan_report.get("schema")) if tranche_plan_report else "",
            "tranchePlanReportStatus": _safe_text(tranche_plan_report.get("status")) if tranche_plan_report else "",
            "requestedPaperIds": sorted(requested_papers),
            "targetTranche": TRANCHE_FIGURE_CAPTION_PILOT,
        },
        "counts": counts,
        "heldOutSection": held_out_summary,
        "gate": {
            "figureCaptionPilotReady": (
                bool(candidate_rows)
                and candidate_rows == figure_caption_pilot_input_rows
                and not input_schema_violations
            ),
            "strictEligibleMutationAllowed": False,
            "strictEvidenceStoreWriteAllowed": False,
            "strictEvidenceCreated": False,
            "citationReady": False,
            "runtimeEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "sectionRowsProcessedAsActivePilot": False,
            "schemaViolations": _dedupe(input_schema_violations),
            "decision": (
                "strict_evidence_figure_caption_pilot_ready"
                if status == "ok"
                else "blocked"
            ),
            "recommendedNextTranche": (
                "strict_evidence_figure_caption_pilot_executor_dry_run"
                if status == "ok"
                else "parsed_artifact_strict_evidence_promotion_tranche_plan_repair"
            ),
        },
        "policy": {
            "reportOnly": True,
            "figureCaptionPilotOnly": True,
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
            "gate",
            "policy",
            "warnings",
            "rows",
        )
        if key in report
    }


def render_strict_evidence_figure_caption_pilot_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    held_out = dict(report.get("heldOutSection") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byPilotStatus") or {})).items())
    ]
    return "\n".join(
        [
            "# Strict Evidence Figure Caption Pilot",
            "",
            f"- status: {report.get('status', '')}",
            f"- report-only: {json.dumps(report.get('policy', {}).get('reportOnly'))}",
            f"- figure caption text only (not figure region): {json.dumps(report.get('policy', {}).get('figureCaptionTextOnlyNotFigureRegion'))}",
            f"- input rows: {int(counts.get('inputRows') or 0)}",
            f"- figure caption pilot input rows: {int(counts.get('figureCaptionPilotInputRows') or 0)}",
            f"- figure caption pilot candidate-only rows: {int(counts.get('figureCaptionPilotCandidateOnlyRows') or 0)}",
            f"- held-out section rows: {int(counts.get('heldOutSectionRows') or 0)}",
            f"- strict evidence writes: {int(counts.get('strictEvidenceWriteRows') or 0)}",
            "",
            "## Held-out section summary",
            f"- held-out rows: {int(held_out.get('heldOutSectionRows') or 0)}",
            f"- by planned tranche: {json.dumps(held_out.get('byPlannedTranche') or {})}",
            f"- by artifact type: {json.dumps(held_out.get('byArtifactType') or {})}",
            "",
            "## Pilot status breakdown",
            *[f"- {item}" for item in by_status],
            "",
            f"- recommended next tranche: {report.get('gate', {}).get('recommendedNextTranche', '')}",
        ]
    )


def write_strict_evidence_figure_caption_pilot_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "strict-evidence-figure-caption-pilot.json"
    summary_path = root / "strict-evidence-figure-caption-pilot-summary.json"
    markdown_path = root / "strict-evidence-figure-caption-pilot.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_strict_evidence_figure_caption_pilot_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Plan the figure-caption StrictEvidence pilot from tranche-plan rows without "
            "mutating records or integration surfaces."
        )
    )
    parser.add_argument(
        "--tranche-plan-report",
        default=str(DEFAULT_TRANCHE_PLAN_REPORT_PATH),
        help="Promotion tranche-plan JSON report.",
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_strict_evidence_figure_caption_pilot(
        tranche_plan_report_path=args.tranche_plan_report,
        paper_ids=args.paper_id or None,
    )

    if args.output_dir:
        paths = write_strict_evidence_figure_caption_pilot_reports(report, args.output_dir)
        print(f"wrote report: {paths['report']}")
        print(f"wrote summary: {paths['summary']}")
        print(f"wrote markdown: {paths['markdown']}")

    if args.json or not args.output_dir:
        print(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_SCHEMA_ID",
    "PILOT_STATUS_CANDIDATE_ONLY",
    "PILOT_STATUS_HELD_OUT_SECTION",
    "TRANCHE_FIGURE_CAPTION_PILOT",
    "build_strict_evidence_figure_caption_pilot",
    "render_strict_evidence_figure_caption_pilot_markdown",
    "write_strict_evidence_figure_caption_pilot_reports",
]
