"""Promotion tranche plan for parsed-artifact StrictEvidence policy-gate candidates.

Consumes the StrictEvidence policy-gate report and groups candidate-only rows into
explicit future promotion tranches without mutating records, writing JSONL, or
enabling citation/runtime/answer integration.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_strict_evidence_policy_gate import (
    PARSED_ARTIFACT_STRICT_EVIDENCE_POLICY_GATE_SCHEMA_ID,
    POLICY_STATUS_CANDIDATE_ONLY,
)


PARSED_ARTIFACT_STRICT_EVIDENCE_PROMOTION_TRANCHE_PLAN_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-strict-evidence-promotion-tranche-plan.v1"
)

PLAN_STATUS_CANDIDATE_ONLY = "promotion_tranche_plan_candidate_only"
PLAN_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"
PLAN_STATUS_BLOCKED_POLICY_GATE = "blocked_policy_gate_not_candidate"
PLAN_STATUS_BLOCKED_MISSING_IDENTITY = "blocked_missing_record_identity"
PLAN_STATUS_BLOCKED_UNSUPPORTED_ARTIFACT = "blocked_unsupported_artifact_type"
PLAN_STATUS_BLOCKED_RUNTIME_OR_CITATION = "blocked_runtime_or_citation_flag_violation"

TRANCHE_TEXT_SECTION_PILOT = "strict_evidence_text_section_pilot"
TRANCHE_FIGURE_CAPTION_PILOT = "strict_evidence_figure_caption_pilot"
TRANCHE_MANUAL_OR_EXTRACTOR_HOLDOUT = "strict_evidence_manual_or_extractor_holdout"
TRANCHE_CITATION_GRADE_GATE_LATER = "strict_evidence_citation_grade_gate_later"
TRANCHE_RUNTIME_BINDING_GATE_LATER = "strict_evidence_runtime_binding_gate_later"

PILOT_ARTIFACT_TYPES = frozenset({"section", "figure"})

DEFAULT_POLICY_GATE_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-strict-evidence-policy-gate"
    / "01-parsed-artifact-strict-evidence-policy-gate"
    / "parsed-artifact-strict-evidence-policy-gate.json"
)

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-strict-evidence-promotion-tranche-plan"
    / "01-parsed-artifact-strict-evidence-promotion-tranche-plan"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_bool(value: Any) -> bool:
    return bool(value)


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


def _runtime_or_integration_violation(policy_row: dict[str, Any]) -> list[str]:
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
        if _safe_bool(policy_row.get(field_name)):
            violations.append(f"policy_row.{field_name}_true")
    if _safe_int(policy_row.get("sourceSpanUpdatedRows")):
        violations.append("policy_row.sourceSpanUpdatedRows_nonzero")
    return violations


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _assign_planned_tranche(artifact_type: str) -> str:
    if artifact_type == "section":
        return TRANCHE_TEXT_SECTION_PILOT
    if artifact_type == "figure":
        return TRANCHE_FIGURE_CAPTION_PILOT
    return ""


def _classify_plan_row(policy_row: dict[str, Any]) -> tuple[str, str, list[str]]:
    blockers: list[str] = []
    policy_status = _safe_text(policy_row.get("policy_gate_status"))
    if policy_status != POLICY_STATUS_CANDIDATE_ONLY:
        blockers.append(f"policy_gate_status={policy_status or 'unknown'}")
        if not _safe_bool(policy_row.get("strictEvidencePolicyCandidateOnly")):
            blockers.append("strictEvidencePolicyCandidateOnly_false")
        return PLAN_STATUS_BLOCKED_POLICY_GATE, "", _dedupe(blockers)

    for field_name in ("strictEvidenceId", "sourceSpanId", "candidateRecordId"):
        if not _safe_text(policy_row.get(field_name)):
            blockers.append(f"{field_name}_missing")

    if blockers:
        return PLAN_STATUS_BLOCKED_MISSING_IDENTITY, "", _dedupe(blockers)

    flag_violations = _runtime_or_integration_violation(policy_row)
    if flag_violations:
        return PLAN_STATUS_BLOCKED_RUNTIME_OR_CITATION, "", flag_violations

    artifact_type = _safe_text(policy_row.get("artifact_type"))
    planned_tranche = _assign_planned_tranche(artifact_type)
    if not planned_tranche:
        return PLAN_STATUS_BLOCKED_UNSUPPORTED_ARTIFACT, "", [
            f"unsupported_artifact_type:{artifact_type or 'missing'}"
        ]

    return PLAN_STATUS_CANDIDATE_ONLY, planned_tranche, []


def _plan_rows(policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, policy_row in enumerate(policy_rows):
        source_row = dict(policy_row or {})
        plan_status, planned_tranche, blockers = _classify_plan_row(source_row)
        ready = plan_status == PLAN_STATUS_CANDIDATE_ONLY
        rows.append(
            {
                "plan_row_id": f"parsed-artifact-strict-evidence-promotion-tranche-plan:{index:04d}",
                "policy_gate_row_id": _safe_text(source_row.get("policy_gate_row_id")),
                "readback_review_row_id": _safe_text(source_row.get("readback_review_row_id")),
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
                "policy_gate_status": _safe_text(source_row.get("policy_gate_status")),
                "strictEvidencePolicyCandidateOnly": _safe_bool(
                    source_row.get("strictEvidencePolicyCandidateOnly")
                ),
                "planned_tranche": planned_tranche,
                "planned_tranche_scope": (
                    "text strict evidence section pilot"
                    if planned_tranche == TRANCHE_TEXT_SECTION_PILOT
                    else (
                        "figure caption text strict evidence pilot (not figure region)"
                        if planned_tranche == TRANCHE_FIGURE_CAPTION_PILOT
                        else ""
                    )
                ),
                "plan_status": plan_status,
                "plan_blockers": _dedupe(blockers),
                "promotionTranchePlanCandidateOnly": ready,
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
                    f"queue_for_{planned_tranche}_dry_run"
                    if ready
                    else "repair_policy_gate_row_before_tranche_plan"
                ),
            }
        )
    return rows


def _tranche_catalog(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    section_rows = sum(1 for row in rows if row.get("planned_tranche") == TRANCHE_TEXT_SECTION_PILOT)
    figure_rows = sum(1 for row in rows if row.get("planned_tranche") == TRANCHE_FIGURE_CAPTION_PILOT)
    return [
        {
            "trancheId": TRANCHE_TEXT_SECTION_PILOT,
            "status": "planned_pilot",
            "artifactTypes": ["section"],
            "scope": "text strict evidence for section claim surfaces",
            "rowCount": section_rows,
            "recommendedExecutionOrder": 1,
        },
        {
            "trancheId": TRANCHE_FIGURE_CAPTION_PILOT,
            "status": "planned_pilot",
            "artifactTypes": ["figure"],
            "scope": "figure caption text strict evidence only (not figure region)",
            "rowCount": figure_rows,
            "recommendedExecutionOrder": 2,
        },
        {
            "trancheId": TRANCHE_MANUAL_OR_EXTRACTOR_HOLDOUT,
            "status": "out_of_scope_holdout",
            "artifactTypes": [],
            "scope": "manual follow-up or extractor rows outside the 99 candidate set",
            "rowCount": 0,
            "recommendedExecutionOrder": None,
        },
        {
            "trancheId": TRANCHE_CITATION_GRADE_GATE_LATER,
            "status": "blocked_later",
            "artifactTypes": [],
            "scope": "citation-grade evidence enablement remains explicitly blocked",
            "rowCount": 0,
            "recommendedExecutionOrder": None,
        },
        {
            "trancheId": TRANCHE_RUNTIME_BINDING_GATE_LATER,
            "status": "blocked_later",
            "artifactTypes": [],
            "scope": "runtime evidence binding remains explicitly blocked",
            "rowCount": 0,
            "recommendedExecutionOrder": None,
        },
    ]


def _blocked_later_gates() -> list[dict[str, Any]]:
    return [
        {
            "gateId": TRANCHE_CITATION_GRADE_GATE_LATER,
            "status": "blocked_later",
            "reason": "citation_grade_evidence_not_in_scope_for_layout_parser_pilot",
            "requiresExplicitTranche": True,
        },
        {
            "gateId": TRANCHE_RUNTIME_BINDING_GATE_LATER,
            "status": "blocked_later",
            "reason": "runtime_evidence_binding_not_in_scope_for_layout_parser_pilot",
            "requiresExplicitTranche": True,
        },
        {
            "gateId": "strict_evidence_answer_integration_gate_later",
            "status": "blocked_later",
            "reason": "answer_integration_changes_remain_disabled",
            "requiresExplicitTranche": True,
        },
    ]


def _count_rows(
    *,
    rows: list[dict[str, Any]],
    input_schema_violations: list[str],
    policy_candidate_rows: int,
) -> dict[str, Any]:
    return {
        "inputRows": len(rows),
        "policyCandidateRows": policy_candidate_rows,
        "plannedPromotionRows": sum(
            1 for row in rows if row.get("plan_status") == PLAN_STATUS_CANDIDATE_ONLY
        ),
        "sectionPilotRows": sum(
            1 for row in rows if row.get("planned_tranche") == TRANCHE_TEXT_SECTION_PILOT
        ),
        "figureCaptionPilotRows": sum(
            1 for row in rows if row.get("planned_tranche") == TRANCHE_FIGURE_CAPTION_PILOT
        ),
        "holdoutRows": 0,
        "blockedInputReportSchemaViolationRows": (
            len(rows) if input_schema_violations and rows else int(bool(input_schema_violations))
        ),
        "blockedPolicyGateNotCandidateRows": sum(
            1 for row in rows if row.get("plan_status") == PLAN_STATUS_BLOCKED_POLICY_GATE
        ),
        "blockedMissingRecordIdentityRows": sum(
            1 for row in rows if row.get("plan_status") == PLAN_STATUS_BLOCKED_MISSING_IDENTITY
        ),
        "blockedUnsupportedArtifactTypeRows": sum(
            1 for row in rows if row.get("plan_status") == PLAN_STATUS_BLOCKED_UNSUPPORTED_ARTIFACT
        ),
        "blockedRuntimeOrCitationFlagViolationRows": sum(
            1 for row in rows if row.get("plan_status") == PLAN_STATUS_BLOCKED_RUNTIME_OR_CITATION
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
        "byPaperId": dict(Counter(str(row.get("paper_id") or "") for row in rows)),
        "byArtifactType": dict(Counter(str(row.get("artifact_type") or "") for row in rows)),
        "byPlannedTranche": dict(Counter(str(row.get("planned_tranche") or "") for row in rows if row.get("planned_tranche"))),
        "byPlanStatus": dict(Counter(str(row.get("plan_status") or "") for row in rows)),
        "byRecommendedAction": dict(Counter(str(row.get("recommended_action") or "") for row in rows)),
    }


def build_parsed_artifact_strict_evidence_promotion_tranche_plan(
    *,
    policy_gate_report_path: str | Path = DEFAULT_POLICY_GATE_REPORT_PATH,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    report_path = Path(str(policy_gate_report_path)).expanduser()
    warnings: list[str] = []
    input_schema_violations: list[str] = []
    requested_papers = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}

    policy_gate_report = _read_json(report_path)
    if not policy_gate_report:
        warnings.append("policy_gate_report_missing_or_unreadable")

    validation = validate_payload(
        policy_gate_report,
        PARSED_ARTIFACT_STRICT_EVIDENCE_POLICY_GATE_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        input_schema_violations = [str(error) for error in validation.errors]
        if not policy_gate_report:
            input_schema_violations.append("policy_gate_report_missing_or_unreadable")

    if policy_gate_report and _safe_text(policy_gate_report.get("status")) != "ok":
        input_schema_violations.append(
            f"policy_gate_report_status={_safe_text(policy_gate_report.get('status')) or 'unknown'}"
        )

    policy_rows = [
        row
        for row in policy_gate_report.get("rows", [])
        if isinstance(row, dict)
        and _safe_bool(row.get("strictEvidencePolicyCandidateOnly"))
        and _safe_text(row.get("policy_gate_status")) == POLICY_STATUS_CANDIDATE_ONLY
    ] if isinstance(policy_gate_report, dict) else []

    policy_candidate_count = int(
        (policy_gate_report.get("counts") or {}).get("strictEvidencePolicyCandidateOnlyRows")
        or len(policy_rows)
    ) if policy_gate_report else 0

    if requested_papers:
        found_papers = {_safe_text(row.get("paper_id")) for row in policy_rows if _safe_text(row.get("paper_id"))}
        if requested_papers - found_papers:
            warnings.append("requested_paper_ids_not_found")
        policy_rows = [row for row in policy_rows if _safe_text(row.get("paper_id")) in requested_papers]

    if not policy_rows and not input_schema_violations:
        warnings.append("policy_gate_candidate_rows_missing")

    rows = _plan_rows(policy_rows)
    if input_schema_violations:
        for row in rows:
            row["plan_status"] = PLAN_STATUS_BLOCKED_INPUT_SCHEMA
            row["planned_tranche"] = ""
            row["plan_blockers"] = _dedupe([*row.get("plan_blockers", []), *input_schema_violations])
            row["promotionTranchePlanCandidateOnly"] = False
            row["recommended_action"] = "repair_policy_gate_report_schema_before_tranche_plan"

    counts = _count_rows(
        rows=rows,
        input_schema_violations=_dedupe(input_schema_violations),
        policy_candidate_rows=policy_candidate_count,
    )
    planned_rows = int(counts.get("plannedPromotionRows") or 0)
    status = "ok"
    if input_schema_violations or not rows or planned_rows != len(rows):
        status = "blocked"

    tranches = _tranche_catalog(rows)
    blocked_later = _blocked_later_gates()

    return {
        "schema": PARSED_ARTIFACT_STRICT_EVIDENCE_PROMOTION_TRANCHE_PLAN_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "policyGateReportPath": str(report_path),
            "policyGateSchema": _safe_text(policy_gate_report.get("schema")) if policy_gate_report else "",
            "policyGateReportStatus": _safe_text(policy_gate_report.get("status")) if policy_gate_report else "",
            "requestedPaperIds": sorted(requested_papers),
        },
        "counts": counts,
        "tranches": tranches,
        "blockedLater": blocked_later,
        "gate": {
            "promotionTranchePlanReady": (
                bool(planned_rows) and planned_rows == len(rows) and not input_schema_violations
            ),
            "strictEligibleMutationAllowed": False,
            "strictEvidenceStoreWriteAllowed": False,
            "strictEvidenceCreated": False,
            "citationReady": False,
            "runtimeEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": _dedupe(input_schema_violations),
            "decision": (
                "parsed_artifact_strict_evidence_promotion_tranche_plan_ready"
                if status == "ok"
                else "blocked"
            ),
            "recommendedNextTranche": (
                TRANCHE_TEXT_SECTION_PILOT
                if status == "ok"
                else "parsed_artifact_strict_evidence_policy_gate_repair"
            ),
            "recommendedTrancheExecutionOrder": [
                TRANCHE_TEXT_SECTION_PILOT,
                TRANCHE_FIGURE_CAPTION_PILOT,
            ],
        },
        "policy": {
            "reportOnly": True,
            "promotionTranchePlanOnly": True,
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
            "tranches",
            "blockedLater",
            "gate",
            "policy",
            "warnings",
            "rows",
        )
        if key in report
    }


def render_parsed_artifact_strict_evidence_promotion_tranche_plan_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    tranche_lines = [
        f"- {item.get('trancheId')}: {item.get('rowCount')} rows ({item.get('status')})"
        for item in report.get("tranches", [])
        if isinstance(item, dict)
    ]
    blocked_later_lines = [
        f"- {item.get('gateId')}: {item.get('reason')}"
        for item in report.get("blockedLater", [])
        if isinstance(item, dict)
    ]
    return "\n".join(
        [
            "# Parsed Artifact StrictEvidence Promotion Tranche Plan",
            "",
            f"- status: {report.get('status', '')}",
            f"- report-only: {json.dumps(report.get('policy', {}).get('reportOnly'))}",
            f"- input rows: {int(counts.get('inputRows') or 0)}",
            f"- planned promotion rows: {int(counts.get('plannedPromotionRows') or 0)}",
            f"- section pilot rows: {int(counts.get('sectionPilotRows') or 0)}",
            f"- figure caption pilot rows: {int(counts.get('figureCaptionPilotRows') or 0)}",
            f"- strict evidence writes: {int(counts.get('strictEvidenceWriteRows') or 0)}",
            "",
            "## Planned tranches",
            *tranche_lines,
            "",
            "## Blocked later gates",
            *blocked_later_lines,
            "",
            f"- recommended next tranche: {report.get('gate', {}).get('recommendedNextTranche', '')}",
        ]
    )


def write_parsed_artifact_strict_evidence_promotion_tranche_plan_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-strict-evidence-promotion-tranche-plan.json"
    summary_path = root / "parsed-artifact-strict-evidence-promotion-tranche-plan-summary.json"
    markdown_path = root / "parsed-artifact-strict-evidence-promotion-tranche-plan.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_parsed_artifact_strict_evidence_promotion_tranche_plan_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Plan explicit StrictEvidence promotion tranches from policy-gate candidate rows "
            "without mutating records or integration surfaces."
        )
    )
    parser.add_argument(
        "--policy-gate-report",
        default=str(DEFAULT_POLICY_GATE_REPORT_PATH),
        help="StrictEvidence policy-gate JSON report.",
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_parsed_artifact_strict_evidence_promotion_tranche_plan(
        policy_gate_report_path=args.policy_gate_report,
        paper_ids=args.paper_id or None,
    )

    if args.output_dir:
        paths = write_parsed_artifact_strict_evidence_promotion_tranche_plan_reports(
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
    "PARSED_ARTIFACT_STRICT_EVIDENCE_PROMOTION_TRANCHE_PLAN_SCHEMA_ID",
    "PLAN_STATUS_CANDIDATE_ONLY",
    "TRANCHE_TEXT_SECTION_PILOT",
    "TRANCHE_FIGURE_CAPTION_PILOT",
    "build_parsed_artifact_strict_evidence_promotion_tranche_plan",
    "render_parsed_artifact_strict_evidence_promotion_tranche_plan_markdown",
    "write_parsed_artifact_strict_evidence_promotion_tranche_plan_reports",
]
