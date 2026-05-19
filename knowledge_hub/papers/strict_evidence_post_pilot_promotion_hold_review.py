"""Post-pilot promotion hold review for completed StrictEvidence manifest-only pilot rows.

Consumes the pilot tranche completion gate report and documents that downstream
promotion surfaces remain blocked while the 99-row pilot stays candidate-only.
Report-only: does not mutate StrictEvidence or SourceSpan stores or enable integration.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_pilot_tranche_completion_gate import (
    COMPLETION_STATUS_COMPLETE,
    EXPECTED_FIGURE_CAPTION_MANIFEST_ROWS,
    EXPECTED_POLICY_CANDIDATE_ROWS,
    EXPECTED_SECTION_MANIFEST_ROWS,
    EXPECTED_SOURCE_SPAN_STORE_ROWS,
    EXPECTED_STRICT_EVIDENCE_STORE_ROWS,
    STRICT_EVIDENCE_PILOT_TRANCHE_COMPLETION_GATE_SCHEMA_ID,
)


STRICT_EVIDENCE_POST_PILOT_PROMOTION_HOLD_REVIEW_SCHEMA_ID = (
    "knowledge-hub.paper.strict-evidence-post-pilot-promotion-hold-review.v1"
)

HOLD_STATUS_ACTIVE = "post_pilot_promotion_hold_active"
HOLD_STATUS_BLOCKED_NOT_CANDIDATE = "blocked_completion_gate_not_candidate"
HOLD_STATUS_BLOCKED_NOT_COMPLETE = "blocked_pilot_not_complete"
HOLD_STATUS_BLOCKED_GATE_ENABLED = "blocked_downstream_gate_already_enabled"
HOLD_STATUS_BLOCKED_STORE_COUNT = "blocked_store_row_count_changed"
HOLD_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"

DEFAULT_COMPLETION_GATE_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "strict-evidence-pilot-tranche-completion-gate"
    / "01-strict-evidence-pilot-tranche-completion-gate"
    / "strict-evidence-pilot-tranche-completion-gate.json"
)

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "strict-evidence-post-pilot-promotion-hold-review"
    / "01-strict-evidence-post-pilot-promotion-hold-review"
)

_WRITE_COUNT_FIELDS = (
    "strictEvidenceWriteRows",
    "strictEvidenceCreatedRows",
    "citationGradeEvidenceCreatedRows",
    "runtimeEvidenceCreatedRows",
    "parserRoutingChangedRows",
    "answerIntegrationChangedRows",
    "databaseMutationRows",
    "canonicalParsedArtifactWriteRows",
    "sourceSpanUpdatedRows",
    "manifestWriteRows",
    "reindexOrReembedRows",
)

_DOWNSTREAM_GATE_NAMES = (
    "strictEligibleMutation",
    "citationGradeEvidence",
    "runtimeEvidence",
    "parserRouting",
    "answerIntegration",
    "databaseMutation",
    "reindexOrReembed",
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


def _future_promotion_readiness_checklist() -> list[dict[str, Any]]:
    return [
        {
            "id": "adr_strict_eligible_semantics",
            "title": "ADR or decision record for strictEligible mutation semantics",
            "status": "pending",
            "requiredBeforePromotion": True,
        },
        {
            "id": "rollback_strategy",
            "title": "Rollback strategy for strict/citation/runtime promotion",
            "status": "pending",
            "requiredBeforePromotion": True,
        },
        {
            "id": "eval_no_answer_safety",
            "title": "Eval gate for no-answer safety",
            "status": "pending",
            "requiredBeforePromotion": True,
        },
        {
            "id": "citation_grade_policy_gate",
            "title": "Citation-grade policy gate",
            "status": "pending",
            "requiredBeforePromotion": True,
        },
        {
            "id": "runtime_binding_gate",
            "title": "Runtime binding gate",
            "status": "pending",
            "requiredBeforePromotion": True,
        },
        {
            "id": "answer_integration_gate",
            "title": "Answer integration gate",
            "status": "pending",
            "requiredBeforePromotion": True,
        },
        {
            "id": "store_readback_after_mutation",
            "title": "Full store readback after any mutation",
            "status": "pending",
            "requiredBeforePromotion": True,
        },
        {
            "id": "release_smoke_hygiene",
            "title": "Public release smoke/hygiene after any integration change",
            "status": "pending",
            "requiredBeforePromotion": True,
        },
    ]


def _no_mutation_policy_matrix() -> dict[str, Any]:
    return {
        "reportOnly": True,
        "holdReviewOnly": True,
        "manifestWrite": False,
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
    }


def _blocked_downstream_gate_matrix(
    *,
    completion_gate: dict[str, Any],
) -> dict[str, Any]:
    blocked_later = (
        completion_gate.get("blockedLaterGates")
        if isinstance(completion_gate.get("blockedLaterGates"), dict)
        else {}
    )
    gate = completion_gate.get("gate") if isinstance(completion_gate.get("gate"), dict) else {}
    policy = completion_gate.get("policy") if isinstance(completion_gate.get("policy"), dict) else {}
    matrix: dict[str, Any] = {}
    for name in _DOWNSTREAM_GATE_NAMES:
        section = blocked_later.get(name) if isinstance(blocked_later.get(name), dict) else {}
        allowed = _safe_bool(section.get("allowed"))
        ready = _safe_bool(section.get("ready"))
        matrix[name] = {
            "allowed": allowed,
            "ready": ready,
            "reason": _safe_text(section.get("reason")),
            "holdActive": not allowed,
        }
    matrix["strictEligibleMutation"] = {
        "allowed": _safe_bool(gate.get("strictEligibleMutationAllowed")),
        "ready": False,
        "reason": _safe_text(
            (blocked_later.get("strictEligibleMutation") or {}).get("reason")
            if isinstance(blocked_later.get("strictEligibleMutation"), dict)
            else ""
        )
        or "blocked_until_explicit_post_pilot_strict_eligible_tranche",
        "holdActive": not _safe_bool(gate.get("strictEligibleMutationAllowed")),
    }
    matrix["databaseMutation"] = {
        "allowed": _safe_bool(gate.get("runtimeMutationAllowed")) or _safe_bool(policy.get("databaseMutation")),
        "ready": False,
        "reason": "blocked_until_explicit_post_pilot_database_mutation_tranche",
        "holdActive": not (
            _safe_bool(gate.get("runtimeMutationAllowed")) or _safe_bool(policy.get("databaseMutation"))
        ),
    }
    matrix["reindexOrReembed"] = {
        "allowed": _safe_bool(policy.get("reindexOrReembed")),
        "ready": False,
        "reason": "blocked_until_explicit_post_pilot_reindex_or_reembed_tranche",
        "holdActive": not _safe_bool(policy.get("reindexOrReembed")),
    }
    return matrix


def _downstream_gate_violations(gate_matrix: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for name, section in gate_matrix.items():
        if not isinstance(section, dict):
            continue
        if _safe_bool(section.get("allowed")) or _safe_bool(section.get("ready")):
            violations.append(f"downstream_gate_{name}_already_enabled")
    return violations


def _aggregate_completion_violations(
    *,
    completion_gate: dict[str, Any],
    input_schema_violations: list[str],
    gate_matrix: dict[str, Any],
) -> list[str]:
    violations = list(input_schema_violations)
    counts = completion_gate.get("counts") if isinstance(completion_gate.get("counts"), dict) else {}
    gate = completion_gate.get("gate") if isinstance(completion_gate.get("gate"), dict) else {}

    if _safe_text(completion_gate.get("status")) != "ok":
        violations.append(
            f"completion_gate_report_status={_safe_text(completion_gate.get('status')) or 'unknown'}"
        )
    if _safe_text(gate.get("completionDecision")) != COMPLETION_STATUS_COMPLETE:
        violations.append(
            f"completion_decision={_safe_text(gate.get('completionDecision')) or 'unknown'}"
        )
    if not _safe_bool(gate.get("pilotTrancheCompletionGateReady")):
        violations.append("pilot_tranche_completion_gate_not_ready")

    expectations = {
        "inputPolicyCandidateRows": EXPECTED_POLICY_CANDIDATE_ROWS,
        "validatedPilotRows": EXPECTED_POLICY_CANDIDATE_ROWS,
        "completionCandidateOnlyRows": EXPECTED_POLICY_CANDIDATE_ROWS,
        "sectionValidatedRows": EXPECTED_SECTION_MANIFEST_ROWS,
        "figureCaptionValidatedRows": EXPECTED_FIGURE_CAPTION_MANIFEST_ROWS,
        "strictEvidenceStoreRows": EXPECTED_STRICT_EVIDENCE_STORE_ROWS,
        "sourceSpanStoreRows": EXPECTED_SOURCE_SPAN_STORE_ROWS,
    }
    for field_name, expected in expectations.items():
        actual = _safe_int(counts.get(field_name))
        if actual != expected:
            violations.append(f"{field_name}={actual}_expected_{expected}")

    for field_name in _WRITE_COUNT_FIELDS:
        if _safe_int(counts.get(field_name)) != 0:
            violations.append(f"{field_name}={_safe_int(counts.get(field_name))}_expected_0")

    violations.extend(_downstream_gate_violations(gate_matrix))
    return _dedupe(violations)


def _hold_rows(
    completion_rows: list[dict[str, Any]],
    *,
    aggregate_violations: list[str],
    gate_matrix: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    downstream_enabled = _downstream_gate_violations(gate_matrix)

    for index, completion_row in enumerate(completion_rows):
        source_row = dict(completion_row or {})
        completion_status = _safe_text(source_row.get("completion_status"))
        blockers: list[str] = []
        hold_status = HOLD_STATUS_ACTIVE
        recommended_action = "post_pilot_promotion_hold_active"

        if aggregate_violations:
            hold_status = HOLD_STATUS_BLOCKED_INPUT_SCHEMA
            blockers.extend(aggregate_violations)
            recommended_action = "blocked_input_schema_violation"
        elif downstream_enabled:
            hold_status = HOLD_STATUS_BLOCKED_GATE_ENABLED
            blockers.extend(downstream_enabled)
            recommended_action = "blocked_downstream_gate_already_enabled"
        elif completion_status != COMPLETION_STATUS_COMPLETE:
            if not _safe_bool(source_row.get("pilotTrancheCompleteCandidateOnly")):
                hold_status = HOLD_STATUS_BLOCKED_NOT_CANDIDATE
                blockers.append("completion_row_not_candidate_only")
                recommended_action = "blocked_completion_gate_not_candidate"
            else:
                hold_status = HOLD_STATUS_BLOCKED_NOT_COMPLETE
                blockers.append(f"completion_status={completion_status or 'unknown'}")
                recommended_action = "blocked_pilot_not_complete"

        hold_active = hold_status == HOLD_STATUS_ACTIVE
        rows.append(
            {
                "hold_row_id": f"strict-evidence-post-pilot-promotion-hold-review:{index:04d}",
                "completion_row_id": _safe_text(source_row.get("completion_row_id")),
                "readback_row_id": _safe_text(source_row.get("readback_row_id")),
                "strictEvidenceId": _safe_text(source_row.get("strictEvidenceId")),
                "sourceSpanId": _safe_text(source_row.get("sourceSpanId")),
                "candidateRecordId": _safe_text(source_row.get("candidateRecordId")),
                "paper_id": _safe_text(source_row.get("paper_id")),
                "artifact_type": _safe_text(source_row.get("artifact_type")),
                "manifestType": _safe_text(source_row.get("manifestType")),
                "completion_status": completion_status,
                "hold_status": hold_status,
                "hold_blockers": _dedupe(blockers),
                "postPilotPromotionHoldActive": hold_active,
                "strictEligibleMutationAllowed": False,
                "citationGradeAllowed": False,
                "runtimeEvidenceAllowed": False,
                "parserRoutingAllowed": False,
                "answerIntegrationAllowed": False,
                "databaseMutationAllowed": False,
                "reindexOrReembedAllowed": False,
                "strictEligible": False,
                "strictEvidenceCreated": False,
                "citationGrade": False,
                "runtimeEvidence": False,
                "parserRoutingChanged": False,
                "answerIntegrationChanged": False,
                "databaseMutation": False,
                "recommended_action": recommended_action,
            }
        )
    return rows


def _count_rows(
    *,
    rows: list[dict[str, Any]],
    completion_counts: dict[str, Any],
    input_schema_violations: list[str],
    gate_matrix: dict[str, Any],
) -> dict[str, Any]:
    by_status = Counter(_safe_text(row.get("hold_status")) for row in rows)
    by_artifact = Counter(
        _safe_text(row.get("artifact_type"))
        for row in rows
        if row.get("hold_status") == HOLD_STATUS_ACTIVE
    )
    return {
        "inputPolicyCandidateRows": _safe_int(completion_counts.get("inputPolicyCandidateRows")),
        "validatedPilotRows": _safe_int(completion_counts.get("validatedPilotRows")),
        "holdActiveRows": int(by_status.get(HOLD_STATUS_ACTIVE, 0)),
        "sectionHoldRows": int(by_artifact.get("section", 0)),
        "figureCaptionHoldRows": int(by_artifact.get("figure", 0)),
        "blockedCompletionGateNotCandidateRows": int(by_status.get(HOLD_STATUS_BLOCKED_NOT_CANDIDATE, 0)),
        "blockedPilotNotCompleteRows": int(by_status.get(HOLD_STATUS_BLOCKED_NOT_COMPLETE, 0)),
        "blockedDownstreamGateAlreadyEnabledRows": int(
            by_status.get(HOLD_STATUS_BLOCKED_GATE_ENABLED, 0)
        ),
        "blockedStoreRowCountChangedRows": int(by_status.get(HOLD_STATUS_BLOCKED_STORE_COUNT, 0)),
        "blockedInputSchemaViolationRows": int(by_status.get(HOLD_STATUS_BLOCKED_INPUT_SCHEMA, 0)),
        "strictEligibleMutationAllowedRows": sum(
            1 for row in rows if _safe_bool(row.get("strictEligibleMutationAllowed"))
        ),
        "citationGradeAllowedRows": sum(1 for row in rows if _safe_bool(row.get("citationGradeAllowed"))),
        "runtimeEvidenceAllowedRows": sum(
            1 for row in rows if _safe_bool(row.get("runtimeEvidenceAllowed"))
        ),
        "parserRoutingAllowedRows": sum(1 for row in rows if _safe_bool(row.get("parserRoutingAllowed"))),
        "answerIntegrationAllowedRows": sum(
            1 for row in rows if _safe_bool(row.get("answerIntegrationAllowed"))
        ),
        "strictEvidenceStoreRows": _safe_int(completion_counts.get("strictEvidenceStoreRows")),
        "sourceSpanStoreRows": _safe_int(completion_counts.get("sourceSpanStoreRows")),
        "strictEvidenceWriteRows": 0,
        "strictEvidenceCreatedRows": 0,
        "citationGradeEvidenceCreatedRows": 0,
        "runtimeEvidenceCreatedRows": 0,
        "parserRoutingChangedRows": 0,
        "answerIntegrationChangedRows": 0,
        "databaseMutationRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
        "sourceSpanUpdatedRows": 0,
        "manifestWriteRows": 0,
        "reindexOrReembedRows": 0,
        "schemaViolationCount": len(input_schema_violations),
        "downstreamGateAllowedCount": sum(
            1
            for section in gate_matrix.values()
            if isinstance(section, dict) and _safe_bool(section.get("allowed"))
        ),
        "byArtifactType": dict(by_artifact),
        "byHoldStatus": dict(by_status),
        "byRecommendedAction": dict(Counter(_safe_text(row.get("recommended_action")) for row in rows)),
    }


def build_strict_evidence_post_pilot_promotion_hold_review(
    *,
    completion_gate_report_path: str | Path = DEFAULT_COMPLETION_GATE_REPORT_PATH,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    report_path = Path(str(completion_gate_report_path)).expanduser()
    warnings: list[str] = []
    input_schema_violations: list[str] = []
    requested_papers = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}

    completion_gate = _read_json(report_path)
    if not completion_gate:
        input_schema_violations.append("completion_gate_report_missing_or_unreadable")

    if completion_gate:
        validation = validate_payload(
            completion_gate,
            STRICT_EVIDENCE_PILOT_TRANCHE_COMPLETION_GATE_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            input_schema_violations.extend(str(error) for error in validation.errors)

    completion_rows = [
        row for row in completion_gate.get("rows", []) if isinstance(row, dict)
    ] if completion_gate else []

    if requested_papers:
        found_papers = {_safe_text(row.get("paper_id")) for row in completion_rows if _safe_text(row.get("paper_id"))}
        if requested_papers - found_papers:
            warnings.append("requested_paper_ids_not_found")
        completion_rows = [
            row for row in completion_rows if _safe_text(row.get("paper_id")) in requested_papers
        ]

    if not completion_rows and not input_schema_violations:
        warnings.append("completion_gate_rows_missing")

    input_schema_violations = _dedupe(input_schema_violations)
    gate_matrix = _blocked_downstream_gate_matrix(completion_gate=completion_gate) if completion_gate else {}
    aggregate_violations = (
        _aggregate_completion_violations(
            completion_gate=completion_gate,
            input_schema_violations=input_schema_violations,
            gate_matrix=gate_matrix,
        )
        if completion_gate
        else list(input_schema_violations)
    )

    rows = _hold_rows(
        completion_rows,
        aggregate_violations=aggregate_violations,
        gate_matrix=gate_matrix,
    )
    completion_counts = (
        completion_gate.get("counts") if isinstance(completion_gate.get("counts"), dict) else {}
    )
    counts = _count_rows(
        rows=rows,
        completion_counts=completion_counts,
        input_schema_violations=aggregate_violations,
        gate_matrix=gate_matrix,
    )

    hold_active_rows = int(counts.get("holdActiveRows") or 0)
    status = "ok"
    if (
        aggregate_violations
        or not rows
        or hold_active_rows != EXPECTED_POLICY_CANDIDATE_ROWS
        or hold_active_rows != len(rows)
        or int(counts.get("downstreamGateAllowedCount") or 0) > 0
    ):
        status = "blocked"

    policy_matrix = _no_mutation_policy_matrix()
    checklist = _future_promotion_readiness_checklist()

    return {
        "schema": STRICT_EVIDENCE_POST_PILOT_PROMOTION_HOLD_REVIEW_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "completionGateReportPath": str(report_path),
            "completionGateReportSchema": _safe_text(completion_gate.get("schema")) if completion_gate else "",
            "completionGateReportStatus": _safe_text(completion_gate.get("status")) if completion_gate else "",
            "completionDecision": _safe_text(
                (completion_gate.get("gate") or {}).get("completionDecision")
            )
            if completion_gate
            else "",
            "requestedPaperIds": sorted(requested_papers),
            "sectionRunManifestPath": _safe_text((completion_gate.get("input") or {}).get("sectionRunManifestPath"))
            if completion_gate
            else "",
            "figureCaptionRunManifestPath": _safe_text(
                (completion_gate.get("input") or {}).get("figureCaptionRunManifestPath")
            )
            if completion_gate
            else "",
            "expectedPolicyCandidateRows": EXPECTED_POLICY_CANDIDATE_ROWS,
            "expectedHoldActiveRows": EXPECTED_POLICY_CANDIDATE_ROWS,
        },
        "counts": counts,
        "blockedDownstreamGateMatrix": gate_matrix,
        "futurePromotionReadinessChecklist": checklist,
        "noMutationPolicyMatrix": policy_matrix,
        "gate": {
            "postPilotPromotionHoldReviewReady": status == "ok",
            "holdDecision": (
                "strict_evidence_post_pilot_promotion_hold_active"
                if status == "ok"
                else "strict_evidence_post_pilot_promotion_hold_blocked"
            ),
            "strictEligibleMutationAllowed": False,
            "citationGradeAllowed": False,
            "runtimeEvidenceAllowed": False,
            "parserRoutingAllowed": False,
            "answerIntegrationAllowed": False,
            "databaseMutationAllowed": False,
            "reindexOrReembedAllowed": False,
            "strictEvidenceStoreWriteAllowed": False,
            "sourceSpanStoreWriteAllowed": False,
            "runManifestWriteAllowed": False,
            "schemaViolations": aggregate_violations,
            "recommendedNextTranche": (
                "strict_evidence_strict_eligible_mutation_decision_record"
                if status == "ok"
                else "strict_evidence_pilot_tranche_completion_gate_repair"
            ),
        },
        "policy": policy_matrix,
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
            "blockedDownstreamGateMatrix",
            "futurePromotionReadinessChecklist",
            "noMutationPolicyMatrix",
            "gate",
            "policy",
            "warnings",
        )
        if key in report
    }


def render_strict_evidence_post_pilot_promotion_hold_review_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    gate_matrix = dict(report.get("blockedDownstreamGateMatrix") or {})
    checklist = list(report.get("futurePromotionReadinessChecklist") or [])
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byHoldStatus") or {})).items())
    ]
    matrix_lines = [
        f"- {name}: allowed={json.dumps(section.get('allowed'))}, holdActive={json.dumps(section.get('holdActive'))}"
        for name, section in sorted(gate_matrix.items())
        if isinstance(section, dict)
    ]
    checklist_lines = [
        f"- [{item.get('id', '')}] {item.get('title', '')} (status={item.get('status', '')})"
        for item in checklist
    ]
    return "\n".join(
        [
            "# Strict Evidence Post-Pilot Promotion Hold Review",
            "",
            f"- status: {report.get('status', '')}",
            f"- hold decision: {gate.get('holdDecision', '')}",
            f"- input policy candidate rows: {int(counts.get('inputPolicyCandidateRows') or 0)}",
            f"- validated pilot rows: {int(counts.get('validatedPilotRows') or 0)}",
            f"- hold active rows: {int(counts.get('holdActiveRows') or 0)}",
            f"- section hold rows: {int(counts.get('sectionHoldRows') or 0)}",
            f"- figure caption hold rows: {int(counts.get('figureCaptionHoldRows') or 0)}",
            f"- strict evidence store rows: {int(counts.get('strictEvidenceStoreRows') or 0)}",
            f"- source span store rows: {int(counts.get('sourceSpanStoreRows') or 0)}",
            "",
            "## Blocked downstream gate matrix",
            *matrix_lines,
            "",
            "## Future promotion readiness checklist",
            *checklist_lines,
            "",
            "## Hold status breakdown",
            *[f"- {item}" for item in by_status],
            "",
            f"- recommended next tranche: {gate.get('recommendedNextTranche', '')}",
        ]
    )


def write_strict_evidence_post_pilot_promotion_hold_review_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "strict-evidence-post-pilot-promotion-hold-review.json"
    summary_path = root / "strict-evidence-post-pilot-promotion-hold-review-summary.json"
    markdown_path = root / "strict-evidence-post-pilot-promotion-hold-review.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_strict_evidence_post_pilot_promotion_hold_review_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Review post-pilot promotion hold state for completed StrictEvidence pilot rows "
            "without enabling downstream promotion or mutating stores."
        )
    )
    parser.add_argument(
        "--completion-gate-report",
        default=str(DEFAULT_COMPLETION_GATE_REPORT_PATH),
        help="Path to the pilot tranche completion gate JSON report.",
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for JSON, summary, and markdown reports.",
    )
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_strict_evidence_post_pilot_promotion_hold_review(
        completion_gate_report_path=args.completion_gate_report,
        paper_ids=args.paper_id or None,
    )
    paths = write_strict_evidence_post_pilot_promotion_hold_review_reports(report, args.output_dir)
    print(f"wrote report: {paths['report']}")
    print(f"wrote summary: {paths['summary']}")
    print(f"wrote markdown: {paths['markdown']}")
    if args.json:
        print(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_COMPLETION_GATE_REPORT_PATH",
    "DEFAULT_OUTPUT_DIR",
    "HOLD_STATUS_ACTIVE",
    "STRICT_EVIDENCE_POST_PILOT_PROMOTION_HOLD_REVIEW_SCHEMA_ID",
    "build_strict_evidence_post_pilot_promotion_hold_review",
    "render_strict_evidence_post_pilot_promotion_hold_review_markdown",
    "write_strict_evidence_post_pilot_promotion_hold_review_reports",
]
