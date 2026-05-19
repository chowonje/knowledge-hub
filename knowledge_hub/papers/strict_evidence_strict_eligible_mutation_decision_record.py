"""Decision record for StrictEvidence strictEligible mutation semantics.

Consumes the post-pilot promotion hold review report and records the policy
decision that StrictEvidence rows must not be mutated in place to become
strict-eligible. This helper is report-only: it does not write eligibility
records, mutate StrictEvidence/SourceSpan stores, or enable runtime surfaces.
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
    EXPECTED_FIGURE_CAPTION_MANIFEST_ROWS,
    EXPECTED_POLICY_CANDIDATE_ROWS,
    EXPECTED_SECTION_MANIFEST_ROWS,
    EXPECTED_SOURCE_SPAN_STORE_ROWS,
    EXPECTED_STRICT_EVIDENCE_STORE_ROWS,
)
from knowledge_hub.papers.strict_evidence_post_pilot_promotion_hold_review import (
    DEFAULT_OUTPUT_DIR as DEFAULT_HOLD_REVIEW_OUTPUT_DIR,
    HOLD_STATUS_ACTIVE,
    STRICT_EVIDENCE_POST_PILOT_PROMOTION_HOLD_REVIEW_SCHEMA_ID,
)


STRICT_EVIDENCE_STRICT_ELIGIBLE_MUTATION_DECISION_RECORD_SCHEMA_ID = (
    "knowledge-hub.paper.strict-evidence-strict-eligible-mutation-decision-record.v1"
)

DECISION_STATUS_CANDIDATE_ONLY = "strict_eligible_mutation_decision_candidate_only"
DECISION_STATUS_BLOCKED_HOLD_NOT_ACTIVE = "blocked_post_pilot_hold_not_active"
DECISION_STATUS_BLOCKED_DOWNSTREAM_GATE = "blocked_downstream_gate_already_enabled"
DECISION_STATUS_BLOCKED_STORE_COUNT = "blocked_store_row_count_changed"
DECISION_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"

DECISION_SEPARATE_ELIGIBILITY_RECORD = "separate_append_only_eligibility_record"

DEFAULT_HOLD_REVIEW_REPORT_PATH = (
    DEFAULT_HOLD_REVIEW_OUTPUT_DIR / "strict-evidence-post-pilot-promotion-hold-review.json"
)

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "strict-evidence-strict-eligible-mutation-decision-record"
    / "01-strict-evidence-strict-eligible-mutation-decision-record"
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

_DOWNSTREAM_GATE_ROW_FIELDS = (
    "strictEligibleMutationAllowed",
    "citationGradeAllowed",
    "runtimeEvidenceAllowed",
    "parserRoutingAllowed",
    "answerIntegrationAllowed",
    "databaseMutationAllowed",
    "reindexOrReembedAllowed",
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


def _strict_eligible_semantics_decision() -> dict[str, Any]:
    return {
        "decision": DECISION_SEPARATE_ELIGIBILITY_RECORD,
        "strictEvidenceInPlaceMutationAllowed": False,
        "strictEligibleBooleanMutationAllowed": False,
        "strictEligibleFlagMeaning": "legacy_compatibility_flag_must_remain_false_on_strict_evidence_records",
        "eligibilityRecordRequired": True,
        "eligibilityRecordAppendOnly": True,
        "eligibilityStoreName": "parsed_artifact_strict_evidence_eligibility_store",
        "eligibilityRecordContractRequired": True,
        "eligibilityRecordRuntimeVisible": False,
        "citationGradeAllowedByThisDecision": False,
        "runtimeEvidenceAllowedByThisDecision": False,
        "answerIntegrationAllowedByThisDecision": False,
        "rationale": [
            "preserve StrictEvidence as an immutable audit record",
            "make eligibility decisions independently reviewable and rollbackable",
            "avoid turning legacy boolean flags into runtime authority",
            "keep citation-grade, runtime binding, and answer integration behind separate gates",
        ],
        "alternativesRejected": [
            {
                "alternative": "mutate_strict_evidence_record_in_place",
                "reason": "would obscure when and why a record became eligible and make rollback ambiguous",
            },
            {
                "alternative": "treat_strictEligible_true_as_runtime_answerability",
                "reason": "would bypass citation-grade, runtime binding, and no-answer safety gates",
            },
        ],
    }


def _no_mutation_policy_matrix() -> dict[str, Any]:
    return {
        "reportOnly": True,
        "decisionRecordOnly": True,
        "manifestWrite": False,
        "eligibilityRecordWrite": False,
        "strictEvidenceStoreWrite": False,
        "sourceSpanStoreWrite": False,
        "strictEligibleMutation": False,
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


def _downstream_gate_allowed_count(hold_review: dict[str, Any]) -> int:
    matrix = hold_review.get("blockedDownstreamGateMatrix")
    if not isinstance(matrix, dict):
        return 0
    return sum(
        1
        for section in matrix.values()
        if isinstance(section, dict) and (_safe_bool(section.get("allowed")) or _safe_bool(section.get("ready")))
    )


def _aggregate_hold_violations(
    *,
    hold_review: dict[str, Any],
    input_schema_violations: list[str],
) -> list[str]:
    violations = list(input_schema_violations)
    counts = hold_review.get("counts") if isinstance(hold_review.get("counts"), dict) else {}
    gate = hold_review.get("gate") if isinstance(hold_review.get("gate"), dict) else {}

    if _safe_text(hold_review.get("status")) != "ok":
        violations.append(f"hold_review_status={_safe_text(hold_review.get('status')) or 'unknown'}")
    if _safe_text(gate.get("holdDecision")) != "strict_evidence_post_pilot_promotion_hold_active":
        violations.append(f"hold_decision={_safe_text(gate.get('holdDecision')) or 'unknown'}")
    if not _safe_bool(gate.get("postPilotPromotionHoldReviewReady")):
        violations.append("post_pilot_promotion_hold_review_not_ready")

    expectations = {
        "inputPolicyCandidateRows": EXPECTED_POLICY_CANDIDATE_ROWS,
        "validatedPilotRows": EXPECTED_POLICY_CANDIDATE_ROWS,
        "holdActiveRows": EXPECTED_POLICY_CANDIDATE_ROWS,
        "sectionHoldRows": EXPECTED_SECTION_MANIFEST_ROWS,
        "figureCaptionHoldRows": EXPECTED_FIGURE_CAPTION_MANIFEST_ROWS,
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

    if _downstream_gate_allowed_count(hold_review) > 0:
        violations.append("downstream_gate_already_enabled")

    return _dedupe(violations)


def _decision_rows(
    hold_rows: list[dict[str, Any]],
    *,
    aggregate_violations: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, hold_row in enumerate(hold_rows):
        source_row = dict(hold_row or {})
        hold_status = _safe_text(source_row.get("hold_status"))
        blockers: list[str] = []
        decision_status = DECISION_STATUS_CANDIDATE_ONLY
        recommended_action = "queue_for_strict_evidence_eligibility_record_contract"

        if aggregate_violations:
            decision_status = DECISION_STATUS_BLOCKED_INPUT_SCHEMA
            blockers.extend(aggregate_violations)
            recommended_action = "repair_post_pilot_hold_review_before_eligibility_decision"
        elif hold_status != HOLD_STATUS_ACTIVE or not _safe_bool(source_row.get("postPilotPromotionHoldActive")):
            decision_status = DECISION_STATUS_BLOCKED_HOLD_NOT_ACTIVE
            blockers.append(f"hold_status={hold_status or 'unknown'}")
            recommended_action = "rerun_post_pilot_promotion_hold_review"
        elif any(_safe_bool(source_row.get(field_name)) for field_name in _DOWNSTREAM_GATE_ROW_FIELDS):
            decision_status = DECISION_STATUS_BLOCKED_DOWNSTREAM_GATE
            blockers.append("row_downstream_gate_already_enabled")
            recommended_action = "disable_downstream_gate_before_eligibility_decision"

        rows.append(
            {
                "decision_row_id": (
                    f"strict-evidence-strict-eligible-mutation-decision-record:{index:04d}"
                ),
                "hold_row_id": _safe_text(source_row.get("hold_row_id")),
                "completion_row_id": _safe_text(source_row.get("completion_row_id")),
                "readback_row_id": _safe_text(source_row.get("readback_row_id")),
                "strictEvidenceId": _safe_text(source_row.get("strictEvidenceId")),
                "sourceSpanId": _safe_text(source_row.get("sourceSpanId")),
                "candidateRecordId": _safe_text(source_row.get("candidateRecordId")),
                "paper_id": _safe_text(source_row.get("paper_id")),
                "artifact_type": _safe_text(source_row.get("artifact_type")),
                "hold_status": hold_status,
                "decision_status": decision_status,
                "decision_blockers": _dedupe(blockers),
                "strictEligibleMutationDecision": DECISION_SEPARATE_ELIGIBILITY_RECORD,
                "strictEvidenceInPlaceMutationAllowed": False,
                "strictEligibleBooleanMutationAllowed": False,
                "eligibilityRecordRequired": True,
                "eligibilityRecordWriteAllowed": False,
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
    hold_counts: dict[str, Any],
    schema_violations: list[str],
) -> dict[str, Any]:
    by_status = Counter(_safe_text(row.get("decision_status")) for row in rows)
    by_artifact = Counter(
        _safe_text(row.get("artifact_type"))
        for row in rows
        if row.get("decision_status") == DECISION_STATUS_CANDIDATE_ONLY
    )
    return {
        "inputPolicyCandidateRows": _safe_int(hold_counts.get("inputPolicyCandidateRows")),
        "inputHoldRows": _safe_int(hold_counts.get("holdActiveRows")),
        "decisionRecordCandidateOnlyRows": int(by_status.get(DECISION_STATUS_CANDIDATE_ONLY, 0)),
        "sectionDecisionRows": int(by_artifact.get("section", 0)),
        "figureCaptionDecisionRows": int(by_artifact.get("figure", 0)),
        "blockedPostPilotHoldNotActiveRows": int(
            by_status.get(DECISION_STATUS_BLOCKED_HOLD_NOT_ACTIVE, 0)
        ),
        "blockedDownstreamGateAlreadyEnabledRows": int(
            by_status.get(DECISION_STATUS_BLOCKED_DOWNSTREAM_GATE, 0)
        ),
        "blockedStoreRowCountChangedRows": int(by_status.get(DECISION_STATUS_BLOCKED_STORE_COUNT, 0)),
        "blockedInputSchemaViolationRows": int(by_status.get(DECISION_STATUS_BLOCKED_INPUT_SCHEMA, 0)),
        "strictEvidenceStoreRows": _safe_int(hold_counts.get("strictEvidenceStoreRows")),
        "sourceSpanStoreRows": _safe_int(hold_counts.get("sourceSpanStoreRows")),
        "strictEligibleMutationAllowedRows": sum(
            1 for row in rows if _safe_bool(row.get("strictEligibleMutationAllowed"))
        ),
        "eligibilityRecordWriteRows": 0,
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
        "schemaViolationCount": len(schema_violations),
        "byArtifactType": dict(by_artifact),
        "byDecisionStatus": dict(by_status),
        "byRecommendedAction": dict(Counter(_safe_text(row.get("recommended_action")) for row in rows)),
    }


def build_strict_evidence_strict_eligible_mutation_decision_record(
    *,
    hold_review_report_path: str | Path = DEFAULT_HOLD_REVIEW_REPORT_PATH,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    report_path = Path(str(hold_review_report_path)).expanduser()
    warnings: list[str] = []
    input_schema_violations: list[str] = []
    requested_papers = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}

    hold_review = _read_json(report_path)
    if not hold_review:
        input_schema_violations.append("hold_review_report_missing_or_unreadable")

    if hold_review:
        validation = validate_payload(
            hold_review,
            STRICT_EVIDENCE_POST_PILOT_PROMOTION_HOLD_REVIEW_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            input_schema_violations.extend(str(error) for error in validation.errors)

    hold_rows = [
        row for row in hold_review.get("rows", []) if isinstance(row, dict)
    ] if hold_review else []

    if requested_papers:
        found_papers = {_safe_text(row.get("paper_id")) for row in hold_rows if _safe_text(row.get("paper_id"))}
        if requested_papers - found_papers:
            warnings.append("requested_paper_ids_not_found")
        hold_rows = [row for row in hold_rows if _safe_text(row.get("paper_id")) in requested_papers]

    if not hold_rows and not input_schema_violations:
        warnings.append("hold_review_rows_missing")

    input_schema_violations = _dedupe(input_schema_violations)
    aggregate_violations = (
        _aggregate_hold_violations(
            hold_review=hold_review,
            input_schema_violations=input_schema_violations,
        )
        if hold_review
        else list(input_schema_violations)
    )

    rows = _decision_rows(hold_rows, aggregate_violations=aggregate_violations)
    hold_counts = hold_review.get("counts") if isinstance(hold_review.get("counts"), dict) else {}
    counts = _count_rows(rows=rows, hold_counts=hold_counts, schema_violations=aggregate_violations)

    decision_candidate_rows = int(counts.get("decisionRecordCandidateOnlyRows") or 0)
    status = "ok"
    if (
        aggregate_violations
        or not rows
        or decision_candidate_rows != EXPECTED_POLICY_CANDIDATE_ROWS
        or decision_candidate_rows != len(rows)
        or int(counts.get("strictEligibleMutationAllowedRows") or 0) != 0
    ):
        status = "blocked"

    decision = _strict_eligible_semantics_decision()
    policy_matrix = _no_mutation_policy_matrix()

    return {
        "schema": STRICT_EVIDENCE_STRICT_ELIGIBLE_MUTATION_DECISION_RECORD_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "holdReviewReportPath": str(report_path),
            "holdReviewReportSchema": _safe_text(hold_review.get("schema")) if hold_review else "",
            "holdReviewReportStatus": _safe_text(hold_review.get("status")) if hold_review else "",
            "holdDecision": _safe_text((hold_review.get("gate") or {}).get("holdDecision"))
            if hold_review
            else "",
            "requestedPaperIds": sorted(requested_papers),
            "expectedPolicyCandidateRows": EXPECTED_POLICY_CANDIDATE_ROWS,
            "expectedDecisionCandidateRows": EXPECTED_POLICY_CANDIDATE_ROWS,
            "expectedSectionDecisionRows": EXPECTED_SECTION_MANIFEST_ROWS,
            "expectedFigureCaptionDecisionRows": EXPECTED_FIGURE_CAPTION_MANIFEST_ROWS,
        },
        "counts": counts,
        "strictEligibleSemanticsDecision": decision,
        "blockedDownstreamGateMatrix": (
            hold_review.get("blockedDownstreamGateMatrix")
            if isinstance(hold_review.get("blockedDownstreamGateMatrix"), dict)
            else {}
        ),
        "noMutationPolicyMatrix": policy_matrix,
        "gate": {
            "strictEligibleMutationDecisionRecordReady": status == "ok",
            "decision": decision["decision"],
            "strictEvidenceInPlaceMutationAllowed": False,
            "strictEligibleBooleanMutationAllowed": False,
            "strictEligibleMutationAllowed": False,
            "eligibilityRecordContractRequired": True,
            "eligibilityRecordWriteAllowed": False,
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
                "strict_evidence_eligibility_record_contract"
                if status == "ok"
                else "strict_evidence_post_pilot_promotion_hold_review_repair"
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
            "strictEligibleSemanticsDecision",
            "blockedDownstreamGateMatrix",
            "noMutationPolicyMatrix",
            "gate",
            "policy",
            "warnings",
        )
        if key in report
    }


def render_strict_evidence_strict_eligible_mutation_decision_record_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    decision = dict(report.get("strictEligibleSemanticsDecision") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byDecisionStatus") or {})).items())
    ]
    rationale = [f"- {item}" for item in list(decision.get("rationale") or [])]
    return "\n".join(
        [
            "# Strict Evidence strictEligible Mutation Decision Record",
            "",
            f"- status: {report.get('status', '')}",
            f"- decision: {gate.get('decision', '')}",
            f"- input hold rows: {int(counts.get('inputHoldRows') or 0)}",
            f"- decision candidate rows: {int(counts.get('decisionRecordCandidateOnlyRows') or 0)}",
            f"- section decision rows: {int(counts.get('sectionDecisionRows') or 0)}",
            f"- figure caption decision rows: {int(counts.get('figureCaptionDecisionRows') or 0)}",
            f"- strict eligible mutation allowed rows: {int(counts.get('strictEligibleMutationAllowedRows') or 0)}",
            f"- eligibility record writes: {int(counts.get('eligibilityRecordWriteRows') or 0)}",
            "",
            "## Decision",
            f"- in-place StrictEvidence mutation allowed: {json.dumps(gate.get('strictEvidenceInPlaceMutationAllowed'))}",
            f"- strictEligible boolean mutation allowed: {json.dumps(gate.get('strictEligibleBooleanMutationAllowed'))}",
            f"- eligibility record contract required: {json.dumps(gate.get('eligibilityRecordContractRequired'))}",
            "",
            "## Rationale",
            *rationale,
            "",
            "## Decision status breakdown",
            *[f"- {item}" for item in by_status],
            "",
            f"- recommended next tranche: {gate.get('recommendedNextTranche', '')}",
        ]
    )


def write_strict_evidence_strict_eligible_mutation_decision_record_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "strict-evidence-strict-eligible-mutation-decision-record.json"
    summary_path = root / "strict-evidence-strict-eligible-mutation-decision-record-summary.json"
    markdown_path = root / "strict-evidence-strict-eligible-mutation-decision-record.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_strict_evidence_strict_eligible_mutation_decision_record_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Record strictEligible mutation semantics for StrictEvidence pilot rows "
            "without mutating stores or enabling runtime surfaces."
        )
    )
    parser.add_argument(
        "--hold-review-report",
        default=str(DEFAULT_HOLD_REVIEW_REPORT_PATH),
        help="Path to the post-pilot promotion hold review JSON report.",
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for JSON, summary, and markdown reports.",
    )
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_strict_evidence_strict_eligible_mutation_decision_record(
        hold_review_report_path=args.hold_review_report,
        paper_ids=args.paper_id or None,
    )
    paths = write_strict_evidence_strict_eligible_mutation_decision_record_reports(
        report,
        args.output_dir,
    )
    print(f"wrote report: {paths['report']}")
    print(f"wrote summary: {paths['summary']}")
    print(f"wrote markdown: {paths['markdown']}")
    if args.json:
        print(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_HOLD_REVIEW_REPORT_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DECISION_SEPARATE_ELIGIBILITY_RECORD",
    "DECISION_STATUS_CANDIDATE_ONLY",
    "STRICT_EVIDENCE_STRICT_ELIGIBLE_MUTATION_DECISION_RECORD_SCHEMA_ID",
    "build_strict_evidence_strict_eligible_mutation_decision_record",
    "render_strict_evidence_strict_eligible_mutation_decision_record_markdown",
    "write_strict_evidence_strict_eligible_mutation_decision_record_reports",
]
