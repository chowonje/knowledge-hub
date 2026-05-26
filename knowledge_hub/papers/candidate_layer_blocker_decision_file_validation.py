"""Report-only validation for candidate-layer blocker decision files.

This helper validates a future human/operator decision file against the
decision input pack. It does not record decisions, execute operator approvals,
create evidence, route parsers, or modify canonical artifacts.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


CANDIDATE_LAYER_BLOCKER_DECISION_FILE_VALIDATION_SCHEMA_ID = (
    "knowledge-hub.paper.candidate-layer-blocker-decision-file-validation.v1"
)
CANDIDATE_LAYER_BLOCKER_DECISION_INPUT_PACK_SCHEMA_ID = (
    "knowledge-hub.paper.candidate-layer-blocker-decision-input-pack.v1"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        payload = json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _input_rows(input_pack: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in list(input_pack.get("decisionInputs") or []) if isinstance(item, dict)]


def _decision_rows(decision_file: dict[str, Any]) -> list[dict[str, Any]]:
    rows = decision_file.get("decisions")
    if rows is None:
        rows = decision_file.get("decisionInputs")
    if rows is None:
        rows = decision_file.get("decisionRows")
    return [dict(item) for item in list(rows or []) if isinstance(item, dict)]


def _decision_id(item: dict[str, Any]) -> str:
    return str(
        item.get("source_decision_row_id")
        or item.get("sourceDecisionRowId")
        or item.get("decision_row_id")
        or item.get("decisionRowId")
        or ""
    )


def _decision_value(item: dict[str, Any]) -> str:
    return str(item.get("decision") or item.get("review_decision") or item.get("reviewDecision") or "")


def _unsafe_input_pack_flags(input_pack: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    counts = dict(input_pack.get("counts") or {})
    gate = dict(input_pack.get("gate") or {})
    policy = dict(input_pack.get("policy") or {})
    if input_pack.get("schema") != CANDIDATE_LAYER_BLOCKER_DECISION_INPUT_PACK_SCHEMA_ID:
        flags.append("candidate_layer_blocker_decision_input_pack_schema_mismatch")
    if str(input_pack.get("status") or "") == "blocked":
        flags.append("candidate_layer_blocker_decision_input_pack_blocked")
    for key in ("acceptedDecisionRows", "operatorApprovedRows", "strictEligibleRows", "runtimeEvidenceRows"):
        if _safe_int(counts.get(key)) > 0:
            flags.append(f"decisionInputPack_{key}_nonzero")
    for key in (
        "strictEvidenceReady",
        "parserRoutingReady",
        "answerIntegrationReady",
        "runtimePromotionAllowed",
        "containsAcceptedDecisions",
        "containsOperatorApprovals",
    ):
        if bool(gate.get(key)):
            flags.append(f"decisionInputPack_{key}_true")
    for key in (
        "strictEvidenceCreated",
        "runtimePromotionAllowed",
        "parserRoutingChanged",
        "canonicalParsedArtifactsWritten",
        "databaseMutation",
        "reindexOrReembed",
        "answerIntegrationChanged",
    ):
        if bool(policy.get(key)):
            flags.append(f"decisionInputPack_{key}_true")
    return list(dict.fromkeys(flags))


def _submitted_decisions(decision_file: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    mapped: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    seen: set[str] = set()
    for item in _decision_rows(decision_file):
        row_id = _decision_id(item)
        if not row_id:
            errors.append("decision_file_row_id_missing")
            continue
        if row_id in seen:
            errors.append("decision_file_duplicate_row_id")
        seen.add(row_id)
        if row_id not in mapped:
            mapped[row_id] = dict(item)
    return mapped, list(dict.fromkeys(errors))


def _input_by_source_id(input_pack: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item.get("source_decision_row_id") or ""): item for item in _input_rows(input_pack)}


def _global_file_errors(input_pack: dict[str, Any], decision_file: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    input_ids = set(_input_by_source_id(input_pack))
    for item in _decision_rows(decision_file):
        row_id = _decision_id(item)
        if row_id and row_id not in input_ids:
            errors.append("decision_file_unknown_input_row_id")
    return list(dict.fromkeys(errors))


def _validation_row(index: int, input_row: dict[str, Any], submitted: dict[str, Any] | None) -> dict[str, Any]:
    source_id = str(input_row.get("source_decision_row_id") or "")
    allowed = [str(item) for item in list(input_row.get("allowed_decisions") or [])]
    decision = _decision_value(submitted or {}) if submitted else ""
    reviewer = str((submitted or {}).get("reviewer") or "")
    notes = str((submitted or {}).get("notes") or "")
    errors: list[str] = []
    if not submitted:
        errors.append("decision_missing")
        validation_status = "missing"
        submitted_decision = "needs_review"
    else:
        submitted_decision = decision or "needs_review"
        if submitted_decision not in allowed:
            errors.append("decision_not_allowed_for_review_bucket")
        if submitted_decision != "needs_review" and not reviewer:
            errors.append("reviewer_required_for_non_needs_review_decision")
        validation_status = "valid" if not errors else "invalid"
    return {
        "validation_row_id": f"candidate-layer-blocker-decision-file-validation:{index:04d}",
        "source_input_row_id": str(input_row.get("input_row_id") or ""),
        "source_decision_row_id": source_id,
        "source_review_card_id": str(input_row.get("source_review_card_id") or ""),
        "source_backlog_id": str(input_row.get("source_backlog_id") or ""),
        "blocker": str(input_row.get("blocker") or ""),
        "priority": str(input_row.get("priority") or ""),
        "review_bucket": str(input_row.get("review_bucket") or ""),
        "affected_layers": list(input_row.get("affected_layers") or []),
        "allowed_decisions": allowed,
        "submitted_decision": submitted_decision,
        "reviewer": reviewer,
        "notes": notes,
        "validation_status": validation_status,
        "validation_errors": errors,
        "decision_scope": "candidate_layer_blocker_decision_file_validation_only_no_runtime_or_strict_promotion",
        "evidence_tier": "candidate_layer_blocker_decision_file_validation_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": [
            "candidate_layer_blocker_decision_file_validation_only",
            "decisions_not_recorded_by_validation_report",
            "strict_promotion_requires_later_explicit_tranche",
            "runtime_promotion_disabled_for_tranche",
        ],
        "non_strict_reason": [
            "validation_rows_are_not_decision_records",
            "validation_rows_do_not_authorize_runtime_use",
            "validation_rows_do_not_create_strict_evidence",
        ],
    }


def _counts(rows: list[dict[str, Any]], unsafe_flags: list[str], file_errors: list[str]) -> dict[str, Any]:
    by_status = Counter(str(row.get("validation_status") or "") for row in rows)
    by_bucket = Counter(str(row.get("review_bucket") or "") for row in rows)
    by_decision = Counter(str(row.get("submitted_decision") or "") for row in rows)
    by_layer = Counter(layer for row in rows for layer in list(row.get("affected_layers") or []))
    return {
        "validationRows": len(rows),
        "validRows": by_status.get("valid", 0),
        "invalidRows": by_status.get("invalid", 0),
        "missingRows": by_status.get("missing", 0),
        "needsReviewRows": by_decision.get("needs_review", 0),
        "nonNeedsReviewRows": len(rows) - by_decision.get("needs_review", 0),
        "fileErrorCount": len(file_errors),
        "acceptedDecisionRows": 0,
        "operatorApprovedRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "byStatus": dict(by_status),
        "byBucket": dict(by_bucket),
        "byDecision": dict(by_decision),
        "byLayer": dict(by_layer),
    }


def build_candidate_layer_blocker_decision_file_validation(
    *,
    candidate_layer_blocker_decision_input_pack_report: str | Path,
    candidate_layer_blocker_decisions_file: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a future blocker decision file without recording decisions."""

    input_pack_path = Path(str(candidate_layer_blocker_decision_input_pack_report)).expanduser()
    decision_file_path = (
        Path(str(candidate_layer_blocker_decisions_file)).expanduser()
        if candidate_layer_blocker_decisions_file
        else None
    )
    input_pack = _read_json(input_pack_path)
    decision_file = _read_json(decision_file_path) if decision_file_path else {}
    submitted, duplicate_errors = _submitted_decisions(decision_file)
    file_errors = list(dict.fromkeys([*duplicate_errors, *_global_file_errors(input_pack, decision_file)]))
    unsafe_flags = _unsafe_input_pack_flags(input_pack)
    rows = [
        _validation_row(index, input_row, submitted.get(str(input_row.get("source_decision_row_id") or "")))
        for index, input_row in enumerate(_input_rows(input_pack), start=1)
    ]
    counts = _counts(rows, unsafe_flags, file_errors)
    if unsafe_flags or file_errors:
        status = "blocked"
        decision = "blocked"
    elif not decision_file_path:
        status = "decision_file_required"
        decision = "manual_decision_file_missing"
    elif _safe_int(counts.get("missingRows")) or _safe_int(counts.get("invalidRows")):
        status = "decision_file_incomplete"
        decision = "manual_decision_file_incomplete"
    else:
        status = "decision_file_validated"
        decision = "manual_decision_file_validated_non_runtime"
    return {
        "schema": CANDIDATE_LAYER_BLOCKER_DECISION_FILE_VALIDATION_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "candidateLayerBlockerDecisionInputPackReport": str(input_pack_path),
            "candidateLayerBlockerDecisionInputPackSchema": str(input_pack.get("schema") or ""),
            "candidateLayerBlockerDecisionsFile": str(decision_file_path or ""),
            "candidateLayerBlockerDecisionFileRows": len(_decision_rows(decision_file)),
        },
        "counts": counts,
        "gate": {
            "decisionFileValidationReady": bool(rows) and not unsafe_flags,
            "decisionFileComplete": status == "decision_file_validated",
            "containsRecordedDecisions": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": decision,
            "unsafeUpstreamFlags": unsafe_flags,
            "fileValidationErrors": file_errors,
            "recommendedNextTranche": "manual_fill_candidate_layer_blocker_decision_file"
            if status != "decision_file_validated"
            else "candidate_layer_blocker_decision_record_from_validated_file_requires_explicit_user_approval",
        },
        "policy": {
            "reportOnly": True,
            "decisionFileValidationOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "decision_file_validation_does_not_record_decisions",
            "valid_decision_rows_do_not_authorize_runtime_use",
            "operator_approval_validation_does_not_execute_diagnostic_actions",
            "strict_or_runtime_promotion_requires_a_separate_explicit_tranche",
        ],
        "validationRows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings")
        if key in report
    }


def render_candidate_layer_blocker_decision_file_validation_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# Candidate Layer Blocker Decision File Validation",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Validation rows: `{int(counts.get('validationRows') or 0)}`",
        f"- Valid rows: `{int(counts.get('validRows') or 0)}`",
        f"- Missing rows: `{int(counts.get('missingRows') or 0)}`",
        f"- Invalid rows: `{int(counts.get('invalidRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This validation report is report-only. It does not record decisions, create strict evidence, allow runtime citations, route parsers, write canonical parsed artifacts, mutate DB state, reindex, reembed, or change answer behavior.",
        "",
        "## Counts",
        "",
        f"- By status: `{json.dumps(counts.get('byStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By bucket: `{json.dumps(counts.get('byBucket') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By decision: `{json.dumps(counts.get('byDecision') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_candidate_layer_blocker_decision_file_validation_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    validation_path = root / "candidate-layer-blocker-decision-file-validation.json"
    summary_path = root / "candidate-layer-blocker-decision-file-validation-summary.json"
    markdown_path = root / "candidate-layer-blocker-decision-file-validation.md"
    validation_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_candidate_layer_blocker_decision_file_validation_markdown(report), encoding="utf-8")
    return {"validation": str(validation_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Validate a candidate-layer blocker decision file without applying it.")
    parser.add_argument("--candidate-layer-blocker-decision-input-pack-report", required=True)
    parser.add_argument("--candidate-layer-blocker-decisions-file", default="")
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_candidate_layer_blocker_decision_file_validation(
        candidate_layer_blocker_decision_input_pack_report=args.candidate_layer_blocker_decision_input_pack_report,
        candidate_layer_blocker_decisions_file=args.candidate_layer_blocker_decisions_file or None,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_candidate_layer_blocker_decision_file_validation_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_LAYER_BLOCKER_DECISION_FILE_VALIDATION_SCHEMA_ID",
    "build_candidate_layer_blocker_decision_file_validation",
    "render_candidate_layer_blocker_decision_file_validation_markdown",
    "write_candidate_layer_blocker_decision_file_validation_reports",
]
