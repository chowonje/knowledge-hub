"""Completion gate for the StrictEvidence manifest-only pilot tranche.

Consumes the pilot tranche manifest readback review report and emits a
completion/sign-off decision for the full 99-row pilot set. Report-only: does not
mutate StrictEvidence or SourceSpan stores, create manifests, or enable integration.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_pilot_tranche_manifest_readback_review import (
    EXPECTED_FIGURE_CAPTION_MANIFEST_ROWS,
    EXPECTED_POLICY_CANDIDATE_ROWS,
    EXPECTED_SECTION_MANIFEST_ROWS,
    EXPECTED_SOURCE_SPAN_STORE_ROWS,
    EXPECTED_STRICT_EVIDENCE_STORE_ROWS,
    READBACK_STATUS_BLOCKED_DUPLICATE_ID,
    READBACK_STATUS_BLOCKED_INPUT_SCHEMA,
    READBACK_STATUS_BLOCKED_MISSING_POLICY,
    READBACK_STATUS_BLOCKED_RUNTIME_OR_CITATION,
    READBACK_STATUS_BLOCKED_STORE_COUNT,
    READBACK_STATUS_BLOCKED_UNEXPECTED_MANIFEST,
    READBACK_STATUS_VALIDATED,
    STRICT_EVIDENCE_PILOT_TRANCHE_MANIFEST_READBACK_REVIEW_SCHEMA_ID,
)


STRICT_EVIDENCE_PILOT_TRANCHE_COMPLETION_GATE_SCHEMA_ID = (
    "knowledge-hub.paper.strict-evidence-pilot-tranche-completion-gate.v1"
)

COMPLETION_STATUS_COMPLETE = "strict_evidence_pilot_tranche_complete_candidate_only"
COMPLETION_STATUS_BLOCKED_READBACK = "blocked_manifest_readback_not_validated"
COMPLETION_STATUS_BLOCKED_MISSING_POLICY = "blocked_missing_policy_candidate"
COMPLETION_STATUS_BLOCKED_UNEXPECTED_MANIFEST = "blocked_unexpected_manifest_row"
COMPLETION_STATUS_BLOCKED_DUPLICATE_ID = "blocked_duplicate_strict_evidence_id"
COMPLETION_STATUS_BLOCKED_STORE_COUNT = "blocked_store_row_count_changed"
COMPLETION_STATUS_BLOCKED_RUNTIME_OR_CITATION = "blocked_runtime_or_citation_flag_violation"
COMPLETION_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"

DEFAULT_MANIFEST_READBACK_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "strict-evidence-pilot-tranche-manifest-readback-review"
    / "01-strict-evidence-pilot-tranche-manifest-readback-review"
    / "strict-evidence-pilot-tranche-manifest-readback-review.json"
)

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "strict-evidence-pilot-tranche-completion-gate"
    / "01-strict-evidence-pilot-tranche-completion-gate"
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


def _blocked_later_gates() -> dict[str, Any]:
    return {
        "citationGradeEvidence": {
            "ready": False,
            "allowed": False,
            "reason": "blocked_until_explicit_post_pilot_citation_promotion_tranche",
        },
        "runtimeEvidence": {
            "ready": False,
            "allowed": False,
            "reason": "blocked_until_explicit_post_pilot_runtime_promotion_tranche",
        },
        "parserRouting": {
            "ready": False,
            "allowed": False,
            "reason": "blocked_until_explicit_post_pilot_parser_routing_tranche",
        },
        "answerIntegration": {
            "ready": False,
            "allowed": False,
            "reason": "blocked_until_explicit_post_pilot_answer_integration_tranche",
        },
        "strictEligibleMutation": {
            "ready": False,
            "allowed": False,
            "reason": "blocked_until_explicit_post_pilot_strict_eligible_tranche",
        },
    }


def _no_mutation_policy_matrix() -> dict[str, Any]:
    return {
        "reportOnly": True,
        "completionGateOnly": True,
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


def _mutation_flag_violation(row: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for field_name in (
        "strictEligible",
        "strictEvidenceCreated",
        "citationGrade",
        "runtimeEvidence",
        "parserRoutingChanged",
        "answerIntegrationChanged",
        "databaseMutation",
    ):
        if _safe_bool(row.get(field_name)):
            violations.append(f"{field_name}_true")
    return violations


def _map_readback_status_to_completion(readback_status: str) -> str:
    mapping = {
        READBACK_STATUS_VALIDATED: COMPLETION_STATUS_COMPLETE,
        READBACK_STATUS_BLOCKED_MISSING_POLICY: COMPLETION_STATUS_BLOCKED_MISSING_POLICY,
        READBACK_STATUS_BLOCKED_UNEXPECTED_MANIFEST: COMPLETION_STATUS_BLOCKED_UNEXPECTED_MANIFEST,
        READBACK_STATUS_BLOCKED_DUPLICATE_ID: COMPLETION_STATUS_BLOCKED_DUPLICATE_ID,
        READBACK_STATUS_BLOCKED_STORE_COUNT: COMPLETION_STATUS_BLOCKED_STORE_COUNT,
        READBACK_STATUS_BLOCKED_RUNTIME_OR_CITATION: COMPLETION_STATUS_BLOCKED_RUNTIME_OR_CITATION,
        READBACK_STATUS_BLOCKED_INPUT_SCHEMA: COMPLETION_STATUS_BLOCKED_INPUT_SCHEMA,
    }
    if readback_status in mapping:
        return mapping[readback_status]
    return COMPLETION_STATUS_BLOCKED_READBACK


def _aggregate_readback_violations(
    *,
    readback_report: dict[str, Any],
    input_schema_violations: list[str],
) -> list[str]:
    violations = list(input_schema_violations)
    counts = readback_report.get("counts") if isinstance(readback_report.get("counts"), dict) else {}
    gate = readback_report.get("gate") if isinstance(readback_report.get("gate"), dict) else {}

    if _safe_text(readback_report.get("status")) != "ok":
        violations.append(
            f"manifest_readback_report_status={_safe_text(readback_report.get('status')) or 'unknown'}"
        )
    if not _safe_bool(gate.get("pilotManifestReadbackReviewReady")):
        violations.append("pilot_manifest_readback_review_not_ready")

    expectations = {
        "inputPolicyCandidateRows": EXPECTED_POLICY_CANDIDATE_ROWS,
        "pilotManifestReadbackValidatedRows": EXPECTED_POLICY_CANDIDATE_ROWS,
        "sectionManifestRows": EXPECTED_SECTION_MANIFEST_ROWS,
        "figureCaptionManifestRows": EXPECTED_FIGURE_CAPTION_MANIFEST_ROWS,
        "combinedManifestRows": EXPECTED_POLICY_CANDIDATE_ROWS,
        "missingPolicyCandidateRows": 0,
        "unexpectedManifestRows": 0,
        "duplicateStrictEvidenceIdRows": 0,
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

    diagnostics = (
        readback_report.get("diagnostics")
        if isinstance(readback_report.get("diagnostics"), dict)
        else {}
    )
    for field_name in (
        "duplicateStrictEvidenceIds",
        "missingPolicyCandidateIds",
        "unexpectedManifestRowIds",
    ):
        items = diagnostics.get(field_name)
        if isinstance(items, list) and items:
            violations.append(f"diagnostics.{field_name}_non_empty")

    return _dedupe(violations)


def _completion_rows(
    readback_rows: list[dict[str, Any]],
    *,
    aggregate_violations: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, readback_row in enumerate(readback_rows):
        source_row = dict(readback_row or {})
        readback_status = _safe_text(source_row.get("readback_status"))
        blockers: list[str] = []
        completion_status = _map_readback_status_to_completion(readback_status)

        if aggregate_violations:
            completion_status = COMPLETION_STATUS_BLOCKED_INPUT_SCHEMA
            blockers.extend(aggregate_violations)
        elif readback_status != READBACK_STATUS_VALIDATED:
            blockers.extend(_safe_text(item) for item in (source_row.get("readback_blockers") or []))
            blockers.append(f"readback_status={readback_status or 'unknown'}")
        else:
            flag_violations = _mutation_flag_violation(source_row)
            if flag_violations:
                completion_status = COMPLETION_STATUS_BLOCKED_RUNTIME_OR_CITATION
                blockers.extend(flag_violations)

        complete_candidate = completion_status == COMPLETION_STATUS_COMPLETE
        rows.append(
            {
                "completion_row_id": f"strict-evidence-pilot-tranche-completion-gate:{index:04d}",
                "readback_row_id": _safe_text(source_row.get("readback_row_id")),
                "policy_gate_row_id": _safe_text(source_row.get("policy_gate_row_id")),
                "strictEvidenceId": _safe_text(source_row.get("strictEvidenceId")),
                "sourceSpanId": _safe_text(source_row.get("sourceSpanId")),
                "candidateRecordId": _safe_text(source_row.get("candidateRecordId")),
                "paper_id": _safe_text(source_row.get("paper_id")),
                "artifact_type": _safe_text(source_row.get("artifact_type")),
                "manifestType": _safe_text(source_row.get("manifestType")),
                "readback_status": readback_status,
                "completion_status": completion_status,
                "completion_blockers": _dedupe(blockers),
                "pilotTrancheCompleteCandidateOnly": complete_candidate,
                "strictEligible": False,
                "strictEvidenceCreated": False,
                "citationGrade": False,
                "runtimeEvidence": False,
                "parserRoutingChanged": False,
                "answerIntegrationChanged": False,
                "databaseMutation": False,
                "recommended_action": (
                    "strict_evidence_pilot_tranche_complete_candidate_only"
                    if complete_candidate
                    else "repair_pilot_manifest_readback_before_completion_gate"
                ),
            }
        )
    return rows


def _count_rows(
    *,
    rows: list[dict[str, Any]],
    readback_counts: dict[str, Any],
    input_schema_violations: list[str],
) -> dict[str, Any]:
    by_status = Counter(_safe_text(row.get("completion_status")) for row in rows)
    by_artifact = Counter(
        _safe_text(row.get("artifact_type"))
        for row in rows
        if row.get("completion_status") == COMPLETION_STATUS_COMPLETE
    )
    return {
        "inputPolicyCandidateRows": _safe_int(readback_counts.get("inputPolicyCandidateRows")),
        "validatedPilotRows": _safe_int(readback_counts.get("pilotManifestReadbackValidatedRows")),
        "sectionValidatedRows": _safe_int(readback_counts.get("sectionManifestRows")),
        "figureCaptionValidatedRows": _safe_int(readback_counts.get("figureCaptionManifestRows")),
        "completionCandidateOnlyRows": int(by_status.get(COMPLETION_STATUS_COMPLETE, 0)),
        "blockedManifestReadbackNotValidatedRows": int(
            by_status.get(COMPLETION_STATUS_BLOCKED_READBACK, 0)
        ),
        "blockedMissingPolicyCandidateRows": int(
            by_status.get(COMPLETION_STATUS_BLOCKED_MISSING_POLICY, 0)
        ),
        "blockedUnexpectedManifestRows": int(
            by_status.get(COMPLETION_STATUS_BLOCKED_UNEXPECTED_MANIFEST, 0)
        ),
        "blockedDuplicateStrictEvidenceIdRows": int(
            by_status.get(COMPLETION_STATUS_BLOCKED_DUPLICATE_ID, 0)
        ),
        "blockedStoreRowCountChangedRows": int(
            by_status.get(COMPLETION_STATUS_BLOCKED_STORE_COUNT, 0)
        ),
        "blockedRuntimeOrCitationFlagViolationRows": int(
            by_status.get(COMPLETION_STATUS_BLOCKED_RUNTIME_OR_CITATION, 0)
        ),
        "blockedInputSchemaViolationRows": int(
            by_status.get(COMPLETION_STATUS_BLOCKED_INPUT_SCHEMA, 0)
        ),
        "strictEvidenceStoreRows": _safe_int(readback_counts.get("strictEvidenceStoreRows")),
        "sourceSpanStoreRows": _safe_int(readback_counts.get("sourceSpanStoreRows")),
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
        "schemaViolationCount": len(input_schema_violations),
        "byArtifactType": dict(by_artifact),
        "byCompletionStatus": dict(by_status),
        "byRecommendedAction": dict(Counter(_safe_text(row.get("recommended_action")) for row in rows)),
    }


def build_strict_evidence_pilot_tranche_completion_gate(
    *,
    manifest_readback_report_path: str | Path = DEFAULT_MANIFEST_READBACK_REPORT_PATH,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    report_path = Path(str(manifest_readback_report_path)).expanduser()
    warnings: list[str] = []
    input_schema_violations: list[str] = []
    requested_papers = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}

    readback_report = _read_json(report_path)
    if not readback_report:
        input_schema_violations.append("manifest_readback_report_missing_or_unreadable")

    if readback_report:
        validation = validate_payload(
            readback_report,
            STRICT_EVIDENCE_PILOT_TRANCHE_MANIFEST_READBACK_REVIEW_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            input_schema_violations.extend(str(error) for error in validation.errors)

    readback_rows = [
        row for row in readback_report.get("rows", []) if isinstance(row, dict)
    ] if readback_report else []

    if requested_papers:
        found_papers = {_safe_text(row.get("paper_id")) for row in readback_rows if _safe_text(row.get("paper_id"))}
        if requested_papers - found_papers:
            warnings.append("requested_paper_ids_not_found")
        readback_rows = [row for row in readback_rows if _safe_text(row.get("paper_id")) in requested_papers]

    if not readback_rows and not input_schema_violations:
        warnings.append("manifest_readback_rows_missing")

    input_schema_violations = _dedupe(input_schema_violations)
    aggregate_violations = (
        _aggregate_readback_violations(
            readback_report=readback_report,
            input_schema_violations=input_schema_violations,
        )
        if readback_report
        else list(input_schema_violations)
    )

    rows = _completion_rows(readback_rows, aggregate_violations=aggregate_violations)
    readback_counts = (
        readback_report.get("counts") if isinstance(readback_report.get("counts"), dict) else {}
    )
    counts = _count_rows(
        rows=rows,
        readback_counts=readback_counts,
        input_schema_violations=aggregate_violations,
    )

    completion_rows = int(counts.get("completionCandidateOnlyRows") or 0)
    status = "ok"
    if (
        aggregate_violations
        or not rows
        or completion_rows != EXPECTED_POLICY_CANDIDATE_ROWS
        or completion_rows != len(rows)
    ):
        status = "blocked"

    blocked_later = _blocked_later_gates()
    policy_matrix = _no_mutation_policy_matrix()

    return {
        "schema": STRICT_EVIDENCE_PILOT_TRANCHE_COMPLETION_GATE_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "manifestReadbackReportPath": str(report_path),
            "manifestReadbackReportSchema": _safe_text(readback_report.get("schema")) if readback_report else "",
            "manifestReadbackReportStatus": _safe_text(readback_report.get("status")) if readback_report else "",
            "requestedPaperIds": sorted(requested_papers),
            "sectionRunManifestPath": _safe_text(
                (readback_report.get("input") or {}).get("sectionRunManifestPath")
            )
            if readback_report
            else "",
            "figureCaptionRunManifestPath": _safe_text(
                (readback_report.get("input") or {}).get("figureCaptionRunManifestPath")
            )
            if readback_report
            else "",
            "expectedPolicyCandidateRows": EXPECTED_POLICY_CANDIDATE_ROWS,
            "expectedSectionValidatedRows": EXPECTED_SECTION_MANIFEST_ROWS,
            "expectedFigureCaptionValidatedRows": EXPECTED_FIGURE_CAPTION_MANIFEST_ROWS,
            "expectedStrictEvidenceStoreRows": EXPECTED_STRICT_EVIDENCE_STORE_ROWS,
            "expectedSourceSpanStoreRows": EXPECTED_SOURCE_SPAN_STORE_ROWS,
        },
        "counts": counts,
        "blockedLaterGates": blocked_later,
        "noMutationPolicyMatrix": policy_matrix,
        "gate": {
            "pilotTrancheCompletionGateReady": status == "ok",
            "completionDecision": (
                "strict_evidence_pilot_tranche_complete_candidate_only"
                if status == "ok"
                else "strict_evidence_pilot_tranche_completion_blocked"
            ),
            "strictEligibleMutationAllowed": False,
            "strictEvidenceStoreWriteAllowed": False,
            "sourceSpanStoreWriteAllowed": False,
            "runManifestWriteAllowed": False,
            "strictEvidenceCreated": False,
            "citationReady": False,
            "runtimeEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": aggregate_violations,
            "recommendedNextTranche": (
                "strict_evidence_post_pilot_promotion_hold_review"
                if status == "ok"
                else "strict_evidence_pilot_tranche_manifest_readback_review_repair"
            ),
        },
        "policy": {
            **policy_matrix,
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
            "blockedLaterGates",
            "noMutationPolicyMatrix",
            "gate",
            "policy",
            "warnings",
        )
        if key in report
    }


def render_strict_evidence_pilot_tranche_completion_gate_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    matrix = dict(report.get("noMutationPolicyMatrix") or {})
    blocked_later = dict(report.get("blockedLaterGates") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byCompletionStatus") or {})).items())
    ]
    blocked_lines = [
        f"- {name}: ready={json.dumps(section.get('ready'))}, allowed={json.dumps(section.get('allowed'))}"
        for name, section in sorted(blocked_later.items())
    ]
    return "\n".join(
        [
            "# Strict Evidence Pilot Tranche Completion Gate",
            "",
            f"- status: {report.get('status', '')}",
            f"- completion decision: {gate.get('completionDecision', '')}",
            f"- input policy candidate rows: {int(counts.get('inputPolicyCandidateRows') or 0)}",
            f"- validated pilot rows: {int(counts.get('validatedPilotRows') or 0)}",
            f"- section validated rows: {int(counts.get('sectionValidatedRows') or 0)}",
            f"- figure caption validated rows: {int(counts.get('figureCaptionValidatedRows') or 0)}",
            f"- completion candidate-only rows: {int(counts.get('completionCandidateOnlyRows') or 0)}",
            f"- strict evidence store rows: {int(counts.get('strictEvidenceStoreRows') or 0)}",
            f"- source span store rows: {int(counts.get('sourceSpanStoreRows') or 0)}",
            "",
            "## No-mutation policy matrix",
            f"- report only: {json.dumps(matrix.get('reportOnly'))}",
            f"- strict evidence store write: {json.dumps(matrix.get('strictEvidenceStoreWrite'))}",
            f"- source span store write: {json.dumps(matrix.get('sourceSpanStoreWrite'))}",
            "",
            "## Blocked-later gates",
            *blocked_lines,
            "",
            "## Completion status breakdown",
            *[f"- {item}" for item in by_status],
            "",
            f"- recommended next tranche: {gate.get('recommendedNextTranche', '')}",
        ]
    )


def write_strict_evidence_pilot_tranche_completion_gate_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "strict-evidence-pilot-tranche-completion-gate.json"
    summary_path = root / "strict-evidence-pilot-tranche-completion-gate-summary.json"
    markdown_path = root / "strict-evidence-pilot-tranche-completion-gate.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_strict_evidence_pilot_tranche_completion_gate_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Emit a completion/sign-off gate for the StrictEvidence manifest-only pilot "
            "tranche without mutating stores or integration surfaces."
        )
    )
    parser.add_argument(
        "--manifest-readback-report",
        default=str(DEFAULT_MANIFEST_READBACK_REPORT_PATH),
        help="Path to the pilot tranche manifest readback review JSON report.",
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for JSON, summary, and markdown reports.",
    )
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_strict_evidence_pilot_tranche_completion_gate(
        manifest_readback_report_path=args.manifest_readback_report,
        paper_ids=args.paper_id or None,
    )
    paths = write_strict_evidence_pilot_tranche_completion_gate_reports(report, args.output_dir)
    print(f"wrote report: {paths['report']}")
    print(f"wrote summary: {paths['summary']}")
    print(f"wrote markdown: {paths['markdown']}")
    if args.json:
        print(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "COMPLETION_STATUS_COMPLETE",
    "DEFAULT_MANIFEST_READBACK_REPORT_PATH",
    "DEFAULT_OUTPUT_DIR",
    "STRICT_EVIDENCE_PILOT_TRANCHE_COMPLETION_GATE_SCHEMA_ID",
    "build_strict_evidence_pilot_tranche_completion_gate",
    "render_strict_evidence_pilot_tranche_completion_gate_markdown",
    "write_strict_evidence_pilot_tranche_completion_gate_reports",
]
