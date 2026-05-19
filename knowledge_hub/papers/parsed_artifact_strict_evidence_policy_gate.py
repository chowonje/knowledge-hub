"""Policy gate for parsed-artifact StrictEvidence records after promotion readback.

Consumes the StrictEvidence promotion readback-review report and classifies
readback-validated rows into next-tier policy states without mutating records,
creating citation/runtime evidence, or changing integration surfaces.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_strict_evidence_promotion_readback_review import (
    PARSED_ARTIFACT_STRICT_EVIDENCE_PROMOTION_READBACK_REVIEW_SCHEMA_ID,
    READBACK_STATUS_VALIDATED,
)
from knowledge_hub.papers.parsed_artifact_strict_evidence_record_contract import (
    CHARS_BASIS,
    CHARS_NORMALIZATION_LABEL,
)


PARSED_ARTIFACT_STRICT_EVIDENCE_POLICY_GATE_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-strict-evidence-policy-gate.v1"
)

POLICY_STATUS_CANDIDATE_ONLY = "strict_evidence_policy_candidate_only"
POLICY_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"
POLICY_STATUS_BLOCKED_READBACK_NOT_VALIDATED = "blocked_readback_not_validated"
POLICY_STATUS_BLOCKED_MISSING_VERBATIM_HASH = "blocked_missing_verbatim_hash"
POLICY_STATUS_BLOCKED_MISSING_AUTHORITY_CHARS = "blocked_missing_authority_chars"
POLICY_STATUS_BLOCKED_INVALID_AUTHORITY_BASIS = "blocked_invalid_authority_basis"
POLICY_STATUS_BLOCKED_UNSUPPORTED_NORMALIZATION = "blocked_unsupported_normalization"
POLICY_STATUS_BLOCKED_RUNTIME_OR_CITATION = "blocked_runtime_or_citation_flag_violation"
POLICY_STATUS_BLOCKED_MISSING_SOURCE_SPAN = "blocked_missing_source_span_reference"

DEFAULT_READBACK_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-strict-evidence-promotion-readback-review"
    / "01-parsed-artifact-strict-evidence-promotion-readback-review"
    / "parsed-artifact-strict-evidence-promotion-readback-review.json"
)

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-strict-evidence-policy-gate"
    / "01-parsed-artifact-strict-evidence-policy-gate"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_bool(value: Any) -> bool:
    return bool(value)


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
    readback_row: dict[str, Any],
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
        "sourceSpanUpdatedRows",
    ):
        if _safe_bool(readback_row.get(field_name)):
            violations.append(f"readback_row.{field_name}_true")
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


def _classify_policy_row(
    readback_row: dict[str, Any],
    *,
    strict_evidence_record: dict[str, Any],
) -> tuple[str, list[str]]:
    blockers: list[str] = []

    if _safe_text(readback_row.get("readback_status")) != READBACK_STATUS_VALIDATED:
        blockers.extend(_safe_text(item) for item in (readback_row.get("review_blockers") or []))
        blockers.append(f"readback_status={_safe_text(readback_row.get('readback_status')) or 'unknown'}")
        if not _safe_bool(readback_row.get("readback_validated")):
            blockers.append("readback_validated_false")
        return POLICY_STATUS_BLOCKED_READBACK_NOT_VALIDATED, _dedupe(blockers)

    if not _safe_bool(readback_row.get("source_span_reference_found")):
        return POLICY_STATUS_BLOCKED_MISSING_SOURCE_SPAN, ["source_span_reference_not_found_in_readback"]

    if not _safe_bool(readback_row.get("source_span_reference_hash_match")):
        return POLICY_STATUS_BLOCKED_MISSING_SOURCE_SPAN, ["source_span_reference_hash_mismatch_in_readback"]

    for field_name in ("strictEvidenceId", "sourceSpanId", "candidateRecordId", "sourceContentHash"):
        if not _safe_text(readback_row.get(field_name)):
            blockers.append(f"{field_name}_missing")
    if blockers:
        return POLICY_STATUS_BLOCKED_MISSING_AUTHORITY_CHARS, _dedupe(blockers)

    verbatim_hash = _safe_text(strict_evidence_record.get("verbatimSubstringSha256"))
    if not verbatim_hash:
        return POLICY_STATUS_BLOCKED_MISSING_VERBATIM_HASH, ["verbatimSubstringSha256_missing"]

    authority = (
        strict_evidence_record.get("authority")
        if isinstance(strict_evidence_record.get("authority"), dict)
        else {}
    )
    chars = authority.get("chars") if isinstance(authority.get("chars"), dict) else {}
    start = chars.get("start")
    end = chars.get("end")
    if start is None or end is None:
        return POLICY_STATUS_BLOCKED_MISSING_AUTHORITY_CHARS, ["authority_chars_start_or_end_missing"]

    basis = _safe_text(chars.get("basis"))
    if basis != CHARS_BASIS:
        return POLICY_STATUS_BLOCKED_INVALID_AUTHORITY_BASIS, [
            f"authority_chars_basis_invalid:{basis or 'missing'}"
        ]

    normalization = _safe_text(chars.get("normalization"))
    if normalization != CHARS_NORMALIZATION_LABEL:
        return POLICY_STATUS_BLOCKED_UNSUPPORTED_NORMALIZATION, [
            f"authority_chars_normalization_unsupported:{normalization or 'missing'}"
        ]

    expected_hash = _safe_text(chars.get("expectedSubstringSha256"))
    if not expected_hash:
        return POLICY_STATUS_BLOCKED_MISSING_AUTHORITY_CHARS, ["authority_chars_expectedSubstringSha256_missing"]

    if expected_hash != verbatim_hash:
        return POLICY_STATUS_BLOCKED_MISSING_VERBATIM_HASH, [
            "verbatimSubstringSha256_must_equal_authority_chars_expectedSubstringSha256"
        ]

    flag_violations = _runtime_or_integration_violation(
        readback_row=readback_row,
        strict_evidence_record=strict_evidence_record,
    )
    if flag_violations:
        return POLICY_STATUS_BLOCKED_RUNTIME_OR_CITATION, flag_violations

    return POLICY_STATUS_CANDIDATE_ONLY, []


def _policy_rows(readback_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, readback_row in enumerate(readback_rows):
        source_row = dict(readback_row or {})
        strict_evidence_record = _load_strict_evidence_record_at_store_ref(
            _safe_text(source_row.get("strict_evidence_store_path")),
            _safe_int(source_row.get("strict_evidence_store_line")) or 0,
        )
        status, blockers = _classify_policy_row(source_row, strict_evidence_record=strict_evidence_record)
        ready = status == POLICY_STATUS_CANDIDATE_ONLY
        authority = (
            strict_evidence_record.get("authority")
            if isinstance(strict_evidence_record.get("authority"), dict)
            else {}
        )
        chars = authority.get("chars") if isinstance(authority.get("chars"), dict) else {}
        rows.append(
            {
                "policy_gate_row_id": f"parsed-artifact-strict-evidence-policy-gate:{index:04d}",
                "readback_review_row_id": _safe_text(source_row.get("review_row_id")),
                "strictEvidenceId": _safe_text(
                    source_row.get("strictEvidenceId") or strict_evidence_record.get("strictEvidenceId")
                ),
                "sourceSpanId": _safe_text(source_row.get("sourceSpanId")),
                "candidateRecordId": _safe_text(source_row.get("candidateRecordId")),
                "runId": _safe_text(source_row.get("runId")),
                "paper_id": _safe_text(source_row.get("paper_id")),
                "artifact_type": _safe_text(source_row.get("artifact_type")),
                "sourceContentHash": _safe_text(source_row.get("sourceContentHash")),
                "idempotencyKey": _safe_text(source_row.get("idempotencyKey")),
                "strict_evidence_store_path": _safe_text(source_row.get("strict_evidence_store_path")),
                "strict_evidence_store_line": _safe_int(source_row.get("strict_evidence_store_line")) or 0,
                "readback_status": _safe_text(source_row.get("readback_status")),
                "readback_validated": _safe_bool(source_row.get("readback_validated")),
                "source_span_reference_found": _safe_bool(source_row.get("source_span_reference_found")),
                "source_span_reference_hash_match": _safe_bool(
                    source_row.get("source_span_reference_hash_match")
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
                "policy_gate_status": status,
                "policy_blockers": _dedupe(blockers),
                "strictEvidencePolicyCandidateOnly": ready,
                "strictEligible": False,
                "strictEvidenceCreated": False,
                "citationGrade": False,
                "runtimeEvidence": False,
                "parserRoutingChanged": False,
                "answerIntegrationChanged": False,
                "databaseMutation": False,
                "canonicalParsedArtifactsWritten": False,
                "sourceSpanUpdatedRows": 0,
                "recommended_action": (
                    "queue_for_explicit_strict_evidence_promotion_tranche"
                    if ready
                    else "repair_strict_evidence_or_readback_before_policy_gate"
                ),
            }
        )
    return rows


def _count_rows(
    *,
    rows: list[dict[str, Any]],
    input_schema_violations: list[str],
    readback_validated_rows: int,
) -> dict[str, Any]:
    return {
        "inputRows": len(rows),
        "readbackValidatedRows": readback_validated_rows,
        "strictEvidencePolicyCandidateOnlyRows": sum(
            1 for row in rows if row.get("policy_gate_status") == POLICY_STATUS_CANDIDATE_ONLY
        ),
        "blockedInputReportSchemaViolationRows": (
            len(rows) if input_schema_violations and rows else int(bool(input_schema_violations))
        ),
        "blockedReadbackNotValidatedRows": sum(
            1
            for row in rows
            if row.get("policy_gate_status") == POLICY_STATUS_BLOCKED_READBACK_NOT_VALIDATED
        ),
        "blockedMissingVerbatimHashRows": sum(
            1 for row in rows if row.get("policy_gate_status") == POLICY_STATUS_BLOCKED_MISSING_VERBATIM_HASH
        ),
        "blockedMissingAuthorityCharsRows": sum(
            1 for row in rows if row.get("policy_gate_status") == POLICY_STATUS_BLOCKED_MISSING_AUTHORITY_CHARS
        ),
        "blockedInvalidAuthorityBasisRows": sum(
            1 for row in rows if row.get("policy_gate_status") == POLICY_STATUS_BLOCKED_INVALID_AUTHORITY_BASIS
        ),
        "blockedUnsupportedNormalizationRows": sum(
            1 for row in rows if row.get("policy_gate_status") == POLICY_STATUS_BLOCKED_UNSUPPORTED_NORMALIZATION
        ),
        "blockedRuntimeOrCitationFlagViolationRows": sum(
            1 for row in rows if row.get("policy_gate_status") == POLICY_STATUS_BLOCKED_RUNTIME_OR_CITATION
        ),
        "blockedMissingSourceSpanReferenceRows": sum(
            1 for row in rows if row.get("policy_gate_status") == POLICY_STATUS_BLOCKED_MISSING_SOURCE_SPAN
        ),
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
        "byPolicyGateStatus": dict(Counter(str(row.get("policy_gate_status") or "") for row in rows)),
        "byRecommendedAction": dict(Counter(str(row.get("recommended_action") or "") for row in rows)),
    }


def build_parsed_artifact_strict_evidence_policy_gate(
    *,
    readback_report_path: str | Path = DEFAULT_READBACK_REPORT_PATH,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    report_path = Path(str(readback_report_path)).expanduser()
    warnings: list[str] = []
    input_schema_violations: list[str] = []
    requested_papers = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}

    readback_report = _read_json(report_path)
    if not readback_report:
        warnings.append("readback_report_missing_or_unreadable")

    validation = validate_payload(
        readback_report,
        PARSED_ARTIFACT_STRICT_EVIDENCE_PROMOTION_READBACK_REVIEW_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        input_schema_violations = [str(error) for error in validation.errors]
        if not readback_report:
            input_schema_violations.append("readback_report_missing_or_unreadable")

    if readback_report and _safe_text(readback_report.get("status")) != "ok":
        input_schema_violations.append(
            f"readback_report_status={_safe_text(readback_report.get('status')) or 'unknown'}"
        )

    readback_rows = [
        row
        for row in readback_report.get("rows", [])
        if isinstance(row, dict) and _safe_bool(row.get("readback_validated"))
    ] if isinstance(readback_report, dict) else []

    readback_validated_count = int(
        (readback_report.get("counts") or {}).get("readbackValidatedRows") or len(readback_rows)
    ) if readback_report else 0

    if requested_papers:
        found_papers = {
            _safe_text(row.get("paper_id")) for row in readback_rows if _safe_text(row.get("paper_id"))
        }
        if requested_papers - found_papers:
            warnings.append("requested_paper_ids_not_found")
        readback_rows = [
            row for row in readback_rows if _safe_text(row.get("paper_id")) in requested_papers
        ]

    if not readback_rows and not input_schema_violations:
        warnings.append("readback_validated_rows_missing")

    rows = _policy_rows(readback_rows)
    if input_schema_violations:
        for row in rows:
            row["policy_gate_status"] = POLICY_STATUS_BLOCKED_INPUT_SCHEMA
            row["policy_blockers"] = _dedupe([*row.get("policy_blockers", []), *input_schema_violations])
            row["strictEvidencePolicyCandidateOnly"] = False
            row["recommended_action"] = "repair_readback_report_schema_before_policy_gate"

    counts = _count_rows(
        rows=rows,
        input_schema_violations=_dedupe(input_schema_violations),
        readback_validated_rows=readback_validated_count,
    )
    candidate_rows = int(counts.get("strictEvidencePolicyCandidateOnlyRows") or 0)
    status = "ok"
    if input_schema_violations or not rows or candidate_rows != len(rows):
        status = "blocked"

    return {
        "schema": PARSED_ARTIFACT_STRICT_EVIDENCE_POLICY_GATE_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "readbackReportPath": str(report_path),
            "readbackSchema": _safe_text(readback_report.get("schema")) if readback_report else "",
            "readbackReportStatus": _safe_text(readback_report.get("status")) if readback_report else "",
            "requestedPaperIds": sorted(requested_papers),
        },
        "counts": counts,
        "gate": {
            "strictEvidencePolicyGateReady": (
                bool(candidate_rows) and candidate_rows == len(rows) and not input_schema_violations
            ),
            "strictEligibleMutationAllowed": False,
            "strictEvidenceCreated": False,
            "citationReady": False,
            "runtimeEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": _dedupe(input_schema_violations),
            "decision": (
                "parsed_artifact_strict_evidence_policy_gate_ready"
                if status == "ok"
                else "blocked"
            ),
            "recommendedNextTranche": (
                "parsed_artifact_strict_evidence_promotion_tranche_plan"
                if status == "ok"
                else "parsed_artifact_strict_evidence_promotion_readback_repair"
            ),
        },
        "policy": {
            "reportOnly": True,
            "policyGateOnly": True,
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
            "gate",
            "policy",
            "warnings",
            "rows",
        )
        if key in report
    }


def render_parsed_artifact_strict_evidence_policy_gate_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byPolicyGateStatus") or {})).items())
    ]
    return "\n".join(
        [
            "# Parsed Artifact StrictEvidence Policy Gate",
            "",
            f"- status: {report.get('status', '')}",
            f"- report-only: {json.dumps(report.get('policy', {}).get('reportOnly'))}",
            f"- input rows: {int(counts.get('inputRows') or 0)}",
            f"- readback validated rows: {int(counts.get('readbackValidatedRows') or 0)}",
            f"- policy candidate-only rows: {int(counts.get('strictEvidencePolicyCandidateOnlyRows') or 0)}",
            f"- strict evidence created: {int(counts.get('strictEvidenceCreatedRows') or 0)}",
            f"- citation-grade evidence created: {int(counts.get('citationGradeEvidenceCreatedRows') or 0)}",
            f"- runtime evidence created: {int(counts.get('runtimeEvidenceCreatedRows') or 0)}",
            f"- database mutations: {int(counts.get('databaseMutationRows') or 0)}",
            "",
            "## Policy gate status breakdown",
            *[f"- {item}" for item in by_status],
        ]
    )


def write_parsed_artifact_strict_evidence_policy_gate_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-strict-evidence-policy-gate.json"
    summary_path = root / "parsed-artifact-strict-evidence-policy-gate-summary.json"
    markdown_path = root / "parsed-artifact-strict-evidence-policy-gate.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_parsed_artifact_strict_evidence_policy_gate_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Evaluate StrictEvidence promotion readback rows against the strict-evidence "
            "policy gate without mutating records or integration surfaces."
        )
    )
    parser.add_argument(
        "--readback-report",
        default=str(DEFAULT_READBACK_REPORT_PATH),
        help="StrictEvidence promotion readback-review JSON report.",
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_parsed_artifact_strict_evidence_policy_gate(
        readback_report_path=args.readback_report,
        paper_ids=args.paper_id or None,
    )

    if args.output_dir:
        paths = write_parsed_artifact_strict_evidence_policy_gate_reports(report, args.output_dir)
        print(f"wrote report: {paths['report']}")
        print(f"wrote summary: {paths['summary']}")
        print(f"wrote markdown: {paths['markdown']}")

    if args.json or not args.output_dir:
        print(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PARSED_ARTIFACT_STRICT_EVIDENCE_POLICY_GATE_SCHEMA_ID",
    "POLICY_STATUS_CANDIDATE_ONLY",
    "build_parsed_artifact_strict_evidence_policy_gate",
    "render_parsed_artifact_strict_evidence_policy_gate_markdown",
    "write_parsed_artifact_strict_evidence_policy_gate_reports",
]
