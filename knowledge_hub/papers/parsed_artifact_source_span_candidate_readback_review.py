"""Readback review for parsed-artifact SourceSpan candidate-store records.

This helper reads local candidate-store JSONL files, validates candidate
records, and classifies them for a later promotion gate. It does not promote
SourceSpan records and does not create strict, citation-grade, or runtime
evidence.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_candidate_store_contract import (
    PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_RECORD_SCHEMA_ID,
    PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE,
)


PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_READBACK_REVIEW_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-source-span-candidate-readback-review.v1"
)

READBACK_STATUS_PROMOTION_REVIEW_READY = "promotion_review_ready_candidate_only"
READBACK_STATUS_BLOCKED_SCHEMA_VIOLATION = "blocked_schema_violation"
READBACK_STATUS_BLOCKED_MISSING_SOURCE_HASH = "blocked_missing_source_hash"
READBACK_STATUS_BLOCKED_MISSING_LOCATOR = "blocked_missing_locator"
READBACK_STATUS_BLOCKED_DUPLICATE_IDEMPOTENCY_KEY = "blocked_duplicate_idempotency_key"
READBACK_STATUS_BLOCKED_RUNTIME_OR_STRICT_FLAG = "blocked_runtime_or_strict_flag_violation"
READBACK_STATUS_BLOCKED_UNSUPPORTED_TARGET = "blocked_unsupported_candidate_target"

DEFAULT_PAPERS_DIR = Path.home() / ".khub" / "papers"
DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-source-span-candidate-readback-review"
    / "01-parsed-artifact-source-span-candidate-readback-review"
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


def _safe_list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    return value


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    payload_path = Path(str(path)).expanduser()
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _candidate_store_root(papers_dir: str | Path) -> Path:
    return Path(str(papers_dir)).expanduser() / "structured_evidence_candidates" / "source_span"


def _run_manifest_path(papers_dir: str | Path, run_id: str) -> Path:
    return (
        Path(str(papers_dir)).expanduser()
        / "structured_evidence_candidates"
        / "runs"
        / f"{run_id}.json"
    )


def _iter_candidate_records(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            rows.append(
                {
                    "record_path": str(path),
                    "record_line": line_number,
                    "record": payload,
                }
            )
    return rows


def _has_locator(record: dict[str, Any]) -> bool:
    locator = record.get("locator")
    if not isinstance(locator, dict):
        return False
    page = _safe_int(locator.get("page"))
    bbox = _safe_list(locator.get("bbox"))
    block_indexes = _safe_list(locator.get("blockIndexes"))
    chars = locator.get("chars")
    chars_start = _safe_int(chars.get("start")) if isinstance(chars, dict) else None
    chars_end = _safe_int(chars.get("end")) if isinstance(chars, dict) else None
    return page is not None or bool(bbox) or bool(block_indexes) or (
        chars_start is not None and chars_end is not None
    )


def _runtime_or_strict_flag_violation(record: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if _safe_bool(record.get("strictEligible")):
        violations.append("strictEligible_true")
    if _safe_bool(record.get("citationGrade")):
        violations.append("citationGrade_true")
    if _safe_bool(record.get("runtimeEvidence")):
        violations.append("runtimeEvidence_true")
    write_policy = record.get("writePolicy") if isinstance(record.get("writePolicy"), dict) else {}
    for field_name in (
        "databaseMutation",
        "parserRoutingChanged",
        "answerIntegrationChanged",
        "reindexOrReembed",
        "canonicalParsedArtifactsWritten",
    ):
        if _safe_bool(write_policy.get(field_name)):
            violations.append(f"writePolicy.{field_name}_true")
    return violations


def _classify_record(
    record: dict[str, Any],
    *,
    duplicate_key: bool,
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    if _safe_text(record.get("plannedWriteTarget")) != PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE:
        return READBACK_STATUS_BLOCKED_UNSUPPORTED_TARGET, [
            f"plannedWriteTarget={_safe_text(record.get('plannedWriteTarget')) or 'unknown'}"
        ]

    if not _safe_text(record.get("sourceContentHash")):
        blockers.append("sourceContentHash_missing")
        return READBACK_STATUS_BLOCKED_MISSING_SOURCE_HASH, blockers

    if not _has_locator(record):
        blockers.append("locator_missing_page_bbox_blockIndexes_or_chars")
        return READBACK_STATUS_BLOCKED_MISSING_LOCATOR, blockers

    flag_violations = _runtime_or_strict_flag_violation(record)
    if flag_violations:
        return READBACK_STATUS_BLOCKED_RUNTIME_OR_STRICT_FLAG, flag_violations

    validation = validate_payload(
        record,
        PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_RECORD_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        return READBACK_STATUS_BLOCKED_SCHEMA_VIOLATION, [str(error) for error in validation.errors]

    if duplicate_key:
        return READBACK_STATUS_BLOCKED_DUPLICATE_IDEMPOTENCY_KEY, [
            f"duplicate_idempotencyKey={_safe_text(record.get('idempotencyKey'))}"
        ]

    return READBACK_STATUS_PROMOTION_REVIEW_READY, []


def _review_rows(raw_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    idempotency_counts = Counter(
        _safe_text(row.get("record", {}).get("idempotencyKey"))
        for row in raw_rows
        if _safe_text(row.get("record", {}).get("idempotencyKey"))
    )
    rows: list[dict[str, Any]] = []
    schema_violations: list[str] = []

    for index, raw_row in enumerate(raw_rows):
        record = dict(raw_row.get("record") or {})
        key = _safe_text(record.get("idempotencyKey"))
        status, blockers = _classify_record(
            record,
            duplicate_key=bool(key and idempotency_counts.get(key, 0) > 1),
        )
        if status == READBACK_STATUS_BLOCKED_SCHEMA_VIOLATION:
            schema_violations.extend(
                f"candidate_record_schema_violation:{record.get('candidateRecordId') or index}:{error}"
                for error in blockers
            )

        rows.append(
            {
                "review_row_id": f"parsed-artifact-source-span-candidate-readback-review:{index:04d}",
                "candidateRecordId": _safe_text(record.get("candidateRecordId")),
                "runId": _safe_text(record.get("runId")),
                "paper_id": _safe_text(record.get("paperId")),
                "artifact_type": _safe_text(record.get("artifactType")),
                "source_candidate_id": _safe_text(record.get("sourceCandidateId")),
                "source_readiness_row_id": _safe_text(record.get("sourceReadinessRowId")),
                "sourceContentHash": _safe_text(record.get("sourceContentHash")),
                "source_file": _safe_text(record.get("sourceFile")),
                "locator": record.get("locator") if isinstance(record.get("locator"), dict) else {},
                "idempotencyKey": key,
                "candidate_store_path": _safe_text(raw_row.get("record_path")),
                "candidate_store_line": _safe_int(raw_row.get("record_line")) or 0,
                "readback_status": status,
                "review_blockers": _dedupe(blockers),
                "promotion_review_ready": status == READBACK_STATUS_PROMOTION_REVIEW_READY,
                "sourceSpanCreated": False,
                "strictEligible": False,
                "citationGrade": False,
                "runtimeEvidence": False,
                "parserRoutingChanged": False,
                "answerIntegrationChanged": False,
                "databaseMutation": False,
                "canonicalParsedArtifactsWritten": False,
                "recommended_action": (
                    "queue_for_explicit_source_span_promotion_policy_gate"
                    if status == READBACK_STATUS_PROMOTION_REVIEW_READY
                    else "repair_candidate_store_record_before_promotion"
                ),
            }
        )

    return rows, _dedupe(schema_violations)


def _count_rows(
    *,
    rows: list[dict[str, Any]],
    schema_violations: list[str],
) -> dict[str, Any]:
    return {
        "inputRows": len(rows),
        "candidateRecordRows": len(rows),
        "promotionReviewReadyCandidateOnlyRows": sum(
            1 for row in rows if row.get("readback_status") == READBACK_STATUS_PROMOTION_REVIEW_READY
        ),
        "blockedSchemaViolationRows": sum(
            1 for row in rows if row.get("readback_status") == READBACK_STATUS_BLOCKED_SCHEMA_VIOLATION
        ),
        "blockedMissingSourceHashRows": sum(
            1 for row in rows if row.get("readback_status") == READBACK_STATUS_BLOCKED_MISSING_SOURCE_HASH
        ),
        "blockedMissingLocatorRows": sum(
            1 for row in rows if row.get("readback_status") == READBACK_STATUS_BLOCKED_MISSING_LOCATOR
        ),
        "blockedDuplicateIdempotencyKeyRows": sum(
            1 for row in rows if row.get("readback_status") == READBACK_STATUS_BLOCKED_DUPLICATE_IDEMPOTENCY_KEY
        ),
        "blockedRuntimeOrStrictFlagRows": sum(
            1 for row in rows if row.get("readback_status") == READBACK_STATUS_BLOCKED_RUNTIME_OR_STRICT_FLAG
        ),
        "blockedUnsupportedTargetRows": sum(
            1 for row in rows if row.get("readback_status") == READBACK_STATUS_BLOCKED_UNSUPPORTED_TARGET
        ),
        "sourceSpanCreatedRows": 0,
        "strictEvidenceCreatedRows": 0,
        "citationGradeEvidenceCreatedRows": 0,
        "runtimeEvidenceCreatedRows": 0,
        "parserRoutingChangedRows": 0,
        "answerIntegrationChangedRows": 0,
        "databaseMutationRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
        "schemaViolationCount": len(schema_violations),
        "byPaperId": dict(Counter(str(row.get("paper_id") or "") for row in rows)),
        "byArtifactType": dict(Counter(str(row.get("artifact_type") or "") for row in rows)),
        "byReadbackStatus": dict(Counter(str(row.get("readback_status") or "") for row in rows)),
        "byRecommendedAction": dict(Counter(str(row.get("recommended_action") or "") for row in rows)),
    }


def build_parsed_artifact_source_span_candidate_readback_review(
    *,
    papers_dir: str | Path = DEFAULT_PAPERS_DIR,
    run_id: str | None = None,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    papers_root = Path(str(papers_dir)).expanduser()
    candidate_root = _candidate_store_root(papers_root)
    requested_papers = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}
    requested_run_id = _safe_text(run_id)
    warnings: list[str] = []
    schema_violations: list[str] = []

    if not candidate_root.exists():
        warnings.append("candidate_store_root_missing")

    raw_rows = _iter_candidate_records(candidate_root)
    if requested_run_id:
        manifest_path = _run_manifest_path(papers_root, requested_run_id)
        if not manifest_path.exists():
            warnings.append("run_manifest_missing")
        raw_rows = [
            row for row in raw_rows if _safe_text(row.get("record", {}).get("runId")) == requested_run_id
        ]

    if requested_papers:
        found_papers = {
            _safe_text(row.get("record", {}).get("paperId"))
            for row in raw_rows
            if _safe_text(row.get("record", {}).get("paperId"))
        }
        if requested_papers - found_papers:
            warnings.append("requested_paper_ids_not_found")
        raw_rows = [
            row for row in raw_rows if _safe_text(row.get("record", {}).get("paperId")) in requested_papers
        ]

    if not raw_rows:
        warnings.append("candidate_records_missing")

    rows, row_schema_violations = _review_rows(raw_rows)
    schema_violations.extend(row_schema_violations)
    schema_violations = _dedupe(schema_violations)
    counts = _count_rows(rows=rows, schema_violations=schema_violations)

    status = "ok"
    ready_rows = int(counts.get("promotionReviewReadyCandidateOnlyRows") or 0)
    if schema_violations or not rows or ready_rows != len(rows):
        status = "blocked"

    return {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_READBACK_REVIEW_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "papersDir": str(papers_root),
            "candidateStoreRoot": str(candidate_root),
            "requestedRunId": requested_run_id,
            "requestedPaperIds": sorted(requested_papers),
        },
        "counts": counts,
        "gate": {
            "readyForPromotionPolicyGate": bool(ready_rows) and ready_rows == len(rows) and not schema_violations,
            "sourceSpanCreated": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": schema_violations,
            "decision": (
                "parsed_artifact_source_span_candidate_readback_review_ready"
                if status == "ok"
                else "blocked"
            ),
            "recommendedNextTranche": "parsed_artifact_source_span_candidate_promotion_policy_gate",
        },
        "policy": {
            "reportOnly": True,
            "readbackOnly": True,
            "candidateStoreWrite": False,
            "sourceSpanCreated": False,
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


def render_parsed_artifact_source_span_candidate_readback_review_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byReadbackStatus") or {})).items())
    ]
    return "\n".join(
        [
            "# Parsed Artifact SourceSpan Candidate Readback Review",
            "",
            f"- status: {report.get('status', '')}",
            f"- report-only: {json.dumps(report.get('policy', {}).get('reportOnly'))}",
            f"- readback-only: {json.dumps(report.get('policy', {}).get('readbackOnly'))}",
            f"- candidate records: {int(counts.get('candidateRecordRows') or 0)}",
            f"- promotion-review-ready candidate-only rows: {int(counts.get('promotionReviewReadyCandidateOnlyRows') or 0)}",
            f"- schema violations: {int(counts.get('schemaViolationCount') or 0)}",
            f"- source spans created: {int(counts.get('sourceSpanCreatedRows') or 0)}",
            f"- strict evidence created: {int(counts.get('strictEvidenceCreatedRows') or 0)}",
            f"- database mutations: {int(counts.get('databaseMutationRows') or 0)}",
            "",
            "## Readback status breakdown",
            *[f"- {item}" for item in by_status],
        ]
    )


def write_parsed_artifact_source_span_candidate_readback_review_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-source-span-candidate-readback-review.json"
    summary_path = root / "parsed-artifact-source-span-candidate-readback-review-summary.json"
    markdown_path = root / "parsed-artifact-source-span-candidate-readback-review.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_parsed_artifact_source_span_candidate_readback_review_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Review parsed-artifact SourceSpan candidate-store JSONL records "
            "before any SourceSpan promotion gate."
        )
    )
    parser.add_argument("--papers-dir", default=str(DEFAULT_PAPERS_DIR), help="Local papers_dir root.")
    parser.add_argument("--run-id", default="", help="Filter records by run id.")
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_parsed_artifact_source_span_candidate_readback_review(
        papers_dir=args.papers_dir,
        run_id=args.run_id or None,
        paper_ids=args.paper_id or None,
    )

    if args.output_dir:
        paths = write_parsed_artifact_source_span_candidate_readback_review_reports(
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
    "PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_READBACK_REVIEW_SCHEMA_ID",
    "READBACK_STATUS_PROMOTION_REVIEW_READY",
    "build_parsed_artifact_source_span_candidate_readback_review",
    "render_parsed_artifact_source_span_candidate_readback_review_markdown",
    "write_parsed_artifact_source_span_candidate_readback_review_reports",
]
