"""Readback review for applied parsed-artifact StrictEvidence store records.

Reads local StrictEvidence JSONL files written by the executor apply tranche,
validates records against the StrictEvidence record schema, cross-checks SourceSpan
references, and confirms records remain non-citation-grade and non-runtime.
Does not write StrictEvidence or SourceSpan JSONL or mutate runtime surfaces.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_strict_evidence_executor_apply import (
    PARSED_ARTIFACT_STRICT_EVIDENCE_EXECUTOR_APPLY_SCHEMA_ID,
)
from knowledge_hub.papers.parsed_artifact_strict_evidence_record_contract import (
    CHARS_BASIS,
    CHARS_NORMALIZATION_LABEL,
    PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_SCHEMA_ID,
    PARSED_ARTIFACT_STRICT_EVIDENCE_STORE,
    validate_strict_evidence_record_semantics,
)


PARSED_ARTIFACT_STRICT_EVIDENCE_PROMOTION_READBACK_REVIEW_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-strict-evidence-promotion-readback-review.v1"
)

READBACK_STATUS_VALIDATED = "readback_validated"
READBACK_STATUS_BLOCKED_SCHEMA = "blocked_record_schema_violation"
READBACK_STATUS_BLOCKED_SEMANTIC = "blocked_record_semantic_violation"
READBACK_STATUS_BLOCKED_MISSING_SOURCE_SPAN = "blocked_missing_source_span_reference"
READBACK_STATUS_BLOCKED_SOURCE_HASH_MISMATCH = "blocked_source_hash_mismatch"
READBACK_STATUS_BLOCKED_IDEMPOTENCY_DUPLICATE = "blocked_idempotency_duplicate"
READBACK_STATUS_BLOCKED_RUNTIME_OR_CITATION = "blocked_runtime_or_citation_flag_violation"
READBACK_STATUS_BLOCKED_INPUT_REPORT = "blocked_input_report_schema_violation"

DEFAULT_APPLY_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-strict-evidence-executor-apply"
    / "02-parsed-artifact-strict-evidence-executor-apply"
    / "parsed-artifact-strict-evidence-executor-apply.json"
)

DEFAULT_RUN_MANIFEST_PATH = (
    Path.home()
    / ".khub"
    / "papers"
    / "structured_evidence"
    / "runs"
    / "parsed-artifact-strict-evidence-executor-apply-20260519.json"
)

DEFAULT_PAPERS_DIR = Path.home() / ".khub" / "papers"

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-strict-evidence-promotion-readback-review"
    / "01-parsed-artifact-strict-evidence-promotion-readback-review"
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


def _strict_evidence_store_root(papers_dir: str | Path) -> Path:
    return Path(str(papers_dir)).expanduser() / "structured_evidence" / "strict_evidence"


def _source_span_store_root(papers_dir: str | Path) -> Path:
    return Path(str(papers_dir)).expanduser() / "structured_evidence" / "source_span"


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    payload_path = Path(str(path)).expanduser()
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _primary_source_span_id(record: dict[str, Any]) -> str:
    source_span_ids = _safe_list(record.get("sourceSpanIds"))
    if source_span_ids:
        return _safe_text(source_span_ids[0])
    return _safe_text(record.get("sourceSpanId"))


def _primary_candidate_record_id(record: dict[str, Any]) -> str:
    candidate_ids = _safe_list(record.get("candidateRecordIds"))
    if candidate_ids:
        return _safe_text(candidate_ids[0])
    return _safe_text(record.get("candidateRecordId"))


def _collect_manifest_run_id_candidates(manifest: dict[str, Any], *, manifest_path: str) -> list[str]:
    candidates: list[str] = []

    def add(value: Any) -> None:
        text = _safe_text(value)
        if text:
            candidates.append(text)

    add(manifest.get("runId"))
    input_section = manifest.get("input")
    if isinstance(input_section, dict):
        add(input_section.get("runId"))

    for record in _safe_list(manifest.get("strictEvidenceRecords")):
        if isinstance(record, dict):
            add(record.get("runId"))

    for row in _safe_list(manifest.get("rows")):
        if not isinstance(row, dict):
            continue
        add(row.get("runId"))
        record = row.get("record")
        if isinstance(record, dict):
            add(record.get("runId"))

    if manifest_path:
        add(Path(manifest_path).stem)

    return _dedupe(candidates)


def _manifest_record_match_keys(manifest: dict[str, Any]) -> tuple[set[str], set[str]]:
    strict_evidence_ids: set[str] = set()
    idempotency_keys: set[str] = set()

    def absorb(record: dict[str, Any]) -> None:
        strict_evidence_id = _safe_text(record.get("strictEvidenceId"))
        idempotency_key = _safe_text(record.get("idempotencyKey"))
        if strict_evidence_id:
            strict_evidence_ids.add(strict_evidence_id)
        if idempotency_key:
            idempotency_keys.add(idempotency_key)

    for record in _safe_list(manifest.get("strictEvidenceRecords")):
        if isinstance(record, dict):
            absorb(record)

    for row in _safe_list(manifest.get("rows")):
        if not isinstance(row, dict):
            continue
        record = row.get("record")
        if isinstance(record, dict):
            absorb(record)
        absorb(row)

    return strict_evidence_ids, idempotency_keys


def _observed_record_run_ids(raw_rows: list[dict[str, Any]]) -> list[str]:
    return _dedupe(
        _safe_text(row.get("record", {}).get("runId"))
        for row in raw_rows
        if _safe_text(row.get("record", {}).get("runId"))
    )


def _resolve_record_run_ids_from_manifest(
    *,
    manifest: dict[str, Any],
    manifest_path: str,
    raw_rows: list[dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    manifest_candidate_run_ids = _collect_manifest_run_id_candidates(manifest, manifest_path=manifest_path)
    observed_record_run_ids = _observed_record_run_ids(raw_rows)
    observed_set = set(observed_record_run_ids)

    direct_matches = [run_id for run_id in manifest_candidate_run_ids if run_id in observed_set]
    if direct_matches:
        return _dedupe(direct_matches), {
            "resolution": "manifest_run_id_direct_match",
            "manifestCandidateRunIds": manifest_candidate_run_ids,
            "observedRecordRunIds": observed_record_run_ids,
            "resolvedRecordRunIds": _dedupe(direct_matches),
        }

    strict_evidence_ids, idempotency_keys = _manifest_record_match_keys(manifest)
    inferred_run_ids: list[str] = []
    if strict_evidence_ids or idempotency_keys:
        for raw_row in raw_rows:
            record = raw_row.get("record") if isinstance(raw_row.get("record"), dict) else {}
            strict_evidence_id = _safe_text(record.get("strictEvidenceId"))
            idempotency_key = _safe_text(record.get("idempotencyKey"))
            if strict_evidence_id in strict_evidence_ids or idempotency_key in idempotency_keys:
                run_id = _safe_text(record.get("runId"))
                if run_id:
                    inferred_run_ids.append(run_id)

    inferred_run_ids = _dedupe(inferred_run_ids)
    if inferred_run_ids:
        return inferred_run_ids, {
            "resolution": "manifest_record_metadata_match",
            "manifestCandidateRunIds": manifest_candidate_run_ids,
            "observedRecordRunIds": observed_record_run_ids,
            "resolvedRecordRunIds": inferred_run_ids,
        }

    return [], {
        "resolution": "run_manifest_record_run_id_mismatch",
        "manifestCandidateRunIds": manifest_candidate_run_ids,
        "observedRecordRunIds": observed_record_run_ids,
        "resolvedRecordRunIds": [],
    }


def _iter_strict_evidence_records(root: Path) -> list[dict[str, Any]]:
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


def _load_source_span_index(root: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    if not root.exists():
        return index
    for path in sorted(root.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                payload = {}
            if not isinstance(payload, dict):
                continue
            source_span_id = _safe_text(payload.get("sourceSpanId"))
            if source_span_id:
                index[source_span_id] = payload
    return index


def _runtime_or_citation_violation(record: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if _safe_bool(record.get("citationGrade")):
        violations.append("citationGrade_true")
    if _safe_bool(record.get("runtimeEvidence")):
        violations.append("runtimeEvidence_true")
    if _safe_bool(record.get("strictEligible")):
        violations.append("strictEligible_true")
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


def _semantic_field_violations(record: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if not _safe_text(record.get("strictEvidenceId")):
        violations.append("strictEvidenceId_missing")
    if not _primary_source_span_id(record):
        violations.append("sourceSpanId_missing")
    if not _primary_candidate_record_id(record):
        violations.append("candidateRecordId_missing")
    if not _safe_text(record.get("paperId")):
        violations.append("paperId_missing")
    if not _safe_text(record.get("artifactType")):
        violations.append("artifactType_missing")
    if not _safe_text(record.get("sourceContentHash")):
        violations.append("sourceContentHash_missing")

    authority = record.get("authority") if isinstance(record.get("authority"), dict) else {}
    chars = authority.get("chars") if isinstance(authority.get("chars"), dict) else {}
    start = chars.get("start")
    end = chars.get("end")
    if start is None or end is None:
        violations.append("authority_chars_start_or_end_missing")
    else:
        try:
            if int(end) <= int(start):
                violations.append("authority_chars_end_must_be_greater_than_start")
        except Exception:
            violations.append("authority_chars_start_or_end_invalid")

    if _safe_text(chars.get("basis")) != CHARS_BASIS:
        violations.append(f"authority_chars_basis_invalid:{_safe_text(chars.get('basis')) or 'missing'}")
    if _safe_text(chars.get("normalization")) != CHARS_NORMALIZATION_LABEL:
        violations.append(
            f"authority_chars_normalization_invalid:{_safe_text(chars.get('normalization')) or 'missing'}"
        )

    expected_hash = _safe_text(chars.get("expectedSubstringSha256"))
    verbatim_hash = _safe_text(record.get("verbatimSubstringSha256"))
    if expected_hash and verbatim_hash and expected_hash != verbatim_hash:
        violations.append("verbatimSubstringSha256_must_equal_authority_chars_expectedSubstringSha256")

    violations.extend(validate_strict_evidence_record_semantics(record))
    return _dedupe(violations)


def _source_span_reference_violations(
    record: dict[str, Any],
    *,
    source_span_index: dict[str, dict[str, Any]],
) -> tuple[str, list[str]]:
    source_span_id = _primary_source_span_id(record)
    if not source_span_id:
        return READBACK_STATUS_BLOCKED_MISSING_SOURCE_SPAN, ["sourceSpanId_missing"]

    source_span = source_span_index.get(source_span_id)
    if not source_span:
        return READBACK_STATUS_BLOCKED_MISSING_SOURCE_SPAN, [
            f"source_span_record_missing:{source_span_id}"
        ]

    span_violations: list[str] = []
    if _safe_bool(source_span.get("strictEligible")):
        span_violations.append("source_span_strictEligible_true")
    if _safe_bool(source_span.get("citationGrade")):
        span_violations.append("source_span_citationGrade_true")
    if _safe_bool(source_span.get("runtimeEvidence")):
        span_violations.append("source_span_runtimeEvidence_true")
    if span_violations:
        return READBACK_STATUS_BLOCKED_RUNTIME_OR_CITATION, span_violations

    record_hash = _safe_text(record.get("sourceContentHash"))
    span_hash = _safe_text(source_span.get("sourceContentHash"))
    if record_hash and span_hash and record_hash != span_hash:
        return READBACK_STATUS_BLOCKED_SOURCE_HASH_MISMATCH, [
            f"sourceContentHash_mismatch:strict={record_hash}:source_span={span_hash}"
        ]

    return READBACK_STATUS_VALIDATED, []


def _classify_record(
    record: dict[str, Any],
    *,
    duplicate_key: bool,
    requested_run_id: str,
    source_span_index: dict[str, dict[str, Any]],
) -> tuple[str, list[str]]:
    if _safe_text(record.get("plannedWriteTarget")) != PARSED_ARTIFACT_STRICT_EVIDENCE_STORE:
        return READBACK_STATUS_BLOCKED_SCHEMA, [
            f"plannedWriteTarget={_safe_text(record.get('plannedWriteTarget')) or 'unknown'}"
        ]

    record_run_id = _safe_text(record.get("runId"))
    if requested_run_id and record_run_id != requested_run_id:
        return READBACK_STATUS_BLOCKED_SCHEMA, [
            f"runId_mismatch:expected={requested_run_id}:actual={record_run_id or 'missing'}"
        ]

    flag_violations = _runtime_or_citation_violation(record)
    if flag_violations:
        return READBACK_STATUS_BLOCKED_RUNTIME_OR_CITATION, flag_violations

    validation = validate_payload(
        record,
        PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        return READBACK_STATUS_BLOCKED_SCHEMA, [str(error) for error in validation.errors]

    semantic_violations = _semantic_field_violations(record)
    if semantic_violations:
        return READBACK_STATUS_BLOCKED_SEMANTIC, semantic_violations

    if duplicate_key:
        return READBACK_STATUS_BLOCKED_IDEMPOTENCY_DUPLICATE, [
            f"duplicate_idempotencyKey={_safe_text(record.get('idempotencyKey'))}"
        ]

    ref_status, ref_blockers = _source_span_reference_violations(
        record,
        source_span_index=source_span_index,
    )
    if ref_status != READBACK_STATUS_VALIDATED:
        return ref_status, ref_blockers

    return READBACK_STATUS_VALIDATED, []


def _review_rows(
    raw_rows: list[dict[str, Any]],
    *,
    requested_run_id: str,
    source_span_index: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
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
            requested_run_id=requested_run_id,
            source_span_index=source_span_index,
        )
        if status in {READBACK_STATUS_BLOCKED_SCHEMA, READBACK_STATUS_BLOCKED_SEMANTIC}:
            schema_violations.extend(
                f"strict_evidence_record_violation:{record.get('strictEvidenceId') or index}:{error}"
                for error in blockers
            )

        source_span_id = _primary_source_span_id(record)
        source_span = source_span_index.get(source_span_id, {})
        source_span_hash_match = (
            _safe_text(record.get("sourceContentHash"))
            == _safe_text(source_span.get("sourceContentHash"))
            if source_span
            else False
        )

        rows.append(
            {
                "review_row_id": (
                    f"parsed-artifact-strict-evidence-promotion-readback-review:{index:04d}"
                ),
                "strictEvidenceId": _safe_text(record.get("strictEvidenceId")),
                "sourceSpanId": source_span_id,
                "candidateRecordId": _primary_candidate_record_id(record),
                "runId": _safe_text(record.get("runId")),
                "paper_id": _safe_text(record.get("paperId")),
                "artifact_type": _safe_text(record.get("artifactType")),
                "sourceContentHash": _safe_text(record.get("sourceContentHash")),
                "idempotencyKey": key,
                "strict_evidence_store_path": _safe_text(raw_row.get("record_path")),
                "strict_evidence_store_line": _safe_int(raw_row.get("record_line")) or 0,
                "source_span_reference_found": bool(source_span),
                "source_span_reference_hash_match": source_span_hash_match,
                "readback_status": status,
                "review_blockers": _dedupe(blockers),
                "readback_validated": status == READBACK_STATUS_VALIDATED,
                "strictEvidenceCreated": False,
                "strictEvidenceWriteRows": 0,
                "sourceSpanUpdatedRows": 0,
                "citationGrade": False,
                "runtimeEvidence": False,
                "parserRoutingChanged": False,
                "answerIntegrationChanged": False,
                "databaseMutation": False,
                "canonicalParsedArtifactsWritten": False,
                "recommended_action": (
                    "queue_for_explicit_strict_evidence_policy_gate"
                    if status == READBACK_STATUS_VALIDATED
                    else "repair_strict_evidence_store_record_before_policy_gate"
                ),
            }
        )

    return rows, _dedupe(schema_violations)


def _count_rows(
    *,
    rows: list[dict[str, Any]],
    schema_violations: list[str],
    expected_input_rows: int,
) -> dict[str, Any]:
    return {
        "inputRows": expected_input_rows or len(rows),
        "strictEvidenceRecordRows": len(rows),
        "readbackValidatedRows": sum(
            1 for row in rows if row.get("readback_status") == READBACK_STATUS_VALIDATED
        ),
        "blockedRecordSchemaViolationRows": sum(
            1 for row in rows if row.get("readback_status") == READBACK_STATUS_BLOCKED_SCHEMA
        ),
        "blockedRecordSemanticViolationRows": sum(
            1 for row in rows if row.get("readback_status") == READBACK_STATUS_BLOCKED_SEMANTIC
        ),
        "blockedMissingSourceSpanReferenceRows": sum(
            1 for row in rows if row.get("readback_status") == READBACK_STATUS_BLOCKED_MISSING_SOURCE_SPAN
        ),
        "blockedSourceHashMismatchRows": sum(
            1 for row in rows if row.get("readback_status") == READBACK_STATUS_BLOCKED_SOURCE_HASH_MISMATCH
        ),
        "blockedIdempotencyDuplicateRows": sum(
            1 for row in rows if row.get("readback_status") == READBACK_STATUS_BLOCKED_IDEMPOTENCY_DUPLICATE
        ),
        "blockedRuntimeOrCitationFlagViolationRows": sum(
            1 for row in rows if row.get("readback_status") == READBACK_STATUS_BLOCKED_RUNTIME_OR_CITATION
        ),
        "blockedInputReportSchemaViolationRows": sum(
            1 for row in rows if row.get("readback_status") == READBACK_STATUS_BLOCKED_INPUT_REPORT
        ),
        "strictEvidenceCreatedRows": 0,
        "citationGradeEvidenceCreatedRows": 0,
        "runtimeEvidenceCreatedRows": 0,
        "parserRoutingChangedRows": 0,
        "answerIntegrationChangedRows": 0,
        "databaseMutationRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
        "sourceSpanUpdatedRows": 0,
        "schemaViolationCount": len(schema_violations),
        "byPaperId": dict(Counter(str(row.get("paper_id") or "") for row in rows)),
        "byArtifactType": dict(Counter(str(row.get("artifact_type") or "") for row in rows)),
        "byReadbackStatus": dict(Counter(str(row.get("readback_status") or "") for row in rows)),
        "byRecommendedAction": dict(Counter(str(row.get("recommended_action") or "") for row in rows)),
    }


def build_parsed_artifact_strict_evidence_promotion_readback_review(
    *,
    apply_report_path: str | Path = DEFAULT_APPLY_REPORT_PATH,
    run_manifest_path: str | Path = DEFAULT_RUN_MANIFEST_PATH,
    papers_dir: str | Path = DEFAULT_PAPERS_DIR,
    run_id: str | None = None,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    apply_path = Path(str(apply_report_path)).expanduser()
    manifest_path = Path(str(run_manifest_path)).expanduser()
    papers_root = Path(str(papers_dir)).expanduser()
    strict_evidence_root = _strict_evidence_store_root(papers_root)
    source_span_root = _source_span_store_root(papers_root)
    requested_papers = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}
    requested_run_id = _safe_text(run_id)

    warnings: list[str] = []
    schema_violations: list[str] = []

    apply_report = _read_json(apply_path)
    manifest = _read_json(manifest_path)

    if not apply_report:
        schema_violations.append("apply_report_missing_or_unreadable")
    else:
        validation = validate_payload(
            apply_report,
            PARSED_ARTIFACT_STRICT_EVIDENCE_EXECUTOR_APPLY_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            schema_violations.extend(str(error) for error in validation.errors)
        if _safe_text(apply_report.get("status")) != "ok":
            schema_violations.append(
                f"apply_report_status={_safe_text(apply_report.get('status')) or 'unknown'}"
            )
        if not _safe_bool((apply_report.get("input") or {}).get("apply")):
            warnings.append("apply_report_not_in_apply_mode")

    if not requested_run_id and apply_report:
        requested_run_id = _safe_text((apply_report.get("input") or {}).get("runId"))

    expected_input_rows = 0
    if apply_report:
        counts = apply_report.get("counts") if isinstance(apply_report.get("counts"), dict) else {}
        expected_input_rows = int(counts.get("strictEvidenceWriteRows") or counts.get("inputRows") or 0)

    run_identity: dict[str, Any] = {
        "requestedRunId": requested_run_id,
        "requestedRunManifestPath": str(manifest_path),
        "runIdFilterMode": "none",
        "manifestCandidateRunIds": [],
        "observedRecordRunIds": [],
        "resolvedRecordRunIds": [],
        "resolution": "",
    }

    if schema_violations:
        rows = [
            {
                "review_row_id": "parsed-artifact-strict-evidence-promotion-readback-review:0000",
                "strictEvidenceId": "",
                "sourceSpanId": "",
                "candidateRecordId": "",
                "runId": "",
                "paper_id": "",
                "artifact_type": "",
                "sourceContentHash": "",
                "idempotencyKey": "",
                "strict_evidence_store_path": "",
                "strict_evidence_store_line": 0,
                "source_span_reference_found": False,
                "source_span_reference_hash_match": False,
                "readback_status": READBACK_STATUS_BLOCKED_INPUT_REPORT,
                "review_blockers": _dedupe(schema_violations),
                "readback_validated": False,
                "strictEvidenceCreated": False,
                "strictEvidenceWriteRows": 0,
                "sourceSpanUpdatedRows": 0,
                "citationGrade": False,
                "runtimeEvidence": False,
                "parserRoutingChanged": False,
                "answerIntegrationChanged": False,
                "databaseMutation": False,
                "canonicalParsedArtifactsWritten": False,
                "recommended_action": "repair_apply_report_before_readback_review",
            }
        ]
        counts = _count_rows(
            rows=rows,
            schema_violations=_dedupe(schema_violations),
            expected_input_rows=expected_input_rows,
        )
        return {
            "schema": PARSED_ARTIFACT_STRICT_EVIDENCE_PROMOTION_READBACK_REVIEW_SCHEMA_ID,
            "status": "blocked",
            "generatedAt": _now_iso(),
            "input": {
                "applyReportPath": str(apply_path),
                "applyReportSchema": _safe_text(apply_report.get("schema")),
                "applyReportStatus": _safe_text(apply_report.get("status")),
                "runManifestPath": str(manifest_path),
                "papersDir": str(papers_root),
                "strictEvidenceStoreRoot": str(strict_evidence_root),
                "sourceSpanStoreRoot": str(source_span_root),
                "requestedRunId": requested_run_id,
                "requestedPaperIds": sorted(requested_papers),
                "runIdentity": run_identity,
                "expectedInputRowsFromApplyReport": expected_input_rows,
            },
            "counts": counts,
            "gate": {
                "readbackReviewReady": False,
                "strictEvidencePolicyGateReady": False,
                "strictEvidenceCreated": False,
                "citationReady": False,
                "runtimeEvidenceReady": False,
                "parserRoutingReady": False,
                "answerIntegrationReady": False,
                "runtimeMutationAllowed": False,
                "schemaViolations": _dedupe(schema_violations),
                "decision": "blocked",
                "recommendedNextTranche": "repair_apply_report_before_strict_evidence_readback_review",
            },
            "policy": {
                "reportOnly": True,
                "strictEvidenceStoreWrite": False,
                "sourceSpanStoreWrite": False,
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

    if not strict_evidence_root.exists():
        warnings.append("strict_evidence_store_root_missing")
    if not source_span_root.exists():
        warnings.append("source_span_store_root_missing")

    all_raw_rows = _iter_strict_evidence_records(strict_evidence_root)
    raw_rows = list(all_raw_rows)
    run_identity["observedRecordRunIds"] = _observed_record_run_ids(all_raw_rows)

    if manifest_path.is_file():
        if not manifest:
            warnings.append("run_manifest_unreadable")
        resolved_run_ids, resolution = _resolve_record_run_ids_from_manifest(
            manifest=manifest,
            manifest_path=str(manifest_path),
            raw_rows=all_raw_rows,
        )
        run_identity.update(resolution)
        run_identity["runIdFilterMode"] = "run_manifest_resolved_record_run_id"
        if resolved_run_ids:
            resolved_set = set(resolved_run_ids)
            raw_rows = [
                row
                for row in raw_rows
                if _safe_text(row.get("record", {}).get("runId")) in resolved_set
            ]
        elif requested_run_id:
            raw_rows = [
                row
                for row in raw_rows
                if _safe_text(row.get("record", {}).get("runId")) == requested_run_id
            ]
        else:
            warnings.append("run_manifest_record_run_id_mismatch")
            raw_rows = []
    elif requested_run_id:
        run_identity["runIdFilterMode"] = "record_run_id_exact"
        raw_rows = [
            row for row in raw_rows if _safe_text(row.get("record", {}).get("runId")) == requested_run_id
        ]
        run_identity["resolvedRecordRunIds"] = [requested_run_id] if raw_rows else []

    if requested_papers:
        found_papers = {
            _safe_text(row.get("record", {}).get("paperId"))
            for row in raw_rows
            if _safe_text(row.get("record", {}).get("paperId"))
        }
        if requested_papers - found_papers:
            warnings.append("requested_paper_ids_not_found")
        raw_rows = [
            row
            for row in raw_rows
            if _safe_text(row.get("record", {}).get("paperId")) in requested_papers
        ]

    if expected_input_rows and len(raw_rows) != expected_input_rows:
        warnings.append(
            f"strict_evidence_record_count_mismatch:expected={expected_input_rows}:actual={len(raw_rows)}"
        )

    classify_run_id = requested_run_id
    if run_identity.get("resolvedRecordRunIds") and len(run_identity["resolvedRecordRunIds"]) == 1:
        classify_run_id = str(run_identity["resolvedRecordRunIds"][0])

    source_span_index = _load_source_span_index(source_span_root)
    rows, row_schema_violations = _review_rows(
        raw_rows,
        requested_run_id=classify_run_id,
        source_span_index=source_span_index,
    )
    schema_violations = _dedupe([*schema_violations, *row_schema_violations])

    counts = _count_rows(
        rows=rows,
        schema_violations=schema_violations,
        expected_input_rows=expected_input_rows or len(rows),
    )

    validated_count = int(counts.get("readbackValidatedRows") or 0)
    status = "ok"
    if schema_violations or validated_count != len(rows) or (
        expected_input_rows and validated_count != expected_input_rows
    ):
        status = "blocked"

    source_span_refs_found = sum(1 for row in rows if row.get("source_span_reference_found"))
    source_span_hash_matches = sum(1 for row in rows if row.get("source_span_reference_hash_match"))

    return {
        "schema": PARSED_ARTIFACT_STRICT_EVIDENCE_PROMOTION_READBACK_REVIEW_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "applyReportPath": str(apply_path),
            "applyReportSchema": _safe_text(apply_report.get("schema")),
            "applyReportStatus": _safe_text(apply_report.get("status")),
            "runManifestPath": str(manifest_path),
            "papersDir": str(papers_root),
            "strictEvidenceStoreRoot": str(strict_evidence_root),
            "sourceSpanStoreRoot": str(source_span_root),
            "requestedRunId": requested_run_id,
            "requestedPaperIds": sorted(requested_papers),
            "runIdentity": run_identity,
            "expectedInputRowsFromApplyReport": expected_input_rows,
            "sourceSpanReferenceSummary": {
                "recordsWithSourceSpanReference": source_span_refs_found,
                "recordsWithMatchingSourceContentHash": source_span_hash_matches,
            },
        },
        "counts": counts,
        "gate": {
            "readbackReviewReady": status == "ok" and validated_count == len(rows),
            "strictEvidencePolicyGateReady": status == "ok" and validated_count == len(rows),
            "strictEvidenceCreated": False,
            "citationReady": False,
            "runtimeEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": schema_violations,
            "decision": (
                "parsed_artifact_strict_evidence_promotion_readback_review_ready"
                if status == "ok"
                else "blocked"
            ),
            "recommendedNextTranche": (
                "parsed_artifact_strict_evidence_policy_gate"
                if status == "ok"
                else "repair_strict_evidence_store_records_before_policy_gate"
            ),
        },
        "policy": {
            "reportOnly": True,
            "strictEvidenceStoreWrite": False,
            "sourceSpanStoreWrite": False,
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


def render_parsed_artifact_strict_evidence_promotion_readback_review_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byReadbackStatus") or {})).items())
    ]
    ref_summary = dict((report.get("input") or {}).get("sourceSpanReferenceSummary") or {})
    return "\n".join(
        [
            "# Parsed Artifact StrictEvidence Promotion Readback Review",
            "",
            f"- status: {report.get('status', '')}",
            f"- input rows: {int(counts.get('inputRows') or 0)}",
            f"- strict evidence record rows: {int(counts.get('strictEvidenceRecordRows') or 0)}",
            f"- readback validated rows: {int(counts.get('readbackValidatedRows') or 0)}",
            f"- source span references found: {int(ref_summary.get('recordsWithSourceSpanReference') or 0)}",
            f"- source content hash matches: {int(ref_summary.get('recordsWithMatchingSourceContentHash') or 0)}",
            f"- strict evidence created: {int(counts.get('strictEvidenceCreatedRows') or 0)}",
            f"- citation-grade evidence created: {int(counts.get('citationGradeEvidenceCreatedRows') or 0)}",
            f"- runtime evidence created: {int(counts.get('runtimeEvidenceCreatedRows') or 0)}",
            "",
            "## Readback status breakdown",
            *[f"- {item}" for item in by_status],
        ]
    )


def write_parsed_artifact_strict_evidence_promotion_readback_review_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-strict-evidence-promotion-readback-review.json"
    summary_path = root / "parsed-artifact-strict-evidence-promotion-readback-review-summary.json"
    markdown_path = root / "parsed-artifact-strict-evidence-promotion-readback-review.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_parsed_artifact_strict_evidence_promotion_readback_review_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description="Readback review for applied parsed-artifact StrictEvidence JSONL records."
    )
    parser.add_argument("--apply-report", default=str(DEFAULT_APPLY_REPORT_PATH))
    parser.add_argument("--run-manifest", default=str(DEFAULT_RUN_MANIFEST_PATH))
    parser.add_argument("--papers-dir", default=str(DEFAULT_PAPERS_DIR))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--paper-id", action="append", default=[])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = build_parsed_artifact_strict_evidence_promotion_readback_review(
        apply_report_path=args.apply_report,
        run_manifest_path=args.run_manifest,
        papers_dir=args.papers_dir,
        run_id=args.run_id or None,
        paper_ids=args.paper_id or None,
    )
    paths = write_parsed_artifact_strict_evidence_promotion_readback_review_reports(
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
    "PARSED_ARTIFACT_STRICT_EVIDENCE_PROMOTION_READBACK_REVIEW_SCHEMA_ID",
    "READBACK_STATUS_VALIDATED",
    "DEFAULT_APPLY_REPORT_PATH",
    "DEFAULT_RUN_MANIFEST_PATH",
    "build_parsed_artifact_strict_evidence_promotion_readback_review",
    "write_parsed_artifact_strict_evidence_promotion_readback_review_reports",
]
