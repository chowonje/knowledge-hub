"""Read-only full-apply readiness gate for evidence chunk candidates.

This gate consumes the parsed-artifact evidence chunk candidate dry-run and the
canary apply/readback report. It rechecks canary readback against the local
candidate store, but it does not write records or promote anything to runtime
evidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_apply_executor import (
    _contains_private_path,
    normalize_text,
    sanitized_report_ref,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_canary_apply_readback import (
    APPLIED_DECISION as CANARY_APPLIED_DECISION,
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_CANARY_APPLY_READBACK_SCHEMA_ID,
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RECORD_SCHEMA_ID,
    load_json,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_dry_run import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_DRY_RUN_SCHEMA_ID,
    READY_DECISION as CANDIDATE_DRY_RUN_READY_DECISION,
    ZERO_COUNTER_FIELDS,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_READINESS_GATE_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-candidate-full-apply-readiness-gate.v1"
)

READINESS_STATUS_VALIDATED = "full_apply_readiness_validated_canary_record"
READINESS_STATUS_BLOCKED_MISSING_STORE = "blocked_missing_store_record"
READINESS_STATUS_BLOCKED_DUPLICATE_STORE = "blocked_duplicate_store_record"
READINESS_STATUS_BLOCKED_HASH_MISMATCH = "blocked_hash_mismatch"
READINESS_STATUS_BLOCKED_POLICY_VIOLATION = "blocked_policy_violation"

READY_DECISION = "parsed_artifact_evidence_chunk_candidate_full_apply_readiness_gate_ready"
BLOCKED_DECISION = "blocked"
NEXT_TRANCHE = "parsed_artifact_evidence_chunk_candidate_full_apply_executor_apply_readback"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _safe_filename(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return text.strip("._") or "unknown"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _record_hash(record: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(record).encode("utf-8")).hexdigest()


def _store_path(papers_dir: str | Path, paper_id: str) -> Path:
    return (
        Path(str(papers_dir)).expanduser()
        / "structured_evidence_candidates"
        / "evidence_chunk"
        / f"{_safe_filename(paper_id)}.jsonl"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _source_dry_run_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers: list[str] = []
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_DRY_RUN_SCHEMA_ID:
        blockers.append("invalid_candidate_dry_run_schema")
    if report.get("status") != "ready":
        blockers.append("candidate_dry_run_not_ready")
    if report.get("decision") != CANDIDATE_DRY_RUN_READY_DECISION:
        blockers.append("candidate_dry_run_invalid_decision")
    if dict(report.get("gate") or {}).get("passed") is not True:
        blockers.append("candidate_dry_run_gate_not_passed")
    if _int(counts.get("selectedCandidateRows")) <= 0:
        blockers.append("candidate_dry_run_has_no_selected_rows")
    for field_name in ("blockedRows", "privatePathLeakRows", "schemaViolationCount", *ZERO_COUNTER_FIELDS):
        if _int(counts.get(field_name)) != 0:
            blockers.append(f"candidate_dry_run_has_{field_name}")
    if _contains_private_path(report):
        blockers.append("candidate_dry_run_has_private_path_leak")
    return sorted(set(blockers))


def _source_canary_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers: list[str] = []
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_CANARY_APPLY_READBACK_SCHEMA_ID:
        blockers.append("invalid_canary_apply_readback_schema")
    if report.get("status") != "applied":
        blockers.append("canary_apply_readback_not_applied")
    if report.get("decision") != CANARY_APPLIED_DECISION:
        blockers.append("canary_apply_readback_invalid_decision")
    if _int(counts.get("selectedCanaryRows")) != 10:
        blockers.append("canary_selected_rows_not_10")
    if _int(counts.get("readbackValidatedRows")) != _int(counts.get("selectedCanaryRows")):
        blockers.append("canary_readback_count_mismatch")
    if _int(counts.get("candidateStoreWriteRows")) + _int(counts.get("alreadyCorrectRows")) < _int(counts.get("selectedCanaryRows")):
        blockers.append("canary_store_write_or_already_correct_count_mismatch")
    for field_name in ("blockedRows", "privatePathLeakRows", "schemaViolationCount"):
        if _int(counts.get(field_name)) != 0:
            blockers.append(f"canary_apply_readback_has_{field_name}")
    for field_name in (
        "sourceSpanCreatedRows",
        "strictEvidenceRows",
        "citationGradeRows",
        "runtimeEvidenceRows",
        "answerVisibleRows",
        "answerGenerationRows",
        "databaseMutationRows",
        "indexMutationRows",
        "reindexOrReembedRows",
        "parserExecutionRows",
        "canonicalParsedArtifactWriteRows",
        "vaultScanRows",
        "externalDownloadRows",
    ):
        if _int(counts.get(field_name)) != 0:
            blockers.append(f"canary_apply_readback_has_{field_name}")
    if _contains_private_path(report):
        blockers.append("canary_apply_readback_has_private_path_leak")
    return sorted(set(blockers))


def _record_policy_ok(record: dict[str, Any]) -> bool:
    return (
        record.get("schema") == PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RECORD_SCHEMA_ID
        and record.get("candidateOnly") is True
        and record.get("answerEvidenceCandidate") is True
        and record.get("answerabilityCandidate") is True
        and record.get("strictEvidence") is False
        and record.get("citationGrade") is False
        and record.get("runtimeEvidence") is False
        and record.get("answerVisible") is False
    )


def _dry_run_summary(report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = dict(report.get("counts") or {})
    return {
        "schema": normalize_text(report.get("schema")),
        "status": normalize_text(report.get("status")),
        "decision": normalize_text(report.get("decision")),
        "reportRef": normalize_text(report_ref),
        "selectedCandidateRows": _int(counts.get("selectedCandidateRows")),
        "candidateRows": _int(counts.get("candidateRows")),
        "heldCandidateRows": _int(counts.get("heldCandidateRows")),
        "paragraphCandidateRows": _int(counts.get("paragraphCandidateRows")),
        "sectionCandidateRows": _int(counts.get("sectionCandidateRows")),
        "blockedRows": _int(counts.get("blockedRows")),
        "privatePathLeakRows": _int(counts.get("privatePathLeakRows")),
        "schemaViolationCount": _int(counts.get("schemaViolationCount")),
    }


def _canary_summary(report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = dict(report.get("counts") or {})
    return {
        "schema": normalize_text(report.get("schema")),
        "status": normalize_text(report.get("status")),
        "decision": normalize_text(report.get("decision")),
        "reportRef": normalize_text(report_ref),
        "selectedCanaryRows": _int(counts.get("selectedCanaryRows")),
        "heldRows": _int(counts.get("heldRows")),
        "candidateStoreWriteRows": _int(counts.get("candidateStoreWriteRows")),
        "alreadyCorrectRows": _int(counts.get("alreadyCorrectRows")),
        "readbackValidatedRows": _int(counts.get("readbackValidatedRows")),
        "blockedRows": _int(counts.get("blockedRows")),
        "privatePathLeakRows": _int(counts.get("privatePathLeakRows")),
        "schemaViolationCount": _int(counts.get("schemaViolationCount")),
    }


def build_parsed_artifact_evidence_chunk_candidate_full_apply_readiness_gate(
    *,
    candidate_dry_run_report: dict[str, Any],
    source_candidate_dry_run_report_ref: str,
    canary_apply_readback_report: dict[str, Any],
    source_canary_apply_readback_report_ref: str,
    papers_dir: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    schema_violations = sorted(
        set(
            [
                *_source_dry_run_blockers(candidate_dry_run_report),
                *_source_canary_blockers(canary_apply_readback_report),
            ]
        )
    )
    dry_run_rows = [
        dict(row)
        for row in list(candidate_dry_run_report.get("candidateRows") or [])
        if isinstance(row, dict)
    ]
    dry_run_ids = {normalize_text(row.get("candidateRowId")) for row in dry_run_rows}
    expected_records = [
        dict(row)
        for row in list(canary_apply_readback_report.get("candidateRecords") or [])
        if isinstance(row, dict)
    ]
    rows: list[dict[str, Any]] = []
    store_rows_by_record_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in expected_records:
        for store_row in _read_jsonl(_store_path(papers_dir, normalize_text(record.get("paperId")))):
            store_rows_by_record_id[normalize_text(store_row.get("candidateRecordId"))].append(store_row)

    for index, expected in enumerate(expected_records, start=1):
        record_id = normalize_text(expected.get("candidateRecordId"))
        source_row_id = normalize_text(expected.get("sourceCandidateRowId"))
        matches = store_rows_by_record_id.get(record_id, [])
        stored = matches[0] if matches else {}
        checks = {
            "sourceCandidateRowInDryRun": source_row_id in dry_run_ids,
            "storeRecordPresent": bool(matches),
            "singleStoreRecordForCandidateRecordId": len(matches) == 1,
            "recordHashMatches": bool(stored) and _record_hash(stored) == _record_hash(expected),
            "recordBytesMatchCanonicalJson": bool(stored) and stored == expected,
            "paperIdMatches": normalize_text(stored.get("paperId")) == normalize_text(expected.get("paperId")),
            "sourceContentHashMatches": normalize_text(stored.get("sourceContentHash")) == normalize_text(expected.get("sourceContentHash")),
            "spanLocatorMatches": normalize_text(stored.get("spanLocator")) == normalize_text(expected.get("spanLocator")),
            "snippetHashMatches": normalize_text(stored.get("snippetHash")) == normalize_text(expected.get("snippetHash")),
            "policyCandidateOnly": bool(stored) and _record_policy_ok(stored),
            "recordNotPrivatePathLeaking": bool(stored) and not _contains_private_path(stored),
        }
        blockers = [name for name, passed in checks.items() if not passed]
        if not matches:
            readiness_status = READINESS_STATUS_BLOCKED_MISSING_STORE
        elif len(matches) != 1:
            readiness_status = READINESS_STATUS_BLOCKED_DUPLICATE_STORE
        elif not checks["policyCandidateOnly"] or not checks["recordNotPrivatePathLeaking"]:
            readiness_status = READINESS_STATUS_BLOCKED_POLICY_VIOLATION
        elif blockers:
            readiness_status = READINESS_STATUS_BLOCKED_HASH_MISMATCH
        else:
            readiness_status = READINESS_STATUS_VALIDATED
        rows.append(
            {
                "readinessRowId": f"parsed-artifact-evidence-chunk-candidate-full-apply-readiness-gate:{index:04d}",
                "sourceCandidateRowId": source_row_id,
                "candidateRecordId": record_id,
                "paperId": normalize_text(expected.get("paperId")),
                "artifactType": normalize_text(expected.get("artifactType")),
                "sourceContentHash": normalize_text(expected.get("sourceContentHash")),
                "spanLocator": normalize_text(expected.get("spanLocator")),
                "snippetHash": normalize_text(expected.get("snippetHash")),
                "expectedRecordSha256": _record_hash(expected),
                "storedRecordSha256": _record_hash(stored) if stored else "",
                "matchingStoreRecordRows": len(matches),
                "readbackValidated": readiness_status == READINESS_STATUS_VALIDATED,
                "candidateOnly": True,
                "strictEvidence": False,
                "citationGrade": False,
                "runtimeEvidence": False,
                "answerVisible": False,
                "readinessStatus": readiness_status,
                "readinessBlockers": sorted(set(blockers)),
                "checks": checks,
            }
        )

    private_path_leak_rows = 1 if _contains_private_path(rows) else 0
    if private_path_leak_rows:
        schema_violations.append("private_path_leak")
    by_status = Counter(row["readinessStatus"] for row in rows)
    blocked_rows = sum(1 for row in rows if row["readinessStatus"] != READINESS_STATUS_VALIDATED)
    dry_counts = dict(candidate_dry_run_report.get("counts") or {})
    canary_counts = dict(canary_apply_readback_report.get("counts") or {})
    counts = {
        "inputCandidateRows": _int(dry_counts.get("selectedCandidateRows")),
        "fullApplyCandidateRows": _int(dry_counts.get("selectedCandidateRows")),
        "canaryCandidateRows": len(expected_records),
        "heldRows": max(0, _int(dry_counts.get("selectedCandidateRows")) - len(expected_records)),
        "canaryReadbackValidatedRows": _int(canary_counts.get("readbackValidatedRows")),
        "storeValidatedCanaryRows": by_status.get(READINESS_STATUS_VALIDATED, 0),
        "missingStoreRows": by_status.get(READINESS_STATUS_BLOCKED_MISSING_STORE, 0),
        "duplicateStoreRows": by_status.get(READINESS_STATUS_BLOCKED_DUPLICATE_STORE, 0),
        "hashMismatchRows": by_status.get(READINESS_STATUS_BLOCKED_HASH_MISMATCH, 0),
        "policyViolationRows": by_status.get(READINESS_STATUS_BLOCKED_POLICY_VIOLATION, 0),
        "blockedRows": blocked_rows,
        "candidateStoreWriteRows": 0,
        "sourceSpanCreatedRows": 0,
        "strictEvidenceRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "answerVisibleRows": 0,
        "answerGenerationRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "reindexOrReembedRows": 0,
        "parserExecutionRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
        "vaultScanRows": 0,
        "externalDownloadRows": 0,
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(set(schema_violations)),
        "byArtifactType": dict(Counter(row["artifactType"] for row in rows)),
        "byReadinessStatus": dict(by_status),
    }
    ready = not schema_violations and blocked_rows == 0 and counts["storeValidatedCanaryRows"] == len(expected_records) == 10
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_READINESS_GATE_SCHEMA_ID,
        "status": "ready" if ready else "blocked",
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if ready else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE,
        "input": {
            "sourceCandidateDryRunReportRef": normalize_text(source_candidate_dry_run_report_ref),
            "sourceCanaryApplyReadbackReportRef": normalize_text(source_canary_apply_readback_report_ref),
            "papersDirRef": "papers_dir",
            "candidateStoreRefTemplate": "papers_dir/structured_evidence_candidates/evidence_chunk/{paperId}.jsonl",
        },
        "sourceCandidateDryRun": _dry_run_summary(candidate_dry_run_report, report_ref=source_candidate_dry_run_report_ref),
        "sourceCanaryApplyReadback": _canary_summary(canary_apply_readback_report, report_ref=source_canary_apply_readback_report_ref),
        "policy": {
            "readOnlyReview": True,
            "candidateStoreWrite": False,
            "fullApplyPerformed": False,
            "sourceSpanCreated": False,
            "strictEvidenceCreated": False,
            "citationGradeEvidenceCreated": False,
            "runtimeEvidenceCreated": False,
            "answerVisibleExposure": False,
            "answerGeneration": False,
            "parserRoutingChanged": False,
            "answerIntegrationChanged": False,
            "databaseMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "parserExecution": False,
            "canonicalParsedArtifactsWritten": False,
        },
        "counts": counts,
        "gate": {
            "readyForFullApplyExecutor": ready,
            "canaryOnlyReviewed": True,
            "fullApplyAllowedByThisReport": ready,
            "fullApplyPerformed": False,
            "candidateStoreWriteAllowedInThisTranche": False,
            "runtimeVisibilityAllowed": False,
            "evidencePromotionAllowed": False,
            "schemaViolations": sorted(set(schema_violations)),
        },
        "rows": rows,
        "warnings": [],
    }


def render_parsed_artifact_evidence_chunk_candidate_full_apply_readiness_gate_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    by_status = dict(counts.get("byReadinessStatus") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Candidate Full Apply Readiness Gate",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- fullApplyCandidateRows: `{counts.get('fullApplyCandidateRows')}`",
        f"- canaryCandidateRows: `{counts.get('canaryCandidateRows')}`",
        f"- heldRows: `{counts.get('heldRows')}`",
        f"- canaryReadbackValidatedRows: `{counts.get('canaryReadbackValidatedRows')}`",
        f"- storeValidatedCanaryRows: `{counts.get('storeValidatedCanaryRows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Mutation Guarantees",
        "",
        f"- candidateStoreWriteRows: `{counts.get('candidateStoreWriteRows')}`",
        f"- sourceSpanCreatedRows: `{counts.get('sourceSpanCreatedRows')}`",
        f"- strictEvidenceRows: `{counts.get('strictEvidenceRows')}`",
        f"- citationGradeRows: `{counts.get('citationGradeRows')}`",
        f"- runtimeEvidenceRows: `{counts.get('runtimeEvidenceRows')}`",
        f"- answerVisibleRows: `{counts.get('answerVisibleRows')}`",
        f"- databaseMutationRows: `{counts.get('databaseMutationRows')}`",
        f"- indexMutationRows: `{counts.get('indexMutationRows')}`",
        f"- vaultScanRows: `{counts.get('vaultScanRows')}`",
        f"- externalDownloadRows: `{counts.get('externalDownloadRows')}`",
        "",
        "## Readiness Status",
        "",
    ]
    for status, count in sorted(by_status.items()):
        lines.append(f"- `{status}`: `{count}`")
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_evidence_chunk_candidate_full_apply_readiness_gate(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(render_parsed_artifact_evidence_chunk_candidate_full_apply_readiness_gate_markdown(report), encoding="utf-8")
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "BLOCKED_DECISION",
    "NEXT_TRANCHE",
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_READINESS_GATE_SCHEMA_ID",
    "READINESS_STATUS_VALIDATED",
    "READY_DECISION",
    "build_parsed_artifact_evidence_chunk_candidate_full_apply_readiness_gate",
    "load_json",
    "sanitized_report_ref",
    "write_parsed_artifact_evidence_chunk_candidate_full_apply_readiness_gate",
]
