"""Readback review for applied StrictEvidence runtime binding store records.

Reconciles runtime binding JSONL rows written by the runtime binding executor apply
tranche against StrictEvidence, eligibility, and SourceSpan stores. Read-only:
does not write runtime binding, eligibility, StrictEvidence, or SourceSpan JSONL or
mutate runtime/integration surfaces.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_runtime_binding_executor_apply import (
    APPLY_STATUS_APPLIED,
    STRICT_EVIDENCE_RUNTIME_BINDING_EXECUTOR_APPLY_SCHEMA_ID,
)
from knowledge_hub.papers.strict_evidence_runtime_binding_record_contract import (
    RUNTIME_BINDING_DECISION,
    RUNTIME_BINDING_POLICY_VERSION,
    RUNTIME_BINDING_STATE,
    STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_CONTRACT_SCHEMA_ID,
    STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_SCHEMA_ID,
    RUNTIME_BINDING_STORE,
    validate_runtime_binding_record_semantics,
)


STRICT_EVIDENCE_RUNTIME_BINDING_EXECUTOR_APPLY_READBACK_REVIEW_SCHEMA_ID = (
    "knowledge-hub.paper.strict-evidence-runtime-binding-executor-apply-readback-review.v1"
)

READBACK_STATUS_VALIDATED = "runtime_binding_readback_validated"
READBACK_STATUS_BLOCKED_MISSING_STORE = "blocked_missing_runtime_binding_store"
READBACK_STATUS_BLOCKED_SCHEMA = "blocked_runtime_binding_record_schema_violation"
READBACK_STATUS_BLOCKED_SEMANTIC = "blocked_runtime_binding_record_semantic_violation"
READBACK_STATUS_BLOCKED_MISSING_STRICT_EVIDENCE = "blocked_missing_strict_evidence_reference"
READBACK_STATUS_BLOCKED_MISSING_ELIGIBILITY = "blocked_missing_eligibility_reference"
READBACK_STATUS_BLOCKED_MISSING_SOURCE_SPAN = "blocked_missing_source_span_reference"
READBACK_STATUS_BLOCKED_MISSING_CANDIDATE = "blocked_missing_candidate_record_id"
READBACK_STATUS_BLOCKED_IDENTITY = "blocked_candidate_identity_mismatch"
READBACK_STATUS_BLOCKED_DUPLICATE_IDEMPOTENCY = "blocked_duplicate_idempotency_key"
READBACK_STATUS_BLOCKED_DUPLICATE_RECORD_ID = "blocked_duplicate_runtime_binding_record_id"
READBACK_STATUS_BLOCKED_RUNTIME_OR_ANSWER = "blocked_runtime_or_answer_flag_violation"
READBACK_STATUS_BLOCKED_STORE_COUNT = "blocked_store_row_count_changed"
READBACK_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"

EXPECTED_INPUT_ROWS = 99
EXPECTED_RUNTIME_BINDING_RECORD_ROWS = 99
EXPECTED_CITATION_GRADE_STORE_ROWS = 99
EXPECTED_STRICT_EVIDENCE_STORE_ROWS = 99
EXPECTED_ELIGIBILITY_STORE_ROWS = 99
EXPECTED_SOURCE_SPAN_STORE_ROWS = 102

DEFAULT_APPLY_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "strict-evidence-runtime-binding-executor-apply"
    / "02-strict-evidence-runtime-binding-executor-apply"
    / "strict-evidence-runtime-binding-executor-apply.json"
)

DEFAULT_DRY_RUN_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "strict-evidence-runtime-binding-executor-apply"
    / "01-strict-evidence-runtime-binding-executor-apply-dry-run"
    / "strict-evidence-runtime-binding-executor-apply.json"
)

DEFAULT_CONTRACT_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "strict-evidence-runtime-binding-record-contract"
    / "01-strict-evidence-runtime-binding-record-contract"
    / "strict-evidence-runtime-binding-record-contract.json"
)

DEFAULT_PAPERS_DIR = Path.home() / ".khub" / "papers"

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "strict-evidence-runtime-binding-executor-apply-readback-review"
    / "01-strict-evidence-runtime-binding-executor-apply-readback-review"
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


def _runtime_binding_store_root(papers_dir: str | Path) -> Path:
    return Path(str(papers_dir)).expanduser() / "structured_evidence" / "strict_evidence_runtime_binding"


def _citation_grade_store_root(papers_dir: str | Path) -> Path:
    return Path(str(papers_dir)).expanduser() / "structured_evidence" / "strict_evidence_citation_grade"


def _strict_evidence_store_root(papers_dir: str | Path) -> Path:
    return Path(str(papers_dir)).expanduser() / "structured_evidence" / "strict_evidence"


def _eligibility_store_root(papers_dir: str | Path) -> Path:
    return Path(str(papers_dir)).expanduser() / "structured_evidence" / "strict_evidence_eligibility"


def _source_span_store_root(papers_dir: str | Path) -> Path:
    return Path(str(papers_dir)).expanduser() / "structured_evidence" / "source_span"


def _count_jsonl_rows(root: Path) -> int:
    if not root.is_dir():
        return 0
    total = 0
    for path in sorted(root.glob("*.jsonl")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if not text:
            continue
        total += sum(1 for line in text.splitlines() if line.strip())
    return total


def _iter_jsonl_records(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
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


def _load_index(root: Path, key_field: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for item in _iter_jsonl_records(root):
        record = item.get("record") if isinstance(item.get("record"), dict) else {}
        key = _safe_text(record.get(key_field))
        if key:
            index[key] = record
    return index


def _no_mutation_policy_matrix() -> dict[str, Any]:
    return {
        "readbackOnly": True,
        "runtimeBindingRecordWrite": False,
        "eligibilityRecordWrite": False,
        "citationGradeRecordWrite": False,
        "citationGradeBooleanMutation": False,
        "strictEvidenceStoreWrite": False,
        "sourceSpanStoreWrite": False,
        "strictEvidenceCreated": False,
        "strictEligibleMutation": False,
        "runtimeEvidenceCreated": False,
        "runtimeVisible": False,
        "answerIntegrationVisible": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "databaseMutation": False,
        "vaultScan": False,
        "reindexOrReembed": False,
        "canonicalParsedArtifactsWritten": False,
        "manifestWrite": False,
    }


def _runtime_or_answer_violations(record: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for field_name in (
        "runtimeBindingMutationApplied",
        "strictEligibleMutationApplied",
        "runtimeEvidence",
        "runtimeVisible",
        "answerIntegrationVisible",
        "strictEligible",
        "strictEvidenceCreated",
        "parserRoutingChanged",
        "answerIntegrationChanged",
        "databaseMutation",
    ):
        if _safe_bool(record.get(field_name)):
            violations.append(f"{field_name}_true")
    write_policy = record.get("writePolicy") if isinstance(record.get("writePolicy"), dict) else {}
    for field_name in (
        "runtimeBindingRecordWrite",
        "runtimeEvidenceCreated",
        "runtimeVisible",
        "answerIntegrationVisible",
        "citationGradeRecordWrite",
        "citationGradeBooleanMutation",
        "eligibilityRecordWrite",
        "strictEvidenceStoreWrite",
        "sourceSpanStoreWrite",
        "strictEligibleMutation",
        "databaseMutation",
        "parserRoutingChanged",
        "answerIntegrationChanged",
        "reindexOrReembed",
        "canonicalParsedArtifactsWritten",
    ):
        if _safe_bool(write_policy.get(field_name)):
            violations.append(f"writePolicy.{field_name}_true")
    return _dedupe(violations)


def _parent_record_violations(parent: dict[str, Any], *, label: str) -> list[str]:
    violations: list[str] = []
    if _safe_bool(parent.get("strictEligible")):
        violations.append(f"{label}_strictEligible_true")
    if _safe_bool(parent.get("citationGrade")):
        violations.append(f"{label}_citationGrade_true")
    if _safe_bool(parent.get("runtimeEvidence")):
        violations.append(f"{label}_runtimeEvidence_true")
    if _safe_bool(parent.get("runtimeVisible")):
        violations.append(f"{label}_runtimeVisible_true")
    if _safe_bool(parent.get("answerIntegrationVisible")):
        violations.append(f"{label}_answerIntegrationVisible_true")
    if _safe_bool(parent.get("runtimeBindingMutationApplied")):
        violations.append(f"{label}_runtimeBindingMutationApplied_true")
    if _safe_bool(parent.get("citationGradeMutationApplied")):
        violations.append(f"{label}_citationGradeMutationApplied_true")
    if _safe_bool(parent.get("strictEligibleMutationApplied")):
        violations.append(f"{label}_strictEligibleMutationApplied_true")
    return violations


def _extended_semantic_violations(record: dict[str, Any]) -> list[str]:
    violations = list(validate_runtime_binding_record_semantics(record))
    if _safe_text(record.get("runtimeBindingState")) != RUNTIME_BINDING_STATE:
        violations.append(
            f"runtimeBindingState_mismatch:{_safe_text(record.get('runtimeBindingState')) or 'missing'}"
        )
    if _safe_text(record.get("policyVersion")) != RUNTIME_BINDING_POLICY_VERSION:
        violations.append("policyVersion_mismatch")
    if _safe_text(record.get("runtimeBindingDecision")) != RUNTIME_BINDING_DECISION:
        violations.append("runtimeBindingDecision_mismatch")
    return _dedupe(violations)


def _candidate_identity_violations(
    record: dict[str, Any],
    *,
    apply_record: dict[str, Any],
    citation_grade: dict[str, Any],
    strict_evidence: dict[str, Any],
    eligibility: dict[str, Any],
    source_span: dict[str, Any],
) -> list[str]:
    violations: list[str] = []
    candidate_record_id = _safe_text(record.get("candidateRecordId"))
    if not candidate_record_id:
        return ["candidateRecordId_missing"]

    if _safe_text(apply_record.get("candidateRecordId")) not in ("", candidate_record_id):
        violations.append("apply_report_candidateRecordId_mismatch")
    if _safe_text(citation_grade.get("candidateRecordId")) != candidate_record_id:
        violations.append("citation_grade_candidateRecordId_mismatch")
    if _safe_text(eligibility.get("candidateRecordId")) != candidate_record_id:
        violations.append("eligibility_candidateRecordId_mismatch")
    if _safe_text(source_span.get("candidateRecordId")) != candidate_record_id:
        violations.append("source_span_candidateRecordId_mismatch")

    strict_candidates = strict_evidence.get("candidateRecordIds")
    if isinstance(strict_candidates, list):
        if candidate_record_id not in {_safe_text(item) for item in strict_candidates}:
            violations.append("strict_evidence_candidateRecordIds_missing_candidate")
    elif _safe_text(strict_evidence.get("candidateRecordId")) != candidate_record_id:
        violations.append("strict_evidence_candidateRecordId_mismatch")

    source_span_ids = strict_evidence.get("sourceSpanIds")
    source_span_id = _safe_text(record.get("sourceSpanId"))
    if isinstance(source_span_ids, list) and source_span_id:
        if source_span_id not in {_safe_text(item) for item in source_span_ids}:
            violations.append("strict_evidence_sourceSpanIds_missing_source_span")

    for field_name in (
        "citationGradeRecordId",
        "strictEvidenceId",
        "eligibilityRecordId",
        "sourceSpanId",
    ):
        record_value = _safe_text(record.get(field_name))
        apply_value = _safe_text(apply_record.get(field_name))
        if apply_value and apply_value != record_value:
            violations.append(f"apply_report_{field_name}_mismatch")

    citation_grade_id = _safe_text(record.get("citationGradeRecordId"))
    if _safe_text(citation_grade.get("citationGradeRecordId")) != citation_grade_id:
        violations.append("citation_grade_record_id_mismatch")

    record_hash = _safe_text(record.get("sourceContentHash"))
    provenance = record.get("provenanceTrace") if isinstance(record.get("provenanceTrace"), dict) else {}
    if _safe_text(provenance.get("sourceContentHash")) not in ("", record_hash):
        violations.append("provenanceTrace_sourceContentHash_mismatch")
    apply_hash = _safe_text(apply_record.get("sourceContentHash"))
    if apply_hash and apply_hash != record_hash:
        violations.append("apply_report_sourceContentHash_mismatch")

    if _safe_text(provenance.get("candidateRecordId")) not in ("", candidate_record_id):
        violations.append("provenanceTrace_candidateRecordId_mismatch")
    if _safe_text(provenance.get("citationGradeRecordId")) not in (
        "",
        _safe_text(record.get("citationGradeRecordId")),
    ):
        violations.append("provenanceTrace_citationGradeRecordId_mismatch")

    return _dedupe(violations)


def _classify_runtime_binding_record(
    record: dict[str, Any],
    *,
    apply_record_index: dict[str, dict[str, Any]],
    citation_grade_index: dict[str, dict[str, Any]],
    strict_evidence_index: dict[str, dict[str, Any]],
    eligibility_index: dict[str, dict[str, Any]],
    source_span_index: dict[str, dict[str, Any]],
    duplicate_idempotency: bool,
    duplicate_record_id: bool,
    store_exists: bool,
) -> tuple[str, list[str]]:
    if not store_exists:
        return READBACK_STATUS_BLOCKED_MISSING_STORE, ["runtime_binding_store_missing"]

    flag_violations = _runtime_or_answer_violations(record)
    if flag_violations:
        return READBACK_STATUS_BLOCKED_RUNTIME_OR_ANSWER, flag_violations

    validation = validate_payload(record, STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_SCHEMA_ID, strict=True)
    if not validation.ok:
        return READBACK_STATUS_BLOCKED_SCHEMA, [str(error) for error in validation.errors]

    semantic_violations = _extended_semantic_violations(record)
    if semantic_violations:
        return READBACK_STATUS_BLOCKED_SEMANTIC, semantic_violations

    if duplicate_idempotency:
        return READBACK_STATUS_BLOCKED_DUPLICATE_IDEMPOTENCY, [
            f"duplicate_idempotencyKey={_safe_text(record.get('idempotencyKey'))}"
        ]
    if duplicate_record_id:
        return READBACK_STATUS_BLOCKED_DUPLICATE_RECORD_ID, [
            f"duplicate_runtimeBindingRecordId={_safe_text(record.get('runtimeBindingRecordId'))}"
        ]

    if not _safe_text(record.get("candidateRecordId")):
        return READBACK_STATUS_BLOCKED_MISSING_CANDIDATE, ["candidateRecordId_missing"]

    runtime_binding_record_id = _safe_text(record.get("runtimeBindingRecordId"))
    apply_record = apply_record_index.get(runtime_binding_record_id, {})

    citation_grade_record_id = _safe_text(record.get("citationGradeRecordId"))
    citation_grade = citation_grade_index.get(citation_grade_record_id)
    if not citation_grade:
        return READBACK_STATUS_BLOCKED_IDENTITY, [
            f"citation_grade_record_missing:{citation_grade_record_id or 'unknown'}"
        ]
    citation_grade_parent_violations = _parent_record_violations(
        citation_grade, label="citation_grade"
    )
    if citation_grade_parent_violations:
        return READBACK_STATUS_BLOCKED_RUNTIME_OR_ANSWER, citation_grade_parent_violations

    strict_evidence_id = _safe_text(record.get("strictEvidenceId"))
    strict_evidence = strict_evidence_index.get(strict_evidence_id)
    if not strict_evidence:
        return READBACK_STATUS_BLOCKED_MISSING_STRICT_EVIDENCE, [
            f"strict_evidence_record_missing:{strict_evidence_id or 'unknown'}"
    ]
    strict_parent_violations = _parent_record_violations(strict_evidence, label="strict_evidence")
    if strict_parent_violations:
        return READBACK_STATUS_BLOCKED_RUNTIME_OR_ANSWER, strict_parent_violations

    eligibility_record_id = _safe_text(record.get("eligibilityRecordId"))
    eligibility = eligibility_index.get(eligibility_record_id)
    if not eligibility:
        return READBACK_STATUS_BLOCKED_MISSING_ELIGIBILITY, [
            f"eligibility_record_missing:{eligibility_record_id or 'unknown'}"
    ]
    eligibility_parent_violations = _parent_record_violations(eligibility, label="eligibility")
    if eligibility_parent_violations:
        return READBACK_STATUS_BLOCKED_RUNTIME_OR_ANSWER, eligibility_parent_violations

    source_span_id = _safe_text(record.get("sourceSpanId"))
    source_span = source_span_index.get(source_span_id)
    if not source_span:
        return READBACK_STATUS_BLOCKED_MISSING_SOURCE_SPAN, [
            f"source_span_record_missing:{source_span_id or 'unknown'}"
    ]
    source_span_parent_violations = _parent_record_violations(source_span, label="source_span")
    if source_span_parent_violations:
        return READBACK_STATUS_BLOCKED_RUNTIME_OR_ANSWER, source_span_parent_violations

    identity_violations = _candidate_identity_violations(
        record,
        apply_record=apply_record,
        citation_grade=citation_grade,
        strict_evidence=strict_evidence,
        eligibility=eligibility,
        source_span=source_span,
    )
    if identity_violations:
        if identity_violations == ["candidateRecordId_missing"]:
            return READBACK_STATUS_BLOCKED_MISSING_CANDIDATE, identity_violations
        return READBACK_STATUS_BLOCKED_IDENTITY, identity_violations

    return READBACK_STATUS_VALIDATED, []


def build_strict_evidence_runtime_binding_executor_apply_readback_review(
    *,
    apply_report_path: str | Path = DEFAULT_APPLY_REPORT_PATH,
    dry_run_report_path: str | Path = DEFAULT_DRY_RUN_REPORT_PATH,
    runtime_binding_record_contract_report_path: str | Path = DEFAULT_CONTRACT_REPORT_PATH,
    papers_dir: str | Path = DEFAULT_PAPERS_DIR,
    paper_ids: list[str] | None = None,
    expected_input_rows: int = EXPECTED_INPUT_ROWS,
    expected_runtime_binding_record_rows: int = EXPECTED_RUNTIME_BINDING_RECORD_ROWS,
    expected_citation_grade_store_rows: int = EXPECTED_CITATION_GRADE_STORE_ROWS,
    expected_strict_evidence_store_rows: int = EXPECTED_STRICT_EVIDENCE_STORE_ROWS,
    expected_eligibility_store_rows: int = EXPECTED_ELIGIBILITY_STORE_ROWS,
    expected_source_span_store_rows: int = EXPECTED_SOURCE_SPAN_STORE_ROWS,
) -> dict[str, Any]:
    apply_path = Path(str(apply_report_path)).expanduser()
    dry_run_path = Path(str(dry_run_report_path)).expanduser()
    contract_path = Path(str(runtime_binding_record_contract_report_path)).expanduser()
    papers_root = Path(str(papers_dir)).expanduser()
    requested_papers = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}

    warnings: list[str] = []
    schema_violations: list[str] = []

    apply_report = _read_json(apply_path)
    dry_run_report = _read_json(dry_run_path)
    contract_report = _read_json(contract_path)

    if not apply_report:
        schema_violations.append("apply_report_missing_or_unreadable")
    else:
        validation = validate_payload(
            apply_report,
            STRICT_EVIDENCE_RUNTIME_BINDING_EXECUTOR_APPLY_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            schema_violations.extend(f"apply_report:{error}" for error in validation.errors)
        if _safe_text(apply_report.get("status")) != "ok":
            schema_violations.append(
                f"apply_report_status={_safe_text(apply_report.get('status')) or 'unknown'}"
            )

    if not dry_run_report:
        schema_violations.append("dry_run_report_missing_or_unreadable")
    else:
        validation = validate_payload(
            dry_run_report,
            STRICT_EVIDENCE_RUNTIME_BINDING_EXECUTOR_APPLY_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            schema_violations.extend(f"dry_run_report:{error}" for error in validation.errors)

    if not contract_report:
        schema_violations.append("contract_report_missing_or_unreadable")
    else:
        validation = validate_payload(
            contract_report,
            STRICT_EVIDENCE_RUNTIME_BINDING_RECORD_CONTRACT_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            schema_violations.extend(f"contract_report:{error}" for error in validation.errors)

    apply_rows = [
        row
        for row in apply_report.get("rows", [])
        if isinstance(row, dict)
        and _safe_text(row.get("apply_status")) == APPLY_STATUS_APPLIED
    ] if apply_report else []

    if requested_papers:
        found = {_safe_text(row.get("paper_id")) for row in apply_rows if _safe_text(row.get("paper_id"))}
        if requested_papers - found:
            warnings.append("requested_paper_ids_not_found_in_apply_report")
        apply_rows = [row for row in apply_rows if _safe_text(row.get("paper_id")) in requested_papers]

    runtime_binding_root = _runtime_binding_store_root(papers_root)
    citation_grade_root = _citation_grade_store_root(papers_root)
    strict_evidence_root = _strict_evidence_store_root(papers_root)
    eligibility_root = _eligibility_store_root(papers_root)
    source_span_root = _source_span_store_root(papers_root)
    store_exists = runtime_binding_root.is_dir()

    runtime_binding_raw = _iter_jsonl_records(runtime_binding_root)
    if requested_papers:
        runtime_binding_raw = [
            item
            for item in runtime_binding_raw
            if _safe_text((item.get("record") or {}).get("paperId")) in requested_papers
        ]

    citation_grade_index = _load_index(citation_grade_root, "citationGradeRecordId")
    strict_evidence_index = _load_index(strict_evidence_root, "strictEvidenceId")
    eligibility_index = _load_index(eligibility_root, "eligibilityRecordId")
    source_span_index = _load_index(source_span_root, "sourceSpanId")

    idempotency_counts: Counter[str] = Counter()
    record_id_counts: Counter[str] = Counter()
    for item in runtime_binding_raw:
        record = item.get("record") if isinstance(item.get("record"), dict) else {}
        idempotency_key = _safe_text(record.get("idempotencyKey"))
        record_id = _safe_text(record.get("runtimeBindingRecordId"))
        if idempotency_key:
            idempotency_counts[idempotency_key] += 1
        if record_id:
            record_id_counts[record_id] += 1

    apply_record_by_id = {}
    for record in apply_report.get("runtimeBindingRecords", []) if apply_report else []:
        if isinstance(record, dict) and _safe_text(record.get("runtimeBindingRecordId")):
            apply_record_by_id[_safe_text(record.get("runtimeBindingRecordId"))] = record

    apply_by_runtime_binding_id = {
        _safe_text(row.get("runtimeBindingRecordId")): row
        for row in apply_rows
        if _safe_text(row.get("runtimeBindingRecordId"))
    }

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(runtime_binding_raw):
        record = dict(item.get("record") or {})
        strict_evidence_id = _safe_text(record.get("strictEvidenceId"))
        record_id = _safe_text(record.get("runtimeBindingRecordId"))
        apply_row = apply_by_runtime_binding_id.get(record_id, {})
        apply_record = apply_record_by_id.get(record_id, {})
        idempotency_key = _safe_text(record.get("idempotencyKey"))

        readback_status, blockers = _classify_runtime_binding_record(
            record,
            apply_record_index=apply_record_by_id,
            citation_grade_index=citation_grade_index,
            strict_evidence_index=strict_evidence_index,
            eligibility_index=eligibility_index,
            source_span_index=source_span_index,
            duplicate_idempotency=idempotency_counts.get(idempotency_key, 0) > 1,
            duplicate_record_id=record_id_counts.get(record_id, 0) > 1,
            store_exists=store_exists,
        )

        rows.append(
            {
                "readback_row_id": (
                    f"strict-evidence-runtime-binding-executor-apply-readback-review:{index:04d}"
                ),
                "apply_row_id": _safe_text(apply_row.get("apply_row_id")),
                "dry_run_row_id": _safe_text(apply_row.get("dry_run_row_id")),
                "policy_design_row_id": _safe_text(apply_row.get("policy_design_row_id")),
                "hold_row_id": _safe_text(apply_row.get("hold_row_id")),
                "strictEvidenceId": strict_evidence_id,
                "sourceSpanId": _safe_text(record.get("sourceSpanId")),
                "candidateRecordId": _safe_text(record.get("candidateRecordId")),
                "eligibilityRecordId": _safe_text(record.get("eligibilityRecordId")),
                "citationGradeRecordId": _safe_text(record.get("citationGradeRecordId")),
                "runtimeBindingRecordId": record_id,
                "idempotencyKey": idempotency_key,
                "paper_id": _safe_text(record.get("paperId")),
                "artifact_type": _safe_text(record.get("artifactType")),
                "runtime_binding_store_path": _safe_text(item.get("record_path")),
                "runtime_binding_store_line": int(item.get("record_line") or 0),
                "plannedWriteTarget": _safe_text(record.get("plannedWriteTarget")),
                "policyVersion": _safe_text(record.get("policyVersion")),
                "runtimeBindingDecision": _safe_text(record.get("runtimeBindingDecision")),
                "runtimeBindingState": _safe_text(record.get("runtimeBindingState")),
                "sourceContentHash": _safe_text(record.get("sourceContentHash")),
                "readback_status": readback_status,
                "readback_blockers": _dedupe(blockers),
                "runtimeBindingReadbackValidated": readback_status == READBACK_STATUS_VALIDATED,
                "writeMatrix": _no_mutation_policy_matrix(),
                "runtimeBindingRecordWriteRows": 0,
                "citationGradeRecordWriteRows": 0,
                "citationGradeBooleanMutationRows": 0,
                "eligibilityRecordWriteRows": 0,
                "strictEvidenceWriteRows": 0,
                "sourceSpanUpdatedRows": 0,
                "strictEligibleMutationRows": 0,
                "runtimeEvidence": False,
                "runtimeVisible": False,
                "answerIntegrationVisible": False,
                "recommended_action": (
                    "queue_for_strict_evidence_runtime_binding_post_apply_promotion_hold_review"
                    if readback_status == READBACK_STATUS_VALIDATED
                    else "repair_runtime_binding_record_before_post_apply_promotion"
                ),
            }
        )

    runtime_binding_record_rows = len(runtime_binding_raw)
    citation_grade_store_rows = _count_jsonl_rows(citation_grade_root)
    strict_evidence_store_rows = _count_jsonl_rows(strict_evidence_root)
    eligibility_store_rows = _count_jsonl_rows(eligibility_root)
    source_span_store_rows = _count_jsonl_rows(source_span_root)

    if runtime_binding_record_rows != expected_runtime_binding_record_rows:
        schema_violations.append(
            "runtime_binding_record_rows="
            f"{runtime_binding_record_rows}:expected={expected_runtime_binding_record_rows}"
        )
    if citation_grade_store_rows != expected_citation_grade_store_rows:
        schema_violations.append(
            "citation_grade_store_rows="
            f"{citation_grade_store_rows}:expected={expected_citation_grade_store_rows}"
        )
    if strict_evidence_store_rows != expected_strict_evidence_store_rows:
        schema_violations.append(
            "strict_evidence_store_rows="
            f"{strict_evidence_store_rows}:expected={expected_strict_evidence_store_rows}"
        )
    if eligibility_store_rows != expected_eligibility_store_rows:
        schema_violations.append(
            f"eligibility_store_rows={eligibility_store_rows}:expected={expected_eligibility_store_rows}"
        )
    if source_span_store_rows != expected_source_span_store_rows:
        schema_violations.append(
            f"source_span_store_rows={source_span_store_rows}:expected={expected_source_span_store_rows}"
        )

    counts = _count_rows(rows=rows, schema_violations=schema_violations, input_rows=len(apply_rows))
    counts["runtimeBindingRecordRows"] = runtime_binding_record_rows
    counts["citationGradeStoreRows"] = citation_grade_store_rows
    counts["strictEvidenceStoreRows"] = strict_evidence_store_rows
    counts["eligibilityStoreRows"] = eligibility_store_rows
    counts["sourceSpanStoreRows"] = source_span_store_rows

    readback_validated_rows = int(counts.get("readbackValidatedRows") or 0)
    status = "ok"
    if (
        schema_violations
        or len(apply_rows) != expected_input_rows
        or runtime_binding_record_rows != expected_runtime_binding_record_rows
        or citation_grade_store_rows != expected_citation_grade_store_rows
        or readback_validated_rows != expected_runtime_binding_record_rows
        or readback_validated_rows != len(rows)
    ):
        status = "blocked"

    return {
        "schema": STRICT_EVIDENCE_RUNTIME_BINDING_EXECUTOR_APPLY_READBACK_REVIEW_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "applyReportPath": str(apply_path),
            "applyReportSchema": _safe_text(apply_report.get("schema")) if apply_report else "",
            "applyReportStatus": _safe_text(apply_report.get("status")) if apply_report else "",
            "dryRunReportPath": str(dry_run_path),
            "dryRunReportSchema": _safe_text(dry_run_report.get("schema")) if dry_run_report else "",
            "contractReportPath": str(contract_path),
            "contractReportSchema": _safe_text(contract_report.get("schema")) if contract_report else "",
            "papersDir": str(papers_root),
            "requestedPaperIds": sorted(requested_papers),
            "expectedInputRows": expected_input_rows,
            "expectedRuntimeBindingRecordRows": expected_runtime_binding_record_rows,
            "expectedCitationGradeStoreRows": expected_citation_grade_store_rows,
            "expectedStrictEvidenceStoreRows": expected_strict_evidence_store_rows,
            "expectedEligibilityStoreRows": expected_eligibility_store_rows,
            "expectedSourceSpanStoreRows": expected_source_span_store_rows,
        },
        "counts": counts,
        "readbackOnlyPolicyMatrix": _no_mutation_policy_matrix(),
        "gate": {
            "readbackReviewReady": status == "ok",
            "runtimeBindingRecordWriteAllowed": False,
            "runtimeVisibleAllowed": False,
            "answerIntegrationVisibleAllowed": False,
            "citationGradeRecordWriteAllowed": False,
            "citationGradeBooleanMutationAllowed": False,
            "strictEligibleMutationAllowed": False,
            "eligibilityRecordWriteAllowed": False,
            "strictEvidenceStoreWriteAllowed": False,
            "sourceSpanStoreWriteAllowed": False,
            "runtimeEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": _dedupe(schema_violations),
            "decision": (
                "strict_evidence_runtime_binding_executor_apply_readback_review_ready"
                if status == "ok"
                else "strict_evidence_runtime_binding_executor_apply_readback_review_blocked"
            ),
            "recommendedNextTranche": (
                "strict_evidence_runtime_binding_post_apply_promotion_hold_review"
                if status == "ok"
                else "strict_evidence_runtime_binding_executor_apply_repair"
            ),
        },
        "policy": {
            "readbackOnly": True,
            **_no_mutation_policy_matrix(),
        },
        "warnings": _dedupe(warnings),
        "rows": rows,
    }


def _count_rows(
    *,
    rows: list[dict[str, Any]],
    schema_violations: list[str],
    input_rows: int,
) -> dict[str, Any]:
    by_status = Counter(_safe_text(row.get("readback_status")) for row in rows)
    return {
        "inputRows": input_rows,
        "runtimeBindingRecordRows": len(rows),
        "readbackValidatedRows": int(by_status.get(READBACK_STATUS_VALIDATED, 0)),
        "blockedMissingRuntimeBindingStoreRows": int(by_status.get(READBACK_STATUS_BLOCKED_MISSING_STORE, 0)),
        "blockedRuntimeBindingRecordSchemaViolationRows": int(by_status.get(READBACK_STATUS_BLOCKED_SCHEMA, 0)),
        "blockedRuntimeBindingRecordSemanticViolationRows": int(
            by_status.get(READBACK_STATUS_BLOCKED_SEMANTIC, 0)
        ),
        "blockedMissingStrictEvidenceReferenceRows": int(
            by_status.get(READBACK_STATUS_BLOCKED_MISSING_STRICT_EVIDENCE, 0)
        ),
        "blockedMissingEligibilityReferenceRows": int(
            by_status.get(READBACK_STATUS_BLOCKED_MISSING_ELIGIBILITY, 0)
        ),
        "blockedMissingSourceSpanReferenceRows": int(
            by_status.get(READBACK_STATUS_BLOCKED_MISSING_SOURCE_SPAN, 0)
        ),
        "blockedMissingCandidateRecordIdRows": int(by_status.get(READBACK_STATUS_BLOCKED_MISSING_CANDIDATE, 0)),
        "blockedCandidateIdentityMismatchRows": int(by_status.get(READBACK_STATUS_BLOCKED_IDENTITY, 0)),
        "blockedDuplicateIdempotencyKeyRows": int(by_status.get(READBACK_STATUS_BLOCKED_DUPLICATE_IDEMPOTENCY, 0)),
        "blockedDuplicateRuntimeBindingRecordIdRows": int(
            by_status.get(READBACK_STATUS_BLOCKED_DUPLICATE_RECORD_ID, 0)
        ),
        "blockedRuntimeOrAnswerFlagViolationRows": int(
            by_status.get(READBACK_STATUS_BLOCKED_RUNTIME_OR_ANSWER, 0)
        ),
        "blockedStoreRowCountChangedRows": int(by_status.get(READBACK_STATUS_BLOCKED_STORE_COUNT, 0)),
        "blockedInputSchemaViolationRows": int(by_status.get(READBACK_STATUS_BLOCKED_INPUT_SCHEMA, 0)),
        "runtimeBindingRecordWriteRows": 0,
        "citationGradeRecordWriteRows": 0,
        "citationGradeBooleanMutationRows": 0,
        "eligibilityRecordWriteRows": 0,
        "strictEvidenceWriteRows": 0,
        "sourceSpanUpdatedRows": 0,
        "strictEligibleMutationRows": 0,
        "runtimeEvidenceCreatedRows": 0,
        "runtimeVisibleRows": 0,
        "answerIntegrationVisibleRows": 0,
        "parserRoutingChangedRows": 0,
        "answerIntegrationChangedRows": 0,
        "databaseMutationRows": 0,
        "reindexOrReembedRows": 0,
        "manifestWriteRows": 0,
        "schemaViolationCount": len(schema_violations),
        "byPaperId": dict(
            Counter(_safe_text(row.get("paper_id")) for row in rows if row.get("runtimeBindingReadbackValidated"))
        ),
        "byArtifactType": dict(
            Counter(
                _safe_text(row.get("artifact_type"))
                for row in rows
                if row.get("runtimeBindingReadbackValidated")
            )
        ),
        "byReadbackStatus": dict(by_status),
        "byRecommendedAction": dict(Counter(_safe_text(row.get("recommended_action")) for row in rows)),
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
            "readbackOnlyPolicyMatrix",
            "gate",
            "policy",
            "warnings",
            "rows",
        )
        if key in report
    }


def render_strict_evidence_runtime_binding_executor_apply_readback_review_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byReadbackStatus") or {})).items())
    ]
    return "\n".join(
        [
            "# Strict Evidence Runtime Binding Executor Apply Readback Review",
            "",
            f"- status: {report.get('status', '')}",
            f"- input rows: {int(counts.get('inputRows') or 0)}",
            f"- runtime binding record rows: {int(counts.get('runtimeBindingRecordRows') or 0)}",
            f"- readback validated rows: {int(counts.get('readbackValidatedRows') or 0)}",
            f"- citation-grade store rows: {int(counts.get('citationGradeStoreRows') or 0)}",
            f"- strict evidence store rows: {int(counts.get('strictEvidenceStoreRows') or 0)}",
            f"- eligibility store rows: {int(counts.get('eligibilityStoreRows') or 0)}",
            f"- source span store rows: {int(counts.get('sourceSpanStoreRows') or 0)}",
            f"- runtime binding record writes: {int(counts.get('runtimeBindingRecordWriteRows') or 0)}",
            f"- runtime visible rows: {int(counts.get('runtimeVisibleRows') or 0)}",
            f"- answer integration visible rows: {int(counts.get('answerIntegrationVisibleRows') or 0)}",
            "",
            "## Readback status breakdown",
            *[f"- {item}" for item in by_status],
            "",
            f"- recommended next tranche: {report.get('gate', {}).get('recommendedNextTranche', '')}",
        ]
    )


def write_strict_evidence_runtime_binding_executor_apply_readback_review_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "strict-evidence-runtime-binding-executor-apply-readback-review.json"
    summary_path = root / "strict-evidence-runtime-binding-executor-apply-readback-review-summary.json"
    markdown_path = root / "strict-evidence-runtime-binding-executor-apply-readback-review.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_strict_evidence_runtime_binding_executor_apply_readback_review_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Read-only readback review for applied StrictEvidence runtime binding JSONL records."
        )
    )
    parser.add_argument(
        "--apply-report",
        default=str(DEFAULT_APPLY_REPORT_PATH),
        help="Path to runtime binding executor apply JSON report.",
    )
    parser.add_argument(
        "--dry-run-report",
        default=str(DEFAULT_DRY_RUN_REPORT_PATH),
        help="Path to runtime binding executor apply dry-run plan JSON report.",
    )
    parser.add_argument(
        "--contract-report",
        default=str(DEFAULT_CONTRACT_REPORT_PATH),
        help="Path to runtime binding record contract JSON report.",
    )
    parser.add_argument(
        "--papers-dir",
        default=str(DEFAULT_PAPERS_DIR),
        help=(
            "Local papers_dir root for runtime binding/strict_evidence/"
            "strict_evidence_eligibility/source_span JSONL stores."
        ),
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for JSON, summary, and markdown reports.",
    )
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_strict_evidence_runtime_binding_executor_apply_readback_review(
        apply_report_path=args.apply_report,
        dry_run_report_path=args.dry_run_report,
        runtime_binding_record_contract_report_path=args.contract_report,
        papers_dir=args.papers_dir,
        paper_ids=args.paper_id or None,
    )
    paths = write_strict_evidence_runtime_binding_executor_apply_readback_review_reports(
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
    "DEFAULT_APPLY_REPORT_PATH",
    "DEFAULT_CONTRACT_REPORT_PATH",
    "DEFAULT_DRY_RUN_REPORT_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PAPERS_DIR",
    "EXPECTED_CITATION_GRADE_STORE_ROWS",
    "EXPECTED_RUNTIME_BINDING_RECORD_ROWS",
    "EXPECTED_ELIGIBILITY_STORE_ROWS",
    "EXPECTED_INPUT_ROWS",
    "EXPECTED_SOURCE_SPAN_STORE_ROWS",
    "EXPECTED_STRICT_EVIDENCE_STORE_ROWS",
    "READBACK_STATUS_VALIDATED",
    "STRICT_EVIDENCE_RUNTIME_BINDING_EXECUTOR_APPLY_READBACK_REVIEW_SCHEMA_ID",
    "RUNTIME_BINDING_STORE",
    "build_strict_evidence_runtime_binding_executor_apply_readback_review",
    "render_strict_evidence_runtime_binding_executor_apply_readback_review_markdown",
    "write_strict_evidence_runtime_binding_executor_apply_readback_review_reports",
]
