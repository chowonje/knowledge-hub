"""Readback review for StrictEvidence pilot tranche run manifests.

Reconciles section and figure-caption manifest-only apply outcomes against the
99-row policy-gate candidate set. Read-only: does not write manifests, StrictEvidence
JSONL, SourceSpan JSONL, or mutate runtime/integration surfaces.
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
from knowledge_hub.papers.strict_evidence_figure_caption_pilot_executor_apply import (
    APPLY_STATUS_APPLIED as FIGURE_APPLY_STATUS_APPLIED,
    STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_EXECUTOR_APPLY_SCHEMA_ID,
)
from knowledge_hub.papers.strict_evidence_text_section_pilot_executor_apply import (
    APPLY_STATUS_APPLIED as SECTION_APPLY_STATUS_APPLIED,
    STRICT_EVIDENCE_TEXT_SECTION_PILOT_EXECUTOR_APPLY_SCHEMA_ID,
)


STRICT_EVIDENCE_PILOT_TRANCHE_MANIFEST_READBACK_REVIEW_SCHEMA_ID = (
    "knowledge-hub.paper.strict-evidence-pilot-tranche-manifest-readback-review.v1"
)

READBACK_STATUS_VALIDATED = "pilot_manifest_readback_validated"
READBACK_STATUS_BLOCKED_MANIFEST_MISSING = "blocked_manifest_missing"
READBACK_STATUS_BLOCKED_MANIFEST_SHAPE = "blocked_manifest_schema_or_shape_violation"
READBACK_STATUS_BLOCKED_MISSING_POLICY = "blocked_missing_policy_candidate"
READBACK_STATUS_BLOCKED_UNEXPECTED_MANIFEST = "blocked_unexpected_manifest_row"
READBACK_STATUS_BLOCKED_DUPLICATE_ID = "blocked_duplicate_strict_evidence_id"
READBACK_STATUS_BLOCKED_ARTIFACT_MISMATCH = "blocked_artifact_type_mismatch"
READBACK_STATUS_BLOCKED_RUNTIME_OR_CITATION = "blocked_runtime_or_citation_flag_violation"
READBACK_STATUS_BLOCKED_STORE_COUNT = "blocked_store_row_count_changed"
READBACK_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"

MANIFEST_TYPE_SECTION = "strict_evidence_text_section_pilot_executor_apply"
MANIFEST_TYPE_FIGURE_CAPTION = "strict_evidence_figure_caption_pilot_executor_apply"

EXPECTED_SECTION_MANIFEST_ROWS = 45
EXPECTED_FIGURE_CAPTION_MANIFEST_ROWS = 54
EXPECTED_POLICY_CANDIDATE_ROWS = 99
EXPECTED_STRICT_EVIDENCE_STORE_ROWS = 99
EXPECTED_SOURCE_SPAN_STORE_ROWS = 102

DEFAULT_SECTION_APPLY_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "strict-evidence-text-section-pilot-executor-apply"
    / "02-strict-evidence-text-section-pilot-executor-apply"
    / "strict-evidence-text-section-pilot-executor-apply.json"
)

DEFAULT_FIGURE_CAPTION_APPLY_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "strict-evidence-figure-caption-pilot-executor-apply"
    / "02-strict-evidence-figure-caption-pilot-executor-apply"
    / "strict-evidence-figure-caption-pilot-executor-apply.json"
)

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

DEFAULT_PAPERS_DIR = Path.home() / ".khub" / "papers"

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "strict-evidence-pilot-tranche-manifest-readback-review"
    / "01-strict-evidence-pilot-tranche-manifest-readback-review"
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


def _strict_evidence_store_root(papers_dir: str | Path) -> Path:
    return Path(str(papers_dir)).expanduser() / "structured_evidence" / "strict_evidence"


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


def _no_mutation_policy_matrix() -> dict[str, Any]:
    return {
        "readbackOnly": True,
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
    checks = {
        "strictEligible": _safe_bool(row.get("strictEligible")),
        "strictEvidenceCreated": _safe_bool(row.get("strictEvidenceCreated")),
        "citationGrade": _safe_bool(row.get("citationGrade")),
        "runtimeEvidence": _safe_bool(row.get("runtimeEvidence")),
        "parserRoutingChanged": _safe_bool(row.get("parserRoutingChanged")),
        "answerIntegrationChanged": _safe_bool(row.get("answerIntegrationChanged")),
        "databaseMutation": _safe_bool(row.get("databaseMutation")),
    }
    for field_name, enabled in checks.items():
        if enabled:
            violations.append(f"{field_name}_true")
    write_matrix = row.get("writeMatrix")
    if isinstance(write_matrix, dict):
        if _safe_bool(write_matrix.get("strictEvidenceStoreWrite")):
            violations.append("writeMatrix.strictEvidenceStoreWrite_true")
        if _safe_bool(write_matrix.get("sourceSpanStoreWrite")):
            violations.append("writeMatrix.sourceSpanStoreWrite_true")
        if _safe_bool(write_matrix.get("writeEnabled")):
            violations.append("writeMatrix.writeEnabled_true")
    return _dedupe(violations)


def _resolve_manifest_path(apply_report: dict[str, Any]) -> str:
    gate = apply_report.get("gate") if isinstance(apply_report.get("gate"), dict) else {}
    return _safe_text(gate.get("runManifestPath"))


def _active_manifest_rows(manifest: dict[str, Any], *, applied_status: str) -> list[dict[str, Any]]:
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        return []
    active: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if _safe_text(row.get("apply_status")) == applied_status:
            active.append(row)
    return active


def _policy_candidate_rows(policy_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = policy_report.get("rows")
    if not isinstance(rows, list):
        return []
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if _safe_text(row.get("policy_gate_status")) == POLICY_STATUS_CANDIDATE_ONLY:
            candidates.append(row)
    return candidates


def _manifest_index(
    rows: list[dict[str, Any]],
    *,
    manifest_type: str,
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        strict_evidence_id = _safe_text(row.get("strictEvidenceId"))
        if not strict_evidence_id:
            continue
        index[strict_evidence_id] = {
            **row,
            "manifestType": manifest_type,
        }
    return index


def _count_readback_rows(
    rows: list[dict[str, Any]],
    *,
    policy_candidate_rows: int,
    input_schema_violations: list[str],
    section_manifest_rows: int,
    figure_manifest_rows: int,
    strict_evidence_store_rows: int,
    source_span_store_rows: int,
) -> dict[str, Any]:
    by_status = Counter(_safe_text(row.get("readback_status")) for row in rows)
    by_paper = Counter(
        _safe_text(row.get("paper_id")) for row in rows if row.get("readback_status") == READBACK_STATUS_VALIDATED
    )
    by_artifact = Counter(
        _safe_text(row.get("artifact_type"))
        for row in rows
        if row.get("readback_status") == READBACK_STATUS_VALIDATED
    )
    by_manifest_type = Counter(
        _safe_text(row.get("manifestType"))
        for row in rows
        if row.get("readback_status") == READBACK_STATUS_VALIDATED
    )
    by_recommended = Counter(_safe_text(row.get("recommended_action")) for row in rows)

    return {
        "inputPolicyCandidateRows": policy_candidate_rows,
        "sectionManifestRows": section_manifest_rows,
        "figureCaptionManifestRows": figure_manifest_rows,
        "combinedManifestRows": section_manifest_rows + figure_manifest_rows,
        "pilotManifestReadbackValidatedRows": int(by_status.get(READBACK_STATUS_VALIDATED, 0)),
        "missingPolicyCandidateRows": int(by_status.get(READBACK_STATUS_BLOCKED_MISSING_POLICY, 0)),
        "unexpectedManifestRows": int(by_status.get(READBACK_STATUS_BLOCKED_UNEXPECTED_MANIFEST, 0)),
        "duplicateStrictEvidenceIdRows": int(by_status.get(READBACK_STATUS_BLOCKED_DUPLICATE_ID, 0)),
        "blockedManifestMissingRows": int(by_status.get(READBACK_STATUS_BLOCKED_MANIFEST_MISSING, 0)),
        "blockedManifestSchemaOrShapeViolationRows": int(
            by_status.get(READBACK_STATUS_BLOCKED_MANIFEST_SHAPE, 0)
        ),
        "blockedArtifactTypeMismatchRows": int(
            by_status.get(READBACK_STATUS_BLOCKED_ARTIFACT_MISMATCH, 0)
        ),
        "blockedRuntimeOrCitationFlagViolationRows": int(
            by_status.get(READBACK_STATUS_BLOCKED_RUNTIME_OR_CITATION, 0)
        ),
        "blockedStoreRowCountChangedRows": int(by_status.get(READBACK_STATUS_BLOCKED_STORE_COUNT, 0)),
        "blockedInputSchemaViolationRows": int(by_status.get(READBACK_STATUS_BLOCKED_INPUT_SCHEMA, 0)),
        "strictEvidenceStoreRows": strict_evidence_store_rows,
        "sourceSpanStoreRows": source_span_store_rows,
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
        "byPaperId": dict(by_paper),
        "byArtifactType": dict(by_artifact),
        "byManifestType": dict(by_manifest_type),
        "byReadbackStatus": dict(by_status),
        "byRecommendedAction": dict(by_recommended),
    }


def _readback_rows(
    *,
    policy_rows: list[dict[str, Any]],
    section_index: dict[str, dict[str, Any]],
    figure_index: dict[str, dict[str, Any]],
    input_schema_violations: list[str],
    store_count_blocked: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    duplicate_ids: list[str] = []
    missing_ids: list[str] = []
    unexpected_ids: list[str] = []

    section_ids = set(section_index)
    figure_ids = set(figure_index)
    duplicate_ids = sorted(section_ids & figure_ids)
    policy_ids = {
        _safe_text(row.get("strictEvidenceId"))
        for row in policy_rows
        if _safe_text(row.get("strictEvidenceId"))
    }
    manifest_ids = section_ids | figure_ids
    missing_ids = sorted(policy_ids - manifest_ids)
    unexpected_ids = sorted(manifest_ids - policy_ids)

    for index, policy_row in enumerate(policy_rows):
        strict_evidence_id = _safe_text(policy_row.get("strictEvidenceId"))
        artifact_type = _safe_text(policy_row.get("artifact_type"))
        paper_id = _safe_text(policy_row.get("paper_id"))
        blockers: list[str] = []
        readback_status = READBACK_STATUS_VALIDATED
        manifest_type = ""
        recommended_action = "pilot_manifest_readback_validated"

        if input_schema_violations:
            readback_status = READBACK_STATUS_BLOCKED_INPUT_SCHEMA
            blockers.extend(input_schema_violations)
            recommended_action = "blocked_input_schema_violation"
        elif store_count_blocked:
            readback_status = READBACK_STATUS_BLOCKED_STORE_COUNT
            blockers.append("store_row_count_mismatch")
            recommended_action = "blocked_store_row_count_changed"
        elif strict_evidence_id in duplicate_ids:
            readback_status = READBACK_STATUS_BLOCKED_DUPLICATE_ID
            blockers.append("duplicate_strict_evidence_id_across_manifests")
            recommended_action = "blocked_duplicate_strict_evidence_id"
        elif strict_evidence_id in missing_ids:
            readback_status = READBACK_STATUS_BLOCKED_MISSING_POLICY
            blockers.append("missing_from_both_pilot_manifests")
            recommended_action = "blocked_missing_policy_candidate"
        else:
            in_section = strict_evidence_id in section_index
            in_figure = strict_evidence_id in figure_index
            manifest_row: dict[str, Any] = {}
            if in_section and in_figure:
                readback_status = READBACK_STATUS_BLOCKED_DUPLICATE_ID
                blockers.append("duplicate_strict_evidence_id_across_manifests")
                recommended_action = "blocked_duplicate_strict_evidence_id"
            elif artifact_type == "section" and in_section:
                manifest_type = MANIFEST_TYPE_SECTION
                manifest_row = section_index[strict_evidence_id]
                if _safe_text(manifest_row.get("artifact_type")) != "section":
                    readback_status = READBACK_STATUS_BLOCKED_ARTIFACT_MISMATCH
                    blockers.append("section_policy_row_not_section_in_manifest")
                    recommended_action = "blocked_artifact_type_mismatch"
            elif artifact_type == "figure" and in_figure:
                manifest_type = MANIFEST_TYPE_FIGURE_CAPTION
                manifest_row = figure_index[strict_evidence_id]
                if _safe_text(manifest_row.get("artifact_type")) != "figure":
                    readback_status = READBACK_STATUS_BLOCKED_ARTIFACT_MISMATCH
                    blockers.append("figure_policy_row_not_figure_in_manifest")
                    recommended_action = "blocked_artifact_type_mismatch"
            elif artifact_type == "section":
                readback_status = READBACK_STATUS_BLOCKED_MISSING_POLICY
                blockers.append("section_policy_row_missing_from_section_manifest")
                recommended_action = "blocked_missing_policy_candidate"
            elif artifact_type == "figure":
                readback_status = READBACK_STATUS_BLOCKED_MISSING_POLICY
                blockers.append("figure_policy_row_missing_from_figure_manifest")
                recommended_action = "blocked_missing_policy_candidate"
            else:
                readback_status = READBACK_STATUS_BLOCKED_ARTIFACT_MISMATCH
                blockers.append(f"unsupported_artifact_type={artifact_type or 'missing'}")
                recommended_action = "blocked_artifact_type_mismatch"

            if readback_status == READBACK_STATUS_VALIDATED and manifest_row:
                flag_violations = _mutation_flag_violation(manifest_row)
                if flag_violations:
                    readback_status = READBACK_STATUS_BLOCKED_RUNTIME_OR_CITATION
                    blockers.extend(flag_violations)
                    recommended_action = "blocked_runtime_or_citation_flag_violation"

        rows.append(
            {
                "readback_row_id": (
                    f"strict-evidence-pilot-tranche-manifest-readback-review:{index:04d}"
                ),
                "policy_gate_row_id": _safe_text(policy_row.get("policy_gate_row_id")),
                "strictEvidenceId": strict_evidence_id,
                "sourceSpanId": _safe_text(policy_row.get("sourceSpanId")),
                "candidateRecordId": _safe_text(policy_row.get("candidateRecordId")),
                "paper_id": paper_id,
                "artifact_type": artifact_type,
                "manifestType": manifest_type,
                "readback_status": readback_status,
                "readback_blockers": _dedupe(blockers),
                "policy_gate_status": _safe_text(policy_row.get("policy_gate_status")),
                "sectionManifestFound": strict_evidence_id in section_index,
                "figureCaptionManifestFound": strict_evidence_id in figure_index,
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

    for strict_evidence_id in unexpected_ids:
        manifest_type = (
            MANIFEST_TYPE_SECTION
            if strict_evidence_id in section_index
            else MANIFEST_TYPE_FIGURE_CAPTION
        )
        manifest_row = section_index.get(strict_evidence_id) or figure_index.get(strict_evidence_id) or {}
        rows.append(
            {
                "readback_row_id": (
                    f"strict-evidence-pilot-tranche-manifest-readback-review:unexpected:{strict_evidence_id}"
                ),
                "policy_gate_row_id": "",
                "strictEvidenceId": strict_evidence_id,
                "sourceSpanId": _safe_text(manifest_row.get("sourceSpanId")),
                "candidateRecordId": _safe_text(manifest_row.get("candidateRecordId")),
                "paper_id": _safe_text(manifest_row.get("paper_id")),
                "artifact_type": _safe_text(manifest_row.get("artifact_type")),
                "manifestType": manifest_type,
                "readback_status": READBACK_STATUS_BLOCKED_UNEXPECTED_MANIFEST,
                "readback_blockers": ["unexpected_manifest_row_outside_policy_candidates"],
                "policy_gate_status": "",
                "sectionManifestFound": strict_evidence_id in section_index,
                "figureCaptionManifestFound": strict_evidence_id in figure_index,
                "strictEligible": False,
                "strictEvidenceCreated": False,
                "citationGrade": False,
                "runtimeEvidence": False,
                "parserRoutingChanged": False,
                "answerIntegrationChanged": False,
                "databaseMutation": False,
                "recommended_action": "blocked_unexpected_manifest_row",
            }
        )

    diagnostics = {
        "duplicateStrictEvidenceIds": duplicate_ids,
        "missingPolicyCandidateIds": missing_ids,
        "unexpectedManifestRowIds": unexpected_ids,
    }
    return rows, diagnostics


def build_strict_evidence_pilot_tranche_manifest_readback_review(
    *,
    section_apply_report_path: str | Path = DEFAULT_SECTION_APPLY_REPORT_PATH,
    figure_caption_apply_report_path: str | Path = DEFAULT_FIGURE_CAPTION_APPLY_REPORT_PATH,
    policy_gate_report_path: str | Path = DEFAULT_POLICY_GATE_REPORT_PATH,
    papers_dir: str | Path = DEFAULT_PAPERS_DIR,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    section_apply_path = Path(str(section_apply_report_path)).expanduser()
    figure_apply_path = Path(str(figure_caption_apply_report_path)).expanduser()
    policy_gate_path = Path(str(policy_gate_report_path)).expanduser()
    papers_root = Path(str(papers_dir)).expanduser()

    warnings: list[str] = []
    input_schema_violations: list[str] = []
    requested_papers = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}

    section_apply_report = _read_json(section_apply_path)
    figure_apply_report = _read_json(figure_apply_path)
    policy_gate_report = _read_json(policy_gate_path)

    if not section_apply_report:
        input_schema_violations.append("section_apply_report_missing_or_unreadable")
    if not figure_apply_report:
        input_schema_violations.append("figure_caption_apply_report_missing_or_unreadable")
    if not policy_gate_report:
        input_schema_violations.append("policy_gate_report_missing_or_unreadable")

    for label, report, schema_id in (
        ("section_apply", section_apply_report, STRICT_EVIDENCE_TEXT_SECTION_PILOT_EXECUTOR_APPLY_SCHEMA_ID),
        (
            "figure_caption_apply",
            figure_apply_report,
            STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_EXECUTOR_APPLY_SCHEMA_ID,
        ),
        ("policy_gate", policy_gate_report, PARSED_ARTIFACT_STRICT_EVIDENCE_POLICY_GATE_SCHEMA_ID),
    ):
        if not report:
            continue
        validation = validate_payload(report, schema_id, strict=True)
        if not validation.ok:
            input_schema_violations.extend(f"{label}:{error}" for error in validation.errors)
        if _safe_text(report.get("status")) != "ok":
            input_schema_violations.append(
                f"{label}_report_status={_safe_text(report.get('status')) or 'unknown'}"
            )

    section_manifest_path = _resolve_manifest_path(section_apply_report)
    figure_manifest_path = _resolve_manifest_path(figure_apply_report)
    if not section_manifest_path:
        input_schema_violations.append("section_run_manifest_path_missing")
    if not figure_manifest_path:
        input_schema_violations.append("figure_caption_run_manifest_path_missing")

    section_manifest = _read_json(section_manifest_path)
    figure_manifest = _read_json(figure_manifest_path)

    if section_manifest_path and not section_manifest:
        input_schema_violations.append("section_run_manifest_missing_or_unreadable")
    if figure_manifest_path and not figure_manifest:
        input_schema_violations.append("figure_caption_run_manifest_missing_or_unreadable")

    for label, manifest, schema_id in (
        ("section_manifest", section_manifest, STRICT_EVIDENCE_TEXT_SECTION_PILOT_EXECUTOR_APPLY_SCHEMA_ID),
        (
            "figure_caption_manifest",
            figure_manifest,
            STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_EXECUTOR_APPLY_SCHEMA_ID,
        ),
    ):
        if not manifest:
            continue
        validation = validate_payload(manifest, schema_id, strict=True)
        if not validation.ok:
            input_schema_violations.extend(f"{label}:{error}" for error in validation.errors)
        if _safe_text(manifest.get("status")) != "ok":
            input_schema_violations.append(
                f"{label}_status={_safe_text(manifest.get('status')) or 'unknown'}"
            )

    section_active_rows = _active_manifest_rows(
        section_manifest,
        applied_status=SECTION_APPLY_STATUS_APPLIED,
    )
    figure_active_rows = _active_manifest_rows(
        figure_manifest,
        applied_status=FIGURE_APPLY_STATUS_APPLIED,
    )

    if (
        section_manifest
        and len(section_active_rows) != EXPECTED_SECTION_MANIFEST_ROWS
    ):
        input_schema_violations.append(
            f"section_manifest_active_rows={len(section_active_rows)}_expected_{EXPECTED_SECTION_MANIFEST_ROWS}"
        )
    if (
        figure_manifest
        and len(figure_active_rows) != EXPECTED_FIGURE_CAPTION_MANIFEST_ROWS
    ):
        input_schema_violations.append(
            "figure_caption_manifest_active_rows="
            f"{len(figure_active_rows)}_expected_{EXPECTED_FIGURE_CAPTION_MANIFEST_ROWS}"
        )

    policy_rows = _policy_candidate_rows(policy_gate_report)
    if policy_gate_report and len(policy_rows) != EXPECTED_POLICY_CANDIDATE_ROWS:
        input_schema_violations.append(
            f"policy_candidate_rows={len(policy_rows)}_expected_{EXPECTED_POLICY_CANDIDATE_ROWS}"
        )

    if requested_papers:
        policy_rows = [row for row in policy_rows if _safe_text(row.get("paper_id")) in requested_papers]
        section_active_rows = [
            row for row in section_active_rows if _safe_text(row.get("paper_id")) in requested_papers
        ]
        figure_active_rows = [
            row for row in figure_active_rows if _safe_text(row.get("paper_id")) in requested_papers
        ]

    strict_evidence_store_rows = _count_jsonl_rows(_strict_evidence_store_root(papers_root))
    source_span_store_rows = _count_jsonl_rows(_source_span_store_root(papers_root))
    store_count_blocked = (
        strict_evidence_store_rows != EXPECTED_STRICT_EVIDENCE_STORE_ROWS
        or source_span_store_rows != EXPECTED_SOURCE_SPAN_STORE_ROWS
    )
    if store_count_blocked:
        warnings.append("store_row_count_mismatch")

    input_schema_violations = _dedupe(input_schema_violations)
    section_index = _manifest_index(section_active_rows, manifest_type=MANIFEST_TYPE_SECTION)
    figure_index = _manifest_index(figure_active_rows, manifest_type=MANIFEST_TYPE_FIGURE_CAPTION)

    rows, diagnostics = _readback_rows(
        policy_rows=policy_rows,
        section_index=section_index,
        figure_index=figure_index,
        input_schema_violations=input_schema_violations,
        store_count_blocked=store_count_blocked and not requested_papers,
    )

    counts = _count_readback_rows(
        rows,
        policy_candidate_rows=len(policy_rows),
        input_schema_violations=input_schema_violations,
        section_manifest_rows=len(section_active_rows),
        figure_manifest_rows=len(figure_active_rows),
        strict_evidence_store_rows=strict_evidence_store_rows,
        source_span_store_rows=source_span_store_rows,
    )

    validated_rows = int(counts.get("pilotManifestReadbackValidatedRows") or 0)
    status = "ok"
    if (
        input_schema_violations
        or validated_rows != EXPECTED_POLICY_CANDIDATE_ROWS
        or int(counts.get("missingPolicyCandidateRows") or 0)
        or int(counts.get("unexpectedManifestRows") or 0)
        or int(counts.get("duplicateStrictEvidenceIdRows") or 0)
        or store_count_blocked
    ):
        status = "blocked"

    policy_matrix = _no_mutation_policy_matrix()
    report = {
        "schema": STRICT_EVIDENCE_PILOT_TRANCHE_MANIFEST_READBACK_REVIEW_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "sectionApplyReportPath": str(section_apply_path),
            "sectionApplyReportSchema": _safe_text(section_apply_report.get("schema")),
            "sectionApplyReportStatus": _safe_text(section_apply_report.get("status")),
            "figureCaptionApplyReportPath": str(figure_apply_path),
            "figureCaptionApplyReportSchema": _safe_text(figure_apply_report.get("schema")),
            "figureCaptionApplyReportStatus": _safe_text(figure_apply_report.get("status")),
            "policyGateReportPath": str(policy_gate_path),
            "policyGateReportSchema": _safe_text(policy_gate_report.get("schema")),
            "policyGateReportStatus": _safe_text(policy_gate_report.get("status")),
            "papersDir": str(papers_root),
            "strictEvidenceStoreRoot": str(_strict_evidence_store_root(papers_root)),
            "sourceSpanStoreRoot": str(_source_span_store_root(papers_root)),
            "requestedPaperIds": sorted(requested_papers),
            "sectionRunManifestPath": section_manifest_path,
            "figureCaptionRunManifestPath": figure_manifest_path,
            "expectedPolicyCandidateRows": EXPECTED_POLICY_CANDIDATE_ROWS,
            "expectedSectionManifestRows": EXPECTED_SECTION_MANIFEST_ROWS,
            "expectedFigureCaptionManifestRows": EXPECTED_FIGURE_CAPTION_MANIFEST_ROWS,
            "expectedStrictEvidenceStoreRows": EXPECTED_STRICT_EVIDENCE_STORE_ROWS,
            "expectedSourceSpanStoreRows": EXPECTED_SOURCE_SPAN_STORE_ROWS,
        },
        "counts": counts,
        "diagnostics": diagnostics,
        "noMutationPolicyMatrix": policy_matrix,
        "gate": {
            "pilotManifestReadbackReviewReady": status == "ok",
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
            "schemaViolations": input_schema_violations,
            "decision": (
                "strict_evidence_pilot_tranche_manifest_readback_validated"
                if status == "ok"
                else "strict_evidence_pilot_tranche_manifest_readback_blocked"
            ),
            "recommendedNextTranche": (
                "strict_evidence_pilot_tranche_completion_gate"
                if status == "ok"
                else "strict_evidence_pilot_tranche_manifest_readback_review_repair"
            ),
        },
        "policy": {
            "reportOnly": True,
            "readbackOnly": True,
            **policy_matrix,
        },
        "warnings": _dedupe(warnings),
        "rows": rows,
    }
    return report


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in (
            "schema",
            "status",
            "generatedAt",
            "input",
            "counts",
            "diagnostics",
            "noMutationPolicyMatrix",
            "gate",
            "policy",
            "warnings",
        )
        if key in report
    }


def render_strict_evidence_pilot_tranche_manifest_readback_review_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    diagnostics = dict(report.get("diagnostics") or {})
    matrix = dict(report.get("noMutationPolicyMatrix") or {})
    input_section = dict(report.get("input") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byReadbackStatus") or {})).items())
    ]
    return "\n".join(
        [
            "# Strict Evidence Pilot Tranche Manifest Readback Review",
            "",
            f"- status: {report.get('status', '')}",
            f"- policy candidate rows: {int(counts.get('inputPolicyCandidateRows') or 0)}",
            f"- section manifest rows: {int(counts.get('sectionManifestRows') or 0)}",
            f"- figure caption manifest rows: {int(counts.get('figureCaptionManifestRows') or 0)}",
            f"- combined manifest rows: {int(counts.get('combinedManifestRows') or 0)}",
            f"- readback validated rows: {int(counts.get('pilotManifestReadbackValidatedRows') or 0)}",
            f"- strict evidence store rows: {int(counts.get('strictEvidenceStoreRows') or 0)}",
            f"- source span store rows: {int(counts.get('sourceSpanStoreRows') or 0)}",
            "",
            "## Manifest paths",
            f"- section apply report: {input_section.get('sectionApplyReportPath', '')}",
            f"- section run manifest: {input_section.get('sectionRunManifestPath', '')}",
            f"- figure caption apply report: {input_section.get('figureCaptionApplyReportPath', '')}",
            f"- figure caption run manifest: {input_section.get('figureCaptionRunManifestPath', '')}",
            "",
            "## Diagnostics",
            f"- duplicate strict evidence ids: {json.dumps(diagnostics.get('duplicateStrictEvidenceIds') or [])}",
            f"- missing policy candidate ids: {json.dumps(diagnostics.get('missingPolicyCandidateIds') or [])}",
            f"- unexpected manifest row ids: {json.dumps(diagnostics.get('unexpectedManifestRowIds') or [])}",
            "",
            "## No-mutation policy matrix",
            f"- readback only: {json.dumps(matrix.get('readbackOnly'))}",
            f"- strict evidence store write: {json.dumps(matrix.get('strictEvidenceStoreWrite'))}",
            f"- source span store write: {json.dumps(matrix.get('sourceSpanStoreWrite'))}",
            "",
            "## Readback status breakdown",
            *[f"- {item}" for item in by_status],
            "",
            f"- recommended next tranche: {report.get('gate', {}).get('recommendedNextTranche', '')}",
        ]
    )


def write_strict_evidence_pilot_tranche_manifest_readback_review_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "strict-evidence-pilot-tranche-manifest-readback-review.json"
    summary_path = root / "strict-evidence-pilot-tranche-manifest-readback-review-summary.json"
    markdown_path = root / "strict-evidence-pilot-tranche-manifest-readback-review.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_strict_evidence_pilot_tranche_manifest_readback_review_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Read back and reconcile StrictEvidence pilot tranche run manifests "
            "against the policy-gate candidate set without mutating stores."
        )
    )
    parser.add_argument(
        "--section-apply-report",
        default=str(DEFAULT_SECTION_APPLY_REPORT_PATH),
        help="Path to the text-section pilot executor apply JSON report.",
    )
    parser.add_argument(
        "--figure-caption-apply-report",
        default=str(DEFAULT_FIGURE_CAPTION_APPLY_REPORT_PATH),
        help="Path to the figure-caption pilot executor apply JSON report.",
    )
    parser.add_argument(
        "--policy-gate-report",
        default=str(DEFAULT_POLICY_GATE_REPORT_PATH),
        help="Path to the parsed-artifact StrictEvidence policy gate JSON report.",
    )
    parser.add_argument(
        "--papers-dir",
        default=str(DEFAULT_PAPERS_DIR),
        help="Local papers_dir root used to count strict_evidence and source_span JSONL rows.",
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for JSON, summary, and markdown reports.",
    )
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_strict_evidence_pilot_tranche_manifest_readback_review(
        section_apply_report_path=args.section_apply_report,
        figure_caption_apply_report_path=args.figure_caption_apply_report,
        policy_gate_report_path=args.policy_gate_report,
        papers_dir=args.papers_dir,
        paper_ids=args.paper_id or None,
    )
    paths = write_strict_evidence_pilot_tranche_manifest_readback_review_reports(
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
    "DEFAULT_FIGURE_CAPTION_APPLY_REPORT_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_POLICY_GATE_REPORT_PATH",
    "DEFAULT_SECTION_APPLY_REPORT_PATH",
    "READBACK_STATUS_VALIDATED",
    "STRICT_EVIDENCE_PILOT_TRANCHE_MANIFEST_READBACK_REVIEW_SCHEMA_ID",
    "build_strict_evidence_pilot_tranche_manifest_readback_review",
    "render_strict_evidence_pilot_tranche_manifest_readback_review_markdown",
    "write_strict_evidence_pilot_tranche_manifest_readback_review_reports",
]
