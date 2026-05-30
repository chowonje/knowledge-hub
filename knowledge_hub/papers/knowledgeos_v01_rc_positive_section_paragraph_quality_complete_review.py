"""Complete review for the KnowledgeOS v0.1 RC positive section/paragraph QA slice."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.knowledgeos_v01_rc_corpus_scale_answer_quality_gate_refresh_after_positive_answer_execution import (
    KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_GATE_REFRESH_AFTER_POSITIVE_ANSWER_EXECUTION_SCHEMA_ID,
    READY_DECISION as PROVENANCE_REFRESH_READY_DECISION,
)
from knowledge_hub.papers.knowledgeos_v01_rc_corpus_scale_answer_quality_positive_answer_execution_gate import (
    KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_POSITIVE_ANSWER_EXECUTION_GATE_SCHEMA_ID,
    READY_DECISION as POSITIVE_EXECUTION_READY_DECISION,
)
from knowledge_hub.papers.knowledgeos_v01_rc_corpus_scale_answer_quality_positive_section_paragraph_seed import (
    KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_POSITIVE_SECTION_PARAGRAPH_SEED_SCHEMA_ID,
    READY_DECISION as POSITIVE_SEED_READY_DECISION,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed import (
    ZERO_COUNTER_FIELDS,
    _clean_text,
    _contains_private_path,
    _int,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture import (
    _read_json,
)


KNOWLEDGEOS_V01_RC_POSITIVE_SECTION_PARAGRAPH_QUALITY_COMPLETE_REVIEW_SCHEMA_ID = (
    "knowledge-hub.product.knowledgeos-v01-rc-positive-section-paragraph-quality-complete-review.v1"
)

READY_DECISION = "knowledgeos_v01_rc_positive_section_paragraph_quality_complete_review_ready"
BLOCKED_DECISION = "knowledgeos_v01_rc_positive_section_paragraph_quality_complete_review_blocked"
NEXT_TRANCHE_READY = "knowledgeos_v01_rc_research_preview_release_readiness_decision_gate"
NEXT_TRANCHE_BLOCKED = "knowledgeos_v01_rc_positive_section_paragraph_quality_complete_review_repair"

DEFAULT_POSITIVE_SEED_REPORT = Path(
    "eval/knowledgeos/reports/knowledgeos_v01_rc_corpus_scale_answer_quality_positive_section_paragraph_seed.v1.json"
)
DEFAULT_POSITIVE_EXECUTION_REPORT = Path(
    "eval/knowledgeos/reports/knowledgeos_v01_rc_corpus_scale_answer_quality_positive_answer_execution_gate.v1.json"
)
DEFAULT_PROVENANCE_REFRESH_REPORT = Path(
    "eval/knowledgeos/reports/knowledgeos_v01_rc_corpus_scale_answer_quality_gate_refresh_after_positive_answer_execution.v1.json"
)

EXPECTED_PR_REFS = (
    {
        "prNumber": 180,
        "headBranch": "codex/corpus-scale-answer-quality-positive-section-paragraph-seed-20260530-20260530",
        "phase": "positive_section_paragraph_seed",
    },
    {
        "prNumber": 181,
        "headBranch": "codex/corpus-scale-answer-quality-positive-answer-execution-gate-20260530-20260530",
        "phase": "positive_answer_execution",
    },
    {
        "prNumber": 182,
        "headBranch": "codex/corpus-scale-answer-quality-positive-provenance-refresh-20260530-20260530",
        "phase": "positive_provenance_refresh",
    },
)
DEFAULT_BASE_BRANCH = "main"
EXPECTED_CASE_ROWS = 50
EXPECTED_POSITIVE_ROWS = 7
EXPECTED_EXPECTED_NO_ANSWER_ROWS = 17
EXPECTED_STRUCTURED_MODALITY_ROWS = 34
EXPECTED_STRICT_PROVENANCE_ROWS = 20

EXTRA_ZERO_COUNTER_FIELDS = (
    "externalLlmCallRows",
    "modelApiCallRows",
    "judgeModelCallRows",
    "githubPrMutationRows",
    "mergeRows",
    "branchDeletionRows",
    "releaseTagRows",
    "packagePublishRows",
    "rawPayloadPersistedRows",
    "rawGithubPayloadPersistedRows",
    "defaultMcpToolRows",
    "defaultKhubAskRouteRows",
)

SELF_REVIEW_ALLOWED_DIRTY_PATHS = {
    "CHANGELOG.md",
    "docs/PROJECT_STATE.md",
    "docs/schemas/knowledgeos-v01-rc-positive-section-paragraph-quality-complete-review.v1.json",
    "eval/knowledgeos/reports/knowledgeos_v01_rc_positive_section_paragraph_quality_complete_review.v1.json",
    "eval/knowledgeos/reports/knowledgeos_v01_rc_positive_section_paragraph_quality_complete_review.v1.md",
    "eval/knowledgeos/scripts/build_knowledgeos_v01_rc_positive_section_paragraph_quality_complete_review.py",
    "knowledge_hub/core/schema_validator.py",
    "knowledge_hub/papers/knowledgeos_v01_rc_positive_section_paragraph_quality_complete_review.py",
    "tests/test_knowledgeos_v01_rc_positive_section_paragraph_quality_complete_review.py",
}

FORBIDDEN_RAW_PAYLOAD_KEYS = {"question", "answer", "citations", "sources", "excerpt", "quote", "text"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _zero_counter_fields() -> tuple[str, ...]:
    return tuple(dict.fromkeys((*ZERO_COUNTER_FIELDS, *EXTRA_ZERO_COUNTER_FIELDS)))


def _schema_blockers(report: dict[str, Any], schema_id: str, prefix: str) -> list[str]:
    if report.get("schema") != schema_id:
        return [f"{prefix}_schema_mismatch"]
    validation = validate_payload(report, schema_id, strict=True)
    if not validation.ok:
        return [f"{prefix}_schema_validation_failed"]
    return []


def _unsafe_counter_blockers(report: dict[str, Any], prefix: str) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers: list[str] = []
    for field in _zero_counter_fields():
        if _int(counts.get(field)) != 0:
            blockers.append(f"unsafe_counter_nonzero:{prefix}:{field}")
    return sorted(set(blockers))


def _raw_payload_key_count(value: Any) -> int:
    if isinstance(value, dict):
        count = sum(1 for key in value if key in FORBIDDEN_RAW_PAYLOAD_KEYS)
        return count + sum(_raw_payload_key_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(_raw_payload_key_count(item) for item in value)
    return 0


def _positive_seed_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers = _schema_blockers(
        report,
        KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_POSITIVE_SECTION_PARAGRAPH_SEED_SCHEMA_ID,
        "positive_seed",
    )
    if report.get("status") != "ready":
        blockers.append("positive_seed_not_ready")
    if report.get("decision") != POSITIVE_SEED_READY_DECISION:
        blockers.append("positive_seed_decision_not_ready")
    if report.get("nextRecommendedTranche") != "corpus_scale_answer_quality_positive_answer_execution_gate":
        blockers.append("positive_seed_next_tranche_not_execution_gate")
    if _int(counts.get("inputCaseRows")) != EXPECTED_CASE_ROWS:
        blockers.append("positive_seed_input_case_rows_not_50")
    if _int(counts.get("positiveSeedRows")) != EXPECTED_POSITIVE_ROWS:
        blockers.append("positive_seed_rows_not_7")
    if _int(counts.get("positiveProbePassRows")) != EXPECTED_POSITIVE_ROWS:
        blockers.append("positive_seed_probe_pass_rows_not_7")
    if _int(counts.get("heldExpectedNoAnswerRows")) != EXPECTED_EXPECTED_NO_ANSWER_ROWS:
        blockers.append("positive_seed_expected_no_answer_hold_mismatch")
    if _int(counts.get("heldStructuredModalityRows")) != EXPECTED_STRUCTURED_MODALITY_ROWS:
        blockers.append("positive_seed_structured_modality_hold_mismatch")
    if _int(counts.get("controlledExecutionUnexpectedAnswerableRows")) != 0:
        blockers.append("positive_seed_unexpected_answerable_rows_present")
    if _int(counts.get("controlledExecutionNoAnswerSafetyFailRows")) != 0:
        blockers.append("positive_seed_no_answer_safety_fail_rows_present")
    if _int(counts.get("citationCount")) != EXPECTED_STRICT_PROVENANCE_ROWS:
        blockers.append("positive_seed_citation_count_not_20")
    if _int(counts.get("evidencePacketContractSpanRows")) != EXPECTED_STRICT_PROVENANCE_ROWS:
        blockers.append("positive_seed_contract_span_rows_not_20")
    if _int(counts.get("publicDefaultPromotionHeldRows")) < 1:
        blockers.append("positive_seed_public_default_hold_missing")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("positive_seed_private_path_leak")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("positive_seed_schema_violations_present")
    return sorted(set(blockers + _unsafe_counter_blockers(report, "positive_seed")))


def _positive_execution_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers = _schema_blockers(
        report,
        KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_POSITIVE_ANSWER_EXECUTION_GATE_SCHEMA_ID,
        "positive_execution",
    )
    if report.get("status") != "ready":
        blockers.append("positive_execution_not_ready")
    if report.get("decision") != POSITIVE_EXECUTION_READY_DECISION:
        blockers.append("positive_execution_decision_not_ready")
    if report.get("nextRecommendedTranche") != "corpus_scale_answer_quality_gate_refresh_after_positive_answer_execution":
        blockers.append("positive_execution_next_tranche_not_provenance_refresh")
    if _int(counts.get("inputPositiveSeedRows")) != EXPECTED_POSITIVE_ROWS:
        blockers.append("positive_execution_input_positive_rows_not_7")
    if _int(counts.get("attemptedPositiveAnswerRows")) != EXPECTED_POSITIVE_ROWS:
        blockers.append("positive_execution_attempted_rows_not_7")
    if _int(counts.get("positiveAnswerPassRows")) != EXPECTED_POSITIVE_ROWS:
        blockers.append("positive_execution_pass_rows_not_7")
    if _int(counts.get("positiveAnswerFailRows")) != 0:
        blockers.append("positive_execution_fail_rows_present")
    if _int(counts.get("heldExpectedNoAnswerRows")) != EXPECTED_EXPECTED_NO_ANSWER_ROWS:
        blockers.append("positive_execution_expected_no_answer_hold_mismatch")
    if _int(counts.get("heldStructuredModalityRows")) != EXPECTED_STRUCTURED_MODALITY_ROWS:
        blockers.append("positive_execution_structured_modality_hold_mismatch")
    if _int(counts.get("controlledExecutionUnexpectedAnswerableRows")) != 0:
        blockers.append("positive_execution_unexpected_answerable_rows_present")
    if _int(counts.get("controlledExecutionNoAnswerSafetyFailRows")) != 0:
        blockers.append("positive_execution_no_answer_safety_fail_rows_present")
    if _int(counts.get("citationCount")) != EXPECTED_STRICT_PROVENANCE_ROWS:
        blockers.append("positive_execution_citation_count_not_20")
    if _int(counts.get("evidencePacketContractSpanRows")) != EXPECTED_STRICT_PROVENANCE_ROWS:
        blockers.append("positive_execution_contract_span_rows_not_20")
    if _int(counts.get("answerTextIncludedRows")) != 0:
        blockers.append("positive_execution_answer_text_included")
    if _int(counts.get("publicDefaultPromotionHeldRows")) < 1:
        blockers.append("positive_execution_public_default_hold_missing")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("positive_execution_private_path_leak")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("positive_execution_schema_violations_present")
    return sorted(set(blockers + _unsafe_counter_blockers(report, "positive_execution")))


def _provenance_refresh_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers = _schema_blockers(
        report,
        KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_GATE_REFRESH_AFTER_POSITIVE_ANSWER_EXECUTION_SCHEMA_ID,
        "provenance_refresh",
    )
    if report.get("status") != "ready":
        blockers.append("provenance_refresh_not_ready")
    if report.get("decision") != PROVENANCE_REFRESH_READY_DECISION:
        blockers.append("provenance_refresh_decision_not_ready")
    if report.get("nextRecommendedTranche") != "knowledgeos_v01_rc_positive_section_paragraph_quality_complete_review":
        blockers.append("provenance_refresh_next_tranche_not_complete_review")
    if _int(counts.get("inputPositiveExecutionRows")) != EXPECTED_POSITIVE_ROWS:
        blockers.append("provenance_refresh_input_rows_not_7")
    if _int(counts.get("attemptedProvenanceRows")) != EXPECTED_POSITIVE_ROWS:
        blockers.append("provenance_refresh_attempted_rows_not_7")
    if _int(counts.get("provenancePassRows")) != EXPECTED_POSITIVE_ROWS:
        blockers.append("provenance_refresh_pass_rows_not_7")
    if _int(counts.get("provenanceFailRows")) != 0:
        blockers.append("provenance_refresh_fail_rows_present")
    if _int(counts.get("heldExpectedNoAnswerRows")) != EXPECTED_EXPECTED_NO_ANSWER_ROWS:
        blockers.append("provenance_refresh_expected_no_answer_hold_mismatch")
    if _int(counts.get("heldStructuredModalityRows")) != EXPECTED_STRUCTURED_MODALITY_ROWS:
        blockers.append("provenance_refresh_structured_modality_hold_mismatch")
    if _int(counts.get("strictProvenanceSpanRows")) != EXPECTED_STRICT_PROVENANCE_ROWS:
        blockers.append("provenance_refresh_strict_span_rows_not_20")
    if _int(counts.get("sourceContentHashRows")) != EXPECTED_STRICT_PROVENANCE_ROWS:
        blockers.append("provenance_refresh_source_hash_rows_not_20")
    if _int(counts.get("charsLocatorRows")) != EXPECTED_STRICT_PROVENANCE_ROWS:
        blockers.append("provenance_refresh_chars_locator_rows_not_20")
    if _int(counts.get("answerContractCitationProvenanceRows")) != EXPECTED_STRICT_PROVENANCE_ROWS:
        blockers.append("provenance_refresh_answer_contract_provenance_rows_not_20")
    if _int(counts.get("publicDefaultPromotionHeldRows")) < 1:
        blockers.append("provenance_refresh_public_default_hold_missing")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("provenance_refresh_private_path_leak")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("provenance_refresh_schema_violations_present")
    if _raw_payload_key_count(report):
        blockers.append("provenance_refresh_raw_payload_key_present")
    return sorted(set(blockers + _unsafe_counter_blockers(report, "provenance_refresh")))


def _cross_report_blockers(seed_report: dict[str, Any], execution_report: dict[str, Any], provenance_report: dict[str, Any]) -> list[str]:
    seed_counts = dict(seed_report.get("counts") or {})
    execution_counts = dict(execution_report.get("counts") or {})
    provenance_counts = dict(provenance_report.get("counts") or {})
    blockers: list[str] = []
    if _int(seed_counts.get("positiveSeedRows")) != _int(execution_counts.get("inputPositiveSeedRows")):
        blockers.append("seed_execution_positive_row_count_mismatch")
    if _int(execution_counts.get("positiveAnswerPassRows")) != _int(provenance_counts.get("provenancePassRows")):
        blockers.append("execution_provenance_pass_count_mismatch")
    for field in ("heldExpectedNoAnswerRows", "heldStructuredModalityRows"):
        values = {_int(seed_counts.get(field)), _int(execution_counts.get(field)), _int(provenance_counts.get(field))}
        if len(values) != 1:
            blockers.append(f"cross_report_{field}_mismatch")
    if _int(execution_counts.get("evidencePacketContractSpanRows")) != _int(provenance_counts.get("strictProvenanceSpanRows")):
        blockers.append("execution_provenance_contract_span_count_mismatch")
    if _int(provenance_counts.get("positiveSectionParagraphQualityCompleteRows")) != 1:
        blockers.append("provenance_quality_complete_row_missing")
    return sorted(set(blockers))


def _count_check_runs(github_pr_state: dict[str, Any]) -> tuple[int, int]:
    total = 0
    success = 0
    for row in list(github_pr_state.get("statusCheckRollup") or []):
        item = dict(row)
        if item.get("__typename") != "CheckRun":
            continue
        total += 1
        if item.get("status") == "COMPLETED" and item.get("conclusion") == "SUCCESS":
            success += 1
    return total, success


def _pr_blockers(pr_state: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if pr_state.get("lookupStatus") != "ok":
        blockers.append(f"pr_{expected['prNumber']}_lookup_failed")
    if _int(pr_state.get("number")) != _int(expected.get("prNumber")):
        blockers.append(f"pr_{expected['prNumber']}_number_mismatch")
    if pr_state.get("state") != "MERGED":
        blockers.append(f"pr_{expected['prNumber']}_not_merged")
    if pr_state.get("isDraft") is not False:
        blockers.append(f"pr_{expected['prNumber']}_still_draft")
    if _clean_text(pr_state.get("baseRefName")) != DEFAULT_BASE_BRANCH:
        blockers.append(f"pr_{expected['prNumber']}_base_not_main")
    if _clean_text(pr_state.get("headRefName")) != _clean_text(expected.get("headBranch")):
        blockers.append(f"pr_{expected['prNumber']}_head_branch_mismatch")
    if not _clean_text(pr_state.get("mergedAt")):
        blockers.append(f"pr_{expected['prNumber']}_merged_at_missing")
    if not _clean_text(dict(pr_state.get("mergeCommit") or {}).get("oid")):
        blockers.append(f"pr_{expected['prNumber']}_merge_commit_missing")
    check_total, check_success = _count_check_runs(pr_state)
    if check_total != 7:
        blockers.append(f"pr_{expected['prNumber']}_ci_check_count_not_7")
    if check_success != 7:
        blockers.append(f"pr_{expected['prNumber']}_ci_checks_not_green")
    if _contains_private_path(pr_state):
        blockers.append(f"pr_{expected['prNumber']}_private_path_marker")
    return sorted(set(blockers))


def _status_rows(git_state: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in list(git_state.get("statusRows") or []):
        row = dict(item)
        rows.append({"statusCode": _clean_text(row.get("statusCode")), "path": _clean_text(row.get("path"))})
    return rows


def _blocking_dirty_rows(git_state: dict[str, Any]) -> list[dict[str, str]]:
    return [row for row in _status_rows(git_state) if row.get("path") not in SELF_REVIEW_ALLOWED_DIRTY_PATHS]


def _git_state_blockers(git_state: dict[str, Any], latest_merge_commit_oid: str) -> list[str]:
    blockers: list[str] = []
    origin_main = _clean_text(git_state.get("originMainSha"))
    remote_main = _clean_text(git_state.get("remoteMainSha"))
    head = _clean_text(git_state.get("headSha"))
    if not origin_main:
        blockers.append("origin_main_sha_missing")
    if not remote_main:
        blockers.append("remote_main_sha_missing")
    if origin_main and remote_main and origin_main != remote_main:
        blockers.append("origin_main_remote_main_mismatch")
    if latest_merge_commit_oid and origin_main and latest_merge_commit_oid != origin_main:
        blockers.append("origin_main_does_not_match_latest_quality_merge_commit")
    if head and origin_main and head != origin_main:
        blockers.append("worktree_head_not_at_origin_main")
    if _blocking_dirty_rows(git_state):
        blockers.append("blocking_dirty_worktree_rows_present")
    if _contains_private_path(git_state):
        blockers.append("git_state_private_path_marker")
    return sorted(set(blockers))


def _hygiene_blockers(hygiene_result: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if hygiene_result.get("status") != "ok":
        blockers.append("public_hygiene_not_ok")
    if _int(hygiene_result.get("issueCount")) != 0:
        blockers.append("public_hygiene_issues_present")
    if _contains_private_path(hygiene_result):
        blockers.append("public_hygiene_private_path_marker")
    return sorted(set(blockers))


def _release_smoke_blockers(release_smoke_result: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if release_smoke_result.get("status") != "ok":
        blockers.append("release_smoke_not_ok")
    if _int(release_smoke_result.get("checkedCount")) <= 0:
        blockers.append("release_smoke_no_checks")
    if _int(release_smoke_result.get("checkedCount")) != _int(release_smoke_result.get("passedCount")):
        blockers.append("release_smoke_not_all_passed")
    if _contains_private_path(release_smoke_result):
        blockers.append("release_smoke_private_path_marker")
    return sorted(set(blockers))


def build_knowledgeos_v01_rc_positive_section_paragraph_quality_complete_review(
    *,
    positive_seed_report_path: str | Path = DEFAULT_POSITIVE_SEED_REPORT,
    positive_execution_report_path: str | Path = DEFAULT_POSITIVE_EXECUTION_REPORT,
    provenance_refresh_report_path: str | Path = DEFAULT_PROVENANCE_REFRESH_REPORT,
    positive_seed_report: dict[str, Any] | None = None,
    positive_execution_report: dict[str, Any] | None = None,
    provenance_refresh_report: dict[str, Any] | None = None,
    git_state: dict[str, Any] | None = None,
    github_pr_states: list[dict[str, Any]] | None = None,
    release_smoke_result: dict[str, Any] | None = None,
    hygiene_result: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    seed_report = dict(positive_seed_report or _read_json(positive_seed_report_path))
    execution_report = dict(positive_execution_report or _read_json(positive_execution_report_path))
    provenance_report = dict(provenance_refresh_report or _read_json(provenance_refresh_report_path))
    git_payload = dict(git_state or {})
    pr_payloads = [dict(row or {}) for row in list(github_pr_states or [])]
    smoke_payload = dict(release_smoke_result or {})
    hygiene_payload = dict(hygiene_result or {})
    pr_by_number = {_int(row.get("number")): row for row in pr_payloads}

    seed_blockers = _positive_seed_blockers(seed_report)
    execution_blockers = _positive_execution_blockers(execution_report)
    provenance_blockers = _provenance_refresh_blockers(provenance_report)
    cross_blockers = _cross_report_blockers(seed_report, execution_report, provenance_report)
    pr_check_rows: list[dict[str, Any]] = []
    pr_blockers: list[str] = []
    for expected in EXPECTED_PR_REFS:
        pr_state = pr_by_number.get(_int(expected.get("prNumber")), {})
        blockers = _pr_blockers(pr_state, expected)
        pr_blockers.extend(blockers)
        check_total, check_success = _count_check_runs(pr_state)
        pr_check_rows.append(
            {
                "prNumber": _int(expected.get("prNumber")),
                "phase": _clean_text(expected.get("phase")),
                "status": "pass" if not blockers else "fail",
                "merged": pr_state.get("state") == "MERGED",
                "mergeCommitShortSha": _clean_text(dict(pr_state.get("mergeCommit") or {}).get("oid"))[:7],
                "ciCheckRows": check_total,
                "ciCheckSuccessRows": check_success,
                "blockers": blockers,
            }
        )
    latest_merge_commit_oid = _clean_text(dict(pr_by_number.get(182, {}).get("mergeCommit") or {}).get("oid"))
    git_blockers = _git_state_blockers(git_payload, latest_merge_commit_oid)
    smoke_blockers = _release_smoke_blockers(smoke_payload)
    hygiene_blockers = _hygiene_blockers(hygiene_payload)

    private_path_leak_rows = (
        1
        if _contains_private_path(seed_report)
        or _contains_private_path(execution_report)
        or _contains_private_path(provenance_report)
        or _contains_private_path(git_payload)
        or _contains_private_path(pr_payloads)
        or _contains_private_path(smoke_payload)
        or _contains_private_path(hygiene_payload)
        else 0
    )
    semantic_violations = sorted(
        set(
            seed_blockers
            + execution_blockers
            + provenance_blockers
            + cross_blockers
            + pr_blockers
            + git_blockers
            + smoke_blockers
            + hygiene_blockers
        )
    )
    if private_path_leak_rows:
        semantic_violations.append("positive_section_paragraph_quality_complete_private_path_marker")
    semantic_violations = sorted(set(semantic_violations))
    status = "ready" if not semantic_violations else "blocked"

    seed_counts = dict(seed_report.get("counts") or {})
    execution_counts = dict(execution_report.get("counts") or {})
    provenance_counts = dict(provenance_report.get("counts") or {})
    remote_branch_rows = sum(1 for row in list(git_payload.get("remoteFeatureBranches") or []) if _clean_text(row.get("sha")))
    ci_total = sum(_count_check_runs(row)[0] for row in pr_payloads)
    ci_success = sum(_count_check_runs(row)[1] for row in pr_payloads)
    counts = {
        "qualityCompleteReviewRows": 1,
        "positiveSeedReportRows": 1,
        "positiveExecutionReportRows": 1,
        "provenanceRefreshReportRows": 1,
        "positiveSeedReadyRows": 1 if not seed_blockers else 0,
        "positiveExecutionReadyRows": 1 if not execution_blockers else 0,
        "provenanceRefreshReadyRows": 1 if not provenance_blockers else 0,
        "inputCaseRows": _int(seed_counts.get("inputCaseRows")),
        "positiveSeedRows": _int(seed_counts.get("positiveSeedRows")),
        "positiveAnswerPassRows": _int(execution_counts.get("positiveAnswerPassRows")),
        "positiveAnswerFailRows": _int(execution_counts.get("positiveAnswerFailRows")),
        "provenancePassRows": _int(provenance_counts.get("provenancePassRows")),
        "provenanceFailRows": _int(provenance_counts.get("provenanceFailRows")),
        "positiveMethodComparisonRows": _int(provenance_counts.get("positiveMethodComparisonRows")),
        "positiveLimitationRows": _int(provenance_counts.get("positiveLimitationRows")),
        "heldExpectedNoAnswerRows": _int(provenance_counts.get("heldExpectedNoAnswerRows")),
        "heldStructuredModalityRows": _int(provenance_counts.get("heldStructuredModalityRows")),
        "controlledExecutionUnexpectedAnswerableRows": _int(provenance_counts.get("controlledExecutionUnexpectedAnswerableRows")),
        "controlledExecutionNoAnswerSafetyFailRows": _int(provenance_counts.get("controlledExecutionNoAnswerSafetyFailRows")),
        "strictProvenanceSpanRows": _int(provenance_counts.get("strictProvenanceSpanRows")),
        "sourceContentHashRows": _int(provenance_counts.get("sourceContentHashRows")),
        "charsLocatorRows": _int(provenance_counts.get("charsLocatorRows")),
        "answerContractCitationRows": _int(provenance_counts.get("answerContractCitationRows")),
        "answerContractCitationProvenanceRows": _int(provenance_counts.get("answerContractCitationProvenanceRows")),
        "rawPayloadKeyRows": _raw_payload_key_count(provenance_report),
        "prRows": len(EXPECTED_PR_REFS),
        "prMergedRows": sum(1 for row in pr_payloads if row.get("state") == "MERGED"),
        "ciCheckRows": ci_total,
        "ciCheckSuccessRows": ci_success,
        "mainLatestQualityMergeCommitMatchRows": 1
        if latest_merge_commit_oid and latest_merge_commit_oid == _clean_text(git_payload.get("originMainSha"))
        else 0,
        "remoteMainMatchesLocalOriginRows": 1
        if _clean_text(git_payload.get("originMainSha")) == _clean_text(git_payload.get("remoteMainSha"))
        and _clean_text(git_payload.get("originMainSha"))
        else 0,
        "releaseSmokeCheckedRows": _int(smoke_payload.get("checkedCount")),
        "releaseSmokePassedRows": _int(smoke_payload.get("passedCount")),
        "publicHygieneIssueRows": _int(hygiene_payload.get("issueCount")),
        "positiveSectionParagraphQualityCompleteRows": 1 if status == "ready" else 0,
        "researchPreviewReleaseReadinessDecisionRecommendedRows": 1 if status == "ready" else 0,
        "publicDefaultPromotionReadyRows": 0,
        "publicDefaultPromotionHeldRows": 1,
        "generalRcReadyRows": 0,
        "corpusScalePositiveSliceProvenRows": 1 if status == "ready" else 0,
        "corpusScaleClaimProvenRows": 0,
        "remoteFeatureBranchStillExistsRows": remote_branch_rows,
        "branchCleanupRecommendedRows": remote_branch_rows if status == "ready" else 0,
        "branchCleanupAppliedRows": 0,
        "blockedRows": len(semantic_violations),
        **{field: 0 for field in _zero_counter_fields()},
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(semantic_violations),
    }
    return {
        "schema": KNOWLEDGEOS_V01_RC_POSITIVE_SECTION_PARAGRAPH_QUALITY_COMPLETE_REVIEW_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "inputs": {
            "positiveSeedReportRef": DEFAULT_POSITIVE_SEED_REPORT.as_posix(),
            "positiveExecutionReportRef": DEFAULT_POSITIVE_EXECUTION_REPORT.as_posix(),
            "provenanceRefreshReportRef": DEFAULT_PROVENANCE_REFRESH_REPORT.as_posix(),
            "baseBranch": DEFAULT_BASE_BRANCH,
            "headShortSha": _clean_text(git_payload.get("headShortSha")),
            "originMainShortSha": _clean_text(git_payload.get("originMainShortSha")),
            "remoteMainShortSha": _clean_text(git_payload.get("remoteMainShortSha")),
            "latestQualityMergeCommitShortSha": latest_merge_commit_oid[:7] if latest_merge_commit_oid else "",
        },
        "qualityDecision": {
            "sectionParagraphQuality": "complete" if status == "ready" else "blocked",
            "positiveAnswerSlice": "7_positive_rows_ready_with_hash_chars_citation_provenance",
            "noAnswerSafety": "expected_no_answer_and_structured_modality_rows_held",
            "publicDefaultDecision": "hold_public_default_promotion",
            "generalRcDecision": "not_ready_from_this_gate_alone",
            "nextGate": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        },
        "counts": counts,
        "gate": {
            "qualityCompleteReviewReady": status == "ready",
            "positiveSeedReady": not seed_blockers,
            "positiveExecutionReady": not execution_blockers,
            "provenanceRefreshReady": not provenance_blockers,
            "crossReportConsistent": not cross_blockers,
            "allQualityPrsMerged": counts["prMergedRows"] == len(EXPECTED_PR_REFS),
            "ciChecksGreen": counts["ciCheckRows"] == 21 and counts["ciCheckSuccessRows"] == 21,
            "mainContainsLatestQualityMergeCommit": counts["mainLatestQualityMergeCommitMatchRows"] == 1,
            "remoteMainVerified": counts["remoteMainMatchesLocalOriginRows"] == 1,
            "releaseSmokeReady": not smoke_blockers,
            "publicHygieneReady": not hygiene_blockers,
            "positiveSectionParagraphQualityComplete": counts["positiveSectionParagraphQualityCompleteRows"] == 1,
            "expectedNoAnswerRowsHeld": counts["heldExpectedNoAnswerRows"] == EXPECTED_EXPECTED_NO_ANSWER_ROWS,
            "structuredModalityRowsHeld": counts["heldStructuredModalityRows"] == EXPECTED_STRUCTURED_MODALITY_ROWS,
            "publicDefaultPromotionAllowed": False,
            "generalRcReady": False,
            "semanticViolations": semantic_violations,
        },
        "checkRows": [
            {"checkId": "positive_seed_report", "status": "pass" if not seed_blockers else "fail", "blockers": seed_blockers},
            {
                "checkId": "positive_answer_execution_report",
                "status": "pass" if not execution_blockers else "fail",
                "blockers": execution_blockers,
            },
            {
                "checkId": "positive_provenance_refresh_report",
                "status": "pass" if not provenance_blockers else "fail",
                "blockers": provenance_blockers,
            },
            {"checkId": "cross_report_consistency", "status": "pass" if not cross_blockers else "fail", "blockers": cross_blockers},
            {"checkId": "git_main_state", "status": "pass" if not git_blockers else "fail", "blockers": git_blockers},
            {"checkId": "release_smoke", "status": "pass" if not smoke_blockers else "fail", "blockers": smoke_blockers},
            {"checkId": "public_hygiene", "status": "pass" if not hygiene_blockers else "fail", "blockers": hygiene_blockers},
        ],
        "pullRequestRows": pr_check_rows,
        "cleanupRows": [
            {
                "cleanupId": "remote_feature_branches",
                "targetRefs": [f"origin/{row.get('headBranch')}" for row in EXPECTED_PR_REFS],
                "status": "recommended_not_applied" if remote_branch_rows else "not_needed",
                "requiresExplicitApproval": True,
                "summary": "Remote feature branch cleanup is separate from this read-only complete review.",
            }
        ],
        "warnings": [
            "complete_review_closes_positive_section_paragraph_quality_slice_only",
            "public_default_promotion_remains_held",
            "table_equation_figure_evidence_remains_outside_this_gate",
            "branch_cleanup_not_applied_in_this_tranche",
        ],
    }


def render_knowledgeos_v01_rc_positive_section_paragraph_quality_complete_review_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    decision = dict(report.get("qualityDecision") or {})
    lines = [
        "# KnowledgeOS v0.1 RC Positive Section/Paragraph Quality Complete Review",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- sectionParagraphQuality: `{decision.get('sectionParagraphQuality')}`",
        f"- publicDefaultDecision: `{decision.get('publicDefaultDecision')}`",
        f"- inputCaseRows: `{counts.get('inputCaseRows')}`",
        f"- positiveSeedRows: `{counts.get('positiveSeedRows')}`",
        f"- positiveAnswerPassRows: `{counts.get('positiveAnswerPassRows')}`",
        f"- provenancePassRows: `{counts.get('provenancePassRows')}`",
        f"- strictProvenanceSpanRows: `{counts.get('strictProvenanceSpanRows')}`",
        f"- sourceContentHashRows: `{counts.get('sourceContentHashRows')}`",
        f"- charsLocatorRows: `{counts.get('charsLocatorRows')}`",
        f"- answerContractCitationProvenanceRows: `{counts.get('answerContractCitationProvenanceRows')}`",
        f"- heldExpectedNoAnswerRows: `{counts.get('heldExpectedNoAnswerRows')}`",
        f"- heldStructuredModalityRows: `{counts.get('heldStructuredModalityRows')}`",
        f"- prMergedRows: `{counts.get('prMergedRows')}`",
        f"- ciCheckSuccessRows: `{counts.get('ciCheckSuccessRows')}`",
        f"- releaseSmokePassedRows: `{counts.get('releaseSmokePassedRows')}`",
        f"- publicHygieneIssueRows: `{counts.get('publicHygieneIssueRows')}`",
        f"- positiveSectionParagraphQualityCompleteRows: `{counts.get('positiveSectionParagraphQualityCompleteRows')}`",
        f"- publicDefaultPromotionHeldRows: `{counts.get('publicDefaultPromotionHeldRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Checks",
        "",
    ]
    for row in list(report.get("checkRows") or []):
        blockers = ", ".join(list(row.get("blockers") or [])) or "none"
        lines.append(f"- `{row.get('checkId')}`: `{row.get('status')}`; blockers=`{blockers}`")
    lines.extend(["", "## Pull Requests", ""])
    for row in list(report.get("pullRequestRows") or []):
        blockers = ", ".join(list(row.get("blockers") or [])) or "none"
        lines.append(
            f"- `#{row.get('prNumber')}` `{row.get('phase')}`: `{row.get('status')}`; "
            f"merged=`{row.get('merged')}`; ci=`{row.get('ciCheckSuccessRows')}/{row.get('ciCheckRows')}`; blockers=`{blockers}`"
        )
    lines.extend(["", "## Mutation Guarantees", ""])
    for field in _zero_counter_fields():
        lines.append(f"- {field}: `{counts.get(field)}`")
    return "\n".join(lines).rstrip() + "\n"


def write_knowledgeos_v01_rc_positive_section_paragraph_quality_complete_review(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_knowledgeos_v01_rc_positive_section_paragraph_quality_complete_review_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "KNOWLEDGEOS_V01_RC_POSITIVE_SECTION_PARAGRAPH_QUALITY_COMPLETE_REVIEW_SCHEMA_ID",
    "READY_DECISION",
    "BLOCKED_DECISION",
    "build_knowledgeos_v01_rc_positive_section_paragraph_quality_complete_review",
    "write_knowledgeos_v01_rc_positive_section_paragraph_quality_complete_review",
]
