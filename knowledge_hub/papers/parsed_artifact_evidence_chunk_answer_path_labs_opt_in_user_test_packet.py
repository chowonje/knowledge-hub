"""User-test packet for the labs parsed-artifact evidence chunk surface."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
from typing import Any

from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_runner import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_QUALITY_EVAL_RUNNER_SCHEMA_ID,
    READY_DECISION as QUALITY_EVAL_RUNNER_READY_DECISION,
    ZERO_COUNTER_FIELDS,
    _clean_text,
    _contains_private_path,
    _int,
    _read_json,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_PACKET_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-labs-opt-in-user-test-packet.v1"
)

READY_DECISION = "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_packet_ready"
BLOCKED_DECISION = "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_packet_blocked"
NEXT_TRANCHE_READY = "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture"
NEXT_TRANCHE_BLOCKED = "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_packet_repair"
DEFAULT_QUALITY_EVAL_RUNNER_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_runner.v1.json"
)
DEFAULT_USER_TEST_QUESTIONS = {
    "alphafold_protein_structure_seed": (
        "What parsed section or paragraph evidence is available about AlphaFold protein structure prediction?"
    ),
    "word_vectors_similarity_seed": (
        "What parsed section or paragraph evidence is available about continuous word representations?"
    ),
    "resolved_pair_compare_seed": "Compare available parsed evidence for the AlphaFold and word vector papers.",
    "missing_candidate_store_no_evidence_seed": (
        "What parsed section or paragraph evidence is available for this missing paper?"
    ),
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _runner_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    blockers: list[str] = []
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_QUALITY_EVAL_RUNNER_SCHEMA_ID:
        blockers.append("quality_eval_runner_schema_mismatch")
    if report.get("status") != "ready":
        blockers.append("quality_eval_runner_not_ready")
    if report.get("decision") != QUALITY_EVAL_RUNNER_READY_DECISION:
        blockers.append("quality_eval_runner_decision_not_ready")
    if gate.get("readyForLabsOptInUserTestPacket") is not True:
        blockers.append("quality_eval_runner_gate_not_ready_for_user_test_packet")
    if _int(counts.get("qualityPassRows")) != _int(counts.get("inputCaseRows")):
        blockers.append("quality_eval_runner_cases_not_all_passed")
    if float(counts.get("minQualityScore") or 0.0) < 1.0:
        blockers.append("quality_eval_runner_min_quality_score_below_threshold")
    if _int(counts.get("noEvidenceLlmCallRows")) != 0:
        blockers.append("quality_eval_runner_no_evidence_called_llm")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("quality_eval_runner_schema_violations_present")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("quality_eval_runner_private_path_leak")
    if _contains_private_path(report):
        blockers.append("quality_eval_runner_private_path_marker")
    return sorted(set(blockers))


def _command_for_row(row: dict[str, Any], *, allow_external: bool = False) -> str:
    case_id = _clean_text(row.get("caseId"))
    question = _clean_text(row.get("question")) or DEFAULT_USER_TEST_QUESTIONS.get(case_id, "")
    parts = [
        "khub",
        "labs",
        "paper",
        "evidence-chunk-ask",
        question or f"Run evidence chunk preview for {case_id}",
    ]
    for paper_id in list(row.get("paperIds") or []):
        parts.extend(["--paper-id", _clean_text(paper_id)])
    if allow_external:
        parts.append("--allow-external")
    parts.append("--json")
    return " ".join(shlex.quote(part) for part in parts if _clean_text(part))


def _assertions_for_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    assertions: list[dict[str, Any]] = [
        {
            "jsonPath": "$.schema",
            "operator": "equals",
            "expected": "knowledge-hub.paper.evidence-chunk-answer-preview.result.v1",
        },
        {"jsonPath": "$.status", "operator": "equals", "expected": _clean_text(row.get("observedStatus"))},
        {"jsonPath": "$.answerable", "operator": "equals", "expected": bool(row.get("observedAnswerable"))},
        {"jsonPath": "$.sourceType", "operator": "equals", "expected": "paper"},
        {"jsonPath": "$.allowExternal", "operator": "equals", "expected": False},
        {
            "jsonPath": "$.queryPlan.parsed_artifact_evidence_chunk_adapter",
            "operator": "equals",
            "expected": "runtime_v1",
        },
        {
            "jsonPath": "$.evidencePacketSummary.adapterRowsAdded",
            "operator": "atLeast",
            "expected": _int(row.get("adapterRowsAdded")),
        },
        {
            "jsonPath": "$.evidencePacketSummary.citationCount",
            "operator": "atLeast",
            "expected": _int(row.get("citationCount")),
        },
        {
            "jsonPath": "$.evidencePacketContractSummary.spanRows",
            "operator": "atLeast",
            "expected": _int(row.get("evidencePacketContractSpanRows")),
        },
    ]
    for term in list(row.get("observedEvidenceTerms") or []):
        assertions.append(
            {
                "jsonPath": "$.sources[*].text",
                "operator": "containsCaseInsensitive",
                "expected": _clean_text(term),
            }
        )
    return assertions


def _packet_row(row: dict[str, Any]) -> dict[str, Any]:
    expected_answerable = bool(row.get("expectedAnswerable"))
    return {
        "packetRowId": f"user-test:{_clean_text(row.get('caseId'))}",
        "caseId": _clean_text(row.get("caseId")),
        "testType": "expected_answerable" if expected_answerable else "expected_no_evidence",
        "surface": "khub_labs_cli",
        "command": _command_for_row(row),
        "expectedExitCode": 0,
        "paperIds": list(row.get("paperIds") or []),
        "expectedStatus": _clean_text(row.get("observedStatus")),
        "expectedAnswerable": bool(row.get("observedAnswerable")),
        "expectedMinCitations": _int(row.get("citationCount")),
        "expectedMinSpanRows": _int(row.get("evidencePacketContractSpanRows")),
        "expectedEvidenceTerms": list(row.get("observedEvidenceTerms") or []),
        "expectedNoEvidence": not expected_answerable,
        "expectedNoExternalCall": True,
        "expectedNoPrivatePathLeak": True,
        "jsonAssertions": _assertions_for_row(row),
        "answerTextIncludedInPacket": False,
        "citationPayloadIncludedInPacket": False,
        "sourcePayloadIncludedInPacket": False,
        "excerptIncludedInPacket": False,
    }


def _external_rejection_row(seed_row: dict[str, Any] | None) -> dict[str, Any]:
    row = dict(seed_row or {})
    if not row:
        row = {
            "caseId": "external_rejection",
            "question": "What parsed section or paragraph evidence is available?",
            "paperIds": ["1207.0580"],
        }
    return {
        "packetRowId": "user-test:external-rejection",
        "caseId": "external_rejection",
        "testType": "external_rejection",
        "surface": "khub_labs_cli",
        "command": _command_for_row(row, allow_external=True),
        "expectedExitCode": 1,
        "paperIds": list(row.get("paperIds") or []),
        "expectedErrorContains": "--allow-external is not enabled",
        "expectedNoExternalCall": True,
        "expectedNoPrivatePathLeak": True,
    }


def build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_packet(
    *,
    quality_eval_runner_report_path: str | Path = DEFAULT_QUALITY_EVAL_RUNNER_REPORT,
    quality_eval_runner_report: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    runner_report = dict(quality_eval_runner_report or _read_json(quality_eval_runner_report_path))
    source_blockers = _runner_blockers(runner_report)
    runner_rows = [dict(row or {}) for row in list(runner_report.get("rows") or [])]
    packet_rows = [] if source_blockers else [_packet_row(row) for row in runner_rows]
    external_row = None if source_blockers else _external_rejection_row(runner_rows[0] if runner_rows else None)
    private_path_leak_rows = 1 if "quality_eval_runner_private_path_marker" in source_blockers else 0
    private_path_leak_rows += sum(1 for row in packet_rows if _contains_private_path(row))
    semantic_violations = list(source_blockers)
    if external_row and _contains_private_path(external_row):
        private_path_leak_rows += 1
    if private_path_leak_rows:
        semantic_violations.append("private_path_leak")
    counts = {
        "inputRunnerRows": 1 if runner_report else 0,
        "runnerReadyInputRows": 1 if runner_report and not source_blockers else 0,
        "sourceCaseRows": len(runner_rows),
        "packetRows": len(packet_rows),
        "expectedAnswerableCommandRows": sum(1 for row in packet_rows if row.get("testType") == "expected_answerable"),
        "expectedNoEvidenceCommandRows": sum(1 for row in packet_rows if row.get("testType") == "expected_no_evidence"),
        "externalRejectionCommandRows": 1 if external_row else 0,
        "cliCommandRows": len(packet_rows) + (1 if external_row else 0),
        "mcpInstructionRows": 1 if not source_blockers else 0,
        "jsonAssertionRows": sum(len(row.get("jsonAssertions") or []) for row in packet_rows),
        "answerTextIncludedRows": 0,
        "citationPayloadIncludedRows": 0,
        "sourcePayloadIncludedRows": 0,
        "excerptIncludedRows": 0,
        **{field: 0 for field in ZERO_COUNTER_FIELDS},
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(set(semantic_violations)),
    }
    status = "ready" if not semantic_violations and packet_rows else "blocked"
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_PACKET_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "inputs": {
            "qualityEvalRunnerReportRef": (
                "eval/knowledgeos/reports/"
                "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_runner.v1.json"
            ),
            "qualityEvalRunnerSchema": _clean_text(runner_report.get("schema")),
            "qualityEvalRunnerStatus": _clean_text(runner_report.get("status")),
            "qualityEvalRunnerDecision": _clean_text(runner_report.get("decision")),
        },
        "policy": {
            "packetOnly": True,
            "labsOnly": True,
            "manualUserRunRequired": True,
            "defaultPublicAskRemainsClosed": True,
            "mcpDefaultProfileRemainsClosed": True,
            "externalModelCallsAllowed": False,
            "answerTextExcludedFromPacket": True,
            "citationPayloadExcludedFromPacket": True,
            "sourcePayloadExcludedFromPacket": True,
            "excerptExcludedFromPacket": True,
            "candidateStoreWrites": False,
            "sourceSpanCreation": False,
            "strictEvidenceCreation": False,
            "runtimeDefaultChange": False,
        },
        "counts": counts,
        "gate": {
            "readyForLabsOptInUserTestOutputCapture": status == "ready",
            "qualityEvalRunnerReady": not source_blockers,
            "packetHasExpectedAnswerableCommands": counts["expectedAnswerableCommandRows"] > 0,
            "packetHasExpectedNoEvidenceCommand": counts["expectedNoEvidenceCommandRows"] > 0,
            "packetHasExternalRejectionCommand": counts["externalRejectionCommandRows"] == 1,
            "semanticViolations": sorted(set(semantic_violations)),
        },
        "instructions": {
            "runFromRepoRoot": True,
            "cliSurface": "khub labs paper evidence-chunk-ask",
            "mcpSurface": "paper_evidence_chunk_answer_preview",
            "mcpProfileRequirement": "labs_or_all_only",
            "defaultMcpProfileExpected": "tool_hidden",
            "doNotUseAllowExternal": True,
            "captureOutputPolicy": "save_sanitized_json_only_if_user_requests_capture",
        },
        "rows": packet_rows,
        "externalRejection": external_row or {},
        "warnings": [
            "packet_commands_are_for_manual_user_testing_and_are_not_executed_by_this_report",
            "actual_cli_output_may_include_answer_or_source_payloads_but_this_packet_does_not_include_them",
        ],
    }


def render_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_packet_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Answer Path Labs Opt-in User Test Packet",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- packetRows: `{counts.get('packetRows')}`",
        f"- expectedAnswerableCommandRows: `{counts.get('expectedAnswerableCommandRows')}`",
        f"- expectedNoEvidenceCommandRows: `{counts.get('expectedNoEvidenceCommandRows')}`",
        f"- externalRejectionCommandRows: `{counts.get('externalRejectionCommandRows')}`",
        f"- jsonAssertionRows: `{counts.get('jsonAssertionRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Instructions",
        "",
        "- Run from the product repo root.",
        "- Use the labs CLI surface only: `khub labs paper evidence-chunk-ask`.",
        "- MCP testing requires the `labs` or `all` profile; the default MCP profile should not expose this tool.",
        "- Do not use `--allow-external` except for the explicit rejection check.",
        "",
        "## Commands",
        "",
    ]
    for row in list(report.get("rows") or []):
        lines.extend(
            [
                f"### `{row.get('caseId')}`",
                "",
                f"```bash\n{row.get('command')}\n```",
                "",
                f"- expectedStatus: `{row.get('expectedStatus')}`",
                f"- expectedAnswerable: `{row.get('expectedAnswerable')}`",
                f"- expectedMinCitations: `{row.get('expectedMinCitations')}`",
                f"- expectedMinSpanRows: `{row.get('expectedMinSpanRows')}`",
                f"- expectedEvidenceTerms: `{row.get('expectedEvidenceTerms')}`",
                "",
            ]
        )
    external = dict(report.get("externalRejection") or {})
    if external:
        lines.extend(
            [
                "### `external_rejection`",
                "",
                f"```bash\n{external.get('command')}\n```",
                "",
                f"- expectedExitCode: `{external.get('expectedExitCode')}`",
                f"- expectedErrorContains: `{external.get('expectedErrorContains')}`",
                "",
            ]
        )
    lines.extend(["## Mutation Guarantees", ""])
    for field in ZERO_COUNTER_FIELDS:
        lines.append(f"- {field}: `{counts.get(field)}`")
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_packet(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_packet_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_PACKET_SCHEMA_ID",
    "READY_DECISION",
    "build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_packet",
    "write_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_packet",
]
