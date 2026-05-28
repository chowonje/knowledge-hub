"""Sanitized output capture for the labs parsed-artifact evidence chunk user-test packet."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
from types import SimpleNamespace
from typing import Any, Callable

from click.testing import CliRunner

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.interfaces.cli.commands.paper_labs_cmd import paper_labs_group
from knowledge_hub.papers.evidence_chunk_answer_preview import PAPER_EVIDENCE_CHUNK_ANSWER_PREVIEW_SCHEMA_ID
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed import (
    ZERO_COUNTER_FIELDS,
    _clean_text,
    _contains_private_path,
    _int,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_packet import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_PACKET_SCHEMA_ID,
    READY_DECISION as USER_TEST_PACKET_READY_DECISION,
    _read_json,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_searcher_ingress_live_smoke import (
    DEFAULT_PAPERS_DIR,
    _build_searcher,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_OUTPUT_CAPTURE_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-labs-opt-in-user-test-output-capture.v1"
)

READY_DECISION = "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture_ready"
BLOCKED_DECISION = "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture_blocked"
NEXT_TRANCHE_READY = "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_promotion_review"
NEXT_TRANCHE_BLOCKED = "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture_repair"
DEFAULT_USER_TEST_PACKET_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_packet.v1.json"
)


CaptureInvoker = Callable[[dict[str, Any], str | Path], dict[str, Any]]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_json(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(data.encode("utf-8")).hexdigest()


def _sha256_text(value: Any) -> str:
    return "sha256:" + hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _packet_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    blockers: list[str] = []
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_PACKET_SCHEMA_ID:
        blockers.append("user_test_packet_schema_mismatch")
    if report.get("status") != "ready":
        blockers.append("user_test_packet_not_ready")
    if report.get("decision") != USER_TEST_PACKET_READY_DECISION:
        blockers.append("user_test_packet_decision_not_ready")
    if gate.get("readyForLabsOptInUserTestOutputCapture") is not True:
        blockers.append("user_test_packet_gate_not_ready_for_output_capture")
    if _int(counts.get("packetRows")) <= 0:
        blockers.append("user_test_packet_has_no_commands")
    if _int(counts.get("externalRejectionCommandRows")) != 1:
        blockers.append("user_test_packet_missing_external_rejection_command")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("user_test_packet_schema_violations_present")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("user_test_packet_private_path_leak")
    if _contains_private_path(report):
        blockers.append("user_test_packet_private_path_marker")
    return sorted(set(blockers))


def _cli_args_from_command(command: str) -> list[str]:
    tokens = shlex.split(str(command or ""))
    try:
        idx = tokens.index("evidence-chunk-ask")
    except ValueError:
        return []
    return tokens[idx:]


class _CaptureFactory:
    def __init__(self, searcher: Any) -> None:
        self._searcher = searcher

    def searcher(self) -> Any:
        return self._searcher


class _CaptureKhub:
    def __init__(self, searcher: Any) -> None:
        self.factory = _CaptureFactory(searcher)
        self.config = SimpleNamespace()


def _invoke_packet_row(row: dict[str, Any], papers_dir: str | Path) -> dict[str, Any]:
    searcher, llm = _build_searcher(papers_dir=papers_dir)
    args = _cli_args_from_command(_clean_text(row.get("command")))
    result = CliRunner().invoke(paper_labs_group, args, obj={"khub": _CaptureKhub(searcher)})
    payload: dict[str, Any] = {}
    if result.exit_code == 0:
        try:
            parsed = json.loads(result.output)
            payload = parsed if isinstance(parsed, dict) else {}
        except Exception:
            payload = {}
    return {
        "exitCode": int(result.exit_code),
        "stdout": result.output,
        "stderr": "",
        "exception": str(result.exception or ""),
        "payload": payload,
        "localFakeLlmCallRows": int(llm.calls),
    }


def _invoke_external_rejection(row: dict[str, Any], papers_dir: str | Path) -> dict[str, Any]:
    _ = papers_dir
    args = _cli_args_from_command(_clean_text(row.get("command")))
    result = CliRunner().invoke(paper_labs_group, args, obj={})
    return {
        "exitCode": int(result.exit_code),
        "stdout": result.output,
        "stderr": "",
        "exception": str(result.exception or ""),
        "payload": {},
        "localFakeLlmCallRows": 0,
    }


def _path_value(payload: dict[str, Any], json_path: str) -> Any:
    if json_path == "$.sources[*].text":
        sources = [dict(source or {}) for source in list(payload.get("sources") or [])]
        return " ".join(_clean_text(source.get("text") or source.get("excerpt")) for source in sources)
    if not json_path.startswith("$."):
        return None
    current: Any = payload
    for part in json_path[2:].split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _assertion_pass(payload: dict[str, Any], assertion: dict[str, Any]) -> bool:
    observed = _path_value(payload, _clean_text(assertion.get("jsonPath")))
    expected = assertion.get("expected")
    operator = _clean_text(assertion.get("operator"))
    if operator == "equals":
        return observed == expected
    if operator == "atLeast":
        return _int(observed) >= _int(expected)
    if operator == "containsCaseInsensitive":
        return _clean_text(expected).casefold() in _clean_text(observed).casefold()
    return False


def _observed_terms(payload: dict[str, Any], expected_terms: list[str]) -> list[str]:
    evidence_text = _path_value(payload, "$.sources[*].text")
    return [term for term in expected_terms if _clean_text(term).casefold() in _clean_text(evidence_text).casefold()]


def _capture_row(
    row: dict[str, Any],
    *,
    papers_dir: str | Path,
    invoke_row: CaptureInvoker,
) -> dict[str, Any]:
    captured = invoke_row(row, papers_dir)
    payload = dict(captured.get("payload") or {})
    expected_terms = [_clean_text(term) for term in list(row.get("expectedEvidenceTerms") or []) if _clean_text(term)]
    observed_terms = _observed_terms(payload, expected_terms)
    assertions = [dict(assertion or {}) for assertion in list(row.get("jsonAssertions") or [])]
    assertion_results = [_assertion_pass(payload, assertion) for assertion in assertions]
    summary = dict(payload.get("evidencePacketSummary") or {})
    contract = dict(payload.get("evidencePacketContractSummary") or {})
    schema_valid = validate_payload(payload, PAPER_EVIDENCE_CHUNK_ANSWER_PREVIEW_SCHEMA_ID, strict=True).ok
    exit_code = _int(captured.get("exitCode"))
    observed_status = _clean_text(payload.get("status"))
    observed_answerable = bool(payload.get("answerable"))
    local_fake_llm_calls = _int(captured.get("localFakeLlmCallRows"))
    failure_reasons: list[str] = []
    if exit_code != _int(row.get("expectedExitCode")):
        failure_reasons.append("exit_code_unexpected")
    if not schema_valid:
        failure_reasons.append("payload_schema_invalid")
    if observed_status != _clean_text(row.get("expectedStatus")):
        failure_reasons.append("status_unexpected")
    if observed_answerable is not bool(row.get("expectedAnswerable")):
        failure_reasons.append("answerable_unexpected")
    if any(not item for item in assertion_results):
        failure_reasons.append("json_assertions_failed")
    if len(observed_terms) != len(expected_terms):
        failure_reasons.append("expected_evidence_terms_missing")
    if bool(row.get("expectedAnswerable")) and local_fake_llm_calls != 1:
        failure_reasons.append("answerable_case_llm_call_count_unexpected")
    if bool(row.get("expectedNoEvidence")) and local_fake_llm_calls != 0:
        failure_reasons.append("no_evidence_case_called_llm")
    if _contains_private_path({"payload": payload, "stdout": captured.get("stdout"), "stderr": captured.get("stderr")}):
        failure_reasons.append("private_path_leak")
    return {
        "packetRowId": _clean_text(row.get("packetRowId")),
        "caseId": _clean_text(row.get("caseId")),
        "testType": _clean_text(row.get("testType")),
        "surface": "khub_labs_cli",
        "commandHash": _sha256_text(row.get("command")),
        "expectedExitCode": _int(row.get("expectedExitCode")),
        "observedExitCode": exit_code,
        "expectedStatus": _clean_text(row.get("expectedStatus")),
        "observedStatus": observed_status,
        "expectedAnswerable": bool(row.get("expectedAnswerable")),
        "observedAnswerable": observed_answerable,
        "expectedMinCitations": _int(row.get("expectedMinCitations")),
        "observedCitationCount": _int(summary.get("citationCount")),
        "expectedMinSpanRows": _int(row.get("expectedMinSpanRows")),
        "observedSpanRows": _int(contract.get("spanRows")),
        "expectedEvidenceTerms": expected_terms,
        "observedEvidenceTerms": observed_terms,
        "payloadSchema": _clean_text(payload.get("schema")),
        "payloadSchemaValid": schema_valid,
        "payloadHash": _sha256_json(payload) if payload else "",
        "assertionRows": len(assertions),
        "assertionPassRows": sum(1 for item in assertion_results if item),
        "assertionFailRows": sum(1 for item in assertion_results if not item),
        "adapterRowsAdded": _int(summary.get("adapterRowsAdded")),
        "selectedEvidenceCount": _int(summary.get("selectedEvidenceCount")),
        "localFakeLlmCallRows": local_fake_llm_calls,
        "answerTextIncludedInReport": False,
        "citationPayloadIncludedInReport": False,
        "sourcePayloadIncludedInReport": False,
        "excerptIncludedInReport": False,
        "rawOutputPersisted": False,
        "failureReasons": sorted(set(failure_reasons)),
        "pass": not failure_reasons,
    }


def _capture_external_row(
    row: dict[str, Any],
    *,
    papers_dir: str | Path,
    invoke_external: CaptureInvoker,
) -> dict[str, Any]:
    captured = invoke_external(row, papers_dir)
    output_text = _clean_text(str(captured.get("stdout") or "") + " " + str(captured.get("stderr") or ""))
    expected_error = _clean_text(row.get("expectedErrorContains"))
    exit_code = _int(captured.get("exitCode"))
    failure_reasons: list[str] = []
    if exit_code != _int(row.get("expectedExitCode")):
        failure_reasons.append("external_rejection_exit_code_unexpected")
    if expected_error and expected_error not in output_text:
        failure_reasons.append("external_rejection_error_text_missing")
    if _int(captured.get("localFakeLlmCallRows")) != 0:
        failure_reasons.append("external_rejection_called_llm")
    if _contains_private_path({"stdout": captured.get("stdout"), "stderr": captured.get("stderr")}):
        failure_reasons.append("private_path_leak")
    return {
        "packetRowId": _clean_text(row.get("packetRowId")),
        "caseId": "external_rejection",
        "testType": "external_rejection",
        "surface": "khub_labs_cli",
        "commandHash": _sha256_text(row.get("command")),
        "expectedExitCode": _int(row.get("expectedExitCode")),
        "observedExitCode": exit_code,
        "expectedErrorContains": expected_error,
        "observedErrorHash": _sha256_text(output_text),
        "localFakeLlmCallRows": _int(captured.get("localFakeLlmCallRows")),
        "rawOutputPersisted": False,
        "failureReasons": sorted(set(failure_reasons)),
        "pass": not failure_reasons,
    }


def build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture(
    *,
    papers_dir: str | Path = DEFAULT_PAPERS_DIR,
    user_test_packet_report_path: str | Path = DEFAULT_USER_TEST_PACKET_REPORT,
    user_test_packet_report: dict[str, Any] | None = None,
    generated_at: str | None = None,
    invoke_row: CaptureInvoker | None = None,
    invoke_external: CaptureInvoker | None = None,
) -> dict[str, Any]:
    packet_report = dict(user_test_packet_report or _read_json(user_test_packet_report_path))
    source_blockers = _packet_blockers(packet_report)
    packet_rows = [dict(row or {}) for row in list(packet_report.get("rows") or [])]
    external_packet_row = dict(packet_report.get("externalRejection") or {})
    row_invoker = invoke_row or _invoke_packet_row
    external_invoker = invoke_external or _invoke_external_rejection
    rows = [] if source_blockers else [
        _capture_row(row, papers_dir=papers_dir, invoke_row=row_invoker) for row in packet_rows
    ]
    external_row = (
        {}
        if source_blockers or not external_packet_row
        else _capture_external_row(external_packet_row, papers_dir=papers_dir, invoke_external=external_invoker)
    )
    all_rows = rows + ([external_row] if external_row else [])
    fail_rows = sum(1 for row in all_rows if not bool(row.get("pass")))
    private_path_leak_rows = sum(
        1 for row in all_rows if "private_path_leak" in list(row.get("failureReasons") or [])
    )
    semantic_violations = list(source_blockers)
    if fail_rows:
        semantic_violations.append(f"user_test_output_capture_failures:{fail_rows}")
    if private_path_leak_rows:
        semantic_violations.append("private_path_leak")
    expected_answerable_rows = sum(1 for row in rows if bool(row.get("expectedAnswerable")))
    expected_no_evidence_rows = sum(1 for row in rows if row.get("testType") == "expected_no_evidence")
    counts = {
        "inputPacketRows": 1 if packet_report else 0,
        "packetReadyInputRows": 1 if packet_report and not source_blockers else 0,
        "packetCommandRows": len(packet_rows) + (1 if external_packet_row else 0),
        "capturedCommandRows": len(all_rows),
        "outputCapturePassRows": sum(1 for row in all_rows if bool(row.get("pass"))),
        "outputCaptureFailRows": fail_rows,
        "expectedAnswerableOutputRows": expected_answerable_rows,
        "observedAnswerableOutputRows": sum(1 for row in rows if bool(row.get("observedAnswerable"))),
        "expectedNoEvidenceOutputRows": expected_no_evidence_rows,
        "observedNoEvidenceOutputRows": sum(1 for row in rows if row.get("observedStatus") == "no_evidence"),
        "externalRejectionOutputRows": 1 if external_row else 0,
        "externalRejectionPassRows": 1 if external_row and bool(external_row.get("pass")) else 0,
        "surfacePayloadSchemaValidRows": sum(1 for row in rows if bool(row.get("payloadSchemaValid"))),
        "jsonAssertionRows": sum(_int(row.get("assertionRows")) for row in rows),
        "jsonAssertionPassRows": sum(_int(row.get("assertionPassRows")) for row in rows),
        "jsonAssertionFailRows": sum(_int(row.get("assertionFailRows")) for row in rows),
        "adapterRowsAdded": sum(_int(row.get("adapterRowsAdded")) for row in rows),
        "selectedEvidenceCount": sum(_int(row.get("selectedEvidenceCount")) for row in rows),
        "citationCount": sum(_int(row.get("observedCitationCount")) for row in rows),
        "evidencePacketContractSpanRows": sum(_int(row.get("observedSpanRows")) for row in rows),
        "localFakeLlmCallRows": sum(_int(row.get("localFakeLlmCallRows")) for row in all_rows),
        "noEvidenceLlmCallRows": sum(
            _int(row.get("localFakeLlmCallRows")) for row in rows if row.get("testType") == "expected_no_evidence"
        ),
        "rawOutputPersistedRows": 0,
        "answerTextIncludedRows": 0,
        "citationPayloadIncludedRows": 0,
        "sourcePayloadIncludedRows": 0,
        "excerptIncludedRows": 0,
        **{field: 0 for field in ZERO_COUNTER_FIELDS},
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(set(semantic_violations)),
    }
    status = "ready" if not semantic_violations and all_rows else "blocked"
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_OUTPUT_CAPTURE_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "inputs": {
            "userTestPacketReportRef": (
                "eval/knowledgeos/reports/"
                "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_packet.v1.json"
            ),
            "userTestPacketSchema": _clean_text(packet_report.get("schema")),
            "userTestPacketStatus": _clean_text(packet_report.get("status")),
            "userTestPacketDecision": _clean_text(packet_report.get("decision")),
            "papersDirRef": "papers_dir",
        },
        "policy": {
            "reportOnly": True,
            "labsOnly": True,
            "inProcessCliInvocation": True,
            "localFakeLlmOnly": True,
            "rawOutputPersisted": False,
            "answerTextExcludedFromReport": True,
            "citationPayloadExcludedFromReport": True,
            "sourcePayloadExcludedFromReport": True,
            "excerptExcludedFromReport": True,
            "externalModelCallsAllowed": False,
            "judgeModelCallsAllowed": False,
            "candidateStoreWrites": False,
            "sourceSpanCreation": False,
            "strictEvidenceCreation": False,
            "runtimeDefaultChange": False,
        },
        "counts": counts,
        "gate": {
            "readyForLabsOptInUserTestPromotionReview": status == "ready",
            "userTestPacketReady": not source_blockers,
            "allCapturedCommandsPassed": fail_rows == 0 and bool(all_rows),
            "allJsonAssertionsPassed": counts["jsonAssertionRows"] == counts["jsonAssertionPassRows"],
            "expectedNoEvidenceCasesStayedNoEvidence": counts["expectedNoEvidenceOutputRows"]
            == counts["observedNoEvidenceOutputRows"],
            "externalRequestRejected": counts["externalRejectionPassRows"] == 1,
            "noRawOutputPersisted": counts["rawOutputPersistedRows"] == 0,
            "semanticViolations": sorted(set(semantic_violations)),
        },
        "rows": rows,
        "externalRejection": external_row,
        "warnings": [
            "raw_cli_json_output_is_not_persisted_in_this_report",
            "payload_hashes_are_for_reproducibility_without_exposing_answer_or_source_payloads",
        ],
    }


def render_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Answer Path Labs Opt-in User Test Output Capture",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- capturedCommandRows: `{counts.get('capturedCommandRows')}`",
        f"- outputCapturePassRows: `{counts.get('outputCapturePassRows')}`",
        f"- outputCaptureFailRows: `{counts.get('outputCaptureFailRows')}`",
        f"- jsonAssertionRows: `{counts.get('jsonAssertionRows')}`",
        f"- jsonAssertionPassRows: `{counts.get('jsonAssertionPassRows')}`",
        f"- externalRejectionPassRows: `{counts.get('externalRejectionPassRows')}`",
        f"- rawOutputPersistedRows: `{counts.get('rawOutputPersistedRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Captured Rows",
        "",
    ]
    for row in list(report.get("rows") or []):
        lines.extend(
            [
                f"- `{row.get('caseId')}`: pass=`{row.get('pass')}`, "
                f"status=`{row.get('observedStatus')}`, answerable=`{row.get('observedAnswerable')}`, "
                f"citations=`{row.get('observedCitationCount')}`, spans=`{row.get('observedSpanRows')}`, "
                f"payloadHash=`{row.get('payloadHash')}`",
            ]
        )
    external = dict(report.get("externalRejection") or {})
    if external:
        lines.extend(
            [
                "",
                "## External Rejection",
                "",
                f"- pass: `{external.get('pass')}`",
                f"- observedExitCode: `{external.get('observedExitCode')}`",
                f"- observedErrorHash: `{external.get('observedErrorHash')}`",
            ]
        )
    lines.extend(["", "## Mutation Guarantees", ""])
    for field in ZERO_COUNTER_FIELDS:
        lines.append(f"- {field}: `{counts.get(field)}`")
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_OUTPUT_CAPTURE_SCHEMA_ID",
    "READY_DECISION",
    "build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture",
    "write_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture",
]
