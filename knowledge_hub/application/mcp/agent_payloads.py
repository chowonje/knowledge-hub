from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Callable

from knowledge_hub.application.agent_gateway import build_gateway_metadata
from knowledge_hub.core.sanitizer import redact_payload

AGENT_CONTEXT_PACKET_SCHEMA = "knowledge-hub.agent.context-packet.v1"
AGENT_SOURCE_TEXT_ROLE = "evidence_not_instruction"
PLAYBOOK_SCHEMA = "knowledge-hub.foundry.agent.run.playbook.v1"
VALID_AGENT_ROLES = {"planner", "researcher", "analyst", "summarizer", "auditor", "coach"}
VALID_ORCHESTRATOR_MODES = {"single-pass", "adaptive", "strict"}
VALID_PLAYBOOK_TOOLS = {"ask_knowledge", "search_knowledge", "build_task_context"}
_RAW_TEXT_KEYS = {
    "body",
    "content",
    "full_text",
    "fulltext",
    "note_body",
    "notebody",
    "raw",
    "raw_body",
    "rawbody",
    "raw_text",
    "rawtext",
}
_MACOS_USER_ROOT = "/" + "Users"
_FILE_URI_RE = re.compile(r"file://[^\s\"'<>)]*", re.IGNORECASE)
_LOCAL_USER_PATH_RE = re.compile(rf"{re.escape(_MACOS_USER_ROOT)}/[^/\s\"'<>)]*(/[^\s\"'<>)]*)?")
_ARTIFACT_PATH_RE = re.compile(rf"(?:~|{re.escape(_MACOS_USER_ROOT)}/[^/]+)/(?:(?:\.khub/)?runs|artifacts|eval/knowledgeos/runs)/[^\s\"'<>)]*")
_SECRET_VALUE_RE = re.compile(r"\b(?:sk|xox[baprs]|gh[pousr])-[A-Za-z0-9_=-]{6,}\b")


def default_agent_policy(
    *,
    classification: str = "P2",
    allow_external: bool = False,
    policy_mode: str = "local-only",
    policy_allowed: bool = True,
    policy_errors: list[str] | None = None,
    stage_allowed: bool = False,
    writeback_allowed: bool = False,
) -> dict[str, object]:
    errors = [str(item) for item in (policy_errors or []) if str(item).strip()]
    return {
        "allowExternal": bool(allow_external),
        "policyMode": str(policy_mode or "local-only"),
        "classification": str(classification or "P2").upper(),
        "policyAllowed": bool(policy_allowed and not errors),
        "externalSendAllowed": False,
        "stageAllowed": bool(stage_allowed),
        "writebackAllowed": bool(writeback_allowed),
        "finalApplyAllowed": False,
        "errors": errors,
    }


def _redact_local_path_token(match: re.Match[str]) -> str:
    token = match.group(0)
    tail = token.split("/")[-1]
    if tail and "." in tail:
        return f"~/[REDACTED_LOCAL_PATH]/{tail}"
    return "~/[REDACTED_LOCAL_PATH]"


def sanitize_agent_string(value: str) -> tuple[str, bool]:
    text = str(value)
    sanitized = redact_payload(text)
    if not isinstance(sanitized, str):
        sanitized = text
    home = str(Path.home())
    if home and home in sanitized:
        sanitized = sanitized.replace(home, "~")
    sanitized = _FILE_URI_RE.sub("[REDACTED_FILE_URI]", sanitized)
    sanitized = _ARTIFACT_PATH_RE.sub("~/[REDACTED_ARTIFACT_PATH]", sanitized)
    sanitized = _LOCAL_USER_PATH_RE.sub(_redact_local_path_token, sanitized)
    sanitized = _SECRET_VALUE_RE.sub("[REDACTED_SECRET]", sanitized)
    return sanitized, sanitized != text


def sanitize_agent_packet_value(value: Any, *, key: str = "") -> tuple[Any, bool]:
    normalized_key = key.replace("-", "_").replace(" ", "_").lower()
    if normalized_key in _RAW_TEXT_KEYS and value not in (None, "", [], {}):
        return "[OMITTED_RAW_TEXT]", True
    if isinstance(value, str):
        return sanitize_agent_string(value)
    if isinstance(value, dict):
        changed = False
        output: dict[str, Any] = {}
        redacted = redact_payload(value)
        source = redacted if isinstance(redacted, dict) else value
        if source is not value:
            changed = True
        for nested_key, nested_value in source.items():
            sanitized, nested_changed = sanitize_agent_packet_value(nested_value, key=str(nested_key))
            output[str(nested_key)] = sanitized
            changed = changed or nested_changed
        return output, changed
    if isinstance(value, list):
        changed = False
        output = []
        for item in value:
            sanitized, nested_changed = sanitize_agent_packet_value(item)
            output.append(sanitized)
            changed = changed or nested_changed
        return output, changed
    if isinstance(value, tuple):
        sanitized_list, changed = sanitize_agent_packet_value(list(value))
        return tuple(sanitized_list), changed
    return value, False


def _contract_present(value: object | None) -> bool:
    return isinstance(value, dict) and bool(value)


def _verification_is_unsafe(verification_verdict: dict[str, object] | None) -> bool:
    if not isinstance(verification_verdict, dict) or not verification_verdict:
        return False
    verdict = str(
        verification_verdict.get("verdict")
        or verification_verdict.get("status")
        or verification_verdict.get("result")
        or ""
    ).strip().lower()
    recommended = str(
        verification_verdict.get("recommended_action")
        or verification_verdict.get("recommendedAction")
        or verification_verdict.get("action")
        or ""
    ).strip().lower()
    unsupported_count = verification_verdict.get("unsupportedClaimCount")
    if unsupported_count is None:
        unsupported_count = verification_verdict.get("unsupported_claim_count")
    try:
        unsupported = int(unsupported_count or 0)
    except Exception:
        unsupported = 0
    unsupported_claims = verification_verdict.get("unsupportedClaims") or verification_verdict.get("unsupported_claims")
    if isinstance(unsupported_claims, list):
        unsupported = max(unsupported, len(unsupported_claims))
    return verdict in {"fail", "failed", "abstain", "blocked"} or recommended in {"abstain", "block"} or unsupported > 0


def infer_agent_safety(
    *,
    policy: dict[str, object],
    evidence_packet_contract: dict[str, object] | None = None,
    answer_contract: dict[str, object] | None = None,
    verification_verdict: dict[str, object] | None = None,
    require_answer_contracts: bool = False,
) -> tuple[bool, bool, list[str]]:
    warnings: list[str] = []
    if not bool(policy.get("policyAllowed", True)):
        warnings.extend(str(item) for item in policy.get("errors", []) if str(item).strip())
        warnings.append("policy gate blocked this packet")
    if bool(policy.get("externalSendAllowed", False)):
        warnings.append("external sends are disabled for agent runtime v1")
    if require_answer_contracts and not _contract_present(evidence_packet_contract):
        warnings.append("missing evidencePacketContract for answer-mode agent packet")
    if require_answer_contracts and not _contract_present(answer_contract):
        warnings.append("missing answerContract for answer-mode agent packet")
    if _verification_is_unsafe(verification_verdict):
        warnings.append("verification verdict requires human review")

    safe = not warnings
    required_review = not safe
    return safe, required_review, warnings


def build_agent_context_packet(
    *,
    request_id: str,
    tool: str,
    goal: str = "",
    query: str = "",
    policy: dict[str, object] | None = None,
    context: dict[str, object] | None = None,
    evidence_packet_contract: dict[str, object] | None = None,
    answer_contract: dict[str, object] | None = None,
    verification_verdict: dict[str, object] | None = None,
    safe_to_use: bool | None = None,
    required_human_review: bool | None = None,
    warnings: list[str] | None = None,
    next_actions: list[str] | None = None,
    require_answer_contracts: bool = False,
    stage_only: bool = False,
    final_apply: bool = False,
    source_text_role: str = AGENT_SOURCE_TEXT_ROLE,
    redaction_applied: bool | None = None,
    blocked_reason: str | None = None,
) -> dict[str, object]:
    packet_policy = policy or default_agent_policy()
    inferred_safe, inferred_review, inferred_warnings = infer_agent_safety(
        policy=packet_policy,
        evidence_packet_contract=evidence_packet_contract,
        answer_contract=answer_contract,
        verification_verdict=verification_verdict,
        require_answer_contracts=require_answer_contracts,
    )
    merged_warnings: list[str] = []
    for item in [*(warnings or []), *inferred_warnings]:
        text = str(item).strip()
        if text and text not in merged_warnings:
            merged_warnings.append(text)

    safe = inferred_safe if safe_to_use is None else bool(safe_to_use)
    review = inferred_review if required_human_review is None else bool(required_human_review)
    if merged_warnings and safe_to_use is None:
        safe = False
    if not safe and required_human_review is None:
        review = True
    sanitized_context, context_redacted = sanitize_agent_packet_value(context or {})
    sanitized_evidence_packet, evidence_redacted = sanitize_agent_packet_value(evidence_packet_contract or {})
    sanitized_answer_contract, answer_redacted = sanitize_agent_packet_value(answer_contract or {})
    sanitized_verification, verification_redacted = sanitize_agent_packet_value(verification_verdict or {})
    sanitized_goal, goal_redacted = sanitize_agent_string(str(goal or ""))
    sanitized_query, query_redacted = sanitize_agent_string(str(query or ""))
    sanitizer_changed = any(
        [
            context_redacted,
            evidence_redacted,
            answer_redacted,
            verification_redacted,
            goal_redacted,
            query_redacted,
        ]
    )
    if redaction_applied is None:
        redaction_applied = sanitizer_changed
    resolved_blocked_reason = str(blocked_reason or "").strip()
    if not resolved_blocked_reason and not safe:
        policy_errors = packet_policy.get("errors")
        if isinstance(policy_errors, list):
            resolved_blocked_reason = next((str(item) for item in policy_errors if str(item).strip()), "")
        if not resolved_blocked_reason:
            resolved_blocked_reason = next((item for item in merged_warnings if item), "not_safe_to_use")

    return {
        "schema": AGENT_CONTEXT_PACKET_SCHEMA,
        "requestId": str(request_id or ""),
        "tool": str(tool or ""),
        "goal": sanitized_goal,
        "query": sanitized_query,
        "policy": packet_policy,
        "context": sanitized_context,
        "evidencePacketContract": sanitized_evidence_packet,
        "answerContract": sanitized_answer_contract,
        "verificationVerdict": sanitized_verification,
        "safeToUse": bool(safe),
        "requiredHumanReview": bool(review),
        "stageOnly": bool(stage_only),
        "finalApply": bool(final_apply),
        "sourceTextRole": AGENT_SOURCE_TEXT_ROLE if source_text_role != AGENT_SOURCE_TEXT_ROLE else source_text_role,
        "redactionApplied": bool(redaction_applied),
        "blockedReason": resolved_blocked_reason,
        "warnings": merged_warnings,
        "nextActions": [str(item) for item in (next_actions or []) if str(item).strip()],
    }


def transition_code(item: dict[str, object], stage_fallback: str = "PLAN", status_fallback: str = "STEP") -> str:
    stage = str(item.get("stage", stage_fallback)).strip().upper() or stage_fallback
    raw_status = str(item.get("status", item.get("action", item.get("step", status_fallback)))).strip().upper() or status_fallback

    if stage == "PLAN":
        status = raw_status if raw_status in {"PLAN", "SKIP"} else "PLAN"
    elif stage == "ACT":
        status = raw_status if raw_status in {"TOOL", "WRITE", "READ", "SEARCH", "ASK"} else "TOOL"
    elif stage == "VERIFY":
        if raw_status in {"PASS", "OK", "DONE", "COMPLETE", "COMPLETED"}:
            status = "PASS"
        elif raw_status in {"FAIL", "ERROR"}:
            status = "FAIL"
        elif raw_status in {"DENY", "DENIED", "BLOCKED", "BLOCK"}:
            status = "BLOCK"
        else:
            status = raw_status or "PASS"
    elif stage == "WRITEBACK":
        status = "DONE" if raw_status in {"DONE", "OK", "SUCCESS", "SUCCEEDED"} else raw_status
    else:
        status = raw_status

    return f"{stage}.{status}"


def coerce_foundry_payload(raw: str) -> dict[str, object]:
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {"status": "ok", "payload_type": "raw-text", "data": data}
    except Exception:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        for line in reversed(lines):
            try:
                data = json.loads(line)
            except Exception:
                continue
            if isinstance(data, dict):
                return data
        return {
            "status": "ok",
            "payload_type": "raw-text",
            "data": raw.strip()[:4000],
        }


def write_agent_run_report(payload: dict[str, object], report_path: str | None, source: str) -> None:
    if not report_path:
        return
    try:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema": "knowledge-hub.foundry.agent.run.report.v1",
                    "generatedAt": datetime.utcnow().isoformat(),
                    "source": source,
                    "run": payload,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        return


def _text_or_none(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_role(value: object | None) -> str:
    normalized = str(value or "planner").strip().lower()
    if normalized in VALID_AGENT_ROLES:
        return normalized
    return "planner"


def _normalize_orchestrator_mode(value: object | None) -> str:
    normalized = str(value or "adaptive").strip().lower()
    if normalized in VALID_ORCHESTRATOR_MODES:
        return normalized
    if normalized in {"single", "singlepass"}:
        return "single-pass"
    return "adaptive"


def _coerce_int(value: object | None, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return default
    return parsed if parsed > 0 else default


def _normalize_string_list(value: object | None) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = _text_or_none(item)
        if text:
            items.append(text)
    return items


def _default_step_fields(tool_name: str) -> tuple[str, str]:
    if tool_name == "build_task_context":
        return (
            "assemble read-only task context",
            "combine persistent knowledge with ephemeral workspace evidence before synthesis",
        )
    if tool_name == "search_knowledge":
        return (
            "collect evidence",
            "gather candidate sources before synthesis",
        )
    return (
        "synthesize answer",
        "build answer from collected evidence",
    )


def _default_playbook_steps(plan: list[str]) -> list[dict[str, object]]:
    steps: list[dict[str, object]] = []
    for index, tool_name in enumerate(plan):
        tool = _text_or_none(tool_name)
        if not tool or tool not in VALID_PLAYBOOK_TOOLS:
            continue
        objective, rationale = _default_step_fields(tool)
        steps.append(
            {
                "order": index + 1,
                "tool": tool,
                "objective": objective,
                "rationale": rationale,
            }
        )
    return steps


def _normalize_playbook(
    value: object | None,
    *,
    goal: str,
    role: str,
    orchestrator_mode: str,
    max_rounds: int,
    source: str,
    plan: list[str],
    now: str,
) -> dict[str, object]:
    raw = value if isinstance(value, dict) else {}
    assumptions = _normalize_string_list(raw.get("assumptions"))
    warnings = _normalize_string_list(raw.get("warnings"))

    raw_steps = raw.get("steps")
    normalized_steps: list[dict[str, object]] = []
    if isinstance(raw_steps, list):
        for index, item in enumerate(raw_steps):
            if not isinstance(item, dict):
                continue
            tool = _text_or_none(item.get("tool"))
            if not tool or tool not in VALID_PLAYBOOK_TOOLS:
                continue
            objective = _text_or_none(item.get("objective"))
            rationale = _text_or_none(item.get("rationale"))
            if not objective or not rationale:
                objective, rationale = _default_step_fields(tool)
            order = _coerce_int(item.get("order"), index + 1)
            normalized = {
                "order": order,
                "tool": tool,
                "objective": objective,
                "rationale": rationale,
            }
            inputs = item.get("inputs")
            if isinstance(inputs, dict) and inputs:
                normalized["inputs"] = inputs
            normalized_steps.append(normalized)

    if not normalized_steps:
        normalized_steps = _default_playbook_steps(plan)

    return {
        "schema": PLAYBOOK_SCHEMA,
        "source": _text_or_none(raw.get("source")) or source,
        "goal": _text_or_none(raw.get("goal")) or goal,
        "role": _normalize_role(raw.get("role") or role),
        "orchestratorMode": _normalize_orchestrator_mode(raw.get("orchestratorMode") or raw.get("orchestrator_mode") or orchestrator_mode),
        "maxRounds": _coerce_int(raw.get("maxRounds") or raw.get("max_rounds"), max_rounds),
        "assumptions": assumptions,
        "warnings": warnings,
        "steps": normalized_steps,
        "generatedAt": _text_or_none(raw.get("generatedAt")) or now,
    }


def _normalize_transition_item(
    item: dict[str, object],
    *,
    base_source: str,
    now: str,
    transition_code_fn: Callable[[dict[str, object], str, str], str],
) -> dict[str, object]:
    message = _text_or_none(item.get("message", item.get("action", item.get("step", "")))) or "step"
    normalized: dict[str, object] = {
        "stage": str(item.get("stage", "PLAN")).strip().upper() or "PLAN",
        "status": str(item.get("status", item.get("action", item.get("step", "STEP")))).strip().upper() or "STEP",
        "message": message,
        "at": _text_or_none(item.get("at")) or now,
    }
    for field in ("tool", "step", "action", "source"):
        value = _text_or_none(item.get(field))
        if value:
            normalized[field] = value
    if "source" not in normalized and base_source:
        normalized["source"] = base_source
    code = _text_or_none(item.get("code")) or transition_code_fn(normalized)
    stage, _, status = code.partition(".")
    normalized["stage"] = stage or "PLAN"
    normalized["status"] = status or "STEP"
    normalized["code"] = code
    return normalized


def _normalize_artifact(value: object | None, *, now: str) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        return {
            "jsonContent": value,
            "classification": "P2",
            "generatedAt": now,
        }

    classification = str(value.get("classification", "P2")).strip().upper()
    if classification not in {"P0", "P1", "P2", "P3"}:
        classification = "P2"

    artifact: dict[str, object] = {
        "jsonContent": value.get("jsonContent") if "jsonContent" in value else value,
        "classification": classification,
        "generatedAt": _text_or_none(value.get("generatedAt")) or now,
    }
    artifact_id = _text_or_none(value.get("id"))
    if artifact_id:
        artifact["id"] = artifact_id
    metadata = value.get("metadata")
    if isinstance(metadata, dict) and metadata:
        artifact["metadata"] = metadata
    return artifact


def _local_only_agent_run_policy(*, decision_source: str, requested: bool | None = False) -> dict[str, object]:
    return {
        "contractRole": "agent_run_external_policy",
        "surface": "mcp-agent-run",
        "scope": "agent_run",
        "allowExternal": False,
        "allowExternalRequested": requested,
        "externalSendAllowed": False,
        "decisionSource": str(decision_source or "local_only_default"),
        "policyMode": "local-only",
        "mode": "local-only",
    }


def _normalize_agent_run_external_policy(payload: dict[str, object], *, default_source: str) -> tuple[dict[str, object], bool, list[str]]:
    raw = payload.get("externalPolicy")
    explicit = isinstance(raw, dict)
    if explicit:
        allow_external = bool(raw.get("allowExternal", raw.get("allow_external", False)))
        external_send_allowed = bool(raw.get("externalSendAllowed", raw.get("external_send_allowed", allow_external)))
        policy_mode = str(
            raw.get("policyMode")
            or raw.get("mode")
            or ("external-allowed" if allow_external else "local-only")
        ).strip().lower()
        requested = raw.get("allowExternalRequested", raw.get("allow_external_requested", allow_external))
        policy = {
            "contractRole": str(raw.get("contractRole") or "agent_run_external_policy"),
            "surface": str(raw.get("surface") or "mcp-agent-run"),
            "scope": str(raw.get("scope") or "agent_run"),
            "allowExternal": allow_external,
            "allowExternalRequested": bool(requested) if requested is not None else None,
            "externalSendAllowed": external_send_allowed,
            "decisionSource": str(raw.get("decisionSource") or raw.get("decision_source") or "delegated_payload"),
            "policyMode": policy_mode,
            "mode": str(raw.get("mode") or policy_mode).strip().lower(),
        }
    else:
        policy = _local_only_agent_run_policy(decision_source="missing_delegated_policy")

    errors: list[str] = []
    local_only_ok = (
        explicit
        and bool(policy.get("allowExternal")) is False
        and bool(policy.get("externalSendAllowed")) is False
        and str(policy.get("policyMode") or "").strip().lower() == "local-only"
    )
    source = str(payload.get("source") or default_source or "")
    delegated = source.startswith("foundry-core/")
    if delegated and not explicit:
        errors.append("agent run blocked: delegated payload missing local-only externalPolicy")
    if bool(policy.get("allowExternal")) or bool(policy.get("externalSendAllowed")) or str(policy.get("policyMode") or "").strip().lower() != "local-only":
        errors.append("agent run blocked: external calls are disabled for agent runtime v1")
    if not delegated and not explicit:
        local_only_ok = True
        policy = _local_only_agent_run_policy(decision_source="knowledge_hub_fallback_local_only")
    return policy, local_only_ok, errors


def normalize_foundry_payload(
    payload: dict[str, object],
    *,
    goal: str,
    max_rounds: int,
    dry_run: bool,
    default_source: str,
    evaluate_policy_gate_fn: Callable[[object | None], tuple[bool, list[str], str]],
    transition_code_fn: Callable[[dict[str, object], str, str], str] = transition_code,
) -> dict[str, object]:
    now = datetime.utcnow().isoformat()
    run_id = str(payload.get("runId") or payload.get("run_id") or f"foundry_{int(datetime.utcnow().timestamp())}")
    role = _normalize_role(payload.get("role"))
    orchestrator_mode = _normalize_orchestrator_mode(payload.get("orchestratorMode", payload.get("orchestrator_mode")))
    raw_status = str(payload.get("status", "")).upper().strip()
    if raw_status in {"RUNNING", "COMPLETED", "BLOCKED", "FAILED"}:
        status = raw_status.lower()
    elif raw_status in {"DONE", "OK", "SUCCESS", "DRY_RUN_OK", "VERIFY_OK"}:
        status = "completed"
    elif raw_status.startswith("DRY_RUN"):
        status = "blocked" if dry_run else "failed"
    elif "FAIL" in raw_status or "ERROR" in raw_status:
        status = "failed"
    elif not raw_status:
        status = "running" if not dry_run else "blocked"
    elif dry_run:
        status = "blocked"
    else:
        status = "completed"

    stage = str(payload.get("stage", "DONE" if status in {"completed", "blocked", "failed"} else "PLAN")).upper()
    source = str(payload.get("source", default_source))

    plan = payload.get("plan")
    if not isinstance(plan, list):
        trace_like = payload.get("trace")
        if isinstance(trace_like, list):
            plan = [
                str(step.get("step", step.get("action", "")))
                for step in trace_like
                if isinstance(step, dict) and (step.get("step") or step.get("action"))
            ]
        else:
            plan = []
    plan = [str(value) for value in plan if value]

    raw_transitions = payload.get("transitions")
    if not isinstance(raw_transitions, list):
        raw_transitions = payload.get("trace", [])

    base_source = str(payload.get("source", source))
    transitions: list[dict[str, object]] = []
    if isinstance(raw_transitions, list):
        for item in raw_transitions:
            if not isinstance(item, dict):
                continue
            transitions.append(
                _normalize_transition_item(
                    item,
                    base_source=base_source,
                    now=now,
                    transition_code_fn=transition_code_fn,
                )
            )
    if not transitions:
        transitions = [
            {
                "stage": "PLAN",
                "status": "SKIP",
                "message": "no transitions",
                "tool": "knowledge-hub-fallback",
                "source": source,
                "code": "PLAN.SKIP",
                "at": now,
            }
        ]

    verify_raw = payload.get("verify")
    is_completed = status == "completed"
    if isinstance(verify_raw, dict):
        verify = {
            "allowed": bool(verify_raw.get("allowed", is_completed)),
            "schemaValid": bool(verify_raw.get("schemaValid", verify_raw.get("schema_valid", is_completed))),
            "policyAllowed": bool(verify_raw.get("policyAllowed", verify_raw.get("policy_allowed", is_completed))),
            "schemaErrors": verify_raw.get("schemaErrors", verify_raw.get("errors", [])) or [],
        }
    else:
        verify = {
            "allowed": is_completed,
            "schemaValid": is_completed,
            "policyAllowed": is_completed,
            "schemaErrors": payload.get("errors", []),
        }
    if not is_completed:
        verify["allowed"] = False
        verify["schemaValid"] = False

    writeback_raw = payload.get("writeback")
    if isinstance(writeback_raw, dict):
        writeback = {
            "ok": bool(writeback_raw.get("ok", status in {"completed", "blocked"})),
            "detail": str(writeback_raw.get("detail", "")),
        }
    else:
        writeback = {
            "ok": bool(status in {"completed", "blocked"}),
            "detail": str(payload.get("writeback", "") if not isinstance(payload.get("writeback"), dict) else ""),
        }

    artifact = _normalize_artifact(payload.get("artifact"), now=now)
    external_policy, external_policy_ok, external_policy_errors = _normalize_agent_run_external_policy(
        payload,
        default_source=source,
    )

    policy_allowed, policy_errors, _artifact_classification = evaluate_policy_gate_fn(artifact)
    if not policy_allowed:
        status = "blocked"
        stage = "VERIFY"
        artifact = (
            {
                "id": artifact.get("id") if isinstance(artifact, dict) else None,
                "jsonContent": "[REDACTED_BY_POLICY]",
                "classification": "P0",
                "generatedAt": now,
            }
            if artifact is not None
            else None
        )

    merged_errors = list(verify.get("schemaErrors") or [])
    if policy_errors:
        merged_errors.extend(policy_errors)
    if external_policy_errors:
        merged_errors.extend(external_policy_errors)
        status = "blocked"
        stage = "VERIFY"

    combined_policy_allowed = bool(policy_allowed and external_policy_ok)
    verify["policyAllowed"] = combined_policy_allowed
    verify["allowed"] = bool(verify.get("allowed", status == "completed")) and combined_policy_allowed
    verify["schemaValid"] = bool(verify.get("schemaValid", status == "completed")) and combined_policy_allowed
    verify["schemaErrors"] = merged_errors

    if policy_errors or external_policy_errors:
        writeback["ok"] = False
        writeback["detail"] = "policy gate blocked"

    normalized: dict[str, object] = {
        "schema": "knowledge-hub.foundry.agent.run.result.v1",
        "source": source,
        "runId": run_id,
        "status": status,
        "goal": str(payload.get("goal", goal)),
        "role": role,
        "orchestratorMode": orchestrator_mode,
        "stage": stage,
        "plan": plan,
        "playbook": _normalize_playbook(
            payload.get("playbook"),
            goal=str(payload.get("goal", goal)),
            role=role,
            orchestrator_mode=orchestrator_mode,
            max_rounds=_coerce_int(payload.get("maxRounds", max_rounds), max_rounds),
            source=source,
            plan=plan,
            now=now,
        ),
        "transitions": transitions,
        "verify": verify,
        "writeback": writeback,
        "externalPolicy": external_policy,
        "artifact": artifact,
        "createdAt": str(payload.get("createdAt", now)),
        "updatedAt": str(payload.get("updatedAt", now)),
        "dryRun": bool(payload.get("dryRun", payload.get("dry_run", dry_run))),
        "maxRounds": _coerce_int(payload.get("maxRounds", payload.get("max_rounds", max_rounds)), max_rounds),
    }
    if bool(normalized["dryRun"]):
        normalized["gateway"] = build_gateway_metadata(surface="agent_run", mode="dry_run")
    tool_value = _text_or_none(payload.get("tool"))
    if tool_value:
        normalized["tool"] = tool_value
    return normalized


def format_agent_result_text(payload: dict[str, object], compact: bool = False) -> str:
    transitions = payload.get("transitions", [])
    lines = [
        f"[runId] {payload.get('runId')}",
        f"[status] {payload.get('status')}",
        f"[stage] {payload.get('stage')}",
        f"[goal] {payload.get('goal')}",
    ]

    plan = payload.get("plan") or []
    if isinstance(plan, list) and plan:
        lines.append(f"[plan] {' -> '.join([str(step) for step in plan])}")

    verify = payload.get("verify")
    if isinstance(verify, dict):
        if not bool(verify.get("allowed", False)):
            lines.append(f"[verify] blocked: {verify.get('schemaErrors', [])}")
        else:
            lines.append("[verify] allowed")

    if isinstance(transitions, list) and transitions:
        lines.append("[trace]")
        limit = 3 if compact else 20
        for item in transitions[:limit]:
            if not isinstance(item, dict):
                continue
            at = item.get("at", "")
            stage = item.get("stage", "")
            status = item.get("status", "")
            message = item.get("message", item.get("action", item.get("step", "")))
            tool = item.get("tool")
            step = item.get("step")
            tool_text = f" [tool={tool}]" if tool else ""
            step_text = f" [step={step}]" if step else ""
            lines.append(f"- {at} {stage}.{status}:{tool_text}{step_text} {message}")

    artifact = payload.get("artifact")
    if isinstance(artifact, dict):
        content = artifact.get("jsonContent", "")
        content_preview = str(content)[:1200 if not compact else 300]
        if content_preview:
            lines.append(f"[artifact] {content_preview}")
        if compact:
            errors = []
            if isinstance(payload.get("verify"), dict):
                errors = payload["verify"].get("schemaErrors", [])
            if errors:
                lines.append(f"[errors] {'; '.join([str(error) for error in errors])}")
    return "\n".join(lines)


def build_fallback_agent_payload(
    *,
    goal: str,
    max_rounds: int,
    dry_run: bool,
    plan: list[str],
    artifact: Any,
    verify_ok: bool,
    errors: list[str],
    trace: list[dict[str, Any]],
    role: str | None = None,
    orchestrator_mode: str | None = None,
    playbook: dict[str, object] | None = None,
    source: str,
    evaluate_policy_gate_fn: Callable[[object | None], tuple[bool, list[str], str]],
    transition_code_fn: Callable[[dict[str, object], str, str], str] = transition_code,
) -> str:
    now = datetime.utcnow().isoformat()
    transitions = []
    for item in trace:
        normalized_item = {
            "stage": str(item.get("stage", "PLAN")).upper(),
            "status": str(item.get("status", item.get("action", item.get("step", "STEP"))).upper()).strip() or "STEP",
            "message": str(item.get("message", item.get("action", item.get("step", "")))),
            "tool": str(item.get("tool", "knowledge-hub-fallback")),
            "step": item.get("step"),
            "action": item.get("action"),
            "at": now,
        }
        code = transition_code_fn(normalized_item)
        normalized_item["code"] = code
        normalized_item["status"] = code.split(".", 1)[-1]
        normalized_item["stage"] = code.split(".", 1)[0]
        transitions.append(normalized_item)
    if not transitions:
        transitions = [
            {
                "stage": "PLAN",
                "status": "SKIP",
                "message": "fallback-no-steps",
                "tool": "knowledge-hub-fallback",
                "code": "PLAN.SKIP",
                "at": now,
            }
        ]
    artifact_payload = {
        "id": f"fallback_artifact_{int(datetime.utcnow().timestamp())}",
        "jsonContent": artifact,
        "classification": "P2",
        "generatedAt": now,
    }
    policy_allowed, policy_errors, _artifact_classification = evaluate_policy_gate_fn(artifact_payload)
    status = "completed" if verify_ok and policy_allowed else "blocked" if (dry_run or not policy_allowed) else "failed"
    if not policy_allowed:
        errors.extend(policy_errors)
        artifact_payload["jsonContent"] = "[REDACTED_BY_POLICY]"
        artifact_payload["classification"] = "P0"
    stage = "VERIFY" if not policy_allowed else ("DONE" if verify_ok else "FAILED")
    normalized_role = _normalize_role(role)
    normalized_orchestrator_mode = _normalize_orchestrator_mode(orchestrator_mode)
    return json.dumps(
        {
            "schema": "knowledge-hub.foundry.agent.run.result.v1",
            "source": source,
            "runId": f"fallback_{int(datetime.utcnow().timestamp())}",
            "goal": goal,
            "role": normalized_role,
            "orchestratorMode": normalized_orchestrator_mode,
            "maxRounds": max_rounds,
            "status": status,
            "plan": plan,
            "stage": stage,
            "playbook": _normalize_playbook(
                playbook,
                goal=goal,
                role=normalized_role,
                orchestrator_mode=normalized_orchestrator_mode,
                max_rounds=max_rounds,
                source=source,
                plan=plan,
                now=now,
            ),
            "transitions": transitions,
            "verify": {
                "allowed": verify_ok and policy_allowed and not errors,
                "schemaValid": verify_ok and policy_allowed and not errors,
                "policyAllowed": policy_allowed,
                "schemaErrors": errors,
            },
            "artifact": artifact_payload,
            "writeback": {
                "ok": bool(verify_ok and policy_allowed),
                "detail": "fallback writeback skipped" if dry_run else "fallback writeback completed",
            },
            "createdAt": now,
            "updatedAt": now,
            "dryRun": dry_run,
        },
        ensure_ascii=False,
    )
