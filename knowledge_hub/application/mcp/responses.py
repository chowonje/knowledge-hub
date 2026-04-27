from __future__ import annotations

from datetime import datetime
import json
from typing import Any, Callable, Sequence

from mcp.types import TextContent

from knowledge_hub.core.sanitizer import classify_payload_level, redact_payload
from knowledge_hub.core.schema_validator import validate_payload

MCP_RESULT_SCHEMA = "knowledge-hub.mcp.result.v1"
MCP_TOOL_STATUS_OK = "ok"
MCP_TOOL_STATUS_QUEUED = "queued"
MCP_TOOL_STATUS_RUNNING = "running"
MCP_TOOL_STATUS_BLOCKED = "blocked"
MCP_TOOL_STATUS_FAILED = "failed"
MCP_TOOL_STATUS_DONE = "done"
MCP_TOOL_STATUS_EXPIRED = "expired"
DEFAULT_MCP_TOOL_TIMEOUT_SECONDS = 1800
POLICY_REPORT_TOOL_NAMES = {"agent_policy_check", "agent_stage_memory"}

LEARNING_TOOL_NAMES = {
    "learning_start_or_resume_topic",
    "learning_get_session_state",
    "learning_explain_topic",
    "learning_checkpoint",
    "learn_map",
    "learn_assess_template",
    "learn_grade",
    "learn_next",
    "learn_run",
    "learn_analyze_gaps",
    "learn_generate_quiz",
    "learn_grade_quiz",
    "learn_reinforce",
    "learn_suggest_patch",
    "learn_graph_build",
    "learn_graph_pending_list",
    "learn_graph_pending_apply",
    "learn_graph_pending_reject",
    "learn_path_generate",
    "run_learning_pipeline",
}
CRAWL_TOOL_NAMES = {
    "crawl_web_ingest",
    "crawl_youtube_ingest",
    "crawl_pending_list",
    "crawl_pending_apply",
    "crawl_pending_reject",
    "crawl_pipeline_run",
    "crawl_pipeline_resume",
    "crawl_pipeline_status",
    "crawl_pipeline_benchmark",
    "crawl_domain_policy_list",
    "crawl_domain_policy_apply",
    "crawl_domain_policy_reject",
}
CRAWL_LABS_TOOL_NAMES = {
    "crawl_youtube_ingest",
    "crawl_pending_list",
    "crawl_pending_apply",
    "crawl_pending_reject",
    "crawl_pipeline_run",
    "crawl_pipeline_resume",
    "crawl_pipeline_status",
    "crawl_pipeline_benchmark",
    "crawl_domain_policy_list",
    "crawl_domain_policy_apply",
    "crawl_domain_policy_reject",
}
KO_NOTE_TOOL_NAMES = {
    "ko_note_status",
    "ko_note_report",
    "ko_note_review_list",
    "ko_note_review_approve",
    "ko_note_review_reject",
    "ko_note_remediate",
}
OPS_TOOL_NAMES = {
    "rag_report",
    "ops_action_list",
    "ops_action_ack",
    "ops_action_execute",
    "ops_action_receipts",
    "ops_action_resolve",
}
TRANSFORM_TOOL_NAMES = {
    "transform_list",
    "transform_preview",
    "transform_run",
    "ask_graph",
    "notebook_workbench_search",
    "notebook_workbench_chat",
}
PAPER_LABS_TOOL_NAMES = {
    "paper_topic_synthesize",
}
ONTOLOGY_TOOL_NAMES = {
    "ontology_profile_list",
    "ontology_profile_show",
    "ontology_profile_activate",
    "ontology_profile_import",
    "ontology_profile_export",
    "ontology_proposal_submit",
    "ontology_proposal_list",
    "ontology_proposal_apply",
    "ontology_proposal_reject",
}
EPISTEMIC_TOOL_NAMES = {
    "belief_list",
    "belief_show",
    "belief_upsert",
    "belief_review",
    "decision_create",
    "decision_list",
    "decision_review",
    "outcome_record",
    "outcome_show",
}
FOUNDRY_CONFLICT_TOOL_NAMES = {
    "foundry_conflict_list",
    "foundry_conflict_apply",
    "foundry_conflict_reject",
}
ENTITY_MERGE_TOOL_NAMES = {
    "entity_merge_list",
    "entity_merge_apply",
    "entity_merge_reject",
}
AGENT_TOOL_NAMES = {
    "agent_build_context",
    "agent_search_knowledge",
    "agent_ask_knowledge",
    "agent_get_evidence",
    "agent_policy_check",
    "agent_stage_memory",
}
AGENT_CORE_ONLY_TOOL_NAMES = {
    "agent_policy_check",
    "agent_stage_memory",
}
DEFAULT_TOOL_NAMES = {
    "search_knowledge",
    "ask_knowledge",
    "build_task_context",
    "discover_and_ingest",
    "get_paper_detail",
    "paper_lookup_and_summarize",
    "get_hub_stats",
    "mcp_job_status",
    "mcp_job_list",
    "mcp_job_cancel",
}
LABS_TOOL_NAMES = (
    LEARNING_TOOL_NAMES
    | KO_NOTE_TOOL_NAMES
    | OPS_TOOL_NAMES
    | ONTOLOGY_TOOL_NAMES
    | EPISTEMIC_TOOL_NAMES
    | FOUNDRY_CONFLICT_TOOL_NAMES
    | ENTITY_MERGE_TOOL_NAMES
    | CRAWL_LABS_TOOL_NAMES
    | TRANSFORM_TOOL_NAMES
    | PAPER_LABS_TOOL_NAMES
    | AGENT_TOOL_NAMES
)
HEAVY_TOOL_NAMES = {
    "crawl_web_ingest",
    "discover_and_ingest",
    "run_paper_ingest_flow",
    "run_agentic_query",
    "index_paper_keywords",
    "learn_run",
    "run_learning_pipeline",
}
JOB_TOOL_NAMES = {
    "mcp_job_status",
    "mcp_job_list",
    "mcp_job_cancel",
    "ops_action_list",
    "ops_action_ack",
    "ops_action_execute",
    "ops_action_receipts",
    "ops_action_resolve",
}
CORE_RUNTIME_TOOL_NAMES = (
    LEARNING_TOOL_NAMES
    | CRAWL_TOOL_NAMES
    | KO_NOTE_TOOL_NAMES
    | FOUNDRY_CONFLICT_TOOL_NAMES
    | ENTITY_MERGE_TOOL_NAMES
    | OPS_TOOL_NAMES
)
CORE_ONLY_TOOL_NAMES = CORE_RUNTIME_TOOL_NAMES | AGENT_CORE_ONLY_TOOL_NAMES
JOB_TOOLS = JOB_TOOL_NAMES
SCHEMA_BY_TOOL = {
    "build_paper_memory": "knowledge-hub.paper-memory.build.result.v1",
    "get_paper_memory_card": "knowledge-hub.paper-memory.card.result.v1",
    "search_paper_memory": "knowledge-hub.paper-memory.search.result.v1",
    "build_task_context": "knowledge-hub.task-context.result.v1",
    "transform_list": "knowledge-hub.transform.list.result.v1",
    "transform_preview": "knowledge-hub.transform.preview.result.v1",
    "transform_run": "knowledge-hub.transform.run.result.v1",
    "ask_graph": "knowledge-hub.ask-graph.result.v1",
    "notebook_workbench_search": "knowledge-hub.workbench.search.result.v1",
    "notebook_workbench_chat": "knowledge-hub.workbench.chat.result.v1",
    "learn_map": "knowledge-hub.learning.map.result.v1",
    "learning_start_or_resume_topic": "knowledge-hub.learning.start-resume.result.v1",
    "learning_get_session_state": "knowledge-hub.learning.session-state.result.v1",
    "learning_explain_topic": "knowledge-hub.learning.explain.result.v1",
    "learning_checkpoint": "knowledge-hub.learning.checkpoint.result.v1",
    "learn_assess_template": "knowledge-hub.learning.template.result.v1",
    "learn_grade": "knowledge-hub.learning.grade.result.v1",
    "learn_next": "knowledge-hub.learning.next.result.v1",
    "learn_run": "knowledge-hub.learning.run.result.v1",
    "learn_analyze_gaps": "knowledge-hub.learning.gap.result.v1",
    "learn_generate_quiz": "knowledge-hub.learning.quiz.generate.result.v1",
    "learn_grade_quiz": "knowledge-hub.learning.quiz.grade.result.v1",
    "learn_suggest_patch": "knowledge-hub.learning.patch.suggest.result.v1",
    "learn_graph_build": "knowledge-hub.learning.graph.build.result.v1",
    "learn_graph_pending_list": "knowledge-hub.learning.graph.pending.result.v1",
    "learn_graph_pending_apply": "knowledge-hub.learning.graph.pending.result.v1",
    "learn_graph_pending_reject": "knowledge-hub.learning.graph.pending.result.v1",
    "learn_path_generate": "knowledge-hub.learning.path.result.v1",
    "run_learning_pipeline": "knowledge-hub.learning.run.result.v1",
    "crawl_web_ingest": "knowledge-hub.crawl.ingest.result.v1",
    "crawl_youtube_ingest": "knowledge-hub.crawl.ingest.result.v1",
    "crawl_pending_list": "knowledge-hub.crawl.pending.list.result.v1",
    "crawl_pending_apply": "knowledge-hub.crawl.pending.apply.result.v1",
    "crawl_pending_reject": "knowledge-hub.crawl.pending.reject.result.v1",
    "crawl_pipeline_run": "knowledge-hub.crawl.pipeline.run.result.v1",
    "crawl_pipeline_resume": "knowledge-hub.crawl.pipeline.run.result.v1",
    "crawl_pipeline_benchmark": "knowledge-hub.crawl.benchmark.result.v1",
    "crawl_domain_policy_list": "knowledge-hub.crawl.domain.policy.result.v1",
    "crawl_domain_policy_apply": "knowledge-hub.crawl.domain.policy.result.v1",
    "crawl_domain_policy_reject": "knowledge-hub.crawl.domain.policy.result.v1",
    "ko_note_status": "knowledge-hub.ko-note.status.result.v1",
    "ko_note_report": "knowledge-hub.ko-note.report.result.v1",
    "ko_note_review_list": "knowledge-hub.ko-note.review.list.result.v1",
    "ko_note_review_approve": "knowledge-hub.ko-note.review.result.v1",
    "ko_note_review_reject": "knowledge-hub.ko-note.review.result.v1",
    "ko_note_remediate": "knowledge-hub.ko-note.remediate.result.v1",
    "rag_report": "knowledge-hub.rag.report.result.v1",
    "ops_action_list": "knowledge-hub.ops.action.list.result.v1",
    "ops_action_ack": "knowledge-hub.ops.action.result.v1",
    "ops_action_execute": "knowledge-hub.ops.action.execute.result.v1",
    "ops_action_receipts": "knowledge-hub.ops.action.receipts.result.v1",
    "ops_action_resolve": "knowledge-hub.ops.action.result.v1",
    "entity_merge_list": "knowledge-hub.entity.merge.list.result.v1",
    "entity_merge_apply": "knowledge-hub.entity.merge.apply.result.v1",
    "entity_merge_reject": "knowledge-hub.entity.merge.reject.result.v1",
    "paper_topic_synthesize": "knowledge-hub.paper-topic-synthesis.result.v1",
    "agent_build_context": "knowledge-hub.agent.context-packet.v1",
    "agent_search_knowledge": "knowledge-hub.agent.context-packet.v1",
    "agent_ask_knowledge": "knowledge-hub.agent.context-packet.v1",
    "agent_get_evidence": "knowledge-hub.agent.context-packet.v1",
    "agent_policy_check": "knowledge-hub.agent.context-packet.v1",
    "agent_stage_memory": "knowledge-hub.agent.context-packet.v1",
}


def payload_text(payload: dict[str, Any], compact: bool = False) -> str:
    if compact:
        return json.dumps(payload, ensure_ascii=False)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _normalize_classification(value: object | None) -> str:
    normalized = str(value or "P2").strip().upper()
    if normalized in {"P0", "P1", "P2", "P3"}:
        return normalized
    return "P2"


_CLASSIFICATION_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


def _most_sensitive_classification(*values: str) -> str:
    best = "P3"
    best_rank = _CLASSIFICATION_RANK[best]
    for value in values:
        normalized = _normalize_classification(value)
        rank = _CLASSIFICATION_RANK.get(normalized, _CLASSIFICATION_RANK["P2"])
        if rank < best_rank:
            best = normalized
            best_rank = rank
    return best


def normalize_classification(value: object | None) -> str:
    return _normalize_classification(value)


_POLICY_METADATA_KEYS = {
    "classification",
    "policyclass",
    "policy_class",
}


def _extract_text_fragments_for_policy(value: Any) -> list[str]:
    fragments: list[str] = []
    if value is None:
        return fragments
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, dict):
        for key, nested_value in value.items():
            if str(key or "").replace("-", "_").replace(" ", "_").lower() in _POLICY_METADATA_KEYS:
                continue
            fragments.extend(_extract_text_fragments_for_policy(nested_value))
        return fragments
    if isinstance(value, (list, tuple, set)):
        for nested_value in value:
            fragments.extend(_extract_text_fragments_for_policy(nested_value))
        return fragments
    return [str(value)]


def evaluate_policy_gate(artifact: object | None) -> tuple[bool, list[str], str]:
    if artifact is None:
        return True, [], "P2"

    payload = artifact if isinstance(artifact, dict) else {"value": artifact}
    declared_classifications: list[str] = []
    policy_node = payload.get("policy")
    if isinstance(policy_node, dict):
        explicit = _normalize_classification(policy_node.get("classification"))
        if explicit == "P0":
            return False, ["policy denied: P0 policy node detected"], explicit
        declared_classifications.append(explicit)

    explicit = _normalize_classification(payload.get("classification"))
    if explicit == "P0":
        return False, ["policy denied: P0 artifact classification"], explicit
    declared_classifications.append(explicit)

    detected = classify_payload_level(_extract_text_fragments_for_policy(payload))
    if detected == "P0":
        return False, ["policy denied: detected potential P0 data"], "P0"

    return True, [], _most_sensitive_classification(*declared_classifications, detected)


def now_iso() -> str:
    return datetime.utcnow().isoformat()


def to_int(value: Any, default: int | None, minimum: int | None = None, maximum: int | None = None) -> int | None:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    if parsed is None:
        return parsed
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def to_float(value: Any, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    bool_text = str(value).strip().lower()
    if bool_text in {"1", "true", "yes", "y", "on"}:
        return True
    if bool_text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def normalize_source(value: Any) -> str:
    source = str(value or "all").strip().lower()
    if source not in {"all", "note", "paper", "web"}:
        return "all"
    return source


def to_tool_status(record_status: str | None) -> str:
    status = str(record_status or MCP_TOOL_STATUS_FAILED)
    lowered = status.lower()
    if lowered in {
        MCP_TOOL_STATUS_QUEUED,
        MCP_TOOL_STATUS_RUNNING,
        MCP_TOOL_STATUS_OK,
        MCP_TOOL_STATUS_DONE,
        MCP_TOOL_STATUS_FAILED,
        MCP_TOOL_STATUS_BLOCKED,
        MCP_TOOL_STATUS_EXPIRED,
    }:
        if lowered == MCP_TOOL_STATUS_DONE:
            return MCP_TOOL_STATUS_OK
        return lowered
    if lowered in {"completed", "success", "finished"}:
        return MCP_TOOL_STATUS_OK
    if lowered in {"error", "failure", "exception"}:
        return MCP_TOOL_STATUS_FAILED
    return MCP_TOOL_STATUS_FAILED


def _coerce_status_message(status: str, error: str | None = None) -> str:
    if error:
        return error
    return {
        MCP_TOOL_STATUS_OK: "completed",
        MCP_TOOL_STATUS_QUEUED: "queued",
        MCP_TOOL_STATUS_RUNNING: "running",
        MCP_TOOL_STATUS_BLOCKED: "blocked",
        MCP_TOOL_STATUS_FAILED: "failed",
        MCP_TOOL_STATUS_EXPIRED: "expired",
    }.get(status, str(status))


def infer_classification(payload: Any) -> str:
    return evaluate_policy_gate(payload)[2]


def collect_source_refs(payload: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(payload, dict):
        for key in ("sourceRefs", "sources", "source_refs"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        refs.append(item)
                    elif isinstance(item, dict):
                        for field in ("id", "url", "title", "path", "file_path"):
                            candidate = item.get(field)
                            if isinstance(candidate, str):
                                refs.append(candidate)
        transitions = payload.get("transitions")
        if isinstance(transitions, list):
            for transition in transitions:
                if isinstance(transition, dict):
                    for field in ("source", "url", "step", "tool"):
                        candidate = transition.get(field)
                        if isinstance(candidate, str) and candidate not in refs:
                            refs.append(candidate)
    return refs[:20]


def infer_next_actions(tool: str, status: str, *, job_id: str | None = None) -> list[dict[str, Any]]:
    if status in {MCP_TOOL_STATUS_QUEUED, MCP_TOOL_STATUS_RUNNING} and job_id:
        return [{"tool": "mcp_job_status", "reason": "poll", "arguments": {"job_id": job_id}}]
    if status == MCP_TOOL_STATUS_OK and tool == "mcp_job_status":
        return [{"tool": "mcp_job_list", "reason": "check other active jobs"}]
    return []


def build_verify_block(
    payload: dict[str, Any] | Any,
    status: str,
    tool_name: str,
    *,
    validate_payload_fn: Callable[[dict[str, Any], str], Any] | None = None,
) -> dict[str, Any]:
    materialized = payload if isinstance(payload, dict) else {}
    policy_report = tool_name in POLICY_REPORT_TOOL_NAMES and materialized.get("schema") == "knowledge-hub.agent.context-packet.v1"
    policy_allowed, policy_errors, inferred_classification = evaluate_policy_gate(materialized)
    if policy_report:
        policy_node = materialized.get("policy")
        if isinstance(policy_node, dict):
            policy_allowed = bool(policy_node.get("policyAllowed", True))
            policy_errors = [str(item) for item in list(policy_node.get("errors") or []) if str(item).strip()]
            inferred_classification = _normalize_classification(policy_node.get("classification"))
    schema_id = SCHEMA_BY_TOOL.get(tool_name)
    if not schema_id:
        payload_schema = materialized.get("schema")
        if isinstance(payload_schema, str) and payload_schema.strip():
            schema_id = payload_schema.strip()
    schema_errors: list[str] = []
    schema_valid = True
    if schema_id:
        validator = validate_payload_fn or (lambda current_payload, current_schema_id: validate_payload(current_payload, current_schema_id, strict=True))
        result = validator(materialized, schema_id)
        schema_valid = bool(result.ok)
        schema_errors = list(result.errors or [])

    if schema_errors:
        materialized.setdefault("schemaErrors", [])
        if isinstance(materialized.get("schemaErrors"), list):
            for item in schema_errors:
                if item not in materialized["schemaErrors"]:
                    materialized["schemaErrors"].append(item)
    if not policy_report:
        for item in policy_errors:
            if item not in schema_errors:
                schema_errors.append(item)
                if isinstance(materialized.get("schemaErrors"), list):
                    materialized["schemaErrors"].append(item)

    if status in {MCP_TOOL_STATUS_FAILED, MCP_TOOL_STATUS_EXPIRED, MCP_TOOL_STATUS_BLOCKED}:
        schema_valid = False

    profile_block = bool(status == MCP_TOOL_STATUS_BLOCKED and materialized.get("blockReason") == "profile")
    if status == MCP_TOOL_STATUS_BLOCKED and not profile_block:
        policy_allowed = False
        schema_valid = False

    allowed = bool(status == MCP_TOOL_STATUS_OK and schema_valid and policy_allowed)
    return {
        "schemaValid": bool(schema_valid if policy_report else schema_valid and policy_allowed),
        "policyAllowed": bool(policy_allowed),
        "allowed": allowed,
        "schemaErrors": schema_errors,
        "classification": inferred_classification,
    }


def build_mcp_tool_response(
    *,
    tool: str,
    status: str,
    payload: dict[str, Any] | Any,
    request_echo: dict[str, Any] | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    job_id: str | None = None,
    status_message: str | None = None,
    artifact: Any | None = None,
    source_refs: Sequence[str] | None = None,
    build_verify_block_fn: Callable[[dict[str, Any] | Any, str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    safe_payload = payload if isinstance(payload, dict) else {"value": payload}
    now = now_iso()
    started = started_at or now
    if finished_at is None and status in {MCP_TOOL_STATUS_OK, MCP_TOOL_STATUS_BLOCKED, MCP_TOOL_STATUS_FAILED}:
        finished_at = now
    if status not in {MCP_TOOL_STATUS_QUEUED, MCP_TOOL_STATUS_RUNNING} and finished_at:
        try:
            started_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
            finished_dt = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
            run_time_ms = int((finished_dt - started_dt).total_seconds() * 1000)
        except Exception:
            run_time_ms = None
    else:
        run_time_ms = None

    policy_report = tool in POLICY_REPORT_TOOL_NAMES and safe_payload.get("schema") == "knowledge-hub.agent.context-packet.v1"
    verify = (build_verify_block_fn or build_verify_block)(safe_payload, status, tool)
    policy_allowed, policy_errors, classification = evaluate_policy_gate(artifact if artifact is not None else safe_payload)
    if policy_report:
        policy_node = safe_payload.get("policy")
        if isinstance(policy_node, dict):
            policy_allowed = bool(policy_node.get("policyAllowed", True))
            policy_errors = [str(item) for item in list(policy_node.get("errors") or []) if str(item).strip()]
            classification = _normalize_classification(policy_node.get("classification"))
    policy_blocked = False if policy_report else (not verify["policyAllowed"]) or (not policy_allowed)
    if not policy_allowed:
        verify["policyAllowed"] = False
        verify["allowed"] = False
        if not policy_report:
            verify["schemaValid"] = False
            for error in policy_errors:
                if error not in verify["schemaErrors"]:
                    verify["schemaErrors"].append(error)

    safe_request = redact_payload(request_echo or {})
    warnings = [str(item) for item in verify["schemaErrors"]] if verify["schemaErrors"] else []
    if policy_report:
        for error in policy_errors:
            if error not in warnings:
                warnings.append(error)
    payload_for_artifact = safe_payload if artifact is None else artifact
    if policy_blocked:
        status = MCP_TOOL_STATUS_BLOCKED
        payload_for_artifact = "[REDACTED_BY_POLICY]"
        verify["allowed"] = False
        verify["policyAllowed"] = False
        verify["schemaValid"] = False
        if not verify["schemaErrors"]:
            warnings.append("policy blocked")
        safe_payload = {
            "status": "blocked",
            "reason": "policy denied",
            "message": "policy redaction applied",
        }

    artifact_dict = {
        "id": f"{tool}_{(job_id or now).replace('-', '').replace(':', '').replace('.', '')[-12:]}",
        "jsonContent": payload_for_artifact,
        "classification": classification if not policy_blocked else "P0",
        "generatedAt": now,
    }

    if verify["schemaErrors"]:
        for error in verify["schemaErrors"]:
            if str(error) not in warnings:
                warnings.append(str(error))

    if status == MCP_TOOL_STATUS_OK:
        verify["allowed"] = bool(verify["policyAllowed"] and verify["schemaValid"])

    return {
        "schema": MCP_RESULT_SCHEMA,
        "tool": tool,
        "status": status,
        "statusMessage": status_message or _coerce_status_message(status),
        "jobId": job_id,
        "startedAt": started,
        "finishedAt": finished_at,
        "runTimeMs": run_time_ms,
        "payload": safe_payload,
        "artifact": artifact_dict,
        "verify": verify,
        "warnings": warnings,
        "warningsCount": len(warnings),
        "requestEcho": safe_request,
        "sourceRefs": list(source_refs or collect_source_refs(safe_payload)),
        "nextSuggestedActions": infer_next_actions(tool, status, job_id=job_id),
    }


def build_text_response(result: dict[str, Any], compact: bool = False) -> list[TextContent]:
    return [TextContent(type="text", text=payload_text(result, compact=compact))]
