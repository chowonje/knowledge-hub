from __future__ import annotations

from pathlib import Path
from typing import Any

from knowledge_hub.application.paper_query_export import DEFAULT_QUERY_MODEL
from knowledge_hub.application.paper_query_run import query_text_for_id, validate_query_embedding_run
from knowledge_hub.application.paper_retrieval_harness import DEFAULT_QWEN8_NAMESPACE, retrieve_paper_evidence_pack
from knowledge_hub.application.task_context import build_task_context, classify_task_mode


def _synthesize_from_task_context(searcher: Any, goal: str, task_context: dict[str, Any]) -> dict[str, Any]:
    llm = getattr(searcher, "llm", None)
    generate = getattr(llm, "generate", None)
    if callable(generate):
        try:
            answer = str(
                generate(
                    prompt=(
                        "Use the provided task context to answer the goal. "
                        "Treat workspace evidence as ephemeral project context and do not invent missing code details."
                    ),
                    context=str(task_context.get("suggested_prompt_context", "")),
                    max_tokens=1200,
                )
            ).strip()
        except Exception:
            answer = ""
        if answer:
            return {
                "answer": answer,
                "sources": list(task_context.get("knowledge_hits", [])),
                "warnings": list(task_context.get("warnings", [])),
                "synthesisMode": "task_context_llm",
            }

    answer_fn = getattr(searcher, "generate_answer", None)
    if callable(answer_fn):
        fallback = answer_fn(goal, top_k=5, allow_external=False)
        normalized = fallback if isinstance(fallback, dict) else {"answer": str(fallback), "sources": []}
        warnings = list(normalized.get("warnings", []))
        warnings.append("workspace context was not synthesized directly; fell back to ask_knowledge")
        normalized["warnings"] = warnings
        normalized["synthesisMode"] = "ask_knowledge_fallback"
        return normalized
    return {"answer": "", "sources": [], "warnings": ["no synthesis runtime available"], "synthesisMode": "none"}


def _paper_arg(arguments: dict[str, Any], camel_key: str, snake_key: str = "") -> str:
    value = arguments.get(camel_key)
    if value is None and snake_key:
        value = arguments.get(snake_key)
    return str(value or "").strip()


def _blocked_paper_evidence_pack(
    *,
    query: str,
    blockers: list[str] | tuple[str, ...],
    qwen_namespace: str,
    qwen_model: str,
    run_dir: str,
    query_id: str,
) -> dict[str, Any]:
    return {
        "schema": "knowledge-hub.labs.paper-retrieval-harness.v1",
        "status": "blocked",
        "query": query,
        "topK": 0,
        "blockers": list(blockers),
        "arms": [
            {
                "name": "qwen8",
                "status": "blocked",
                "namespace": qwen_namespace,
                "model": qwen_model,
                "resultCount": 0,
                "blockers": list(blockers),
                "queryEmbeddingSource": "artifact",
            }
        ],
        "evidence": [],
        "runtimeDiagnostics": {"queryRunDir": run_dir, "queryId": query_id, "qwenNamespace": qwen_namespace},
    }


def _build_paper_evidence_pack(arguments: dict[str, Any], searcher: Any) -> dict[str, Any] | None:
    paper_query_run = _paper_arg(arguments, "paperQueryRun", "paper_query_run")
    if not paper_query_run:
        return None
    paper_query_id = _paper_arg(arguments, "paperQueryId", "paper_query_id")
    qwen_namespace = _paper_arg(arguments, "paperQwenNamespace", "paper_qwen_namespace") or DEFAULT_QWEN8_NAMESPACE
    qwen_model = _paper_arg(arguments, "paperQwenModel", "paper_qwen_model") or DEFAULT_QUERY_MODEL
    validation = validate_query_embedding_run(Path(paper_query_run).expanduser(), qwen_model)
    if validation.status == "blocked":
        return _blocked_paper_evidence_pack(
            query="",
            blockers=validation.blockers,
            qwen_namespace=qwen_namespace,
            qwen_model=qwen_model,
            run_dir=paper_query_run,
            query_id=paper_query_id,
        )
    query = query_text_for_id(validation.query_rows, paper_query_id)
    if not query:
        return _blocked_paper_evidence_pack(
            query="",
            blockers=["query_id_not_found"],
            qwen_namespace=qwen_namespace,
            qwen_model=qwen_model,
            run_dir=paper_query_run,
            query_id=paper_query_id,
        )
    payload = retrieve_paper_evidence_pack(
        khub=searcher,
        query=query,
        use_bge=True,
        use_keyword=True,
        use_qwen8=True,
        qwen_query_embeddings_path=validation.query_embedding_path,
        qwen_query_id=paper_query_id,
        qwen_namespace=qwen_namespace,
        qwen_model=qwen_model,
        top_k=5,
    )
    diagnostics = dict(payload.get("runtimeDiagnostics") or {})
    diagnostics["queryRunDir"] = paper_query_run
    diagnostics["queryId"] = paper_query_id
    payload["runtimeDiagnostics"] = diagnostics
    return payload


def _attach_paper_evidence(artifact: Any, paper_evidence_pack: dict[str, Any] | None) -> Any:
    if not paper_evidence_pack:
        return artifact
    if isinstance(artifact, dict):
        merged = dict(artifact)
    else:
        merged = {"answer": str(artifact or ""), "sources": []}
    merged["paperEvidencePack"] = paper_evidence_pack
    if paper_evidence_pack.get("status") == "blocked":
        warnings = [str(item) for item in list(merged.get("warnings") or [])]
        blockers = ", ".join(str(item) for item in list(paper_evidence_pack.get("blockers") or []))
        warnings.append(f"qwen8 paper evidence blocked: {blockers}")
        merged["warnings"] = warnings
    return merged


async def handle_tool(name: str, arguments: dict[str, Any], ctx: dict[str, Any]):
    if name != "run_agentic_query":
        return None

    emit = ctx["emit"]
    to_bool = ctx["to_bool"]
    to_int = ctx["to_int"]
    run_async_tool = ctx["run_async_tool"]
    request_echo = ctx["request_echo"]
    searcher = ctx["searcher"]

    run_foundry_agent_goal = ctx["run_foundry_agent_goal"]
    coerce_foundry_payload = ctx["coerce_foundry_payload"]
    normalize_foundry_payload = ctx["normalize_foundry_payload"]
    write_agent_run_report = ctx["write_agent_run_report"]
    build_fallback_agent_payload = ctx["build_fallback_agent_payload"]

    status_failed = ctx["MCP_TOOL_STATUS_FAILED"]
    status_queued = ctx["MCP_TOOL_STATUS_QUEUED"]

    goal = str(arguments.get("goal", "")).strip()
    if not goal:
        return emit(status_failed, {"error": "goal이 필요합니다."}, status_message="goal required")

    max_rounds = to_int(arguments.get("max_rounds"), 2, minimum=1, maximum=20)
    role = str(arguments.get("role", "planner"))
    if role not in {"planner", "researcher", "analyst", "coach", "summarizer"}:
        role = "planner"
    report_path = arguments.get("reportPath") if isinstance(arguments.get("reportPath"), str) else None
    orchestrator_mode = arguments.get("orchestratorMode")
    dry_run = to_bool(arguments.get("dry_run"), default=False)
    dump_json = to_bool(arguments.get("dump_json"), default=False)
    mode = classify_task_mode(goal)
    repo_path = str(arguments.get("repo_path", "") or "").strip() or None
    max_workspace_files = to_int(arguments.get("max_workspace_files"), 8, minimum=1, maximum=32) or 8
    include_workspace_arg = arguments.get("include_workspace")
    include_workspace = (
        to_bool(include_workspace_arg, default=True)
        if include_workspace_arg is not None
        else bool(repo_path and mode in {"coding", "design", "debug"})
    )
    paper_evidence_pack = _build_paper_evidence_pack(arguments, searcher)

    async def _runner() -> dict[str, Any]:
        delegated, delegated_err = run_foundry_agent_goal(
            goal=goal,
            max_rounds=max_rounds,
            role=role,
            report_path=report_path,
            orchestrator_mode=str(orchestrator_mode or "adaptive"),
            dry_run=dry_run,
            dump_json=dump_json,
            repo_path=repo_path,
            include_workspace=include_workspace,
            max_workspace_files=max_workspace_files,
        )
        if delegated:
            payload = coerce_foundry_payload(delegated)
            if paper_evidence_pack:
                artifact = payload.get("artifact")
                if isinstance(artifact, dict):
                    updated_artifact = dict(artifact)
                    updated_artifact["jsonContent"] = _attach_paper_evidence(
                        updated_artifact.get("jsonContent"),
                        paper_evidence_pack,
                    )
                    payload["artifact"] = updated_artifact
                else:
                    payload["artifact"] = {
                        "jsonContent": _attach_paper_evidence(artifact, paper_evidence_pack),
                        "classification": "P2",
                    }
                transitions = list(payload.get("transitions") or payload.get("trace") or [])
                transitions.append(
                    {
                        "stage": "ACT",
                        "status": "PAPER_HARNESS_QWEN8",
                        "message": f"paper_harness_qwen8 status={paper_evidence_pack.get('status')}",
                    }
                )
                payload["transitions"] = transitions
            normalized = normalize_foundry_payload(payload, goal=goal, max_rounds=max_rounds, dry_run=dry_run)
            normalized["source"] = "foundry-core/cli-agent"
            if report_path:
                write_agent_run_report(normalized, report_path, "foundry-core/cli-agent")
            return normalized

        errors = [f"foundry bridge unavailable: {delegated_err}"] if delegated_err else []
        fallback_tokens = goal.lower()
        include_search = (
            role in {"planner", "researcher", "analyst", "coach"}
            or any(
                k in fallback_tokens
                for k in ["비교", "compare", "차이", "대조", "찾아", "검색", "search", "목록", "리스트", "추천"]
            )
            or str(orchestrator_mode or "").lower() == "strict"
        )
        if role == "summarizer":
            include_search = False

        trace: list[dict[str, Any]] = []
        artifact: Any = None
        verify_ok = True
        if mode in {"coding", "design", "debug"}:
            plan = ["build_task_context", "ask_knowledge"]
            trace.append({"stage": "PLAN", "step": "build_task_context"})
            task_context = build_task_context(
                searcher,
                goal=goal,
                repo_path=repo_path,
                include_workspace=include_workspace,
                include_vault=True,
                include_papers=True,
                include_web=True,
                max_workspace_files=max_workspace_files,
                max_knowledge_hits=5,
            )
            trace.append(
                {
                    "stage": "ACT",
                    "step": "build_task_context",
                    "workspace_count": len(task_context.get("workspace_files", [])),
                    "knowledge_count": len(task_context.get("knowledge_hits", [])),
                }
            )
            synthesis = _synthesize_from_task_context(searcher, goal, task_context)
            artifact = {
                "mode": mode,
                "taskContext": task_context,
                "persistentKnowledgeEvidence": task_context.get("knowledge_hits", []),
                "workspaceEvidence": task_context.get("workspace_files", []),
                "answer": synthesis.get("answer", ""),
                "sources": synthesis.get("sources", []),
                "warnings": synthesis.get("warnings", []),
                "synthesisMode": synthesis.get("synthesisMode", ""),
            }
            artifact = _attach_paper_evidence(artifact, paper_evidence_pack)
            if paper_evidence_pack:
                trace.append(
                    {
                        "stage": "ACT",
                        "step": "paper_harness_qwen8",
                        "status": paper_evidence_pack.get("status"),
                    }
                )
            trace.append(
                {
                    "stage": "ACT",
                    "step": "ask_knowledge",
                    "has_answer": bool(str(artifact.get("answer", "")).strip()),
                    "synthesis_mode": artifact.get("synthesisMode", ""),
                }
            )
            if not str(artifact.get("answer", "")).strip():
                verify_ok = False
                errors.append("task-context synthesis produced empty artifact")
        else:
            plan = ["search_knowledge", "ask_knowledge"] if include_search else ["ask_knowledge"]
            for step in plan:
                trace.append({"stage": "PLAN", "step": step})
                if step == "search_knowledge":
                    search_results = searcher.search(goal, top_k=5)
                    artifact = [
                        {
                            "title": r.metadata.get("title", "Untitled"),
                            "score": r.score,
                            "source_type": r.metadata.get("source_type", ""),
                        }
                        for r in search_results
                    ]
                    trace.append({"stage": "ACT", "step": step, "count": len(artifact)})
                else:
                    answer = searcher.generate_answer(goal, top_k=5, allow_external=False)
                    artifact = answer
                    if isinstance(answer, dict):
                        trace.append({"stage": "ACT", "step": step, "has_answer": bool(answer.get("answer"))})
                    else:
                        trace.append({"stage": "ACT", "step": step, "has_answer": bool(str(answer))})
                    if not artifact:
                        verify_ok = False
                        errors.append(f"{step} produced empty artifact")
            artifact = _attach_paper_evidence(artifact, paper_evidence_pack)
            if paper_evidence_pack:
                trace.append(
                    {
                        "stage": "ACT",
                        "step": "paper_harness_qwen8",
                        "status": paper_evidence_pack.get("status"),
                    }
                )

        fallback = build_fallback_agent_payload(
            goal=goal,
            max_rounds=max_rounds,
            dry_run=dry_run,
            plan=plan,
            artifact=artifact,
            verify_ok=verify_ok,
            errors=errors,
            trace=trace,
            role=role,
            orchestrator_mode=str(orchestrator_mode or "adaptive"),
            playbook={
                "schema": "knowledge-hub.foundry.agent.run.playbook.v1",
                "source": "knowledge-hub/mcp_server.fallback",
                "goal": goal,
                "role": role,
                "orchestratorMode": orchestrator_mode or "adaptive",
                "maxRounds": max_rounds,
                "plan": plan,
                "taskMode": mode,
                "inputs": {
                    "repoPath": repo_path or "",
                    "includeWorkspace": include_workspace,
                    "maxWorkspaceFiles": max_workspace_files,
                },
            },
        )
        payload = normalize_foundry_payload(
            coerce_foundry_payload(fallback),
            goal=goal,
            max_rounds=max_rounds,
            dry_run=dry_run,
        )
        if report_path:
            write_agent_run_report(payload, report_path, "knowledge-hub/mcp_server.fallback")
        return payload

    job_id, queued = await run_async_tool(name=name, request_echo=request_echo, sync_job=_runner)
    return emit(status_queued, queued["payload"], job_id=job_id, status_message="queued")
