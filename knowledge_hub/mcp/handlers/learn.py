from __future__ import annotations

from typing import Any


def _topic_session(arguments: dict[str, Any]) -> tuple[str, str]:
    return str(arguments.get("topic", "")).strip(), str(arguments.get("session_id", "")).strip()


async def handle_tool(name: str, arguments: dict[str, Any], ctx: dict[str, Any]):
    emit = ctx["emit"]
    config = ctx["config"]
    normalize_source = ctx["normalize_source"]
    to_bool = ctx["to_bool"]
    to_int = ctx["to_int"]
    learning_service_cls = ctx["LearningCoachService"]
    run_async_tool = ctx["run_async_tool"]
    request_echo = ctx["request_echo"]

    status_ok = ctx["MCP_TOOL_STATUS_OK"]
    status_failed = ctx["MCP_TOOL_STATUS_FAILED"]
    status_queued = ctx["MCP_TOOL_STATUS_QUEUED"]

    if name == "learn_map":
        topic = str(arguments.get("topic", "")).strip()
        if not topic:
            return emit(status_failed, {"error": "topic이 필요합니다."}, status_message="topic required")
        dry_run = to_bool(arguments.get("dry_run"), default=False)
        writeback = to_bool(arguments.get("writeback"), default=False) and not dry_run
        service = learning_service_cls(config)
        payload = service.map(
            topic=topic,
            source=normalize_source(arguments.get("source")),
            days=to_int(arguments.get("days"), 180, minimum=1),
            top_k=to_int(arguments.get("top_k"), 12, minimum=1),
            writeback=writeback,
            write_canvas=to_bool(arguments.get("canvas"), default=False) and not dry_run,
            allow_external=to_bool(arguments.get("allow_external"), default=False),
        )
        return emit(status_ok, payload)

    if name == "learning_start_or_resume_topic":
        topic = str(arguments.get("topic", "")).strip()
        if not topic:
            return emit(status_failed, {"error": "topic이 필요합니다."}, status_message="topic required")
        service = learning_service_cls(config)
        payload = service.start_or_resume_topic(
            topic=topic,
            force_new_session=to_bool(arguments.get("force_new_session"), default=False),
            source=normalize_source(arguments.get("source")),
            days=to_int(arguments.get("days"), 180, minimum=1),
            top_k=to_int(arguments.get("top_k"), 12, minimum=1),
            concept_count=to_int(arguments.get("concept_count"), 6, minimum=1),
        )
        return emit(status_ok, payload)

    if name == "learning_get_session_state":
        topic = str(arguments.get("topic", "")).strip()
        session_id = str(arguments.get("session_id", "")).strip()
        if not topic and not session_id:
            return emit(status_failed, {"error": "topic 또는 session_id가 필요합니다."}, status_message="topic or session_id required")
        service = learning_service_cls(config)
        payload = service.get_session_state(topic=topic or None, session_id=session_id or None)
        tool_status = status_ok if str(payload.get("status")) != "not_found" else status_failed
        return emit(tool_status, payload)

    if name == "learning_explain_topic":
        topic = str(arguments.get("topic", "")).strip()
        question = str(arguments.get("question", "")).strip()
        if not topic:
            return emit(status_failed, {"error": "topic이 필요합니다."}, status_message="topic required")
        if not question:
            return emit(status_failed, {"error": "question이 필요합니다."}, status_message="question required")
        service = learning_service_cls(config)
        payload = service.explain_topic(
            topic=topic,
            question=question,
            session_id=str(arguments.get("session_id", "")).strip() or None,
            source=normalize_source(arguments.get("source")),
            top_k=to_int(arguments.get("top_k"), 5, minimum=1),
            min_score=float(arguments.get("min_score", 0.3) or 0.3),
        )
        return emit(status_ok, payload)

    if name == "learning_checkpoint":
        topic, session_id = _topic_session(arguments)
        summary = str(arguments.get("summary", "")).strip()
        if not topic:
            return emit(status_failed, {"error": "topic이 필요합니다."}, status_message="topic required")
        if not session_id:
            return emit(status_failed, {"error": "session_id가 필요합니다."}, status_message="session_id required")
        if not summary:
            return emit(status_failed, {"error": "summary가 필요합니다."}, status_message="summary required")
        service = learning_service_cls(config)
        payload = service.checkpoint(
            topic=topic,
            session_id=session_id,
            summary=summary,
            known_items=arguments.get("known_items") if isinstance(arguments.get("known_items"), list) else [],
            shaky_items=arguments.get("shaky_items") if isinstance(arguments.get("shaky_items"), list) else [],
            unknown_items=arguments.get("unknown_items") if isinstance(arguments.get("unknown_items"), list) else [],
            misconceptions=arguments.get("misconceptions") if isinstance(arguments.get("misconceptions"), list) else [],
            writeback=to_bool(arguments.get("writeback"), default=False),
        )
        return emit(status_ok, payload)

    if name == "learn_assess_template":
        topic, session_id = _topic_session(arguments)
        if not topic:
            return emit(status_failed, {"error": "topic이 필요합니다."}, status_message="topic required")
        if not session_id:
            return emit(status_failed, {"error": "session_id가 필요합니다."}, status_message="session_id required")
        dry_run = to_bool(arguments.get("dry_run"), default=False)
        writeback = to_bool(arguments.get("writeback"), default=False) and not dry_run
        service = learning_service_cls(config)
        payload = service.assess_template(
            topic=topic,
            session_id=session_id,
            concept_count=to_int(arguments.get("concept_count"), 6, minimum=1),
            source=normalize_source(arguments.get("source")),
            days=to_int(arguments.get("days"), 180, minimum=1),
            top_k=to_int(arguments.get("top_k"), 12, minimum=1),
            writeback=writeback,
            allow_external=to_bool(arguments.get("allow_external"), default=False),
        )
        return emit(status_ok, payload)

    if name == "learn_grade":
        topic, session_id = _topic_session(arguments)
        if not topic:
            return emit(status_failed, {"error": "topic이 필요합니다."}, status_message="topic required")
        if not session_id:
            return emit(status_failed, {"error": "session_id가 필요합니다."}, status_message="session_id required")
        dry_run = to_bool(arguments.get("dry_run"), default=False)
        writeback = to_bool(arguments.get("writeback"), default=False) and not dry_run
        service = learning_service_cls(config)
        payload = service.grade(
            topic=topic,
            session_id=session_id,
            writeback=writeback,
            allow_external=to_bool(arguments.get("allow_external"), default=False),
        )
        return emit(status_ok, payload)

    if name == "learn_next":
        topic, session_id = _topic_session(arguments)
        if not topic:
            return emit(status_failed, {"error": "topic이 필요합니다."}, status_message="topic required")
        if not session_id:
            return emit(status_failed, {"error": "session_id가 필요합니다."}, status_message="session_id required")
        dry_run = to_bool(arguments.get("dry_run"), default=False)
        writeback = to_bool(arguments.get("writeback"), default=False) and not dry_run
        service = learning_service_cls(config)
        payload = service.next(
            topic=topic,
            session_id=session_id,
            source=normalize_source(arguments.get("source")),
            days=to_int(arguments.get("days"), 180, minimum=1),
            top_k=to_int(arguments.get("top_k"), 12, minimum=1),
            writeback=writeback,
            allow_external=to_bool(arguments.get("allow_external"), default=False),
        )
        return emit(status_ok, payload)

    if name == "learn_run":
        topic, session_id = _topic_session(arguments)
        if not topic:
            return emit(status_failed, {"error": "topic이 필요합니다."}, status_message="topic required")
        if not session_id:
            return emit(status_failed, {"error": "session_id가 필요합니다."}, status_message="session_id required")

        async def _runner() -> dict[str, Any]:
            service = learning_service_cls(config)
            return service.run(
                topic=topic,
                session_id=session_id,
                source=normalize_source(arguments.get("source")),
                days=to_int(arguments.get("days"), 180, minimum=1),
                top_k=to_int(arguments.get("top_k"), 12, minimum=1),
                concept_count=to_int(arguments.get("concept_count"), 6, minimum=1),
                auto_next=to_bool(arguments.get("auto_next"), default=False),
                writeback=to_bool(arguments.get("writeback"), default=False) and not to_bool(arguments.get("dry_run"), default=False),
                canvas=to_bool(arguments.get("canvas"), default=False) and not to_bool(arguments.get("dry_run"), default=False),
                allow_external=to_bool(arguments.get("allow_external"), default=False),
            )

        job_id, queued = await run_async_tool(name=name, request_echo=request_echo, sync_job=_runner)
        return emit(status_queued, queued["payload"], job_id=job_id, status_message="queued")

    if name == "learn_analyze_gaps":
        topic, session_id = _topic_session(arguments)
        if not topic:
            return emit(status_failed, {"error": "topic이 필요합니다."}, status_message="topic required")
        if not session_id:
            return emit(status_failed, {"error": "session_id가 필요합니다."}, status_message="session_id required")
        dry_run = to_bool(arguments.get("dry_run"), default=False)
        writeback = to_bool(arguments.get("writeback"), default=False) and not dry_run
        service = learning_service_cls(config)
        payload = service.analyze_gaps(
            topic=topic,
            session_id=session_id,
            source=normalize_source(arguments.get("source")),
            days=to_int(arguments.get("days"), 180, minimum=1),
            top_k=to_int(arguments.get("top_k"), 12, minimum=1),
            writeback=writeback,
            allow_external=to_bool(arguments.get("allow_external"), default=False),
        )
        return emit(status_ok, payload)

    if name == "learn_generate_quiz":
        topic, session_id = _topic_session(arguments)
        if not topic:
            return emit(status_failed, {"error": "topic이 필요합니다."}, status_message="topic required")
        if not session_id:
            return emit(status_failed, {"error": "session_id가 필요합니다."}, status_message="session_id required")
        dry_run = to_bool(arguments.get("dry_run"), default=False)
        writeback = to_bool(arguments.get("writeback"), default=False) and not dry_run
        service = learning_service_cls(config)
        payload = service.generate_quiz(
            topic=topic,
            session_id=session_id,
            source=normalize_source(arguments.get("source")),
            days=to_int(arguments.get("days"), 180, minimum=1),
            top_k=to_int(arguments.get("top_k"), 12, minimum=1),
            mix=str(arguments.get("mix", "mixed")).strip().lower(),
            question_count=to_int(arguments.get("question_count"), 6, minimum=1),
            writeback=writeback,
            allow_external=to_bool(arguments.get("allow_external"), default=False),
        )
        return emit(status_ok, payload)

    if name == "learn_grade_quiz":
        topic, session_id = _topic_session(arguments)
        if not topic:
            return emit(status_failed, {"error": "topic이 필요합니다."}, status_message="topic required")
        if not session_id:
            return emit(status_failed, {"error": "session_id가 필요합니다."}, status_message="session_id required")
        dry_run = to_bool(arguments.get("dry_run"), default=False)
        writeback = to_bool(arguments.get("writeback"), default=False) and not dry_run
        raw_answers = arguments.get("answers", [])
        answers = raw_answers if isinstance(raw_answers, list) else []
        service = learning_service_cls(config)
        payload = service.grade_quiz(
            topic=topic,
            session_id=session_id,
            answers=[item for item in answers if isinstance(item, dict)],
            source=normalize_source(arguments.get("source")),
            days=to_int(arguments.get("days"), 180, minimum=1),
            top_k=to_int(arguments.get("top_k"), 12, minimum=1),
            writeback=writeback,
            allow_external=to_bool(arguments.get("allow_external"), default=False),
        )
        return emit(status_ok, payload)

    if name == "learn_reinforce":
        topic, session_id = _topic_session(arguments)
        if not topic:
            return emit(status_failed, {"error": "topic이 필요합니다."}, status_message="topic required")
        if not session_id:
            return emit(status_failed, {"error": "session_id가 필요합니다."}, status_message="session_id required")
        dry_run = to_bool(arguments.get("dry_run"), default=False)
        writeback = to_bool(arguments.get("writeback"), default=False) and not dry_run
        service = learning_service_cls(config)
        payload = service.reinforce(
            topic=topic,
            session_id=session_id,
            source=normalize_source(arguments.get("source")),
            days=to_int(arguments.get("days"), 180, minimum=1),
            top_k=to_int(arguments.get("top_k"), 12, minimum=1),
            top_k_per_gap=to_int(arguments.get("top_k_per_gap"), 3, minimum=1),
            writeback=writeback,
            allow_external=to_bool(arguments.get("allow_external"), default=False),
        )
        return emit(status_ok, payload)

    if name == "learn_suggest_patch":
        topic, session_id = _topic_session(arguments)
        if not topic:
            return emit(status_failed, {"error": "topic이 필요합니다."}, status_message="topic required")
        if not session_id:
            return emit(status_failed, {"error": "session_id가 필요합니다."}, status_message="session_id required")
        dry_run = to_bool(arguments.get("dry_run"), default=False)
        writeback = to_bool(arguments.get("writeback"), default=False) and not dry_run
        service = learning_service_cls(config)
        payload = service.suggest_patch(
            topic=topic,
            session_id=session_id,
            source=normalize_source(arguments.get("source")),
            days=to_int(arguments.get("days"), 180, minimum=1),
            top_k=to_int(arguments.get("top_k"), 12, minimum=1),
            writeback=writeback,
            allow_external=to_bool(arguments.get("allow_external"), default=False),
        )
        return emit(status_ok, payload)

    if name == "learn_graph_build":
        topic = str(arguments.get("topic", "")).strip()
        if not topic:
            return emit(status_failed, {"error": "topic이 필요합니다."}, status_message="topic required")
        service = learning_service_cls(config)
        payload = service.build_learning_graph(
            topic=topic,
            top_k=to_int(arguments.get("top_k"), 12, minimum=1),
            allow_external=to_bool(arguments.get("allow_external"), default=False),
        )
        return emit(status_ok, payload)

    if name == "learn_graph_pending_list":
        service = learning_service_cls(config)
        payload = service.list_learning_graph_pending(
            topic=str(arguments.get("topic", "")).strip() or None,
            item_type=str(arguments.get("item_type", "all")).strip() or "all",
            limit=to_int(arguments.get("limit"), 100, minimum=1),
        )
        return emit(status_ok, payload)

    if name == "learn_graph_pending_apply":
        pending_id = to_int(arguments.get("pending_id"), 0, minimum=1)
        if pending_id < 1:
            return emit(status_failed, {"error": "pending_id가 필요합니다."}, status_message="pending_id required")
        service = learning_service_cls(config)
        payload = service.apply_learning_graph_pending(pending_id=pending_id)
        return emit(status_ok, payload)

    if name == "learn_graph_pending_reject":
        pending_id = to_int(arguments.get("pending_id"), 0, minimum=1)
        if pending_id < 1:
            return emit(status_failed, {"error": "pending_id가 필요합니다."}, status_message="pending_id required")
        service = learning_service_cls(config)
        payload = service.reject_learning_graph_pending(pending_id=pending_id)
        return emit(status_ok, payload)

    if name == "learn_path_generate":
        topic = str(arguments.get("topic", "")).strip()
        if not topic:
            return emit(status_failed, {"error": "topic이 필요합니다."}, status_message="topic required")
        service = learning_service_cls(config)
        payload = service.generate_learning_path(
            topic=topic,
            approved_only=to_bool(arguments.get("approved_only"), default=True),
            writeback=to_bool(arguments.get("writeback"), default=False),
        )
        return emit(status_ok, payload)

    if name == "run_learning_pipeline":
        topic, session_id = _topic_session(arguments)
        if not topic:
            return emit(status_failed, {"error": "topic이 필요합니다."}, status_message="topic required")
        if not session_id:
            return emit(status_failed, {"error": "session_id가 필요합니다."}, status_message="session_id required")

        async def _runner() -> dict[str, Any]:
            service = learning_service_cls(config)
            return service.run(
                topic=topic,
                session_id=session_id,
                source=normalize_source(arguments.get("source")),
                days=to_int(arguments.get("days"), 180, minimum=1),
                top_k=to_int(arguments.get("top_k"), 12, minimum=1),
                concept_count=to_int(arguments.get("concept_count"), 6, minimum=1),
                auto_next=to_bool(arguments.get("auto_next"), default=False),
                writeback=to_bool(arguments.get("writeback"), default=False) and not to_bool(arguments.get("dry_run"), default=False),
                canvas=to_bool(arguments.get("canvas"), default=False) and not to_bool(arguments.get("dry_run"), default=False),
                allow_external=to_bool(arguments.get("allow_external"), default=False),
            )

        job_id, queued = await run_async_tool(name=name, request_echo=request_echo, sync_job=_runner)
        return emit(status_queued, queued["payload"], job_id=job_id, status_message="queued")

    return None
