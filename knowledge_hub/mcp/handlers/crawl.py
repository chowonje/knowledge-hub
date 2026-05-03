from __future__ import annotations

import asyncio
from typing import Any

from knowledge_hub.application.ko_note_reports import build_ko_note_report
from knowledge_hub.notes import KoNoteEnricher, KoNoteMaterializer


async def _queue_async_tool(run_async_tool, *, name: str, request_echo: dict[str, Any], sync_job: Any):
    try:
        return await run_async_tool(tool=name, request_echo=request_echo, sync_job=sync_job)
    except TypeError as error:
        if "unexpected keyword argument 'tool'" not in str(error):
            raise
        return await run_async_tool(name=name, request_echo=request_echo, sync_job=sync_job)


async def handle_tool(name: str, arguments: dict[str, Any], ctx: dict[str, Any]):
    emit = ctx["emit"]
    config = ctx["config"]
    to_bool = ctx["to_bool"]
    to_int = ctx["to_int"]
    to_float = ctx["to_float"]
    run_async_tool = ctx["run_async_tool"]
    request_echo = ctx["request_echo"]
    learning_service_cls = ctx["LearningCoachService"]
    web_ingest_service_cls = ctx["WebIngestService"]
    ko_note_materializer_cls = ctx.get("KoNoteMaterializer") or KoNoteMaterializer
    ko_note_enricher_cls = ctx.get("KoNoteEnricher") or KoNoteEnricher
    sqlite_db = ctx.get("sqlite_db")

    status_ok = ctx["MCP_TOOL_STATUS_OK"]
    status_failed = ctx["MCP_TOOL_STATUS_FAILED"]
    status_queued = ctx["MCP_TOOL_STATUS_QUEUED"]

    if name == "ko_note_status":
        run_id = str(arguments.get("run_id", "")).strip()
        if not run_id:
            return emit(status_failed, {"error": "run_id가 필요합니다."}, status_message="missing run_id")
        materializer = ko_note_materializer_cls(config, sqlite_db=sqlite_db)
        payload = await asyncio.to_thread(materializer.status, run_id=run_id)
        return emit(status_ok, payload)

    if name == "ko_note_report":
        run_id = str(arguments.get("run_id", "")).strip()
        if not run_id:
            return emit(status_failed, {"error": "run_id가 필요합니다."}, status_message="missing run_id")
        materializer = ko_note_materializer_cls(config, sqlite_db=sqlite_db)
        payload = await asyncio.to_thread(
            build_ko_note_report,
            materializer.sqlite_db,
            run_id=run_id,
            recent_runs=to_int(arguments.get("recent_runs"), 10, minimum=1, maximum=100),
        )
        tool_status = status_ok if str(payload.get("status")) == "ok" else status_failed
        return emit(tool_status, payload)

    if name == "ko_note_review_list":
        run_id = str(arguments.get("run_id", "")).strip()
        if not run_id:
            return emit(status_failed, {"error": "run_id가 필요합니다."}, status_message="missing run_id")
        materializer = ko_note_materializer_cls(config, sqlite_db=sqlite_db)
        payload = await asyncio.to_thread(
            materializer.review_list,
            run_id=run_id,
            item_type=str(arguments.get("item_type", "all")).strip() or "all",
            quality_flag=str(arguments.get("quality_flag", "all")).strip() or "all",
            limit=to_int(arguments.get("limit"), 50, minimum=1, maximum=500),
        )
        return emit(status_ok, payload)

    if name == "ko_note_review_approve":
        item_id = to_int(arguments.get("item_id"), None, minimum=1)
        if item_id is None:
            return emit(status_failed, {"error": "item_id가 필요합니다."}, status_message="missing item_id")
        materializer = ko_note_materializer_cls(config, sqlite_db=sqlite_db)
        payload = await asyncio.to_thread(
            materializer.review_approve,
            item_id=item_id,
            reviewer=str(arguments.get("reviewer", "mcp-user")).strip() or "mcp-user",
            note=str(arguments.get("note", "")).strip(),
        )
        tool_status = status_ok if str(payload.get("status")) == "ok" else status_failed
        return emit(tool_status, payload)

    if name == "ko_note_review_reject":
        item_id = to_int(arguments.get("item_id"), None, minimum=1)
        if item_id is None:
            return emit(status_failed, {"error": "item_id가 필요합니다."}, status_message="missing item_id")
        materializer = ko_note_materializer_cls(config, sqlite_db=sqlite_db)
        payload = await asyncio.to_thread(
            materializer.review_reject,
            item_id=item_id,
            reviewer=str(arguments.get("reviewer", "mcp-user")).strip() or "mcp-user",
            note=str(arguments.get("note", "")).strip(),
        )
        tool_status = status_ok if str(payload.get("status")) == "ok" else status_failed
        return emit(tool_status, payload)

    if name == "ko_note_remediate":
        run_id = str(arguments.get("run_id", "")).strip()
        if not run_id:
            return emit(status_failed, {"error": "run_id가 필요합니다."}, status_message="missing run_id")

        async def _runner() -> dict[str, Any]:
            enricher = ko_note_enricher_cls(config, sqlite_db=sqlite_db)
            return await asyncio.to_thread(
                enricher.remediate,
                run_id=run_id,
                item_type=str(arguments.get("item_type", "all")).strip() or "all",
                quality_flag=str(arguments.get("quality_flag", "all")).strip() or "all",
                item_id=to_int(arguments.get("item_id"), 0, minimum=0),
                limit=to_int(arguments.get("limit"), 50, minimum=1, maximum=500),
                strategy=str(arguments.get("strategy", "section")).strip() or "section",
                allow_external=to_bool(arguments.get("allow_external"), default=False),
                llm_mode=str(arguments.get("llm_mode", "auto")).strip() or "auto",
            )

        job_id, queued = await _queue_async_tool(run_async_tool, name=name, request_echo=request_echo, sync_job=_runner)
        return emit(status_queued, queued["payload"], job_id=job_id, status_message="queued")

    if name in {"crawl_web_ingest", "crawl_youtube_ingest"}:
        raw_urls = arguments.get("urls", [])
        if isinstance(raw_urls, str):
            urls = [raw_urls]
        elif isinstance(raw_urls, list):
            urls = [str(item).strip() for item in raw_urls if str(item).strip()]
        else:
            urls = []
        if not urls:
            return emit(status_failed, {"error": "urls가 필요합니다."}, status_message="urls required")

        topic = str(arguments.get("topic", "")).strip()
        learn_map = to_bool(arguments.get("learn_map"), default=False)
        is_youtube = name == "crawl_youtube_ingest"

        async def _runner() -> dict[str, Any]:
            service = web_ingest_service_cls(config)
            payload = await asyncio.to_thread(
                service.crawl_and_ingest,
                urls=urls,
                topic=topic,
                engine=("youtube" if is_youtube else str(arguments.get("engine", "auto"))),
                timeout=to_int(arguments.get("timeout"), 30 if is_youtube else 15, minimum=1),
                delay=to_float(arguments.get("delay"), 0.0 if is_youtube else 0.5, minimum=0.0),
                index=to_bool(arguments.get("index"), default=True),
                extract_concepts=to_bool(arguments.get("extract_concepts"), default=True),
                allow_external=to_bool(arguments.get("allow_external"), default=False),
                writeback=to_bool(arguments.get("writeback"), default=False),
                input_source=("youtube" if is_youtube else "web"),
                transcript_language=(str(arguments.get("transcript_language", "")).strip() or None),
                asr_model=str(arguments.get("asr_model", "tiny")).strip() or "tiny",
                index_autofix_mode=("youtube_single_retry" if is_youtube else "none"),
                concept_threshold=to_float(arguments.get("concept_threshold"), 0.78, minimum=0.0),
                relation_threshold=to_float(arguments.get("relation_threshold"), 0.75, minimum=0.0),
                emit_ontology_graph=to_bool(arguments.get("emit_ontology_graph"), default=False),
                ontology_ttl_path=arguments.get("ontology_ttl_path"),
                validate_ontology_graph=to_bool(arguments.get("validate_ontology_graph"), default=False),
            )
            if learn_map and topic:
                learn_svc = learning_service_cls(config)
                map_result = learn_svc.map(
                    topic=topic,
                    source="all",
                    days=180,
                    top_k=12,
                    writeback=False,
                    allow_external=False,
                )
                payload["learningMap"] = {
                    "status": map_result.get("status"),
                    "trunkCount": len(map_result.get("trunks") or []),
                    "branchCount": len(map_result.get("branches") or []),
                    "schema": map_result.get("schema"),
                }
            return payload

        job_id, queued = await _queue_async_tool(run_async_tool, name=name, request_echo=request_echo, sync_job=_runner)
        return emit(status_queued, queued["payload"], job_id=job_id, status_message="queued")

    if name == "crawl_pipeline_run":
        raw_urls = arguments.get("urls", [])
        if isinstance(raw_urls, str):
            urls = [raw_urls]
        elif isinstance(raw_urls, list):
            urls = [str(item).strip() for item in raw_urls if str(item).strip()]
        else:
            urls = []
        if not urls:
            return emit(status_failed, {"error": "urls가 필요합니다."}, status_message="urls required")

        async def _runner() -> dict[str, Any]:
            service = web_ingest_service_cls(config)
            return await asyncio.to_thread(
                service.run_pipeline,
                urls=urls,
                topic=str(arguments.get("topic", "")).strip(),
                source=str(arguments.get("source", "web")).strip() or "web",
                profile=str(arguments.get("profile", "safe")).strip() or "safe",
                source_policy=str(arguments.get("source_policy", "hybrid")).strip() or "hybrid",
                limit=to_int(arguments.get("limit"), 0, minimum=0),
                engine=str(arguments.get("engine", "auto")),
                timeout=to_int(arguments.get("timeout"), 15, minimum=1),
                delay=to_float(arguments.get("delay"), 0.5, minimum=0.0),
                index=to_bool(arguments.get("index"), default=True),
                extract_concepts=to_bool(arguments.get("extract_concepts"), default=True),
                allow_external=to_bool(arguments.get("allow_external"), default=False),
            )

        job_id, queued = await _queue_async_tool(run_async_tool, name=name, request_echo=request_echo, sync_job=_runner)
        return emit(status_queued, queued["payload"], job_id=job_id, status_message="queued")

    if name == "crawl_pipeline_resume":
        job_id_value = str(arguments.get("job_id", "")).strip()
        if not job_id_value:
            return emit(status_failed, {"error": "job_id가 필요합니다."}, status_message="missing job_id")

        async def _runner() -> dict[str, Any]:
            service = web_ingest_service_cls(config)
            return await asyncio.to_thread(
                service.resume_pipeline,
                job_id=job_id_value,
                profile=(str(arguments.get("profile", "")).strip() or None),
                source_policy=(str(arguments.get("source_policy", "")).strip() or None),
                limit=to_int(arguments.get("limit"), 0, minimum=0),
                engine=str(arguments.get("engine", "auto")),
                timeout=to_int(arguments.get("timeout"), 15, minimum=1),
                delay=to_float(arguments.get("delay"), 0.5, minimum=0.0),
                index=to_bool(arguments.get("index"), default=True),
                extract_concepts=to_bool(arguments.get("extract_concepts"), default=True),
                allow_external=to_bool(arguments.get("allow_external"), default=False),
            )

        job_id, queued = await _queue_async_tool(run_async_tool, name=name, request_echo=request_echo, sync_job=_runner)
        return emit(status_queued, queued["payload"], job_id=job_id, status_message="queued")

    if name == "crawl_pipeline_status":
        job_id_value = str(arguments.get("job_id", "")).strip()
        if not job_id_value:
            return emit(status_failed, {"error": "job_id가 필요합니다."}, status_message="missing job_id")
        service = web_ingest_service_cls(config)
        payload = await asyncio.to_thread(service.pipeline_status, job_id_value)
        return emit(status_ok, payload)

    if name == "crawl_domain_policy_list":
        service = web_ingest_service_cls(config)
        payload = await asyncio.to_thread(
            service.list_domain_policy,
            str(arguments.get("status", "")).strip(),
            to_int(arguments.get("limit"), 200, minimum=1),
        )
        return emit(status_ok, payload)

    if name == "crawl_domain_policy_apply":
        domain = str(arguments.get("domain", "")).strip()
        if not domain:
            return emit(status_failed, {"error": "domain이 필요합니다."}, status_message="missing domain")
        service = web_ingest_service_cls(config)
        payload = await asyncio.to_thread(
            service.apply_domain_policy,
            domain,
            str(arguments.get("reason", "")).strip(),
        )
        return emit(status_ok, payload)

    if name == "crawl_domain_policy_reject":
        domain = str(arguments.get("domain", "")).strip()
        if not domain:
            return emit(status_failed, {"error": "domain이 필요합니다."}, status_message="missing domain")
        service = web_ingest_service_cls(config)
        payload = await asyncio.to_thread(
            service.reject_domain_policy,
            domain,
            str(arguments.get("reason", "")).strip(),
        )
        return emit(status_ok, payload)

    if name == "crawl_pipeline_benchmark":
        raw_urls = arguments.get("urls", [])
        if isinstance(raw_urls, str):
            urls = [raw_urls]
        elif isinstance(raw_urls, list):
            urls = [str(item).strip() for item in raw_urls if str(item).strip()]
        else:
            urls = []
        if not urls:
            return emit(status_failed, {"error": "urls가 필요합니다."}, status_message="urls required")

        async def _runner() -> dict[str, Any]:
            service = web_ingest_service_cls(config)
            return await asyncio.to_thread(
                service.benchmark_pipeline,
                urls=urls,
                sample=to_int(arguments.get("sample"), 20, minimum=1),
                profile=str(arguments.get("profile", "safe")).strip() or "safe",
                source_policy=str(arguments.get("source_policy", "hybrid")).strip() or "hybrid",
                topic=str(arguments.get("topic", "")).strip(),
                engine=str(arguments.get("engine", "auto")),
            )

        job_id, queued = await _queue_async_tool(run_async_tool, name=name, request_echo=request_echo, sync_job=_runner)
        return emit(status_queued, queued["payload"], job_id=job_id, status_message="queued")

    if name == "crawl_pending_list":
        service = web_ingest_service_cls(config)
        payload = await asyncio.to_thread(
            service.list_pending,
            str(arguments.get("topic", "")).strip(),
            to_int(arguments.get("limit"), 50, minimum=1),
        )
        return emit(status_ok, payload)

    if name == "crawl_pending_apply":
        if "id" not in arguments:
            return emit(status_failed, {"error": "id가 필요합니다."}, status_message="missing id")
        service = web_ingest_service_cls(config)
        payload = await asyncio.to_thread(service.apply_pending, to_int(arguments.get("id"), 0, minimum=1))
        return emit(status_ok, payload)

    if name == "crawl_pending_reject":
        if "id" not in arguments:
            return emit(status_failed, {"error": "id가 필요합니다."}, status_message="missing id")
        service = web_ingest_service_cls(config)
        payload = await asyncio.to_thread(service.reject_pending, to_int(arguments.get("id"), 0, minimum=1))
        return emit(status_ok, payload)

    return None
