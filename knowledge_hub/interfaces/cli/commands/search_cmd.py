"""
khub search / ask - 벡터 검색 및 RAG 질의 (논문 + 개념 통합 검색)
"""

from __future__ import annotations

import inspect
import click
from rich.console import Console
from rich.table import Table

from knowledge_hub.application.claim_signals import build_claim_signal_payload
from knowledge_hub.application.ask_contracts import ensure_ask_contract_payload, external_policy_contract
from knowledge_hub.application.related_notes import build_related_note_suggestions
from knowledge_hub.application.runtime_diagnostics import build_runtime_diagnostics
from knowledge_hub.application.rag_reports import build_rag_ops_report
from knowledge_hub.ai.memory_prefilter import normalize_memory_route_mode
from knowledge_hub.ai.retrieval_fit import normalize_source_type
from knowledge_hub.knowledge.graph_signals import analyze_graph_query
from knowledge_hub.core.schema_validator import annotate_schema_errors
from knowledge_hub.papers.prefilter import normalize_paper_memory_mode

console = Console()
_LOCAL_PROVIDER_NAMES = {"ollama", "pplx-local", "pplx-st", "pplx_st"}


def _validate_cli_payload(config, payload: dict, schema_id: str) -> None:
    strict = bool(config.get_nested("validation", "schema", "strict", default=False))
    result = annotate_schema_errors(payload, schema_id, strict=strict)
    if not result.ok:
        problems = ", ".join(result.errors[:5]) or "unknown schema validation error"
        raise click.ClickException(f"schema validation failed for {schema_id}: {problems}")


def _get_searcher(khub_ctx):
    """RAGSearcher 인스턴스 생성"""
    return khub_ctx.factory.get_searcher()


def _ask_allow_external_default(khub_ctx, searcher) -> bool:
    config = getattr(khub_ctx, "config", None) or getattr(searcher, "config", None)
    provider = str(getattr(config, "summarization_provider", "") or "").strip().lower()
    if not provider and config is not None and hasattr(config, "get_nested"):
        provider = str(config.get_nested("summarization", "provider", default="") or "").strip().lower()
    if not provider:
        return False
    return provider not in _LOCAL_PROVIDER_NAMES


def _ask_external_policy_contract(*, surface: str, allow_external: bool, requested: bool | None) -> dict:
    return external_policy_contract(
        surface=surface,
        allow_external=allow_external,
        requested=requested,
        decision_source="explicit_option" if requested is not None else "configured_summarization_provider",
    )


def _filter_supported_kwargs(func, kwargs: dict) -> dict:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return dict(kwargs)
    return {
        key: value
        for key, value in dict(kwargs).items()
        if key in signature.parameters
    }


def _search_with_diagnostics(searcher, query: str, **kwargs):
    search_with_diagnostics = getattr(searcher, "search_with_diagnostics", None)
    if callable(search_with_diagnostics):
        payload = search_with_diagnostics(query, **_filter_supported_kwargs(search_with_diagnostics, kwargs))
        return list(payload.get("results") or []), dict(payload.get("diagnostics") or {})
    return searcher.search(query, **_filter_supported_kwargs(searcher.search, kwargs)), {}


def _generate_answer_compat(searcher, query: str, **kwargs):
    generate_answer = getattr(searcher, "generate_answer")
    supported_kwargs = _filter_supported_kwargs(generate_answer, kwargs)
    try:
        return generate_answer(query, **supported_kwargs)
    except TypeError as error:
        if "unexpected keyword argument" not in str(error):
            raise
        fallback_kwargs = dict(supported_kwargs)
        fallback_kwargs.pop("memory_route_mode", None)
        fallback_kwargs.pop("paper_memory_mode", None)
        fallback_kwargs.pop("allow_external", None)
        fallback_kwargs.pop("retrieval_mode", None)
        fallback_kwargs.pop("alpha", None)
        return generate_answer(query, **fallback_kwargs)


def _runtime_diagnostics(searcher, *, searcher_error: str = "") -> dict:
    config = getattr(searcher, "config", None)
    return build_runtime_diagnostics(config, searcher=searcher, searcher_error=searcher_error)


def _selected_answer_route_fields(result: dict | None) -> dict[str, str]:
    router = dict((result or {}).get("router") or {})
    selected = dict(router.get("selected") or {})
    return {
        "answerRouteApplied": str(selected.get("route") or ""),
        "answerProviderApplied": str(selected.get("provider") or ""),
        "answerModelApplied": str(selected.get("model") or ""),
    }


def _emit_init_failure_json(
    *,
    schema_id: str,
    config,
    query_key: str,
    query_value: str,
    source,
    retrieval_mode: str,
    alpha: float,
    searcher_error: str,
    extra_payload: dict | None = None,
) -> None:
    runtime_diagnostics = build_runtime_diagnostics(config, searcher=None, searcher_error=searcher_error)
    payload = {
        "schema": schema_id,
        query_key: query_value,
        "sourceType": source,
        "retrievalMode": retrieval_mode,
        "alpha": alpha,
        "runtimeDiagnostics": runtime_diagnostics,
        "graphQuerySignal": {},
        "graph_query_signal": {},
        "initError": searcher_error,
        "status": "init_error",
    }
    payload.update(dict(extra_payload or {}))
    console.print_json(data=payload)


def _graph_query_signal(searcher, query: str) -> dict:
    repository = getattr(searcher, "sqlite_db", None)
    if repository is None:
        return {}
    try:
        return analyze_graph_query(query, repository).to_dict()
    except Exception:
        return {}


def _ranking_fields(extras: dict | None, *, source_type: str = "") -> dict:
    data = dict(extras or {})
    ranking = dict(data.get("ranking_signals") or {})
    normalized = str(
        data.get("normalized_source_type")
        or ranking.get("normalized_source_type")
        or normalize_source_type(source_type)
        or ""
    )
    top_signals = list(data.get("top_ranking_signals") or ranking.get("top_ranking_signals") or [])
    return {
        "quality_flag": str(data.get("quality_flag") or "unscored"),
        "source_trust_score": data.get("source_trust_score", 0.0),
        "reference_role": str(data.get("reference_role") or ""),
        "reference_tier": str(data.get("reference_tier") or ""),
        "normalized_source_type": normalized,
        "duplicate_collapsed": bool(data.get("duplicate_collapsed") or ranking.get("duplicate_collapsed")),
        "top_ranking_signals": top_signals,
        "ranking_signals": ranking,
    }


def _source_label(source_type: str) -> str:
    labels = {
        "paper": "[cyan]논문[/cyan]",
        "concept": "[magenta]개념[/magenta]",
        "vault": "[green]노트[/green]",
        "web": "[blue]웹[/blue]",
    }
    return labels.get(source_type, f"[dim]{source_type}[/dim]")


@click.command("rag-report")
@click.option("--limit", type=int, default=100, show_default=True)
@click.option("--days", type=int, default=7, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def rag_report(ctx, limit, days, as_json):
    """최근 RAG answer verification/rewrite 운영 리포트"""
    try:
        searcher = _get_searcher(ctx.obj["khub"])
    except Exception as error:
        console.print(f"[red]초기화 실패: {error}[/red]")
        return

    khub = ctx.obj["khub"]
    payload = build_rag_ops_report(searcher.sqlite_db, limit=max(1, int(limit)), days=max(0, int(days)))
    _validate_cli_payload(khub.config, payload, "knowledge-hub.rag.report.result.v1")
    if as_json:
        console.print_json(data=payload)
        return

    counts = payload.get("counts") or {}
    rates = payload.get("rates") or {}
    console.print(
        f"[bold]rag-report[/bold] total={counts.get('total', 0)} "
        f"needsCaution={counts.get('needsCaution', 0)} "
        f"rewriteApplied={counts.get('rewriteApplied', 0)} "
        f"fallback={counts.get('conservativeFallback', 0)} "
        f"unsupportedRate={rates.get('unsupportedClaimRate', 0.0)}"
    )
    for alert in list(payload.get("alerts") or [])[:5]:
        console.print(f"[yellow]! {alert.get('severity')} {alert.get('code')}: {alert.get('summary')}[/yellow]")
    for action in list(payload.get("recommendedActions") or [])[:3]:
        command = " ".join([str(action.get("command") or ""), *[str(item) for item in (action.get("args") or [])]]).strip()
        console.print(f"[cyan]> {action.get('summary')}[/cyan]")
        if command:
            console.print(f"[dim]  {command}[/dim]")
    for sample in list(payload.get("samples") or [])[:5]:
        console.print(
            f"- {sample.get('createdAt')} status={sample.get('verificationStatus')} "
            f"needsCaution={sample.get('needsCaution')} "
            f"rewrite={sample.get('finalAnswerSource')} "
            f"digest={sample.get('queryDigest')}"
        )
    for warning in list(payload.get("warnings") or [])[:10]:
        console.print(f"[yellow]- {warning}[/yellow]")


@click.command("search")
@click.argument("query")
@click.option("--top-k", "-k", default=10, help="결과 수")
@click.option("--source", "-s", default=None, type=click.Choice(["concept", "paper", "vault", "web"], case_sensitive=False), help="소스 필터: concept, paper, vault, web")
@click.option(
    "--mode",
    "retrieval_mode",
    type=click.Choice(["semantic", "keyword", "hybrid"], case_sensitive=False),
    default="hybrid",
    help="검색 모드: semantic/keyword/hybrid",
)
@click.option(
    "--alpha",
    type=click.FloatRange(0.0, 1.0),
    default=0.7,
    help="hybrid 모드에서 semantic 가중치 (0~1)",
)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def search(ctx, query, top_k, source, retrieval_mode, alpha, as_json):
    """벡터 유사도 검색 (논문 + 개념 통합)"""
    try:
        searcher = _get_searcher(ctx.obj["khub"])
    except Exception as error:
        if as_json:
            _emit_init_failure_json(
                schema_id="knowledge-hub.search.result.v1",
                config=getattr(ctx.obj.get("khub"), "config", None),
                query_key="query",
                query_value=query,
                source=source,
                retrieval_mode=retrieval_mode,
                alpha=alpha,
                searcher_error=str(error),
                extra_payload={
                    "results": [],
                    "rerankSignals": {},
                    "rerank_signals": {},
                    "retrievalPlan": {},
                    "retrieval_plan": {},
                    "claimSignals": {},
                    "claim_signals": {},
                },
            )
            return
        console.print(f"[red]초기화 실패: {error}[/red]")
        return

    with console.status("검색 중..."):
        results, search_diagnostics = _search_with_diagnostics(
            searcher,
            query,
            top_k=top_k,
            source_type=source,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            expand_parent_context=True,
        )
    runtime_diagnostics = _runtime_diagnostics(searcher)

    if not results:
        graph_query_signal = _graph_query_signal(searcher, query)
        claim_signals = build_claim_signal_payload(getattr(searcher, "sqlite_db", None), [])
        if as_json:
            console.print_json(
                data={
                    "schema": "knowledge-hub.search.result.v1",
                    "query": query,
                    "sourceType": source,
                    "retrievalMode": retrieval_mode,
                    "alpha": alpha,
                    "results": [],
                    "runtimeDiagnostics": runtime_diagnostics,
                    "graphQuerySignal": graph_query_signal,
                    "graph_query_signal": graph_query_signal,
                    "rerankSignals": dict(search_diagnostics.get("rerankSignals") or {}),
                    "rerank_signals": dict(search_diagnostics.get("rerankSignals") or {}),
                    "retrievalPlan": dict(search_diagnostics.get("retrievalPlan") or {}),
                    "retrieval_plan": dict(search_diagnostics.get("retrievalPlan") or {}),
                    "claimSignals": dict(claim_signals.get("summary") or {}),
                    "claim_signals": dict(claim_signals.get("summary") or {}),
                }
            )
        else:
            console.print("[yellow]검색 결과가 없습니다.[/yellow]")
        return

    graph_query_signal = _graph_query_signal(searcher, query)
    if as_json:
        related_notes = build_related_note_suggestions(results, query=query, limit=5)
        claim_payload = build_claim_signal_payload(getattr(searcher, "sqlite_db", None), results)
        claim_entries = list(claim_payload.get("items") or [])
        payload = {
            "schema": "knowledge-hub.search.result.v1",
            "query": query,
            "sourceType": source,
            "retrievalMode": retrieval_mode,
            "alpha": alpha,
            "runtimeDiagnostics": runtime_diagnostics,
            "graphQuerySignal": graph_query_signal,
            "graph_query_signal": graph_query_signal,
            "relatedNotes": related_notes,
            "related_notes": related_notes,
            "rerankSignals": dict(search_diagnostics.get("rerankSignals") or {}),
            "rerank_signals": dict(search_diagnostics.get("rerankSignals") or {}),
            "retrievalPlan": dict(search_diagnostics.get("retrievalPlan") or {}),
            "retrieval_plan": dict(search_diagnostics.get("retrievalPlan") or {}),
            "claimSignals": dict(claim_payload.get("summary") or {}),
            "claim_signals": dict(claim_payload.get("summary") or {}),
            "results": [
                {
                    "title": result.metadata.get("title", "Untitled"),
                    "sourceType": result.metadata.get("source_type", ""),
                    "source_type": result.metadata.get("source_type", ""),
                    "score": result.score,
                    "semanticScore": result.semantic_score,
                    "semantic_score": result.semantic_score,
                    "lexicalScore": result.lexical_score,
                    "lexical_score": result.lexical_score,
                    "distance": result.distance,
                    "documentId": result.document_id,
                    "document_id": result.document_id,
                    "parentId": result.metadata.get("resolved_parent_id", ""),
                    "parent_id": result.metadata.get("resolved_parent_id", ""),
                    "parentLabel": result.metadata.get("resolved_parent_label", ""),
                    "parent_label": result.metadata.get("resolved_parent_label", ""),
                    "parentChunkSpan": result.metadata.get("resolved_parent_chunk_span", ""),
                    "parent_chunk_span": result.metadata.get("resolved_parent_chunk_span", ""),
                    **{
                        "normalizedSourceType": ranking["normalized_source_type"],
                        "qualityFlag": ranking["quality_flag"],
                        "sourceTrustScore": ranking["source_trust_score"],
                        "referenceRole": ranking["reference_role"],
                        "referenceTier": ranking["reference_tier"],
                        "duplicateCollapsed": ranking["duplicate_collapsed"],
                        "topRankingSignals": ranking["top_ranking_signals"],
                        "rankingSignals": ranking["ranking_signals"],
                    },
                    **ranking,
                    "claimSignals": dict(claim_signal),
                    "claim_signals": dict(claim_signal),
                    "metadata": result.metadata,
                    "document": result.document[:1200],
                }
                for result, claim_signal in zip(results, claim_entries or [{} for _ in results], strict=False)
                for ranking in [
                    _ranking_fields(
                        getattr(result, "lexical_extras", None),
                        source_type=result.metadata.get("source_type", ""),
                    )
                ]
            ],
        }
        console.print_json(data=payload)
        return

    console.print(f"\n[bold]'{query}' 검색 결과 ({len(results)}개):[/bold]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", width=3, style="dim")
    table.add_column("유형", width=6)
    table.add_column("제목", max_width=45)
    table.add_column("유사도", width=6, justify="right")
    table.add_column("S", width=6, justify="right")
    table.add_column("K", width=6, justify="right")
    table.add_column("상세", max_width=50)

    for index, result in enumerate(results, 1):
        title = result.metadata.get("title", "Untitled")
        source_type = result.metadata.get("source_type", "")
        detail = ""
        if source_type == "concept":
            related = result.metadata.get("related_concepts", "")
            if related:
                detail = f"관련: {related}"
        elif source_type == "paper":
            keywords = result.metadata.get("keywords", "")
            field = result.metadata.get("field", "")
            parts = []
            if field:
                parts.append(field)
            if keywords:
                parts.append(keywords[:40])
            detail = " | ".join(parts)
        else:
            parent_label = str(result.metadata.get("resolved_parent_label", "")).strip()
            parent_span = str(result.metadata.get("resolved_parent_chunk_span", "")).strip()
            if parent_label:
                detail = f"parent: {parent_label}"
                if parent_span:
                    detail += f" ({parent_span})"

        table.add_row(
            str(index),
            _source_label(source_type),
            title,
            f"{result.score:.3f}",
            f"{result.semantic_score:.3f}",
            f"{result.lexical_score:.3f}",
            detail[:50],
        )

    console.print(table)
    related_notes = build_related_note_suggestions(results, query=query, limit=5)
    if related_notes:
        console.print("\n[dim]같이 볼 노트:[/dim]")
        for item in related_notes[:5]:
            reason = ", ".join(str(part) for part in (item.get("reasons") or [])[:3])
            suffix = f" [dim]({reason})[/dim]" if reason else ""
            console.print(f"  - {item.get('title', '')}{suffix}")
    console.print(f"\n[dim]모드: [yellow]{retrieval_mode}[/yellow] (alpha={alpha})[/dim]")
    console.print("[dim]khub ask \"질문\" 으로 RAG 답변 생성 가능[/dim]")


@click.command("ask")
@click.argument("question")
@click.option("--top-k", "-k", default=8, help="참고 문서 수")
@click.option("--source", "-s", default=None, help="소스 필터: concept, paper, vault, web")
@click.option(
    "--mode",
    "retrieval_mode",
    type=click.Choice(["semantic", "keyword", "hybrid"], case_sensitive=False),
    default="hybrid",
    help="검색 모드: semantic/keyword/hybrid",
)
@click.option(
    "--alpha",
    type=click.FloatRange(0.0, 1.0),
    default=0.7,
    help="hybrid 모드에서 semantic 가중치 (0~1)",
)
@click.option(
    "--memory-route-mode",
    type=click.Choice(["off", "compat", "on", "prefilter"], case_sensitive=False),
    default="off",
    show_default=True,
    help="ask retrieval memory prefilter/prior mode: off/compat/on (prefilter는 deprecated compat alias)",
)
@click.option(
    "--paper-memory-mode",
    type=click.Choice(["off", "compat", "on", "prefilter"], case_sensitive=False),
    default="off",
    show_default=True,
    help="paper-source memory prefilter mode: off/compat/on (prefilter는 deprecated compat alias)",
)
@click.option(
    "--allow-external/--no-allow-external",
    default=None,
    help="답변 생성에서 외부 LLM 사용 허용 여부. 기본값은 설정된 요약 provider를 따릅니다.",
)
@click.option(
    "--answer-route",
    type=click.Choice(["auto", "local", "api", "codex"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="답변 생성 route override: auto/local/api/codex",
)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def ask(ctx, question, top_k, source, retrieval_mode, alpha, memory_route_mode, paper_memory_mode, allow_external, answer_route, as_json):
    """RAG 기반 질의응답 (논문 + 개념 지식 그래프 활용)"""
    try:
        searcher = _get_searcher(ctx.obj["khub"])
    except Exception as error:
        if as_json:
            effective_memory_route_mode = normalize_memory_route_mode(
                memory_route_mode,
                paper_memory_mode=paper_memory_mode,
            )
            effective_paper_memory_mode = normalize_paper_memory_mode(paper_memory_mode)
            _emit_init_failure_json(
                schema_id="knowledge-hub.ask.result.v1",
                config=getattr(ctx.obj.get("khub"), "config", None),
                query_key="question",
                query_value=question,
                source=source,
                retrieval_mode=retrieval_mode,
                alpha=alpha,
                searcher_error=str(error),
                extra_payload={
                    "answer": "",
                    "sources": [],
                    "citations": [],
                    "warnings": [f"searcher init failed: {error}"],
                    "memoryRouteMode": effective_memory_route_mode,
                    "paperMemoryMode": effective_paper_memory_mode,
                    "allowExternal": False if allow_external is None else bool(allow_external),
                    "externalPolicy": _ask_external_policy_contract(
                        surface="cli",
                        allow_external=False if allow_external is None else bool(allow_external),
                        requested=allow_external,
                    ),
                    "answerRouteRequested": str(answer_route or "auto"),
                },
            )
            return
        console.print(f"[red]초기화 실패: {error}[/red]")
        return
    allow_external_effective = _ask_allow_external_default(ctx.obj["khub"], searcher) if allow_external is None else bool(allow_external)

    with console.status("답변 생성 중..."):
        result = _generate_answer_compat(
            searcher,
            question,
            top_k=top_k,
            source_type=source,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            allow_external=allow_external_effective,
            memory_route_mode=memory_route_mode,
            paper_memory_mode=paper_memory_mode,
            answer_route_override=None if str(answer_route or "auto").strip().lower() == "auto" else str(answer_route or "").strip().lower(),
        )
    runtime_diagnostics = _runtime_diagnostics(searcher)
    graph_query_signal = _graph_query_signal(searcher, question)

    if as_json:
        effective_memory_route_mode = normalize_memory_route_mode(
            memory_route_mode,
            paper_memory_mode=paper_memory_mode,
        )
        effective_paper_memory_mode = normalize_paper_memory_mode(paper_memory_mode)
        external_policy = _ask_external_policy_contract(
            surface="cli",
            allow_external=allow_external_effective,
            requested=allow_external,
        )
        payload = ensure_ask_contract_payload(
            dict(result),
            source_type=source,
            memory_route_mode=memory_route_mode,
            paper_memory_mode=paper_memory_mode,
            external_policy=external_policy,
        )
        payload["question"] = question
        payload["schema"] = "knowledge-hub.ask.result.v1"
        payload["sourceType"] = source
        payload["retrievalMode"] = retrieval_mode
        payload["alpha"] = alpha
        payload["memoryRouteMode"] = effective_memory_route_mode
        payload["paperMemoryMode"] = effective_paper_memory_mode
        payload["answerRouteRequested"] = str(answer_route or "auto")
        payload.update(_selected_answer_route_fields(payload))
        payload["runtimeDiagnostics"] = runtime_diagnostics
        payload["graphQuerySignal"] = graph_query_signal
        payload["graph_query_signal"] = graph_query_signal
        console.print_json(data=payload)
        return

    console.print(f"\n[bold cyan]Q: {question}[/bold cyan]\n")
    console.print(result["answer"])
    console.print(f"[dim]allow_external={allow_external_effective}[/dim]")
    external_policy = _ask_external_policy_contract(
        surface="cli",
        allow_external=allow_external_effective,
        requested=allow_external,
    )
    console.print(f"[dim]external policy={external_policy['policyMode']}[/dim]")
    selected_route = _selected_answer_route_fields(result)
    if any(selected_route.values()):
        console.print(
            "[dim]"
            f"answer_route={selected_route['answerRouteApplied'] or 'unknown'} "
            f"provider={selected_route['answerProviderApplied'] or ''} "
            f"model={selected_route['answerModelApplied'] or ''}"
            "[/dim]"
        )

    verification = dict(result.get("answerVerification") or {})
    rewrite = dict(result.get("answerRewrite") or {})
    if rewrite:
        rewrite_status = "rewritten" if rewrite.get("applied") else "original"
        console.print(f"\n[bold blue]재작성: {rewrite_status}[/bold blue]")
        if rewrite.get("summary"):
            console.print(str(rewrite.get("summary")))
        console.print(
            f"[dim]attempted={bool(rewrite.get('attempted'))}, "
            f"applied={bool(rewrite.get('applied'))}, "
            f"finalAnswerSource={str(rewrite.get('finalAnswerSource') or 'original')}[/dim]"
        )

    if verification:
        status = str(verification.get("status") or "").strip().lower()
        summary = str(verification.get("summary") or "").strip()
        unsupported = int(verification.get("unsupportedClaimCount") or 0)
        uncertain = int(verification.get("uncertainClaimCount") or 0)
        style = "green" if status == "verified" and not verification.get("needsCaution") else "yellow"
        console.print(f"\n[bold {style}]검증: {status or 'unknown'}[/bold {style}]")
        if summary:
            console.print(summary)
        console.print(
            f"[dim]supported={int(verification.get('supportedClaimCount') or 0)}, "
            f"uncertain={uncertain}, unsupported={unsupported}, "
            f"conflictMentioned={bool(verification.get('conflictMentioned'))}[/dim]"
        )

    warnings = list(result.get("warnings") or [])
    if warnings:
        console.print("\n[bold yellow]경고:[/bold yellow]")
        for warning in warnings[:5]:
            console.print(f"  - {warning}")

    generation = dict(result.get("answerGeneration") or {})
    if generation:
        console.print("\n[bold blue]생성 경로:[/bold blue]")
        console.print(
            f"[dim]status={generation.get('status', 'unknown')}, "
            f"fallback={bool(generation.get('fallbackUsed'))}, "
            f"stage={generation.get('stage', '')}, "
            f"errorType={generation.get('errorType', '')}[/dim]"
        )

    prefilter = dict(result.get("paperMemoryPrefilter") or {})
    memory_prefilter = dict(result.get("memoryPrefilter") or {})
    memory_route = dict(result.get("memoryRoute") or {})
    if memory_route:
        console.print(
            "\n[dim]ask memory prefilter:"
            f" requested={memory_route.get('requestedMode', 'off')}"
            f" effective={memory_route.get('effectiveMode', memory_route.get('requestedMode', 'off'))}"
            f" aliasDeprecated={bool(memory_route.get('aliasDeprecated'))}"
            f" applied={bool(memory_route.get('applied'))}"
            f" source={memory_route.get('sourceType', 'all')}[/dim]"
        )
    if memory_prefilter and not prefilter:
        console.print(
            "\n[dim]memory prefilter detail:"
            f" effective={memory_prefilter.get('effectiveMode', memory_prefilter.get('requestedMode', 'off'))}"
            f" applied={bool(memory_prefilter.get('applied'))}"
            f" fallback={bool(memory_prefilter.get('fallbackUsed'))}"
            f" reason={memory_prefilter.get('reason', '')}[/dim]"
        )
    if prefilter:
        console.print(
            "\n[dim]paper-memory route:"
            f" requested={prefilter.get('requestedMode', 'off')}"
            f" effective={prefilter.get('effectiveMode', prefilter.get('requestedMode', 'off'))}"
            f" applied={bool(prefilter.get('applied'))}"
            f" fallback={bool(prefilter.get('fallbackUsed'))}"
            f" reason={prefilter.get('reason', '')}[/dim]"
        )
        matched_papers = list(prefilter.get("matchedPaperIds") or [])
        if matched_papers:
            console.print(f"[dim]matched papers: {', '.join(str(item) for item in matched_papers)}[/dim]")

    citations = list(result.get("citations") or [])
    if citations:
        console.print("\n[dim]인용/근거:[/dim]")
        for item in citations[:6]:
            console.print(f"  - {item.get('label')}: {item.get('title')} [{item.get('kind')}]")

    if result.get("sources"):
        console.print("\n[dim]참고 자료:[/dim]")
        for index, source_item in enumerate(result["sources"], 1):
            source_type = source_item.get("source_type", "")
            label = "개념" if source_type == "concept" else "논문" if source_type == "paper" else source_type
            console.print(
                f"  {index}. {source_item['title']} [{label}] "
                f"(종합: {source_item['score']:.2f}, S:{source_item.get('semantic_score', 0):.2f}, "
                f"K:{source_item.get('lexical_score', 0):.2f})"
            )
