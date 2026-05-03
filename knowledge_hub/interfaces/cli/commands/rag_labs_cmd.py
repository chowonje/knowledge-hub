"""khub labs rag - RAG vNext read-only diagnostics."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from knowledge_hub.application.rag_corrective_report import (
    ADAPTIVE_PLAN_SCHEMA,
    ANSWERABILITY_RERANK_EVAL_SCHEMA,
    ANSWERABILITY_RERANK_SCHEMA,
    CORRECTIVE_EVAL_SCHEMA,
    CORRECTIVE_EXECUTION_REVIEW_SCHEMA,
    CORRECTIVE_REPORT_SCHEMA,
    CORRECTIVE_RUN_SCHEMA,
    DEFAULT_ANSWERABILITY_RERANK_EVAL_PATH,
    GRAPH_GLOBAL_PLAN_SCHEMA,
    build_rag_adaptive_plan,
    build_rag_answerability_rerank,
    build_rag_answerability_rerank_eval_report,
    build_rag_corrective_eval_report,
    build_rag_corrective_execution_review,
    build_rag_corrective_report,
    build_rag_corrective_run,
    build_rag_graph_global_plan,
)
from knowledge_hub.application.rag_observation_loop import (
    DEFAULT_CORRECTIVE_EVAL_PATH,
    RAG_VNEXT_OBSERVATION_SCHEMA,
    build_rag_vnext_observation_report,
)
from knowledge_hub.application.rag_visualization import (
    RAG_VISUALIZATION_SCHEMA,
    build_rag_visualization_payload,
    render_rag_visualization_html,
)
from knowledge_hub.core.schema_validator import annotate_schema_errors

console = Console()


def _get_searcher(khub_ctx):
    return khub_ctx.factory.get_searcher()


def _validate_payload(config, payload: dict, schema_id: str) -> None:
    strict = bool(config.get_nested("validation", "schema", "strict", default=False)) if hasattr(config, "get_nested") else False
    result = annotate_schema_errors(payload, schema_id, strict=strict)
    if not result.ok:
        problems = ", ".join(result.errors[:5]) or "unknown schema validation error"
        raise click.ClickException(f"schema validation failed for {schema_id}: {problems}")


@click.group("rag")
def rag_labs_group():
    """RAG vNext diagnostics and experiments."""


def _normalized_source(source: str) -> str | None:
    return None if source == "all" else source


def _print_json_payload(config, payload: dict, schema_id: str) -> None:
    _validate_payload(config, payload, schema_id)
    console.print_json(data=payload)


@rag_labs_group.command("visualize")
@click.argument("query")
@click.option("-k", "--top-k", type=int, default=20, show_default=True)
@click.option("-s", "--source", type=click.Choice(["all", "concept", "paper", "vault", "web"]), default="all", show_default=True)
@click.option("--mode", "retrieval_mode", type=click.Choice(["semantic", "keyword", "hybrid"]), default="hybrid", show_default=True)
@click.option("--alpha", type=click.FloatRange(0.0, 1.0), default=0.7, show_default=True)
@click.option("--output", "output_path", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def visualize(ctx, query, top_k, source, retrieval_mode, alpha, output_path, as_json):
    """Build a read-only Three.js RAG retrieval map for one query."""

    try:
        searcher = _get_searcher(ctx.obj["khub"])
        payload = build_rag_visualization_payload(
            searcher,
            query=query,
            top_k=max(1, int(top_k)),
            source_type=_normalized_source(source),
            retrieval_mode=retrieval_mode,
            alpha=float(alpha),
            output_path=str(output_path or ""),
        )
    except Exception as error:
        payload = {
            "schema": RAG_VISUALIZATION_SCHEMA,
            "status": "error",
            "query": str(query),
            "sourceType": source,
            "retrievalMode": retrieval_mode,
            "topK": max(1, int(top_k)),
            "alpha": float(alpha),
            "readOnly": True,
            "labsOnly": True,
            "runtimeApplied": False,
            "artifactWritten": False,
            "artifactPath": str(output_path or ""),
            "createdAt": "",
            "resultCount": 0,
            "summary": {},
            "retrievalStrategy": {},
            "retrievalQuality": {},
            "answerabilityRerank": {},
            "correctiveRetrieval": {},
            "artifactHealth": {},
            "graph": {"layout": "score_radial_v1", "encoding": {}, "nodes": [], "edges": []},
            "warnings": [str(error)],
        }
        if as_json:
            console.print_json(data=payload)
            return
        raise click.ClickException(str(error)) from error

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload["artifactWritten"] = True
        payload["artifactPath"] = str(output_path)
        output_path.write_text(render_rag_visualization_html(payload), encoding="utf-8")

    if as_json:
        _print_json_payload(ctx.obj["khub"].config, payload, RAG_VISUALIZATION_SCHEMA)
        return

    console.print(
        f"[bold]rag visualize[/bold] status={payload.get('status')} "
        f"results={payload.get('resultCount')} nodes={(payload.get('summary') or {}).get('nodeCount')} "
        f"edges={(payload.get('summary') or {}).get('edgeCount')}"
    )
    if output_path is not None:
        console.print(f"[cyan]html[/cyan] {output_path}")


@rag_labs_group.command("corrective-report")
@click.argument("query")
@click.option("-k", "--top-k", type=int, default=5, show_default=True)
@click.option("-s", "--source", type=click.Choice(["all", "concept", "paper", "vault", "web"]), default="all", show_default=True)
@click.option("--mode", "retrieval_mode", type=click.Choice(["semantic", "keyword", "hybrid"]), default="hybrid", show_default=True)
@click.option("--alpha", type=click.FloatRange(0.0, 1.0), default=0.7, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def corrective_report(ctx, query, top_k, source, retrieval_mode, alpha, as_json):
    """Show read-only corrective RAG diagnostics for one query."""

    try:
        searcher = _get_searcher(ctx.obj["khub"])
    except Exception as error:
        if as_json:
            console.print_json(
                data={
                    "schema": CORRECTIVE_REPORT_SCHEMA,
                    "status": "error",
                    "query": query,
                    "sourceType": source,
                    "retrievalMode": retrieval_mode,
                    "topK": max(1, int(top_k)),
                    "alpha": float(alpha),
                    "readOnly": True,
                    "resultCount": 0,
                    "retrievalPlan": {},
                    "retrievalStrategy": {},
                    "retrievalQuality": {},
                    "answerabilityRerank": {},
                    "correctiveRetrieval": {},
                    "artifactHealth": {},
                    "candidateSources": [],
                    "rerankSignals": {},
                    "memoryRoute": {},
                    "memoryPrefilter": {},
                    "paperMemoryPrefilter": {},
                    "suggestedActions": [],
                    "actionsApplied": [],
                    "resultsPreview": [],
                    "warnings": [f"searcher init failed: {error}"],
                }
            )
            return
        console.print(f"[red]초기화 실패: {error}[/red]")
        return

    normalized_source = _normalized_source(source)
    payload = build_rag_corrective_report(
        searcher,
        query=query,
        top_k=max(1, int(top_k)),
        source_type=normalized_source,
        retrieval_mode=retrieval_mode,
        alpha=float(alpha),
    )
    _validate_payload(ctx.obj["khub"].config, payload, CORRECTIVE_REPORT_SCHEMA)

    if as_json:
        console.print_json(data=payload)
        return

    quality = dict(payload.get("retrievalQuality") or {})
    answerability = dict(payload.get("answerabilityRerank") or {})
    corrective = dict(payload.get("correctiveRetrieval") or {})
    strategy = dict(payload.get("retrievalStrategy") or {})
    console.print(
        f"[bold]rag corrective-report[/bold] status={payload.get('status')} "
        f"class={strategy.get('complexityClass', '-')} "
        f"quality={quality.get('label', '-')}({quality.get('score', '-')}) "
        f"answerability={answerability.get('label', '-')}({answerability.get('score', '-')}) "
        f"retryCandidate={bool(corrective.get('retryCandidate'))} "
        f"action={corrective.get('candidateAction', 'none')}"
    )
    for action in list(payload.get("suggestedActions") or [])[:3]:
        console.print(f"[cyan]> {action.get('actionType')}[/cyan] {action.get('description')}")
    for result in list(payload.get("resultsPreview") or [])[:5]:
        console.print(f"- {result.get('title')} [{result.get('sourceType')}] score={result.get('score'):.3f}")


@rag_labs_group.command("eval-corrective")
@click.option("--queries", "queries_path", default=DEFAULT_CORRECTIVE_EVAL_PATH, show_default=True)
@click.option("-k", "--top-k", type=int, default=5, show_default=True)
@click.option("--mode", "retrieval_mode", type=click.Choice(["semantic", "keyword", "hybrid"]), default="hybrid", show_default=True)
@click.option("--alpha", type=click.FloatRange(0.0, 1.0), default=0.7, show_default=True)
@click.option("--limit", type=int, default=None)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def eval_corrective(ctx, queries_path, top_k, retrieval_mode, alpha, limit, as_json):
    """Evaluate corrective diagnostics against a query CSV."""

    try:
        searcher = _get_searcher(ctx.obj["khub"])
        payload = build_rag_corrective_eval_report(
            searcher,
            queries_path=queries_path,
            top_k=max(1, int(top_k)),
            retrieval_mode=retrieval_mode,
            alpha=float(alpha),
            limit=limit,
        )
    except Exception as error:
        if as_json:
            console.print_json(
                data={
                    "schema": CORRECTIVE_EVAL_SCHEMA,
                    "status": "error",
                    "evalPath": str(queries_path),
                    "readOnly": True,
                    "rowCount": 0,
                    "passCount": 0,
                    "failCount": 0,
                    "metrics": {},
                    "rows": [],
                    "warnings": [str(error)],
                }
            )
            return
        raise click.ClickException(str(error)) from error

    if as_json:
        _print_json_payload(ctx.obj["khub"].config, payload, CORRECTIVE_EVAL_SCHEMA)
        return
    console.print(
        f"[bold]rag eval-corrective[/bold] status={payload.get('status')} "
        f"rows={payload.get('rowCount')} pass={payload.get('passCount')} fail={payload.get('failCount')} "
        f"passRate={(payload.get('metrics') or {}).get('passRate')}"
    )
    for row in [item for item in payload.get("rows", []) if not item.get("passed")][:5]:
        console.print(
            f"- row={row.get('row')} scenario={row.get('scenario')} "
            f"expected={row.get('expected')} observed={row.get('observed')}"
        )


@rag_labs_group.command("adaptive-plan")
@click.argument("query")
@click.option("-k", "--top-k", type=int, default=5, show_default=True)
@click.option("-s", "--source", type=click.Choice(["all", "concept", "paper", "vault", "web"]), default="all", show_default=True)
@click.option("--mode", "retrieval_mode", type=click.Choice(["semantic", "keyword", "hybrid"]), default="hybrid", show_default=True)
@click.option("--alpha", type=click.FloatRange(0.0, 1.0), default=0.7, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def adaptive_plan(ctx, query, top_k, source, retrieval_mode, alpha, as_json):
    """Plan an adaptive RAG route without changing runtime behavior."""

    searcher = _get_searcher(ctx.obj["khub"])
    payload = build_rag_adaptive_plan(
        searcher,
        query=query,
        top_k=max(1, int(top_k)),
        source_type=_normalized_source(source),
        retrieval_mode=retrieval_mode,
        alpha=float(alpha),
    )
    if as_json:
        _print_json_payload(ctx.obj["khub"].config, payload, ADAPTIVE_PLAN_SCHEMA)
        return
    plan = dict(payload.get("plan") or {})
    console.print(
        f"[bold]rag adaptive-plan[/bold] status={payload.get('status')} "
        f"class={plan.get('complexityClass')} route={plan.get('route')}"
    )
    for step in list(plan.get("steps") or [])[:8]:
        console.print(f"- {step}")


@rag_labs_group.command("corrective-run")
@click.argument("query")
@click.option("-k", "--top-k", type=int, default=5, show_default=True)
@click.option("-s", "--source", type=click.Choice(["all", "concept", "paper", "vault", "web"]), default="all", show_default=True)
@click.option("--mode", "retrieval_mode", type=click.Choice(["semantic", "keyword", "hybrid"]), default="hybrid", show_default=True)
@click.option("--alpha", type=click.FloatRange(0.0, 1.0), default=0.7, show_default=True)
@click.option("--execute/--dry-run", default=False, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def corrective_run(ctx, query, top_k, source, retrieval_mode, alpha, execute, as_json):
    """Run an opt-in, retrieval-only corrective retry."""

    searcher = _get_searcher(ctx.obj["khub"])
    payload = build_rag_corrective_run(
        searcher,
        query=query,
        top_k=max(1, int(top_k)),
        source_type=_normalized_source(source),
        retrieval_mode=retrieval_mode,
        alpha=float(alpha),
        execute=bool(execute),
    )
    if as_json:
        _print_json_payload(ctx.obj["khub"].config, payload, CORRECTIVE_RUN_SCHEMA)
        return
    console.print(
        f"[bold]rag corrective-run[/bold] status={payload.get('status')} "
        f"execute={payload.get('execute')} actions={len(payload.get('actionsApplied') or [])}"
    )
    for action in list(payload.get("actionsApplied") or [])[:3]:
        console.print(f"- applied {action.get('actionType')} retryResults={action.get('retryResultCount')}")
    for action in list(payload.get("suggestedActions") or [])[:3]:
        console.print(f"- suggested {action.get('actionType')} mode={action.get('mode')}")


@rag_labs_group.command("corrective-execution-review")
@click.option("--queries", "queries_path", default=DEFAULT_CORRECTIVE_EVAL_PATH, show_default=True)
@click.option("-k", "--top-k", type=int, default=5, show_default=True)
@click.option("--mode", "retrieval_mode", type=click.Choice(["semantic", "keyword", "hybrid"]), default="hybrid", show_default=True)
@click.option("--alpha", type=click.FloatRange(0.0, 1.0), default=0.7, show_default=True)
@click.option("--limit", type=int, default=None)
@click.option("--execute/--dry-run", default=False, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def corrective_execution_review(ctx, queries_path, top_k, retrieval_mode, alpha, limit, execute, as_json):
    """Review opt-in corrective retry execution against eval rows."""

    searcher = _get_searcher(ctx.obj["khub"])
    payload = build_rag_corrective_execution_review(
        searcher,
        queries_path=queries_path,
        top_k=max(1, int(top_k)),
        retrieval_mode=retrieval_mode,
        alpha=float(alpha),
        limit=limit,
        execute=bool(execute),
    )
    if as_json:
        _print_json_payload(ctx.obj["khub"].config, payload, CORRECTIVE_EXECUTION_REVIEW_SCHEMA)
        return
    summary = dict(payload.get("summary") or {})
    console.print(
        f"[bold]rag corrective-execution-review[/bold] status={payload.get('status')} "
        f"execute={payload.get('request', {}).get('execute')} "
        f"candidates={summary.get('retryCandidateCount')} applied={summary.get('retryAppliedCount')} "
        f"improved={summary.get('retrievalImprovedCount')} noHarm={summary.get('retrievalNoHarmCount')} "
        f"regressed={summary.get('retrievalRegressedCount')}"
    )
    for row in list(payload.get("rows") or [])[:8]:
        review = dict(row.get("retryExecutionReview") or {})
        console.print(
            f"- row={row.get('row')} scenario={row.get('scenario')} action={review.get('candidateAction')} "
            f"noHarm={review.get('noHarm')} improved={review.get('improved')} regressed={review.get('regressed')}"
        )


@rag_labs_group.command("answerability-rerank")
@click.argument("query")
@click.option("-k", "--top-k", type=int, default=5, show_default=True)
@click.option("-s", "--source", type=click.Choice(["all", "concept", "paper", "vault", "web"]), default="all", show_default=True)
@click.option("--mode", "retrieval_mode", type=click.Choice(["semantic", "keyword", "hybrid"]), default="hybrid", show_default=True)
@click.option("--alpha", type=click.FloatRange(0.0, 1.0), default=0.7, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def answerability_rerank(ctx, query, top_k, source, retrieval_mode, alpha, as_json):
    """Rerank candidates by deterministic answerability in labs only."""

    searcher = _get_searcher(ctx.obj["khub"])
    payload = build_rag_answerability_rerank(
        searcher,
        query=query,
        top_k=max(1, int(top_k)),
        source_type=_normalized_source(source),
        retrieval_mode=retrieval_mode,
        alpha=float(alpha),
    )
    if as_json:
        _print_json_payload(ctx.obj["khub"].config, payload, ANSWERABILITY_RERANK_SCHEMA)
        return
    console.print(
        f"[bold]rag answerability-rerank[/bold] results={payload.get('resultCount')} "
        f"changed={payload.get('changedRankCount')}"
    )
    for result in list(payload.get("rerankedResults") or [])[:5]:
        console.print(
            f"- #{result.get('rerankedRank')} was={result.get('originalRank')} "
            f"score={result.get('answerabilityScore')} {result.get('title')}"
        )


@rag_labs_group.command("eval-answerability-rerank")
@click.option("--queries", "queries_path", default=DEFAULT_ANSWERABILITY_RERANK_EVAL_PATH, show_default=True)
@click.option("--limit", type=int, default=None)
@click.option("-k", "--top-k", type=int, default=5, show_default=True)
@click.option("-s", "--source", type=click.Choice(["all", "concept", "paper", "vault", "web"]), default="all", show_default=True)
@click.option("--mode", "retrieval_mode", type=click.Choice(["semantic", "keyword", "hybrid"]), default="hybrid", show_default=True)
@click.option("--alpha", type=click.FloatRange(0.0, 1.0), default=0.7, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def eval_answerability_rerank(ctx, queries_path, limit, top_k, source, retrieval_mode, alpha, as_json):
    """Evaluate answerability rerank as labs-only shadow evidence."""

    try:
        searcher = _get_searcher(ctx.obj["khub"])
        payload = build_rag_answerability_rerank_eval_report(
            searcher,
            queries_path=queries_path,
            limit=limit,
            top_k=max(1, int(top_k)),
            source_type=_normalized_source(source),
            retrieval_mode=retrieval_mode,
            alpha=float(alpha),
        )
    except Exception as error:
        if as_json:
            console.print_json(
                data={
                    "schema": ANSWERABILITY_RERANK_EVAL_SCHEMA,
                    "status": "error",
                    "generatedAt": "",
                    "readOnly": True,
                    "writeFree": True,
                    "labsOnly": True,
                    "runtimeApplied": False,
                    "evalPath": str(queries_path),
                    "request": {
                        "topK": max(1, int(top_k)),
                        "sourceType": source,
                        "retrievalMode": retrieval_mode,
                        "alpha": float(alpha),
                        "limit": limit,
                    },
                    "summary": {},
                    "rows": [],
                    "promotionReadiness": {
                        "status": "not_ready",
                        "blockers": ["answerability_rerank_shadow_eval_error"],
                    },
                    "warnings": [str(error)],
                }
            )
            return
        raise click.ClickException(str(error)) from error

    if as_json:
        _print_json_payload(ctx.obj["khub"].config, payload, ANSWERABILITY_RERANK_EVAL_SCHEMA)
        return
    summary = dict(payload.get("summary") or {})
    readiness = dict(payload.get("promotionReadiness") or {})
    console.print(
        f"[bold]rag eval-answerability-rerank[/bold] status={payload.get('status')} "
        f"rows={summary.get('rowCount')} improved={summary.get('improvedCount')} "
        f"neutral={summary.get('neutralCount')} regressed={summary.get('regressedCount')} "
        f"invalidGold={summary.get('invalidGoldCount')} readiness={readiness.get('status')}"
    )
    for row in [item for item in payload.get("rows", []) if item.get("verdict") in {"regressed", "invalid_gold"}][:5]:
        console.print(
            f"- row={row.get('row')} queryId={row.get('queryId')} verdict={row.get('verdict')} "
            f"blockers={row.get('blockers')}"
        )


@rag_labs_group.command("graph-global-plan")
@click.argument("query")
@click.option("-k", "--top-k", type=int, default=5, show_default=True)
@click.option("-s", "--source", type=click.Choice(["all", "concept", "paper", "vault", "web"]), default="all", show_default=True)
@click.option("--mode", "retrieval_mode", type=click.Choice(["semantic", "keyword", "hybrid"]), default="hybrid", show_default=True)
@click.option("--alpha", type=click.FloatRange(0.0, 1.0), default=0.7, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def graph_global_plan(ctx, query, top_k, source, retrieval_mode, alpha, as_json):
    """Plan a graph/global lane candidate without executing GraphRAG."""

    searcher = _get_searcher(ctx.obj["khub"])
    payload = build_rag_graph_global_plan(
        searcher,
        query=query,
        top_k=max(1, int(top_k)),
        source_type=_normalized_source(source),
        retrieval_mode=retrieval_mode,
        alpha=float(alpha),
    )
    if as_json:
        _print_json_payload(ctx.obj["khub"].config, payload, GRAPH_GLOBAL_PLAN_SCHEMA)
        return
    lane = dict(payload.get("graphGlobalLane") or {})
    console.print(
        f"[bold]rag graph-global-plan[/bold] status={payload.get('status')} "
        f"candidate={lane.get('candidate')} route={lane.get('route')}"
    )
    for step in list(lane.get("plannedSteps") or [])[:8]:
        console.print(f"- {step}")


@rag_labs_group.command("observe-loop")
@click.option("--queries", "queries_path", default=DEFAULT_CORRECTIVE_EVAL_PATH, show_default=True)
@click.option("-k", "--top-k", type=int, default=5, show_default=True)
@click.option("--mode", "retrieval_mode", type=click.Choice(["semantic", "keyword", "hybrid"]), default="hybrid", show_default=True)
@click.option("--alpha", type=click.FloatRange(0.0, 1.0), default=0.7, show_default=True)
@click.option("--limit", type=int, default=None)
@click.option("--retry-limit", type=int, default=None)
@click.option("--rerank-limit", type=int, default=None)
@click.option("--graph-limit", type=int, default=None)
@click.option("--observation-count", type=int, default=1, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def observe_loop(ctx, queries_path, top_k, retrieval_mode, alpha, limit, retry_limit, rerank_limit, graph_limit, observation_count, as_json):
    """Run the RAG vNext observation loop without writing reports."""

    searcher = _get_searcher(ctx.obj["khub"])
    payload = build_rag_vnext_observation_report(
        searcher,
        queries_path=queries_path,
        top_k=max(1, int(top_k)),
        retrieval_mode=retrieval_mode,
        alpha=float(alpha),
        limit=limit,
        retry_limit=retry_limit,
        rerank_limit=rerank_limit,
        graph_limit=graph_limit,
        observation_count=max(1, int(observation_count)),
    )
    if as_json:
        _print_json_payload(ctx.obj["khub"].config, payload, RAG_VNEXT_OBSERVATION_SCHEMA)
        return
    summary = dict(payload.get("summary") or {})
    readiness = dict(payload.get("promotionReadiness") or {})
    console.print(
        f"[bold]rag observe-loop[/bold] status={payload.get('status')} "
        f"rows={summary.get('rowCount')} passRate={summary.get('correctivePassRate')} "
        f"retryCandidates={summary.get('retryCandidateCount')} "
        f"retryApplied={summary.get('retryAppliedCount')} "
        f"graphCandidates={summary.get('graphCandidateCount')} "
        f"readiness={readiness.get('status')}"
    )
    for blocker in list(readiness.get("blockers") or [])[:8]:
        console.print(f"- blocker: {blocker}")


__all__ = [
    "rag_labs_group",
    "corrective_report",
    "eval_corrective",
    "adaptive_plan",
    "corrective_run",
    "corrective_execution_review",
    "answerability_rerank",
    "eval_answerability_rerank",
    "graph_global_plan",
    "observe_loop",
]
