"""khub labs ask-graph - bounded multi-step ask planner."""

from __future__ import annotations

import click
from rich.console import Console

from knowledge_hub.application.ask_graph import run_ask_graph

console = Console()


@click.command("ask-graph")
@click.argument("question")
@click.option("--source", default=None)
@click.option("--mode", "retrieval_mode", type=click.Choice(["semantic", "keyword", "hybrid"], case_sensitive=False), default="hybrid", show_default=True)
@click.option("--alpha", type=click.FloatRange(0.0, 1.0), default=0.7, show_default=True)
@click.option("--max-steps", default=4, show_default=True)
@click.option("--top-k", default=5, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def ask_graph_cmd(ctx, question, source, retrieval_mode, alpha, max_steps, top_k, as_json):
    khub = ctx.obj["khub"]
    payload = run_ask_graph(
        khub.searcher(),
        question=question,
        source=source,
        mode=retrieval_mode,
        alpha=alpha,
        max_steps=max_steps,
        top_k=top_k,
        return_trace=True,
    )
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[bold cyan]Q: {question}[/bold cyan]")
    console.print(payload.get("answer") or "")
    console.print(f"[dim]intent={payload['decomposition']['intent']} steps={len(payload['decomposition']['subqueries'])}[/dim]")
    for item in payload.get("trace") or []:
        console.print(f"- {item['subquery']}")
