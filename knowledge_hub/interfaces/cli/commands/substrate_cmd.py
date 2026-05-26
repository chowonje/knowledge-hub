from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from knowledge_hub.application.evidence_registry import register_answer_trace, register_packet
from knowledge_hub.application.evidence_substrate import (
    build_compare_payload,
    build_inspect_payload,
    build_trace_payload,
)
from knowledge_hub.interfaces.cli.commands.search_cmd import (
    _generate_answer_compat,
    _get_searcher,
)

console = Console()


def _registry_db(khub):
    return khub.sqlite_db()


@click.command("inspect")
@click.argument("target", required=False, default="corpus")
@click.argument("identifier", required=False, default="")
@click.option("--limit", default=20, show_default=True, help="source/chunk lookup match limit")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def inspect_cmd(ctx, target, identifier, limit, as_json):
    """Inspect corpus/index/source/chunk contracts without mutating stores."""

    khub = ctx.obj["khub"]
    sqlite_db = None
    try:
        sqlite_db = khub.sqlite_db(read_only=True, bootstrap=False)
    except TypeError:
        try:
            sqlite_db = khub.sqlite_db()
        except Exception:
            sqlite_db = None
    except Exception:
        sqlite_db = None
    payload = build_inspect_payload(
        config=khub.config,
        sqlite_db=sqlite_db,
        target=target,
        identifier=identifier,
        limit=limit,
    )
    if as_json:
        console.print_json(data=payload)
        return

    console.print(f"[bold]khub inspect[/bold] target={payload['target']} status={payload['status']}")
    vector = dict((payload.get("stores") or {}).get("vector") or {})
    console.print(
        f"[dim]vector collection={vector.get('collection_name', '')} "
        f"documents={vector.get('total_documents', vector.get('chroma_embeddings', 0))} "
        f"lexical={vector.get('lexical_documents', 0)}[/dim]"
    )
    if payload.get("matches"):
        table = Table(show_header=True, header_style="bold")
        table.add_column("chunk", max_width=32)
        table.add_column("title", max_width=44)
        table.add_column("source", max_width=12)
        for row in payload["matches"]:
            metadata = dict(row.get("metadata") or {})
            table.add_row(
                str(row.get("chunkId") or ""),
                str(row.get("title") or ""),
                str(metadata.get("source_type") or ""),
            )
        console.print(table)
    for warning in list(payload.get("warnings") or [])[:5]:
        console.print(f"[yellow]- {warning}[/yellow]")


def _run_answer_facade(
    khub,
    question: str,
    *,
    top_k: int,
    source: str | None,
    retrieval_mode: str,
    alpha: float,
    allow_external: bool,
) -> dict:
    searcher = _get_searcher(khub)
    result = _generate_answer_compat(
        searcher,
        question,
        top_k=top_k,
        source_type=source,
        retrieval_mode=retrieval_mode,
        alpha=alpha,
        allow_external=allow_external,
        memory_route_mode="off",
        paper_memory_mode="off",
    )
    payload = dict(result or {})
    payload.setdefault("question", question)
    payload.setdefault("sourceType", source)
    payload.setdefault("retrievalMode", retrieval_mode)
    payload.setdefault("alpha", alpha)
    payload.setdefault("allowExternal", allow_external)
    return payload


@click.command("compare")
@click.argument("query")
@click.option("--top-k", "-k", default=8, show_default=True, help="참고 문서 수")
@click.option("--source", "-s", default=None, help="소스 필터: concept, paper, vault, web")
@click.option(
    "--mode",
    "retrieval_mode",
    type=click.Choice(["semantic", "keyword", "hybrid"], case_sensitive=False),
    default="hybrid",
    show_default=True,
)
@click.option("--alpha", type=click.FloatRange(0.0, 1.0), default=0.7, show_default=True)
@click.option("--allow-external/--no-allow-external", default=False, show_default=True)
@click.option("--save-registry", is_flag=True, help="Persist trace/packet lookup records for khub://packet/{id}.")
@click.option("--registry-expires-at", default="", help="Optional ISO timestamp after which saved registry records may be pruned.")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def compare_cmd(ctx, query, top_k, source, retrieval_mode, alpha, allow_external, save_registry, registry_expires_at, as_json):
    """Compare sources/claims through the existing ask/evidence path."""

    try:
        answer_payload = _run_answer_facade(
            ctx.obj["khub"],
            query,
            top_k=top_k,
            source=source,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            allow_external=allow_external,
        )
        payload = build_compare_payload(answer_payload, query=query)
        if save_registry:
            db = _registry_db(ctx.obj["khub"])
            registry = register_answer_trace(
                db,
                answer_payload=answer_payload,
                trace_payload=dict(payload.get("trace") or {}),
                expires_at=registry_expires_at,
            )
            compare_packet = dict(payload.get("comparePacket") or {})
            if compare_packet:
                registry["records"].append(register_packet(db, compare_packet, expires_at=registry_expires_at))
            else:
                registry["warnings"].append("compare payload has no comparePacketContract to persist")
            payload["registry"] = registry
    except Exception as error:
        payload = {
            "schema": "knowledge-hub.compare.result.v1",
            "status": "failed",
            "query": query,
            "answer": "",
            "comparePacket": {},
            "trace": {},
            "citations": [],
            "sources": [],
            "warnings": [str(error)],
        }

    if as_json:
        console.print_json(data=payload)
        return

    console.print(f"[bold]khub compare[/bold] status={payload['status']}")
    if payload.get("answer"):
        console.print(str(payload["answer"]))
    citations = list(payload.get("citations") or [])
    if citations:
        console.print("\n[dim]citations:[/dim]")
        for item in citations[:6]:
            console.print(f"  - {item.get('label', '')}: {item.get('title', '')}")
    for warning in list(payload.get("warnings") or [])[:5]:
        console.print(f"[yellow]- {warning}[/yellow]")


@click.command("trace")
@click.argument("question", required=False, default="")
@click.option("--from-json", "json_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option("--top-k", "-k", default=8, show_default=True, help="참고 문서 수")
@click.option("--source", "-s", default=None, help="소스 필터: concept, paper, vault, web")
@click.option(
    "--mode",
    "retrieval_mode",
    type=click.Choice(["semantic", "keyword", "hybrid"], case_sensitive=False),
    default="hybrid",
    show_default=True,
)
@click.option("--alpha", type=click.FloatRange(0.0, 1.0), default=0.7, show_default=True)
@click.option("--allow-external/--no-allow-external", default=False, show_default=True)
@click.option("--save-registry", is_flag=True, help="Persist answer trace and evidence packet lookup records.")
@click.option("--registry-expires-at", default="", help="Optional ISO timestamp after which saved registry records may be pruned.")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def trace_cmd(ctx, question, json_path, top_k, source, retrieval_mode, alpha, allow_external, save_registry, registry_expires_at, as_json):
    """Trace an answer payload back to citations, evidence spans, and sources."""

    if json_path is None and not str(question or "").strip():
        raise click.ClickException("QUESTION 또는 --from-json 이 필요합니다.")

    try:
        if json_path is not None:
            answer_payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
            payload = build_trace_payload(answer_payload, question=question)
        else:
            answer_payload = _run_answer_facade(
                ctx.obj["khub"],
                question,
                top_k=top_k,
                source=source,
                retrieval_mode=retrieval_mode,
                alpha=alpha,
                allow_external=allow_external,
            )
            payload = build_trace_payload(answer_payload, question=question)
        if save_registry:
            payload["registry"] = register_answer_trace(
                _registry_db(ctx.obj["khub"]),
                answer_payload=answer_payload,
                trace_payload=payload,
                expires_at=registry_expires_at,
            )
    except Exception as error:
        payload = {
            "schema": "knowledge-hub.answer-trace.result.v1",
            "status": "failed",
            "question": question,
            "answerId": "",
            "evidencePacketId": "",
            "answer": "",
            "citations": [],
            "sources": [],
            "evidenceSpans": [],
            "gaps": [],
            "comparePacket": {},
            "warnings": [str(error)],
        }

    if as_json:
        console.print_json(data=payload)
        return

    console.print(f"[bold]khub trace[/bold] status={payload['status']}")
    console.print(f"[dim]answerId={payload.get('answerId', '') or '-'} evidencePacketId={payload.get('evidencePacketId', '') or '-'}[/dim]")
    citations = list(payload.get("citations") or [])
    if citations:
        console.print("\n[dim]citations:[/dim]")
        for item in citations[:8]:
            console.print(f"  - {item.get('label', '')}: {item.get('title', '')}")
    spans = list(payload.get("evidenceSpans") or [])
    if spans:
        console.print(f"[dim]evidence spans: {len(spans)}[/dim]")
    for warning in list(payload.get("warnings") or [])[:5]:
        console.print(f"[yellow]- {warning}[/yellow]")
