from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console

from knowledge_hub.application.paper_query_export import DEFAULT_QUERY_MODEL, export_query_embedding_bundle
from knowledge_hub.application.paper_query_remote_plan import write_query_remote_plan
from knowledge_hub.application.paper_query_run import query_text_for_id, validate_query_embedding_run
from knowledge_hub.application.paper_retrieval_harness import retrieve_paper_evidence_pack
from knowledge_hub.application.remote_index_contract import JsonObject

console = Console()


def _emit_payload(payload: JsonObject, as_json: bool) -> None:
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    console.print(
        "[bold]paper-harness[/bold] "
        f"status={payload.get('status')} evidence={len(list(payload.get('evidence') or []))}"
    )


@click.group("paper-harness")
def paper_harness_group() -> None:
    pass


@paper_harness_group.command("query-export")
@click.option("--query", "queries", required=True, multiple=True, help="query text to embed remotely")
@click.option("--out", "out_dir", required=True, type=click.Path(path_type=Path), help="output bundle directory")
@click.option("--model", default=DEFAULT_QUERY_MODEL, show_default=True)
@click.option("--json", "as_json", is_flag=True, default=False)
def query_export(queries: tuple[str, ...], out_dir: Path, model: str, as_json: bool) -> None:
    payload = export_query_embedding_bundle(queries=queries, out_dir=out_dir, model=model)
    _emit_payload(payload, as_json)


@paper_harness_group.command("query-validate")
@click.option("--run", "run_dir", required=True, type=click.Path(path_type=Path), help="local query embedding run directory")
@click.option("--model", default=DEFAULT_QUERY_MODEL, show_default=True)
@click.option("--json", "as_json", is_flag=True, default=False)
def query_validate(run_dir: Path, model: str, as_json: bool) -> None:
    payload = validate_query_embedding_run(run_dir, model).to_json()
    _emit_payload(payload, as_json)


@paper_harness_group.command("query-remote-plan")
@click.option("--run", "run_dir", required=True, type=click.Path(path_type=Path), help="local query-export run directory")
@click.option("--model", default=DEFAULT_QUERY_MODEL, show_default=True)
@click.option("--remote-host", default="oracle-hermes", show_default=True)
@click.option("--remote-root", default="~/knowledgeos-remote-indexing", show_default=True)
@click.option("--session", default="", help="tmux session name; defaults to a run-derived qwen8 query session")
@click.option("--json", "as_json", is_flag=True, default=False)
def query_remote_plan(
    run_dir: Path,
    model: str,
    remote_host: str,
    remote_root: str,
    session: str,
    as_json: bool,
) -> None:
    payload = write_query_remote_plan(
        run_dir=run_dir,
        model=model,
        remote_host=remote_host,
        remote_root=remote_root,
        session=session,
    )
    _emit_payload(payload, as_json)


@paper_harness_group.command("retrieve")
@click.option("--query", required=True, help="user question or paper retrieval query")
@click.option("--top-k", default=5, show_default=True)
@click.option("--use-bge/--no-use-bge", default=True, show_default=True)
@click.option("--use-keyword/--no-use-keyword", default=True, show_default=True)
@click.option("--use-qwen8/--no-use-qwen8", default=True, show_default=True)
@click.option(
    "--qwen-query-embeddings",
    "qwen_query_embeddings_path",
    type=click.Path(path_type=Path),
    default=None,
    help="qwen8 query embedding JSONL artifact",
)
@click.option("--qwen-query-id", default="", help="query id inside --qwen-query-embeddings")
@click.option("--qwen-namespace", default="qwen3_8b_full_candidate", show_default=True)
@click.option("--qwen-model", default="qwen3-embedding:8b", show_default=True)
@click.option("--json", "as_json", is_flag=True, default=False)
@click.pass_context
def retrieve_paper_harness(
    ctx,
    query: str,
    top_k: int,
    use_bge: bool,
    use_keyword: bool,
    use_qwen8: bool,
    qwen_query_embeddings_path: Path | None,
    qwen_query_id: str,
    qwen_namespace: str,
    qwen_model: str,
    as_json: bool,
) -> None:
    payload = retrieve_paper_evidence_pack(
        khub=ctx.obj["khub"],
        query=query,
        use_bge=use_bge,
        use_keyword=use_keyword,
        use_qwen8=use_qwen8,
        qwen_query_embeddings_path=qwen_query_embeddings_path,
        qwen_query_id=qwen_query_id,
        qwen_namespace=qwen_namespace,
        qwen_model=qwen_model,
        top_k=top_k,
    )
    _emit_payload(payload, as_json)


@paper_harness_group.command("retrieve-from-run")
@click.option("--run", "run_dir", required=True, type=click.Path(path_type=Path), help="validated query embedding run directory")
@click.option("--query-id", required=True, help="query id inside the run queries.jsonl")
@click.option("--top-k", default=5, show_default=True)
@click.option("--use-bge/--no-use-bge", default=True, show_default=True)
@click.option("--use-keyword/--no-use-keyword", default=True, show_default=True)
@click.option("--use-qwen8/--no-use-qwen8", default=True, show_default=True)
@click.option("--qwen-namespace", default="qwen3_8b_full_candidate", show_default=True)
@click.option("--qwen-model", default=DEFAULT_QUERY_MODEL, show_default=True)
@click.option("--json", "as_json", is_flag=True, default=False)
@click.pass_context
def retrieve_from_query_run(
    ctx,
    run_dir: Path,
    query_id: str,
    top_k: int,
    use_bge: bool,
    use_keyword: bool,
    use_qwen8: bool,
    qwen_namespace: str,
    qwen_model: str,
    as_json: bool,
) -> None:
    validation = validate_query_embedding_run(run_dir, qwen_model)
    if validation.status == "blocked":
        _emit_payload(validation.to_json(), as_json)
        return
    query = query_text_for_id(validation.query_rows, query_id)
    if not query:
        payload = validation.to_json()
        payload["status"] = "blocked"
        payload["blockers"] = ["query_id_not_found"]
        _emit_payload(payload, as_json)
        return
    payload = retrieve_paper_evidence_pack(
        khub=ctx.obj["khub"],
        query=query,
        use_bge=use_bge,
        use_keyword=use_keyword,
        use_qwen8=use_qwen8,
        qwen_query_embeddings_path=validation.query_embedding_path,
        qwen_query_id=query_id,
        qwen_namespace=qwen_namespace,
        qwen_model=qwen_model,
        top_k=top_k,
    )
    diagnostics = dict(payload.get("runtimeDiagnostics") or {})
    diagnostics["queryRunDir"] = str(run_dir)
    diagnostics["queryId"] = query_id
    payload["runtimeDiagnostics"] = diagnostics
    _emit_payload(payload, as_json)


__all__ = ["paper_harness_group"]
