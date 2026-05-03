"""
khub status - 시스템 상태 표시
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from knowledge_hub.ai.reranker import reranker_runtime_status
from knowledge_hub.application.runtime_diagnostics import build_runtime_diagnostics

console = Console()


def _sqlite_db(khub_ctx):
    if hasattr(khub_ctx, "sqlite_db"):
        return khub_ctx.sqlite_db()
    from knowledge_hub.infrastructure.persistence import SQLiteDatabase

    return SQLiteDatabase(khub_ctx.config.sqlite_path)


def _vector_db(khub_ctx):
    if hasattr(khub_ctx, "vector_db"):
        return khub_ctx.vector_db(repair_on_init=False)
    from knowledge_hub.infrastructure.persistence import VectorDatabase

    config = khub_ctx.config
    return VectorDatabase(config.vector_db_path, config.collection_name, repair_on_init=False)


def run_status(khub_ctx):
    """시스템 상태 및 통계"""
    config = khub_ctx.config
    runtime = build_runtime_diagnostics(config)
    semantic = dict(runtime.get("semanticRetrieval") or {})
    vector_corpus = dict(runtime.get("vectorCorpus") or {})
    searcher_error = ""

    lines = []
    lines.append(f"[bold]Knowledge Hub v{_get_version()}[/bold]\n")
    lines.append(f"설정: {config.config_path or '(기본값)'}")
    lines.append(f"번역: {config.translation_provider}/{config.translation_model}")
    lines.append(f"요약: {config.summarization_provider}/{config.summarization_model}")
    lines.append(f"임베딩: {config.embedding_provider}/{config.embedding_model}")
    lines.append(f"Obsidian: {'활성 (' + config.vault_path + ')' if config.vault_enabled else '비활성'}")

    console.print(Panel("\n".join(lines), title="시스템 정보", border_style="cyan"))

    try:
        sqlite_db = _sqlite_db(khub_ctx)
        sql_stats = sqlite_db.get_stats()

        table = Table(title="데이터 현황")
        table.add_column("항목", style="cyan")
        table.add_column("수량", justify="right")

        table.add_row("논문", str(sql_stats["papers"]))
        table.add_row("노트", str(sql_stats["notes"]))
        table.add_row("태그", str(sql_stats["tags"]))
        table.add_row("링크", str(sql_stats["links"]))
        table.add_row("벡터 문서", str(vector_corpus.get("total_documents", 0)))

        console.print(table)
    except Exception as error:
        console.print(f"[yellow]DB 접근 불가: {error}[/yellow]")

    runtime_table = Table(title="Retrieval Runtime")
    runtime_table.add_column("항목", style="cyan")
    runtime_table.add_column("값")
    runtime_table.add_column("이유", style="yellow")

    runtime_table.add_row(
        "상태",
        str(runtime.get("status") or "unknown"),
        ", ".join(str(item) for item in runtime.get("warnings") or []) or "-",
    )
    runtime_table.add_row(
        "semantic provider",
        f"{semantic.get('display_name') or semantic.get('provider') or '-'} / {semantic.get('model') or '-'}",
        ", ".join(str(item) for item in semantic.get("reasons") or []) or "-",
    )
    runtime_table.add_row(
        "available",
        "예" if semantic.get("available") else "아니오",
        f"degraded={bool(semantic.get('degraded'))}",
    )
    runtime_status = semantic.get("runtime_status") or {}
    runtime_summary = []
    if isinstance(runtime_status, dict):
        retries = runtime_status.get("retries")
        failures = runtime_status.get("failures")
        failure_count = len(failures) if isinstance(failures, list) else 0
        if retries not in (None, 0):
            runtime_summary.append(f"retries={retries}")
        if failure_count > 0:
            runtime_summary.append(f"failures={failure_count}")
    runtime_table.add_row("embedder runtime", ", ".join(runtime_summary) or "-", "-")
    vector_reasons = ", ".join(str(item) for item in vector_corpus.get("reasons") or []) or "-"
    vector_value = f"{vector_corpus.get('collection_name') or '-'} / {vector_corpus.get('total_documents', 0)}"
    runtime_table.add_row("vector corpus", vector_value, vector_reasons)

    reranker = reranker_runtime_status(config)
    reranker_reasons = ", ".join(str(item) for item in reranker.get("reasons") or []) or "-"
    runtime_table.add_row("reranker enabled", "예" if reranker.get("enabled") else "아니오", reranker_reasons)
    runtime_table.add_row("reranker model", str(reranker.get("model") or "-"), f"ready={bool(reranker.get('ready'))}")
    runtime_table.add_row("reranker window", str(reranker.get("candidate_window") or "-"), "labs opt-in")
    runtime_table.add_row("reranker timeout", str(reranker.get("timeout_ms") or "-"), "ms")
    runtime_table.add_row("reranker runtime", "ready" if reranker.get("ready") else "unavailable", str(reranker.get("reason") or "-"))
    console.print(runtime_table)

    for warning in list(runtime.get("warnings") or [])[:5]:
        console.print(f"[yellow]- {warning}[/yellow]")

    from knowledge_hub.infrastructure.providers import list_providers

    providers = list_providers()
    prov_names = ", ".join(f"{info.display_name}" for info in providers.values())
    console.print(f"\n[dim]사용 가능 프로바이더: {prov_names}[/dim]")


def _get_version() -> str:
    try:
        from knowledge_hub import __version__

        return __version__
    except Exception:
        return "unknown"
