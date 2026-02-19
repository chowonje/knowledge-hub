"""
khub status - 시스템 상태 표시
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def run_status(khub_ctx):
    """시스템 상태 및 통계"""
    config = khub_ctx.config

    lines = []
    lines.append(f"[bold]Knowledge Hub v{_get_version()}[/bold]\n")

    # Config info
    lines.append(f"설정: {config.config_path or '(기본값)'}")
    lines.append(f"번역: {config.translation_provider}/{config.translation_model}")
    lines.append(f"요약: {config.summarization_provider}/{config.summarization_model}")
    lines.append(f"임베딩: {config.embedding_provider}/{config.embedding_model}")
    lines.append(f"Obsidian: {'활성 (' + config.vault_path + ')' if config.vault_enabled else '비활성'}")

    console.print(Panel("\n".join(lines), title="시스템 정보", border_style="cyan"))

    # DB stats
    try:
        from knowledge_hub.core.database import VectorDatabase, SQLiteDatabase

        sqlite_db = SQLiteDatabase(config.sqlite_path)
        sql_stats = sqlite_db.get_stats()

        vector_db = VectorDatabase(config.vector_db_path, config.collection_name)
        vec_stats = vector_db.get_stats()

        table = Table(title="데이터 현황")
        table.add_column("항목", style="cyan")
        table.add_column("수량", justify="right")

        table.add_row("논문", str(sql_stats["papers"]))
        table.add_row("노트", str(sql_stats["notes"]))
        table.add_row("태그", str(sql_stats["tags"]))
        table.add_row("링크", str(sql_stats["links"]))
        table.add_row("벡터 문서", str(vec_stats["total_documents"]))

        console.print(table)
    except Exception as e:
        console.print(f"[yellow]DB 접근 불가: {e}[/yellow]")

    # Providers
    from knowledge_hub.providers.registry import list_providers
    providers = list_providers()
    prov_names = ", ".join(f"{info.display_name}" for info in providers.values())
    console.print(f"\n[dim]사용 가능 프로바이더: {prov_names}[/dim]")


def _get_version() -> str:
    try:
        from knowledge_hub import __version__
        return __version__
    except Exception:
        return "unknown"
