"""
khub setup - 초보자용 시작 가이드
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from knowledge_hub.core.config import Config, DEFAULT_CONFIG_PATH

console = Console()


def _write_minimal_local_config(config: Config):
    """번역/요약/임베딩을 Ollama 로컬 환경으로 빠르게 구성."""
    config.set_nested("translation", "provider", "ollama")
    config.set_nested("translation", "model", "qwen2.5:7b")
    config.set_nested("summarization", "provider", "ollama")
    config.set_nested("summarization", "model", "qwen2.5:7b")
    config.set_nested("embedding", "provider", "ollama")
    config.set_nested("embedding", "model", "nomic-embed-text")
    config.set_nested("providers", "ollama", "base_url", "http://localhost:11434")
    config.set_nested("obsidian", "enabled", False)
    config.set_nested("obsidian", "vault_path", "")
    config.set_nested("storage", "papers_dir", str(config.papers_dir))
    config.set_nested("storage", "vector_db", str(config.vector_db_path))
    config.set_nested("storage", "sqlite", str(config.sqlite_path))


@click.command("setup")
@click.option("--quick", is_flag=True, help="Ollama 기본값으로 바로 설정")
@click.pass_context
def setup_cmd(ctx, quick):
    """처음 시작자가 바로 쓸 수 있는 설정 마법사"""
    config = Config()
    existing = bool(config.config_path or Path(DEFAULT_CONFIG_PATH).exists())

    console.print(
        Panel.fit(
            "[bold]Knowledge Hub 시작 도우미[/bold]\n"
            "- 첫 실행: 1) 번역/요약/임베딩 선택, 2) API 키 입력, 3) 저장 경로 설정\n"
            "- 초보자 모드: 로컬 Ollama만 사용(키 불필요)",
            border_style="green",
        )
    )

    if quick:
        mode = "quick"
    elif existing:
        mode = click.prompt(
            "실행 모드를 선택하세요",
            type=click.Choice(["quick", "custom"], case_sensitive=False),
            default="custom",
            show_default=True,
        )
    else:
        mode = click.prompt(
            "실행 모드를 선택하세요",
            type=click.Choice(["quick", "custom"], case_sensitive=False),
            default="quick",
            show_default=True,
        )

    if mode == "quick":
        _write_minimal_local_config(config)
        config.save()
        console.print("[green]빠른 시작 설정이 저장되었습니다.[/green]")
        console.print(f"[dim]설정 파일: {config.config_path or DEFAULT_CONFIG_PATH}[/dim]")
        console.print(
            "[bold]다음 단계[/bold]\n"
            "  1) Ollama 실행 (`ollama serve`)\n"
            "  2) model pull qwen2.5:7b, nomic-embed-text\n"
            "  3) khub discover \"AI agent\" --max-papers 1"
        )
        return

    # custom: 기존 init 그대로 실행
    from knowledge_hub.cli.init_cmd import init_cmd
    if existing:
        console.print("[dim]현재 설정은 초기화되므로 필요한 값만 덮어씁니다.[/dim]")
    ctx.invoke(init_cmd)
