"""
khub config - 설정 관리 명령어
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax

import yaml

console = Console()


@click.group("config")
def config_group():
    """설정 관리 (get/set/list/path)"""
    pass


@config_group.command("get")
@click.argument("key")
@click.pass_context
def config_get(ctx, key):
    """설정 값 조회 (점 표기법: translation.provider)"""
    config = ctx.obj["khub"].config
    keys = key.split(".")
    value = config.get_nested(*keys)

    if value is None:
        console.print(f"[yellow]'{key}' 설정을 찾을 수 없습니다.[/yellow]")
        return

    if isinstance(value, dict):
        text = yaml.dump(value, default_flow_style=False, allow_unicode=True)
        console.print(f"[cyan]{key}:[/cyan]")
        console.print(Syntax(text, "yaml"))
    else:
        console.print(f"[cyan]{key}:[/cyan] {value}")


@config_group.command("set")
@click.argument("key")
@click.argument("value")
@click.pass_context
def config_set(ctx, key, value):
    """설정 값 변경 (점 표기법: translation.provider openai)"""
    config = ctx.obj["khub"].config

    if value.lower() == "true":
        value = True
    elif value.lower() == "false":
        value = False
    elif value.isdigit():
        value = int(value)

    keys = key.split(".")
    config.set_nested(*keys, value)
    config.save()
    console.print(f"[green]{key} = {value}[/green] (저장됨)")


@config_group.command("list")
@click.pass_context
def config_list(ctx):
    """전체 설정 표시"""
    config = ctx.obj["khub"].config

    table = Table(title="Knowledge Hub 설정")
    table.add_column("섹션", style="cyan", width=14)
    table.add_column("키", style="white", width=20)
    table.add_column("값", style="green")

    sections = [
        ("translation", ["provider", "model"]),
        ("summarization", ["provider", "model"]),
        ("embedding", ["provider", "model"]),
        ("storage", ["papers_dir", "vector_db", "sqlite"]),
        ("obsidian", ["enabled", "vault_path"]),
    ]
    for section, keys in sections:
        for i, key in enumerate(keys):
            value = config.get_nested(section, key, default="(미설정)")
            section_label = section if i == 0 else ""
            table.add_row(section_label, key, str(value))

    console.print(table)

    if config.config_path:
        console.print(f"\n[dim]설정 파일: {config.config_path}[/dim]")
    else:
        console.print("\n[dim]설정 파일: (기본값 사용 중 - khub init으로 생성)[/dim]")


@config_group.command("path")
@click.pass_context
def config_path(ctx):
    """설정 파일 경로 표시"""
    config = ctx.obj["khub"].config
    from knowledge_hub.core.config import DEFAULT_CONFIG_PATH

    if config.config_path:
        console.print(config.config_path)
    else:
        console.print(f"{DEFAULT_CONFIG_PATH} (아직 생성되지 않음)")


@config_group.command("providers")
@click.option("--models", "-m", is_flag=True, help="각 프로바이더의 사용 가능 모델 전체 표시")
@click.pass_context
def config_providers(ctx, models):
    """사용 가능한 AI 프로바이더 목록"""
    from knowledge_hub.providers.registry import list_providers

    providers = list_providers()
    if not providers:
        console.print("[yellow]설치된 프로바이더가 없습니다.[/yellow]")
        console.print("설치: pip install knowledge-hub[openai,ollama]")
        return

    table = Table(title="사용 가능한 프로바이더")
    table.add_column("이름", style="cyan", width=14)
    table.add_column("LLM", justify="center", width=4)
    table.add_column("Embed", justify="center", width=5)
    table.add_column("로컬", justify="center", width=4)
    table.add_column("기본 LLM 모델", max_width=30)
    table.add_column("기본 Embed 모델", max_width=25)

    for name, info in providers.items():
        table.add_row(
            name,
            "[green]O[/green]" if info.supports_llm else "-",
            "[green]O[/green]" if info.supports_embedding else "-",
            "[green]O[/green]" if info.is_local else "-",
            info.default_llm_model or "-",
            info.default_embed_model or "-",
        )

    console.print(table)

    if models:
        console.print()
        for name, info in providers.items():
            if info.available_models:
                console.print(f"[bold cyan]{name}[/bold cyan] 사용 가능 모델:")
                for m in info.available_models:
                    console.print(f"  - {m}")
                console.print()

        if "openai-compat" in providers:
            console.print("[bold cyan]openai-compat[/bold cyan] 사전 구성된 서비스:")
            try:
                from knowledge_hub.providers.openai_compat import KNOWN_SERVICES
                for svc_name, svc in KNOWN_SERVICES.items():
                    env = svc["env_key"] or "no key needed"
                    console.print(f"  [cyan]{svc_name}[/cyan] ({svc['base_url']}) [{env}]")
                    for m in svc["llm_models"][:5]:
                        console.print(f"    - {m}")
            except ImportError:
                pass
    else:
        console.print("\n[dim]--models 플래그로 사용 가능한 모델 전체를 확인할 수 있습니다.[/dim]")

    console.print("\n[dim]커스텀 모델: khub config set translation.model <모델명>[/dim]")
    console.print("[dim]커스텀 엔드포인트: khub config set providers.openai-compat.base_url <URL>[/dim]")
