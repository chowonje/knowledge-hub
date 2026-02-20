"""
khub init - 인터랙티브 설정 마법사
"""

from __future__ import annotations

import os

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def _load_dotenv():
    """프로젝트 .env 파일에서 환경 변수 로드"""
    try:
        from dotenv import load_dotenv
        from pathlib import Path
        candidates = [
            Path.cwd() / ".env",
            Path(__file__).resolve().parents[2] / ".env",
        ]
        for env_path in candidates:
            if env_path.exists():
                load_dotenv(env_path, override=False)
                return
    except ImportError:
        pass


def _detect_obsidian_vault() -> str:
    """macOS에서 Obsidian vault 경로 자동 탐지"""
    from pathlib import Path
    candidates = [
        Path.home() / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents",
        Path.home() / "Documents" / "Obsidian",
        Path.home() / "Obsidian",
    ]
    for base in candidates:
        if base.exists():
            vaults = [d for d in base.iterdir() if d.is_dir() and not d.name.startswith(".")]
            if len(vaults) == 1:
                return str(vaults[0])
            if vaults:
                return str(vaults[0])
    return ""


def _prompt_provider(role: str, default: str, exclude_embed_only: bool = True) -> tuple[str, str]:
    """프로바이더 선택 프롬프트"""
    from knowledge_hub.providers.registry import list_providers

    providers = list_providers()
    choices = []
    for name, info in providers.items():
        if exclude_embed_only and not info.supports_llm:
            continue
        tag = ""
        if info.is_local:
            tag = " (local, free)"
        elif info.requires_api_key:
            tag = " (API key required)"
        choices.append((name, info.display_name, info.default_llm_model, tag))

    console.print(f"\n[bold cyan]{role} 프로바이더 선택:[/bold cyan]")
    for i, (name, display, model, tag) in enumerate(choices, 1):
        marker = " <- recommended" if name == default else ""
        console.print(f"  {i}. {display} ({model}){tag}{marker}")

    while True:
        raw = click.prompt("  선택", default=str(next((i for i, (n, *_) in enumerate(choices, 1) if n == default), 1)))
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(choices):
                name = choices[idx][0]
                model = choices[idx][2]
                if name == "openai-compat":
                    return _prompt_compat_service(role)
                return name, model
        except ValueError:
            if raw in [c[0] for c in choices]:
                for c in choices:
                    if c[0] == raw:
                        if c[0] == "openai-compat":
                            return _prompt_compat_service(role)
                        return c[0], c[2]
        console.print("  [red]잘못된 선택입니다.[/red]")


def _prompt_compat_service(role: str) -> tuple[str, str]:
    """OpenAI-compatible 서비스 세부 선택"""
    from knowledge_hub.providers.openai_compat import KNOWN_SERVICES

    services = list(KNOWN_SERVICES.items())
    console.print(f"\n  [bold]OpenAI-compatible 서비스 선택:[/bold]")
    for i, (svc_name, svc) in enumerate(services, 1):
        models_str = ", ".join(svc["llm_models"][:3]) if svc["llm_models"] else "(custom model)"
        env = svc["env_key"] or "no key"
        console.print(f"    {i}. [cyan]{svc_name}[/cyan] — {models_str} [{env}]")
    console.print(f"    {len(services)+1}. [dim]Custom URL (직접 입력)[/dim]")

    raw = click.prompt("    선택", default="1")
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(services):
            svc_name, svc = services[idx]
            model = svc["llm_models"][0] if svc["llm_models"] else click.prompt("    모델명", default="")
            return "openai-compat", model
        else:
            model = click.prompt("    모델명")
            return "openai-compat", model
    except (ValueError, IndexError):
        return "openai-compat", click.prompt("    모델명", default="deepseek-chat")


def _prompt_embed_provider(default: str) -> tuple[str, str]:
    """임베딩 프로바이더 선택"""
    from knowledge_hub.providers.registry import list_providers

    providers = list_providers()
    choices = []
    for name, info in providers.items():
        if info.supports_embedding:
            tag = " (local)" if info.is_local else ""
            choices.append((name, info.display_name, info.default_embed_model, tag))

    console.print("\n[bold cyan]임베딩 프로바이더 선택:[/bold cyan]")
    for i, (name, display, model, tag) in enumerate(choices, 1):
        marker = " <- recommended" if name == default else ""
        console.print(f"  {i}. {display} ({model}){tag}{marker}")

    while True:
        raw = click.prompt("  선택", default=str(next((i for i, (n, *_) in enumerate(choices, 1) if n == default), 1)))
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(choices):
                return choices[idx][0], choices[idx][2]
        except ValueError:
            pass
        console.print("  [red]잘못된 선택입니다.[/red]")


@click.command("init")
@click.option("--non-interactive", is_flag=True, help="기본값으로 자동 설정")
@click.pass_context
def init_cmd(ctx, non_interactive):
    """초기 설정 마법사 - 프로바이더, API 키, 저장 경로 등을 설정합니다"""
    from knowledge_hub.core.config import Config, DEFAULT_CONFIG, DEFAULT_CONFIG_DIR, DEFAULT_CONFIG_PATH

    console.print(Panel.fit(
        "[bold]Welcome to Knowledge Hub![/bold]\n\n"
        "AI 논문 검색 → 번역 → 요약 → 지식 연결 파이프라인을 설정합니다.",
        border_style="cyan",
    ))

    config = Config()

    if non_interactive:
        config.save()
        console.print(f"\n[green]기본 설정 저장: {DEFAULT_CONFIG_PATH}[/green]")
        return

    # 1. Translation provider
    trans_provider, trans_model = _prompt_provider("번역(Translation)", default="openai")
    config.set_nested("translation", "provider", trans_provider)
    trans_model_input = click.prompt("  번역 모델", default=trans_model)
    config.set_nested("translation", "model", trans_model_input)

    # 2. Summarization provider
    summ_provider, summ_model = _prompt_provider("요약(Summarization)", default="ollama")
    config.set_nested("summarization", "provider", summ_provider)
    summ_model_input = click.prompt("  요약 모델", default=summ_model)
    config.set_nested("summarization", "model", summ_model_input)

    # 3. Embedding provider
    embed_provider, embed_model = _prompt_embed_provider(default="ollama")
    config.set_nested("embedding", "provider", embed_provider)
    embed_model_input = click.prompt("  임베딩 모델", default=embed_model)
    config.set_nested("embedding", "model", embed_model_input)

    # 4. API keys - .env 파일도 확인
    _load_dotenv()
    needed_providers = {trans_provider, summ_provider, embed_provider}
    from knowledge_hub.providers.registry import get_provider_info
    for prov in needed_providers:
        info = get_provider_info(prov)

        if prov == "openai-compat":
            _setup_compat_provider(config, trans_model_input, summ_model_input, embed_model_input)
            continue

        if info and info.requires_api_key:
            env_var = f"{prov.upper()}_API_KEY"
            existing = os.environ.get(env_var, "")
            if existing:
                masked = existing[:8] + "..." + existing[-4:] if len(existing) > 12 else "***"
                console.print(f"\n  [green]{env_var} 감지됨: {masked}[/green]")
                config.set_nested("providers", prov, "api_key", f"${{{env_var}}}")
            else:
                api_key = click.prompt(f"\n  {info.display_name} API Key", default="", hide_input=True)
                if api_key:
                    config.set_nested("providers", prov, "api_key", api_key)

        if info and info.is_local:
            base_url = click.prompt(
                f"\n  {info.display_name} Base URL",
                default=config.get_provider_config(prov).get("base_url", "http://localhost:11434"),
            )
            config.set_nested("providers", prov, "base_url", base_url)


def _setup_compat_provider(config, *models):
    """openai-compat provider의 base_url과 api_key를 설정"""
    from knowledge_hub.providers.openai_compat import KNOWN_SERVICES

    model_to_check = next((m for m in models if m), "")

    detected_svc = None
    for svc_name, svc in KNOWN_SERVICES.items():
        if model_to_check in svc.get("llm_models", []) + svc.get("embed_models", []):
            detected_svc = (svc_name, svc)
            break

    if detected_svc:
        svc_name, svc = detected_svc
        console.print(f"\n  [cyan]감지된 서비스: {svc_name}[/cyan]")
        base_url = click.prompt("  Base URL", default=svc["base_url"])
        config.set_nested("providers", "openai-compat", "base_url", base_url)

        if svc["env_key"]:
            existing = os.environ.get(svc["env_key"], "")
            if existing:
                masked = existing[:8] + "..." + existing[-4:] if len(existing) > 12 else "***"
                console.print(f"  [green]{svc['env_key']} 감지됨: {masked}[/green]")
                config.set_nested("providers", "openai-compat", "api_key", f"${{{svc['env_key']}}}")
            else:
                api_key = click.prompt(f"  {svc_name} API Key", default="", hide_input=True)
                if api_key:
                    config.set_nested("providers", "openai-compat", "api_key", api_key)
    else:
        base_url = click.prompt("\n  OpenAI-compatible Base URL", default="http://localhost:1234/v1")
        config.set_nested("providers", "openai-compat", "base_url", base_url)
        api_key = click.prompt("  API Key (없으면 Enter)", default="", hide_input=True)
        if api_key:
            config.set_nested("providers", "openai-compat", "api_key", api_key)

    # 5. Storage paths
    console.print("\n[bold cyan]저장 경로 설정:[/bold cyan]")
    papers_dir = click.prompt("  논문 저장 경로", default=str(DEFAULT_CONFIG_DIR / "papers"))
    config.set_nested("storage", "papers_dir", papers_dir)

    # 6. Obsidian
    obsidian = click.confirm("\n  Obsidian vault 연동을 활성화할까요?", default=False)
    config.set_nested("obsidian", "enabled", obsidian)
    if obsidian:
        default_vault = _detect_obsidian_vault()
        if default_vault:
            console.print(f"  [dim]감지된 vault: {default_vault}[/dim]")
        vault_path = click.prompt("  Obsidian vault 경로", default=default_vault or "")
        if vault_path:
            config.set_nested("obsidian", "vault_path", vault_path)
        else:
            console.print("  [yellow]vault 경로 생략 - 나중에 khub config set obsidian.vault_path 로 설정 가능[/yellow]")
            config.set_nested("obsidian", "enabled", False)

    # Save
    config.save()
    console.print(f"\n[bold green]설정 완료![/bold green] 저장 위치: {DEFAULT_CONFIG_PATH}")

    # Summary table
    table = Table(title="설정 요약")
    table.add_column("항목", style="cyan")
    table.add_column("값")
    table.add_row("번역", f"{config.translation_provider}/{config.translation_model}")
    table.add_row("요약", f"{config.summarization_provider}/{config.summarization_model}")
    table.add_row("임베딩", f"{config.embedding_provider}/{config.embedding_model}")
    table.add_row("논문 저장", config.papers_dir)
    table.add_row("Obsidian", f"{'활성' if config.vault_enabled else '비활성'}")
    console.print(table)

    console.print("\n[dim]시작: khub discover \"AI agent\" --max-papers 5[/dim]")
