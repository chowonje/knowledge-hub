"""
khub setup - 초보자용 시작 가이드
"""

from __future__ import annotations

from pathlib import Path

import click
import requests
from rich.console import Console
from rich.panel import Panel

from knowledge_hub.infrastructure.config import (
    Config,
    DEFAULT_CONFIG_PATH,
    PUBLIC_PARSER_DEFAULT,
    PUBLIC_SETUP_PROFILE_CHOICES,
    apply_public_setup_profile,
)
from knowledge_hub.interfaces.cli.commands.init_cmd import init_cmd
from knowledge_hub.providers import registry

console = Console()


def _resolve_profile(profile: str | None, quick: bool) -> str:
    if quick:
        return "local"
    token = str(profile or "").strip().lower()
    if token:
        return token
    return "hybrid"


def _apply_profile(config: Config, profile_name: str) -> dict[str, object]:
    profile = apply_public_setup_profile(config, profile_name)
    config.set_nested("paper", "summary", "parser", PUBLIC_PARSER_DEFAULT)
    if profile_name == "hybrid" and not registry.is_provider_available("ollama"):
        console.print("[yellow]Ollama가 감지되지 않아 임베딩을 OpenAI 기본값으로 전환합니다.[/yellow]")
        config.set_nested("embedding", "provider", "openai")
        config.set_nested("embedding", "model", "text-embedding-3-small")
    return profile


def _ollama_base_url(config: Config) -> str:
    return str(config.get_provider_config("ollama").get("base_url", "http://localhost:11434")).rstrip("/")


def _configured_ollama_models(config: Config) -> list[str]:
    models: list[str] = []
    for provider_name, model in (
        (str(config.translation_provider or ""), str(config.translation_model or "")),
        (str(config.summarization_provider or ""), str(config.summarization_model or "")),
        (str(config.embedding_provider or ""), str(config.embedding_model or "")),
    ):
        if provider_name != "ollama":
            continue
        token = model.strip()
        if token and token not in models:
            models.append(token)
    return models


def _ollama_runtime_available(config: Config) -> bool:
    if not _configured_ollama_models(config):
        return True
    try:
        response = requests.get(f"{_ollama_base_url(config)}/api/tags", timeout=1.5)
        return response.ok
    except Exception:
        return False


def _profile_summary(profile_name: str, config: Config) -> None:
    console.print(
        Panel.fit(
            "[bold]Knowledge Hub 시작 도우미[/bold]\n"
            f"- 프로필: {profile_name}\n"
            f"- 번역: {config.translation_provider}/{config.translation_model}\n"
            f"- 요약: {config.summarization_provider}/{config.summarization_model}\n"
            f"- 임베딩: {config.embedding_provider}/{config.embedding_model}\n"
            f"- 파서: {config.paper_summary_parser}\n"
            f"- Obsidian: {'활성' if config.vault_enabled else '비활성'}",
            border_style="green",
        )
    )


def _next_steps(profile_name: str, config: Config, ollama_runtime_ready: bool) -> list[str]:
    ollama_models = _configured_ollama_models(config)
    doctor_cmd = "python -m knowledge_hub.interfaces.cli.main doctor"
    pull_steps = [f"`ollama pull {model}`" for model in ollama_models]
    if profile_name == "local":
        steps = [
            "이 명령은 설정만 저장합니다. local runtime을 자동으로 시작하지는 않습니다.",
            f"Ollama 시작: `ollama serve`",
        ]
        if pull_steps:
            steps.append(f"필수 모델 pull: {', '.join(pull_steps)}")
        steps.append(f"상태 재확인: `{doctor_cmd}`")
        return steps
    if profile_name == "hybrid":
        steps = [
            "OpenAI API 키 확인",
        ]
        if config.embedding_provider == "ollama":
            steps.append("로컬 임베딩을 계속 쓰려면 Ollama runtime이 필요합니다.")
            steps.append("Ollama 시작: `ollama serve`")
            if str(config.embedding_model or "").strip():
                steps.append(f"임베딩 모델 pull: `ollama pull {config.embedding_model}`")
            steps.append(f"상태 재확인: `{doctor_cmd}`")
        elif not ollama_runtime_ready:
            steps.append("현재 임베딩은 fallback으로 저장되었지만 local runtime은 아직 응답하지 않습니다.")
            steps.append(f"원하면 나중에 Ollama를 띄운 뒤 `{doctor_cmd}` 로 확인하세요.")
        else:
            steps.append(f"상태 재확인: `{doctor_cmd}`")
        return steps
    if profile_name == "custom":
        return [
            "khub init 으로 세부 provider/model/API key를 설정하세요.",
        ]
    return [
        "khub init 으로 세부 provider/model/API key를 설정하세요.",
    ]


def _print_runtime_note(profile_name: str, config: Config, ollama_runtime_ready: bool) -> None:
    if ollama_runtime_ready or not _configured_ollama_models(config):
        return
    console.print(
        Panel.fit(
            "[bold yellow]로컬 runtime 미준비[/bold yellow]\n"
            f"- profile: {profile_name}\n"
            "- 설정은 저장됐지만 `blocked/degraded` 상태는 아직 정상입니다.\n"
            f"- 먼저 `ollama serve`\n"
            f"- 확인: `python -m knowledge_hub.interfaces.cli.main doctor`",
            border_style="yellow",
        )
    )


def _print_next_steps(profile_name: str, config: Config, ollama_runtime_ready: bool) -> None:
    console.print("[bold]다음 단계[/bold]")
    for step in _next_steps(profile_name, config, ollama_runtime_ready):
        console.print(f"  - {step}")


@click.command("setup")
@click.option(
    "--profile",
    type=click.Choice(list(PUBLIC_SETUP_PROFILE_CHOICES), case_sensitive=False),
    default=None,
    help="setup profile: local, hybrid, custom",
)
@click.option("--quick", is_flag=True, help="local profile을 바로 적용하는 짧은 별칭")
@click.option("--non-interactive", is_flag=True, help="프롬프트 없이 기본값으로 저장")
@click.pass_context
def setup_cmd(ctx, profile, quick, non_interactive):
    """처음 시작자가 바로 쓸 수 있는 설정 마법사"""
    config = Config()
    existing = bool(config.config_path or Path(DEFAULT_CONFIG_PATH).exists())
    resolved_profile = _resolve_profile(profile, quick)

    if not quick and not profile and not non_interactive:
        console.print(
            Panel.fit(
                "[bold]Knowledge Hub 시작 도우미[/bold]\n"
                "- profile 기반 설정: local, hybrid, custom\n"
                "- quick 별칭: local profile을 바로 적용\n"
                "- custom: khub init 으로 세부 설정",
                border_style="green",
            )
        )
        resolved_profile = click.prompt(
            "실행 프로필을 선택하세요",
            type=click.Choice(list(PUBLIC_SETUP_PROFILE_CHOICES), case_sensitive=False),
            default="hybrid",
            show_default=True,
        )

    if resolved_profile == "custom":
        if existing:
            console.print("[dim]현재 설정은 초기화되므로 필요한 값만 덮어씁니다.[/dim]")
        ctx.invoke(init_cmd, non_interactive=bool(non_interactive))
        return

    if resolved_profile not in {"local", "hybrid"}:
        raise click.ClickException(f"unknown setup profile: {resolved_profile}")

    _apply_profile(config, resolved_profile)
    config.save()
    ollama_runtime_ready = _ollama_runtime_available(config)
    _profile_summary(resolved_profile, config)
    console.print(f"[dim]설정 파일: {config.config_path or DEFAULT_CONFIG_PATH}[/dim]")
    _print_runtime_note(resolved_profile, config, ollama_runtime_ready)
    _print_next_steps(resolved_profile, config, ollama_runtime_ready)


__all__ = ["setup_cmd"]
