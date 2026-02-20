"""
khub health - 진단/트러블슈팅 명령어
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from knowledge_hub.core.config import Config, mask_secret, resolve_api_key
from knowledge_hub.providers import registry

console = Console()


def _check_api_key_status(config: Config, provider: str, selected_model: str) -> tuple[str, str, str]:
    """(상태, 출처, 상세) 반환"""
    config_value = ""
    raw_providers = config._data.get("providers", {}) if hasattr(config, "_data") else {}
    raw_provider_cfg = raw_providers.get(provider, {}) if isinstance(raw_providers, dict) else {}
    if isinstance(raw_provider_cfg, dict):
        config_value = str(raw_provider_cfg.get("api_key", ""))
    config_value = config_value.strip()

    info = registry.get_provider_info(provider)
    if not (info and info.requires_api_key):
        return "OK", "불필요", "-"

    if config_value.startswith("${") and config_value.endswith("}"):
        env_key = config_value[2:-1]
        actual = resolve_api_key(provider, config_value)
        if actual:
            return "OK", f"env:{env_key}", mask_secret(actual, 6)
        return "WARN", f"env:{env_key}", "환경변수 미설정"

    actual = resolve_api_key(provider, config_value)
    if actual:
        return "OK", "직접입력", mask_secret(actual, 4)

    return "WARN", "미설정", "번역/요약/임베딩 실행 시 실패 가능"


def run_health(khub_ctx):
    config = khub_ctx.config

    lines = [
        f"[bold]Knowledge Hub Health Check[/bold]",
        f"Config: {config.config_path or '(기본값)'}",
    ]
    console.print(Panel("\n".join(lines), title="요약", border_style="blue"))

    table = Table(title="환경/폴더")
    table.add_column("항목", style="cyan")
    table.add_column("상태", justify="center")
    table.add_column("설명")

    checks = []

    config_dir = Path(config.config_path).parent if config.config_path else Path.home() / ".khub"
    if config_dir.exists() or (config.config_path is None or Path(config_dir).exists()):
        checks.append(("설정 저장 경로", "OK", str(config_dir)))
    else:
        checks.append(("설정 저장 경로", "WARN", f"경로 없음: {config_dir}"))

    for label, path in [
        ("논문 저장 디렉토리", config.papers_dir),
        ("벡터DB 경로", config.vector_db_path),
        ("SQLite 경로", config.sqlite_path),
    ]:
        p = Path(path)
        parent = p.parent
        if parent.exists():
            checks.append((label, "OK", str(path)))
        else:
            checks.append((label, "WARN", f"부모 디렉토리 없음: {parent}"))

    for item in checks:
        table.add_row(*item)
    console.print(table)

    providers = registry.list_providers()
    provider_table = Table(title="프로바이더/키 상태")
    provider_table.add_column("타입", style="cyan")
    provider_table.add_column("프로바이더", style="magenta")
    provider_table.add_column("모델")
    provider_table.add_column("설치", justify="center")
    provider_table.add_column("API 키", justify="center")
    provider_table.add_column("키 출처", style="green")
    provider_table.add_column("상세", max_width=50)

    roles = [
        ("번역", config.translation_provider, config.translation_model),
        ("요약", config.summarization_provider, config.summarization_model),
        ("임베딩", config.embedding_provider, config.embedding_model),
    ]

    for role, provider_name, model in roles:
        info = providers.get(provider_name)
        installed = "OK" if info else "NO"
        key_status, key_source, key_detail = _check_api_key_status(config, provider_name, model)

        if not info:
            installed = "NO"
            key_status = "WARN"
            key_source = "-"
            key_detail = f"{provider_name} 모듈을 찾을 수 없음"
        elif key_status == "WARN":
            pass
        provider_table.add_row(role, provider_name, model, installed, key_status, key_source, key_detail)

    console.print(provider_table)

    run_test_table = Table(title="런타임 점검")
    run_test_table.add_column("항목")
    run_test_table.add_column("상태", justify="center")
    run_test_table.add_column("메시지")

    # Runtime probes
    try:
        from knowledge_hub.core.database import SQLiteDatabase, VectorDatabase
        sqlite_db = SQLiteDatabase(config.sqlite_path)
        stats = sqlite_db.get_stats()
        run_test_table.add_row("SQLite", "OK", f"논문 {stats.get('papers', 0)}개")
    except Exception as e:
        run_test_table.add_row("SQLite", "FAIL", str(e))

    try:
        config.validate(require_providers=list(dict.fromkeys([config.translation_provider, config.summarization_provider, config.embedding_provider])))
        run_test_table.add_row("설정 검증", "OK", "필수 값/키 누락 없음")
    except Exception as e:
        run_test_table.add_row("설정 검증", "WARN", str(e).replace("\n", " "))

    try:
        from knowledge_hub.core.database import VectorDatabase
        vector_db = VectorDatabase(config.vector_db_path, config.collection_name)
        run_test_table.add_row("Vector DB", "OK", f"컬렉션: {config.collection_name} ({vector_db.count()})")
    except Exception as e:
        run_test_table.add_row("Vector DB", "WARN", str(e))

    console.print(run_test_table)

    console.print("\n[dim]진단 요약: 키 경고가 있으면 khub init 또는 khub config set providers.<provider>.api_key 로 업데이트하세요.[/dim]")


@click.command("health")
@click.pass_context
def health_cmd(ctx):
    """설치/설정/연결 상태를 한 번에 진단합니다."""
    run_health(ctx.obj["khub"])

