"""User-facing setup/runtime doctor."""

from __future__ import annotations

from pathlib import Path
import importlib.util

import click
import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from knowledge_hub.application.runtime_diagnostics import build_runtime_diagnostics
from knowledge_hub.application.runtime_diagnostics import parser_runtime_status
from knowledge_hub.infrastructure.config import DEFAULT_CONFIG_PATH
from knowledge_hub.core.schema_validator import annotate_schema_errors
from knowledge_hub.ai.reranker import reranker_runtime_status
from knowledge_hub.application.index_freshness import build_index_freshness_check
from knowledge_hub.providers import registry

console = Console()


def _config_nested(config, *path, default=None):
    getter = getattr(config, "get_nested", None)
    if callable(getter):
        try:
            return getter(*path, default=default)
        except Exception:
            return default
    return default


def _validate_cli_payload(config, payload: dict, schema_id: str) -> None:
    strict = bool(_config_nested(config, "validation", "schema", "strict", default=False))
    result = annotate_schema_errors(payload, schema_id, strict=strict)
    if not result.ok:
        problems = ", ".join(result.errors[:5]) or "unknown schema validation error"
        raise click.ClickException(f"schema validation failed for {schema_id}: {problems}")


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _ollama_available(base_url: str) -> bool:
    url = str(base_url or "http://localhost:11434").rstrip("/")
    try:
        response = requests.get(f"{url}/api/tags", timeout=1.5)
        return response.ok
    except Exception:
        return False


_ollama_ok = _ollama_available


def _ollama_base_url(config) -> str:
    return str(config.get_provider_config("ollama").get("base_url", "http://localhost:11434")).rstrip("/")


def _configured_ollama_models(config) -> list[str]:
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


def _humanize_provider_reasons(config, provider_name: str, role: str, model: str, reasons: list[str]) -> str:
    messages: list[str] = []
    base_url = _ollama_base_url(config) if provider_name == "ollama" else ""
    reason_set = set(str(item) for item in reasons)
    if "provider_runtime_unavailable" in reason_set and provider_name == "ollama":
        messages.append(
            f"local runtime unavailable at {base_url}; {role}/{model} stays unavailable until `ollama serve` responds"
        )
    elif "provider_runtime_unavailable" in reason_set:
        messages.append("provider runtime unavailable")
    if "missing_api_key" in reason_set:
        messages.append("required API key is missing")
    if "provider_not_installed" in reason_set:
        messages.append("provider package is not installed")
    if "provider_missing_llm_support" in reason_set:
        messages.append("provider does not support LLM calls")
    if "provider_missing_embedding_support" in reason_set:
        messages.append("provider does not support embeddings")
    if "embedder_runtime_warnings" in reason_set:
        messages.append("recent embedding retries/failures were recorded")
    if messages:
        return "; ".join(messages)
    if reasons:
        return ", ".join(str(item) for item in reasons)
    return "API key OK"


def _provider_check(config, role: str, provider_name: str, model: str, state: dict[str, object]) -> dict[str, object]:
    provider_info = registry.get_provider_info(provider_name)
    installed = bool(provider_info)
    requires_api_key = bool(getattr(provider_info, "requires_api_key", False))
    api_key_status = str(state.get("api_key_status") or "not_required")
    explicit_available = state.get("available")
    available = bool(explicit_available) if explicit_available is not None else (installed and not bool(state.get("degraded")))
    reasons = list(state.get("reasons") or [])
    runtime_unavailable = "provider_runtime_unavailable" in {str(item) for item in reasons}

    if not installed:
        status = "blocked"
        summary = f"{role} 프로바이더 {provider_name}를 찾을 수 없습니다."
        fix_command = f"pip install knowledge-hub[{provider_name}]"
    elif requires_api_key and api_key_status == "missing":
        status = "blocked"
        summary = f"{role} 프로바이더 {provider_name}는 API 키가 필요합니다."
        fix_command = f"export {provider_name.upper().replace('-', '_')}_API_KEY=..."
    elif not available:
        status = "degraded"
        if provider_name == "ollama" and runtime_unavailable:
            summary = f"{role} 프로바이더 {provider_name}/{model}이 local runtime unavailable 상태입니다."
            fix_command = "ollama serve"
        else:
            summary = f"{role} 프로바이더 {provider_name}/{model}이 준비되지 않았습니다."
            fix_command = ""
    else:
        status = "ok"
        summary = f"{role} 프로바이더 {provider_name}/{model}이 준비되었습니다."
        fix_command = ""

    return {
        "area": role,
        "status": status,
        "summary": summary,
        "detail": _humanize_provider_reasons(config, provider_name, role, model, reasons)
        if api_key_status != "missing"
        else "required API key is missing",
        "fixCommand": fix_command,
    }


def _parser_check(config) -> dict[str, object]:
    parser_states = {
        name: parser_runtime_status(name)
        for name in ("pymupdf", "mineru", "opendataloader")
    }
    available_non_raw = [name for name, state in parser_states.items() if bool(state.get("available"))]
    degraded_non_raw = [name for name, state in parser_states.items() if str(state.get("status") or "") != "ok"]
    if available_non_raw and not degraded_non_raw:
        status = "ok"
        summary = "paper parser auto chain이 준비되었습니다."
    elif available_non_raw:
        status = "degraded"
        summary = "paper parser auto chain이 부분적으로만 준비되었습니다."
    else:
        status = "degraded"
        summary = "paper parser auto chain이 raw fallback만 사용할 수 있습니다."
    detail = (
        f"auto={config.paper_summary_parser} "
        f"pymupdf={parser_states['pymupdf']['status']} "
        f"mineru={parser_states['mineru']['status']} "
        f"opendataloader={parser_states['opendataloader']['status']} "
        f"raw=ok"
    )
    fix_commands = [
        str(state.get("fixCommand") or "").strip()
        for state in parser_states.values()
        if str(state.get("status") or "") != "ok"
    ]
    return {
        "area": "paper parser",
        "status": status,
        "summary": summary,
        "detail": detail,
        "fixCommand": next((item for item in fix_commands if item), ""),
    }


def _ollama_check(config) -> dict[str, object]:
    uses_ollama = any(
        provider == "ollama"
        for provider in (
            str(config.translation_provider or ""),
            str(config.summarization_provider or ""),
            str(config.embedding_provider or ""),
        )
    )
    base_url = _ollama_base_url(config)
    configured_models = _configured_ollama_models(config)
    available = _ollama_ok(base_url) if uses_ollama else True
    if uses_ollama and available:
        status = "ok"
        summary = "Ollama 연결이 정상입니다."
    elif uses_ollama:
        status = "blocked"
        summary = "Ollama local runtime이 응답하지 않습니다."
    else:
        status = "ok"
        summary = "현재 프로필은 Ollama를 사용하지 않습니다."
    if uses_ollama:
        detail = f"{base_url} / configured models: {', '.join(configured_models) or '-'}"
    else:
        detail = "not configured"
    return {
        "area": "Ollama",
        "status": status,
        "summary": summary,
        "detail": detail,
        "fixCommand": "ollama serve",
    }


def _storage_check(config) -> dict[str, object]:
    papers_dir = Path(config.papers_dir).expanduser()
    sqlite_path = Path(config.sqlite_path).expanduser()
    ok = papers_dir.parent.exists() and sqlite_path.parent.exists()
    status = "ok" if ok else "needs_setup"
    summary = "논문/SQLite 저장 경로가 준비되었습니다." if ok else "논문/SQLite 저장 경로를 아직 만들지 못했습니다."
    return {
        "area": "sqlite/papers",
        "status": status,
        "summary": summary,
        "detail": f"papers={papers_dir} sqlite={sqlite_path}",
        "fixCommand": "khub setup --profile local --non-interactive",
    }


def _reranker_check(config) -> dict[str, object]:
    state = reranker_runtime_status(config)
    enabled = bool(state.get("enabled"))
    ready = bool(state.get("ready"))
    if ready:
        status = "ok"
        summary = "labs reranker runtime이 준비되었습니다."
    elif enabled:
        status = "degraded"
        summary = "labs reranker가 켜져 있지만 runtime 준비가 부족합니다."
    else:
        status = "ok"
        summary = "labs reranker는 기본 off이며, 필요 시 켤 수 있습니다."
    reasons = ", ".join(str(item) for item in state.get("reasons") or []) or "-"
    return {
        "area": "reranker",
        "status": status,
        "summary": summary,
        "detail": f"enabled={enabled} model={state.get('model')} window={state.get('candidate_window')} timeout_ms={state.get('timeout_ms')} ready={ready} reasons={reasons}",
        "fixCommand": "pip install 'knowledge-hub-cli[st]'" if not ready else "",
    }


def _vector_check(runtime: dict[str, object]) -> dict[str, object]:
    vector = dict(runtime.get("vectorCorpus") or {})
    total_documents = int(vector.get("total_documents", 0) or 0)
    available = bool(vector.get("available"))
    recovery_backup = dict(vector.get("recovery_backup") or {})
    if available:
        status = "ok"
        summary = "벡터 코퍼스가 준비되었습니다."
        fix_command = ""
    elif recovery_backup.get("total_documents") and bool(recovery_backup.get("restorable")):
        status = "needs_setup"
        summary = "벡터 코퍼스가 비어 있지만 복구 가능한 백업이 있습니다."
        fix_command = "khub vector-compare --latest-backup"
    elif recovery_backup.get("total_documents"):
        status = "needs_setup"
        summary = "벡터 코퍼스가 비어 있고 읽기 백업만 남아 있습니다."
        fix_command = "khub index --all"
    elif total_documents <= 0:
        status = "needs_setup"
        summary = "벡터 코퍼스가 비어 있습니다."
        fix_command = "khub index --all"
    else:
        status = "degraded"
        summary = "벡터 코퍼스가 부분적으로만 준비되었습니다."
        fix_command = "khub index --all"
    detail = f"{vector.get('collection_name') or '-'} / {total_documents}"
    if recovery_backup.get("total_documents"):
        detail = f"{detail} | backup={recovery_backup.get('total_documents')} at {recovery_backup.get('path')}"
    return {
        "area": "vector corpus",
        "status": status,
        "summary": summary,
        "detail": detail,
        "fixCommand": fix_command,
    }


def _api_key_check(config, runtime: dict[str, object]) -> dict[str, object]:
    providers = list(runtime.get("providers") or [])
    missing = []
    for state in providers:
        if not bool(state.get("requires_api_key")):
            continue
        if str(state.get("api_key_status") or "") == "missing":
            missing.append(f"{state.get('provider')}")
    status = "blocked" if missing else "ok"
    summary = "필요한 API 키가 준비되었습니다." if not missing else "API 키가 부족합니다."
    if missing:
        detail = ", ".join(missing)
        fix = "khub init" if len(missing) > 1 else f"export {missing[0].upper().replace('-', '_')}_API_KEY=..."
    else:
        detail = "all required keys available"
        fix = ""
    return {
        "area": "API keys",
        "status": status,
        "summary": summary,
        "detail": detail,
        "fixCommand": fix,
    }


def _overall_status(checks: list[dict[str, object]]) -> str:
    if any(check.get("status") == "blocked" for check in checks):
        return "blocked"
    if any(check.get("status") == "needs_setup" for check in checks):
        return "needs_setup"
    if any(check.get("status") == "degraded" for check in checks):
        return "degraded"
    return "ok"


def _append_action(actions: list[str], seen: set[str], action: str) -> None:
    token = str(action or "").strip()
    if not token or token in seen:
        return
    seen.add(token)
    actions.append(token)


def _next_actions(checks: list[dict[str, object]], config=None) -> list[str]:
    actions: list[str] = []
    seen: set[str] = set()
    ollama_blocked = any(
        str(check.get("area") or "") == "Ollama" and str(check.get("status") or "") in {"blocked", "degraded"}
        for check in checks
    )
    if config is not None and ollama_blocked:
        _append_action(
            actions,
            seen,
            f"ollama serve  # start the local runtime at {_ollama_base_url(config)}",
        )
        for model in _configured_ollama_models(config):
            _append_action(actions, seen, f"ollama pull {model}")
        _append_action(
            actions,
            seen,
            "python -m knowledge_hub.interfaces.cli.main doctor  # confirm blocked/degraded areas after Ollama is up",
        )
    for check in checks:
        fix = str(check.get("fixCommand") or "").strip()
        if not fix:
            continue
        if ollama_blocked and fix == "ollama serve":
            continue
        if check.get("status") in {"blocked", "needs_setup", "degraded"}:
            _append_action(actions, seen, fix)
    return actions


def _status_style(status: str) -> str:
    return {
        "ok": "green",
        "needs_setup": "yellow",
        "degraded": "bright_yellow",
        "blocked": "red",
    }.get(str(status or "").strip().lower(), "white")


def build_doctor_payload(khub_ctx) -> dict[str, object]:
    config = khub_ctx.config
    runtime = build_runtime_diagnostics(config)
    provider_states = {str(item.get("role") or ""): dict(item) for item in list(runtime.get("providers") or [])}

    checks = [
        {
            "area": "settings",
            "status": "ok" if (getattr(config, "config_path", None) or Path(DEFAULT_CONFIG_PATH).exists()) else "needs_setup",
            "summary": "설정 파일을 찾았습니다." if (getattr(config, "config_path", None) or Path(DEFAULT_CONFIG_PATH).exists()) else "아직 초기 설정이 필요합니다.",
            "detail": str(getattr(config, "config_path", None) or DEFAULT_CONFIG_PATH),
            "fixCommand": "khub setup --profile local --non-interactive",
        },
        _provider_check(config, "summary", config.summarization_provider, config.summarization_model, provider_states.get("summarization", {})),
        _provider_check(config, "embedding", config.embedding_provider, config.embedding_model, provider_states.get("embedding", {})),
        _api_key_check(config, runtime),
        _ollama_check(config),
        _parser_check(config),
        _vector_check(runtime),
        build_index_freshness_check(config),
        _reranker_check(config),
        _storage_check(config),
    ]

    status = _overall_status(checks)
    next_actions = _next_actions(checks, config=config)
    warnings = list(runtime.get("warnings") or [])
    return {
        "schema": "knowledge-hub.doctor.result.v1",
        "status": status,
        "checks": checks,
        "nextActions": next_actions,
        "warnings": list(dict.fromkeys(warnings)),
    }


def _render_human_report(payload: dict[str, object]) -> None:
    status = str(payload.get("status") or "unknown")
    checks = list(payload.get("checks") or [])
    next_actions = list(payload.get("nextActions") or [])
    style = _status_style(status)
    console.print(
        Panel.fit(
            f"[bold]Knowledge Hub Doctor[/bold]\n"
            f"상태: [{style}]{status}[/{style}]\n"
            f"체크: {len(checks)}개 | 다음 조치: {len(next_actions)}개",
            border_style=style,
        )
    )
    table = Table(title="진단 체크")
    table.add_column("영역", style="cyan", max_width=18)
    table.add_column("상태", justify="center", max_width=12)
    table.add_column("요약", max_width=32)
    table.add_column("상세", max_width=52)
    table.add_column("수정 명령", max_width=44)
    for item in checks:
        if not isinstance(item, dict):
            continue
        item_status = str(item.get("status") or "unknown")
        table.add_row(
            str(item.get("area") or "-"),
            f"[{_status_style(item_status)}]{item_status}[/{_status_style(item_status)}]",
            str(item.get("summary") or "-"),
            str(item.get("detail") or "-"),
            str(item.get("fixCommand") or "-"),
        )
    console.print(table)
    if next_actions:
        console.print("## nextActions")
        for action in next_actions:
            console.print(f"- {action}")
    for warning in list(payload.get("warnings") or [])[:5]:
        console.print(f"[yellow]- {warning}[/yellow]")


def run_doctor(khub_ctx):
    payload = build_doctor_payload(khub_ctx)
    _validate_cli_payload(khub_ctx.config, payload, payload["schema"])
    return payload


build_doctor_report = build_doctor_payload


@click.command("doctor")
@click.option("--json/--no-json", "as_json", default=False, show_default=True, help="결과를 JSON으로 출력")
@click.pass_context
def doctor_cmd(ctx, as_json):
    """사용자용 환경 진단과 다음 조치 안내"""
    payload = run_doctor(ctx.obj["khub"])
    if as_json:
        console.print_json(data=payload)
        return
    _render_human_report(payload)


__all__ = ["doctor_cmd", "run_doctor", "build_doctor_payload", "build_doctor_report"]
