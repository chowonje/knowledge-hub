"""Shared runtime diagnostics for provider/retrieval surfaces."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import shutil
import subprocess
from pathlib import Path
from typing import Any

import requests

from knowledge_hub.infrastructure.config import DEFAULT_CONFIG_PATH
from knowledge_hub.infrastructure.config import resolve_api_key
from knowledge_hub.infrastructure.persistence.vector import inspect_vector_store
from knowledge_hub.providers import registry


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def _provider_supports_role(info: Any, role: str) -> bool:
    if role == "embedding":
        return bool(getattr(info, "supports_embedding", False))
    return bool(getattr(info, "supports_llm", False))


def provider_runtime_probe(config: Any, provider_name: str) -> dict[str, Any]:
    name = _safe_str(provider_name).lower()
    if name == "ollama":
        return dict(_ollama_check(config))
    return {}


def _role_provider_state(
    config: Any,
    *,
    role: str,
    provider_name: str,
    model: str,
    searcher: Any = None,
    runtime_probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provider_name = _safe_str(provider_name)
    model = _safe_str(model)
    if provider_name:
        try:
            provider_info = registry.get_provider_info(provider_name, config=config)
        except TypeError:
            provider_info = registry.get_provider_info(provider_name)
    else:
        provider_info = None
    provider_config = {}
    if config is not None and hasattr(config, "get_provider_config") and provider_name:
        try:
            provider_config = dict(config.get_provider_config(provider_name) or {})
        except Exception:
            provider_config = {}

    requires_api_key = bool(getattr(provider_info, "requires_api_key", False))
    resolved_secret = ""
    api_key_status = "not_required"
    if requires_api_key:
        raw_key = _safe_str(provider_config.get("api_key", ""))
        resolved_secret = resolve_api_key(provider_name, raw_key)
        api_key_status = "ok" if resolved_secret else "missing"

    installed = bool(provider_info)
    role_supported = bool(provider_info) and _provider_supports_role(provider_info, role)
    available = installed and role_supported and (not requires_api_key or bool(resolved_secret))

    reason_codes: list[str] = []
    if not installed:
        reason_codes.append("provider_not_installed")
    elif not role_supported:
        reason_codes.append(
            "provider_missing_embedding_support" if role == "embedding" else "provider_missing_llm_support"
        )
    if requires_api_key and not resolved_secret:
        reason_codes.append("missing_api_key")

    runtime_status: dict[str, Any] = {}
    if runtime_probe:
        runtime_status = dict(runtime_probe)
        if not bool(runtime_probe.get("available", True)):
            available = False
            reason_codes.append("provider_runtime_unavailable")

    if role == "embedding" and searcher is not None:
        embedder = getattr(searcher, "embedder", None)
        if embedder is not None and hasattr(embedder, "get_last_status"):
            try:
                embedder_status = dict(embedder.get_last_status() or {})
            except Exception as error:
                embedder_status = {"error": str(error)}
                reason_codes.append("embedder_runtime_status_error")
            runtime_status = {**runtime_status, **embedder_status}

    if runtime_status:
        retries = int(runtime_status.get("retries", 0) or 0)
        failures = runtime_status.get("failures", [])
        if retries > 0 or (isinstance(failures, list) and failures):
            reason_codes.append("embedder_runtime_warnings")

    degraded = not available or bool(reason_codes)
    return {
        "role": role,
        "provider": provider_name,
        "model": model,
        "display_name": getattr(provider_info, "display_name", provider_name) if provider_info else provider_name,
        "installed": installed,
        "supports": {
            "llm": bool(getattr(provider_info, "supports_llm", False)),
            "embedding": bool(getattr(provider_info, "supports_embedding", False)),
        },
        "requires_api_key": requires_api_key,
        "api_key_status": api_key_status,
        "is_local": bool(getattr(provider_info, "is_local", False)),
        "available": available,
        "degraded": degraded,
        "reasons": reason_codes,
        "runtime_status": runtime_status,
    }


def _vector_corpus_state(searcher: Any = None, *, config: Any = None) -> dict[str, Any]:
    if config is None:
        config = getattr(searcher, "config", None)
    if searcher is None:
        if config is not None:
            return inspect_vector_store(
                getattr(config, "vector_db_path", ""),
                getattr(config, "collection_name", ""),
            )
        return {
            "available": False,
            "degraded": True,
            "reasons": ["vector_corpus_unavailable"],
            "collection_name": "",
            "total_documents": 0,
            "db_path": "",
            "lexical_db_path": "",
        }

    database = getattr(searcher, "database", None)
    if database is None or not hasattr(database, "get_stats"):
        if config is not None:
            return inspect_vector_store(
                getattr(config, "vector_db_path", ""),
                getattr(config, "collection_name", ""),
            )
        return {
            "available": False,
            "degraded": True,
            "reasons": ["vector_corpus_unavailable"],
            "collection_name": "",
            "total_documents": 0,
            "db_path": "",
            "lexical_db_path": "",
        }

    try:
        stats = dict(database.get_stats() or {})
    except Exception as error:
        return {
            "available": False,
            "degraded": True,
            "reasons": ["vector_corpus_stats_error"],
            "error": str(error),
            "collection_name": "",
            "total_documents": 0,
            "db_path": "",
            "lexical_db_path": "",
        }

    total_documents = int(stats.get("total_documents", 0) or 0)
    collection_name = _safe_str(stats.get("collection_name", ""))
    reasons: list[str] = []
    if total_documents <= 0:
        reasons.append("vector_corpus_empty")
    if not collection_name:
        reasons.append("vector_collection_missing")
    degraded = bool(reasons)
    payload = {
        "available": total_documents > 0 and bool(collection_name),
        "degraded": degraded,
        "reasons": reasons,
        "collection_name": collection_name,
        "total_documents": total_documents,
        "db_path": _safe_str(stats.get("db_path", "")),
        "lexical_db_path": _safe_str(stats.get("lexical_db_path", "")),
        "metadata_keys": list(stats.get("metadata_keys") or []),
    }
    if total_documents <= 0 and config is not None:
        fallback = inspect_vector_store(
            getattr(config, "vector_db_path", ""),
            getattr(config, "collection_name", ""),
        )
        if fallback.get("recovery_backup"):
            payload["recovery_backup"] = dict(fallback["recovery_backup"])
    return payload


def build_runtime_diagnostics(config: Any, *, searcher: Any = None, searcher_error: str = "") -> dict[str, Any]:
    """Summarize provider/runtime health for retrieval-facing surfaces."""
    effective_config = config or getattr(searcher, "config", None)
    vector_corpus = _vector_corpus_state(searcher, config=effective_config)
    provider_names = {
        _safe_str(getattr(effective_config, "translation_provider", "")),
        _safe_str(getattr(effective_config, "summarization_provider", "")),
        _safe_str(getattr(effective_config, "embedding_provider", "")),
    }
    runtime_probes = {
        name: provider_runtime_probe(effective_config, name)
        for name in provider_names
        if name
    }
    provider_states = [
        _role_provider_state(
            effective_config,
            role="translation",
            provider_name=getattr(effective_config, "translation_provider", ""),
            model=getattr(effective_config, "translation_model", ""),
            runtime_probe=runtime_probes.get(_safe_str(getattr(effective_config, "translation_provider", ""))),
        ),
        _role_provider_state(
            effective_config,
            role="summarization",
            provider_name=getattr(effective_config, "summarization_provider", ""),
            model=getattr(effective_config, "summarization_model", ""),
            runtime_probe=runtime_probes.get(_safe_str(getattr(effective_config, "summarization_provider", ""))),
        ),
        _role_provider_state(
            effective_config,
            role="embedding",
            provider_name=getattr(effective_config, "embedding_provider", ""),
            model=getattr(effective_config, "embedding_model", ""),
            searcher=searcher,
            runtime_probe=runtime_probes.get(_safe_str(getattr(effective_config, "embedding_provider", ""))),
        ),
    ]
    semantic = next((item for item in provider_states if item.get("role") == "embedding"), {})
    warnings: list[str] = []
    if searcher_error:
        warnings.append(f"searcher init failed: {searcher_error}")
    if vector_corpus.get("reasons"):
        warnings.append(f"vector corpus degraded: {', '.join(str(item) for item in vector_corpus['reasons'])}")
    recovery_backup = vector_corpus.get("recovery_backup") or {}
    if vector_corpus.get("reasons") and isinstance(recovery_backup, dict) and recovery_backup.get("total_documents"):
        warnings.append(
            "vector corpus backup available: "
            f"{recovery_backup.get('total_documents')} docs at {recovery_backup.get('path')}"
        )
    for provider_state in provider_states:
        role = str(provider_state.get("role") or "")
        reasons = [str(item) for item in list(provider_state.get("reasons") or [])]
        if reasons and role != "embedding":
            warnings.append(f"{role} degraded: {', '.join(reasons)}")
    if semantic.get("reasons"):
        warnings.append(f"semantic retrieval degraded: {', '.join(str(item) for item in semantic['reasons'])}")
    runtime_status = semantic.get("runtime_status") or {}
    if runtime_status:
        retries = int(runtime_status.get("retries", 0) or 0)
        failures = runtime_status.get("failures", [])
        failure_count = len(failures) if isinstance(failures, list) else 0
        if retries > 0 or failure_count > 0:
            warnings.append(f"embedding last status: retries={retries} failures={failure_count}")

    degraded = bool(warnings) or any(bool(item.get("degraded")) for item in provider_states)
    return {
        "schema": "knowledge-hub.runtime.diagnostics.v1",
        "status": "ok" if not degraded else "degraded",
        "degraded": degraded,
        "vectorCorpus": vector_corpus,
        "providers": provider_states,
        "semanticRetrieval": semantic,
        "warnings": warnings,
    }


def _status_rank(status: str) -> int:
    return {"ok": 0, "degraded": 1, "needs_setup": 2, "blocked": 3}.get(str(status or "").strip().lower(), 1)


def _worst_status(statuses: list[str]) -> str:
    return max(statuses or ["ok"], key=_status_rank)


def _check_payload(
    *,
    area: str,
    status: str,
    summary: str,
    detail: str = "",
    fix_command: str = "",
) -> dict[str, Any]:
    return {
        "area": area,
        "status": status,
        "summary": summary,
        "detail": detail,
        "fixCommand": fix_command,
    }


def _provider_doctor_status(provider_state: dict[str, Any]) -> str:
    reasons = set(str(item) for item in provider_state.get("reasons") or [])
    if "provider_not_installed" in reasons or "missing_api_key" in reasons:
        return "blocked"
    if "provider_missing_llm_support" in reasons or "provider_missing_embedding_support" in reasons:
        return "blocked"
    if "embedder_runtime_warnings" in reasons or provider_state.get("degraded"):
        return "degraded"
    return "ok" if provider_state.get("available") else "needs_setup"


def _java_runtime_status() -> dict[str, Any]:
    candidates: list[str] = []
    primary = shutil.which("java")
    if primary:
        candidates.append(primary)
    for fallback in (
        "/opt/homebrew/opt/openjdk/bin/java",
        "/usr/local/opt/openjdk/bin/java",
    ):
        if fallback not in candidates and Path(fallback).exists():
            candidates.append(fallback)
    if not candidates:
        return {
            "available": False,
            "status": "needs_setup",
            "detail": "Java runtime을 찾을 수 없습니다.",
            "fixCommand": "brew install openjdk",
        }

    last_detail = "Java runtime을 사용할 수 없습니다."
    for java_path in candidates:
        try:
            result = subprocess.run(
                [java_path, "-version"],
                capture_output=True,
                text=True,
                timeout=3.0,
            )
        except Exception as error:
            last_detail = f"Java runtime 확인 실패: {error}"
            continue
        if result.returncode == 0:
            detail = (result.stderr or result.stdout or "").strip().splitlines()
            return {
                "available": True,
                "status": "ok",
                "detail": detail[0] if detail else java_path,
                "fixCommand": "",
            }
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        last_detail = detail[0] if detail else last_detail
    return {
        "available": False,
        "status": "blocked",
        "detail": last_detail,
        "fixCommand": 'export PATH="$(brew --prefix openjdk)/bin:$PATH"',
    }


def _mineru_dependency_status() -> dict[str, Any]:
    missing: list[str] = []
    if importlib.util.find_spec("addict") is None:
        missing.append("addict")

    transformers_version = ""
    try:
        transformers_version = str(importlib.metadata.version("transformers"))
    except Exception:
        transformers_version = ""

    try:
        from transformers.pytorch_utils import find_pruneable_heads_and_indices as _heads_helper  # noqa: PLC0415
    except Exception as error:
        version_detail = f" transformers={transformers_version}" if transformers_version else ""
        return {
            "available": False,
            "status": "blocked",
            "detail": f"MinerU runtime dependency check failed:{version_detail} {error}",
            "fixCommand": "python -m pip install -e '.[mineru]'",
        }
    _ = _heads_helper

    if missing:
        return {
            "available": False,
            "status": "blocked",
            "detail": f"MinerU runtime dependency check failed: missing {', '.join(missing)}",
            "fixCommand": "python -m pip install -e '.[mineru]'",
        }

    return {
        "available": True,
        "status": "ok",
        "detail": f"MinerU runtime dependencies 확인됨 (transformers={transformers_version or 'unknown'})",
        "fixCommand": "",
    }


def parser_runtime_status(parser_name: str) -> dict[str, Any]:
    parser_id = str(parser_name or "").strip().lower()
    if parser_id == "raw":
        return {
            "available": True,
            "status": "ok",
            "detail": "raw fallback는 항상 사용 가능합니다.",
            "fixCommand": "",
        }
    if parser_id == "pymupdf":
        if importlib.util.find_spec("fitz") is None:
            return {
                "available": False,
                "status": "needs_setup",
                "detail": "PyMuPDF 패키지가 설치되어 있지 않습니다.",
                "fixCommand": "python -m pip install -e '.[pymupdf]'",
            }
        ocrmypdf_path = shutil.which("ocrmypdf")
        tesseract_path = shutil.which("tesseract")
        gs_path = shutil.which("gs")
        missing = [name for name, path in (("ocrmypdf", ocrmypdf_path), ("tesseract", tesseract_path), ("gs", gs_path)) if not path]
        if missing:
            return {
                "available": True,
                "status": "degraded",
                "detail": f"PyMuPDF parser는 사용 가능하지만 scanned PDF OCR 경로는 비활성입니다 (missing {', '.join(missing)}).",
                "fixCommand": "python -m pip install -e '.[pymupdf,ocrmypdf]'",
            }
        return {
            "available": True,
            "status": "ok",
            "detail": "PyMuPDF lightweight parser + OCR prerequisites 확인됨.",
            "fixCommand": "",
        }
    if parser_id == "opendataloader":
        if importlib.util.find_spec("opendataloader_pdf") is None:
            return {
                "available": False,
                "status": "needs_setup",
                "detail": "opendataloader_pdf 패키지가 설치되어 있지 않습니다.",
                "fixCommand": "python -m pip install -e '.[opendataloader]'",
            }
        java_status = _java_runtime_status()
        if not java_status["available"]:
            return {
                "available": False,
                "status": str(java_status["status"]),
                "detail": f"opendataloader_pdf는 설치되어 있지만 Java runtime이 없어 실행할 수 없습니다. {java_status['detail']}",
                "fixCommand": str(java_status["fixCommand"]),
            }
        return {
            "available": True,
            "status": "ok",
            "detail": f"opendataloader_pdf + Java runtime 확인됨 ({java_status['detail']})",
            "fixCommand": "",
        }
    if parser_id == "mineru":
        cli_path = shutil.which("mineru")
        try:
            version = importlib.metadata.version("mineru")
        except Exception:
            version = ""
        if not version and not cli_path:
            return {
                "available": False,
                "status": "needs_setup",
                "detail": "MinerU 패키지 또는 CLI가 설치되어 있지 않습니다.",
                "fixCommand": "python -m pip install -e '.[mineru]'",
            }
        if not cli_path:
            return {
                "available": True,
                "status": "degraded",
                "detail": f"MinerU 패키지는 설치되어 있지만 CLI가 감지되지 않았습니다 (version={version or 'unknown'}).",
                "fixCommand": "python -m pip install -e '.[mineru]'",
            }
        dependency_status = _mineru_dependency_status()
        if not dependency_status["available"]:
            return dependency_status
        return {
            "available": True,
            "status": "ok",
            "detail": f"MinerU CLI 감지됨 ({version or 'version unknown'}).",
            "fixCommand": "",
        }
    return {
        "available": True,
        "status": "ok",
        "detail": "raw fallback는 항상 사용 가능합니다.",
        "fixCommand": "",
    }


def _parser_doctor_status(parser_name: str) -> dict[str, Any]:
    return parser_runtime_status(parser_name)


def _ollama_check(config: Any) -> dict[str, Any]:
    provider_info = registry.get_provider_info("ollama")
    base_url = ""
    if config is not None and hasattr(config, "get_provider_config"):
        try:
            base_url = str(config.get_provider_config("ollama").get("base_url", "") or "").strip()
        except Exception:
            base_url = ""
    base_url = base_url or "http://localhost:11434"
    if not provider_info:
        return {
            "available": False,
            "status": "blocked",
            "summary": "Ollama provider 패키지가 설치되어 있지 않습니다.",
            "detail": "knowledge_hub.providers.ollama 모듈을 찾을 수 없습니다.",
            "fixCommand": "pip install knowledge-hub[ollama]",
        }
    url = base_url.rstrip("/") + "/api/tags"
    try:
        response = requests.get(url, timeout=2.0)
        if response.ok:
            return {
                "available": True,
                "status": "ok",
                "summary": "Ollama 서버 연결 확인됨.",
                "detail": f"{base_url} /api/tags",
                "fixCommand": "",
            }
        return {
            "available": False,
            "status": "degraded",
            "summary": "Ollama 서버가 응답했지만 성공 상태가 아닙니다.",
            "detail": f"HTTP {response.status_code} at {base_url}",
            "fixCommand": "ollama serve",
        }
    except Exception as error:
        return {
            "available": False,
            "status": "blocked",
            "summary": "Ollama 서버에 연결할 수 없습니다.",
            "detail": str(error),
            "fixCommand": "ollama serve",
        }


def build_doctor_diagnostics(config: Any, *, searcher: Any = None, searcher_error: str = "") -> dict[str, Any]:
    runtime = build_runtime_diagnostics(config, searcher=searcher, searcher_error=searcher_error)
    provider_states = list(runtime.get("providers") or [])
    provider_map = {str(item.get("role") or ""): dict(item) for item in provider_states if isinstance(item, dict)}
    checks: list[dict[str, Any]] = []

    config_path = Path(str(getattr(config, "config_path", "") or DEFAULT_CONFIG_PATH)).expanduser()
    config_exists = bool(getattr(config, "config_path", None)) or DEFAULT_CONFIG_PATH.exists()
    checks.append(
        _check_payload(
            area="config",
            status="ok" if config_exists else "needs_setup",
            summary="설정 파일 상태",
            detail=str(config_path if config_exists else DEFAULT_CONFIG_PATH),
            fix_command="khub setup --profile local" if not config_exists else "",
        )
    )

    for role, label, fix in [
        ("translation", "번역 프로바이더", "khub config set translation.provider openai"),
        ("summarization", "요약 프로바이더", "khub config set summarization.provider ollama"),
        ("embedding", "임베딩 프로바이더", "khub config set embedding.provider ollama"),
    ]:
        state = dict(provider_map.get(role) or {})
        status = _provider_doctor_status(state)
        provider_name = str(state.get("provider") or getattr(config, f"{role}_provider", "") or "-")
        model = str(state.get("model") or getattr(config, f"{role}_model", "") or "-")
        detail_bits = [provider_name, model]
        if state.get("reasons"):
            detail_bits.append(", ".join(str(item) for item in state.get("reasons") or []))
        checks.append(
            _check_payload(
                area=label,
                status=status,
                summary=f"{provider_name}/{model}",
                detail=" | ".join(bit for bit in detail_bits if bit),
                fix_command=fix if status != "ok" else "",
            )
        )

    checks.append(
        _check_payload(
            area="ollama",
            status=_ollama_check(config)["status"],
            summary=_ollama_check(config)["summary"],
            detail=_ollama_check(config)["detail"],
            fix_command=_ollama_check(config)["fixCommand"],
        )
    )

    parser_checks = []
    for parser_name in ("pymupdf", "mineru", "opendataloader", "raw"):
        parser_state = _parser_doctor_status(parser_name)
        parser_checks.append(
            _check_payload(
                area=f"parser/{parser_name}",
                status=str(parser_state["status"]),
                summary=f"{parser_name} parser",
                detail=str(parser_state["detail"]),
                fix_command=str(parser_state["fixCommand"]),
            )
        )
    checks.extend(parser_checks)

    vector_corpus = dict(runtime.get("vectorCorpus") or {})
    vector_reasons = [str(item) for item in vector_corpus.get("reasons") or []]
    if vector_corpus.get("available"):
        vector_status = "ok"
    elif "vector_corpus_empty" in vector_reasons or "vector_collection_missing" in vector_reasons:
        vector_status = "needs_setup"
    elif "vector_corpus_unavailable" in vector_reasons:
        vector_status = "needs_setup"
    else:
        vector_status = "degraded" if vector_reasons else "needs_setup"
    checks.append(
        _check_payload(
            area="vector",
            status=vector_status,
            summary=f"{vector_corpus.get('collection_name') or '-'} / {vector_corpus.get('total_documents', 0)}",
            detail=", ".join(vector_reasons) or "-",
            fix_command="khub discover \"AI agent\" --max-papers 1" if vector_status == "needs_setup" else "",
        )
    )

    sqlite_path = Path(str(getattr(config, "sqlite_path", "") or "")).expanduser()
    sqlite_exists = sqlite_path.exists()
    checks.append(
        _check_payload(
            area="sqlite",
            status="ok" if sqlite_exists else "needs_setup",
            summary=str(sqlite_path),
            detail="SQLite 데이터베이스 파일" if sqlite_exists else "SQLite 파일이 아직 생성되지 않았습니다.",
            fix_command="khub setup --profile local" if not sqlite_exists else "",
        )
    )

    papers_dir = Path(str(getattr(config, "papers_dir", "") or "")).expanduser()
    papers_exists = papers_dir.exists()
    checks.append(
        _check_payload(
            area="papers",
            status="ok" if papers_exists else "needs_setup",
            summary=str(papers_dir),
            detail="논문 저장 디렉토리" if papers_exists else "논문 저장 디렉토리가 아직 없습니다.",
            fix_command=f"mkdir -p {papers_dir}" if not papers_exists else "",
        )
    )

    top_status = _worst_status([item["status"] for item in checks])
    if any(item["status"] == "blocked" for item in checks):
        top_status = "blocked"
    elif any(item["status"] == "needs_setup" for item in checks):
        top_status = "needs_setup"
    elif any(item["status"] == "degraded" for item in checks):
        top_status = "degraded"
    else:
        top_status = "ok"

    next_actions: list[dict[str, Any]] = []
    for item in checks:
        if item["status"] in {"blocked", "needs_setup", "degraded"} and item.get("fixCommand"):
            next_actions.append(
                {
                    "area": item["area"],
                    "status": item["status"],
                    "command": item["fixCommand"],
                    "summary": item["summary"],
                }
            )
    seen: set[tuple[str, str]] = set()
    unique_actions: list[dict[str, Any]] = []
    for action in next_actions:
        key = (str(action["area"]), str(action["command"]))
        if key in seen:
            continue
        seen.add(key)
        unique_actions.append(action)

    return {
        "schema": "knowledge-hub.doctor.result.v1",
        "status": top_status,
        "checks": checks,
        "nextActions": unique_actions,
        "runtime": runtime,
    }


__all__ = ["build_doctor_diagnostics", "build_runtime_diagnostics"]
