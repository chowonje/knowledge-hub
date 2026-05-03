"""
khub index - 논문 + 개념 노트를 벡터DB에 통합 인덱싱 (배치 임베딩)

논문: 제목 + 요약 + 키워드를 임베딩
개념: 설명 + 관련 개념 + 관련 논문을 임베딩
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from uuid import uuid4

import click
import requests
from rich.console import Console

from knowledge_hub.infrastructure.persistence.stores.derivative_lifecycle import (
    fallback_source_hash,
    mark_derivatives_stale_for_document,
)
from knowledge_hub.interfaces.cli.commands.paper_shared_runtime import _build_paper_embedding_text
from knowledge_hub.vault.concepts import iter_concept_note_paths

console = Console()
log = logging.getLogger("khub.index")

BATCH_SIZE = 20
EMBED_MAX_RETRIES = 3
EMBED_RETRY_BASE_SEC = 1.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sqlite_db(khub):
    if hasattr(khub, "sqlite_db"):
        return khub.sqlite_db()
    from knowledge_hub.infrastructure.persistence import SQLiteDatabase

    return SQLiteDatabase(khub.config.sqlite_path)


def _vector_db(khub):
    if hasattr(khub, "vector_db"):
        try:
            return khub.vector_db(repair_on_init=False)
        except TypeError:
            return khub.vector_db()
    from knowledge_hub.infrastructure.persistence import VectorDatabase

    config = khub.config
    return VectorDatabase(config.vector_db_path, config.collection_name, repair_on_init=False)


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _failure(stage: str, code: str, message: str, file_path: str = "") -> dict:
    return {
        "stage": stage,
        "errorCode": code,
        "message": message,
        "file": file_path,
    }


def _detect_competing_khub_processes() -> list[str]:
    """Detect concurrent khub processes that may contend for model/DB resources."""
    try:
        result = subprocess.run(
            ["ps", "-ax", "-o", "pid=,command="],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return []

    current_pid = os.getpid()
    commands: list[str] = []
    for raw in (result.stdout or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        first_sep = line.find(" ")
        if first_sep <= 0:
            continue
        pid_raw = line[:first_sep].strip()
        command = line[first_sep + 1 :].strip()
        pid = _safe_int(pid_raw, default=-1)
        if pid <= 0 or pid == current_pid:
            continue
        lowered = command.lower()
        if (
            "khub" not in lowered
            and "knowledge_hub.cli.main" not in lowered
            and "knowledge_hub.interfaces.cli.main" not in lowered
        ):
            continue
        if any(token in lowered for token in (" index", " search", " ask ")):
            commands.append(command)
    return commands


def _estimate_vault_chunks(vault_path: str, excludes: list[str], chunk_size: int, chunk_overlap: int) -> tuple[int, int]:
    root = Path(vault_path).expanduser()
    if not root.exists():
        return 0, 0

    md_count = 0
    estimated_chunks = 0
    step = max(1, int(chunk_size) - int(chunk_overlap))
    for md in root.rglob("*.md"):
        if any(ex in md.parts for ex in excludes):
            continue
        md_count += 1
        try:
            size = len(md.read_text(encoding="utf-8"))
        except Exception:
            continue
        estimated_chunks += 1 if size <= chunk_size else max(1, (size + step - 1) // step)
    return md_count, estimated_chunks


def _collect_embedder_status(embedder) -> tuple[int, list[dict]]:
    if not hasattr(embedder, "get_last_status"):
        return 0, []
    try:
        status = embedder.get_last_status() or {}
    except Exception:
        return 0, []

    retries = _safe_int(status.get("retries"), default=0) if isinstance(status, dict) else 0
    failures_raw = status.get("failures", []) if isinstance(status, dict) else []
    failures: list[dict] = []
    if isinstance(failures_raw, list):
        for item in failures_raw:
            if not isinstance(item, dict):
                continue
            failures.append(
                {
                    "stage": str(item.get("stage", "embedder")),
                    "errorCode": str(item.get("errorCode", "EMBEDDER_WARNING")),
                    "message": str(item.get("message", "")),
                    "item": str(item.get("itemIndex", "")),
                }
            )
    return retries, failures


def _save_index_report(config, run_id: str, payload: dict) -> str:
    report_dir = Path(config.indexing_failure_report_dir).expanduser()
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"index-{run_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _redact_provider_config(provider_config: dict) -> dict:
    if not isinstance(provider_config, dict):
        return {}
    redacted = dict(provider_config)
    for key in ("api_key", "token", "access_token", "secret"):
        if key in redacted and redacted[key]:
            value = str(redacted[key])
            redacted[key] = f"{value[:4]}***" if len(value) > 4 else "***"
    return redacted


def _build_next_actions(
    status: str,
    failures: list[dict],
    warnings: list[dict],
    retry_diagnostics: dict[str, int] | None,
    provider: str,
) -> list[str]:
    actions: list[str] = []
    retry_diagnostics = dict(retry_diagnostics or {})
    provider_retries = _safe_int(retry_diagnostics.get("providerRetries"), default=0)
    batch_retries = _safe_int(retry_diagnostics.get("batchRetries"), default=0)
    isolated_retries = _safe_int(retry_diagnostics.get("isolatedRetries"), default=0)
    total_adaptive_retries = _safe_int(retry_diagnostics.get("totalAdaptiveRetries"), default=0)

    if status == "ok" and not warnings:
        actions.append("No action required.")
        return actions

    if status == "ok":
        if provider_retries > 0 or total_adaptive_retries > 0:
            actions.append(
                "Indexing completed with recoverable warnings; monitor retry volume and lower batch size if this becomes frequent."
            )
        else:
            actions.append("Indexing completed with non-fatal warnings; inspect the run report if the same warnings recur.")
        if provider == "pplx-st":
            actions.append(
                "If recoverable provider stalls recur, set providers.pplx-st.torch_num_threads=1 and disable_tokenizers_parallelism=true."
            )
        return actions[:5]

    if batch_retries > 0 or isolated_retries > 0:
        actions.append("Retry failed items with lower batch size (e.g., providers.pplx-st.batch_size=4).")
    if provider == "pplx-st":
        actions.append("If lock-like stalls recur, set providers.pplx-st.torch_num_threads=1 and disable_tokenizers_parallelism=true.")
        actions.append("If model load fails on mps, retry with providers.pplx-st.device=cpu.")

    error_codes = {str(item.get("errorCode", "")) for item in failures}
    if "MODEL_LOAD_TIMEOUT" in error_codes or "MODEL_LOAD_FAILED" in error_codes:
        actions.append("Clear Hugging Face cache for the model and retry indexing.")
    if "PARSE_FAILED" in error_codes:
        actions.append("Inspect parse-failed markdown files in the report and fix malformed frontmatter/content.")
    if "EMBEDDING_NONE" in error_codes or "ENCODE_TIMEOUT" in error_codes:
        actions.append("Reduce max_chars_per_chunk/chunk_size and rerun failed vault indexing.")

    if not actions:
        actions.append("Inspect failures in the run report and rerun with --verbose.")
    return actions[:5]


def _config_nested(config, *path: str, default=None):
    getter = getattr(config, "get_nested", None)
    if callable(getter):
        try:
            return getter(*path, default=default)
        except TypeError:
            try:
                return getter(*path)
            except Exception:
                return default
        except Exception:
            return default
    return default


def _config_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _index_resource_thresholds(config) -> tuple[float, float, float, float]:
    if hasattr(config, "get_nested"):
        try:
            from knowledge_hub.web.ingest_pipeline_support import resource_thresholds

            return resource_thresholds(SimpleNamespace(config=config))
        except Exception:
            pass
    return (0.82, 0.90, 1.0, 30.0)


def _index_resource_ratios(config) -> tuple[float, float]:
    del config
    try:
        from knowledge_hub.web.ingest_pipeline_support import sample_resource_ratio

        return sample_resource_ratio(SimpleNamespace(config=None))
    except Exception:
        return (0.0, 0.0)


def _apply_resource_backoff(config, *, backoff_round: int) -> dict[str, float | int]:
    mem_high, cpu_high, backoff_base, backoff_max = _index_resource_thresholds(config)
    memory_ratio, cpu_ratio = _index_resource_ratios(config)
    if memory_ratio < mem_high and cpu_ratio < cpu_high:
        return {
            "memory_ratio": float(memory_ratio),
            "cpu_ratio": float(cpu_ratio),
            "sleep_sec": 0.0,
            "next_round": 0,
        }

    sleep_sec = min(backoff_max, backoff_base * (2 ** max(0, int(backoff_round))))
    time.sleep(sleep_sec)
    return {
        "memory_ratio": float(memory_ratio),
        "cpu_ratio": float(cpu_ratio),
        "sleep_sec": float(sleep_sec),
        "next_round": min(max(0, int(backoff_round)) + 1, 12),
    }


def _run_index_batches(
    *,
    config,
    items: list[Any],
    embedder,
    vector_db,
    build_text: Callable[[Any], str],
    build_metadata: Callable[[Any], dict[str, Any]],
    build_id: Callable[[Any], str],
    failure_builder: Callable[[Any | None, str], dict[str, Any]],
    warning_builder: Callable[[Any | None, str], dict[str, Any]] | None = None,
    before_add: Callable[[list[Any]], None] | None = None,
    mark_indexed: Callable[[list[Any]], None] | None = None,
) -> dict[str, Any]:
    if not items:
        return {
            "succeeded": 0,
            "failures": [],
            "warnings": [],
            "retries": 0,
            "retryDiagnostics": {
                "providerRetries": 0,
                "batchRetries": 0,
                "isolatedRetries": 0,
                "totalAdaptiveRetries": 0,
            },
            "adaptive": {
                "batchSizeInitial": max(1, BATCH_SIZE),
                "batchSizeMin": 1,
                "batchSizeHistory": [max(1, BATCH_SIZE)],
                "batchRetryCount": 0,
                "isolatedRetryCount": 0,
                "resourceBackoffCount": 0,
                "resourceBackoffSeconds": 0.0,
            },
        }

    initial_batch_size = max(
        1,
        _safe_int(_config_nested(config, "indexing", "embed_batch_size", default=BATCH_SIZE), default=BATCH_SIZE),
    )
    min_batch_size = max(
        1,
        min(
            initial_batch_size,
            _safe_int(_config_nested(config, "indexing", "min_batch_size", default=1), default=1),
        ),
    )
    max_batch_retries = max(
        0,
        _safe_int(_config_nested(config, "indexing", "max_batch_retries", default=3), default=3),
    )
    pause_ms = max(
        0,
        _safe_int(_config_nested(config, "indexing", "embed_pause_ms", default=50), default=50),
    )
    auto_batch_backoff = _config_bool(
        _config_nested(config, "indexing", "auto_batch_backoff", default=True),
        True,
    )

    provider_retries = 0
    succeeded = 0
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    batch_retry_count = 0
    isolated_retry_count = 0
    resource_backoff_count = 0
    resource_backoff_seconds = 0.0
    batch_size_history = [initial_batch_size]
    cursor = 0
    backoff_round = 0

    def _collect_status(item: Any | None = None) -> None:
        nonlocal provider_retries
        retries, embedder_failures = _collect_embedder_status(embedder)
        provider_retries += retries
        builder = warning_builder or failure_builder
        for warning in embedder_failures:
            warnings.append(builder(item, f"{warning['errorCode']}: {warning['message']}"))

    def _index_single(item: Any) -> bool:
        nonlocal succeeded, isolated_retry_count
        isolated_retry_count += 1
        status_collected = False
        try:
            text = build_text(item)
            embs = _embed_batch_via_provider(embedder, [text])
            _collect_status(item)
            status_collected = True
            if not embs or embs[0] is None:
                raise RuntimeError("embedder returned None for isolated item")
            if before_add is not None:
                before_add([item])
            vector_db.add_documents(
                documents=[text],
                embeddings=[embs[0]],
                metadatas=[build_metadata(item)],
                ids=[build_id(item)],
            )
            if mark_indexed is not None:
                mark_indexed([item])
            succeeded += 1
            return True
        except Exception as error:
            if not status_collected:
                _collect_status(item)
            failures.append(failure_builder(item, str(error)))
            return False

    while cursor < len(items):
        backoff_meta = _apply_resource_backoff(config, backoff_round=backoff_round)
        backoff_round = int(backoff_meta["next_round"])
        if float(backoff_meta["sleep_sec"]) > 0.0:
            resource_backoff_count += 1
            resource_backoff_seconds += float(backoff_meta["sleep_sec"])

        candidate_size = min(initial_batch_size, len(items) - cursor)
        local_batch_retries = 0

        while True:
            batch = items[cursor : cursor + candidate_size]
            if not batch:
                break
            status_collected = False
            try:
                texts = [build_text(item) for item in batch]
                embs = _embed_batch_via_provider(embedder, texts)
                _collect_status()
                status_collected = True
                if before_add is not None:
                    before_add(batch)
                vector_db.add_documents(
                    documents=texts,
                    embeddings=embs,
                    metadatas=[build_metadata(item) for item in batch],
                    ids=[build_id(item) for item in batch],
                )
                if mark_indexed is not None:
                    mark_indexed(batch)
                succeeded += len(batch)
                cursor += len(batch)
                if pause_ms:
                    time.sleep(pause_ms / 1000.0)
                break
            except Exception as error:
                if not status_collected:
                    _collect_status()
                if len(batch) > 1:
                    batch_retry_count += 1
                log.warning(
                    "adaptive index batch failed at cursor=%d size=%d: %s",
                    cursor,
                    len(batch),
                    error,
                )
                if auto_batch_backoff and local_batch_retries < max_batch_retries:
                    next_size = max(min_batch_size, len(batch) // 2)
                    if next_size < len(batch):
                        candidate_size = next_size
                        local_batch_retries += 1
                        if batch_size_history[-1] != candidate_size:
                            batch_size_history.append(candidate_size)
                        continue

                for item in batch:
                    _index_single(item)
                cursor += len(batch)
                if pause_ms:
                    time.sleep(pause_ms / 1000.0)
                break

    return {
        "succeeded": succeeded,
        "failures": failures,
        "warnings": warnings,
        "retries": int(provider_retries + batch_retry_count + isolated_retry_count),
        "retryDiagnostics": {
            "providerRetries": int(provider_retries),
            "batchRetries": int(batch_retry_count),
            "isolatedRetries": int(isolated_retry_count),
            "totalAdaptiveRetries": int(batch_retry_count + isolated_retry_count),
        },
        "adaptive": {
            "batchSizeInitial": initial_batch_size,
            "batchSizeMin": min_batch_size,
            "batchSizeHistory": batch_size_history,
            "batchRetryCount": batch_retry_count,
            "isolatedRetryCount": isolated_retry_count,
            "resourceBackoffCount": resource_backoff_count,
            "resourceBackoffSeconds": round(resource_backoff_seconds, 3),
        },
    }


def _get_embedder(config, khub=None):
    """config 기반으로 적절한 Embedder 인스턴스 생성"""
    if khub is not None and hasattr(khub, "build_embedder"):
        return khub.build_embedder(config.embedding_provider, config.embedding_model)
    from knowledge_hub.infrastructure.providers import get_embedder

    embed_cfg = config.get_provider_config(config.embedding_provider)
    return get_embedder(config.embedding_provider, model=config.embedding_model, **embed_cfg)


def _embed_batch_openai(texts: list[str], model: str, api_key: str | None = None) -> list[list[float]]:
    """OpenAI 임베딩 직접 호출(레거시 재시도 테스트 호환용)."""
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai package is required for openai embedding mode") from exc

    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def _embed_with_retry(
    texts: list[str],
    provider: str,
    model: str,
    api_key: str | None = None,
) -> list[list[float]]:
    """429/5xx에 대해 재시도하는 임베딩 호출 래퍼(테스트 계약 유지)."""
    if provider != "openai":
        raise ValueError(f"unsupported provider for retry helper: {provider}")

    for attempt in range(EMBED_MAX_RETRIES + 1):
        try:
            return _embed_batch_openai(texts, model=model, api_key=api_key)
        except requests.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = status in {429, 500, 502, 503, 504}
            final_attempt = attempt >= EMBED_MAX_RETRIES
            if (not retryable) or final_attempt:
                raise
            sleep_for = EMBED_RETRY_BASE_SEC * (2 ** attempt)
            time.sleep(sleep_for)
    # 루프 구조상 도달하지 않음
    raise RuntimeError("embed retry exhausted")


def _embed_batch_via_provider(embedder, texts: list[str]) -> list[list[float]]:
    """프로바이더 추상화를 통해 배치 임베딩"""
    results = embedder.embed_batch(texts, show_progress=False)
    failed = [i for i, r in enumerate(results) if r is None]
    if failed:
        raise RuntimeError(f"{len(failed)}개 텍스트 임베딩 실패 (인덱스: {failed[:5]})")
    return results


def _get_paper_keywords(vault_path: str) -> dict[str, list[str]]:
    """Obsidian 논문 노트에서 arxiv_id → 키워드 목록 매핑 추출"""
    papers_dir = Path(vault_path) / "Papers"
    if not papers_dir.exists():
        alt = Path(vault_path) / "Projects" / "AI" / "AI_Papers"
        if alt.exists():
            papers_dir = alt
        else:
            return {}

    mapping: dict[str, list[str]] = {}
    for md_path in papers_dir.glob("*.md"):
        if md_path.name == "00_Concept_Index.md":
            continue
        try:
            content = md_path.read_text(encoding="utf-8")
        except Exception:
            continue
        arxiv_match = re.search(r'arxiv_id:\s*"?([0-9]+\.[0-9]+)"?', content)
        if not arxiv_match:
            continue
        aid = arxiv_match.group(1)
        concepts = re.findall(r'\[\[([^\]]+)\]\]', content)
        concepts = [c for c in concepts if c != "00_Concept_Index"]
        if concepts:
            mapping[aid] = concepts
    return mapping


def _load_concept_notes(vault_path: str) -> list[dict]:
    """Obsidian 개념 노트 로드 → 임베딩용 데이터 리스트 반환"""
    candidates = [
        Path(vault_path) / "AI" / "AI_Papers" / "Concepts",
        Path(vault_path) / "Papers" / "Concepts",
        Path(vault_path) / "Projects" / "AI" / "AI_Papers" / "Concepts",
        Path(vault_path) / "Concepts",
    ]
    concepts_dir = None
    for c in candidates:
        if c.exists():
            concepts_dir = c
            break
    if concepts_dir is None:
        return []

    results = []
    for md_path in iter_concept_note_paths(concepts_dir):
        try:
            content = md_path.read_text(encoding="utf-8")
        except Exception:
            continue
        name = md_path.stem
        note_key = str(md_path.relative_to(concepts_dir).with_suffix("")).replace("\\", "/")

        desc_match = re.search(r'^# .+\n\n(.+?)(?:\n\n##|\Z)', content, re.MULTILINE | re.DOTALL)
        description = desc_match.group(1).strip() if desc_match else ""

        related = re.findall(r'## 관련 개념\n((?:- \[\[.+?\]\]\n)*)', content)
        related_names = re.findall(r'\[\[([^\]]+)\]\]', related[0]) if related else []

        papers = re.findall(r'## 관련 논문\n((?:- \[\[.+?\]\]\n)*)', content)
        paper_names = re.findall(r'\[\[([^\]]+)\]\]', papers[0]) if papers else []

        text_parts = [f"Concept: {name}"]
        if description:
            text_parts.append(description)
        if related_names:
            text_parts.append(f"Related concepts: {', '.join(related_names)}")
        if paper_names:
            text_parts.append(f"Papers: {', '.join(paper_names)}")

        results.append({
            "name": name,
            "note_key": note_key,
            "text": "\n\n".join(text_parts),
            "related": related_names,
            "papers": paper_names,
        })
    return results


@click.command("index")
@click.option("--all", "index_all", is_flag=True, help="이미 인덱싱된 논문도 재인덱싱")
@click.option("--concepts-only", is_flag=True, help="개념 노트만 인덱싱")
@click.option("--vault-all", is_flag=True, help="Obsidian vault의 전체 Markdown 파일을 청크 단위로 인덱싱")
@click.option("--vault-clear", is_flag=True, help="vault 인덱싱 전에 source_type=vault 문서를 삭제")
@click.option("--json/--no-json", "as_json", default=False, show_default=True, help="요약 결과를 JSON으로 출력")
@click.pass_context
def index_cmd(ctx, index_all, concepts_only, vault_all, vault_clear, as_json):
    """논문 + 개념 노트를 벡터DB에 통합 인덱싱"""
    from knowledge_hub.vault.indexer import VaultIndexer

    khub = ctx.obj["khub"]
    config = khub.config

    try:
        if hasattr(khub, "build_embedder"):
            embedder = _get_embedder(config, khub=khub)
        else:
            embedder = _get_embedder(config)
    except Exception as e:
        console.print(f"[red]임베딩 프로바이더 초기화 실패: {e}[/red]")
        console.print("[dim]khub config providers 로 사용 가능한 프로바이더를 확인하세요.[/dim]")
        raise SystemExit(1)

    try:
        vector_db = _vector_db(khub)
    except Exception as e:
        console.print(f"[red]벡터DB 초기화 실패: {e}[/red]")
        message = str(e or "").lower()
        if "chroma" in message or "tenant" in message or "default_tenant" in message:
            console.print(
                "[dim]복구 후보가 있으면 `khub vector-restore --latest-backup`로 preview 후 "
                "`--apply --confirm`으로 복구하세요.[/dim]"
            )
        raise SystemExit(1)

    run_id = f"idx_{uuid4().hex[:12]}"
    started_at = _now_iso()
    console.print(f"[dim]run_id={run_id}[/dim]")
    console.print(f"[dim]임베딩: {config.embedding_provider}/{config.embedding_model}[/dim]")

    provider_snapshot = {
        "provider": config.embedding_provider,
        "model": config.embedding_model,
        "collectionName": config.collection_name,
        "vectorDbPath": config.vector_db_path,
        "sqlitePath": config.sqlite_path,
        "providerConfig": _redact_provider_config(config.get_provider_config(config.embedding_provider)),
    }

    preflight_warnings: list[str] = []
    competing = _detect_competing_khub_processes()
    if competing:
        preflight_warnings.append(f"competing khub processes detected: {len(competing)}")
        console.print("[yellow]동시 khub 프로세스 감지됨 (락/경합 가능):[/yellow]")
        for command in competing[:5]:
            console.print(f"  - {command[:160]}")
        if len(competing) > 5:
            console.print(f"  - ... and {len(competing) - 5} more")

    if config.embedding_provider == "pplx-st":
        st_cfg = config.get_provider_config("pplx-st")
        if _safe_int(st_cfg.get("torch_num_threads"), default=1) != 1:
            preflight_warnings.append("pplx-st torch_num_threads is not 1; lock contention risk")
            console.print("[yellow]권장: providers.pplx-st.torch_num_threads=1[/yellow]")
        if not bool(st_cfg.get("disable_tokenizers_parallelism", True)):
            preflight_warnings.append("pplx-st disable_tokenizers_parallelism is false; lock contention risk")
            console.print("[yellow]권장: providers.pplx-st.disable_tokenizers_parallelism=true[/yellow]")

    preflight_vault_files = 0
    preflight_estimated_chunks = 0
    if vault_all and config.vault_path:
        preflight_vault_files, preflight_estimated_chunks = _estimate_vault_chunks(
            vault_path=config.vault_path,
            excludes=config.vault_excludes,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        if vault_clear:
            preflight_warnings.append("--vault-clear is redundant; vault-all now performs authoritative vault resync")
            console.print("[yellow]--vault-clear는 호환용입니다. vault-all은 이제 기본적으로 vault를 재동기화합니다.[/yellow]")
        console.print(
            f"[dim]vault preflight: files={preflight_vault_files}, "
            f"estimated_chunks={preflight_estimated_chunks}[/dim]"
        )

    t_start = time.time()
    failed_papers: list[dict] = []
    failed_concepts: list[dict] = []
    failed_vault: list[dict] = []
    runtime_warnings: list[dict] = []
    total_retries = 0
    retry_diagnostics = {
        "providerRetries": 0,
        "batchRetries": 0,
        "isolatedRetries": 0,
        "totalAdaptiveRetries": 0,
    }
    adaptive_indexing = {
        "papers": {
            "batchSizeInitial": max(1, BATCH_SIZE),
            "batchSizeMin": 1,
            "batchSizeHistory": [max(1, BATCH_SIZE)],
            "batchRetryCount": 0,
            "isolatedRetryCount": 0,
            "resourceBackoffCount": 0,
            "resourceBackoffSeconds": 0.0,
        },
        "concepts": {
            "batchSizeInitial": max(1, BATCH_SIZE),
            "batchSizeMin": 1,
            "batchSizeHistory": [max(1, BATCH_SIZE)],
            "batchRetryCount": 0,
            "isolatedRetryCount": 0,
            "resourceBackoffCount": 0,
            "resourceBackoffSeconds": 0.0,
        },
    }
    planned_papers = 0
    planned_concepts = 0

    keyword_map = {}
    if config.vault_path:
        try:
            keyword_map = _get_paper_keywords(config.vault_path)
        except Exception as e:
            console.print(f"[yellow]키워드 로드 실패 (계속 진행): {e}[/yellow]")

    # ── Phase 1: 논문 인덱싱 ──
    paper_success = 0
    if not concepts_only:
        sqlite_db = _sqlite_db(khub)
        papers = sqlite_db.list_papers(limit=999)
        unindexed = papers if index_all else [p for p in papers if not p.get("indexed")]
        planned_papers = len(unindexed)

        if unindexed:
            console.print(f"[bold]논문 인덱싱: {len(unindexed)}편[/bold]")

            def _mark_papers_indexed(batch: list[dict[str, Any]]) -> None:
                for paper in batch:
                    sqlite_db.conn.execute(
                        "UPDATE papers SET indexed = 1 WHERE arxiv_id = ?",
                        (paper["arxiv_id"],),
                    )
                sqlite_db.conn.commit()

            def _paper_source_hash(paper: dict[str, Any]) -> str:
                text = _build_paper_embedding_text(
                    sqlite_db,
                    paper=paper,
                    config=khub.config,
                    keywords=keyword_map.get(paper["arxiv_id"], []),
                )
                return fallback_source_hash(text, paper.get("arxiv_id"))

            def _prepare_paper_vector_replace(batch: list[dict[str, Any]]) -> None:
                delete_by_metadata = getattr(vector_db, "delete_by_metadata", None)
                for paper in batch:
                    arxiv_id = str(paper.get("arxiv_id") or "").strip()
                    if not arxiv_id:
                        continue
                    source_hash = _paper_source_hash(paper)
                    if source_hash:
                        mark_derivatives_stale_for_document(
                            sqlite_db.conn,
                            document_id=f"paper:{arxiv_id}",
                            source_content_hash=source_hash,
                            source_type="paper",
                        )
                    if callable(delete_by_metadata):
                        delete_by_metadata({"source_type": "paper", "arxiv_id": arxiv_id})

            paper_run = _run_index_batches(
                config=config,
                items=unindexed,
                embedder=embedder,
                vector_db=vector_db,
                build_text=lambda paper: _build_paper_embedding_text(
                    sqlite_db,
                    paper=paper,
                    config=khub.config,
                    keywords=keyword_map.get(paper["arxiv_id"], []),
                ),
                build_metadata=lambda paper: {
                    "title": paper["title"] or "",
                    "arxiv_id": paper["arxiv_id"],
                    "source_type": "paper",
                    "field": paper.get("field", ""),
                    "keywords": ", ".join(keyword_map.get(paper["arxiv_id"], [])[:10]),
                    "document_id": f"paper:{paper['arxiv_id']}",
                    "source_content_hash": _paper_source_hash(paper),
                    "stale": 0,
                    "chunk_index": 0,
                },
                build_id=lambda paper: f"paper_{paper['arxiv_id']}_0",
                failure_builder=lambda paper, message: {
                    "arxiv_id": paper["arxiv_id"] if paper else "",
                    "title": paper["title"] if paper else "",
                    "error": message,
                },
                warning_builder=lambda paper, message: {
                    "arxiv_id": paper["arxiv_id"] if paper else "",
                    "title": paper["title"] if paper else "",
                    "error": message,
                },
                before_add=_prepare_paper_vector_replace,
                mark_indexed=_mark_papers_indexed,
            )
            paper_success = int(paper_run["succeeded"])
            total_retries += int(paper_run["retries"])
            failed_papers.extend(list(paper_run["failures"]))
            runtime_warnings.extend(
                _failure("paper", "PAPER_INDEX_WARNING", item.get("error", ""), item.get("arxiv_id", ""))
                for item in paper_run.get("warnings", [])
            )
            paper_retry_diagnostics = dict(paper_run.get("retryDiagnostics") or {})
            retry_diagnostics["providerRetries"] += _safe_int(paper_retry_diagnostics.get("providerRetries"), default=0)
            retry_diagnostics["batchRetries"] += _safe_int(paper_retry_diagnostics.get("batchRetries"), default=0)
            retry_diagnostics["isolatedRetries"] += _safe_int(paper_retry_diagnostics.get("isolatedRetries"), default=0)
            retry_diagnostics["totalAdaptiveRetries"] += _safe_int(
                paper_retry_diagnostics.get("totalAdaptiveRetries"),
                default=0,
            )
            adaptive_indexing["papers"] = dict(paper_run["adaptive"])

            console.print(f"  [bold green]논문 {paper_success}/{len(unindexed)}편 완료[/bold green]")
        else:
            console.print("[green]모든 논문이 이미 인덱싱되어 있습니다.[/green]")

    # ── Phase 2: 개념 노트 인덱싱 ──
    concept_success = 0
    if config.vault_path:
        try:
            concept_notes = _load_concept_notes(config.vault_path)
        except Exception as e:
            console.print(f"[red]개념 노트 로드 실패: {e}[/red]")
            concept_notes = []

        if concept_notes:
            existing_ids = set()
            try:
                existing = vector_db.collection.get(
                    where={"source_type": "concept"},
                    include=[],
                )
                existing_ids = set(existing.get("ids", []))
            except Exception as error:
                log.warning("failed to load existing concept ids for dedupe: %s", error)

            if not index_all:
                concept_notes = [c for c in concept_notes
                                 if f"concept_{c.get('note_key') or c['name']}" not in existing_ids]
            planned_concepts = len(concept_notes)

            if concept_notes:
                console.print(f"\n[bold]개념 노트 인덱싱: {len(concept_notes)}개[/bold]")
                concept_run = _run_index_batches(
                    config=config,
                    items=concept_notes,
                    embedder=embedder,
                    vector_db=vector_db,
                    build_text=lambda concept: concept["text"],
                    build_metadata=lambda concept: {
                        "title": concept["name"],
                        "note_key": concept.get("note_key") or concept["name"],
                        "source_type": "concept",
                        "related_concepts": ", ".join(concept["related"][:5]),
                        "related_papers": ", ".join(concept["papers"][:5]),
                        "chunk_index": 0,
                    },
                    build_id=lambda concept: f"concept_{concept.get('note_key') or concept['name']}",
                    failure_builder=lambda concept, message: {
                        "name": concept["name"] if concept else "",
                        "error": message,
                    },
                    warning_builder=lambda concept, message: {
                        "name": concept["name"] if concept else "",
                        "error": message,
                    },
                )
                concept_success = int(concept_run["succeeded"])
                total_retries += int(concept_run["retries"])
                failed_concepts.extend(list(concept_run["failures"]))
                runtime_warnings.extend(
                    _failure("concept", "CONCEPT_INDEX_WARNING", item.get("error", ""), item.get("name", ""))
                    for item in concept_run.get("warnings", [])
                )
                concept_retry_diagnostics = dict(concept_run.get("retryDiagnostics") or {})
                retry_diagnostics["providerRetries"] += _safe_int(concept_retry_diagnostics.get("providerRetries"), default=0)
                retry_diagnostics["batchRetries"] += _safe_int(concept_retry_diagnostics.get("batchRetries"), default=0)
                retry_diagnostics["isolatedRetries"] += _safe_int(concept_retry_diagnostics.get("isolatedRetries"), default=0)
                retry_diagnostics["totalAdaptiveRetries"] += _safe_int(
                    concept_retry_diagnostics.get("totalAdaptiveRetries"),
                    default=0,
                )
                adaptive_indexing["concepts"] = dict(concept_run["adaptive"])

                console.print(f"  [bold green]개념 {concept_success}/{len(concept_notes)}개 완료[/bold green]")
            else:
                console.print("[green]모든 개념 노트가 이미 인덱싱되어 있습니다.[/green]")
    else:
        console.print("[dim]Obsidian vault 미설정 - 개념 노트 인덱싱 건너뜀[/dim]")

    # ── Phase 3: Vault 전체 인덱싱(옵션) ──
    vault_summary = None
    if vault_all:
        if not config.vault_path:
            console.print("[red]vault_path 미설정: khub config set obsidian.vault_path <경로>[/red]")
            raise SystemExit(1)

        console.print("\n[bold]Vault 전체 인덱싱 시작[/bold]")
        sqlite_db = _sqlite_db(khub)
        vault_indexer = VaultIndexer(
            config=config,
            vector_db=vector_db,
            sqlite_db=sqlite_db,
            embedder=embedder,
        )
        try:
            vault_summary = vault_indexer.index(
                vault_path=config.vault_path,
                clear=False,
                authoritative=True,
            )
            if isinstance(vault_summary, dict):
                total_retries += _safe_int(vault_summary.get("retries"), default=0)
                failed_vault.extend(vault_summary.get("failures", []) or [])
                runtime_warnings.extend(vault_summary.get("warnings", []) or [])
                vault_retry_diagnostics = dict(vault_summary.get("retryDiagnostics") or {})
                retry_diagnostics["providerRetries"] += _safe_int(vault_retry_diagnostics.get("providerRetries"), default=0)
                retry_diagnostics["batchRetries"] += _safe_int(vault_retry_diagnostics.get("batchRetries"), default=0)
                retry_diagnostics["isolatedRetries"] += _safe_int(vault_retry_diagnostics.get("isolatedRetries"), default=0)
                retry_diagnostics["totalAdaptiveRetries"] += _safe_int(
                    vault_retry_diagnostics.get("totalAdaptiveRetries"),
                    default=0,
                )
        except Exception as e:
            console.print(f"[red]Vault 인덱싱 실패: {e}[/red]")
            raise SystemExit(1)

    elapsed = max(0.001, time.time() - t_start)
    vault_processed = _safe_int(vault_summary.get("processed"), 0) if isinstance(vault_summary, dict) else 0
    vault_succeeded = _safe_int(vault_summary.get("succeeded"), 0) if isinstance(vault_summary, dict) else 0
    vault_failed = _safe_int(vault_summary.get("failed"), 0) if isinstance(vault_summary, dict) else 0

    processed_total = planned_papers + planned_concepts + vault_processed
    succeeded_total = paper_success + concept_success + vault_succeeded

    failures_payload = [
        _failure("paper", "PAPER_INDEX_FAILED", item.get("error", ""), item.get("arxiv_id", ""))
        for item in failed_papers
    ] + [
        _failure("concept", "CONCEPT_INDEX_FAILED", item.get("error", ""), item.get("name", ""))
        for item in failed_concepts
    ] + [
        _failure(
            str(item.get("stage", "vault")),
            str(item.get("errorCode", "VAULT_INDEX_FAILED")),
            str(item.get("message", "")),
            str(item.get("file", "")),
        )
        for item in failed_vault
    ]
    failed_total = len(failures_payload)
    status = "ok"
    if failed_total > 0 and succeeded_total > 0:
        status = "partial"
    elif failed_total > 0 and succeeded_total == 0:
        status = "error"

    payload = {
        "schema": "knowledge-hub.index.result.v1",
        "runId": run_id,
        "status": status,
        "startedAt": started_at,
        "finishedAt": _now_iso(),
        "providerConfigSnapshot": provider_snapshot,
        "preflight": {
            "warnings": preflight_warnings,
            "competingProcesses": competing,
            "vaultFiles": preflight_vault_files,
            "vaultEstimatedChunks": preflight_estimated_chunks,
        },
        "processed": processed_total,
        "processedBreakdown": {
            "papers": planned_papers,
            "concepts": planned_concepts,
            "vault": {
                "processed": vault_processed,
                "succeeded": vault_succeeded,
                "failed": vault_failed,
            },
        },
        "succeeded": succeeded_total,
        "failed": failed_total,
        "retries": total_retries,
        "warnings": runtime_warnings,
        "failures": failures_payload,
        "retryDiagnostics": retry_diagnostics,
        "adaptiveIndexing": adaptive_indexing,
        "durationSec": round(elapsed, 3),
        "throughput": {
            "totalPerMin": round((processed_total / elapsed) * 60.0, 3),
            "vaultChunksPerMin": (
                float(vault_summary.get("throughputChunksPerMin", 0.0))
                if isinstance(vault_summary, dict)
                else 0.0
            ),
        },
        "vectorDbCount": vector_db.count(),
    }
    payload["nextActions"] = _build_next_actions(
        status=status,
        failures=payload["failures"],
        warnings=payload["warnings"],
        retry_diagnostics=payload["retryDiagnostics"],
        provider=config.embedding_provider,
    )
    report_path = _save_index_report(config=config, run_id=run_id, payload=payload)
    console.print(f"[dim]failure report: {report_path}[/dim]")
    payload["reportPath"] = report_path

    console.print(
        f"\n[bold]통합 인덱싱 완료 ({elapsed:.1f}초)[/bold]"
        f"\n  상태: {status}"
        f"\n  논문: {paper_success}편 | 개념: {concept_success}개"
        f"\n  vault 실패 청크: {_safe_int(payload['processedBreakdown']['vault']['failed'], 0)}"
        f"\n  재시도: {total_retries}"
        f"\n  벡터DB 총: {vector_db.count()}개 문서"
    )

    if payload["failures"]:
        console.print("\n[bold red]실패 항목(최대 20):[/bold red]")
        for item in payload["failures"][:20]:
            console.print(
                f"  [{item['stage']}] {item['file'] or '-'} "
                f"{item['errorCode']}: {item['message'][:120]}"
            )

    if as_json:
        console.print_json(data=payload)
