"""Paper CLI runtime helpers for paper materialization flows."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import click
from rich.markdown import Markdown

from knowledge_hub.interfaces.cli.commands.paper_shared_runtime import (
    _build_paper_embedding_text,
    _resolve_summary_build_options,
)
from knowledge_hub.papers.source_guard import review_downloaded_source


def _safe_title(title: str) -> str:
    normalized = re.sub(r'[\\/:*?"<>|]', "", title).strip()
    return re.sub(r"\s+", " ", normalized)[:100].strip()


def _extract_json_payload(raw: str) -> dict[str, Any]:
    token = str(raw or "").strip()
    if not token:
        raise ValueError("empty JSON payload")
    try:
        payload = json.loads(token)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    lines = token.splitlines()
    for index, line in enumerate(lines):
        if line.lstrip().startswith("{"):
            candidate = "\n".join(lines[index:])
            try:
                payload = json.loads(candidate)
                if isinstance(payload, dict):
                    return payload
            except Exception:
                continue

    start = token.find("{")
    end = token.rfind("}")
    if start >= 0 and end > start:
        payload = json.loads(token[start : end + 1])
        if isinstance(payload, dict):
            return payload
    raise ValueError("no JSON object found in worker output")


def _default_summary_batch_checkpoint_path(config: Any) -> Path:
    target = Path(str(getattr(config, "papers_dir", "") or "")).expanduser() / "summaries" / "_summarize_all.checkpoint.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _append_summary_batch_checkpoint(checkpoint_file: Path | None, payload: dict[str, Any]) -> None:
    if checkpoint_file is None:
        return
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _summary_quality_target(
    *,
    khub: Any,
    paper_row: dict[str, Any],
    assess_summary_quality_fn: Callable[[str], dict[str, Any]],
    render_structured_summary_notes_fn: Callable[[dict[str, Any]], str],
    summary_payload: dict[str, Any],
    build_public_summary_card_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    paper_id = str(paper_row.get("arxiv_id") or "").strip()
    try:
        public_payload = build_public_summary_card_fn(khub, paper_id=paper_id)
        quality = dict(public_payload.get("quality") or {})
        score = quality.get("score")
        if score is not None:
            return {
                "score": int(score),
                "band": str(quality.get("band") or ""),
                "summaryStatus": str(quality.get("summaryStatus") or ""),
                "reasons": [str(item) for item in list(quality.get("reasons") or []) if str(item or "").strip()],
            }
    except Exception:
        pass

    source_text = render_structured_summary_notes_fn(summary_payload) if summary_payload else str(paper_row.get("notes") or "")
    fallback_quality = dict(assess_summary_quality_fn(source_text) or {})
    return {
        "score": int(fallback_quality.get("score") or 0),
        "band": "",
        "summaryStatus": "",
        "reasons": [str(item) for item in list(fallback_quality.get("reasons") or []) if str(item or "").strip()],
    }


def _run_structured_summary_batch_worker(
    *,
    config: Any,
    paper_id: str,
    paper_parser: str,
    quick: bool,
    provider: str | None,
    model: str | None,
    allow_external: bool,
    llm_mode: str,
    timeout_sec: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "knowledge_hub.interfaces.cli.main",
        "labs",
        "paper-summary",
        "build",
        "--paper-id",
        str(paper_id).strip(),
        "--paper-parser",
        str(paper_parser or "auto").strip().lower(),
        "--json",
    ]
    if quick:
        command.append("--quick")
    if provider:
        command.extend(["--provider", str(provider)])
    if model:
        command.extend(["--model", str(model)])
    command.append("--allow-external" if allow_external else "--no-allow-external")
    command.extend(["--llm-mode", str(llm_mode or "auto")])

    cwd = None
    config_path = str(getattr(config, "config_path", "") or "").strip()
    if config_path:
        candidate = Path(config_path).expanduser().resolve().parent
        if candidate.exists():
            cwd = str(candidate)

    env = dict(os.environ)
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_sec or 1)),
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(f"summary worker timed out after {int(timeout_sec or 0)}s") from error

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        detail = " ".join(detail.split())[:400] or f"worker exited with code {completed.returncode}"
        raise RuntimeError(detail)

    output = completed.stdout or ""
    if not output.strip() and completed.stderr:
        output = completed.stderr
    return _extract_json_payload(output)


def run_paper_download(
    *,
    khub: Any,
    arxiv_id: str,
    console: Any,
    sqlite_db_fn: Callable[..., Any],
    downloader_factory: Callable[[str], Any],
) -> None:
    config = khub.config
    downloader = downloader_factory(config.papers_dir)
    sqlite_db = sqlite_db_fn(config, khub=khub)

    existing = sqlite_db.get_paper(arxiv_id)
    title = existing["title"] if existing else arxiv_id

    try:
        with console.status(f"다운로드 중: {arxiv_id}..."):
            result = downloader.download_single(arxiv_id, title)
    except Exception as error:
        console.print(f"[red]다운로드 실패: {error}[/red]")
        return

    if result["success"]:
        review = review_downloaded_source(
            sqlite_db,
            paper_id=str(arxiv_id or "").strip(),
            title=str(title or arxiv_id).strip(),
            pdf_path=str(result.get("pdf") or ""),
            text_path=str(result.get("text") or ""),
        )
        guard = dict(review.get("guard") or {})
        if bool(review.get("blocked")):
            console.print(f"[yellow]source guard blocked import[/yellow]: {guard.get('reason')}")
            return
        paper_data = {
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": existing.get("authors", "") if existing else "",
            "year": existing.get("year", 0) if existing else 0,
            "field": existing.get("field", "") if existing else "",
            "importance": existing.get("importance", 3) if existing else 3,
            "notes": existing.get("notes", "") if existing else "",
            "pdf_path": str(review.get("finalPdfPath") or result.get("pdf") or "") or None,
            "text_path": str(review.get("finalTextPath") or result.get("text") or "") or None,
            "translated_path": existing.get("translated_path") if existing else None,
        }
        sqlite_db.upsert_paper(paper_data)
        if str(guard.get("decision") or "") == "relink_to_canonical":
            console.print(
                f"[yellow]source guard relink[/yellow]: {guard.get('canonicalPaperId') or '-'}"
            )
        console.print(f"[green]다운로드 완료: {paper_data.get('pdf_path') or result.get('pdf', 'N/A')}[/green]")
        return

    console.print(f"[red]다운로드 실패: {arxiv_id}[/red]")


def run_paper_translate(
    *,
    khub: Any,
    arxiv_id: str,
    provider: str | None,
    model: str | None,
    allow_external: bool,
    llm_mode: str,
    console: Any,
    sqlite_db_fn: Callable[..., Any],
    resolve_routed_llm_fn: Callable[..., tuple[Any, Any, list[str]]],
    fallback_to_mini_llm_fn: Callable[..., tuple[Any, Any, list[str]]],
) -> None:
    config = khub.config
    sqlite_db = sqlite_db_fn(config, khub=khub)
    paper = sqlite_db.get_paper(arxiv_id)

    if not paper:
        console.print(f"[red]논문을 찾을 수 없습니다: {arxiv_id}[/red]")
        return

    text_path = paper.get("text_path")
    if not text_path:
        console.print("[red]텍스트 파일이 없습니다. khub paper download 먼저 실행하세요.[/red]")
        return

    console.print(f"번역 중: [bold]{paper['title'][:60]}[/bold]")

    try:
        text = Path(text_path).read_text(encoding="utf-8")
    except Exception as error:
        console.print(f"[red]텍스트 파일 읽기 실패: {error}[/red]")
        return

    output_dir = Path(config.papers_dir) / "translated"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{_safe_title(paper['title'])}_translated.md"

    chunk_size = 6000
    chunks = [text[index:index + chunk_size] for index in range(0, len(text), chunk_size)]

    llm, decision, warnings = resolve_routed_llm_fn(
        config,
        task_type="translation",
        allow_external=allow_external,
        llm_mode=llm_mode,
        query=paper["title"],
        context=text[:4000],
        source_count=1,
        provider_override=provider,
        model_override=model,
    )
    if llm is None:
        raise click.ClickException("사용 가능한 번역 LLM을 초기화하지 못했습니다.")
    console.print(f"[dim]프로바이더: {decision.provider}/{decision.model} route={decision.route}[/dim]")
    for warning in warnings:
        console.print(f"[yellow]{warning}[/yellow]")

    translated_parts = []
    active_llm = llm
    active_decision = decision
    for index, chunk in enumerate(chunks, 1):
        console.print(f"  [{index}/{len(chunks)}] 번역 중...")
        try:
            result = active_llm.translate(chunk, source_lang="en", target_lang="ko")
        except Exception as error:
            if active_decision.route == "local" and allow_external:
                fallback_llm, fallback_decision, fallback_warnings = fallback_to_mini_llm_fn(
                    config,
                    task_type="translation",
                    allow_external=allow_external,
                    query=paper["title"],
                    context=chunk[:4000],
                )
                for warning in fallback_warnings:
                    console.print(f"[yellow]{warning}[/yellow]")
                if fallback_llm is None:
                    raise click.ClickException(f"번역 실패(로컬/mini fallback 모두 실패): {error}") from error
                active_llm = fallback_llm
                active_decision = fallback_decision
                result = active_llm.translate(chunk, source_lang="en", target_lang="ko")
            else:
                raise click.ClickException(f"번역 실패: {error}") from error
        translated_parts.append(result)

    full_translation = "\n\n".join(translated_parts)
    header = (
        f"# {paper['title']}\n\n"
        f"> arXiv: {arxiv_id} | 번역: {decision.provider}/{decision.model}\n\n---\n\n"
    )
    output_path.write_text(header + full_translation, encoding="utf-8")

    sqlite_db.conn.execute(
        "UPDATE papers SET translated_path = ? WHERE arxiv_id = ?",
        (str(output_path), arxiv_id),
    )
    sqlite_db.conn.commit()
    console.print(f"[green]번역 완료: {output_path.name}[/green]")


def run_paper_summarize(
    *,
    khub: Any,
    arxiv_id: str,
    provider: str | None,
    model: str | None,
    quick: bool,
    allow_external: bool | None,
    llm_mode: str,
    console: Any,
    sqlite_db_fn: Callable[..., Any],
    structured_summary_service_factory: Callable[[Any, Any], Any],
    paper_summary_parser_fn: Callable[[Any], str],
    sync_structured_summary_view_fn: Callable[..., str],
) -> None:
    config = khub.config
    sqlite_db = sqlite_db_fn(config, khub=khub)
    paper = sqlite_db.get_paper(arxiv_id)

    if not paper:
        console.print(f"[red]논문을 찾을 수 없습니다: {arxiv_id}[/red]")
        return

    console.print(f"요약 중: [bold]{paper['title'][:60]}[/bold]")
    summary_service = structured_summary_service_factory(sqlite_db, config)
    summary_options = _resolve_summary_build_options(
        config,
        provider=provider,
        model=model,
        allow_external=allow_external,
        llm_mode=llm_mode,
    )
    console.print(
        f"[dim]parser={paper_summary_parser_fn(config)} "
        f"provider={summary_options['provider_override'] or summary_options['configured_provider'] or '(router)'} "
        f"model={summary_options['model_override'] or summary_options['configured_model'] or '(router)'} "
        f"external={bool(summary_options['allow_external'])} "
        f"mode={llm_mode}[/dim]"
    )
    with console.status("구조화 요약 생성 중..."):
        try:
            payload = summary_service.build(
                paper_id=arxiv_id,
                paper_parser=paper_summary_parser_fn(config),
                refresh_parse=False,
                quick=bool(quick),
                allow_external=bool(summary_options["allow_external"]),
                llm_mode=str(llm_mode or "auto"),
                provider_override=summary_options["provider_override"],
                model_override=summary_options["model_override"],
            )
        except Exception as error:
            raise click.ClickException(f"요약 실패: {error}") from error
    if str(payload.get("status") or "") == "blocked":
        detail = "; ".join(str(item) for item in list(payload.get("warnings") or [])[:3]) or f"paper summary blocked: {arxiv_id}"
        raise click.ClickException(detail)
    rendered_summary = sync_structured_summary_view_fn(sqlite_db, paper=paper, payload=payload, config=config)
    console.print(
        f"[dim]parser={payload.get('parserUsed') or 'raw'} "
        f"route={payload.get('llmRoute') or 'fallback-only'} "
        f"fallback={bool(payload.get('fallbackUsed'))}[/dim]"
    )
    for warning in list(payload.get("warnings") or [])[:5]:
        console.print(f"[yellow]{warning}[/yellow]")
    console.print(f"\n[bold]요약: {paper['title']}[/bold]\n")
    console.print(Markdown(rendered_summary))
    console.print("\n[green]구조화 요약 저장 완료[/green]")


def run_paper_embed(
    *,
    khub: Any,
    arxiv_id: str,
    console: Any,
    sqlite_db_fn: Callable[..., Any],
    build_embedder_fn: Callable[..., Any],
    vector_db_fn: Callable[..., Any],
) -> None:
    config = khub.config
    sqlite_db = sqlite_db_fn(config, khub=khub)
    paper = sqlite_db.get_paper(arxiv_id)

    if not paper:
        console.print(f"[red]논문을 찾을 수 없습니다: {arxiv_id}[/red]")
        return

    console.print(f"임베딩 중: [bold]{paper['title'][:60]}[/bold]")
    text = _build_paper_embedding_text(sqlite_db, paper=paper, config=config)

    try:
        embedder = build_embedder_fn(config, khub=khub)
        embedding = embedder.embed_text(text)
    except Exception as error:
        console.print(f"[red]임베딩 실패: {error}[/red]")
        return

    vector_db = vector_db_fn(config, khub=khub)
    vector_db.add_documents(
        documents=[text],
        embeddings=[embedding],
        metadatas=[{
            "title": paper["title"],
            "arxiv_id": arxiv_id,
            "source_type": "paper",
            "field": paper.get("field", ""),
            "chunk_index": 0,
        }],
        ids=[f"paper_{arxiv_id}_0"],
    )

    sqlite_db.conn.execute("UPDATE papers SET indexed = 1 WHERE arxiv_id = ?", (arxiv_id,))
    sqlite_db.conn.commit()
    console.print(f"[green]임베딩 완료 (벡터DB: {vector_db.count()}개 문서)[/green]")


def run_paper_translate_all(
    *,
    khub: Any,
    limit: int,
    field: str | None,
    provider: str | None,
    model: str | None,
    console: Any,
    sqlite_db_fn: Callable[..., Any],
    build_llm_fn: Callable[..., Any],
) -> None:
    config = khub.config
    sqlite_db = sqlite_db_fn(config, khub=khub)
    papers = sqlite_db.list_papers(field=field, limit=999)
    untranslated = [paper for paper in papers if not paper.get("translated_path") and paper.get("text_path")]

    if limit > 0:
        untranslated = untranslated[:limit]

    if not untranslated:
        console.print("[green]모든 논문이 이미 번역되었거나 텍스트 파일이 없습니다.[/green]")
        return

    prov = provider or config.translation_provider
    mdl = model or config.translation_model

    console.print(f"[bold]미번역 논문 {len(untranslated)}편 번역 시작[/bold]")
    console.print(f"[dim]프로바이더: {prov}/{mdl}[/dim]\n")

    llm = build_llm_fn(config, prov, mdl, khub=khub)

    output_dir = Path(config.papers_dir) / "translated"
    output_dir.mkdir(parents=True, exist_ok=True)
    success = 0
    failed: list[dict[str, str]] = []

    for index, paper in enumerate(untranslated, 1):
        arxiv_id = paper["arxiv_id"]
        title = paper["title"]
        console.print(f"[{index}/{len(untranslated)}] {title[:55]}...", end=" ")

        try:
            text = Path(paper["text_path"]).read_text(encoding="utf-8")
        except Exception as error:
            console.print(f"[red]읽기 실패: {error}[/red]")
            failed.append({"arxiv_id": arxiv_id, "error": f"파일 읽기: {error}"})
            continue

        chunk_size = 6000
        chunks = [text[offset:offset + chunk_size] for offset in range(0, len(text), chunk_size)]
        translated_parts = []
        chunk_failed = False
        for chunk_index, chunk in enumerate(chunks, 1):
            try:
                translated_parts.append(llm.translate(chunk, source_lang="en", target_lang="ko"))
            except Exception as error:
                console.print(f"[red]청크 {chunk_index} 실패[/red]")
                failed.append({"arxiv_id": arxiv_id, "error": f"청크 {chunk_index}: {error}"})
                chunk_failed = True
                break

        if chunk_failed:
            continue

        output_path = output_dir / f"{_safe_title(title)}_translated.md"
        header = f"# {title}\n\n> arXiv: {arxiv_id} | 번역: {prov}/{mdl}\n\n---\n\n"
        output_path.write_text(header + "\n\n".join(translated_parts), encoding="utf-8")

        sqlite_db.conn.execute(
            "UPDATE papers SET translated_path = ? WHERE arxiv_id = ?",
            (str(output_path), arxiv_id),
        )
        sqlite_db.conn.commit()
        success += 1
        console.print(f"[green]OK ({len(chunks)}청크)[/green]")

    console.print(f"\n[bold green]{success}/{len(untranslated)}편 번역 완료[/bold green]")
    if failed:
        console.print(f"[bold red]⚠ 실패: {len(failed)}편[/bold red]")
        for item in failed:
            console.print(f"  {item['arxiv_id']}: {item['error'][:80]}")


def run_paper_summarize_all(
    *,
    khub: Any,
    limit: int,
    field: str | None,
    quick: bool,
    resummary: bool,
    bad_only: bool,
    threshold: int,
    provider: str | None,
    model: str | None,
    allow_external: bool | None,
    llm_mode: str,
    paper_timeout_sec: int,
    checkpoint_file: Path | None,
    console: Any,
    sqlite_db_fn: Callable[..., Any],
    assess_summary_quality_fn: Callable[[str], dict[str, Any]],
    structured_summary_service_factory: Callable[[Any, Any], Any],
    paper_summary_parser_fn: Callable[[Any], str],
    render_structured_summary_notes_fn: Callable[[dict[str, Any]], str],
    sync_structured_summary_view_fn: Callable[..., str],
    build_public_summary_card_fn: Callable[..., dict[str, Any]],
    summary_batch_worker_fn: Callable[..., dict[str, Any]],
) -> None:
    config = khub.config
    sqlite_db = sqlite_db_fn(config, khub=khub)
    summary_service = structured_summary_service_factory(sqlite_db, config)
    summary_options = _resolve_summary_build_options(
        config,
        provider=provider,
        model=model,
        allow_external=allow_external,
        llm_mode=llm_mode,
    )
    papers = sqlite_db.list_papers(field=field, limit=999)
    artifact_cache: dict[str, dict[str, Any]] = {}

    def _artifact_for(paper_row: dict[str, Any]) -> dict[str, Any]:
        paper_id = str(paper_row.get("arxiv_id") or "").strip()
        if paper_id not in artifact_cache:
            artifact_cache[paper_id] = dict(summary_service.load_artifact(paper_id=paper_id) or {})
        return artifact_cache[paper_id]

    def _quality_source_text(paper_row: dict[str, Any]) -> str:
        payload = _artifact_for(paper_row)
        if payload:
            return render_structured_summary_notes_fn(payload)
        return str(paper_row.get("notes") or "")

    if bad_only:
        targets = [
            paper
            for paper in papers
            if _summary_quality_target(
                khub=khub,
                paper_row=paper,
                assess_summary_quality_fn=assess_summary_quality_fn,
                render_structured_summary_notes_fn=render_structured_summary_notes_fn,
                summary_payload=_artifact_for(paper),
                build_public_summary_card_fn=build_public_summary_card_fn,
            )["score"]
            < threshold
        ]
        console.print(f"[dim]품질 점수 {threshold}점 미만 논문 필터링[/dim]")
    elif resummary:
        targets = papers
    else:
        targets = [paper for paper in papers if not _artifact_for(paper)]

    if limit > 0:
        targets = targets[:limit]

    if not targets:
        console.print("[green]모든 논문이 이미 요약되어 있습니다.[/green]")
        return

    mode_label = "간단" if quick else "심층"
    if bad_only:
        mode_label += " (품질 미달 재요약)"
    elif resummary:
        mode_label += " (전체 재요약)"
    console.print(f"[bold]{len(targets)}편 {mode_label} 요약 시작[/bold]")
    console.print(
        f"[dim]parser={paper_summary_parser_fn(config)} "
        f"provider={summary_options['provider_override'] or summary_options['configured_provider'] or '(router)'} "
        f"model={summary_options['model_override'] or summary_options['configured_model'] or '(router)'} "
        f"external={bool(summary_options['allow_external'])} "
        f"mode={llm_mode} "
        f"paper-timeout={int(paper_timeout_sec or 0)}s[/dim]\n"
    )

    success = 0
    failed: list[dict[str, str]] = []
    checkpoint_target = Path(checkpoint_file).expanduser() if checkpoint_file is not None else _default_summary_batch_checkpoint_path(config)
    for index, paper in enumerate(targets, 1):
        arxiv_id = paper["arxiv_id"]
        title = paper["title"]
        console.print(f"  [{index}/{len(targets)}] {title[:50]}...", end=" ")
        started_at = time.time()

        try:
            payload = summary_batch_worker_fn(
                config=config,
                paper_id=arxiv_id,
                paper_parser=paper_summary_parser_fn(config),
                quick=bool(quick),
                provider=summary_options["provider_override"],
                model=summary_options["model_override"],
                allow_external=bool(summary_options["allow_external"]),
                llm_mode=str(llm_mode or "auto"),
                timeout_sec=int(paper_timeout_sec or 0),
            )
            if str(payload.get("status") or "") == "blocked":
                detail = "; ".join(str(item) for item in list(payload.get("warnings") or [])[:3]) or "blocked"
                raise RuntimeError(detail)
            sync_structured_summary_view_fn(sqlite_db, paper=paper, payload=payload, config=config)
            artifact_cache[str(arxiv_id)] = dict(payload)
            success += 1
            _append_summary_batch_checkpoint(
                checkpoint_target,
                {
                    "paperId": str(arxiv_id),
                    "title": str(title or ""),
                    "status": "ok",
                    "parserUsed": str(payload.get("parserUsed") or ""),
                    "llmRoute": str(payload.get("llmRoute") or ""),
                    "fallbackUsed": bool(payload.get("fallbackUsed")),
                    "durationSec": round(time.time() - started_at, 3),
                },
            )
            console.print(
                f"[green]OK[/green] [dim](route={payload.get('llmRoute') or 'fallback-only'} "
                f"fallback={bool(payload.get('fallbackUsed'))})[/dim]"
            )
        except TimeoutError as error:
            failed.append({"arxiv_id": arxiv_id, "error": str(error)})
            _append_summary_batch_checkpoint(
                checkpoint_target,
                {
                    "paperId": str(arxiv_id),
                    "title": str(title or ""),
                    "status": "timeout",
                    "error": str(error),
                    "durationSec": round(time.time() - started_at, 3),
                },
            )
            console.print(f"[red]TIMEOUT ({error})[/red]")
        except Exception as error:
            failed.append({"arxiv_id": arxiv_id, "error": str(error)})
            _append_summary_batch_checkpoint(
                checkpoint_target,
                {
                    "paperId": str(arxiv_id),
                    "title": str(title or ""),
                    "status": "failed",
                    "error": str(error),
                    "durationSec": round(time.time() - started_at, 3),
                },
            )
            console.print(f"[red]FAIL ({error})[/red]")

    console.print(f"\n[bold green]{success}/{len(targets)}편 요약 완료[/bold green]")
    console.print(f"[dim]checkpoint={checkpoint_target}[/dim]")
    if failed:
        console.print(f"[bold red]실패: {len(failed)}편[/bold red]")
        for item in failed:
            console.print(f"  {item['arxiv_id']}: {item['error'][:80]}")


def run_paper_embed_all(
    *,
    khub: Any,
    index_all: bool,
    console: Any,
    sqlite_db_fn: Callable[..., Any],
    build_embedder_fn: Callable[..., Any],
    vector_db_fn: Callable[..., Any],
) -> None:
    config = khub.config
    sqlite_db = sqlite_db_fn(config, khub=khub)
    papers = sqlite_db.list_papers(limit=999)
    unindexed = papers if index_all else [paper for paper in papers if not paper.get("indexed")]

    if not unindexed:
        console.print("[green]모든 논문이 이미 인덱싱되어 있습니다.[/green]")
        return

    console.print(f"[bold]인덱싱 시작: {len(unindexed)}편[/bold]")
    console.print(f"[dim]임베딩: {config.embedding_provider}/{config.embedding_model}[/dim]")

    embedder = build_embedder_fn(config, khub=khub)
    vector_db = vector_db_fn(config, khub=khub)
    batch_size = 20
    success = 0
    failed = 0
    started_at = time.time()

    for offset in range(0, len(unindexed), batch_size):
        batch = unindexed[offset:offset + batch_size]
        texts = [_build_paper_embedding_text(sqlite_db, paper=paper, config=config) for paper in batch]

        successful_items: list[tuple[dict[str, Any], str, list[float]]] = []
        batch_failures: list[tuple[str, str]] = []

        try:
            raw_embeddings = embedder.embed_batch(texts, show_progress=False)
            if len(raw_embeddings) != len(texts):
                raise RuntimeError(
                    f"embed_batch returned {len(raw_embeddings)} embeddings for {len(texts)} texts"
                )
            for paper, text, embedding in zip(batch, texts, raw_embeddings):
                if embedding is None:
                    batch_failures.append((paper["arxiv_id"], "embed_batch returned None"))
                else:
                    successful_items.append((paper, text, embedding))
        except Exception as error:
            batch_failures = [(paper["arxiv_id"], str(error)) for paper in batch]
            successful_items = []

        if batch_failures:
            recovered_items: list[tuple[dict[str, Any], str, list[float]]] = []
            recovered_failures: list[tuple[str, str]] = []
            failure_ids = {paper_id for paper_id, _ in batch_failures}
            paper_lookup = {paper["arxiv_id"]: (paper, text) for paper, text in zip(batch, texts)}
            for paper_id in failure_ids:
                paper, text = paper_lookup[paper_id]
                try:
                    embedding = embedder.embed_text(text)
                    recovered_items.append((paper, text, embedding))
                except Exception as error:
                    recovered_failures.append((paper_id, str(error)))
            successful_items.extend(recovered_items)
            batch_failures = recovered_failures

        if successful_items:
            documents, embeddings, metadatas, ids = [], [], [], []
            for paper, text, embedding in successful_items:
                documents.append(text)
                embeddings.append(embedding)
                metadatas.append({
                    "title": paper["title"] or "",
                    "arxiv_id": paper["arxiv_id"],
                    "source_type": "paper",
                    "field": paper.get("field", ""),
                    "chunk_index": 0,
                })
                ids.append(f"paper_{paper['arxiv_id']}_0")

            vector_db.add_documents(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )

            for paper, _, _ in successful_items:
                sqlite_db.conn.execute("UPDATE papers SET indexed = 1 WHERE arxiv_id = ?", (paper["arxiv_id"],))
            sqlite_db.conn.commit()
            success += len(successful_items)

        if batch_failures:
            failed += len(batch_failures)
            console.print(
                f"  [{success}/{len(unindexed)}] 배치: "
                f"[yellow]{len(successful_items)}편 OK[/yellow], "
                f"[red]{len(batch_failures)}편 실패[/red]"
            )
        else:
            console.print(f"  [{success}/{len(unindexed)}] 배치: [green]{len(successful_items)}편 OK[/green]")

    elapsed = time.time() - started_at
    summary_line = f"\n[bold green]{success}/{len(unindexed)}편 인덱싱 완료 ({elapsed:.1f}초)[/bold green]"
    if failed:
        summary_line += f" [bold red](실패 {failed}편)[/bold red]"
    console.print(summary_line)
