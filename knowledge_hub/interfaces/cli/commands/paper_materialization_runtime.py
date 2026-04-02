"""Paper CLI runtime helpers for paper materialization flows."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable

import click
from rich.markdown import Markdown


def _safe_title(title: str) -> str:
    normalized = re.sub(r'[\\/:*?"<>|]', "", title).strip()
    return re.sub(r"\s+", " ", normalized)[:100].strip()


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
        paper_data = {
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": existing.get("authors", "") if existing else "",
            "year": existing.get("year", 0) if existing else 0,
            "field": existing.get("field", "") if existing else "",
            "importance": existing.get("importance", 3) if existing else 3,
            "notes": existing.get("notes", "") if existing else "",
            "pdf_path": result.get("pdf"),
            "text_path": result.get("text"),
            "translated_path": existing.get("translated_path") if existing else None,
        }
        sqlite_db.upsert_paper(paper_data)
        console.print(f"[green]다운로드 완료: {result.get('pdf', 'N/A')}[/green]")
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
    allow_external: bool,
    llm_mode: str,
    console: Any,
    sqlite_db_fn: Callable[..., Any],
    collect_paper_text_fn: Callable[[dict[str, Any], Any], str],
    resolve_routed_llm_fn: Callable[..., tuple[Any, Any, list[str]]],
    fallback_to_mini_llm_fn: Callable[..., tuple[Any, Any, list[str]]],
    update_obsidian_summary_fn: Callable[[dict[str, Any], str, Any], None],
) -> None:
    config = khub.config
    sqlite_db = sqlite_db_fn(config, khub=khub)
    paper = sqlite_db.get_paper(arxiv_id)

    if not paper:
        console.print(f"[red]논문을 찾을 수 없습니다: {arxiv_id}[/red]")
        return

    console.print(f"요약 중: [bold]{paper['title'][:60]}[/bold]")

    text = collect_paper_text_fn(paper, config)
    source_label = "전문" if len(text) > 2000 else "abstract"
    console.print(f"[dim]입력 소스: {source_label} ({len(text):,}자)[/dim]")

    llm, decision, warnings = resolve_routed_llm_fn(
        config,
        task_type="materialization_summary" if quick else "rag_answer",
        allow_external=allow_external,
        llm_mode=llm_mode,
        query=paper["title"],
        context=text[:8000],
        source_count=1,
        provider_override=provider,
        model_override=model,
    )
    if llm is None:
        raise click.ClickException("사용 가능한 요약 LLM을 초기화하지 못했습니다.")
    console.print(f"[dim]프로바이더: {decision.provider}/{decision.model} route={decision.route}[/dim]")
    for warning in warnings:
        console.print(f"[yellow]{warning}[/yellow]")

    with console.status("심층 요약 생성 중..."):
        try:
            if quick:
                summary = llm.summarize(text, language="ko", max_sentences=5)
            else:
                summary = llm.summarize_paper(text, title=paper["title"], language="ko")
        except Exception as error:
            if decision.route == "local" and allow_external:
                fallback_llm, fallback_decision, fallback_warnings = fallback_to_mini_llm_fn(
                    config,
                    task_type="materialization_summary" if quick else "rag_answer",
                    allow_external=allow_external,
                    query=paper["title"],
                    context=text[:8000],
                )
                for warning in fallback_warnings:
                    console.print(f"[yellow]{warning}[/yellow]")
                if fallback_llm is None:
                    raise click.ClickException(f"요약 실패(로컬/mini fallback 모두 실패): {error}") from error
                decision = fallback_decision
                if quick:
                    summary = fallback_llm.summarize(text, language="ko", max_sentences=5)
                else:
                    summary = fallback_llm.summarize_paper(text, title=paper["title"], language="ko")
            else:
                raise click.ClickException(f"요약 실패: {error}") from error

    console.print(f"\n[bold]요약: {paper['title']}[/bold]\n")
    console.print(Markdown(summary))

    sqlite_db.conn.execute(
        "UPDATE papers SET notes = ? WHERE arxiv_id = ?",
        (summary, arxiv_id),
    )
    sqlite_db.conn.commit()

    update_obsidian_summary_fn(paper, summary, config)
    console.print("\n[green]요약 저장 완료[/green]")


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

    text = f"Title: {paper['title']}"
    if paper.get("notes"):
        text += f"\n\n{paper['notes']}"

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
    console: Any,
    sqlite_db_fn: Callable[..., Any],
    build_llm_fn: Callable[..., Any],
    collect_paper_text_fn: Callable[[dict[str, Any], Any], str],
    assess_summary_quality_fn: Callable[[str], dict[str, Any]],
    update_obsidian_summary_fn: Callable[[dict[str, Any], str, Any], None],
    requests_post_fn: Callable[..., Any],
    log: Any,
) -> None:
    config = khub.config
    sqlite_db = sqlite_db_fn(config, khub=khub)
    papers = sqlite_db.list_papers(field=field, limit=999)

    if bad_only:
        targets = [paper for paper in papers if assess_summary_quality_fn(paper.get("notes", ""))["score"] < threshold]
        console.print(f"[dim]품질 점수 {threshold}점 미만 논문 필터링[/dim]")
    elif resummary:
        targets = papers
    else:
        targets = [paper for paper in papers if not paper.get("notes") or len(paper.get("notes", "")) < 100]

    if limit > 0:
        targets = targets[:limit]

    if not targets:
        console.print("[green]모든 논문이 이미 요약되어 있습니다.[/green]")
        return

    prov = provider or config.summarization_provider
    mdl = model or config.summarization_model
    llm = build_llm_fn(config, prov, mdl, khub=khub)

    mode_label = "간단" if quick else "심층"
    if bad_only:
        mode_label += " (품질 미달 재요약)"
    elif resummary:
        mode_label += " (전체 재요약)"
    console.print(f"[bold]{len(targets)}편 {mode_label} 요약 시작[/bold]")
    console.print(f"[dim]프로바이더: {prov}/{mdl}[/dim]\n")

    missing_abstract = [
        paper
        for paper in targets
        if not collect_paper_text_fn(paper, config) or len(collect_paper_text_fn(paper, config)) < 100
    ]
    abstract_map: dict[str, str] = {}
    if missing_abstract:
        arxiv_ids = [paper["arxiv_id"] for paper in missing_abstract]
        for offset in range(0, len(arxiv_ids), 50):
            chunk = arxiv_ids[offset:offset + 50]
            try:
                response = requests_post_fn(
                    "https://api.semanticscholar.org/graph/v1/paper/batch",
                    params={"fields": "title,abstract,externalIds"},
                    json={"ids": [f"ArXiv:{arxiv_id}" for arxiv_id in chunk]},
                    timeout=60,
                )
                if response.status_code == 200:
                    for paper_data in response.json():
                        if paper_data and paper_data.get("abstract"):
                            external_ids = paper_data.get("externalIds", {})
                            arxiv_id = external_ids.get("ArXiv", "")
                            if arxiv_id:
                                abstract_map[arxiv_id] = paper_data["abstract"]
            except Exception as error:
                log.warning("Semantic Scholar abstract backfill batch failed: %s", error)
        console.print(f"[dim]Semantic Scholar에서 {len(abstract_map)}편 abstract 보충[/dim]\n")

    success = 0
    failed: list[dict[str, str]] = []
    for index, paper in enumerate(targets, 1):
        arxiv_id = paper["arxiv_id"]
        title = paper["title"]

        text = collect_paper_text_fn(paper, config)
        if len(text) < 100:
            extra = abstract_map.get(arxiv_id, "")
            if extra:
                text = f"제목: {title}\n초록: {extra}"

        if len(text) < 50:
            console.print(f"  [{index}/{len(targets)}] {arxiv_id} - 텍스트 부족, 스킵")
            continue

        source = "전문" if len(text) > 2000 else "abstract"
        console.print(f"  [{index}/{len(targets)}] {title[:50]}... ({source})", end=" ")

        try:
            if quick:
                summary = llm.summarize(text, language="ko", max_sentences=5)
            else:
                summary = llm.summarize_paper(text, title=title, language="ko")

            sqlite_db.conn.execute(
                "UPDATE papers SET notes = ? WHERE arxiv_id = ?",
                (summary, arxiv_id),
            )
            sqlite_db.conn.commit()

            update_obsidian_summary_fn(paper, summary, config)
            success += 1
            console.print("[green]OK[/green]")
        except Exception as error:
            log.error("요약 실패 %s: %s", arxiv_id, error)
            failed.append({"arxiv_id": arxiv_id, "error": str(error)})
            console.print(f"[red]FAIL ({error})[/red]")

    console.print(f"\n[bold green]{success}/{len(targets)}편 요약 완료[/bold green]")
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
    started_at = time.time()

    for offset in range(0, len(unindexed), batch_size):
        batch = unindexed[offset:offset + batch_size]
        texts = []
        for paper in batch:
            text = f"Title: {paper['title'] or paper['arxiv_id']}"
            if paper.get("notes"):
                text += f"\n\n{paper['notes']}"
            texts.append(text)

        try:
            raw_embeddings = embedder.embed_batch(texts, show_progress=False)
            embeddings = [embedding for embedding in raw_embeddings if embedding is not None]
            if len(embeddings) != len(texts):
                raise RuntimeError(f"{len(texts) - len(embeddings)}개 텍스트 임베딩 실패")

            documents, metadatas, ids = [], [], []
            for paper, text, embedding in zip(batch, texts, embeddings):
                documents.append(text)
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

            for paper in batch:
                sqlite_db.conn.execute("UPDATE papers SET indexed = 1 WHERE arxiv_id = ?", (paper["arxiv_id"],))
            sqlite_db.conn.commit()

            success += len(batch)
            console.print(f"  [{success}/{len(unindexed)}] 배치: [green]{len(batch)}편 OK[/green]")
        except Exception as error:
            console.print(f"  배치 실패: [red]{error}[/red]")

    elapsed = time.time() - started_at
    console.print(f"\n[bold green]{success}/{len(unindexed)}편 인덱싱 완료 ({elapsed:.1f}초)[/bold green]")
