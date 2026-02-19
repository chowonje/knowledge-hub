"""
khub discover - 논문 자동 검색 → 다운로드 → 요약 → 인덱싱 → Obsidian 연결
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

console = Console()
log = logging.getLogger("khub.discover")

_sqlite_lock = Lock()


def _build_pipeline(khub_ctx, need_translator=False):
    """Config에서 프로바이더를 읽어 파이프라인 객체 구성 (ChromaDB 제외 - 인덱싱은 별도 프로세스)"""
    from knowledge_hub.core.database import SQLiteDatabase
    from knowledge_hub.providers.registry import get_llm

    config = khub_ctx.config

    summ_provider_cfg = config.get_provider_config(config.summarization_provider)
    summarizer = get_llm(
        config.summarization_provider,
        model=config.summarization_model,
        **summ_provider_cfg,
    )

    translator = None
    if need_translator:
        trans_provider_cfg = config.get_provider_config(config.translation_provider)
        translator = get_llm(
            config.translation_provider,
            model=config.translation_model,
            **trans_provider_cfg,
        )

    sqlite_db = SQLiteDatabase(config.sqlite_path)

    return config, summarizer, translator, sqlite_db


def _embed_via_requests(texts, config):
    """requests로 직접 임베딩 API 호출 (openai 라이브러리 우회)"""
    import os
    import requests

    if config.embedding_provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            api_key = config.get_provider_config("openai").get("api_key", "")
        resp = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": config.embedding_model, "input": texts},
            timeout=30,
        )
        resp.raise_for_status()
        return [x["embedding"] for x in sorted(resp.json()["data"], key=lambda x: x["index"])]
    else:
        base_url = config.get_provider_config(config.embedding_provider).get("base_url", "http://localhost:11434")
        embs = []
        for text in texts:
            resp = requests.post(
                f"{base_url}/api/embeddings",
                json={"model": config.embedding_model, "prompt": text},
                timeout=30,
            )
            resp.raise_for_status()
            embs.append(resp.json()["embedding"])
        return embs


def _index_papers_inline(papers_to_index, config):
    """배치 임베딩으로 벡터 인덱싱 (requests + ChromaDB 지연 로드)"""
    from knowledge_hub.core.database import VectorDatabase

    vector_db = VectorDatabase(config.vector_db_path, config.collection_name)
    results = {}

    texts, paper_refs = [], []
    for paper in papers_to_index:
        text_parts = [f"Title: {paper['title']}"]
        if paper.get("abstract"):
            text_parts.append(f"Abstract: {paper['abstract']}")
        if paper.get("summary"):
            text_parts.append(f"Summary: {paper['summary']}")
        texts.append("\n\n".join(text_parts))
        paper_refs.append(paper)

    if not texts:
        return results

    try:
        embeddings = _embed_via_requests(texts, config)

        docs, embs, metas, ids = [], [], [], []
        for paper, text, emb in zip(paper_refs, texts, embeddings):
            docs.append(text)
            embs.append(emb)
            metas.append({
                "title": paper["title"],
                "arxiv_id": paper["arxiv_id"],
                "source_type": "paper",
                "field": paper.get("field", ""),
                "chunk_index": 0,
            })
            ids.append(f"paper_{paper['arxiv_id']}_0")

        vector_db.add_documents(
            documents=docs, embeddings=embs, metadatas=metas, ids=ids,
        )
        for paper in paper_refs:
            results[paper["arxiv_id"]] = {"ok": True, "chunks": 1}
    except Exception as e:
        for paper in paper_refs:
            results[paper["arxiv_id"]] = {"ok": False, "chunks": 0, "error": str(e)}

    return results


def _download_one(paper, downloader):
    """단일 논문 PDF/텍스트 다운로드 (스레드용)"""
    try:
        result = downloader.download_single(paper.arxiv_id, paper.title)
        return paper, result, None
    except Exception as e:
        return paper, None, str(e)


def _show_candidate_table(new_papers, skipped):
    """검색 결과를 테이블로 보여주고 사용자 확인"""
    table = Table(title=f"발견된 새 논문 ({len(new_papers)}편, {skipped}편 중복 건너뜀)")
    table.add_column("#", width=3, style="dim")
    table.add_column("arXiv", style="cyan", width=13)
    table.add_column("제목", max_width=55)
    table.add_column("연도", width=5)
    table.add_column("인용", width=6, justify="right")
    table.add_column("분야", max_width=25, style="magenta")

    for i, p in enumerate(new_papers, 1):
        fields = ", ".join(p.fields_of_study[:2]) if p.fields_of_study else ""
        table.add_row(
            str(i), p.arxiv_id, p.title[:55],
            str(p.year), str(p.citation_count), fields,
        )

    console.print(table)


@click.command("discover")
@click.argument("topic")
@click.option("--max-papers", "-n", default=5, help="수집할 최대 논문 수")
@click.option("--year", "year_start", type=int, default=None, help="검색 시작 연도 (예: 2024)")
@click.option("--min-citations", type=int, default=0, help="최소 인용수 필터")
@click.option("--sort", "sort_by", type=click.Choice(["relevance", "citationCount"]), default="relevance")
@click.option("--translate/--no-translate", default=True, help="초록 한국어 번역 여부")
@click.option("--summarize/--no-summarize", "gen_summary", default=True, help="요약 생성 여부")
@click.option("--obsidian/--no-obsidian", "create_obsidian", default=None, help="Obsidian 노트 생성")
@click.option("--yes", "-y", "auto_confirm", is_flag=True, help="확인 없이 바로 진행")
@click.option("--workers", "-w", default=4, help="병렬 다운로드 워커 수 (기본: 4)")
@click.option("--index/--no-index", "do_index", default=True, help="벡터DB 인덱싱 여부")
@click.pass_context
def discover(ctx, topic, max_papers, year_start, min_citations, sort_by,
             translate, gen_summary, create_obsidian, auto_confirm, workers, do_index):
    """논문 자동 검색 → 다운로드 → 요약 → 인덱싱 → Obsidian 연결

    \b
    예시:
      khub discover "large language model agent" -n 5 --year 2024
      khub discover "RAG" -n 10 --min-citations 50 --sort citationCount
      khub discover "AI agent" -n 3 -y          # 확인 없이 바로 진행
      khub discover "transformer" -w 8           # 8개 병렬 다운로드
    """
    console.print(Panel.fit(
        f"[bold]논문 탐색: {topic}[/bold]\n"
        f"최대 {max_papers}편 | 연도: {year_start or '전체'} | 정렬: {sort_by} | 워커: {workers}",
        border_style="cyan",
    ))

    try:
        config, summarizer, translator, sqlite_db = _build_pipeline(
            ctx.obj["khub"], need_translator=translate,
        )
    except Exception as e:
        console.print(f"[red]초기화 실패: {e}[/red]")
        console.print("[dim]khub init을 먼저 실행하세요.[/dim]")
        return

    if create_obsidian is None:
        create_obsidian = config.vault_enabled

    from knowledge_hub.papers.discoverer import discover_papers
    from knowledge_hub.papers.downloader import PaperDownloader

    # ── Phase 1: Search ──
    try:
        with console.status("[cyan]Semantic Scholar + arXiv 검색 중...[/cyan]"):
            discovered = discover_papers(
                topic=topic,
                max_papers=max_papers * 3,
                year_start=year_start,
                min_citations=min_citations,
                sort_by=sort_by,
            )
    except Exception as e:
        console.print(f"[red]논문 검색 실패: {e}[/red]")
        log.exception("검색 API 오류")
        return

    if not discovered:
        console.print("[yellow]검색 결과가 없습니다.[/yellow]")
        return

    console.print(f"[green]{len(discovered)}개 논문 발견[/green]")

    # ── Phase 2: Deduplicate (SQLite) ──
    new_papers = []
    skipped = 0
    for paper in discovered:
        if sqlite_db.get_paper(paper.arxiv_id):
            skipped += 1
            continue
        new_papers.append(paper)
        if len(new_papers) >= max_papers:
            break

    if not new_papers:
        console.print(f"[yellow]새로운 논문이 없습니다. {skipped}편 모두 이미 수집되었습니다.[/yellow]")
        return

    # ── Phase 3: 확인 프롬프트 ──
    _show_candidate_table(new_papers, skipped)

    if not auto_confirm:
        action = click.prompt(
            "\n  진행할까요? [Y]전체 / [n]취소 / 번호 선택(예: 1,3,5)",
            default="Y",
        )
        if action.strip().lower() == "n":
            console.print("[dim]취소됨[/dim]")
            return
        if action.strip().upper() != "Y" and action.strip() != "":
            try:
                indices = [int(x.strip()) - 1 for x in action.split(",")]
                new_papers = [new_papers[i] for i in indices if 0 <= i < len(new_papers)]
                console.print(f"[cyan]{len(new_papers)}편 선택됨[/cyan]")
            except (ValueError, IndexError):
                console.print("[red]잘못된 입력. 취소합니다.[/red]")
                return

    if not new_papers:
        return

    # ── Phase 4: 병렬 다운로드 ──
    downloader = PaperDownloader(config.papers_dir)
    dl_results = {}
    dl_errors = {}

    console.print(f"\n[bold]다운로드 시작 ({len(new_papers)}편, {workers}개 병렬)[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_dl = progress.add_task("다운로드 중...", total=len(new_papers))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_download_one, p, downloader): p for p in new_papers}

            for future in as_completed(futures):
                paper, result, error = future.result()
                if error:
                    dl_errors[paper.arxiv_id] = error
                else:
                    dl_results[paper.arxiv_id] = result
                progress.advance(task_dl)
                progress.update(task_dl, description=f"다운로드: {paper.arxiv_id}")

    console.print(f"  [green]{len(dl_results)}편 다운로드 완료[/green]", end="")
    if dl_errors:
        console.print(f", [red]{len(dl_errors)}편 실패[/red]")
    else:
        console.print()

    # ── Phase 5: 요약 + 번역 + Obsidian ──
    results = []
    papers_to_index = []

    for idx, paper in enumerate(new_papers, 1):
        console.print(f"\n[bold cyan]── [{idx}/{len(new_papers)}] {paper.title[:60]} ──[/bold cyan]")
        entry = {"arxiv_id": paper.arxiv_id, "title": paper.title, "steps": []}

        dl = dl_results.get(paper.arxiv_id)
        if not dl:
            entry["success"] = False
            entry["error"] = dl_errors.get(paper.arxiv_id, "다운로드 실패")
            results.append(entry)
            console.print(f"  [red]다운로드 실패[/red]")
            continue

        try:
            entry["steps"].append("PDF" if dl.get("pdf") else "텍스트")
            console.print(f"  [dim]PDF 저장됨[/dim]")

            importance = 5 if paper.citation_count >= 500 else 4 if paper.citation_count >= 100 else 3 if paper.citation_count >= 20 else 2 if paper.citation_count >= 5 else 1
            paper_data = {
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "authors": paper.authors,
                "year": paper.year,
                "field": ", ".join(paper.fields_of_study[:3]),
                "importance": importance,
                "notes": f"citations: {paper.citation_count}",
                "pdf_path": dl.get("pdf"),
                "text_path": dl.get("text"),
                "translated_path": None,
            }
            with _sqlite_lock:
                sqlite_db.upsert_paper(paper_data)

            # Summarize — PDF 전문이 있으면 심층 요약, 없으면 abstract 기반
            summary = ""
            if gen_summary and (paper.abstract or dl.get("text")):
                console.print("  요약 중...", end=" ")
                try:
                    text_file = dl.get("text")
                    if text_file and Path(text_file).exists():
                        full_text = Path(text_file).read_text(encoding="utf-8")[:30000]
                        source_label = "전문"
                    else:
                        full_text = f"제목: {paper.title}\n초록: {paper.abstract}"
                        source_label = "abstract"

                    summary = summarizer.summarize_paper(
                        full_text, title=paper.title, language="ko",
                    )
                    console.print(f"[green]완료 ({source_label})[/green]")
                    entry["steps"].append(f"요약({source_label})")
                except Exception as e:
                    try:
                        summary = summarizer.summarize(
                            f"제목: {paper.title}\n초록: {paper.abstract}",
                            language="ko", max_sentences=5,
                        )
                        console.print(f"[yellow]간단 요약 ({e})[/yellow]")
                        entry["steps"].append("요약(간단)")
                    except Exception:
                        summary = paper.abstract[:500] if paper.abstract else ""
                        console.print(f"[yellow]fallback[/yellow]")
                        entry["steps"].append("요약(fallback)")

            # Translate abstract
            translated_abstract = ""
            if translate and translator and paper.abstract:
                console.print("  번역 중...", end=" ")
                try:
                    translated_abstract = translator.translate(
                        paper.abstract, source_lang="en", target_lang="ko",
                    )
                    console.print("[green]완료[/green]")
                    entry["steps"].append("번역")
                except Exception as e:
                    console.print(f"[yellow]실패 ({e})[/yellow]")
                    entry["steps"].append("번역실패")

            if summary:
                with _sqlite_lock:
                    sqlite_db.conn.execute(
                        "UPDATE papers SET notes = ? WHERE arxiv_id = ?",
                        (summary, paper.arxiv_id),
                    )
                    sqlite_db.conn.commit()

            if do_index:
                papers_to_index.append({
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "abstract": paper.abstract or "",
                    "summary": summary,
                    "field": paper_data["field"],
                })

            # Obsidian note
            if create_obsidian and config.vault_path:
                console.print("  Obsidian 노트...", end=" ")
                vault = Path(config.vault_path)
                papers_dir_path = vault / "Papers"
                papers_dir_path.mkdir(parents=True, exist_ok=True)
                safe_title = re.sub(r'[\\/:*?"<>|]', '', paper.title).strip()
                safe_title = re.sub(r'\s+', ' ', safe_title)[:100].strip()
                note_path = papers_dir_path / f"{safe_title}.md"

                fields_tags = ", ".join(paper.fields_of_study[:3])
                lines = [
                    "---",
                    f'title: "{paper.title}"',
                    f'arxiv_id: "{paper.arxiv_id}"',
                    f'authors: "{paper.authors}"',
                    f"year: {paper.year}",
                    f"citations: {paper.citation_count}",
                    f'tags: [paper, {fields_tags}]',
                    "type: paper-summary",
                    "---",
                    "",
                    f"# {paper.title}",
                    "",
                    f"**arXiv:** [{paper.arxiv_id}](https://arxiv.org/abs/{paper.arxiv_id})",
                    f"**저자:** {paper.authors}",
                    f"**연도:** {paper.year} | **인용수:** {paper.citation_count}",
                    "",
                ]
                if summary:
                    lines.extend(["## 요약", "", summary, ""])
                if translated_abstract:
                    lines.extend(["## 초록 (한국어)", "", translated_abstract, ""])
                if paper.abstract:
                    lines.extend(["## Abstract", "", paper.abstract, ""])
                lines.extend([
                    "",
                    "---",
                    "",
                    "## 관련 개념",
                    "- [[00_Concept_Index]]",
                    "",
                ])

                note_path.write_text("\n".join(lines), encoding="utf-8")
                console.print("[green]완료[/green]")
                entry["steps"].append("Obsidian")

            entry["summary"] = summary[:200] if summary else ""
            entry["success"] = True
            console.print(f"  [bold green]✓ 완료[/bold green]")

        except Exception as e:
            entry["success"] = False
            entry["error"] = str(e)
            console.print(f"  [red]오류: {e}[/red]")

        results.append(entry)

    # ── Phase 6: 벡터 인덱싱 ──
    if do_index and papers_to_index:
        console.print(f"\n[bold]벡터 인덱싱 ({len(papers_to_index)}편)...[/bold]", end=" ")
        try:
            idx_results = _index_papers_inline(papers_to_index, config)
            indexed_count = sum(1 for v in idx_results.values() if v.get("ok"))
            failed_count = len(papers_to_index) - indexed_count
            console.print(f"[green]{indexed_count}편 완료[/green]", end="")
            if failed_count > 0:
                console.print(f", [red]{failed_count}편 실패[/red]")
            else:
                console.print()
            for r in results:
                idx_r = idx_results.get(r["arxiv_id"])
                if idx_r and idx_r.get("ok"):
                    r["steps"].append(f"인덱싱({idx_r['chunks']})")
                    sqlite_db.conn.execute(
                        "UPDATE papers SET indexed = 1 WHERE arxiv_id = ?", (r["arxiv_id"],)
                    )
                elif idx_r:
                    r["steps"].append("인덱싱실패")
                    log.warning("인덱싱 실패 %s: %s", r["arxiv_id"], idx_r.get("error", ""))
            sqlite_db.conn.commit()
        except Exception as e:
            log.exception("벡터 인덱싱 전체 실패")
            console.print(f"[yellow]실패: {e}[/yellow]")

    # ── 결과 요약 ──
    console.print()
    table = Table(title=f"수집 결과: {topic}")
    table.add_column("#", width=3)
    table.add_column("arXiv", style="cyan", width=13)
    table.add_column("논문", max_width=45)
    table.add_column("파이프라인", style="green")
    table.add_column("상태")

    for i, r in enumerate(results, 1):
        status = "[green]완료[/green]" if r.get("success") else f"[red]{r.get('error', '실패')[:25]}[/red]"
        table.add_row(
            str(i), r["arxiv_id"], r["title"][:45],
            " → ".join(r.get("steps", [])), status,
        )

    console.print(table)

    success_count = sum(1 for r in results if r.get("success"))
    console.print(f"\n[bold green]{success_count}/{len(results)}편 수집 완료[/bold green], {skipped}편 중복 건너뜀")

    if success_count > 0:
        if not do_index:
            console.print("[dim]khub index 로 벡터 인덱싱 | khub paper list 로 목록 확인[/dim]")
        else:
            console.print("[dim]khub paper list 로 목록 확인 | khub search \"키워드\" 로 검색[/dim]")
