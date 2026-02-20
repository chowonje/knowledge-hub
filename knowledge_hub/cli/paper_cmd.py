"""
khub paper - 논문 개별 관리 명령어

개별 작업:
  khub paper add <URL>           URL로 논문 추가 (arXiv, OpenReview, HuggingFace 등)
  khub paper download <ID>       단일 다운로드
  khub paper translate <ID>      단일 번역
  khub paper summarize <ID>      단일 요약
  khub paper embed <ID>          단일 임베딩
  khub paper info <ID>           상세 정보

배치 작업:
  khub paper translate-all       미번역 전체 번역
  khub paper summarize-all       미요약 전체 요약
  khub paper embed-all           미인덱싱 전체 임베딩
  khub paper list                목록 조회
"""

from __future__ import annotations

import logging
import os
import re
import time

import click
import requests
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()
log = logging.getLogger("khub.paper")

_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")

API_MAX_RETRIES = 3
API_RETRY_BASE_SEC = 2.0


def _validate_arxiv_id(arxiv_id: str) -> str:
    """arXiv ID 형식 검증. 유효하면 반환, 아니면 ClickException."""
    arxiv_id = arxiv_id.strip()
    if not _ARXIV_ID_RE.match(arxiv_id):
        raise click.BadParameter(
            f"유효하지 않은 arXiv ID: '{arxiv_id}' (예: 2501.06322)",
            param_hint="arxiv_id",
        )
    return arxiv_id


def _api_call_with_retry(fn, *args, **kwargs):
    """API 호출을 재시도하는 범용 래퍼. fn은 requests 호출을 수행해야 함."""
    last_err: Exception | None = None
    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except requests.HTTPError as e:
            last_err = e
            status = getattr(e.response, "status_code", 0)
            if status == 429 or status >= 500:
                wait = API_RETRY_BASE_SEC * (2 ** (attempt - 1))
                log.warning("API %d 에러, %d/%d 재시도 (%.1fs 대기)",
                            status, attempt, API_MAX_RETRIES, wait)
                time.sleep(wait)
                continue
            raise
        except (requests.ConnectionError, requests.Timeout) as e:
            last_err = e
            wait = API_RETRY_BASE_SEC * (2 ** (attempt - 1))
            log.warning("네트워크 오류, %d/%d 재시도 (%.1fs 대기)",
                        attempt, API_MAX_RETRIES, wait)
            time.sleep(wait)
    raise last_err  # type: ignore[misc]


MAX_SUMMARIZE_CHARS = 30000


def _resolve_vault_papers_dir(vault_path: str) -> Path | None:
    """Obsidian vault 내 논문 폴더를 동적으로 탐색"""
    candidates = [
        Path(vault_path) / "Papers",
        Path(vault_path) / "Projects" / "AI" / "AI_Papers",
        Path(vault_path) / "papers",
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path(vault_path) / "Papers"


def _resolve_vault_concepts_dir(vault_path: str) -> Path:
    """Obsidian vault 내 개념 폴더를 동적으로 탐색"""
    papers_dir = _resolve_vault_papers_dir(vault_path)
    if papers_dir:
        concepts = papers_dir / "Concepts"
        if concepts.exists():
            return concepts
    candidates = [
        Path(vault_path) / "Papers" / "Concepts",
        Path(vault_path) / "Projects" / "AI" / "AI_Papers" / "Concepts",
        Path(vault_path) / "Concepts",
    ]
    for c in candidates:
        if c.exists():
            return c
    return (papers_dir or Path(vault_path) / "Papers") / "Concepts"


def _collect_paper_text(paper: dict, config) -> str:
    """논문 전문 텍스트 수집: 번역본 → 원문 텍스트 → abstract 순으로 fallback"""
    translated = paper.get("translated_path")
    if translated and Path(translated).exists():
        text = Path(translated).read_text(encoding="utf-8")
        if len(text) > 200:
            return text[:MAX_SUMMARIZE_CHARS]

    text_path = paper.get("text_path")
    if text_path and Path(text_path).exists():
        text = Path(text_path).read_text(encoding="utf-8")
        if len(text) > 200:
            return text[:MAX_SUMMARIZE_CHARS]

    papers_dir = Path(config.papers_dir)
    for pattern in [f"*{paper['arxiv_id']}*.txt", f"*{paper['title'][:30]}*.txt"]:
        for p in papers_dir.glob(pattern):
            text = p.read_text(encoding="utf-8")
            if len(text) > 200:
                return text[:MAX_SUMMARIZE_CHARS]

    title = paper.get("title", "")
    authors = paper.get("authors", "")
    field = paper.get("field", "")
    notes = paper.get("notes", "")
    return f"제목: {title}\n저자: {authors}\n분야: {field}\n{notes}"


def _update_obsidian_summary(paper: dict, summary: str, config):
    """Obsidian 노트가 있으면 요약 섹션을 업데이트"""
    if not config.vault_path:
        return
    vault = Path(config.vault_path)
    safe_title = re.sub(r'[\\/:*?"<>|]', '', paper["title"]).strip()
    safe_title = re.sub(r'\s+', ' ', safe_title)[:100].strip()

    papers_dir = _resolve_vault_papers_dir(str(vault))
    if papers_dir:
        note_path = papers_dir / f"{safe_title}.md"
        if not note_path.exists():
            return
        content = note_path.read_text(encoding="utf-8")
        placeholder = "요약본/번역본이 아직 등록되지 않았습니다"
        old_summary_section = None

        if placeholder in content:
            content = content.replace(placeholder, summary)
            note_path.write_text(content, encoding="utf-8")
            console.print(f"[dim]Obsidian 노트 업데이트: {note_path.name}[/dim]")
            return

        if "## 요약" in content:
            lines = content.split("\n")
            start = None
            end = None
            for i, line in enumerate(lines):
                if line.strip() == "## 요약":
                    start = i
                elif start is not None and line.startswith("## ") and i > start:
                    end = i
                    break
            if start is not None:
                if end is None:
                    end = len(lines)
                new_lines = lines[:start] + ["## 요약", "", summary, ""] + lines[end:]
                note_path.write_text("\n".join(new_lines), encoding="utf-8")
                console.print(f"[dim]Obsidian 노트 업데이트: {note_path.name}[/dim]")
                return


@click.group("paper")
def paper_group():
    """논문 관리 (add/download/translate/summarize/embed/list/info)"""
    pass


# ─────────────────────────────────────────────
# paper list
# ─────────────────────────────────────────────
@paper_group.command("list")
@click.option("--field", "-f", default=None, help="분야 필터")
@click.option("--limit", "-n", default=50, help="표시할 최대 수")
@click.pass_context
def paper_list(ctx, field, limit):
    """수집된 논문 목록"""
    config = ctx.obj["khub"].config
    from knowledge_hub.core.database import SQLiteDatabase

    sqlite_db = SQLiteDatabase(config.sqlite_path)
    papers = sqlite_db.list_papers(field=field, limit=limit)

    if not papers:
        console.print("[yellow]수집된 논문이 없습니다. khub discover로 시작하세요.[/yellow]")
        return

    table = Table(title=f"논문 목록 ({len(papers)}개)")
    table.add_column("arXiv ID", style="cyan", width=14)
    table.add_column("제목", max_width=50)
    table.add_column("연도", width=5)
    table.add_column("분야", style="magenta", max_width=20)
    table.add_column("PDF", width=4)
    table.add_column("요약", width=4)
    table.add_column("번역", width=4)
    table.add_column("벡터", width=4)

    for p in papers:
        notes = p.get("notes") or ""
        has_summary = len(notes) > 30
        table.add_row(
            p["arxiv_id"],
            p["title"][:50],
            str(p.get("year", "")),
            p.get("field", "")[:20],
            "[green]O[/green]" if p.get("pdf_path") else "-",
            "[green]O[/green]" if has_summary else "-",
            "[green]O[/green]" if p.get("translated_path") else "-",
            "[green]O[/green]" if p.get("indexed") else "-",
        )

    console.print(table)


# ─────────────────────────────────────────────
# paper add <URL>
# ─────────────────────────────────────────────
@paper_group.command("add")
@click.argument("url")
@click.option("--download/--no-download", default=True, help="PDF 다운로드 여부")
@click.pass_context
def paper_add(ctx, url, download):
    """URL로 논문 추가 (arXiv, OpenReview, PapersWithCode, HuggingFace, S2, PDF URL)"""
    config = ctx.obj["khub"].config
    from knowledge_hub.papers.url_resolver import resolve_url
    from knowledge_hub.papers.downloader import PaperDownloader
    from knowledge_hub.core.database import SQLiteDatabase

    with console.status(f"[cyan]URL 분석 중: {url[:60]}...[/cyan]"):
        paper = resolve_url(url)

    if not paper:
        console.print("[red]논문을 찾을 수 없습니다.[/red]")
        return

    console.print(f"[bold]{paper.title}[/bold]")
    console.print(f"  저자: {paper.authors}")
    console.print(f"  연도: {paper.year} | 인용: {paper.citation_count} | 소스: {paper.source}")
    if paper.abstract:
        console.print(f"  초록: {paper.abstract[:120]}...")

    sqlite_db = SQLiteDatabase(config.sqlite_path)
    existing = sqlite_db.get_paper(paper.arxiv_id) if paper.arxiv_id else None
    if existing:
        console.print(f"[yellow]이미 등록된 논문입니다: {paper.arxiv_id}[/yellow]")
        return

    paper_data = {
        "arxiv_id": paper.arxiv_id or re.sub(r'[^\w]', '_', paper.title)[:30],
        "title": paper.title,
        "authors": paper.authors,
        "year": paper.year,
        "field": ", ".join(paper.fields_of_study[:3]),
        "importance": 3,
        "notes": f"citations: {paper.citation_count}",
        "pdf_path": None,
        "text_path": None,
        "translated_path": None,
    }

    if download:
        downloader = PaperDownloader(config.papers_dir)
        with console.status("다운로드 중..."):
            result = downloader.download_single(paper.arxiv_id, paper.title)
        paper_data["pdf_path"] = result.get("pdf")
        paper_data["text_path"] = result.get("text")
        if result["success"]:
            console.print(f"  [green]PDF 다운로드 완료[/green]")
        else:
            console.print(f"  [yellow]PDF 다운로드 실패[/yellow]")

    sqlite_db.upsert_paper(paper_data)
    console.print(f"[green]논문 등록 완료: {paper_data['arxiv_id']}[/green]")
    console.print("[dim]khub paper summarize / translate / embed 로 후속 작업 가능[/dim]")


# ─────────────────────────────────────────────
# paper download <arxiv_id>
# ─────────────────────────────────────────────
@paper_group.command("download")
@click.argument("arxiv_id")
@click.pass_context
def paper_download(ctx, arxiv_id):
    """단일 논문 PDF/텍스트 다운로드"""
    arxiv_id = _validate_arxiv_id(arxiv_id)
    config = ctx.obj["khub"].config
    from knowledge_hub.papers.downloader import PaperDownloader
    from knowledge_hub.core.database import SQLiteDatabase

    downloader = PaperDownloader(config.papers_dir)
    sqlite_db = SQLiteDatabase(config.sqlite_path)

    existing = sqlite_db.get_paper(arxiv_id)
    title = existing["title"] if existing else arxiv_id

    try:
        with console.status(f"다운로드 중: {arxiv_id}..."):
            result = downloader.download_single(arxiv_id, title)
    except Exception as e:
        console.print(f"[red]다운로드 실패: {e}[/red]")
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
    else:
        console.print(f"[red]다운로드 실패: {arxiv_id}[/red]")


# ─────────────────────────────────────────────
# paper translate <arxiv_id>
# ─────────────────────────────────────────────
@paper_group.command("translate")
@click.argument("arxiv_id")
@click.option("--provider", "-p", default=None, help="번역 프로바이더 (기본: config)")
@click.option("--model", "-m", default=None, help="번역 모델 (기본: config)")
@click.pass_context
def paper_translate(ctx, arxiv_id, provider, model):
    """논문 전체 번역 (arXiv ID 지정)"""
    arxiv_id = _validate_arxiv_id(arxiv_id)
    config = ctx.obj["khub"].config
    from knowledge_hub.core.database import SQLiteDatabase

    sqlite_db = SQLiteDatabase(config.sqlite_path)
    paper = sqlite_db.get_paper(arxiv_id)

    if not paper:
        console.print(f"[red]논문을 찾을 수 없습니다: {arxiv_id}[/red]")
        return

    text_path = paper.get("text_path")
    if not text_path:
        console.print("[red]텍스트 파일이 없습니다. khub paper download 먼저 실행하세요.[/red]")
        return

    prov = provider or config.translation_provider
    mdl = model or config.translation_model

    console.print(f"번역 중: [bold]{paper['title'][:60]}[/bold]")
    console.print(f"[dim]프로바이더: {prov}/{mdl}[/dim]")

    from knowledge_hub.providers.registry import get_llm

    llm = get_llm(prov, model=mdl, **config.get_provider_config(prov))

    try:
        text = Path(text_path).read_text(encoding="utf-8")
    except Exception as e:
        console.print(f"[red]텍스트 파일 읽기 실패: {e}[/red]")
        return

    output_dir = Path(config.papers_dir) / "translated"
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_title = re.sub(r'[\\/:*?"<>|]', '', paper['title']).strip()
    safe_title = re.sub(r'\s+', ' ', safe_title)[:100].strip()
    output_path = output_dir / f"{safe_title}_translated.md"

    chunk_size = 6000
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    translated_parts = []
    for i, chunk in enumerate(chunks):
        console.print(f"  [{i + 1}/{len(chunks)}] 번역 중...")
        result = llm.translate(chunk, source_lang="en", target_lang="ko")
        translated_parts.append(result)

    full_translation = "\n\n".join(translated_parts)
    header = f"# {paper['title']}\n\n> arXiv: {arxiv_id} | 번역: {prov}/{mdl}\n\n---\n\n"
    output_path.write_text(header + full_translation, encoding="utf-8")

    sqlite_db.conn.execute(
        "UPDATE papers SET translated_path = ? WHERE arxiv_id = ?",
        (str(output_path), arxiv_id),
    )
    sqlite_db.conn.commit()
    console.print(f"[green]번역 완료: {output_path.name}[/green]")


# ─────────────────────────────────────────────
# paper summarize <arxiv_id>
# ─────────────────────────────────────────────
@paper_group.command("summarize")
@click.argument("arxiv_id")
@click.option("--provider", "-p", default=None, help="요약 프로바이더 (기본: config)")
@click.option("--model", "-m", default=None, help="요약 모델 (기본: config)")
@click.option("--quick", is_flag=True, help="간단 요약 (5문장, abstract만 사용)")
@click.pass_context
def paper_summarize(ctx, arxiv_id, provider, model, quick):
    """논문 심층 요약 생성 (구조화된 분석)"""
    arxiv_id = _validate_arxiv_id(arxiv_id)
    config = ctx.obj["khub"].config
    from knowledge_hub.core.database import SQLiteDatabase

    sqlite_db = SQLiteDatabase(config.sqlite_path)
    paper = sqlite_db.get_paper(arxiv_id)

    if not paper:
        console.print(f"[red]논문을 찾을 수 없습니다: {arxiv_id}[/red]")
        return

    prov = provider or config.summarization_provider
    mdl = model or config.summarization_model

    console.print(f"요약 중: [bold]{paper['title'][:60]}[/bold]")
    console.print(f"[dim]프로바이더: {prov}/{mdl}[/dim]")

    from knowledge_hub.providers.registry import get_llm
    llm = get_llm(prov, model=mdl, **config.get_provider_config(prov))

    text = _collect_paper_text(paper, config)
    source_label = "전문" if len(text) > 2000 else "abstract"
    console.print(f"[dim]입력 소스: {source_label} ({len(text):,}자)[/dim]")

    with console.status("심층 요약 생성 중..."):
        if quick:
            summary = llm.summarize(text, language="ko", max_sentences=5)
        else:
            summary = llm.summarize_paper(text, title=paper["title"], language="ko")

    console.print(f"\n[bold]요약: {paper['title']}[/bold]\n")
    from rich.markdown import Markdown
    console.print(Markdown(summary))

    sqlite_db.conn.execute(
        "UPDATE papers SET notes = ? WHERE arxiv_id = ?",
        (summary, arxiv_id),
    )
    sqlite_db.conn.commit()

    _update_obsidian_summary(paper, summary, config)
    console.print(f"\n[green]요약 저장 완료[/green]")


# ─────────────────────────────────────────────
# paper embed <arxiv_id>
# ─────────────────────────────────────────────
@paper_group.command("embed")
@click.argument("arxiv_id")
@click.pass_context
def paper_embed(ctx, arxiv_id):
    """단일 논문 벡터 임베딩"""
    arxiv_id = _validate_arxiv_id(arxiv_id)
    config = ctx.obj["khub"].config
    from knowledge_hub.core.database import SQLiteDatabase, VectorDatabase

    sqlite_db = SQLiteDatabase(config.sqlite_path)
    paper = sqlite_db.get_paper(arxiv_id)

    if not paper:
        console.print(f"[red]논문을 찾을 수 없습니다: {arxiv_id}[/red]")
        return

    console.print(f"임베딩 중: [bold]{paper['title'][:60]}[/bold]")

    text = f"Title: {paper['title']}"
    if paper.get("notes"):
        text += f"\n\n{paper['notes']}"

    from knowledge_hub.providers.registry import get_embedder as _get_embedder
    try:
        embed_cfg = config.get_provider_config(config.embedding_provider)
        embedder = _get_embedder(config.embedding_provider, model=config.embedding_model, **embed_cfg)
        emb = embedder.embed_text(text)
    except Exception as e:
        console.print(f"[red]임베딩 실패: {e}[/red]")
        return

    vector_db = VectorDatabase(config.vector_db_path, config.collection_name)
    vector_db.add_documents(
        documents=[text],
        embeddings=[emb],
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


# ─────────────────────────────────────────────
# paper translate-all
# ─────────────────────────────────────────────
@paper_group.command("translate-all")
@click.option("--limit", "-n", default=0, help="최대 번역 수 (0=전체)")
@click.option("--field", "-f", default=None, help="분야 필터")
@click.option("--provider", "-p", default=None, help="번역 프로바이더")
@click.option("--model", "-m", default=None, help="번역 모델")
@click.pass_context
def paper_translate_all(ctx, limit, field, provider, model):
    """미번역 논문 전체 번역"""
    config = ctx.obj["khub"].config
    from knowledge_hub.core.database import SQLiteDatabase

    sqlite_db = SQLiteDatabase(config.sqlite_path)
    papers = sqlite_db.list_papers(field=field, limit=999)
    untranslated = [p for p in papers if not p.get("translated_path") and p.get("text_path")]

    if limit > 0:
        untranslated = untranslated[:limit]

    if not untranslated:
        console.print("[green]모든 논문이 이미 번역되었거나 텍스트 파일이 없습니다.[/green]")
        return

    prov = provider or config.translation_provider
    mdl = model or config.translation_model

    console.print(f"[bold]미번역 논문 {len(untranslated)}편 번역 시작[/bold]")
    console.print(f"[dim]프로바이더: {prov}/{mdl}[/dim]\n")

    from knowledge_hub.providers.registry import get_llm
    llm = get_llm(prov, model=mdl, **config.get_provider_config(prov))

    output_dir = Path(config.papers_dir) / "translated"
    output_dir.mkdir(parents=True, exist_ok=True)
    success = 0
    failed: list[dict] = []

    for idx, paper in enumerate(untranslated, 1):
        aid = paper["arxiv_id"]
        title = paper["title"]
        console.print(f"[{idx}/{len(untranslated)}] {title[:55]}...", end=" ")

        try:
            text = Path(paper["text_path"]).read_text(encoding="utf-8")
        except Exception as e:
            console.print(f"[red]읽기 실패: {e}[/red]")
            failed.append({"arxiv_id": aid, "error": f"파일 읽기: {e}"})
            continue

        chunk_size = 6000
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        translated_parts = []
        chunk_failed = False
        for ci, chunk in enumerate(chunks):
            try:
                translated_parts.append(llm.translate(chunk, source_lang="en", target_lang="ko"))
            except Exception as e:
                log.error("번역 실패 %s 청크 %d: %s", aid, ci, e)
                console.print(f"[red]청크 {ci+1} 실패[/red]")
                failed.append({"arxiv_id": aid, "error": f"청크 {ci+1}: {e}"})
                chunk_failed = True
                break

        if chunk_failed:
            continue

        safe_title = re.sub(r'[\\/:*?"<>|]', '', title).strip()
        safe_title = re.sub(r'\s+', ' ', safe_title)[:100].strip()
        out_path = output_dir / f"{safe_title}_translated.md"

        header = f"# {title}\n\n> arXiv: {aid} | 번역: {prov}/{mdl}\n\n---\n\n"
        out_path.write_text(header + "\n\n".join(translated_parts), encoding="utf-8")

        sqlite_db.conn.execute(
            "UPDATE papers SET translated_path = ? WHERE arxiv_id = ?",
            (str(out_path), aid),
        )
        sqlite_db.conn.commit()
        success += 1
        console.print(f"[green]OK ({len(chunks)}청크)[/green]")

    console.print(f"\n[bold green]{success}/{len(untranslated)}편 번역 완료[/bold green]")
    if failed:
        console.print(f"[bold red]⚠ 실패: {len(failed)}편[/bold red]")
        for f in failed:
            console.print(f"  {f['arxiv_id']}: {f['error'][:80]}")


# ─────────────────────────────────────────────
# paper summarize-all
# ─────────────────────────────────────────────
@paper_group.command("summarize-all")
@click.option("--limit", "-n", default=0, help="최대 요약 수 (0=전체)")
@click.option("--field", "-f", default=None, help="분야 필터")
@click.option("--quick", is_flag=True, help="간단 요약 (구조화 분석 대신 3-5문장)")
@click.option("--resummary", is_flag=True, help="이미 요약된 논문도 재요약")
@click.pass_context
def paper_summarize_all(ctx, limit, field, quick, resummary):
    """전체 논문 심층 요약 (구조화된 분석)"""
    config = ctx.obj["khub"].config
    from knowledge_hub.core.database import SQLiteDatabase
    from knowledge_hub.providers.registry import get_llm

    sqlite_db = SQLiteDatabase(config.sqlite_path)
    papers = sqlite_db.list_papers(field=field, limit=999)

    if resummary:
        targets = papers
    else:
        targets = [p for p in papers if not p.get("notes") or len(p.get("notes", "")) < 100]

    if limit > 0:
        targets = targets[:limit]

    if not targets:
        console.print("[green]모든 논문이 이미 요약되어 있습니다.[/green]")
        return

    prov = config.summarization_provider
    mdl = config.summarization_model
    llm = get_llm(prov, model=mdl, **config.get_provider_config(prov))

    console.print(f"[bold]{len(targets)}편 {'간단' if quick else '심층'} 요약 시작[/bold]")
    console.print(f"[dim]프로바이더: {prov}/{mdl}[/dim]\n")

    # abstract가 없는 논문은 Semantic Scholar에서 보충
    missing_abstract = [p for p in targets if not _collect_paper_text(p, config) or len(_collect_paper_text(p, config)) < 100]
    if missing_abstract:
        aids = [p["arxiv_id"] for p in missing_abstract]
        abstract_map = {}
        for i in range(0, len(aids), 50):
            chunk = aids[i:i+50]
            try:
                resp = requests.post(
                    "https://api.semanticscholar.org/graph/v1/paper/batch",
                    params={"fields": "title,abstract,externalIds"},
                    json={"ids": [f"ArXiv:{a}" for a in chunk]},
                    timeout=60,
                )
                if resp.status_code == 200:
                    for paper_data in resp.json():
                        if paper_data and paper_data.get("abstract"):
                            ext = paper_data.get("externalIds", {})
                            aid = ext.get("ArXiv", "")
                            if aid:
                                abstract_map[aid] = paper_data["abstract"]
            except Exception:
                pass
        console.print(f"[dim]Semantic Scholar에서 {len(abstract_map)}편 abstract 보충[/dim]\n")

    success = 0
    failed: list[dict] = []
    for idx, p in enumerate(targets, 1):
        aid = p["arxiv_id"]
        title = p["title"]

        text = _collect_paper_text(p, config)
        if len(text) < 100 and 'abstract_map' in dir():
            extra = abstract_map.get(aid, "")
            if extra:
                text = f"제목: {title}\n초록: {extra}"

        if len(text) < 50:
            console.print(f"  [{idx}/{len(targets)}] {aid} - 텍스트 부족, 스킵")
            continue

        source = "전문" if len(text) > 2000 else "abstract"
        console.print(f"  [{idx}/{len(targets)}] {title[:50]}... ({source})", end=" ")

        try:
            if quick:
                summary = llm.summarize(text, language="ko", max_sentences=5)
            else:
                summary = llm.summarize_paper(text, title=title, language="ko")

            sqlite_db.conn.execute(
                "UPDATE papers SET notes = ? WHERE arxiv_id = ?",
                (summary, aid),
            )
            sqlite_db.conn.commit()

            _update_obsidian_summary(p, summary, config)
            success += 1
            console.print("[green]OK[/green]")
        except Exception as e:
            log.error("요약 실패 %s: %s", aid, e)
            failed.append({"arxiv_id": aid, "error": str(e)})
            console.print(f"[red]FAIL ({e})[/red]")

    console.print(f"\n[bold green]{success}/{len(targets)}편 요약 완료[/bold green]")
    if failed:
        console.print(f"[bold red]실패: {len(failed)}편[/bold red]")
        for f in failed:
            console.print(f"  {f['arxiv_id']}: {f['error'][:80]}")


# ─────────────────────────────────────────────
# paper embed-all
# ─────────────────────────────────────────────
@paper_group.command("embed-all")
@click.option("--all", "index_all", is_flag=True, help="이미 인덱싱된 논문도 재인덱싱")
@click.pass_context
def paper_embed_all(ctx, index_all):
    """미인덱싱 논문 전체 벡터 임베딩"""
    from knowledge_hub.core.database import SQLiteDatabase, VectorDatabase

    config = ctx.obj["khub"].config
    sqlite_db = SQLiteDatabase(config.sqlite_path)
    papers = sqlite_db.list_papers(limit=999)
    unindexed = papers if index_all else [p for p in papers if not p.get("indexed")]

    if not unindexed:
        console.print("[green]모든 논문이 이미 인덱싱되어 있습니다.[/green]")
        return

    console.print(f"[bold]인덱싱 시작: {len(unindexed)}편[/bold]")
    console.print(f"[dim]임베딩: {config.embedding_provider}/{config.embedding_model}[/dim]")

    from knowledge_hub.providers.registry import get_embedder as _get_embedder
    embed_cfg = config.get_provider_config(config.embedding_provider)
    embedder = _get_embedder(config.embedding_provider, model=config.embedding_model, **embed_cfg)

    vector_db = VectorDatabase(config.vector_db_path, config.collection_name)
    batch_size = 20
    success = 0
    t_start = time.time()

    for i in range(0, len(unindexed), batch_size):
        batch = unindexed[i:i + batch_size]
        texts = []
        for p in batch:
            t = f"Title: {p['title'] or p['arxiv_id']}"
            if p.get("notes"):
                t += f"\n\n{p['notes']}"
            texts.append(t)

        try:
            raw_embs = embedder.embed_batch(texts, show_progress=False)
            embs = [e for e in raw_embs if e is not None]
            if len(embs) != len(texts):
                raise RuntimeError(f"{len(texts) - len(embs)}개 텍스트 임베딩 실패")

            docs, embeddings, metas, ids = [], [], [], []
            for p, text, emb in zip(batch, texts, embs):
                docs.append(text)
                embeddings.append(emb)
                metas.append({
                    "title": p["title"] or "",
                    "arxiv_id": p["arxiv_id"],
                    "source_type": "paper",
                    "field": p.get("field", ""),
                    "chunk_index": 0,
                })
                ids.append(f"paper_{p['arxiv_id']}_0")

            vector_db.add_documents(documents=docs, embeddings=embeddings, metadatas=metas, ids=ids)

            for p in batch:
                sqlite_db.conn.execute("UPDATE papers SET indexed = 1 WHERE arxiv_id = ?", (p["arxiv_id"],))
            sqlite_db.conn.commit()

            success += len(batch)
            console.print(f"  [{success}/{len(unindexed)}] 배치: [green]{len(batch)}편 OK[/green]")
        except Exception as e:
            console.print(f"  배치 실패: [red]{e}[/red]")

    elapsed = time.time() - t_start
    console.print(f"\n[bold green]{success}/{len(unindexed)}편 인덱싱 완료 ({elapsed:.1f}초)[/bold green]")


# ─────────────────────────────────────────────
# paper info <arxiv_id>
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# paper sync-keywords
# ─────────────────────────────────────────────
@paper_group.command("sync-keywords")
@click.option("--force", is_flag=True, help="이미 키워드가 있는 논문도 재추출")
@click.option("--limit", "-n", default=0, help="최대 처리 수 (0=전체)")
@click.pass_context
def paper_sync_keywords(ctx, force, limit):
    """모든 논문에서 핵심 키워드+근거 추출 → kg_relations + Obsidian 노트 갱신"""
    config = ctx.obj["khub"].config
    from knowledge_hub.core.database import SQLiteDatabase
    import json

    vault_path = config.vault_path
    if not vault_path:
        console.print("[red]Obsidian vault 경로가 설정되지 않았습니다. khub config set obsidian.vault_path <경로>[/red]")
        return

    papers_dir = _resolve_vault_papers_dir(vault_path)
    if not papers_dir or not papers_dir.exists():
        console.print(f"[red]Obsidian 논문 폴더를 찾을 수 없습니다.[/red]")
        console.print("[dim]khub config set obsidian.vault_path 로 vault 경로를 확인하세요.[/dim]")
        return

    from knowledge_hub.providers.registry import get_llm
    prov = config.summarization_provider
    mdl = config.summarization_model
    prov_cfg = config.get_provider_config(prov)

    sqlite_db = SQLiteDatabase(config.sqlite_path)
    md_files = sorted(papers_dir.glob("*.md"))
    md_files = [f for f in md_files if f.name != "00_Concept_Index.md"]

    console.print(f"[bold]Obsidian 논문 노트 {len(md_files)}개 스캔 중...[/bold]\n")

    all_concepts: dict[str, list[str]] = {}
    updated = 0
    skipped = 0
    relations_added = 0

    for idx, md_path in enumerate(md_files):
        content = md_path.read_text(encoding="utf-8")

        arxiv_match = re.search(r'arxiv_id:\s*"?([0-9]+\.[0-9]+)"?', content)
        arxiv_id = arxiv_match.group(1) if arxiv_match else None

        has_good_concepts = "내가 배워야 할 개념" in content and "[[" in content.split("내가 배워야 할 개념")[-1].split("#")[0] if "내가 배워야 할 개념" in content else False

        if has_good_concepts and not force:
            concepts_section = content.split("내가 배워야 할 개념")[-1]
            next_heading = concepts_section.find("\n# ")
            if next_heading > 0:
                concepts_section = concepts_section[:next_heading]
            concepts = re.findall(r'\[\[([^\]]+)\]\]', concepts_section)
            concepts = [c for c in concepts if c != "00_Concept_Index"]
            for c in concepts:
                all_concepts.setdefault(c, []).append(md_path.stem)
            skipped += 1
            continue

        if limit > 0 and updated >= limit:
            break

        title = md_path.stem
        summary_text = _extract_summary_text(content, title, sqlite_db)

        if not summary_text or len(summary_text) < 20:
            console.print(f"  [{idx+1}/{len(md_files)}] {title[:50]}... [dim]텍스트 부족, 스킵[/dim]")
            skipped += 1
            continue

        console.print(f"  [{idx+1}/{len(md_files)}] {title[:50]}...", end=" ")

        try:
            if not hasattr(paper_sync_keywords, '_llm'):
                paper_sync_keywords._llm = get_llm(prov, model=mdl, **prov_cfg)
            evidence_results = _extract_keywords_with_evidence(paper_sync_keywords._llm, title, summary_text, sqlite_db)
        except Exception as e:
            console.print(f"[red]실패: {e}[/red]")
            continue

        if not evidence_results:
            console.print("[yellow]키워드 없음[/yellow]")
            continue

        concepts = [e["concept"] for e in evidence_results]
        for c in concepts:
            all_concepts.setdefault(c, []).append(title)

        # kg_relations에 paper_uses_concept 관계 + 근거 저장
        if arxiv_id:
            for ev in evidence_results:
                cname = ev["concept"]
                cid = _concept_id(cname)
                sqlite_db.upsert_concept(cid, cname)
                sqlite_db.add_relation(
                    source_type="paper", source_id=arxiv_id,
                    relation="paper_uses_concept",
                    target_type="concept", target_id=cid,
                    evidence_text=ev.get("evidence", ""),
                    confidence=ev.get("confidence", 0.7),
                )
                relations_added += 1

        new_content = _update_note_concepts(content, concepts)
        md_path.write_text(new_content, encoding="utf-8")
        updated += 1
        console.print(f"[green]{len(concepts)}개 키워드[/green]")

    console.print(f"\n[bold]업데이트: {updated}개 | 스킵(기존): {skipped}개 | 관계: {relations_added}개[/bold]")

    concept_index_path = papers_dir / "00_Concept_Index.md"
    _regenerate_concept_index(concept_index_path, all_concepts)
    console.print(f"[bold green]Concept Index 갱신 완료 ({len(all_concepts)}개 개념)[/bold green]")


def _extract_summary_text(content: str, title: str, sqlite_db) -> str:
    """노트에서 요약/초록 텍스트 추출, 없으면 DB에서 가져오기"""
    placeholder = "요약본/번역본이 아직 등록되지 않았습니다"

    for heading in ["## 요약", "# 📌 한줄 요약", "## 초록"]:
        if heading in content:
            section = content.split(heading, 1)[1]
            next_h = re.search(r'\n#{1,3} ', section)
            if next_h:
                section = section[:next_h.start()]
            section = section.strip()
            if section and placeholder not in section and len(section) > 20:
                return section[:3000]

    arxiv_match = re.search(r'arxiv_id:\s*"?([0-9]+\.[0-9]+)"?', content)
    if arxiv_match:
        aid = arxiv_match.group(1)
        paper = sqlite_db.get_paper(aid)
        if paper:
            notes = paper.get("notes", "")
            if notes and len(notes) > 30:
                return f"제목: {paper.get('title', title)}\n분야: {paper.get('field', '')}\n{notes}"[:3000]

    return f"제목: {title}"


def _extract_keywords_with_evidence(llm, title: str, text: str,
                                     sqlite_db=None) -> list[dict]:
    """LLM으로 키워드 + 근거 문장을 함께 추출.

    반환: [{"concept": "Transformer", "evidence": "We propose...", "confidence": 0.9}, ...]
    """
    import json as _json

    prompt = (
        "You extract 5-10 core academic concepts from AI/ML papers. "
        "For each concept, provide a short evidence sentence from the text that "
        "shows why this concept is relevant to this paper, plus a confidence score.\n\n"
        "Return ONLY valid JSON: [{\"concept\": \"Name\", \"evidence\": \"sentence\", \"confidence\": 0.9}, ...]\n\n"
        "Rules:\n"
        "- Use SINGULAR form (e.g. 'Neural Network' not 'Neural Networks')\n"
        "- Use full names, not abbreviations\n"
        "- Use standard academic terms\n"
        "- confidence: 0.5-1.0 based on how central the concept is to this paper\n"
        "- evidence: 1 sentence from the text, or a brief paraphrase if exact quote unavailable\n\n"
        f"Paper: {title}\n\n{text[:2500]}"
    )
    raw = llm.generate(prompt).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    items = _json.loads(raw)
    if not isinstance(items, list):
        return []

    results = []
    seen = set()
    for item in items:
        if not isinstance(item, dict) or "concept" not in item:
            continue
        name = str(item["concept"]).strip()
        if not name or len(name) <= 1:
            continue
        if sqlite_db:
            canonical = sqlite_db.resolve_concept(name)
            if canonical:
                name = canonical
        if name.lower() not in seen:
            seen.add(name.lower())
            results.append({
                "concept": name,
                "evidence": str(item.get("evidence", ""))[:500],
                "confidence": min(1.0, max(0.0, float(item.get("confidence", 0.7)))),
            })
    return results


def _extract_keywords_openai(llm, title: str, text: str,
                              sqlite_db=None) -> list[str]:
    """LLM으로 핵심 키워드 5~10개 추출 + DB alias 정규화 적용"""
    import json as _json

    prompt = (
        "You extract 5-10 core academic concepts/keywords from AI/ML papers. "
        "Return ONLY a JSON array of English concept names. "
        "Use standard academic terms (e.g. 'Transformer', 'Attention Mechanism', "
        "'Reinforcement Learning', 'Knowledge Distillation'). "
        "Always use SINGULAR form (e.g. 'Neural Network' not 'Neural Networks'). "
        "Use full names, not abbreviations (e.g. 'Large Language Model' not 'LLM'). "
        "Do NOT include LaTeX commands, paper-specific names, or generic terms like 'AI' or 'deep learning' unless central.\n\n"
        f"Paper: {title}\n\n{text[:2500]}"
    )
    raw = llm.generate(prompt).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    keywords = _json.loads(raw)
    if not isinstance(keywords, list):
        return []

    result = []
    seen = set()
    for k in keywords:
        name = str(k).strip()
        if not name or len(name) <= 1:
            continue
        if sqlite_db:
            canonical = sqlite_db.resolve_concept(name)
            if canonical:
                name = canonical
        if name.lower() not in seen:
            seen.add(name.lower())
            result.append(name)
    return result


def _update_note_concepts(content: str, concepts: list[str]) -> str:
    """노트 내용에서 키워드 섹션을 업데이트 또는 추가"""
    concept_lines = "# 🧩 내가 배워야 할 개념\n- [[00_Concept_Index]]\n"
    for c in concepts:
        concept_lines += f"- [[{c}]]\n"

    placeholder = "요약본/번역본이 아직 등록되지 않았습니다"
    if placeholder in content:
        old_line_pattern = re.compile(r'.*요약본/번역본이 아직.*paper sync-keywords.*\n?', re.DOTALL)
        cleaned = content
        for line in content.split('\n'):
            if placeholder in line:
                cleaned = cleaned.replace(line, '')
                break
        for line in cleaned.split('\n'):
            if 'sync-keywords' in line:
                cleaned = cleaned.replace(line, '')
                break
        content = cleaned.rstrip() + "\n\n"

    if "내가 배워야 할 개념" in content:
        pattern = re.compile(
            r'(#[#\s]*🧩?\s*내가 배워야 할 개념.*?\n)((?:- \[\[.*?\]\]\n)*)',
            re.MULTILINE,
        )
        if pattern.search(content):
            content = pattern.sub(concept_lines, content)
        else:
            content = content.rstrip() + "\n\n" + concept_lines
    elif "핵심 키워드:" in content:
        kw_line = re.search(r'핵심 키워드:.*\n', content)
        if kw_line:
            content = content[:kw_line.start()] + concept_lines + content[kw_line.end():]
    else:
        content = content.rstrip() + "\n\n" + concept_lines

    return content


def _regenerate_concept_index(index_path: Path, all_concepts: dict[str, list[str]]):
    """00_Concept_Index.md를 빈도순으로 재생성"""
    sorted_concepts = sorted(all_concepts.items(), key=lambda x: -len(x[1]))

    lines = [
        "---",
        "title: 00_Concept_Index",
        "---",
        "",
        "# AI Papers Concept Index",
        "",
        "이 폴더 내 요약 노트에서 추출된 개념 링크 목록",
        "",
        "## 개념",
    ]
    for concept, papers in sorted_concepts:
        lines.append(f"- [[{concept}]] ({len(papers)})")

    lines.append("")
    index_path.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────
# paper build-concepts
# ─────────────────────────────────────────────
@paper_group.command("build-concepts")
@click.option("--force", is_flag=True, help="기존 개념 노트도 재생성")
@click.pass_context
def paper_build_concepts(ctx, force):
    """모든 키워드에 대해 개별 개념 노트 생성 + kg_relations에 관계 저장"""
    config = ctx.obj["khub"].config
    import json

    vault_path = config.vault_path
    if not vault_path:
        console.print("[red]Obsidian vault 경로가 설정되지 않았습니다.[/red]")
        return

    papers_dir = _resolve_vault_papers_dir(vault_path)
    concepts_dir = _resolve_vault_concepts_dir(vault_path)
    concepts_dir.mkdir(parents=True, exist_ok=True)

    from knowledge_hub.providers.registry import get_llm as _get_llm
    prov = config.summarization_provider
    mdl = config.summarization_model
    prov_cfg = config.get_provider_config(prov)
    llm = _get_llm(prov, model=mdl, **prov_cfg)

    from knowledge_hub.core.database import SQLiteDatabase
    sqlite_db = SQLiteDatabase(config.sqlite_path)

    # 1) 모든 논문 노트에서 개념 → 논문 매핑 수집
    concept_papers: dict[str, list[str]] = {}
    md_files = sorted(papers_dir.glob("*.md"))
    for md_path in md_files:
        if md_path.name == "00_Concept_Index.md":
            continue
        content = md_path.read_text(encoding="utf-8")
        concepts = re.findall(r'\[\[([^\]]+)\]\]', content)
        for c in concepts:
            if c != "00_Concept_Index":
                concept_papers.setdefault(c, []).append(md_path.stem)

    all_concept_names = sorted(concept_papers.keys())
    console.print(f"[bold]{len(all_concept_names)}개 개념 발견[/bold]")

    if not force:
        existing = {f.stem for f in concepts_dir.glob("*.md")}
        to_process = [c for c in all_concept_names if c not in existing]
    else:
        to_process = list(all_concept_names)

    if not to_process:
        console.print("[green]모든 개념 노트가 이미 생성되어 있습니다. --force로 재생성 가능.[/green]")
        _rebuild_concept_index_with_relations(papers_dir, concepts_dir, concept_papers)
        return

    console.print(f"[bold]{len(to_process)}개 개념 노트 생성 시작[/bold]\n")

    batch_size = 15
    created = 0
    relations_stored = 0

    for i in range(0, len(to_process), batch_size):
        batch = to_process[i:i + batch_size]
        console.print(f"  배치 [{i+1}~{i+len(batch)}/{len(to_process)}]...", end=" ")

        try:
            results = _batch_describe_concepts(llm, batch, all_concept_names)
        except Exception as e:
            console.print(f"[red]API 오류: {e}[/red]")
            continue

        for concept_name, info in results.items():
            desc = info.get("description", "")
            related = info.get("related", [])
            papers = concept_papers.get(concept_name, [])

            cid = _concept_id(concept_name)
            sqlite_db.upsert_concept(cid, concept_name, desc)

            for rel_name in related:
                rel_id = _concept_id(rel_name)
                sqlite_db.upsert_concept(rel_id, rel_name)
                sqlite_db.add_relation(
                    source_type="concept", source_id=cid,
                    relation="concept_related_to",
                    target_type="concept", target_id=rel_id,
                    evidence_text=f"LLM이 {concept_name}의 관련 개념으로 식별",
                    confidence=0.6,
                )
                relations_stored += 1

            note_content = _build_concept_note(concept_name, desc, related, papers)
            safe_name = re.sub(r'[\\/:*?"<>|]', '', concept_name).strip()
            note_path = concepts_dir / f"{safe_name}.md"
            note_path.write_text(note_content, encoding="utf-8")
            created += 1

        console.print(f"[green]{len(results)}개 생성[/green]")

    _rebuild_concept_index_with_relations(papers_dir, concepts_dir, concept_papers)

    console.print(f"\n[bold green]{created}개 개념 노트 생성 완료[/bold green]")
    console.print(f"[dim]concept_related_to 관계: {relations_stored}개 저장[/dim]")
    console.print(f"[dim]위치: {concepts_dir}[/dim]")


def _batch_describe_concepts(llm, batch: list[str], all_concepts: list[str]) -> dict:
    """LLM으로 개념 배치의 설명 + 관련 개념 추출"""
    import json

    concept_list_str = ", ".join(all_concepts[:200])

    prompt = (
        "You are an AI/ML concept expert. For each concept, provide:\n"
        "1. A concise Korean description (1-2 sentences) explaining what it is\n"
        "2. 3-5 related concepts from the provided concept list\n\n"
        "Return ONLY valid JSON: {\"ConceptName\": {\"description\": \"한국어 설명\", \"related\": [\"Related1\", \"Related2\", ...]}, ...}\n"
        "Pick related concepts ONLY from the provided list. Be precise and educational.\n\n"
        f"Concepts to describe:\n{json.dumps(batch, ensure_ascii=False)}\n\n"
        f"Available concepts for relations:\n{concept_list_str}"
    )
    raw = llm.generate(prompt).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


def _build_concept_note(name: str, description: str, related: list[str], papers: list[str]) -> str:
    """개별 개념 노트 마크다운 생성"""
    lines = [
        "---",
        "type: concept",
        f'title: "{name}"',
        "---",
        "",
        f"# {name}",
        "",
        description,
        "",
    ]

    if related:
        lines.append("## 관련 개념")
        for r in related:
            lines.append(f"- [[{r}]]")
        lines.append("")

    if papers:
        lines.append("## 관련 논문")
        for p in papers:
            lines.append(f"- [[{p}]]")
        lines.append("")

    lines.append(f"*[[00_Concept_Index|← 개념 목록으로]]*")
    lines.append("")
    return "\n".join(lines)


def _rebuild_concept_index_with_relations(papers_dir: Path, concepts_dir: Path, concept_papers: dict[str, list[str]]):
    """Concept Index를 관계 정보 포함하여 재생성"""
    sorted_concepts = sorted(concept_papers.items(), key=lambda x: -len(x[1]))

    has_note = {f.stem for f in concepts_dir.glob("*.md")}

    lines = [
        "---",
        "title: 00_Concept_Index",
        "---",
        "",
        "# AI Papers Concept Index",
        "",
        "이 폴더 내 요약 노트에서 추출된 개념 링크 목록",
        f"총 **{len(sorted_concepts)}개** 개념 | **{len(has_note)}개** 설명 노트 생성됨",
        "",
    ]

    freq_groups = {"## 핵심 개념 (3회 이상)": [], "## 주요 개념 (2회)": [], "## 기타 개념 (1회)": []}
    for concept, papers in sorted_concepts:
        count = len(papers)
        status = "📝" if concept in has_note else "📌"
        entry = f"- {status} [[{concept}]] ({count}편)"
        if count >= 3:
            freq_groups["## 핵심 개념 (3회 이상)"].append(entry)
        elif count == 2:
            freq_groups["## 주요 개념 (2회)"].append(entry)
        else:
            freq_groups["## 기타 개념 (1회)"].append(entry)

    for heading, entries in freq_groups.items():
        if entries:
            lines.append(heading)
            lines.extend(entries)
            lines.append("")

    lines.append("")
    (papers_dir / "00_Concept_Index.md").write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────
# paper normalize-concepts
# ─────────────────────────────────────────────
@paper_group.command("normalize-concepts")
@click.option("--dry-run", is_flag=True, help="변경 없이 탐지 결과만 표시")
@click.pass_context
def paper_normalize_concepts(ctx, dry_run):
    """개념 동의어/복수형/약어 탐지 → 정규화 + 병합"""
    config = ctx.obj["khub"].config
    import json as _json
    from knowledge_hub.core.database import SQLiteDatabase

    vault_path = config.vault_path
    if not vault_path:
        console.print("[red]Obsidian vault 경로가 설정되지 않았습니다.[/red]")
        return

    from knowledge_hub.providers.registry import get_llm as _get_llm
    prov = config.summarization_provider
    mdl = config.summarization_model
    prov_cfg = config.get_provider_config(prov)
    llm = _get_llm(prov, model=mdl, **prov_cfg)

    papers_dir = _resolve_vault_papers_dir(vault_path)
    concepts_dir = _resolve_vault_concepts_dir(vault_path)

    # 1) 모든 개념 이름 수집
    concept_names = sorted({f.stem for f in concepts_dir.glob("*.md")}) if concepts_dir.exists() else []

    md_files = sorted(papers_dir.glob("*.md"))
    for md_path in md_files:
        if md_path.name == "00_Concept_Index.md":
            continue
        content = md_path.read_text(encoding="utf-8")
        for c in re.findall(r'\[\[([^\]]+)\]\]', content):
            if c != "00_Concept_Index" and c not in concept_names:
                concept_names.append(c)

    concept_names = sorted(set(concept_names))
    console.print(f"[bold]{len(concept_names)}개 개념 스캔 완료[/bold]\n")

    if len(concept_names) < 2:
        console.print("[green]정규화할 개념이 부족합니다.[/green]")
        return

    # 2) LLM으로 동의어 그룹 탐지 (배치)
    console.print("[bold]동의어/복수형/약어 그룹 탐지 중...[/bold]")
    all_groups: list[dict] = []
    batch_size = 80

    for i in range(0, len(concept_names), batch_size):
        batch = concept_names[i:i + batch_size]
        console.print(f"  배치 [{i+1}~{i+len(batch)}/{len(concept_names)}]...", end=" ")
        try:
            groups = _detect_synonym_groups(llm, batch)
            all_groups.extend(groups)
            console.print(f"[green]{len(groups)}개 그룹[/green]")
        except Exception as e:
            console.print(f"[red]실패: {e}[/red]")

    if not all_groups:
        console.print("[green]동의어 그룹이 발견되지 않았습니다.[/green]")
        return

    # 3) 결과 표시
    table = Table(title=f"동의어 그룹 ({len(all_groups)}개)")
    table.add_column("정규 이름", style="cyan")
    table.add_column("별칭 (병합 대상)", style="yellow")
    for g in all_groups:
        table.add_row(g["canonical"], ", ".join(g["aliases"]))
    console.print(table)

    if dry_run:
        console.print("\n[dim]--dry-run: 변경 없이 종료[/dim]")
        return

    # 4) SQLite에 concepts + aliases 등록
    sqlite_db = SQLiteDatabase(config.sqlite_path)
    registered = 0

    for name in concept_names:
        cid = _concept_id(name)
        sqlite_db.upsert_concept(cid, name)

    for g in all_groups:
        canonical = g["canonical"]
        canonical_id = _concept_id(canonical)
        sqlite_db.upsert_concept(canonical_id, canonical)

        for alias in g["aliases"]:
            sqlite_db.add_alias(alias, canonical_id)

            alias_id = _concept_id(alias)
            existing = sqlite_db.get_concept(alias_id)
            if existing and existing["canonical_name"] != canonical:
                sqlite_db.delete_concept(alias_id)

        registered += 1

    console.print(f"\n[green]{registered}개 정규화 그룹 DB 등록[/green]")

    # 5) Obsidian 노트 병합 + 논문 노트 치환
    merged = 0
    for g in all_groups:
        canonical = g["canonical"]
        for alias in g["aliases"]:
            merged += _merge_obsidian_concept(papers_dir, concepts_dir, alias, canonical)
            _replace_in_paper_notes(papers_dir, alias, canonical)

    console.print(f"[green]Obsidian 노트 {merged}개 병합 완료[/green]")

    # 6) Concept Index 재생성
    concept_papers: dict[str, list[str]] = {}
    for md_path in sorted(papers_dir.glob("*.md")):
        if md_path.name == "00_Concept_Index.md":
            continue
        content = md_path.read_text(encoding="utf-8")
        for c in re.findall(r'\[\[([^\]]+)\]\]', content):
            if c != "00_Concept_Index":
                concept_papers.setdefault(c, []).append(md_path.stem)

    _rebuild_concept_index_with_relations(papers_dir, concepts_dir, concept_papers)
    console.print(f"[bold green]정규화 완료 — {len(all_groups)}개 그룹, {merged}개 노트 병합[/bold green]")


def _concept_id(name: str) -> str:
    """개념 이름에서 안정적인 ID 생성 (소문자, 공백→언더스코어)"""
    return re.sub(r'\s+', '_', name.strip()).lower()


def _detect_synonym_groups(llm, concept_names: list[str]) -> list[dict]:
    """LLM으로 동의어/복수형/약어 그룹 탐지"""
    import json as _json

    prompt = (
        "You are an AI/ML terminology expert. Given a list of concept names, "
        "find groups of synonyms, abbreviations, plural/singular variants, or "
        "near-duplicates that should be merged into a single canonical concept.\n\n"
        "Rules:\n"
        "- Only group terms that truly refer to the SAME concept\n"
        "- Do NOT merge parent-child (e.g. 'Reinforcement Learning' and 'Multi-Agent RL' are different)\n"
        "- Prefer singular form as canonical\n"
        "- Prefer full name over abbreviation as canonical\n"
        "- Return ONLY a JSON array of {\"canonical\": \"...\", \"aliases\": [\"...\"]}\n"
        "- Skip concepts with no duplicates\n\n"
        + _json.dumps(concept_names, ensure_ascii=False)
    )
    raw = llm.generate(prompt).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    groups = _json.loads(raw)
    if not isinstance(groups, list):
        return []
    return [g for g in groups if isinstance(g, dict) and g.get("canonical") and g.get("aliases")]


def _merge_obsidian_concept(papers_dir: Path, concepts_dir: Path, alias: str, canonical: str) -> int:
    """Obsidian 개념 노트 병합: alias 노트의 관련 논문/개념을 canonical에 합산 후 삭제"""
    safe_alias = re.sub(r'[\\/:*?"<>|]', '', alias).strip()
    safe_canonical = re.sub(r'[\\/:*?"<>|]', '', canonical).strip()
    alias_path = concepts_dir / f"{safe_alias}.md"
    canonical_path = concepts_dir / f"{safe_canonical}.md"

    if not alias_path.exists():
        return 0

    alias_content = alias_path.read_text(encoding="utf-8")
    alias_papers = set(re.findall(r'\[\[([^\]]+)\]\]', alias_content))

    if canonical_path.exists():
        can_content = canonical_path.read_text(encoding="utf-8")
        can_papers = set(re.findall(r'\[\[([^\]]+)\]\]', can_content))
        new_papers = alias_papers - can_papers - {canonical, "00_Concept_Index"}

        if new_papers and "## 관련 논문" in can_content:
            insert_point = can_content.index("## 관련 논문") + len("## 관련 논문")
            next_nl = can_content.index("\n", insert_point)
            extra = "\n".join(f"- [[{p}]]" for p in sorted(new_papers))
            can_content = can_content[:next_nl] + "\n" + extra + can_content[next_nl:]
            canonical_path.write_text(can_content, encoding="utf-8")

    alias_path.unlink()
    return 1


def _replace_in_paper_notes(papers_dir: Path, old_name: str, new_name: str):
    """모든 논문 노트에서 [[old_name]] → [[new_name]] 치환"""
    old_link = f"[[{old_name}]]"
    new_link = f"[[{new_name}]]"
    for md_path in papers_dir.glob("*.md"):
        if md_path.name == "00_Concept_Index.md":
            continue
        content = md_path.read_text(encoding="utf-8")
        if old_link in content:
            content = content.replace(old_link, new_link)
            md_path.write_text(content, encoding="utf-8")


# ─────────────────────────────────────────────
# paper info <arxiv_id>
# ─────────────────────────────────────────────
@paper_group.command("info")
@click.argument("arxiv_id")
@click.pass_context
def paper_info(ctx, arxiv_id):
    """논문 상세 정보"""
    arxiv_id = _validate_arxiv_id(arxiv_id)
    config = ctx.obj["khub"].config
    from knowledge_hub.core.database import SQLiteDatabase

    sqlite_db = SQLiteDatabase(config.sqlite_path)
    paper = sqlite_db.get_paper(arxiv_id)

    if not paper:
        console.print(f"[red]논문을 찾을 수 없습니다: {arxiv_id}[/red]")
        return

    table = Table(title=f"논문 정보: {arxiv_id}")
    table.add_column("항목", style="cyan", width=12)
    table.add_column("값")

    table.add_row("제목", paper["title"])
    table.add_row("저자", paper.get("authors", ""))
    table.add_row("연도", str(paper.get("year", "")))
    table.add_row("분야", paper.get("field", ""))
    table.add_row("중요도", str(paper.get("importance", "")))
    table.add_row("PDF", paper.get("pdf_path") or "-")
    table.add_row("텍스트", paper.get("text_path") or "-")
    table.add_row("번역", paper.get("translated_path") or "-")
    table.add_row("인덱싱", "O" if paper.get("indexed") else "-")
    table.add_row("arXiv", f"https://arxiv.org/abs/{arxiv_id}")

    console.print(table)

    notes = paper.get("notes", "")
    if notes and len(notes) > 30:
        console.print(f"\n[bold]요약:[/bold]")
        console.print(notes[:500])
