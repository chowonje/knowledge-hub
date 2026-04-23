"""
khub discover - 논문 자동 검색 → 다운로드 → 요약 → 인덱싱 → Obsidian 연결
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path
from threading import Lock

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from knowledge_hub.papers.judge_feedback import PaperJudgeFeedbackLogger
from knowledge_hub.papers.source_guard import review_downloaded_source, stage_source_guard

console = Console()
log = logging.getLogger("khub.discover")

_sqlite_lock = Lock()


def _build_pipeline(khub_ctx, need_translator=False):
    """Config에서 프로바이더를 읽어 파이프라인 객체 구성 (ChromaDB 제외 - 인덱싱은 별도 프로세스)"""
    factory = khub_ctx.factory
    config = factory.config
    summarizer = factory.build_llm(config.summarization_provider, config.summarization_model)
    translator = None
    if need_translator:
        translator = factory.build_llm(config.translation_provider, config.translation_model)
    sqlite_db = factory.get_sqlite_db()
    return config, summarizer, translator, sqlite_db


def _vector_db(khub_ctx, config):
    if hasattr(khub_ctx, "vector_db"):
        return khub_ctx.vector_db()
    from knowledge_hub.infrastructure.persistence import VectorDatabase

    return VectorDatabase(config.vector_db_path, config.collection_name)


def _get_embedder(config, khub_ctx=None):
    """config 기반 Embedder 인스턴스 생성"""
    if khub_ctx is not None and hasattr(khub_ctx, "build_embedder"):
        return khub_ctx.build_embedder(config.embedding_provider, config.embedding_model)
    from knowledge_hub.infrastructure.providers import get_embedder

    embed_cfg = config.get_provider_config(config.embedding_provider)
    return get_embedder(config.embedding_provider, model=config.embedding_model, **embed_cfg)


def _index_papers_inline(papers_to_index, khub_ctx, config):
    """배치 임베딩으로 벡터 인덱싱 (프로바이더 레지스트리 사용)"""
    vector_db = _vector_db(khub_ctx, config)
    if khub_ctx is not None and hasattr(khub_ctx, "build_embedder"):
        embedder = _get_embedder(config, khub_ctx=khub_ctx)
    else:
        embedder = _get_embedder(config)
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
        raw_embeddings = embedder.embed_batch(texts, show_progress=False)
        valid_pairs = [(p, t, e) for p, t, e in zip(paper_refs, texts, raw_embeddings) if e is not None]

        docs, embs, metas, ids = [], [], [], []
        for paper, text, emb in valid_pairs:
            docs.append(text)
            embs.append(emb)
            metas.append(
                {
                    "title": paper["title"],
                    "arxiv_id": paper["arxiv_id"],
                    "source_type": "paper",
                    "field": paper.get("field", ""),
                    "chunk_index": 0,
                }
            )
            ids.append(f"paper_{paper['arxiv_id']}_0")

        if docs:
            vector_db.add_documents(documents=docs, embeddings=embs, metadatas=metas, ids=ids)
        for paper, _, emb in valid_pairs:
            results[paper["arxiv_id"]] = {"ok": True, "chunks": 1}
        for paper in paper_refs:
            if paper["arxiv_id"] not in results:
                results[paper["arxiv_id"]] = {"ok": False, "chunks": 0, "error": "embedding failed"}
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
            str(i),
            p.arxiv_id,
            p.title[:55],
            str(p.year),
            str(p.citation_count),
            fields,
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
@click.option("--judge/--no-judge", "judge_enabled", default=False, help="optional paper discovery filter 사용")
@click.option("--judge-threshold", type=float, default=0.62, show_default=True, help="optional judge keep threshold")
@click.option("--judge-candidates", type=int, default=None, help="optional judge 평가 후보 수 (기본: max-papers*3)")
@click.option("--allow-external/--no-allow-external", default=False, show_default=True, help="optional judge에서 외부 LLM 사용 허용")
@click.option("--yes", "-y", "auto_confirm", is_flag=True, help="확인 없이 바로 진행")
@click.option("--workers", "-w", default=4, help="병렬 다운로드 워커 수 (기본: 4)")
@click.option("--index/--no-index", "do_index", default=True, help="벡터DB 인덱싱 여부")
@click.option("--json/--no-json", "as_json", default=False, show_default=True, help="결과를 JSON으로 출력")
@click.pass_context
def discover(
    ctx,
    topic,
    max_papers,
    year_start,
    min_citations,
    sort_by,
    translate,
    gen_summary,
    create_obsidian,
    judge_enabled,
    judge_threshold,
    judge_candidates,
    allow_external,
    auto_confirm,
    workers,
    do_index,
    as_json,
):
    """논문 자동 검색 → 다운로드 → 요약 → 인덱싱 → Obsidian 연결

    \b
    예시:
      khub discover "large language model agent" -n 5 --year 2024
      khub discover "RAG" -n 10 --min-citations 50 --sort citationCount
      khub discover "AI agent" -n 3 -y
      khub discover "transformer" -w 8
    """
    khub = ctx.obj["khub"]
    emit = console.print if not as_json else (lambda *args, **kwargs: None)
    status_ctx = console.status if not as_json else (lambda *_args, **_kwargs: nullcontext())

    if not as_json:
        emit(
            Panel.fit(
                f"[bold]논문 탐색: {topic}[/bold]\n"
                f"최대 {max_papers}편 | 연도: {year_start or '전체'} | 정렬: {sort_by} | 워커: {workers}",
                border_style="cyan",
            )
        )

    try:
        config, summarizer, translator, sqlite_db = _build_pipeline(khub, need_translator=translate)
    except Exception as e:
        emit(f"[red]초기화 실패: {e}[/red]")
        emit("[dim]khub init을 먼저 실행하세요.[/dim]")
        return

    if create_obsidian is None:
        create_obsidian = config.vault_enabled

    from knowledge_hub.papers.discoverer import discover_papers
    from knowledge_hub.papers.downloader import PaperDownloader

    try:
        with status_ctx("[cyan]Semantic Scholar + arXiv 검색 중...[/cyan]"):
            discovered = discover_papers(
                topic=topic,
                max_papers=max_papers * 3,
                year_start=year_start,
                min_citations=min_citations,
                sort_by=sort_by,
            )
    except Exception as e:
        emit(f"[red]논문 검색 실패: {e}[/red]")
        log.exception("검색 API 오류")
        return

    if not discovered:
        emit("[yellow]검색 결과가 없습니다.[/yellow]")
        return

    emit(f"[green]{len(discovered)}개 논문 발견[/green]")

    new_papers = []
    skipped = 0
    for paper in discovered:
        if sqlite_db.get_paper(paper.arxiv_id):
            skipped += 1
            continue
        new_papers.append(paper)
        max_candidate_count = judge_candidates if judge_candidates is not None else max_papers * 3
        if len(new_papers) >= (max_candidate_count if judge_enabled else max_papers):
            break

    if not new_papers:
        emit(f"[yellow]새로운 논문이 없습니다. {skipped}편 모두 이미 수집되었습니다.[/yellow]")
        return

    judge_payload = {
        "enabled": False,
        "backend": "",
        "threshold": round(float(judge_threshold or 0.62), 6),
        "candidateCount": len(new_papers),
        "selectedCount": len(new_papers),
        "degraded": False,
        "warnings": [],
        "items": [],
    }
    if judge_enabled:
        from knowledge_hub.papers.judge import JUDGE_BACKEND, PaperJudgeService

        judge_service = PaperJudgeService(
            config,
            llm=summarizer,
            allow_external=allow_external,
            pass_threshold=judge_threshold,
        )
        selected_papers, judge_result = judge_service.select_candidates(
            new_papers,
            topic=topic,
            threshold=judge_threshold,
            top_k=max_papers,
        )
        judge_payload = {
            "enabled": True,
            "backend": judge_result.get("backend", JUDGE_BACKEND),
            "threshold": judge_result.get("threshold", round(float(judge_threshold or 0.62), 6)),
            "candidateCount": int(judge_result.get("candidateCount", len(new_papers)) or 0),
            "selectedCount": int(judge_result.get("selectedCount", len(selected_papers)) or 0),
            "degraded": bool(judge_result.get("degraded", False)),
            "warnings": list(judge_result.get("warnings") or []),
            "items": list(judge_result.get("items") or []),
        }
        PaperJudgeFeedbackLogger(config).log_judge_decisions(
            topic=topic,
            items=judge_payload["items"],
            backend=str(judge_payload["backend"] or JUDGE_BACKEND),
            threshold=float(judge_payload["threshold"] or judge_threshold or 0.62),
            degraded=bool(judge_payload["degraded"]),
            allow_external=bool(allow_external),
            source="discover_cli",
        )
        new_papers = selected_papers
        emit(
            f"[cyan]judge selected {judge_payload['selectedCount']}/{judge_payload['candidateCount']} candidates[/cyan]"
        )
        if judge_payload["warnings"]:
            emit(f"[yellow]{'; '.join(judge_payload['warnings'])}[/yellow]")
        if not new_papers:
            emit("[yellow]judge threshold를 통과한 신규 논문이 없습니다.[/yellow]")
            return

    if not as_json:
        _show_candidate_table(new_papers, skipped)

    if not auto_confirm and not as_json:
        action = click.prompt(
            "\n  진행할까요? [Y]전체 / [n]취소 / 번호 선택(예: 1,3,5)",
            default="Y",
        )
        if action.strip().lower() == "n":
            emit("[dim]취소됨[/dim]")
            return
        if action.strip().upper() != "Y" and action.strip() != "":
            try:
                indices = [int(x.strip()) - 1 for x in action.split(",")]
                new_papers = [new_papers[i] for i in indices if 0 <= i < len(new_papers)]
                emit(f"[cyan]{len(new_papers)}편 선택됨[/cyan]")
            except (ValueError, IndexError):
                emit("[red]잘못된 입력. 취소합니다.[/red]")
                return

    if not new_papers:
        return

    downloader = PaperDownloader(config.papers_dir)
    paper_manager = None
    dl_results = {}
    dl_errors = {}

    emit(f"\n[bold]다운로드 시작 ({len(new_papers)}편, {workers}개 병렬)[/bold]")
    if as_json:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_download_one, p, downloader): p for p in new_papers}
            for future in as_completed(futures):
                paper, result, error = future.result()
                if error:
                    dl_errors[paper.arxiv_id] = error
                else:
                    dl_results[paper.arxiv_id] = result
    else:
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

    emit(f"  [green]{len(dl_results)}편 다운로드 완료[/green]", end="")
    if dl_errors:
        emit(f", [red]{len(dl_errors)}편 실패[/red]")
    else:
        emit()

    results = []
    papers_to_index = []
    obsidian_paths = []

    for idx, paper in enumerate(new_papers, 1):
        emit(f"\n[bold cyan]── [{idx}/{len(new_papers)}] {paper.title[:60]} ──[/bold cyan]")
        entry = {"arxiv_id": paper.arxiv_id, "title": paper.title, "steps": []}

        dl = dl_results.get(paper.arxiv_id)
        if not dl:
            entry["success"] = False
            entry["error"] = dl_errors.get(paper.arxiv_id, "다운로드 실패")
            results.append(entry)
            emit("  [red]다운로드 실패[/red]")
            continue

        try:
            entry["steps"].append("PDF" if dl.get("pdf") else "텍스트")
            emit("  [dim]PDF 저장됨[/dim]")

            importance = 5 if paper.citation_count >= 500 else 4 if paper.citation_count >= 100 else 3 if paper.citation_count >= 20 else 2 if paper.citation_count >= 5 else 1
            source_guard = stage_source_guard(sqlite_db, paper_id=paper.arxiv_id, title=paper.title)
            review = review_downloaded_source(
                sqlite_db,
                paper_id=paper.arxiv_id,
                title=paper.title,
                pdf_path=str(dl.get("pdf") or ""),
                text_path=str(dl.get("text") or ""),
                existing=source_guard,
            )
            entry["sourceGuard"] = dict(review.get("guard") or {})
            paper_data = {
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "authors": paper.authors,
                "year": paper.year,
                "field": ", ".join(paper.fields_of_study[:3]),
                "importance": importance,
                "notes": f"citations: {paper.citation_count}",
                "pdf_path": None,
                "text_path": None,
                "translated_path": None,
            }
            if bool(review.get("blocked")):
                with _sqlite_lock:
                    sqlite_db.upsert_paper(paper_data)
                entry["success"] = False
                entry["error"] = str(entry["sourceGuard"].get("reason") or "source guard blocked import")
                emit(f"  [yellow]source guard blocked[/yellow] {entry['error']}")
                results.append(entry)
                continue
            paper_data["pdf_path"] = str(review.get("finalPdfPath") or dl.get("pdf") or "") or None
            paper_data["text_path"] = str(review.get("finalTextPath") or dl.get("text") or "") or None
            with _sqlite_lock:
                sqlite_db.upsert_paper(paper_data)

            summary = ""
            if gen_summary and (paper.abstract or dl.get("text")):
                emit("  요약 중...", end=" ")
                try:
                    text_file = dl.get("text")
                    if text_file and Path(text_file).exists():
                        full_text = Path(text_file).read_text(encoding="utf-8")[:30000]
                        source_label = "전문"
                    else:
                        full_text = f"제목: {paper.title}\n초록: {paper.abstract}"
                        source_label = "abstract"

                    summary = summarizer.summarize_paper(full_text, title=paper.title, language="ko")
                    emit(f"[green]완료 ({source_label})[/green]")
                    entry["steps"].append(f"요약({source_label})")
                except Exception as e:
                    try:
                        summary = summarizer.summarize(
                            f"제목: {paper.title}\n초록: {paper.abstract}",
                            language="ko",
                            max_sentences=5,
                        )
                        emit(f"[yellow]간단 요약 ({e})[/yellow]")
                        entry["steps"].append("요약(간단)")
                    except Exception:
                        summary = paper.abstract[:500] if paper.abstract else ""
                        emit("[yellow]fallback[/yellow]")
                        entry["steps"].append("요약(fallback)")

            translated_abstract = ""
            if translate and translator and paper.abstract:
                emit("  번역 중...", end=" ")
                try:
                    translated_abstract = translator.translate(
                        paper.abstract,
                        source_lang="en",
                        target_lang="ko",
                    )
                    emit("[green]완료[/green]")
                    entry["steps"].append("번역")
                except Exception as e:
                    emit(f"[yellow]실패 ({e})[/yellow]")
                    entry["steps"].append("번역실패")

            if summary:
                with _sqlite_lock:
                    sqlite_db.conn.execute(
                        "UPDATE papers SET notes = ? WHERE arxiv_id = ?",
                        (summary, paper.arxiv_id),
                    )
                    sqlite_db.conn.commit()

            if do_index:
                papers_to_index.append(
                    {
                        "arxiv_id": paper.arxiv_id,
                        "title": paper.title,
                        "abstract": paper.abstract or "",
                        "summary": summary,
                        "field": paper_data["field"],
                    }
                )

            if create_obsidian and config.vault_path:
                emit("  Obsidian 노트...", end=" ")
                if paper_manager is None:
                    from knowledge_hub.papers.manager import PaperManager

                    paper_manager = PaperManager(
                        config=config,
                        vector_db=_vector_db(khub, config),
                        sqlite_db=sqlite_db,
                        embedder=_get_embedder(config, khub_ctx=khub),
                    )
                judge_assessment = next(
                    (
                        item
                        for item in list(judge_payload.get("items") or [])
                        if str(item.get("paper_id") or "") == str(paper.arxiv_id)
                    ),
                    None,
                )
                obsidian_path = paper_manager._create_obsidian_note(
                    paper,
                    summary,
                    topic,
                    judge_assessment=judge_assessment,
                    translated_abstract=translated_abstract,
                )
                if obsidian_path:
                    obsidian_paths.append(obsidian_path)
                emit("[green]완료[/green]")
                entry["steps"].append("Obsidian")

            entry["summary"] = summary[:200] if summary else ""
            entry["success"] = True
            if str((entry.get("sourceGuard") or {}).get("decision") or "") == "relink_to_canonical":
                entry["steps"].append("source-guard-relink")
            emit("  [bold green]✓ 완료[/bold green]")

        except Exception as e:
            entry["success"] = False
            entry["error"] = str(e)
            emit(f"  [red]오류: {e}[/red]")

        results.append(entry)

    if do_index and papers_to_index:
        emit(f"\n[bold]벡터 인덱싱 ({len(papers_to_index)}편)...[/bold]", end=" ")
        try:
            idx_results = _index_papers_inline(papers_to_index, khub, config)
            indexed_count = sum(1 for v in idx_results.values() if v.get("ok"))
            failed_count = len(papers_to_index) - indexed_count
            emit(f"[green]{indexed_count}편 완료[/green]", end="")
            if failed_count > 0:
                emit(f", [red]{failed_count}편 실패[/red]")
            else:
                emit()
            for r in results:
                idx_r = idx_results.get(r["arxiv_id"])
                if idx_r and idx_r.get("ok"):
                    r["steps"].append(f"인덱싱({idx_r['chunks']})")
                    sqlite_db.conn.execute("UPDATE papers SET indexed = 1 WHERE arxiv_id = ?", (r["arxiv_id"],))
                elif idx_r:
                    r["steps"].append("인덱싱실패")
                    log.warning("인덱싱 실패 %s: %s", r["arxiv_id"], idx_r.get("error", ""))
            sqlite_db.conn.commit()
        except Exception as e:
            log.exception("벡터 인덱싱 전체 실패")
            emit(f"[yellow]실패: {e}[/yellow]")

    success_count = sum(1 for r in results if r.get("success"))
    if as_json:
        report = {
            "schema": "knowledge-hub.paper.discover.result.v1",
            "status": "ok",
            "topic": topic,
            "discovered": len(discovered),
            "duplicates_skipped": skipped,
            "ingested": [
                {
                    "arxiv_id": r["arxiv_id"],
                    "title": r["title"],
                    "year": next((paper.year for paper in new_papers if paper.arxiv_id == r["arxiv_id"]), 0),
                    "citations": next((paper.citation_count for paper in new_papers if paper.arxiv_id == r["arxiv_id"]), 0),
                    "fields": next((paper.fields_of_study[:3] for paper in new_papers if paper.arxiv_id == r["arxiv_id"]), []),
                    "summary": r.get("summary", ""),
                    "pdf_downloaded": "PDF" in r.get("steps", []),
                }
                for r in results
                if r.get("success")
            ],
            "failed": [
                {
                    "arxiv_id": r["arxiv_id"],
                    "title": r["title"],
                    "error": r.get("error", "실패"),
                }
                for r in results
                if not r.get("success")
            ],
            "obsidian_notes_created": obsidian_paths,
            "message": f"{len(discovered)}개 발견, {success_count}개 수집, {skipped}개 중복 건너뜀, {len([r for r in results if not r.get('success')])}개 실패",
            "warnings": list(judge_payload.get("warnings") or []),
            "judge": judge_payload,
            "results": results,
        }
        click.echo(json.dumps(report, ensure_ascii=False, indent=2))
        return

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
            str(i),
            r["arxiv_id"],
            r["title"][:45],
            " → ".join(r.get("steps", [])),
            status,
        )

    console.print(table)

    console.print(f"\n[bold green]{success_count}/{len(results)}편 수집 완료[/bold green], {skipped}편 중복 건너뜀")

    if success_count > 0:
        if not do_index:
            console.print("[dim]khub index 로 벡터 인덱싱 | khub paper list 로 목록 확인[/dim]")
        else:
            console.print('[dim]khub paper list 로 목록 확인 | khub search "키워드" 로 검색[/dim]')
