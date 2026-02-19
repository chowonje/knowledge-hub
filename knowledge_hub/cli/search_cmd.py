"""
khub search / ask - 벡터 검색 및 RAG 질의 (논문 + 개념 통합 검색)
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

console = Console()


def _get_searcher(khub_ctx):
    """RAGSearcher 인스턴스 생성"""
    from knowledge_hub.core.database import VectorDatabase, SQLiteDatabase
    from knowledge_hub.providers.registry import get_llm, get_embedder
    from knowledge_hub.ai.rag import RAGSearcher

    config = khub_ctx.config

    embed_cfg = config.get_provider_config(config.embedding_provider)
    embedder = get_embedder(config.embedding_provider, model=config.embedding_model, **embed_cfg)

    summ_cfg = config.get_provider_config(config.summarization_provider)
    llm = get_llm(config.summarization_provider, model=config.summarization_model, **summ_cfg)

    vector_db = VectorDatabase(config.vector_db_path, config.collection_name)
    return RAGSearcher(embedder, vector_db, llm)


def _source_label(source_type: str) -> str:
    labels = {
        "paper": "[cyan]논문[/cyan]",
        "concept": "[magenta]개념[/magenta]",
        "vault": "[green]노트[/green]",
    }
    return labels.get(source_type, f"[dim]{source_type}[/dim]")


@click.command("search")
@click.argument("query")
@click.option("--top-k", "-k", default=10, help="결과 수")
@click.option("--source", "-s", default=None, help="소스 필터: concept, paper, vault")
@click.pass_context
def search(ctx, query, top_k, source):
    """벡터 유사도 검색 (논문 + 개념 통합)

    \b
    예시:
      khub search "attention mechanism"
      khub search "Transformer" -s concept     # 개념만 검색
      khub search "GAN" -s paper               # 논문만 검색
    """
    try:
        searcher = _get_searcher(ctx.obj["khub"])
    except Exception as e:
        console.print(f"[red]초기화 실패: {e}[/red]")
        return

    with console.status("검색 중..."):
        results = searcher.search(query, top_k=top_k, source_type=source)

    if not results:
        console.print("[yellow]검색 결과가 없습니다.[/yellow]")
        return

    console.print(f"\n[bold]'{query}' 검색 결과 ({len(results)}개):[/bold]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", width=3, style="dim")
    table.add_column("유형", width=6)
    table.add_column("제목", max_width=45)
    table.add_column("유사도", width=6, justify="right")
    table.add_column("상세", max_width=50)

    for i, r in enumerate(results, 1):
        title = r.metadata.get("title", "Untitled")
        src = r.metadata.get("source_type", "")
        score = f"{r.score:.3f}"

        detail = ""
        if src == "concept":
            related = r.metadata.get("related_concepts", "")
            if related:
                detail = f"관련: {related}"
        elif src == "paper":
            kw = r.metadata.get("keywords", "")
            field = r.metadata.get("field", "")
            parts = []
            if field:
                parts.append(field)
            if kw:
                parts.append(kw[:40])
            detail = " | ".join(parts)

        table.add_row(str(i), _source_label(src), title, score, detail[:50])

    console.print(table)

    console.print("\n[dim]khub ask \"질문\" 으로 RAG 답변 생성 가능[/dim]")


@click.command("ask")
@click.argument("question")
@click.option("--top-k", "-k", default=8, help="참고 문서 수")
@click.pass_context
def ask(ctx, question, top_k):
    """RAG 기반 질의응답 (논문 + 개념 지식 그래프 활용)

    \b
    예시:
      khub ask "Transformer의 핵심 아이디어는?"
      khub ask "Attention Mechanism과 CNN의 차이점은?"
    """
    try:
        searcher = _get_searcher(ctx.obj["khub"])
    except Exception as e:
        console.print(f"[red]초기화 실패: {e}[/red]")
        return

    with console.status("답변 생성 중..."):
        result = searcher.generate_answer(question, top_k=top_k)

    console.print(f"\n[bold cyan]Q: {question}[/bold cyan]\n")
    console.print(result["answer"])

    if result.get("sources"):
        console.print("\n[dim]참고 자료:[/dim]")
        for i, s in enumerate(result["sources"], 1):
            src_type = s.get("source_type", "")
            label = "개념" if src_type == "concept" else "논문" if src_type == "paper" else src_type
            console.print(f"  {i}. {s['title']} [{label}] (유사도: {s['score']:.2f})")
