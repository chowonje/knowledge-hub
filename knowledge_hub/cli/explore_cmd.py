"""
khub explore — 저자 검색, 인용 네트워크, 참고문헌, 배치 조회 등
               학술 메타데이터 탐색 명령 그룹
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

console = Console()


@click.group("explore")
def explore_group():
    """학술 메타데이터 탐색 (저자·인용·참고문헌·네트워크)"""
    pass


# ────────────────────────────────────
#  저자 검색
# ────────────────────────────────────

@explore_group.command("author")
@click.argument("query")
@click.option("--limit", "-n", default=10, help="결과 수")
def author_search(query, limit):
    """저자 이름/소속으로 검색

    \b
    예시:
      khub explore author "Yann LeCun"
      khub explore author "Stanford NLP" -n 20
    """
    from knowledge_hub.papers.discoverer import search_authors

    with console.status("[cyan]저자 검색 중...[/cyan]"):
        authors = search_authors(query, limit=limit)

    if not authors:
        console.print("[yellow]검색 결과가 없습니다.[/yellow]")
        return

    table = Table(title=f'저자 검색: "{query}" ({len(authors)}명)')
    table.add_column("#", width=3, style="dim")
    table.add_column("이름", style="cyan", min_width=20)
    table.add_column("소속", max_width=35)
    table.add_column("논문수", justify="right", width=7)
    table.add_column("인용수", justify="right", width=9)
    table.add_column("h-index", justify="right", width=7)
    table.add_column("ID", style="dim", width=12)

    for i, a in enumerate(authors, 1):
        affil = ", ".join(a.affiliations[:2]) if a.affiliations else "-"
        table.add_row(
            str(i), a.name, affil[:35],
            f"{a.paper_count:,}", f"{a.citation_count:,}", str(a.h_index),
            a.author_id,
        )

    console.print(table)
    console.print("\n[dim]저자의 논문 조회: khub explore author-papers <AUTHOR_ID>[/dim]")


# ────────────────────────────────────
#  저자별 논문
# ────────────────────────────────────

@explore_group.command("author-papers")
@click.argument("author_id")
@click.option("--limit", "-n", default=20, help="결과 수")
@click.option("--ingest", is_flag=True, help="결과를 SQLite에 저장 (discover 파이프라인 없이)")
@click.pass_context
def author_papers(ctx, author_id, limit, ingest):
    """특정 저자의 논문 목록 조회

    \b
    예시:
      khub explore author-papers 1741101  -n 30
      khub explore author-papers 1741101 --ingest   # DB에 저장
    """
    from knowledge_hub.papers.discoverer import get_author_papers

    with console.status("[cyan]저자 논문 조회 중...[/cyan]"):
        author, papers = get_author_papers(author_id, limit=limit)

    if author:
        affil = ", ".join(author.affiliations[:2]) if author.affiliations else "-"
        console.print(Panel.fit(
            f"[bold]{author.name}[/bold]\n"
            f"소속: {affil}\n"
            f"논문: {author.paper_count:,} | 인용: {author.citation_count:,} | h-index: {author.h_index}",
            title="저자 정보", border_style="cyan",
        ))

    if not papers:
        console.print("[yellow]논문이 없습니다.[/yellow]")
        return

    table = Table(title=f"논문 목록 ({len(papers)}편)")
    table.add_column("#", width=3, style="dim")
    table.add_column("arXiv", style="cyan", width=13)
    table.add_column("제목", max_width=55)
    table.add_column("연도", width=5)
    table.add_column("인용", justify="right", width=7)
    table.add_column("분야", max_width=20, style="magenta")

    for i, p in enumerate(papers, 1):
        fields = ", ".join(p.fields_of_study[:2]) if p.fields_of_study else ""
        table.add_row(
            str(i), p.arxiv_id or "-", p.title[:55],
            str(p.year), f"{p.citation_count:,}", fields,
        )

    console.print(table)

    if ingest:
        _ingest_papers_to_sqlite(ctx, papers)


# ────────────────────────────────────
#  논문 상세 정보
# ────────────────────────────────────

@explore_group.command("paper")
@click.argument("paper_id")
def paper_detail(paper_id):
    """논문 상세 정보 조회 (abstract, 메타데이터)

    \b
    PAPER_ID는 arXiv ID, DOI, 또는 Semantic Scholar ID
    예시:
      khub explore paper 2301.12345
      khub explore paper DOI:10.1234/example
    """
    from knowledge_hub.papers.discoverer import get_paper_detail

    with console.status("[cyan]논문 조회 중...[/cyan]"):
        data = get_paper_detail(paper_id)

    if not data:
        console.print(f"[yellow]논문 '{paper_id}'를 찾을 수 없습니다.[/yellow]")
        return

    title = data.get("title", "Unknown")
    authors = ", ".join(a.get("name", "") for a in (data.get("authors") or [])[:8])
    year = data.get("year", "?")
    citations = data.get("citationCount", 0)
    references = data.get("referenceCount", 0)
    influential = data.get("influentialCitationCount", 0)
    venue = data.get("venue", "-")
    fields = ", ".join(data.get("fieldsOfStudy") or [])
    arxiv_id = (data.get("externalIds") or {}).get("ArXiv", "")
    doi = (data.get("externalIds") or {}).get("DOI", "")
    abstract = data.get("abstract", "")

    lines = [
        f"[bold]{title}[/bold]",
        f"저자: {authors}",
        f"연도: {year} | Venue: {venue}",
        f"분야: {fields or '-'}",
        f"인용: {citations:,} (influential: {influential:,}) | 참고문헌: {references:,}",
    ]
    if arxiv_id:
        lines.append(f"arXiv: [link=https://arxiv.org/abs/{arxiv_id}]{arxiv_id}[/link]")
    if doi:
        lines.append(f"DOI: {doi}")

    console.print(Panel("\n".join(lines), border_style="cyan"))

    if abstract:
        console.print(Panel(abstract, title="Abstract", border_style="dim"))

    console.print("\n[dim]인용 조회: khub explore citations " + paper_id + "[/dim]")
    console.print("[dim]참고문헌:  khub explore references " + paper_id + "[/dim]")
    console.print("[dim]네트워크:  khub explore network " + paper_id + "[/dim]")


# ────────────────────────────────────
#  피인용 논문 (citations)
# ────────────────────────────────────

@explore_group.command("citations")
@click.argument("paper_id")
@click.option("--limit", "-n", default=20, help="결과 수")
@click.option("--ingest", is_flag=True, help="결과를 SQLite에 저장")
@click.pass_context
def citations(ctx, paper_id, limit, ingest):
    """이 논문을 인용한 논문들 조회

    \b
    예시:
      khub explore citations 2005.14165          # GPT-3를 인용한 논문
      khub explore citations 2005.14165 -n 50
    """
    from knowledge_hub.papers.discoverer import get_paper_citations, get_paper_detail

    detail = get_paper_detail(paper_id)
    if detail:
        console.print(f'[bold]{detail.get("title", "")}[/bold] — 인용: {detail.get("citationCount", 0):,}회')

    with console.status("[cyan]피인용 논문 조회 중...[/cyan]"):
        _, papers = get_paper_citations(paper_id, limit=limit)

    if not papers:
        console.print("[yellow]피인용 논문이 없습니다.[/yellow]")
        return

    _show_paper_table(papers, f"피인용 논문 ({len(papers)}편)")

    if ingest:
        _ingest_papers_to_sqlite(ctx, papers)


# ────────────────────────────────────
#  참고문헌 (references)
# ────────────────────────────────────

@explore_group.command("references")
@click.argument("paper_id")
@click.option("--limit", "-n", default=20, help="결과 수")
@click.option("--ingest", is_flag=True, help="결과를 SQLite에 저장")
@click.pass_context
def references(ctx, paper_id, limit, ingest):
    """이 논문이 참고한 논문들 조회

    \b
    예시:
      khub explore references 2005.14165
    """
    from knowledge_hub.papers.discoverer import get_paper_references, get_paper_detail

    detail = get_paper_detail(paper_id)
    if detail:
        console.print(f'[bold]{detail.get("title", "")}[/bold] — 참고문헌: {detail.get("referenceCount", 0):,}편')

    with console.status("[cyan]참고문헌 조회 중...[/cyan]"):
        _, papers = get_paper_references(paper_id, limit=limit)

    if not papers:
        console.print("[yellow]참고문헌이 없습니다.[/yellow]")
        return

    _show_paper_table(papers, f"참고문헌 ({len(papers)}편)")

    if ingest:
        _ingest_papers_to_sqlite(ctx, papers)


# ────────────────────────────────────
#  인용 네트워크 분석
# ────────────────────────────────────

@explore_group.command("network")
@click.argument("paper_id")
@click.option("--depth", "-d", default=1, type=click.IntRange(1, 2), help="분석 깊이 (1 or 2)")
@click.option("--citations-limit", default=10, help="인용 논문 최대 수")
@click.option("--references-limit", default=10, help="참고문헌 최대 수")
@click.option("--json", "as_json", is_flag=True, help="JSON으로 출력")
def network(paper_id, depth, citations_limit, references_limit, as_json):
    """인용 네트워크 분석 (시각화)

    \b
    예시:
      khub explore network 2005.14165                  # GPT-3 네트워크
      khub explore network 2005.14165 -d 2             # 2-depth 분석
      khub explore network 1706.03762 --json           # Attention 논문, JSON 출력
    """
    from knowledge_hub.papers.discoverer import analyze_citation_network

    with console.status(f"[cyan]인용 네트워크 분석 중 (depth={depth})...[/cyan]"):
        result = analyze_citation_network(
            paper_id, depth=depth,
            citations_limit=citations_limit,
            references_limit=references_limit,
        )

    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
        return

    if as_json:
        console.print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    root = result["root"]
    console.print(Panel.fit(
        f"[bold]{root['title']}[/bold]\n"
        f"인용: {root['citation_count']:,} | 참고문헌: {root['reference_count']:,}",
        title="네트워크 중심", border_style="cyan",
    ))

    tree = Tree(f"[bold cyan]{root['title'][:60]}[/bold cyan]")

    cite_branch = tree.add(f"[green]← 이 논문을 인용 ({len(result['citations'])}편)[/green]")
    for p in result["citations"][:15]:
        label = f"{p['title'][:50]} ({p['year']}) [dim]인용:{p['citations']:,}[/dim]"
        node = cite_branch.add(label)
        if depth >= 2 and p.get("arxiv_id"):
            sub = result.get("citations_of_citations", {}).get(p["arxiv_id"], [])
            for sp in sub[:3]:
                node.add(f"[dim]{sp['title'][:40]} ({sp['year']})[/dim]")

    ref_branch = tree.add(f"[blue]→ 이 논문이 참고 ({len(result['references'])}편)[/blue]")
    for p in result["references"][:15]:
        ref_branch.add(f"{p['title'][:50]} ({p['year']}) [dim]인용:{p['citations']:,}[/dim]")

    console.print(tree)

    if result.get("citation_year_distribution"):
        console.print("\n[bold]인용 연도 분포:[/bold]")
        dist = result["citation_year_distribution"]
        max_count = max(dist.values()) if dist else 1
        for year, count in dist.items():
            bar_len = int(count / max_count * 30)
            bar = "█" * bar_len
            console.print(f"  {year} │ {bar} {count}")

    if result.get("top_fields"):
        console.print("\n[bold]주요 분야:[/bold]")
        for field, count in list(result["top_fields"].items())[:8]:
            console.print(f"  {field}: {count}")


# ────────────────────────────────────
#  배치 조회
# ────────────────────────────────────

@explore_group.command("batch")
@click.argument("paper_ids", nargs=-1, required=True)
@click.option("--ingest", is_flag=True, help="결과를 SQLite에 저장")
@click.pass_context
def batch_lookup(ctx, paper_ids, ingest):
    """복수 논문 일괄 조회

    \b
    예시:
      khub explore batch 2005.14165 1706.03762 2301.00234
    """
    from knowledge_hub.papers.discoverer import get_papers_batch

    ids = list(paper_ids)
    with console.status(f"[cyan]{len(ids)}편 일괄 조회 중...[/cyan]"):
        results = get_papers_batch(ids)

    if not results:
        console.print("[yellow]조회 결과가 없습니다.[/yellow]")
        return

    table = Table(title=f"배치 조회 결과 ({len(results)}편)")
    table.add_column("#", width=3, style="dim")
    table.add_column("arXiv", style="cyan", width=13)
    table.add_column("제목", max_width=50)
    table.add_column("연도", width=5)
    table.add_column("인용", justify="right", width=8)
    table.add_column("분야", max_width=20, style="magenta")

    discovered = []
    for i, p in enumerate(results, 1):
        arxiv_id = ""
        ext = p.get("externalIds") or {}
        if ext.get("ArXiv"):
            arxiv_id = ext["ArXiv"]
        fields = ", ".join((p.get("fieldsOfStudy") or [])[:2])
        table.add_row(
            str(i), arxiv_id or "-",
            (p.get("title") or "")[:50], str(p.get("year", "")),
            f"{p.get('citationCount', 0):,}", fields,
        )
        if ingest and arxiv_id:
            from knowledge_hub.papers.discoverer import DiscoveredPaper
            authors_list = p.get("authors") or []
            authors_str = ", ".join(a.get("name", "") for a in authors_list[:5])
            discovered.append(DiscoveredPaper(
                arxiv_id=arxiv_id, title=p.get("title", ""),
                authors=authors_str, year=p.get("year") or 0,
                abstract=p.get("abstract") or "",
                citation_count=p.get("citationCount") or 0,
                fields_of_study=p.get("fieldsOfStudy") or [],
                source="semantic_scholar",
            ))

    console.print(table)

    if ingest and discovered:
        _ingest_papers_to_sqlite(ctx, discovered)


# ────────────────────────────────────
#  공통 유틸
# ────────────────────────────────────

def _show_paper_table(papers, title: str):
    table = Table(title=title)
    table.add_column("#", width=3, style="dim")
    table.add_column("arXiv", style="cyan", width=13)
    table.add_column("제목", max_width=55)
    table.add_column("연도", width=5)
    table.add_column("인용", justify="right", width=7)
    table.add_column("분야", max_width=20, style="magenta")

    for i, p in enumerate(papers, 1):
        fields = ", ".join(p.fields_of_study[:2]) if p.fields_of_study else ""
        table.add_row(
            str(i), p.arxiv_id or "-", p.title[:55],
            str(p.year), f"{p.citation_count:,}", fields,
        )
    console.print(table)


def _ingest_papers_to_sqlite(ctx, papers):
    """탐색 결과를 SQLite에 간단히 저장"""
    try:
        config = ctx.obj["khub"].config
        from knowledge_hub.core.database import SQLiteDatabase
        sqlite_db = SQLiteDatabase(config.sqlite_path)

        added = 0
        for p in papers:
            if not p.arxiv_id:
                continue
            existing = sqlite_db.get_paper(p.arxiv_id)
            if existing:
                continue
            importance = 5 if p.citation_count >= 500 else 4 if p.citation_count >= 100 else 3 if p.citation_count >= 20 else 2
            sqlite_db.upsert_paper({
                "arxiv_id": p.arxiv_id, "title": p.title,
                "authors": p.authors, "year": p.year,
                "field": ", ".join(p.fields_of_study[:3]),
                "importance": importance,
                "notes": f"citations: {p.citation_count}",
            })
            added += 1

        console.print(f"\n[green]{added}편 SQLite에 저장됨[/green] (중복 제외)")
        if added > 0:
            console.print("[dim]전체 파이프라인: khub discover 또는 khub paper download/summarize/embed[/dim]")
    except Exception as e:
        console.print(f"[red]저장 실패: {e}[/red]")
