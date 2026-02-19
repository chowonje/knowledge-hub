"""
khub graph - 지식 그래프 쿼리 및 통계

  khub graph stats                          전체 통계
  khub graph concept "Transformer"          개념 상세 조회
  khub graph paper 2501.06322               논문의 개념/관계 조회
  khub graph path "Transformer" "RLHF"      두 개념 간 연결 경로
"""

from __future__ import annotations

import re
from collections import deque

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


@click.group("graph")
def graph_group():
    """지식 그래프 쿼리 및 통계"""
    pass


@graph_group.command("stats")
@click.pass_context
def graph_stats(ctx):
    """지식 그래프 통계"""
    from knowledge_hub.core.database import SQLiteDatabase

    config = ctx.obj["khub"].config
    db = SQLiteDatabase(config.sqlite_path)
    stats = db.get_kg_stats()

    panel_text = (
        f"[bold]개념:[/bold] {stats['concepts']}개"
        f"  |  [bold]별칭:[/bold] {stats['aliases']}개"
        f"  |  [bold]논문:[/bold] {stats['papers']}개"
        f"\n[bold]관계:[/bold] {stats['relations']}개"
        f"  |  [bold]고립 개념:[/bold] {stats['isolated_concepts']}개"
    )
    console.print(Panel(panel_text, title="Knowledge Graph", border_style="cyan"))

    if stats["relation_types"]:
        table = Table(title="관계 유형별 수")
        table.add_column("관계", style="cyan")
        table.add_column("수", justify="right")
        for rel, cnt in stats["relation_types"].items():
            table.add_row(rel, str(cnt))
        console.print(table)

    density = 0
    if stats["concepts"] > 1:
        max_edges = stats["concepts"] * (stats["concepts"] - 1) / 2
        concept_rels = stats["relation_types"].get("concept_related_to", 0)
        density = concept_rels / max_edges if max_edges > 0 else 0
    console.print(f"\n[dim]개념 그래프 밀도: {density:.4f}[/dim]")


@graph_group.command("concept")
@click.argument("name")
@click.pass_context
def graph_concept(ctx, name):
    """개념 상세 조회 — 관련 논문 + 관련 개념 + 근거"""
    from knowledge_hub.core.database import SQLiteDatabase

    config = ctx.obj["khub"].config
    db = SQLiteDatabase(config.sqlite_path)

    canonical = db.resolve_concept(name)
    if not canonical:
        cid = re.sub(r'\s+', '_', name.strip()).lower()
        concept = db.get_concept(cid)
        if not concept:
            console.print(f"[red]개념을 찾을 수 없습니다: {name}[/red]")
            console.print("[dim]khub paper sync-keywords 또는 build-concepts를 먼저 실행하세요.[/dim]")
            return
        canonical = concept["canonical_name"]
        cid = concept["id"]
    else:
        cid = re.sub(r'\s+', '_', canonical.strip()).lower()
        concept = db.get_concept(cid)

    if not concept:
        console.print(f"[red]개념 데이터가 없습니다: {cid}[/red]")
        return

    console.print(Panel(
        f"[bold]{canonical}[/bold]\n\n{concept.get('description') or '[설명 없음]'}",
        title="개념 정보",
        border_style="magenta",
    ))

    aliases = db.get_aliases(cid)
    if aliases:
        console.print(f"[dim]별칭: {', '.join(aliases)}[/dim]\n")

    # 관련 논문
    papers = db.get_concept_papers(cid)
    if papers:
        table = Table(title=f"관련 논문 ({len(papers)}편)")
        table.add_column("arXiv", style="cyan", width=14)
        table.add_column("제목", max_width=45)
        table.add_column("신뢰도", width=6, justify="right")
        table.add_column("근거", max_width=60)

        for p in papers:
            conf = f"{p.get('confidence', 0):.2f}"
            evidence = (p.get("evidence_text") or "")[:60]
            table.add_row(p["arxiv_id"], p["title"][:45], conf, evidence)
        console.print(table)

    # 관련 개념
    related = db.get_related_concepts(cid)
    if related:
        table = Table(title=f"관련 개념 ({len(related)}개)")
        table.add_column("개념", style="magenta")
        table.add_column("신뢰도", width=6, justify="right")
        for r in related:
            table.add_row(r["canonical_name"], f"{r.get('confidence', 0):.2f}")
        console.print(table)

    if not papers and not related:
        console.print("[yellow]연결된 관계가 없습니다.[/yellow]")


@graph_group.command("paper")
@click.argument("arxiv_id")
@click.pass_context
def graph_paper(ctx, arxiv_id):
    """논문의 개념/관계 조회"""
    from knowledge_hub.core.database import SQLiteDatabase

    config = ctx.obj["khub"].config
    db = SQLiteDatabase(config.sqlite_path)

    paper = db.get_paper(arxiv_id)
    if not paper:
        console.print(f"[red]논문을 찾을 수 없습니다: {arxiv_id}[/red]")
        return

    console.print(Panel(
        f"[bold]{paper['title']}[/bold]\n"
        f"arXiv: {arxiv_id} | 연도: {paper.get('year', '?')} | 분야: {paper.get('field', '?')}",
        title="논문 정보",
        border_style="cyan",
    ))

    concepts = db.get_paper_concepts(arxiv_id)
    if concepts:
        table = Table(title=f"사용 개념 ({len(concepts)}개)")
        table.add_column("개념", style="magenta")
        table.add_column("신뢰도", width=6, justify="right")
        table.add_column("근거", max_width=65)

        for c in concepts:
            conf = f"{c.get('confidence', 0):.2f}"
            evidence = (c.get("evidence_text") or "")[:65]
            table.add_row(c["canonical_name"], conf, evidence)
        console.print(table)
    else:
        console.print("[yellow]연결된 개념이 없습니다. khub paper sync-keywords를 실행하세요.[/yellow]")


@graph_group.command("path")
@click.argument("source")
@click.argument("target")
@click.option("--max-depth", "-d", default=4, help="최대 탐색 깊이")
@click.pass_context
def graph_path(ctx, source, target, max_depth):
    """두 개념 간 연결 경로 탐색 (BFS)"""
    from knowledge_hub.core.database import SQLiteDatabase

    config = ctx.obj["khub"].config
    db = SQLiteDatabase(config.sqlite_path)

    src_canonical = db.resolve_concept(source)
    tgt_canonical = db.resolve_concept(target)

    if not src_canonical:
        console.print(f"[red]출발 개념을 찾을 수 없습니다: {source}[/red]")
        return
    if not tgt_canonical:
        console.print(f"[red]도착 개념을 찾을 수 없습니다: {target}[/red]")
        return

    src_id = re.sub(r'\s+', '_', src_canonical.strip()).lower()
    tgt_id = re.sub(r'\s+', '_', tgt_canonical.strip()).lower()

    if src_id == tgt_id:
        console.print(f"[yellow]같은 개념입니다: {src_canonical}[/yellow]")
        return

    # BFS for shortest path through concept_related_to relations
    queue: deque[list[str]] = deque([[src_id]])
    visited = {src_id}

    found_path: list[str] | None = None
    while queue:
        path = queue.popleft()
        if len(path) > max_depth + 1:
            break
        current = path[-1]

        related = db.get_related_concepts(current)
        for r in related:
            neighbor_id = r["id"]
            if neighbor_id == tgt_id:
                found_path = path + [neighbor_id]
                break
            if neighbor_id not in visited:
                visited.add(neighbor_id)
                queue.append(path + [neighbor_id])

        if found_path:
            break

    if not found_path:
        console.print(f"[yellow]{src_canonical} → {tgt_canonical}: 깊이 {max_depth} 이내 경로 없음[/yellow]")
        console.print("[dim]--max-depth를 늘리거나, build-concepts로 관계를 추가하세요.[/dim]")
        return

    # Build display
    id_to_name: dict[str, str] = {}
    for nid in found_path:
        c = db.get_concept(nid)
        id_to_name[nid] = c["canonical_name"] if c else nid

    path_str = " → ".join(f"[magenta]{id_to_name[nid]}[/magenta]" for nid in found_path)
    console.print(f"\n[bold]경로 ({len(found_path)-1}홉):[/bold]  {path_str}")
