"""
khub notebook - Google NotebookLM 연동 명령어

  khub notebook create <topic>  주제별 노트북 생성
  khub notebook sync            논문을 NotebookLM에 동기화
  khub notebook list            노트북 목록 조회
  khub notebook study-pack      로컬 스터디 팩 생성 (NotebookLM 없이도 사용 가능)
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


def _get_client(ctx):
    from knowledge_hub.integrations.notebooklm import get_notebooklm_client
    config = ctx.obj["khub"].config
    client = get_notebooklm_client(config)
    if not client:
        console.print("[red]NotebookLM 설정이 필요합니다.[/red]")
        console.print("[dim]GOOGLE_CLOUD_PROJECT 환경변수를 설정하거나")
        console.print("config에 notebooklm.project_number를 추가하세요.[/dim]")
        console.print()
        console.print("[dim]또는 khub notebook study-pack 으로 로컬 스터디 팩을 생성할 수 있습니다.[/dim]")
        return None
    return client


@click.group("notebook")
def notebook_group():
    """NotebookLM 연동 및 스터디 팩 생성"""
    pass


@notebook_group.command("create")
@click.argument("topic")
@click.option("--description", "-d", default="", help="노트북 설명")
@click.pass_context
def notebook_create(ctx, topic, description):
    """주제별 NotebookLM 노트북 생성"""
    client = _get_client(ctx)
    if not client:
        return

    with console.status(f"노트북 생성 중: {topic}..."):
        result = client.create_notebook(topic, description)

    notebook_id = result.get("name", "").split("/")[-1]
    console.print(f"[green]노트북 생성 완료[/green]")
    console.print(f"  ID: {notebook_id}")
    console.print(f"  제목: {topic}")
    console.print(f"[dim]khub notebook sync --notebook {notebook_id} 로 논문을 추가하세요.[/dim]")


@notebook_group.command("list")
@click.pass_context
def notebook_list(ctx):
    """NotebookLM 노트북 목록 조회"""
    client = _get_client(ctx)
    if not client:
        return

    with console.status("노트북 목록 조회 중..."):
        notebooks = client.list_notebooks()

    if not notebooks:
        console.print("[yellow]생성된 노트북이 없습니다.[/yellow]")
        return

    table = Table(title="NotebookLM 노트북")
    table.add_column("ID", style="cyan")
    table.add_column("제목")
    table.add_column("생성일")

    for nb in notebooks:
        name = nb.get("name", "")
        nb_id = name.split("/")[-1] if name else ""
        table.add_row(
            nb_id,
            nb.get("displayName", ""),
            nb.get("createTime", "")[:10],
        )

    console.print(table)


@notebook_group.command("sync")
@click.option("--notebook", "-nb", required=True, help="대상 노트북 ID")
@click.option("--field", "-f", default=None, help="분야 필터")
@click.option("--limit", "-n", default=0, help="최대 동기화 수 (0=전체)")
@click.pass_context
def notebook_sync(ctx, notebook, field, limit):
    """수집된 논문을 NotebookLM 노트북에 동기화"""
    client = _get_client(ctx)
    if not client:
        return

    config = ctx.obj["khub"].config
    from knowledge_hub.core.database import SQLiteDatabase

    sqlite_db = SQLiteDatabase(config.sqlite_path)
    papers = sqlite_db.list_papers(field=field, limit=limit if limit > 0 else 999)

    if not papers:
        console.print("[yellow]동기화할 논문이 없습니다.[/yellow]")
        return

    console.print(f"[bold]{len(papers)}편 논문을 노트북 [{notebook}]에 동기화 중...[/bold]\n")

    success = 0
    for idx, paper in enumerate(papers, 1):
        console.print(f"  [{idx}/{len(papers)}] {paper['title'][:50]}...", end=" ")
        if client.sync_paper(notebook, paper):
            success += 1
            console.print("[green]OK[/green]")
        else:
            console.print("[red]FAIL[/red]")

    console.print(f"\n[bold green]{success}/{len(papers)}편 동기화 완료[/bold green]")


@notebook_group.command("study-pack")
@click.option("--field", "-f", default=None, help="분야 필터")
@click.option("--limit", "-n", default=0, help="최대 생성 수 (0=전체)")
@click.option("--output", "-o", default=None, help="출력 디렉토리")
@click.pass_context
def notebook_study_pack(ctx, field, limit, output):
    """로컬 스터디 팩 생성 (NotebookLM 없이도 사용 가능)

    각 논문별로 학습에 필요한 정보를 구조화된 마크다운으로 정리합니다.
    """
    config = ctx.obj["khub"].config
    from knowledge_hub.core.database import SQLiteDatabase
    from knowledge_hub.integrations.notebooklm import generate_study_pack

    sqlite_db = SQLiteDatabase(config.sqlite_path)
    papers = sqlite_db.list_papers(field=field, limit=limit if limit > 0 else 999)

    if not papers:
        console.print("[yellow]논문이 없습니다.[/yellow]")
        return

    output_dir = output or str(Path(config.papers_dir) / "study_packs")
    console.print(f"[bold]{len(papers)}편 스터디 팩 생성 중...[/bold]")
    console.print(f"[dim]출력: {output_dir}[/dim]\n")

    success = 0
    for idx, paper in enumerate(papers, 1):
        console.print(f"  [{idx}/{len(papers)}] {paper['title'][:55]}...", end=" ")
        try:
            generate_study_pack(paper, output_dir)
            success += 1
            console.print("[green]OK[/green]")
        except Exception as e:
            console.print(f"[red]{e}[/red]")

    console.print(f"\n[bold green]{success}/{len(papers)}편 스터디 팩 생성 완료[/bold green]")
    console.print(f"[dim]{output_dir} 에서 확인[/dim]")
