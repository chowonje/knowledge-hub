"""Read-only paper admin command runtime helpers."""

from __future__ import annotations

from typing import Any, Callable

from rich.markdown import Markdown
from rich.table import Table


def run_paper_list(
    *,
    khub: Any,
    field: str | None,
    limit: int,
    console: Any,
    sqlite_db_fn: Callable[..., Any],
) -> None:
    sqlite_db = sqlite_db_fn(khub.config, khub=khub)
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

    for paper in papers:
        notes = paper.get("notes") or ""
        has_summary = len(notes) > 30
        table.add_row(
            paper["arxiv_id"],
            paper["title"][:50],
            str(paper.get("year", "")),
            paper.get("field", "")[:20],
            "[green]O[/green]" if paper.get("pdf_path") else "-",
            "[green]O[/green]" if has_summary else "-",
            "[green]O[/green]" if paper.get("translated_path") else "-",
            "[green]O[/green]" if paper.get("indexed") else "-",
        )

    console.print(table)


def run_paper_info(
    *,
    khub: Any,
    arxiv_id: str,
    console: Any,
    sqlite_db_fn: Callable[..., Any],
    validate_arxiv_id_fn: Callable[[str], str],
    assess_summary_quality_fn: Callable[[str], dict[str, Any]],
) -> None:
    normalized_arxiv_id = validate_arxiv_id_fn(arxiv_id)
    sqlite_db = sqlite_db_fn(khub.config, khub=khub)
    paper = sqlite_db.get_paper(normalized_arxiv_id)

    if not paper:
        console.print(f"[red]논문을 찾을 수 없습니다: {normalized_arxiv_id}[/red]")
        return

    table = Table(title=f"논문 정보: {normalized_arxiv_id}")
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
    table.add_row("arXiv", f"https://arxiv.org/abs/{normalized_arxiv_id}")

    console.print(table)

    notes = paper.get("notes", "")
    if notes and len(notes) > 30:
        quality = assess_summary_quality_fn(notes)
        console.print(
            f"\n[bold]요약[/bold] [{quality['color']}]({quality['label']}, {quality['score']}점)[/{quality['color']}]"
        )
        if quality["reasons"]:
            console.print(f"[dim]문제점: {', '.join(quality['reasons'])}[/dim]")
        console.print()
        console.print(Markdown(notes))
        return

    console.print(
        "\n[yellow]요약이 없습니다. 'khub paper summarize {arxiv_id}' 로 생성하세요.[/yellow]".format(
            arxiv_id=normalized_arxiv_id
        )
    )


def run_paper_review(
    *,
    khub: Any,
    bad_only: bool,
    threshold: int,
    field: str | None,
    show_summary: bool,
    limit: int,
    console: Any,
    sqlite_db_fn: Callable[..., Any],
    assess_summary_quality_fn: Callable[[str], dict[str, Any]],
) -> None:
    sqlite_db = sqlite_db_fn(khub.config, khub=khub)
    papers = sqlite_db.list_papers(field=field, limit=999)

    if not papers:
        console.print("[yellow]수집된 논문이 없습니다.[/yellow]")
        return

    assessments: list[tuple[dict[str, Any], dict[str, Any]]] = []
    bad_papers = 0
    for paper in papers:
        quality = assess_summary_quality_fn(paper.get("notes", ""))
        assessments.append((paper, quality))
        if quality["score"] < threshold:
            bad_papers += 1

    if bad_only:
        assessments = [(paper, quality) for paper, quality in assessments if quality["score"] < threshold]

    assessments.sort(key=lambda item: item[1]["score"])

    if not assessments:
        console.print("[green]모든 요약이 기준 이상입니다.[/green]")
        return

    assessments = assessments[:limit]

    total = len(papers)
    good_count = total - bad_papers

    console.print("\n[bold]논문 요약 품질 리뷰[/bold]")
    console.print(f"  전체: {total}편 | 우수/보통: {good_count}편 | 미흡/형편없음: {bad_papers}편\n")

    table = Table(title=f"요약 품질 ({len(assessments)}편)")
    table.add_column("arXiv ID", style="cyan", width=14)
    table.add_column("제목", max_width=40)
    table.add_column("점수", width=5, justify="right")
    table.add_column("등급", width=8)
    table.add_column("문제점", max_width=30)
    table.add_column("요약길이", width=8, justify="right")

    for paper, quality in assessments:
        notes_len = len((paper.get("notes") or "").strip())
        table.add_row(
            paper["arxiv_id"],
            paper["title"][:40],
            str(quality["score"]),
            f"[{quality['color']}]{quality['label']}[/{quality['color']}]",
            ", ".join(quality["reasons"][:2]) if quality["reasons"] else "-",
            f"{notes_len:,}자",
        )

    console.print(table)

    if show_summary:
        console.print("\n[bold]요약 미리보기:[/bold]\n")
        for paper, quality in assessments[:10]:
            notes = (paper.get("notes") or "").strip()
            preview = notes[:200] + "..." if len(notes) > 200 else notes
            console.print(
                f"[cyan]{paper['arxiv_id']}[/cyan] "
                f"[{quality['color']}]{quality['label']}[/{quality['color']}] "
                f"{paper['title'][:50]}"
            )
            if preview:
                console.print(f"  [dim]{preview}[/dim]")
            console.print()

    if bad_papers > 0:
        console.print("\n[bold yellow]재요약 가이드:[/bold yellow]")
        console.print("  개별: khub paper summarize <arxiv_id> -p openai -m gpt-4o")
        console.print("  일괄: khub paper summarize-all --bad-only -p openai -m gpt-4o")
        console.print("  전체: khub paper summarize-all --resummary")
