"""
노트 관리 모듈

통합 노트 CRUD 및 검색
"""

from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from knowledge_hub.core.database import SQLiteDatabase

console = Console()


class NoteManager:
    """통합 노트 관리"""

    def __init__(self, sqlite_db: SQLiteDatabase):
        self.db = sqlite_db

    def list(
        self,
        source_type: Optional[str] = None,
        para_category: Optional[str] = None,
        limit: int = 30,
    ):
        """노트 목록 표시"""
        notes = self.db.list_notes(
            source_type=source_type, para_category=para_category, limit=limit
        )

        table = Table(title="노트 목록")
        table.add_column("ID", style="dim", max_width=30)
        table.add_column("제목", max_width=50)
        table.add_column("소스", style="cyan")
        table.add_column("PARA", style="magenta")
        table.add_column("태그")

        for n in notes:
            tags = self.db.get_note_tags(n["id"])
            table.add_row(
                n["id"][:30],
                n["title"][:50],
                n["source_type"],
                n.get("para_category") or "-",
                ", ".join(tags[:3]),
            )

        console.print(table)
        console.print(f"총 {len(notes)}개")

    def show(self, note_id: str):
        """노트 상세 보기"""
        note = self.db.get_note(note_id)
        if not note:
            console.print(f"[red]노트를 찾을 수 없습니다: {note_id}[/red]")
            return

        tags = self.db.get_note_tags(note_id)
        links = self.db.get_links(note_id)

        console.print(Panel(
            f"[bold]{note['title']}[/bold]\n\n"
            f"소스: {note['source_type']} | PARA: {note.get('para_category') or '-'}\n"
            f"태그: {', '.join(tags) if tags else '-'}\n"
            f"링크: {len(links)}개\n"
            f"파일: {note.get('file_path') or '-'}\n\n"
            f"---\n\n"
            f"{note.get('content', '')[:500]}{'...' if len(note.get('content', '')) > 500 else ''}",
            title=note_id,
        ))

    def search(self, query: str, limit: int = 20):
        """노트 텍스트 검색"""
        notes = self.db.search_notes(query, limit=limit)

        table = Table(title=f"검색: '{query}'")
        table.add_column("제목", max_width=50)
        table.add_column("소스", style="cyan")
        table.add_column("미리보기", max_width=60, style="dim")

        for n in notes:
            preview = (n.get("content") or "")[:60].replace("\n", " ")
            table.add_row(n["title"][:50], n["source_type"], preview)

        console.print(table)
        console.print(f"{len(notes)}개 결과")

    def tags(self):
        """태그 목록 표시"""
        tag_list = self.db.list_tags()

        table = Table(title="태그 목록")
        table.add_column("태그", style="bold")
        table.add_column("항목 수", justify="right", style="cyan")

        for t in tag_list:
            table.add_row(t["name"], str(t["count"]))

        console.print(table)
