"""
PARA 분류 체계 관리

Projects, Areas, Resources, Archives 기반 지식 분류
"""

from typing import Optional
from rich.console import Console
from rich.table import Table

from knowledge_hub.core.database import SQLiteDatabase

console = Console()


class ParaManager:
    """PARA 분류 관리"""

    def __init__(self, sqlite_db: SQLiteDatabase):
        self.db = sqlite_db

    def list_categories(self):
        """PARA 카테고리 목록과 통계 표시"""
        categories = self.db.list_para_categories()
        stats = self.db.get_para_stats()

        table = Table(title="PARA 분류 체계")
        table.add_column("유형", style="bold")
        table.add_column("이름")
        table.add_column("설명")
        table.add_column("항목 수", justify="right", style="cyan")

        for cat in categories:
            count = stats.get(cat["type"], 0)
            table.add_row(
                cat["type"].upper(),
                cat["name"],
                cat["description"] or "",
                str(count),
            )

        console.print(table)

    def list_items(self, category_type: str, limit: int = 50):
        """특정 PARA 카테고리의 항목 목록"""
        notes = self.db.list_notes(para_category=category_type, limit=limit)

        table = Table(title=f"{category_type.upper()} 항목")
        table.add_column("제목")
        table.add_column("소스", style="dim")
        table.add_column("업데이트", style="dim")

        for n in notes:
            table.add_row(n["title"][:60], n["source_type"], n.get("updated_at", ""))

        console.print(table)
        console.print(f"총 {len(notes)}개")

    def move_item(self, note_id: str, target_category: str):
        """항목을 다른 PARA 카테고리로 이동"""
        valid = ["project", "area", "resource", "archive"]
        if target_category not in valid:
            console.print(f"[red]유효하지 않은 카테고리입니다. 선택: {', '.join(valid)}[/red]")
            return

        note = self.db.get_note(note_id)
        if not note:
            console.print(f"[red]항목을 찾을 수 없습니다: {note_id}[/red]")
            return

        self.db.conn.execute(
            "UPDATE notes SET para_category = ? WHERE id = ?", (target_category, note_id)
        )
        self.db.conn.commit()
        console.print(f"[green]{note['title']} → {target_category.upper()} 이동 완료[/green]")
