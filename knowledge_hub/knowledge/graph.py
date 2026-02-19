"""
지식 그래프 관리

노트/논문 간의 연결 관계를 관리하고 시각화합니다.
"""

from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from knowledge_hub.core.database import SQLiteDatabase

console = Console()


class GraphManager:
    """지식 그래프 관리"""

    def __init__(self, sqlite_db: SQLiteDatabase):
        self.db = sqlite_db

    def show_stats(self):
        """그래프 통계 표시"""
        data = self.db.get_graph_data()
        nodes = data["nodes"]
        edges = data["edges"]

        # 유형별 집계
        type_counts = {}
        for n in nodes:
            t = n["type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        group_counts = {}
        for n in nodes:
            g = n["group"]
            group_counts[g] = group_counts.get(g, 0) + 1

        console.print("[bold]지식 그래프 통계[/bold]\n")
        console.print(f"  노드: {len(nodes)}개")
        console.print(f"  엣지: {len(edges)}개")

        if type_counts:
            console.print("\n  [cyan]소스별:[/cyan]")
            for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
                console.print(f"    {t}: {c}")

        if group_counts:
            console.print("\n  [magenta]PARA별:[/magenta]")
            for g, c in sorted(group_counts.items(), key=lambda x: -x[1]):
                console.print(f"    {g}: {c}")

    def show_connections(self, note_id: str):
        """특정 노트의 연결 관계 표시"""
        note = self.db.get_note(note_id)
        if not note:
            console.print(f"[red]노트를 찾을 수 없습니다: {note_id}[/red]")
            return

        links = self.db.get_links(note_id)

        tree = Tree(f"[bold]{note['title']}[/bold]")

        outgoing = [l for l in links if l["source_id"] == note_id]
        incoming = [l for l in links if l["target_id"] == note_id]

        if outgoing:
            out_branch = tree.add("[cyan]나가는 링크[/cyan]")
            for l in outgoing:
                target = self.db.get_note(l["target_id"])
                name = target["title"] if target else l["target_id"]
                out_branch.add(f"{name} ({l['link_type']})")

        if incoming:
            in_branch = tree.add("[green]들어오는 링크[/green]")
            for l in incoming:
                source = self.db.get_note(l["source_id"])
                name = source["title"] if source else l["source_id"]
                in_branch.add(f"{name} ({l['link_type']})")

        console.print(tree)

    def find_isolated(self, limit: int = 20):
        """연결이 없는 고립된 노트 찾기"""
        all_notes = self.db.list_notes(limit=1000)
        isolated = []

        for note in all_notes:
            links = self.db.get_links(note["id"])
            if not links:
                isolated.append(note)

        table = Table(title="고립된 노트 (연결 없음)")
        table.add_column("ID", style="dim", max_width=30)
        table.add_column("제목", max_width=50)
        table.add_column("소스", style="cyan")

        for n in isolated[:limit]:
            table.add_row(n["id"][:30], n["title"][:50], n["source_type"])

        console.print(table)
        console.print(f"총 {len(isolated)}개 고립 노트 (상위 {min(limit, len(isolated))}개 표시)")
