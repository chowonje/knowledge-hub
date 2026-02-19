"""
Vault 인덱서

Obsidian vault를 파싱하고 벡터 DB + SQLite에 인덱싱합니다.
"""

from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from knowledge_hub.core.config import Config
from knowledge_hub.core.database import VectorDatabase, SQLiteDatabase
from knowledge_hub.core.embeddings import OllamaEmbedder
from knowledge_hub.core.models import SourceType
from knowledge_hub.vault.parser import ObsidianParser

console = Console()


class VaultIndexer:
    """Obsidian vault 인덱싱"""

    def __init__(
        self,
        config: Config,
        vector_db: VectorDatabase,
        sqlite_db: SQLiteDatabase,
        embedder: OllamaEmbedder,
    ):
        self.config = config
        self.vector_db = vector_db
        self.sqlite_db = sqlite_db
        self.embedder = embedder

    def index(self, vault_path: Optional[str] = None, clear: bool = False):
        """vault 전체를 인덱싱"""
        path = vault_path or self.config.vault_path
        if not path:
            console.print("[red]vault 경로가 설정되지 않았습니다. config.yaml을 확인하세요.[/red]")
            return

        parser = ObsidianParser(
            vault_path=path,
            exclude_folders=self.config.vault_excludes,
        )

        if clear:
            console.print("[yellow]벡터 DB 초기화 중...[/yellow]")
            self.vector_db.clear_collection()

        console.print(f"[cyan]vault 파싱 중: {path}[/cyan]")
        documents = parser.parse_vault()
        console.print(f"[green]{len(documents)}개 문서 발견[/green]")

        if not documents:
            console.print("[yellow]인덱싱할 문서가 없습니다.[/yellow]")
            return

        all_chunks = []
        for doc in documents:
            chunks = ObsidianParser.chunk_document(
                doc,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
            all_chunks.extend(chunks)

        console.print(f"[cyan]총 {len(all_chunks)}개 청크 임베딩 생성 중...[/cyan]")

        batch_size = 50
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            texts = [c["text"] for c in batch]
            metadatas = [c["metadata"] for c in batch]
            ids = [
                f"{c['metadata']['file_path']}_{c['chunk_index']}" for c in batch
            ]

            embeddings = self.embedder.embed_batch(texts, show_progress=False)

            valid = [
                (t, e, m, doc_id)
                for t, e, m, doc_id in zip(texts, embeddings, metadatas, ids)
                if e is not None
            ]
            if valid:
                v_texts, v_embeds, v_metas, v_ids = zip(*valid)
                self.vector_db.add_documents(
                    documents=list(v_texts),
                    embeddings=list(v_embeds),
                    metadatas=list(v_metas),
                    ids=list(v_ids),
                )

            console.print(
                f"  [{i + len(batch)}/{len(all_chunks)}] 청크 인덱싱 완료"
            )

        # SQLite에도 문서 메타데이터 저장
        for doc in documents:
            self.sqlite_db.upsert_note(
                note_id=doc.file_path,
                title=doc.title,
                content=doc.content[:500],
                file_path=doc.file_path,
                source_type=SourceType.VAULT.value,
                metadata=doc.metadata,
            )
            for tag in doc.tags:
                self.sqlite_db.add_note_tag(doc.file_path, tag)
            for link_target in doc.links:
                self.sqlite_db.add_link(doc.file_path, link_target, "wiki_link")

        total = self.vector_db.count()
        console.print(f"\n[bold green]인덱싱 완료! 총 {total}개 청크가 벡터 DB에 저장됨[/bold green]")
