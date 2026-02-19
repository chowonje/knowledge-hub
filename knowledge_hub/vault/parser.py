"""
Obsidian Vault 파서

Obsidian vault의 마크다운 파일을 읽고 파싱합니다.
- YAML frontmatter 추출
- Obsidian 링크 [[]] 처리
- 태그 추출
- 문서 청킹
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import frontmatter

from knowledge_hub.core.models import Document, SourceType


class ObsidianParser:
    """Obsidian vault 파서"""

    def __init__(self, vault_path: str, exclude_folders: List[str] = None):
        self.vault_path = Path(vault_path)
        self.exclude_folders = exclude_folders or [".obsidian", ".trash", "templates"]
        if not self.vault_path.exists():
            raise ValueError(f"Vault 경로가 존재하지 않습니다: {vault_path}")

    def parse_vault(self) -> List[Document]:
        """전체 vault를 파싱하여 Document 리스트 반환"""
        documents = []
        for md_file in self.vault_path.rglob("*.md"):
            if any(exclude in md_file.parts for exclude in self.exclude_folders):
                continue
            try:
                doc = self.parse_file(md_file)
                if doc:
                    documents.append(doc)
            except Exception as e:
                print(f"파싱 오류 {md_file}: {e}")
        return documents

    def parse_file(self, file_path: Path) -> Optional[Document]:
        """단일 마크다운 파일 파싱"""
        with open(file_path, "r", encoding="utf-8") as f:
            post = frontmatter.load(f)

        content = post.content.strip()
        if not content:
            return None

        title = post.get("title", file_path.stem)
        tags = self._extract_tags(post.metadata, content)
        links = self._extract_links(content)

        return Document(
            content=content,
            metadata=dict(post.metadata),
            file_path=str(file_path.relative_to(self.vault_path)),
            title=title,
            tags=tags,
            links=links,
            source_type=SourceType.VAULT,
        )

    def _extract_tags(self, metadata: Dict, content: str) -> List[str]:
        """frontmatter + 본문 태그 추출"""
        tags = set()
        if "tags" in metadata:
            meta_tags = metadata["tags"]
            if isinstance(meta_tags, list):
                tags.update(meta_tags)
            elif isinstance(meta_tags, str):
                tags.add(meta_tags)

        tag_pattern = r"#([\w가-힣]+(?:[_-][\w가-힣]+)*)"
        tags.update(re.findall(tag_pattern, content))
        return list(tags)

    def _extract_links(self, content: str) -> List[str]:
        """Obsidian [[링크]] 추출"""
        link_pattern = r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]"
        return re.findall(link_pattern, content)

    @staticmethod
    def chunk_document(
        document: Document, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """문서를 청크로 분할"""
        content = document.content
        if len(content) <= chunk_size:
            return [
                {
                    "text": content,
                    "chunk_index": 0,
                    "metadata": {
                        "title": document.title,
                        "file_path": document.file_path,
                        "tags": document.tags,
                        "links": document.links,
                        "source_type": document.source_type.value,
                        **document.metadata,
                    },
                }
            ]

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(content):
            end = start + chunk_size
            chunk_text = content[start:end]

            if end < len(content):
                for delimiter in ["\n\n", "\n", ". ", "? ", "! "]:
                    last_delim = chunk_text.rfind(delimiter)
                    if last_delim > chunk_size * 0.5:
                        chunk_text = chunk_text[: last_delim + len(delimiter)]
                        end = start + last_delim + len(delimiter)
                        break

            chunks.append(
                {
                    "text": chunk_text.strip(),
                    "chunk_index": chunk_index,
                    "metadata": {
                        "title": document.title,
                        "file_path": document.file_path,
                        "tags": document.tags,
                        "links": document.links,
                        "source_type": document.source_type.value,
                        "total_chunks": None,
                        **document.metadata,
                    },
                }
            )
            start = end - chunk_overlap
            chunk_index += 1

        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = len(chunks)
        return chunks
