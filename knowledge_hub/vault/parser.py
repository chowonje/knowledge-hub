"""
Obsidian Vault 파서

Obsidian vault의 마크다운 파일을 읽고 파싱합니다.
- YAML frontmatter 추출
- Obsidian 링크 [[]] 처리
- 태그 추출
- 문서 청킹
"""

import re
from datetime import date, datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml

try:
    import frontmatter
except Exception:  # pragma: no cover - optional dependency fallback
    frontmatter = None

from knowledge_hub.core.models import Document, SourceType
from knowledge_hub.core.vault_paths import resolve_vault_exclude_folders


class ObsidianParser:
    """Obsidian vault 파서"""

    def __init__(self, vault_path: str, exclude_folders: List[str] = None):
        self.vault_path = Path(vault_path)
        self.exclude_folders = resolve_vault_exclude_folders(exclude_folders)
        self.last_errors: list[dict[str, str]] = []
        if not self.vault_path.exists():
            raise ValueError(f"Vault 경로가 존재하지 않습니다: {vault_path}")

    def parse_vault(self) -> List[Document]:
        """전체 vault를 파싱하여 Document 리스트 반환"""
        self.last_errors = []
        documents = []
        for md_file in self.vault_path.rglob("*.md"):
            if any(exclude in md_file.parts for exclude in self.exclude_folders):
                continue
            try:
                doc = self.parse_file(md_file)
                if doc:
                    documents.append(doc)
            except Exception as e:
                self.last_errors.append(
                    {
                        "file_path": str(md_file),
                        "error_type": type(e).__name__,
                        "message": str(e),
                    }
                )
                print(f"파싱 오류 {md_file}: {e}")
        return documents

    def parse_file(self, file_path: Path) -> Optional[Document]:
        """단일 마크다운 파일 파싱"""
        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read()

        metadata, content = self._parse_frontmatter(raw)
        metadata = self._sanitize_metadata_dict(metadata)
        content = str(content or "").strip()
        if not content:
            return None

        title = str(
            metadata.get("title")
            or self._extract_title_from_content(content)
            or file_path.stem
        ).strip()
        tags = self._extract_tags(metadata, content)
        links = self._extract_links(content)

        return Document(
            content=content,
            metadata=dict(metadata),
            file_path=str(file_path.relative_to(self.vault_path)),
            title=title,
            tags=tags,
            links=links,
            source_type=SourceType.VAULT,
        )

    @staticmethod
    def _parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
        if frontmatter is not None:
            try:
                post = frontmatter.loads(raw)
                return dict(post.metadata or {}), str(post.content or "")
            except Exception:
                # Fall through to the tolerant YAML/plaintext parser below.
                pass

        if raw.startswith("---\n"):
            end = raw.find("\n---\n", 4)
            if end != -1:
                fm_raw = raw[4:end]
                body = raw[end + len("\n---\n") :]
                try:
                    metadata = yaml.safe_load(fm_raw) or {}
                    if not isinstance(metadata, dict):
                        metadata = {}
                except Exception:
                    metadata = ObsidianParser._parse_frontmatter_loose(fm_raw)
                return metadata, body
        return {}, raw

    @staticmethod
    def _parse_frontmatter_loose(frontmatter_raw: str) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        active_list_key: str | None = None
        for raw_line in frontmatter_raw.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                active_list_key = None
                continue
            if active_list_key and stripped.startswith("- "):
                value = stripped[2:].strip()
                if value:
                    metadata.setdefault(active_list_key, []).append(ObsidianParser._parse_scalar_value(value))
                continue
            if ":" not in stripped:
                active_list_key = None
                continue
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                active_list_key = None
                continue
            if not value:
                metadata[key] = []
                active_list_key = key
                continue
            metadata[key] = ObsidianParser._parse_scalar_value(value)
            active_list_key = None
        return metadata

    @staticmethod
    def _parse_scalar_value(value: str) -> Any:
        try:
            parsed = yaml.safe_load(value)
        except Exception:
            parsed = value
        return parsed

    @staticmethod
    def _extract_title_from_content(content: str) -> str:
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped in {"---", "***", "___"}:
                continue
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
            break
        return ""

    @classmethod
    def _sanitize_metadata_dict(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = cls._sanitize_metadata_value(metadata)
        return sanitized if isinstance(sanitized, dict) else {}

    @classmethod
    def _sanitize_metadata_value(cls, value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key, item in value.items():
                out[str(key)] = cls._sanitize_metadata_value(item)
            return out
        if isinstance(value, (list, tuple, set)):
            return [cls._sanitize_metadata_value(item) for item in value]
        return str(value)

    def _extract_tags(self, metadata: Dict, content: str) -> List[str]:
        """frontmatter + 본문 태그 추출"""
        tags = set()
        if "tags" in metadata:
            meta_tags = metadata["tags"]
            if isinstance(meta_tags, list):
                tags.update(str(tag).strip() for tag in meta_tags if str(tag).strip())
            elif isinstance(meta_tags, str):
                clean = meta_tags.strip()
                if clean:
                    tags.add(clean)

        tag_pattern = r"#([\w가-힣]+(?:[_-][\w가-힣]+)*)"
        tags.update(re.findall(tag_pattern, content))
        return sorted(str(tag) for tag in tags if str(tag).strip())

    def _extract_links(self, content: str) -> List[str]:
        """Obsidian [[링크]] 추출"""
        link_pattern = r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]"
        seen: set[str] = set()
        links: list[str] = []
        for raw in re.findall(link_pattern, content):
            link = str(raw).strip()
            if not link or link in seen:
                continue
            seen.add(link)
            links.append(link)
        return links

    @staticmethod
    def chunk_document(
        document: Document, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """문서를 청크로 분할"""
        headings = ObsidianParser._extract_headings(document.content)

        document_scope_id = f"{document.source_type.value}:{document.file_path}"

        def locate_section(offset: int) -> tuple[str, str]:
            section_title = ""
            section_path: list[str] = []
            for heading_offset, heading_path, heading in headings:
                if heading_offset <= offset:
                    section_title = heading
                    section_path = heading_path
                else:
                    break

            return section_title, " > ".join(section_path)

        def scope_metadata(section_title: str, section_path: str) -> dict[str, str]:
            parent_type = "section" if section_title else "document"
            parent_scope = (section_path or section_title or document.title or "__document__").strip()
            parent_scope = re.sub(r"\s+", " ", parent_scope) or "__document__"
            section_scope_id = f"{document_scope_id}::section:{parent_scope}" if section_title else ""
            stable_scope_id = section_scope_id or document_scope_id
            return {
                "document_id": document_scope_id,
                "document_scope_id": document_scope_id,
                "section_scope_id": section_scope_id,
                "stable_scope_id": stable_scope_id,
                "scope_level": parent_type,
                "scope_path": parent_scope,
                "parent_id": (
                    f"{document.source_type.value}:{document.file_path}"
                    f"::{parent_type}:{parent_scope}"
                ),
                "parent_title": section_title or document.title,
                "parent_type": parent_type,
            }

        content = document.content
        if len(content) <= chunk_size:
            section_title, section_path = locate_section(0)
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
                        "section_title": section_title,
                        "section_path": section_path,
                        "chunk_size": len(content),
                        "total_chunks": 1,
                        **scope_metadata(section_title, section_path),
                        "contextual_summary": ObsidianParser._build_contextual_summary(
                            document.title,
                            content,
                            section_title,
                            section_path,
                        ),
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
            section_title, section_path = locate_section(start)

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
                        "section_title": section_title,
                        "section_path": section_path,
                        "chunk_size": len(chunk_text),
                        **scope_metadata(section_title, section_path),
                        "contextual_summary": ObsidianParser._build_contextual_summary(
                            document.title,
                            chunk_text,
                            section_title,
                            section_path,
                        ),
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

    @staticmethod
    def _extract_headings(content: str) -> list[tuple[int, list[str], str]]:
        headings: list[tuple[int, list[str], str]] = []
        stack: list[tuple[int, str]] = []

        for match in re.finditer(r"^(#{1,6})\s+(.+?)\s*$", content, flags=re.MULTILINE):
            level = len(match.group(1))
            heading = match.group(2).strip()

            while stack and stack[-1][0] >= level:
                stack.pop()

            stack.append((level, heading))
            headings.append((match.start(), [item[1] for item in stack], heading))

        return headings

    @staticmethod
    def _build_contextual_summary(
        title: str,
        chunk_text: str,
        section_title: str,
        section_path: str,
    ) -> str:
        normalized = re.sub(r"\s+", " ", (chunk_text or "").strip())
        if not normalized:
            return title

        first_sentence = normalized.split(". ")[0].strip()
        if len(first_sentence) > 180:
            first_sentence = f"{first_sentence[:177]}..."

        if section_title:
            if section_path and section_path != section_title:
                return f"[{section_path}] {first_sentence}"
            return f"[{section_title}] {first_sentence}"

        return f"[{title}] {first_sentence}"
