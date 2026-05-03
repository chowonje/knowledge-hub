from __future__ import annotations

from typing import Any

from knowledge_hub.ai.rag_support import safe_int
from knowledge_hub.core.models import SearchResult


class ParentContextService:
    def __init__(self, searcher: Any):
        self.searcher = searcher

    @classmethod
    def from_ctx(cls, ctx: Any) -> "ParentContextService":
        instance = cls.__new__(cls)
        instance.searcher = ctx
        return instance

    @staticmethod
    def build_document_filter(metadata: dict[str, Any]) -> dict[str, Any] | None:
        source = str(metadata.get("source_type", "")).strip()
        if source == "concept":
            return None

        candidates = ("file_path", "arxiv_id", "url")
        filter_dict: dict[str, Any] = {}
        if source:
            filter_dict["source_type"] = source

        for key in candidates:
            value = str(metadata.get(key, "")).strip()
            if value:
                filter_dict[key] = value
                break

        if not any(key in filter_dict for key in candidates):
            return None
        return filter_dict

    @staticmethod
    def doc_cache_key(filter_dict: dict[str, Any]) -> str:
        ordered = ["source_type", "file_path", "arxiv_id", "url"]
        parts = [f"{key}={filter_dict.get(key, '')}" for key in ordered if filter_dict.get(key, "")]
        return "|".join(parts)

    @staticmethod
    def parent_group_key(metadata: dict[str, Any]) -> str:
        parent_id = str(metadata.get("parent_id", "")).strip()
        if parent_id:
            return f"parent:{parent_id}"

        section_path = str(metadata.get("section_path", "")).strip()
        if section_path:
            return f"section_path:{section_path}"

        section_title = str(metadata.get("section_title", "")).strip()
        if section_title:
            return f"section_title:{section_title}"

        return "__document__"

    @staticmethod
    def parent_label(metadata: dict[str, Any]) -> str:
        for key in ("parent_title", "section_path", "section_title", "title"):
            value = str(metadata.get(key, "")).strip()
            if value:
                return value
        return ""

    def load_document_chunks(
        self,
        metadata: dict[str, Any],
        cache: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        filter_dict = self.build_document_filter(metadata)
        if not filter_dict:
            return []

        cache_key = self.doc_cache_key(filter_dict)
        if cache_key in cache:
            return cache[cache_key]

        try:
            raw = self.searcher.database.get_documents(
                filter_dict=filter_dict,
                limit=self.searcher.parent_fetch_limit,
                include_ids=True,
                include_documents=True,
                include_metadatas=True,
            )
        except Exception:
            cache[cache_key] = []
            return []

        docs = raw.get("documents", []) or []
        metas = raw.get("metadatas", []) or []
        ids = raw.get("ids", []) or []
        rows: list[dict[str, Any]] = []
        for idx, document in enumerate(docs):
            row_meta = metas[idx] if idx < len(metas) and isinstance(metas[idx], dict) else {}
            rows.append(
                {
                    "document": document or "",
                    "metadata": row_meta,
                    "chunk_index": safe_int(row_meta.get("chunk_index"), default=idx),
                    "document_id": str(ids[idx]) if idx < len(ids) else "",
                }
            )

        rows.sort(key=lambda item: (item.get("chunk_index", 0), item.get("document_id", "")))
        cache[cache_key] = rows
        return rows

    def resolve_parent_context(
        self,
        result: SearchResult,
        cache: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        default_parent_id = str(result.metadata.get("parent_id", "")).strip()
        default_parent_label = self.parent_label(result.metadata)
        target_parent_key = self.parent_group_key(result.metadata)
        target_chunk = safe_int(result.metadata.get("chunk_index"), default=0)

        rows = self.load_document_chunks(result.metadata, cache)
        if not rows:
            return {
                "parent_id": default_parent_id,
                "parent_label": default_parent_label,
                "chunk_span": str(target_chunk),
                "text": result.document,
            }

        siblings = [row for row in rows if self.parent_group_key(row.get("metadata", {})) == target_parent_key]
        if not siblings:
            siblings = rows

        max_chunks = max(1, int(self.searcher.parent_window_chunks))
        if len(siblings) <= max_chunks:
            selected = siblings
        else:
            hit_pos = min(
                range(len(siblings)),
                key=lambda idx: abs(safe_int(siblings[idx].get("chunk_index"), idx) - target_chunk),
            )
            left = max(0, hit_pos - (max_chunks // 2))
            right = min(len(siblings), left + max_chunks)
            left = max(0, right - max_chunks)
            selected = siblings[left:right]

        text_parts: list[str] = []
        for row in selected:
            row_meta = row.get("metadata", {})
            row_chunk = safe_int(row.get("chunk_index"), default=0)
            row_section = str(row_meta.get("section_title", "")).strip()
            prefix = f"[chunk {row_chunk}]"
            if row_section:
                prefix = f"{prefix} [{row_section}]"
            text_parts.append(f"{prefix}\n{row.get('document', '')}")

        chunk_span = (
            f"{safe_int(selected[0].get('chunk_index'), 0)}-{safe_int(selected[-1].get('chunk_index'), 0)}"
            if selected
            else str(target_chunk)
        )
        parent_meta = selected[0].get("metadata", {}) if selected else result.metadata
        parent_id = str(parent_meta.get("parent_id", "")).strip() or default_parent_id
        parent_label = self.parent_label(parent_meta) or default_parent_label

        return {
            "parent_id": parent_id,
            "parent_label": parent_label,
            "chunk_span": chunk_span,
            "text": "\n\n".join(text_parts).strip() or result.document,
        }

    def apply_parent_context(
        self,
        results: list[SearchResult],
        *,
        include_parent_text: bool = True,
    ) -> list[SearchResult]:
        if not results:
            return results

        doc_cache: dict[str, list[dict[str, Any]]] = {}
        for item in results:
            parent_ctx = self.resolve_parent_context(item, doc_cache)
            item.metadata["resolved_parent_id"] = str(parent_ctx.get("parent_id", "")).strip()
            item.metadata["resolved_parent_label"] = str(parent_ctx.get("parent_label", "")).strip()
            item.metadata["resolved_parent_chunk_span"] = str(parent_ctx.get("chunk_span", "")).strip()
            if include_parent_text:
                parent_text = str(parent_ctx.get("text", "")).strip()
                if parent_text:
                    item.document = parent_text
        return results
