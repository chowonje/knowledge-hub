from __future__ import annotations

from typing import Any, Callable


def should_use_ask_v2(
    *,
    source_type: str | None,
    metadata_filter: dict[str, Any] | None = None,
    sqlite_db: Any | None = None,
    supports_fn: Callable[..., bool],
    normalize_source_type_fn: Callable[[str | None], str],
) -> bool:
    if not supports_fn(source_type=source_type, metadata_filter=metadata_filter):
        return False

    normalized = normalize_source_type_fn(source_type)
    scoped = dict(metadata_filter or {})
    paper_scoped = any(str(scoped.get(key) or "").strip() for key in ("arxiv_id", "paper_id"))
    if normalized == "paper" or (not normalized and paper_scoped):
        # Paper ask_v2 is promoted by default. Do not keep a routine pre-route
        # to legacy here; real compatibility gaps are handled as execution-time
        # fallback in the runtime.
        return True

    if normalized == "web" or any(str(scoped.get(key) or "").strip() for key in ("canonical_url", "url", "source_url")):
        required_methods = (
            "search_web_cards_v2",
            "get_web_card_v2_by_url",
            "replace_web_card_claim_refs_v2",
            "replace_web_evidence_anchors_v2",
            "list_web_evidence_anchors_v2",
        )
        return all(callable(getattr(sqlite_db, name, None)) for name in required_methods)

    if normalized in {"vault", "project"}:
        required_methods = (
            "search_vault_cards_v2" if normalized == "vault" else "search_document_memory_units",
            "get_vault_card_v2" if normalized == "vault" else "get_document_memory_summary",
        )
        return all(callable(getattr(sqlite_db, name, None)) for name in required_methods)

    return False
