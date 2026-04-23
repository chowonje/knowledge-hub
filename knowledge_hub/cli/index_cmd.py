"""Legacy index command shim."""

from knowledge_hub.interfaces.cli.commands.index_cmd import (
    EMBED_RETRY_BASE_SEC,
    _detect_competing_khub_processes,
    _embed_batch_openai,
    _embed_with_retry,
    _get_embedder,
    index_cmd,
)

__all__ = [
    "EMBED_RETRY_BASE_SEC",
    "_detect_competing_khub_processes",
    "_embed_batch_openai",
    "_embed_with_retry",
    "_get_embedder",
    "index_cmd",
]
