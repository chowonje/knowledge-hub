from __future__ import annotations

__all__ = ["RAGSearcher"]


def __getattr__(name: str):
    if name == "RAGSearcher":
        from knowledge_hub.ai.rag import RAGSearcher

        return RAGSearcher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
