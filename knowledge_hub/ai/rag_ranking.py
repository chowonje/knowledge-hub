from __future__ import annotations

from knowledge_hub.core.models import SearchResult
from knowledge_hub.ai.rag_support import safe_float


def retrieval_sort_score(result: SearchResult) -> float:
    extras = dict(getattr(result, "lexical_extras", {}) or {})
    try:
        return float(extras.get("retrieval_sort_score"))
    except Exception:
        return safe_float(getattr(result, "score", 0.0), 0.0)


def retrieval_sort_key(
    result: SearchResult,
    *,
    prefer_lexical: bool = False,
) -> tuple[float, float, float, float]:
    if prefer_lexical:
        return (
            retrieval_sort_score(result),
            safe_float(getattr(result, "score", 0.0), 0.0),
            safe_float(getattr(result, "lexical_score", 0.0), 0.0),
            safe_float(getattr(result, "semantic_score", 0.0), 0.0),
        )
    return (
        retrieval_sort_score(result),
        safe_float(getattr(result, "score", 0.0), 0.0),
        safe_float(getattr(result, "semantic_score", 0.0), 0.0),
        safe_float(getattr(result, "lexical_score", 0.0), 0.0),
    )
