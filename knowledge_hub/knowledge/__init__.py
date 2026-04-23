"""Knowledge package public surface with lazy imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ParaManager",
    "NoteManager",
    "GraphManager",
    "GraphSignalAnalyzer",
    "GraphQuerySignal",
    "GraphEntityHint",
    "GraphCommunityHint",
    "analyze_graph_query",
    "is_graph_heavy_query",
    "Element",
    "EvidenceLink",
    "MemoryCard",
    "SemanticDocument",
    "ClaimSynthesisService",
]


def __getattr__(name: str) -> Any:
    if name == "ParaManager":
        return import_module("knowledge_hub.knowledge.para").ParaManager
    if name == "NoteManager":
        return import_module("knowledge_hub.knowledge.notes").NoteManager
    if name == "GraphManager":
        return import_module("knowledge_hub.knowledge.graph").GraphManager
    if name == "GraphSignalAnalyzer":
        return import_module("knowledge_hub.knowledge.graph_signals").GraphSignalAnalyzer
    if name == "GraphQuerySignal":
        return import_module("knowledge_hub.knowledge.graph_signals").GraphQuerySignal
    if name == "GraphEntityHint":
        return import_module("knowledge_hub.knowledge.graph_signals").GraphEntityHint
    if name == "GraphCommunityHint":
        return import_module("knowledge_hub.knowledge.graph_signals").GraphCommunityHint
    if name == "analyze_graph_query":
        return import_module("knowledge_hub.knowledge.graph_signals").analyze_graph_query
    if name == "is_graph_heavy_query":
        return import_module("knowledge_hub.knowledge.graph_signals").is_graph_heavy_query
    if name in {"Element", "EvidenceLink", "MemoryCard", "SemanticDocument"}:
        return getattr(import_module("knowledge_hub.knowledge.semantic_units"), name)
    if name == "ClaimSynthesisService":
        return import_module("knowledge_hub.knowledge.synthesis").ClaimSynthesisService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
