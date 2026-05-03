"""Bounded graph-signal analysis helpers.

This module does not implement GraphRAG. It only inspects the existing
ontology/graph data that already lives in SQLite and returns inspectable
diagnostics that can later feed retrieval reranking or candidate reduction.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import re
from typing import Any

from knowledge_hub.learning.resolver import EntityResolver, normalize_term


GRAPH_CUE_WEIGHTS: dict[str, float] = {
    "knowledge graph": 0.18,
    "graph-based": 0.16,
    "graph based": 0.16,
    "multi-hop": 0.16,
    "community report": 0.16,
    "community reports": 0.16,
    "neighborhood": 0.14,
    "neighbors": 0.14,
    "neighbor": 0.14,
    "ontology": 0.13,
    "entity": 0.12,
    "entities": 0.12,
    "relation": 0.12,
    "relations": 0.12,
    "relationship": 0.12,
    "relationships": 0.12,
    "path": 0.12,
    "paths": 0.12,
    "connect": 0.1,
    "connected": 0.1,
    "connection": 0.1,
    "connections": 0.1,
    "network": 0.1,
    "cluster": 0.09,
    "clusters": 0.09,
    "link": 0.08,
    "links": 0.08,
    "graph": 0.08,
}


ENTITY_QUERY_TERMS = {
    "ontology",
    "entity",
    "entities",
    "concept",
    "concepts",
    "claim",
    "claims",
    "relation",
    "relations",
    "relationship",
    "relationships",
    "paper",
    "papers",
    "neighborhood",
    "neighbor",
    "neighbors",
    "path",
    "paths",
    "community",
    "cluster",
    "clusters",
}


def _tokenize(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_가-힣]{2,}", str(text or ""))
        if token
    }


def _normalized_phrase_hits(text: str, phrases: dict[str, float]) -> list[str]:
    lowered = normalize_term(text)
    hits = [phrase for phrase in phrases if phrase in lowered]
    hits.sort(key=lambda item: (-len(item), item))
    return hits


def _unique_extend(values: list[str], items: list[str]) -> list[str]:
    for item in items:
        token = str(item or "").strip()
        if token and token not in values:
            values.append(token)
    return values


def _safe_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _entity_name_match_strength(query_norm: str, entity_name: str, aliases: list[str]) -> float:
    entity_norm = normalize_term(entity_name)
    alias_norms = [normalize_term(alias) for alias in aliases if normalize_term(alias)]
    if entity_norm and entity_norm == query_norm:
        return 1.0
    if entity_norm and (entity_norm in query_norm or query_norm in entity_norm):
        return 0.9
    if any(alias and (alias in query_norm or query_norm in alias) for alias in alias_norms):
        return 0.85
    query_tokens = _tokenize(query_norm)
    entity_tokens = _tokenize(entity_norm)
    alias_tokens = set().union(*(_tokenize(alias) for alias in alias_norms)) if alias_norms else set()
    overlap = len(query_tokens & (entity_tokens | alias_tokens))
    if overlap:
        return min(0.8, 0.2 + 0.15 * overlap)
    return 0.0


def _relation_other_side(row: dict[str, Any], entity_type: str, entity_id: str) -> tuple[str, str]:
    source_type = str(row.get("source_type") or "").strip()
    source_id = str(row.get("source_id") or "").strip()
    target_type = str(row.get("target_type") or "").strip()
    target_id = str(row.get("target_id") or "").strip()

    if source_type == entity_type and source_id == entity_id:
        return target_type, target_id
    if target_type == entity_type and target_id == entity_id:
        return source_type, source_id
    source_entity_id = str(row.get("source_entity_id") or "").strip()
    target_entity_id = str(row.get("target_entity_id") or "").strip()
    if source_entity_id == entity_id:
        return target_type or source_type, target_entity_id or target_id
    if target_entity_id == entity_id:
        return source_type or target_type, source_entity_id or source_id
    return "", ""


@dataclass(frozen=True)
class GraphEntityHint:
    entity_id: str
    entity_type: str
    canonical_name: str
    matched_terms: tuple[str, ...]
    aliases: tuple[str, ...] = field(default_factory=tuple)
    relation_count: int = 0
    neighbor_entity_ids: tuple[str, ...] = field(default_factory=tuple)
    neighbor_names: tuple[str, ...] = field(default_factory=tuple)
    neighbor_types: tuple[str, ...] = field(default_factory=tuple)
    score: float = 0.0
    reasons: tuple[str, ...] = field(default_factory=tuple)
    scope_recommendation: str = "baseline"

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "canonical_name": self.canonical_name,
            "matched_terms": list(self.matched_terms),
            "aliases": list(self.aliases),
            "relation_count": self.relation_count,
            "neighbor_entity_ids": list(self.neighbor_entity_ids),
            "neighbor_names": list(self.neighbor_names),
            "neighbor_types": list(self.neighbor_types),
            "score": round(self.score, 6),
            "reasons": list(self.reasons),
            "scope_recommendation": self.scope_recommendation,
        }


@dataclass(frozen=True)
class GraphCommunityHint:
    group: str
    node_count: int
    bridge_node_ids: tuple[str, ...] = field(default_factory=tuple)
    bridge_labels: tuple[str, ...] = field(default_factory=tuple)
    score: float = 0.0
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "node_count": self.node_count,
            "bridge_node_ids": list(self.bridge_node_ids),
            "bridge_labels": list(self.bridge_labels),
            "score": round(self.score, 6),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class GraphQuerySignal:
    query: str
    is_graph_heavy: bool
    confidence: float
    cue_hits: tuple[str, ...] = field(default_factory=tuple)
    matched_entity_count: int = 0
    candidate_hints: tuple[GraphEntityHint, ...] = field(default_factory=tuple)
    community_hints: tuple[GraphCommunityHint, ...] = field(default_factory=tuple)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    recommended_mode: str = "baseline"
    recommended_top_k: int = 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "is_graph_heavy": self.is_graph_heavy,
            "confidence": round(self.confidence, 6),
            "cue_hits": list(self.cue_hits),
            "matched_entity_count": self.matched_entity_count,
            "candidate_hints": [hint.to_dict() for hint in self.candidate_hints],
            "community_hints": [hint.to_dict() for hint in self.community_hints],
            "diagnostics": dict(self.diagnostics),
            "recommended_mode": self.recommended_mode,
            "recommended_top_k": self.recommended_top_k,
        }


class GraphSignalAnalyzer:
    """Inspect existing ontology and note graph data for graph-heavy queries."""

    def __init__(self, repository: Any, *, graph_data: dict[str, Any] | None = None):
        self.repository = repository
        self.graph_data = graph_data or {}

    def _load_entities(self, limit: int = 5000) -> list[dict[str, Any]]:
        getter = getattr(self.repository, "list_ontology_entities", None)
        if not callable(getter):
            return []
        try:
            return [item for item in getter(limit=max(1, int(limit))) if item]
        except Exception:
            return []

    def _resolve_entity_candidates(self, query: str, query_tokens: set[str]) -> list[dict[str, Any]]:
        query_norm = normalize_term(query)
        resolver = None
        try:
            resolver = EntityResolver(self.repository, fuzzy_threshold=0.92)
        except Exception:
            resolver = None

        candidates: dict[str, dict[str, Any]] = {}
        for token in sorted(query_tokens):
            if len(token) < 2:
                continue
            identity = resolver.resolve(token) if resolver else None
            if not identity:
                continue
            candidates[str(identity.canonical_id)] = {
                "entity_id": str(identity.canonical_id),
                "entity_type": "",
                "canonical_name": str(identity.display_name or identity.canonical_id),
                "aliases": list(identity.aliases or []),
                "matched_terms": [token],
                "match_strength": float(identity.resolve_confidence or 0.0),
                "reasons": [str(identity.resolve_method or "resolved")],
            }

        for entity in self._load_entities():
            entity_id = str(entity.get("entity_id") or "").strip()
            canonical_name = str(entity.get("canonical_name") or "").strip()
            entity_type = str(entity.get("entity_type") or "").strip()
            if not entity_id or not canonical_name:
                continue
            aliases = []
            alias_getter = getattr(self.repository, "get_entity_aliases", None)
            if callable(alias_getter):
                try:
                    aliases = [str(alias).strip() for alias in alias_getter(entity_id) if str(alias).strip()]
                except Exception:
                    aliases = []
            if entity_id not in candidates:
                candidates[entity_id] = {
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "canonical_name": canonical_name,
                    "aliases": aliases,
                    "matched_terms": [],
                    "match_strength": 0.0,
                    "reasons": [],
                }
            candidate = candidates[entity_id]
            candidate["entity_type"] = candidate["entity_type"] or entity_type
            candidate["aliases"] = list(dict.fromkeys([*candidate.get("aliases", []), *aliases]))
            match_strength = _entity_name_match_strength(query_norm, canonical_name, aliases)
            if match_strength > 0:
                candidate["match_strength"] = max(float(candidate.get("match_strength", 0.0)), match_strength)
                _unique_extend(candidate["matched_terms"], [canonical_name, *aliases, query_norm])
                _unique_extend(candidate["reasons"], ["name_overlap" if match_strength >= 0.85 else "token_overlap"])

        filtered = [item for item in candidates.values() if item.get("match_strength", 0.0) > 0]
        filtered.sort(
            key=lambda item: (
                float(item.get("match_strength", 0.0)),
                len(item.get("matched_terms", [])),
                str(item.get("canonical_name") or ""),
            ),
            reverse=True,
        )
        return filtered

    def _build_entity_hints(
        self,
        query: str,
        query_tokens: set[str],
        candidate_entities: list[dict[str, Any]],
        *,
        max_entities: int,
        max_neighbors: int,
    ) -> list[GraphEntityHint]:
        cue_hits = _normalized_phrase_hits(query, GRAPH_CUE_WEIGHTS)
        cue_score = sum(GRAPH_CUE_WEIGHTS.get(hit, 0.05) for hit in cue_hits)
        hints: list[GraphEntityHint] = []

        for candidate in candidate_entities[: max(1, int(max_entities))]:
            entity_id = str(candidate.get("entity_id") or "").strip()
            entity_type = str(candidate.get("entity_type") or "").strip() or "concept"
            canonical_name = str(candidate.get("canonical_name") or entity_id)
            aliases = [str(alias).strip() for alias in _safe_list(candidate.get("aliases")) if str(alias).strip()]
            relation_rows = []
            relations_getter = getattr(self.repository, "get_relations", None)
            if callable(relations_getter):
                try:
                    relation_rows = [row for row in relations_getter(entity_type, entity_id) if row]
                except Exception:
                    relation_rows = []

            neighbors: dict[str, dict[str, Any]] = {}
            for row in relation_rows:
                neighbor_type, neighbor_id = _relation_other_side(row, entity_type, entity_id)
                if not neighbor_id:
                    continue
                neighbor_entity = {}
                entity_getter = getattr(self.repository, "get_ontology_entity", None)
                if callable(entity_getter):
                    try:
                        neighbor_entity = entity_getter(neighbor_id) or {}
                    except Exception:
                        neighbor_entity = {}
                neighbor_name = str(neighbor_entity.get("canonical_name") or neighbor_id).strip()
                neighbor_type = str(neighbor_entity.get("entity_type") or neighbor_type).strip()
                neighbors.setdefault(
                    neighbor_id,
                    {
                        "entity_id": neighbor_id,
                        "name": neighbor_name,
                        "type": neighbor_type,
                    },
                )

            neighbor_items = list(neighbors.values())
            neighbor_items.sort(key=lambda item: (item["name"], item["entity_id"]))
            neighbor_names = [str(item["name"]) for item in neighbor_items[: max(1, int(max_neighbors))]]
            neighbor_ids = [str(item["entity_id"]) for item in neighbor_items[: max(1, int(max_neighbors))]]
            neighbor_types = [str(item["type"]) for item in neighbor_items[: max(1, int(max_neighbors))]]
            neighbor_match_count = 0
            for neighbor_name in neighbor_names:
                neighbor_norm = normalize_term(neighbor_name)
                if neighbor_norm and (neighbor_norm in normalize_term(query) or query_tokens & _tokenize(neighbor_norm)):
                    neighbor_match_count += 1

            relation_count = len(relation_rows)
            relation_strength = min(1.0, relation_count / 6.0)
            neighbor_strength = min(1.0, neighbor_match_count / 3.0)
            match_strength = float(candidate.get("match_strength", 0.0))
            reason_set = set(_safe_list(candidate.get("reasons")))
            if relation_count >= 3:
                reason_set.add("relation_density")
            if neighbor_match_count:
                reason_set.add("neighbor_overlap")
            if cue_hits:
                reason_set.add("graph_cue")
            if len(candidate.get("matched_terms", [])) >= 2:
                reason_set.add("multi_term_match")

            score = min(
                1.0,
                (0.4 * match_strength)
                + (0.2 * relation_strength)
                + (0.15 * neighbor_strength)
                + (0.25 * min(1.0, cue_score)),
            )
            if relation_count >= 4 and cue_hits:
                score = min(1.0, score + 0.05)

            if score >= 0.7 or relation_count >= 4:
                scope_recommendation = "candidate_reduction"
            elif match_strength >= 0.8:
                scope_recommendation = "entity_lookup"
            else:
                scope_recommendation = "baseline"

            hints.append(
                GraphEntityHint(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    canonical_name=canonical_name,
                    matched_terms=tuple(dict.fromkeys(str(item) for item in _safe_list(candidate.get("matched_terms")) if str(item).strip())),
                    aliases=tuple(aliases),
                    relation_count=relation_count,
                    neighbor_entity_ids=tuple(neighbor_ids),
                    neighbor_names=tuple(neighbor_names),
                    neighbor_types=tuple(neighbor_types),
                    score=score,
                    reasons=tuple(sorted(reason_set)),
                    scope_recommendation=scope_recommendation,
                )
            )

        hints.sort(key=lambda item: (item.score, item.relation_count, item.canonical_name), reverse=True)
        return hints

    def _build_community_hints(self, query: str) -> list[GraphCommunityHint]:
        graph_data = self.graph_data if isinstance(self.graph_data, dict) else {}
        nodes = _safe_list(graph_data.get("nodes"))
        edges = _safe_list(graph_data.get("edges"))
        if not nodes:
            return []

        query_tokens = _tokenize(query)
        node_lookup: dict[str, dict[str, Any]] = {}
        degree: Counter[str] = Counter()
        adjacency: dict[str, set[str]] = defaultdict(set)

        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id") or node.get("node_id") or "").strip()
            if not node_id:
                continue
            node_lookup[node_id] = node

        for edge in edges:
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("source") or edge.get("source_id") or "").strip()
            target = str(edge.get("target") or edge.get("target_id") or "").strip()
            if not source or not target:
                continue
            degree[source] += 1
            degree[target] += 1
            adjacency[source].add(target)
            adjacency[target].add(source)

        matched_nodes: list[dict[str, Any]] = []
        for node in node_lookup.values():
            label = str(node.get("label") or node.get("title") or "").strip()
            node_id = str(node.get("id") or node.get("node_id") or "").strip()
            node_tokens = _tokenize(f"{label} {node_id}")
            if query_tokens & node_tokens:
                matched_nodes.append(node)

        if not matched_nodes:
            return []

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for node in matched_nodes:
            group = str(node.get("group") or node.get("clusterId") or node.get("cluster_id") or "none").strip() or "none"
            grouped[group].append(node)

        community_hints: list[GraphCommunityHint] = []
        for group, items in grouped.items():
            ranked_items = sorted(
                items,
                key=lambda node: (
                    degree.get(str(node.get("id") or node.get("node_id") or ""), 0),
                    str(node.get("label") or node.get("title") or ""),
                ),
                reverse=True,
            )
            bridge_nodes = ranked_items[:3]
            bridge_node_ids = tuple(str(node.get("id") or node.get("node_id") or "").strip() for node in bridge_nodes if str(node.get("id") or node.get("node_id") or "").strip())
            bridge_labels = tuple(str(node.get("label") or node.get("title") or "").strip() for node in bridge_nodes if str(node.get("label") or node.get("title") or "").strip())
            score = min(1.0, 0.18 + (0.08 * len(items)) + (0.05 * len(bridge_node_ids)) + (0.03 * max((degree.get(node_id, 0) for node_id in bridge_node_ids), default=0)))
            reasons = ("note_graph_overlap", "community_group")
            if any(node_id in adjacency for node_id in bridge_node_ids):
                reasons = (*reasons, "connected_nodes")
            community_hints.append(
                GraphCommunityHint(
                    group=group,
                    node_count=len(items),
                    bridge_node_ids=bridge_node_ids,
                    bridge_labels=bridge_labels,
                    score=score,
                    reasons=reasons,
                )
            )

        community_hints.sort(key=lambda item: (item.score, item.node_count, item.group), reverse=True)
        return community_hints

    def analyze(
        self,
        query: str,
        *,
        max_entities: int = 5,
        max_neighbors: int = 6,
    ) -> GraphQuerySignal:
        query_text = str(query or "").strip()
        query_tokens = _tokenize(query_text)
        cue_hits = tuple(_normalized_phrase_hits(query_text, GRAPH_CUE_WEIGHTS))
        candidate_entities = self._resolve_entity_candidates(query_text, query_tokens)
        entity_hints = self._build_entity_hints(
            query_text,
            query_tokens,
            candidate_entities,
            max_entities=max_entities,
            max_neighbors=max_neighbors,
        )
        community_hints = self._build_community_hints(query_text)

        cue_score = sum(GRAPH_CUE_WEIGHTS.get(hit, 0.05) for hit in cue_hits)
        entity_count = len(entity_hints)
        entity_score = min(1.0, entity_count / 3.0)
        relation_count = sum(h.relation_count for h in entity_hints[: max(1, int(max_entities))])
        relation_score = min(1.0, relation_count / 8.0)
        community_score = min(1.0, len(community_hints) / 3.0)
        entity_heavy_bonus = 0.18 if entity_count >= 2 and relation_count >= 3 else 0.0
        confidence = min(
            1.0,
            (0.35 * cue_score)
            + (0.3 * entity_score)
            + (0.25 * relation_score)
            + (0.1 * community_score)
            + entity_heavy_bonus,
        )
        graph_heavy = bool(
            confidence >= 0.45
            and (
                cue_hits
                or entity_count >= 2
                or any(h.relation_count >= 3 for h in entity_hints)
            )
        )
        recommended_top_k = 3 if graph_heavy else 5
        diagnostics = {
            "graph_cue_hits": list(cue_hits),
            "graph_cue_count": len(cue_hits),
            "query_token_count": len(query_tokens),
            "entity_match_count": len(entity_hints),
            "relation_count": sum(h.relation_count for h in entity_hints),
            "community_hint_count": len(community_hints),
            "candidate_reduction_eligible": graph_heavy,
            "recommended_top_k": recommended_top_k,
            "recommended_mode": "graph" if graph_heavy else "baseline",
            "query_kind": "community" if any(hit in {"community report", "community reports", "cluster"} for hit in cue_hits) else ("path" if any(hit in {"multi-hop", "path", "paths"} for hit in cue_hits) else "entity"),
        }
        return GraphQuerySignal(
            query=query_text,
            is_graph_heavy=graph_heavy,
            confidence=confidence,
            cue_hits=cue_hits,
            matched_entity_count=len(entity_hints),
            candidate_hints=tuple(entity_hints[: max(1, int(max_entities))]),
            community_hints=tuple(community_hints[:3]),
            diagnostics=diagnostics,
            recommended_mode="graph" if graph_heavy else "baseline",
            recommended_top_k=recommended_top_k,
        )


def analyze_graph_query(
    query: str,
    repository: Any,
    *,
    max_entities: int = 5,
    max_neighbors: int = 6,
    graph_data: dict[str, Any] | None = None,
) -> GraphQuerySignal:
    analyzer = GraphSignalAnalyzer(repository, graph_data=graph_data)
    return analyzer.analyze(query, max_entities=max_entities, max_neighbors=max_neighbors)


def is_graph_heavy_query(
    query: str,
    repository: Any,
    *,
    max_entities: int = 5,
    max_neighbors: int = 6,
    graph_data: dict[str, Any] | None = None,
) -> bool:
    return analyze_graph_query(
        query,
        repository,
        max_entities=max_entities,
        max_neighbors=max_neighbors,
        graph_data=graph_data,
    ).is_graph_heavy


__all__ = [
    "GraphCommunityHint",
    "GraphEntityHint",
    "GraphQuerySignal",
    "GraphSignalAnalyzer",
    "analyze_graph_query",
    "is_graph_heavy_query",
]
