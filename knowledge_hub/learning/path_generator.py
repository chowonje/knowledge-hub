"""Approved learning path generation."""

from __future__ import annotations

from collections import defaultdict, deque
import re
from typing import Any
from uuid import uuid4

from knowledge_hub.learning.graph_builder import topic_preferred_prerequisites
from knowledge_hub.learning.graph_models import LearningEdge, LearningPath
from knowledge_hub.learning.prerequisites import transitive_prerequisite_map


def _tarjan_scc(nodes: list[str], adjacency: dict[str, list[str]]) -> list[list[str]]:
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    components: list[list[str]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in adjacency.get(node, []):
            if neighbor not in indices:
                strongconnect(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbor])

        if lowlinks[node] == indices[node]:
            component: list[str] = []
            while stack:
                value = stack.pop()
                on_stack.discard(value)
                component.append(value)
                if value == node:
                    break
            components.append(sorted(component))

    for node in nodes:
        if node not in indices:
            strongconnect(node)
    return components


class LearningPathGenerator:
    def _topic_tokens(self, topic_slug: str) -> set[str]:
        return {
            token
            for token in re.split(r"[^a-z0-9]+", str(topic_slug or "").lower())
            if token and len(token) > 2 and token not in {"the", "and"}
        }

    def _node_matches_topic(self, node: dict[str, Any], topic_tokens: set[str]) -> bool:
        if not topic_tokens:
            return False
        name = str(node.get("canonical_name") or "").lower()
        name_tokens = {
            token
            for token in re.split(r"[^a-z0-9]+", name)
            if token and len(token) > 2
        }
        return bool(name_tokens & topic_tokens)

    def _is_pathworthy_node(self, node: dict[str, Any], topic_tokens: set[str]) -> bool:
        canonical_name = str(node.get("canonical_name") or "").strip()
        if not canonical_name:
            return False
        provenance = node.get("provenance") or {}
        if self._node_matches_topic(node, topic_tokens):
            return True
        if (provenance.get("isTrunk") and float(provenance.get("topicRelevance") or 0.0) >= 0.16):
            return True
        if int(provenance.get("paperSupport") or 0) > 0:
            return True
        if int(provenance.get("parentTrunkCount") or 0) >= 2 and float(provenance.get("topicRelevance") or 0.0) >= 0.08:
            return True
        name = canonical_name.lower()
        return any(
            token in name
            for token in (
                "transformer",
                "attention",
                "embedding",
                "retrieval",
                "rag",
                "reasoning",
                "diffusion",
                "training",
                "fine-tuning",
                "optimization",
                "multimodal",
                "benchmark",
                "safety",
            )
        )

    def _select_path_nodes(
        self,
        topic_slug: str,
        *,
        nodes: list[dict[str, Any]],
        edges: list[LearningEdge],
        resource_links: list[dict[str, Any]],
    ) -> list[str]:
        node_map = {str(node.get("node_id")): node for node in nodes}
        concept_nodes = [
            str(node["node_id"])
            for node in nodes
            if str(node.get("node_type")) in {"concept", "technique"}
        ]
        ordering_edges = [
            edge for edge in edges
            if edge.edge_type in {"prerequisite", "recommended_before"} and edge.source_node_id != edge.target_node_id
        ]
        connected_nodes = {
            node_id
            for edge in ordering_edges
            for node_id in (edge.source_node_id, edge.target_node_id)
            if node_id in node_map
        }
        resource_nodes = {
            str(item.get("concept_node_id"))
            for item in resource_links
            if str(item.get("concept_node_id")) in node_map
        }
        topic_tokens = self._topic_tokens(topic_slug)
        anchor_nodes: set[str] = set()
        for node_id in concept_nodes:
            node = node_map[node_id]
            provenance = node.get("provenance") or {}
            if self._node_matches_topic(node, topic_tokens):
                anchor_nodes.add(node_id)
                continue
            if provenance.get("isTrunk") and float(provenance.get("topicRelevance") or 0.0) >= 0.18:
                anchor_nodes.add(node_id)
                continue
            if int(provenance.get("paperSupport") or 0) > 0 and float(provenance.get("topicRelevance") or 0.0) >= 0.10:
                anchor_nodes.add(node_id)

        quality_nodes = {
            node_id
            for node_id in concept_nodes
            if self._is_pathworthy_node(node_map[node_id], topic_tokens)
        }
        has_ordering_graph = bool(ordering_edges)
        if has_ordering_graph:
            candidate_seed_nodes = connected_nodes | (resource_nodes & connected_nodes) | (anchor_nodes & connected_nodes)
        else:
            candidate_seed_nodes = connected_nodes | resource_nodes | anchor_nodes
        candidate_nodes = (candidate_seed_nodes & quality_nodes) & set(concept_nodes)
        if candidate_nodes:
            return [
                node_id
                for node_id in concept_nodes
                if node_id in candidate_nodes
            ]

        ranked = sorted(
            concept_nodes,
            key=lambda node_id: (
                0 if (node_map[node_id].get("provenance") or {}).get("isTrunk") else 1,
                -float((node_map[node_id].get("provenance") or {}).get("topicRelevance") or 0.0),
                -int((node_map[node_id].get("provenance") or {}).get("paperSupport") or 0),
                -int((node_map[node_id].get("provenance") or {}).get("relationDegree") or 0),
                str(node_map[node_id].get("canonical_name") or ""),
            ),
        )
        return ranked[:12]

    def _preferred_rank_map(self, topic_slug: str) -> dict[str, int]:
        return {
            str(item).strip().lower(): idx
            for idx, item in enumerate(topic_preferred_prerequisites(topic_slug))
            if str(item).strip()
        }

    def _path_order_key(
        self,
        topic_slug: str,
        node_id: str,
        *,
        node_map: dict[str, Any] | None = None,
    ) -> tuple[Any, ...]:
        node = (node_map or {}).get(node_id, {}) if node_map else {}
        preferred_rank = self._preferred_rank_map(topic_slug).get(
            str(node.get("canonical_name") or "").strip().lower(),
            10_000,
        )
        difficulty_score = float(node.get("difficulty_score") or 0.0)
        canonical_name = str(node.get("canonical_name") or node_id)
        return (preferred_rank, difficulty_score, canonical_name)

    def topological_order(
        self,
        topic_slug: str,
        topic_nodes: list[str],
        edges: list[LearningEdge],
        *,
        node_map: dict[str, Any] | None = None,
    ) -> list[str]:
        topic_node_set = set(topic_nodes)
        ordering_edges = [
            edge for edge in edges
            if edge.edge_type in {"prerequisite", "recommended_before"} and edge.source_node_id != edge.target_node_id
            and edge.source_node_id in topic_node_set and edge.target_node_id in topic_node_set
        ]
        adjacency: dict[str, list[str]] = defaultdict(list)
        indegree: dict[str, int] = {node: 0 for node in topic_nodes}
        for edge in ordering_edges:
            adjacency[edge.source_node_id].append(edge.target_node_id)
            indegree[edge.target_node_id] = indegree.get(edge.target_node_id, 0) + 1
            indegree.setdefault(edge.source_node_id, 0)

        queue = deque(
            sorted(
                (node for node in topic_nodes if indegree.get(node, 0) == 0),
                key=lambda node_id: self._path_order_key(topic_slug, node_id, node_map=node_map),
            )
        )
        ordered: list[str] = []
        while queue:
            node = queue.popleft()
            ordered.append(node)
            for neighbor in adjacency.get(node, []):
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)
            if queue:
                queue = deque(
                    sorted(
                        queue,
                        key=lambda node_id: self._path_order_key(topic_slug, node_id, node_map=node_map),
                    )
                )

        if len(ordered) == len(topic_nodes):
            return ordered

        components = _tarjan_scc(topic_nodes, adjacency)
        flattened: list[str] = []
        for component in components:
            component.sort(key=lambda node_id: self._path_order_key(topic_slug, node_id, node_map=node_map))
            flattened.extend(component)
        seen: set[str] = set()
        result: list[str] = []
        for node in ordered + flattened:
            if node not in seen:
                seen.add(node)
                result.append(node)
        return result

    def generate_path(
        self,
        topic_slug: str,
        *,
        nodes: list[dict[str, Any]],
        edges: list[LearningEdge],
        resource_links: list[dict[str, Any]],
        approved_only: bool = True,
    ) -> LearningPath:
        node_map = {str(node.get("node_id")): node for node in nodes}
        concept_nodes = self._select_path_nodes(
            topic_slug,
            nodes=nodes,
            edges=edges,
            resource_links=resource_links,
        )
        ordered = self.topological_order(
            topic_slug,
            concept_nodes,
            edges,
            node_map=node_map,
        )
        prerequisite_closure = transitive_prerequisite_map(
            concept_nodes,
            edges,
            include_recommended=True,
        )

        beginner: list[dict[str, Any]] = []
        intermediate: list[dict[str, Any]] = []
        advanced: list[dict[str, Any]] = []
        stage_lookup = {
            "beginner": beginner,
            "intermediate": intermediate,
            "advanced": advanced,
        }
        resources_by_concept: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in resource_links:
            resources_by_concept[str(item.get("concept_node_id"))].append(item)

        for node_id in ordered:
            node = node_map.get(node_id, {})
            stage = str(node.get("stage") or node.get("difficulty_level") or "intermediate")
            bucket = stage_lookup.get(stage, intermediate)
            papers = sorted(
                resources_by_concept.get(node_id, []),
                key=lambda item: (
                    {"introduced_by": 0, "example_of": 1, "deepened_by": 2}.get(str(item.get("link_type")), 3),
                    str(item.get("resource_node_id")),
                ),
            )
            bucket.append(
                {
                    "nodeId": node_id,
                    "canonicalName": node.get("canonical_name"),
                    "difficultyLevel": node.get("difficulty_level"),
                    "prerequisiteNodeIds": prerequisite_closure.get(node_id, []),
                    "papers": papers,
                }
            )

        score = {
            "conceptCount": len(concept_nodes),
            "edgeCount": len(edges),
            "resourceLinkCount": len(resource_links),
            "transitivePrerequisiteCount": sum(len(items) for items in prerequisite_closure.values()),
            "approvedOnly": approved_only,
        }
        return LearningPath(
            path_id=f"lg_path_{uuid4().hex[:12]}",
            topic_slug=topic_slug,
            nodes=ordered,
            stages={
                "beginner": beginner,
                "intermediate": intermediate,
                "advanced": advanced,
            },
            score=score,
            status="approved" if approved_only else "pending",
            provenance={
                "topicSlug": topic_slug,
                "approvedOnly": approved_only,
                "algorithm": "topological_sort_transitive_v2",
            },
        )
