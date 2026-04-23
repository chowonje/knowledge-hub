"""Deterministic difficulty scoring for learning graph nodes."""

from __future__ import annotations

from typing import Any


FOUNDATIONAL_TOKENS = {
    "linear algebra",
    "optimization",
    "probability",
    "statistics",
    "neural network",
    "attention",
    "transformer",
    "language model",
}

SURVEY_TOKENS = ("survey", "overview", "tutorial", "review")
ADVANCED_TOKENS = ("benchmark", "mixture", "diffusion", "moe", "multimodal", "alignment", "agent")


def map_difficulty_level(score: float) -> str:
    if score < 0.40:
        return "beginner"
    if score < 0.70:
        return "intermediate"
    return "advanced"


def _paper_sophistication(papers: list[dict[str, Any]]) -> float:
    if not papers:
        return 0.2
    values: list[float] = []
    for paper in papers:
        title = str(paper.get("title", "")).lower()
        value = 0.65
        if any(token in title for token in SURVEY_TOKENS):
            value = 0.35
        elif any(token in title for token in ADVANCED_TOKENS):
            value = 0.85
        year = paper.get("year")
        if isinstance(year, int) and year >= 2024:
            value = min(1.0, value + 0.05)
        values.append(value)
    return sum(values) / max(1, len(values))


def _lexical_abstraction(name: str) -> float:
    token = str(name or "").strip().lower()
    if not token:
        return 0.5
    if token in FOUNDATIONAL_TOKENS:
        return 0.2
    word_count = max(1, len(token.split()))
    score = min(1.0, 0.28 + (word_count * 0.10))
    if any(piece in token for piece in ADVANCED_TOKENS):
        score = min(1.0, score + 0.15)
    return score


def score_difficulty(
    *,
    canonical_name: str,
    graph_depth: float,
    ontology_centrality: float,
    papers: list[dict[str, Any]],
    evidence_diversity: float,
) -> dict[str, Any]:
    paper_proxy = _paper_sophistication(papers)
    lexical = _lexical_abstraction(canonical_name)
    score = (
        0.30 * max(0.0, min(1.0, graph_depth))
        + 0.20 * max(0.0, min(1.0, ontology_centrality))
        + 0.20 * paper_proxy
        + 0.15 * lexical
        + 0.15 * max(0.0, min(1.0, evidence_diversity))
    )
    level = map_difficulty_level(score)
    return {
        "difficulty_score": round(float(score), 6),
        "difficulty_level": level,
        "stage": level,
        "components": {
            "graphDepth": round(float(graph_depth), 6),
            "ontologyCentrality": round(float(ontology_centrality), 6),
            "paperSophistication": round(float(paper_proxy), 6),
            "lexicalAbstraction": round(float(lexical), 6),
            "evidenceDiversity": round(float(evidence_diversity), 6),
        },
    }
