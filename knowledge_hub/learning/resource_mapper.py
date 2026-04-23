"""Paper/resource attachment helpers for learning graph."""

from __future__ import annotations

from uuid import uuid4


INTRO_TOKENS = ("survey", "overview", "tutorial", "guide", "introduction")
EXAMPLE_TOKENS = ("benchmark", "case study", "example")


def infer_resource_link(paper: dict, difficulty_level: str) -> tuple[str, str, float]:
    title = str(paper.get("title", "")).lower()
    if any(token in title for token in INTRO_TOKENS):
        return "introduced_by", "beginner", 0.82
    if any(token in title for token in EXAMPLE_TOKENS):
        return "example_of", "intermediate", 0.70
    if difficulty_level == "beginner":
        return "deepened_by", "intermediate", 0.68
    if difficulty_level == "intermediate":
        return "deepened_by", "advanced", 0.74
    return "deepened_by", "advanced", 0.80


def build_resource_link(
    concept_node_id: str,
    resource_node_id: str,
    paper: dict,
    difficulty_level: str,
    topic_slug: str,
) -> dict:
    link_type, reading_stage, confidence = infer_resource_link(paper, difficulty_level)
    return {
        "linkId": f"lg_res_{uuid4().hex[:12]}",
        "conceptNodeId": concept_node_id,
        "resourceNodeId": resource_node_id,
        "linkType": link_type,
        "readingStage": reading_stage,
        "confidence": confidence,
        "status": "pending",
        "provenance": {
            "topicSlug": topic_slug,
            "derivedFrom": "paper_concept_association",
            "paperId": paper.get("arxiv_id"),
        },
    }
