"""Paper lane taxonomy and seed helpers."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

LANE_REVIEW_STATUSES = ("seeded", "reviewed", "locked")
PRIMARY_LANES = (
    "memory_inference",
    "architecture",
    "agent",
    "rag_retrieval",
    "multimodal",
    "safety_evaluation",
)


@dataclass(frozen=True)
class LaneDefinition:
    slug: str
    title: str
    description: str
    inclusion_criteria: str
    questions: tuple[str, ...]
    seed_tags: tuple[str, ...]
    keywords: tuple[str, ...]


LANE_DEFINITIONS: dict[str, LaneDefinition] = {
    "memory_inference": LaneDefinition(
        slug="memory_inference",
        title="Memory Inference",
        description="Inference-time memory hierarchy, KV cache, compression, bandwidth, and serving efficiency work.",
        inclusion_criteria="Use when the core contribution is about inference-time memory, cache, quantization, throughput, or serving bottlenecks.",
        questions=(
            "What is the main inference-time bottleneck this paper targets?",
            "How does it trade memory savings against quality loss?",
            "Does it change KV cache layout, compression, or bandwidth usage?",
            "What hardware or serving assumptions does it rely on?",
            "What is the deployment risk if applied to real systems?",
        ),
        seed_tags=("kv", "cache", "quant", "throughput", "serving", "bandwidth", "compression"),
        keywords=(
            "kv",
            "kv cache",
            "cache",
            "cached",
            "quant",
            "quantization",
            "compression",
            "compress",
            "throughput",
            "serving",
            "bandwidth",
            "latency",
            "memory hierarchy",
            "prefill",
            "decode",
            "inference",
        ),
    ),
    "architecture": LaneDefinition(
        slug="architecture",
        title="Architecture",
        description="Model architecture, attention alternatives, routing, and long-context design changes.",
        inclusion_criteria="Use when the main contribution changes model architecture, attention structure, routing, or sequence modeling.",
        questions=(
            "What architectural change does the paper introduce?",
            "What baseline architecture is it replacing or extending?",
            "How does the architecture affect scaling or context length?",
            "What tradeoffs appear in quality, latency, or training cost?",
            "Is this a practical replacement or a research direction only?",
        ),
        seed_tags=("attention", "mamba", "moe", "routing", "long-context", "transformer-variant"),
        keywords=(
            "architecture",
            "attention",
            "transformer",
            "transformer variant",
            "mamba",
            "ssm",
            "state space",
            "moe",
            "mixture of experts",
            "routing",
            "long context",
            "long-context",
            "context length",
            "recurrent",
        ),
    ),
    "agent": LaneDefinition(
        slug="agent",
        title="Agent",
        description="Agent planning, tool use, agent memory, GUI agents, and agent benchmarks.",
        inclusion_criteria="Use when the paper's core object is an agent loop, tool use, agent memory, or evaluation of agent behavior.",
        questions=(
            "What agent capability is being added or measured?",
            "How much of the gain comes from memory, tools, or planning?",
            "What benchmark or task family does it evaluate?",
            "What failure modes are still visible?",
            "Would this transfer to general-purpose agent systems?",
        ),
        seed_tags=("agent", "tool-use", "planning", "memory-agent", "gui-agent", "benchmark"),
        keywords=(
            "agent",
            "agentic",
            "tool use",
            "tool-use",
            "planning",
            "memory agent",
            "memory-augmented agent",
            "gui agent",
            "web agent",
            "benchmark",
            "multi-agent",
        ),
    ),
    "rag_retrieval": LaneDefinition(
        slug="rag_retrieval",
        title="RAG Retrieval",
        description="Retrieval, indexing, reranking, grounding, context selection, and RAG evaluation.",
        inclusion_criteria="Use when the main contribution is retrieval quality, grounding, indexing, reranking, or RAG system behavior.",
        questions=(
            "What retrieval failure or grounding problem does the paper target?",
            "How are candidates retrieved, reranked, or filtered?",
            "What relevance or answer-quality metric actually improves?",
            "What assumptions does the index or corpus make?",
            "Would this help local-first RAG or only hosted pipelines?",
        ),
        seed_tags=("rag", "retrieval", "rerank", "index", "context", "grounding"),
        keywords=(
            "rag",
            "retrieval",
            "retriever",
            "rerank",
            "reranker",
            "index",
            "indexing",
            "grounding",
            "context",
            "retrieval-augmented",
            "embedding",
        ),
    ),
    "multimodal": LaneDefinition(
        slug="multimodal",
        title="Multimodal",
        description="Vision-language, audio, robotics, video, and cross-modal learning or inference.",
        inclusion_criteria="Use when the paper's inputs or outputs are inherently multimodal or combine modalities beyond pure text.",
        questions=(
            "Which modalities are involved and how are they fused?",
            "What new capability appears because of multimodal reasoning?",
            "How does the paper evaluate cross-modal quality?",
            "What are the compute or data requirements?",
            "Is the method general-purpose or narrow-task specific?",
        ),
        seed_tags=("vision-language", "audio", "robotics", "video", "cross-modal"),
        keywords=(
            "vision-language",
            "vision language",
            "multimodal",
            "multi-modal",
            "audio",
            "speech",
            "robotics",
            "video",
            "cross-modal",
            "cross modal",
            "vlm",
        ),
    ),
    "safety_evaluation": LaneDefinition(
        slug="safety_evaluation",
        title="Safety Evaluation",
        description="Alignment, safety, reliability, red-teaming, and benchmark/evaluation work for model behavior.",
        inclusion_criteria="Use when the main contribution is evaluating, stress-testing, or aligning system behavior rather than capability alone.",
        questions=(
            "What risk or reliability dimension is being measured?",
            "Is this a benchmark, red-team method, or mitigation method?",
            "What counts as a meaningful improvement in safety terms?",
            "How robust is the evaluation to prompt or distribution shift?",
            "Would this matter in deployed systems or only in benchmarks?",
        ),
        seed_tags=("safety", "alignment", "evaluation", "benchmark", "red-team", "reliability"),
        keywords=(
            "safety",
            "alignment",
            "evaluation",
            "benchmark",
            "red team",
            "red-team",
            "reliability",
            "hallucination",
            "robustness",
            "trustworthy",
            "risk",
        ),
    ),
}

LANE_TITLES = {slug: definition.title for slug, definition in LANE_DEFINITIONS.items()}

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def now_lane_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_primary_lane(value: str | None) -> str | None:
    token = str(value or "").strip().lower()
    if not token:
        return None
    if token not in PRIMARY_LANES:
        raise ValueError(f"invalid primary lane: {value}")
    return token


def normalize_lane_review_status(value: str | None) -> str:
    token = str(value or "").strip().lower() or "seeded"
    if token not in LANE_REVIEW_STATUSES:
        raise ValueError(f"invalid lane review status: {value}")
    return token


def slugify_tag(value: str) -> str:
    token = _SLUG_RE.sub("-", str(value or "").strip().lower()).strip("-")
    return token


def normalize_secondary_tags(values: Any, *, limit: int = 6) -> list[str]:
    if values in (None, ""):
        return []
    raw_items: list[Any]
    if isinstance(values, str):
        stripped = values.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except Exception:
                parsed = [part.strip() for part in stripped.split(",")]
            raw_items = list(parsed if isinstance(parsed, list) else [parsed])
        else:
            raw_items = [part.strip() for part in stripped.split(",")]
    elif isinstance(values, (list, tuple, set)):
        raw_items = list(values)
    else:
        raw_items = [values]
    seen: set[str] = set()
    tags: list[str] = []
    for item in raw_items:
        token = slugify_tag(str(item or ""))
        if not token or token in seen:
            continue
        seen.add(token)
        tags.append(token)
        if len(tags) >= max(0, int(limit)):
            break
    return tags


def serialize_secondary_tags(values: Any, *, limit: int = 6) -> str:
    return json.dumps(normalize_secondary_tags(values, limit=limit), ensure_ascii=False)


def lane_title(slug: str) -> str:
    definition = LANE_DEFINITIONS.get(slug)
    if definition is None:
        return str(slug or "").replace("_", " ").title().strip() or "Unknown Lane"
    return definition.title


def lane_seed_tags(slug: str) -> tuple[str, ...]:
    definition = LANE_DEFINITIONS.get(slug)
    return definition.seed_tags if definition is not None else ()


def all_lane_slugs() -> tuple[str, ...]:
    return PRIMARY_LANES


def lane_hub_filename(slug: str) -> str:
    return f"{lane_title(slug)} Lane.md"


def _combined_signal_text(paper: dict[str, Any], paper_memory: dict[str, Any] | None = None) -> str:
    parts = [
        str(paper.get("title") or ""),
        str(paper.get("field") or ""),
        str(paper.get("notes") or ""),
    ]
    if paper_memory:
        parts.extend(
            [
                str(paper_memory.get("paper_core") or paper_memory.get("paperCore") or ""),
                str(paper_memory.get("method_core") or paper_memory.get("methodCore") or ""),
                str(paper_memory.get("evidence_core") or paper_memory.get("evidenceCore") or ""),
                str(paper_memory.get("limitations") or ""),
                str(paper_memory.get("search_text") or paper_memory.get("searchText") or ""),
            ]
        )
    return "\n".join(part for part in parts if str(part).strip()).lower()


def _match_score(text: str, keywords: tuple[str, ...]) -> int:
    score = 0
    for token in keywords:
        key = token.lower().strip()
        if not key:
            continue
        if key in text:
            score += 2 if " " in key or "-" in key else 1
    return score


def seed_lane_metadata(
    paper: dict[str, Any],
    paper_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text = _combined_signal_text(paper, paper_memory)
    scores = {
        slug: _match_score(text, definition.keywords)
        for slug, definition in LANE_DEFINITIONS.items()
    }
    primary_lane = max(PRIMARY_LANES, key=lambda slug: (scores.get(slug, 0), -PRIMARY_LANES.index(slug)))
    if scores.get(primary_lane, 0) <= 0:
        field = str(paper.get("field") or "").lower()
        title = str(paper.get("title") or "").lower()
        if any(token in f"{field} {title}" for token in ("vision", "audio", "robot", "video", "multimodal")):
            primary_lane = "multimodal"
        elif any(token in f"{field} {title}" for token in ("safety", "alignment", "reliability", "evaluation")):
            primary_lane = "safety_evaluation"
        elif "agent" in title:
            primary_lane = "agent"
        elif "retrieval" in title or "rag" in title:
            primary_lane = "rag_retrieval"
        elif any(token in title for token in ("cache", "quant", "serving", "inference", "kv")):
            primary_lane = "memory_inference"
        else:
            primary_lane = "architecture"

    secondary_counter: Counter[str] = Counter()
    for definition in LANE_DEFINITIONS.values():
        for tag in definition.seed_tags:
            if tag.lower() in text:
                secondary_counter[tag] += 2
    for slug, score in scores.items():
        if score <= 0:
            continue
        for tag in lane_seed_tags(slug):
            if tag.lower() in text:
                secondary_counter[tag] += 1
    secondary_tags = [tag for tag, _ in secondary_counter.most_common(6)]
    return {
        "primary_lane": primary_lane,
        "secondary_tags": normalize_secondary_tags(secondary_tags, limit=6),
        "lane_review_status": "seeded",
        "lane_updated_at": now_lane_timestamp(),
        "lane_scores": scores,
    }


def summarize_lane_tag_counts(papers: list[dict[str, Any]]) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for paper in papers:
        for tag in normalize_secondary_tags(paper.get("secondary_tags") or paper.get("secondary_tags_json") or []):
            counter[tag] += 1
    return counter.most_common()
