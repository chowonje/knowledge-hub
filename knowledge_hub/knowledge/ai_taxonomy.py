"""AI knowledge classification helpers.

This module adds a bounded, rule-first AI taxonomy layer on top of existing
ontology concept entities. It intentionally classifies into properties rather
than introducing new top-level entity types.
"""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

from knowledge_hub.learning.resolver import normalize_term


AI_KNOWLEDGE_KINDS = (
    "theory",
    "algorithm",
    "software_system",
    "hardware_system",
    "data",
    "evaluation",
    "operation",
    "application",
)

AI_FACET_VOCAB: dict[str, tuple[str, ...]] = {
    "abstraction_level": ("principle", "component", "system", "product"),
    "lifecycle_stage": ("research", "training", "inference", "evaluation", "operation"),
    "resource_axis": ("compute", "memory", "bandwidth", "data", "energy", "latency", "cost"),
    "modality": ("text", "image", "audio", "video", "multimodal", "action"),
    "evidence_status": ("speculative", "experimental", "replicated", "production"),
}

AI_SUBTYPE_VOCAB: dict[str, tuple[str, ...]] = {
    "theory": (
        "information_theory",
        "optimization",
        "statistical_learning",
        "complexity",
        "representation_learning",
        "generalization",
    ),
    "algorithm": (
        "transformer",
        "attention",
        "diffusion",
        "reinforcement_learning",
        "search",
        "inference_optimization",
        "quantization",
    ),
    "software_system": (
        "training_stack",
        "serving_stack",
        "orchestration",
        "agent_runtime",
        "vector_db",
        "compiler",
    ),
    "hardware_system": (
        "gpu",
        "tpu",
        "npu",
        "memory_hierarchy",
        "interconnect",
        "storage",
        "power_thermal",
    ),
    "data": (
        "dataset",
        "synthetic_data",
        "labeling",
        "curation",
        "filtering",
        "licensing",
    ),
    "evaluation": (
        "benchmark",
        "metric",
        "failure_mode",
        "robustness",
        "safety_eval",
    ),
    "operation": (
        "deployment",
        "monitoring",
        "cost",
        "latency",
        "scaling",
        "reliability",
    ),
    "application": (
        "coding_agent",
        "robotics",
        "multimodal_assistant",
        "search",
        "biotech",
    ),
}

_ENTITY_PATTERNS: tuple[tuple[str, str, float], ...] = (
    ("algorithm", r"\bflashattention(?:-?\d+)?\b", 0.96),
    ("algorithm", r"\btransformer(?:s)?\b", 0.9),
    ("algorithm", r"\bdiffusion\b", 0.9),
    ("algorithm", r"\bquanti[sz]ation\b", 0.88),
    ("algorithm", r"\brlhf\b|\bppo\b|\breinforcement learning\b", 0.88),
    ("algorithm", r"\bpre[- ]?training\b|\bpretrain(?:ing)?\b|\bactor-critic\b|\btransducer\b", 0.8),
    ("algorithm", r"\bbeam search\b|\bsearch\b", 0.78),
    ("algorithm", r"\battention\b", 0.72),
    ("software_system", r"\bvllm\b|\btensorrt\b|\btriton\b|\bkubernetes\b|\bcompiler\b|\bruntime\b", 0.9),
    ("software_system", r"\bserving\b|\borchestrat(?:ion|or)\b|\bvector db\b|\bvector database\b", 0.82),
    ("hardware_system", r"\bh100\b|\bh200\b|\ba100\b|\bl40s\b|\bgpu\b|\btpu\b|\bnpu\b", 0.96),
    ("hardware_system", r"\bhbm\b|\bnvlink\b|\binterconnect\b|\bmemory hierarchy\b|\bpower\b|\bthermal\b", 0.84),
    ("data", r"\bdataset\b|\bcorpus\b|\bsynthetic data\b|\blabel(?:ing|led)\b|\bannotation\b|\blicens(?:e|ing)\b", 0.88),
    ("evaluation", r"\bbenchmark\b|\bmetric\b|\bmmlu\b|\brouge\b|\bbleu\b|\baccuracy\b|\brobustness\b|\bsafety eval\b", 0.9),
    ("operation", r"\bdeployment\b|\bmonitoring\b|\blatency\b|\bthroughput\b|\bscaling\b|\breliability\b|\bcost\b", 0.88),
    ("application", r"\bcoding agent\b|\brobotics\b|\bassistant\b|\bsearch\b|\bbiotech\b|\bcaption(?:ing)?\b", 0.84),
    ("theory", r"\binformation theory\b|\boptimization\b|\bgeneralization\b|\bcomplexity\b|\bstatistical learning\b", 0.88),
)

_SUBTYPE_PATTERNS: dict[str, tuple[tuple[str, str], ...]] = {
    "algorithm": (
        ("attention", r"\bflashattention(?:-?\d+)?\b|\battention\b"),
        ("transformer", r"\btransformer(?:s)?\b"),
        ("diffusion", r"\bdiffusion\b"),
        ("reinforcement_learning", r"\brlhf\b|\bppo\b|\breinforcement learning\b"),
        ("search", r"\bbeam search\b|\bsearch\b"),
        ("quantization", r"\bquanti[sz]ation\b"),
        ("inference_optimization", r"\binference optimization\b|\bkv cache\b|\bpagedattention\b"),
    ),
    "software_system": (
        ("agent_runtime", r"\bagent runtime\b|\bruntime\b"),
        ("compiler", r"\bcompiler\b|\btriton\b"),
        ("vector_db", r"\bvector db\b|\bvector database\b"),
        ("serving_stack", r"\bserving\b|\bvllm\b|\btensorrt\b"),
        ("training_stack", r"\btraining stack\b"),
        ("orchestration", r"\borchestrat(?:ion|or)\b|\bkubernetes\b"),
    ),
    "hardware_system": (
        ("gpu", r"\bh100\b|\bh200\b|\ba100\b|\bl40s\b|\bgpu\b"),
        ("tpu", r"\btpu\b"),
        ("npu", r"\bnpu\b"),
        ("memory_hierarchy", r"\bhbm\b|\bmemory hierarchy\b|\bkv cache\b"),
        ("interconnect", r"\bnvlink\b|\binterconnect\b"),
        ("storage", r"\bstorage\b"),
        ("power_thermal", r"\bpower\b|\bthermal\b"),
    ),
    "data": (
        ("dataset", r"\bdataset\b|\bcorpus\b"),
        ("synthetic_data", r"\bsynthetic data\b"),
        ("labeling", r"\blabel(?:ing|led)\b|\bannotation\b"),
        ("curation", r"\bcuration\b"),
        ("filtering", r"\bfiltering\b"),
        ("licensing", r"\blicens(?:e|ing)\b"),
    ),
    "evaluation": (
        ("benchmark", r"\bbenchmark\b|\bmmlu\b"),
        ("metric", r"\bmetric\b|\baccuracy\b|\bbleu\b|\brouge\b"),
        ("robustness", r"\brobustness\b"),
        ("failure_mode", r"\bfailure mode\b|\bhallucination\b"),
        ("safety_eval", r"\bsafety eval\b|\bsafety evaluation\b"),
    ),
    "operation": (
        ("deployment", r"\bdeployment\b"),
        ("monitoring", r"\bmonitoring\b"),
        ("latency", r"\blatency\b"),
        ("cost", r"\bcost\b"),
        ("scaling", r"\bscaling\b"),
        ("reliability", r"\breliability\b"),
    ),
    "application": (
        ("coding_agent", r"\bcoding agent\b"),
        ("robotics", r"\brobotics\b"),
        ("multimodal_assistant", r"\bassistant\b|\bmultimodal\b"),
        ("search", r"\bsearch\b"),
        ("biotech", r"\bbiotech\b"),
    ),
    "theory": (
        ("information_theory", r"\binformation theory\b"),
        ("optimization", r"\boptimization\b"),
        ("statistical_learning", r"\bstatistical learning\b"),
        ("complexity", r"\bcomplexity\b"),
        ("representation_learning", r"\brepresentation learning\b"),
        ("generalization", r"\bgeneralization\b"),
    ),
}

_TRAINING_RE = re.compile(r"\btraining\b|\bpretrain(?:ing)?\b|\bfine[- ]?tuning\b|\boptimizer\b")
_INFERENCE_RE = re.compile(r"\binference\b|\bserving\b|\bdecode\b|\bkv cache\b|\blatency\b|\bthroughput\b")
_EVALUATION_RE = re.compile(r"\beval(?:uation)?\b|\bbenchmark\b|\bmetric\b|\brobustness\b|\baccuracy\b")
_OPERATION_RE = re.compile(r"\bdeployment\b|\bmonitoring\b|\breliability\b|\bscaling\b|\bcost\b")
_RESEARCH_RE = re.compile(r"\bstudy\b|\bpaper\b|\bresearch\b|\bmethod\b")

_RESOURCE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("compute", r"\bcompute\b|\bflops\b|\bthroughput\b"),
    ("memory", r"\bmemory\b|\bkv cache\b|\bhbm\b"),
    ("bandwidth", r"\bbandwidth\b|\bnvlink\b|\binterconnect\b"),
    ("data", r"\bdataset\b|\bdata\b|\bcorpus\b"),
    ("energy", r"\benergy\b|\bpower\b|\bthermal\b"),
    ("latency", r"\blatency\b"),
    ("cost", r"\bcost\b"),
)

_MODALITY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("text", r"\btext\b|\blanguage\b|\bllm\b"),
    ("image", r"\bimage\b|\bvision\b|\bcaption(?:ing)?\b"),
    ("audio", r"\baudio\b|\bspeech\b"),
    ("video", r"\bvideo\b"),
    ("multimodal", r"\bmultimodal\b"),
    ("action", r"\baction\b|\brobotics\b|\bagent\b"),
)


def default_ai_knowledge_classification() -> dict[str, Any]:
    return {
        "version": "ai-knowledge-v1",
        "knowledge_kinds": list(AI_KNOWLEDGE_KINDS),
        "subtype_vocab": {key: list(values) for key, values in AI_SUBTYPE_VOCAB.items()},
        "facets": {key: list(values) for key, values in AI_FACET_VOCAB.items()},
    }


def validate_ai_knowledge_profile_schema(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["knowledge_classification must be an object"]
    kinds = payload.get("knowledge_kinds")
    if kinds is not None:
        if not isinstance(kinds, list):
            errors.append("knowledge_classification.knowledge_kinds must be a list")
        else:
            invalid = sorted({str(item) for item in kinds if str(item) not in AI_KNOWLEDGE_KINDS})
            if invalid:
                errors.append(f"knowledge_classification.knowledge_kinds invalid values: {', '.join(invalid)}")
    subtype_vocab = payload.get("subtype_vocab")
    if subtype_vocab is not None:
        if not isinstance(subtype_vocab, dict):
            errors.append("knowledge_classification.subtype_vocab must be an object")
        else:
            invalid = sorted(key for key in subtype_vocab if key not in AI_KNOWLEDGE_KINDS)
            if invalid:
                errors.append(f"knowledge_classification.subtype_vocab invalid kinds: {', '.join(invalid)}")
            for key, value in subtype_vocab.items():
                if not isinstance(value, list):
                    errors.append(f"knowledge_classification.subtype_vocab.{key} must be a list")
    facets = payload.get("facets")
    if facets is not None:
        if not isinstance(facets, dict):
            errors.append("knowledge_classification.facets must be an object")
        else:
            invalid_keys = sorted(key for key in facets if key not in AI_FACET_VOCAB)
            if invalid_keys:
                errors.append(f"knowledge_classification.facets invalid keys: {', '.join(invalid_keys)}")
            for key, value in facets.items():
                if not isinstance(value, list):
                    errors.append(f"knowledge_classification.facets.{key} must be a list")
                    continue
                invalid_values = sorted({str(item) for item in value if str(item) not in AI_FACET_VOCAB.get(key, ())})
                if invalid_values:
                    errors.append(
                        f"knowledge_classification.facets.{key} invalid values: {', '.join(invalid_values)}"
                    )
    return errors


def _normalized_context(
    *,
    canonical_name: str,
    aliases: list[str] | None = None,
    title: str = "",
    domain: str = "",
    tags: list[str] | None = None,
    related_names: list[str] | None = None,
    relation_predicates: list[str] | None = None,
    source_type: str = "",
) -> str:
    parts = [
        canonical_name,
        title,
        domain,
        source_type,
        *(aliases or []),
        *(tags or []),
        *(related_names or []),
        *(relation_predicates or []),
    ]
    return normalize_term(" ".join(str(item or "") for item in parts if str(item or "").strip()))


def _determine_subtype(kind: str, normalized_context: str) -> str:
    for subtype, pattern in _SUBTYPE_PATTERNS.get(kind, ()):
        if re.search(pattern, normalized_context):
            return subtype
    return ""


def _infer_facets(kind: str, normalized_context: str, *, source_type: str = "") -> dict[str, Any]:
    facets: dict[str, Any] = {}
    abstraction_defaults = {
        "theory": "principle",
        "algorithm": "component",
        "software_system": "system",
        "hardware_system": "component",
        "data": "component",
        "evaluation": "system",
        "operation": "system",
        "application": "product",
    }
    if kind in abstraction_defaults:
        facets["abstraction_level"] = abstraction_defaults[kind]

    lifecycle: list[str] = []
    if source_type == "paper":
        lifecycle.append("research")
    if _TRAINING_RE.search(normalized_context):
        lifecycle.append("training")
    if _INFERENCE_RE.search(normalized_context):
        lifecycle.append("inference")
    if _EVALUATION_RE.search(normalized_context):
        lifecycle.append("evaluation")
    if _OPERATION_RE.search(normalized_context):
        lifecycle.append("operation")
    if not lifecycle and source_type == "paper" and _RESEARCH_RE.search(normalized_context):
        lifecycle.append("research")
    if lifecycle:
        facets["lifecycle_stage"] = lifecycle

    resource_axis = [name for name, pattern in _RESOURCE_PATTERNS if re.search(pattern, normalized_context)]
    if resource_axis:
        facets["resource_axis"] = resource_axis

    modality = [name for name, pattern in _MODALITY_PATTERNS if re.search(pattern, normalized_context)]
    if modality:
        facets["modality"] = modality

    if source_type == "paper":
        facets["evidence_status"] = "experimental"
    elif _OPERATION_RE.search(normalized_context):
        facets["evidence_status"] = "production"
    return facets


def classify_ai_concept(
    *,
    canonical_name: str,
    aliases: list[str] | None = None,
    title: str = "",
    domain: str = "",
    tags: list[str] | None = None,
    related_names: list[str] | None = None,
    relation_predicates: list[str] | None = None,
    source_type: str = "",
) -> dict[str, Any] | None:
    normalized_context = _normalized_context(
        canonical_name=canonical_name,
        aliases=aliases,
        title=title,
        domain=domain,
        tags=tags,
        related_names=related_names,
        relation_predicates=relation_predicates,
        source_type=source_type,
    )
    if not normalized_context:
        return None

    scores: dict[str, float] = defaultdict(float)
    reasons: dict[str, list[str]] = defaultdict(list)
    for kind, pattern, weight in _ENTITY_PATTERNS:
        if re.search(pattern, normalized_context):
            scores[kind] += weight
            reasons[kind].append(pattern)

    if not scores:
        return None

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_kind, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_score - second_score
    confidence = max(0.0, min(0.99, 0.45 + (0.4 * best_score) + (0.15 * max(0.0, margin))))
    if best_score < 0.7 or margin < 0.08:
        return None

    subtype = _determine_subtype(best_kind, normalized_context)
    facets = _infer_facets(best_kind, normalized_context, source_type=source_type)
    return {
        "knowledge_kind": best_kind,
        "subtype": subtype,
        "facets": facets,
        "classification_confidence": round(confidence, 6),
        "classification_source": "rule",
        "classification_reasons": reasons.get(best_kind, [])[:5],
    }


def merge_ai_classification_properties(
    existing_properties: dict[str, Any] | None,
    classification: dict[str, Any] | None,
) -> tuple[dict[str, Any], bool]:
    current = dict(existing_properties or {})
    if not classification:
        return current, False
    if str(current.get("classification_source") or "").strip().lower() == "manual":
        return current, False

    current_kind = str(current.get("knowledge_kind") or "").strip()
    current_confidence = float(current.get("classification_confidence") or 0.0)
    next_kind = str(classification.get("knowledge_kind") or "").strip()
    next_confidence = float(classification.get("classification_confidence") or 0.0)
    if current_kind and current_kind != next_kind and current_confidence > next_confidence:
        return current, False

    merged = dict(current)
    merged["knowledge_kind"] = next_kind
    subtype = str(classification.get("subtype") or "").strip()
    if subtype:
        merged["subtype"] = subtype
    elif "subtype" in merged:
        merged.pop("subtype", None)
    facets = classification.get("facets") if isinstance(classification.get("facets"), dict) else {}
    if facets:
        merged["facets"] = facets
    merged["classification_confidence"] = round(next_confidence, 6)
    merged["classification_source"] = str(classification.get("classification_source") or "rule")
    reasons = classification.get("classification_reasons")
    if isinstance(reasons, list) and reasons:
        merged["classification_reasons"] = [str(item) for item in reasons if str(item).strip()][:5]
    changed = merged != current
    return merged, changed
