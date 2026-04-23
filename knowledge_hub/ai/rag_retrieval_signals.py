from __future__ import annotations

from typing import Any, Callable

from knowledge_hub.core.models import SearchResult
from knowledge_hub.knowledge.features import reference_prior_boost, source_trust_score


def build_retrieval_ranking_signals(
    result: SearchResult,
    *,
    snapshot: dict[str, Any] | None,
    resolve_result_note_row_fn: Callable[[SearchResult], dict[str, Any] | None],
    resolve_result_quality_fn: Callable[[SearchResult], tuple[str, dict[str, Any]]],
    json_load_dict_fn: Callable[[Any], dict[str, Any]],
    safe_float_fn: Callable[[Any, float], float],
) -> dict[str, Any]:
    metadata = result.metadata or {}
    snapshot = snapshot or {}
    note_row = resolve_result_note_row_fn(result)
    note_meta = json_load_dict_fn((note_row or {}).get("metadata"))

    quality_flag, quality_payload = resolve_result_quality_fn(result)
    quality_boost_map = {
        "ok": 0.04,
        "needs_review": -0.025,
        "reject": -0.05,
        "unscored": 0.0,
    }
    quality_boost = quality_boost_map.get(quality_flag, 0.0)

    source_trust = safe_float_fn(
        snapshot.get("source_trust_score"),
        source_trust_score(
            source_vendor=str(metadata.get("source_vendor") or note_meta.get("source_vendor") or "").strip(),
            source_channel=str(metadata.get("source_channel") or note_meta.get("source_channel") or "").strip(),
            source_type=str(metadata.get("source_type") or (note_row or {}).get("source_type") or "").strip(),
        ),
    )
    source_trust_boost = max(-0.03, min(0.025, 0.06 * (source_trust - 0.6)))

    reference_role = str(
        metadata.get("reference_role")
        or note_meta.get("reference_role")
        or (snapshot.get("payload_json") or {}).get("referenceRole")
        or ""
    ).strip()
    reference_tier = str(
        metadata.get("reference_tier")
        or note_meta.get("reference_tier")
        or (snapshot.get("payload_json") or {}).get("referenceTier")
        or ""
    ).strip()
    reference_prior = safe_float_fn(
        (snapshot.get("payload_json") or {}).get("referencePriorBoost"),
        reference_prior_boost(reference_role=reference_role, reference_tier=reference_tier),
    )

    contradiction = safe_float_fn(snapshot.get("contradiction_score"), 0.0)
    contradiction_penalty = 0.15 * contradiction
    base_score = safe_float_fn(result.score, 0.0)
    adjusted_score = max(
        0.0,
        min(1.0, base_score + quality_boost + source_trust_boost + reference_prior - contradiction_penalty),
    )
    return {
        "quality_flag": quality_flag,
        "quality_boost": round(quality_boost, 6),
        "source_trust_score": round(source_trust, 6),
        "source_trust_boost": round(source_trust_boost, 6),
        "reference_role": reference_role,
        "reference_tier": reference_tier,
        "reference_prior_boost": round(reference_prior, 6),
        "contradiction_penalty": round(contradiction_penalty, 6),
        "quality_version": str(quality_payload.get("version") or ""),
        "retrieval_adjusted_score": round(adjusted_score, 6),
    }
