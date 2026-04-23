"""Suggestion-only patch generator for learning content gaps."""

from __future__ import annotations

from typing import Any


def suggest_patch(
    topic: str,
    *,
    gap_payload: dict[str, Any],
    session_note_path: str,
    max_suggestions: int = 10,
) -> dict[str, Any]:
    missing = gap_payload.get("missingTrunks") if isinstance(gap_payload.get("missingTrunks"), list) else []
    weak_edges = gap_payload.get("weakEdges") if isinstance(gap_payload.get("weakEdges"), list) else []
    evidence_gaps = gap_payload.get("evidenceGaps") if isinstance(gap_payload.get("evidenceGaps"), list) else []

    suggestions: list[dict[str, Any]] = []
    for idx, item in enumerate(missing[: max_suggestions], start=1):
        cid = str(item.get("canonical_id", "unknown"))
        text = (
            f"### Gap Fill: {cid}\n"
            f"- 목적: `{cid}` 개념을 현재 학습 맥락({topic})에 연결\n"
            "- 제안: 이 개념의 정의 1문장 + 기존 개념과의 relation 2개를 추가\n"
            "- 근거: evidence_ptr 형식(path/heading/block_id/snippet_hash)으로 최소 1개 첨부\n"
        )
        suggestions.append(
            {
                "id": f"s_missing_{idx}",
                "targetPath": session_note_path,
                "sectionTitle": "## Concept Map Edges",
                "reason": "missing_target_trunk",
                "confidence": round(float(item.get("priority", 0.6)), 4),
                "mode": "proposal-only",
                "patchText": text,
            }
        )

    offset = len(suggestions)
    for idx, edge in enumerate(weak_edges[: max(0, max_suggestions - offset)], start=1):
        text = (
            "### Weak Edge Reinforcement\n"
            f"- source: `{edge.get('sourceCanonicalId', '-')}`\n"
            f"- relation: `{edge.get('relationNorm', '-')}`\n"
            f"- target: `{edge.get('targetCanonicalId', '-')}`\n"
            "- 제안: relation_norm을 명확한 enum(causes/enables/part_of/...)으로 교체하고 "
            "confidence 및 evidence_ptr를 추가\n"
        )
        suggestions.append(
            {
                "id": f"s_weak_{idx}",
                "targetPath": session_note_path,
                "sectionTitle": "## Concept Map Edges",
                "reason": "weak_edge_or_low_confidence",
                "confidence": round(float(edge.get("priority", 0.5)), 4),
                "mode": "proposal-only",
                "patchText": text,
            }
        )

    if not suggestions and evidence_gaps:
        for idx, edge in enumerate(evidence_gaps[: min(3, max_suggestions)], start=1):
            text = (
                "### Evidence Pointer Addendum\n"
                f"- edge: `{edge.get('sourceCanonicalId', '-')}` -> `{edge.get('relationNorm', '-')}` -> `{edge.get('targetCanonicalId', '-')}`\n"
                "- 제안: evidence_ptr: path=...;heading=...;block_id=...;snippet=... 형식으로 근거 추가\n"
            )
            suggestions.append(
                {
                    "id": f"s_evidence_{idx}",
                    "targetPath": session_note_path,
                    "sectionTitle": "## Concept Map Edges",
                    "reason": "missing_evidence_ptr",
                    "confidence": 0.55,
                    "mode": "proposal-only",
                    "patchText": text,
                }
            )

    return {
        "mode": "proposal-only",
        "suggestionCount": len(suggestions),
        "suggestions": suggestions[:max_suggestions],
        "sourceRefs": [session_note_path] if session_note_path else [],
    }
