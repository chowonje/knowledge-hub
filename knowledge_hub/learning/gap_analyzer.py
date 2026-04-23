"""Gap analysis helpers for ontology-first learning flows."""

from __future__ import annotations

from typing import Any

from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.learning.assessor import parse_edges_from_session
from knowledge_hub.learning.mapper import generate_learning_map
from knowledge_hub.learning.resolver import EntityResolver


def analyze_gaps(
    db: SQLiteDatabase,
    topic: str,
    *,
    session_content: str = "",
    session_note_path: str = "",
    session_id: str = "",
    source: str = "all",
    days: int = 180,
    top_k: int = 12,
    min_confidence: float = 0.7,
    allow_external: bool = False,
    run_id: str | None = None,
) -> dict[str, Any]:
    map_result = generate_learning_map(
        db=db,
        topic=topic,
        source=source,
        days=days,
        top_k=top_k,
        allow_external=allow_external,
        run_id=run_id,
    )
    trunks = map_result.get("trunks") if isinstance(map_result.get("trunks"), list) else []
    target_trunk_ids = [str(item.get("canonical_id", "")).strip() for item in trunks if str(item.get("canonical_id", "")).strip()]

    edges, parse_errors, _ = parse_edges_from_session(session_content or "", session_note_path=session_note_path or "")
    resolver = EntityResolver(db)

    weak_edges: list[dict[str, Any]] = []
    evidence_gaps: list[dict[str, Any]] = []
    used_targets: set[str] = set()
    normalization_failures = 0
    confidence_sum = 0.0
    confidence_count = 0

    for edge in edges:
        src = resolver.resolve(edge.source_canonical_id, entity_type="concept")
        tgt = resolver.resolve(edge.target_canonical_id, entity_type="concept")
        src_id = src.canonical_id if src else "unknown"
        tgt_id = tgt.canonical_id if tgt else "unknown"
        if src:
            used_targets.add(src.canonical_id)
        if tgt:
            used_targets.add(tgt.canonical_id)
        if not src or not tgt:
            normalization_failures += 1

        confidence = float(edge.confidence or 0.0)
        confidence_sum += confidence
        confidence_count += 1

        issues: list[str] = []
        if not src or not tgt:
            issues.append("normalization_failed")
        if edge.relation_norm == "unknown_relation":
            issues.append("unknown_relation")
        if confidence < min_confidence * 5.0:
            issues.append("low_confidence")
        if not edge.evidence_ptrs:
            issues.append("missing_evidence_ptr")

        if issues:
            record = {
                "sourceCanonicalId": src_id,
                "relationNorm": edge.relation_norm,
                "targetCanonicalId": tgt_id,
                "confidence": round(confidence, 4),
                "issues": issues,
                "priority": round(min(1.0, 0.2 * len(issues) + (0.3 if "normalization_failed" in issues else 0.0)), 4),
            }
            weak_edges.append(record)
            if "missing_evidence_ptr" in issues:
                evidence_gaps.append(record)

    missing_trunks: list[dict[str, Any]] = []
    for item in trunks:
        canonical_id = str(item.get("canonical_id", "")).strip()
        if not canonical_id:
            continue
        if canonical_id in used_targets:
            continue
        score = float(item.get("trunkScore", 0.0) or 0.0)
        missing_trunks.append(
            {
                "canonical_id": canonical_id,
                "display_name": str(item.get("display_name", canonical_id)),
                "priority": round(min(1.0, 0.5 + score), 4),
                "reason": "target trunk not covered in session edges",
            }
        )

    missing_trunks.sort(key=lambda x: x["priority"], reverse=True)
    weak_edges.sort(key=lambda x: (x["priority"], x["confidence"]), reverse=True)
    evidence_gaps.sort(key=lambda x: (x["priority"], x["confidence"]), reverse=True)

    target_count = len(target_trunk_ids)
    covered_count = len([item for item in target_trunk_ids if item in used_targets])
    avg_gap_confidence = confidence_sum / confidence_count if confidence_count else 0.0
    normalization_failure_rate = normalization_failures / max(1, len(edges))

    return {
        "status": "ok",
        "summary": {
            "targetTrunkCount": target_count,
            "coveredTrunkCount": covered_count,
            "missingTrunkCount": len(missing_trunks),
            "weakEdgeCount": len(weak_edges),
            "evidenceGapCount": len(evidence_gaps),
            "avgGapConfidence": round(avg_gap_confidence, 6),
            "normalizationFailureRate": round(normalization_failure_rate, 6),
            "parseErrorCount": len(parse_errors),
        },
        "targetTrunkIds": target_trunk_ids,
        "missingTrunks": missing_trunks[:50],
        "weakEdges": weak_edges[:100],
        "evidenceGaps": evidence_gaps[:100],
        "parseErrors": parse_errors[:50],
        "sourceRefs": [session_note_path] if session_note_path else [],
        "mapStatus": map_result.get("status"),
        "policy": map_result.get("policy"),
    }
