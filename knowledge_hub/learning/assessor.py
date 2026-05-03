"""Session template parsing and grading for Learning Coach."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from uuid import uuid4

import yaml

from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.learning.models import (
    GRADE_SCHEMA,
    AssessmentEdge,
    AssessmentScore,
    ConceptIdentity,
    ProgressGateDecision,
)
from knowledge_hub.learning.policy import evaluate_policy_for_payload, pointer_from_text
from knowledge_hub.learning.resolver import EntityResolver

RELATION_SYNONYMS = {
    "causes": ["cause", "causes", "원인", "야기", "유발"],
    "enables": ["enable", "enables", "가능", "가능하게", "촉진"],
    "part_of": ["part_of", "part of", "구성", "포함", "부분"],
    "contrasts": ["contrast", "contrasts", "대조", "차이", "반대"],
    "example_of": ["example", "example_of", "예시", "사례"],
    "requires": ["require", "requires", "필요", "요구"],
    "improves": ["improve", "improves", "개선", "향상"],
    "related_to": ["related", "related_to", "연관", "관련"],
}


def normalize_relation(raw: str) -> str:
    token = (raw or "").strip().lower()
    token = token.replace("-", "_").replace(" ", "_")
    for key, keywords in RELATION_SYNONYMS.items():
        for keyword in keywords:
            norm = keyword.lower().replace("-", "_").replace(" ", "_")
            if norm in token:
                return key
    return "unknown_relation"


def _extract_concept_map_lines(body: str) -> list[str]:
    lines = body.splitlines()
    start_idx: int | None = None

    for idx, line in enumerate(lines):
        if line.strip().lower() == "## concept map edges":
            start_idx = idx + 1
            break

    if start_idx is None:
        return lines

    scoped: list[str] = []
    for line in lines[start_idx:]:
        if line.strip().startswith("## "):
            break
        scoped.append(line)
    return scoped


def parse_frontmatter(content: str) -> tuple[dict, str]:
    if not content.startswith("---\n"):
        return {}, content

    lines = content.splitlines()
    if len(lines) < 3:
        return {}, content

    end_idx = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break
    if end_idx is None:
        return {}, content

    head_text = "\n".join(lines[1:end_idx])
    body = "\n".join(lines[end_idx + 1 :])

    try:
        data = yaml.safe_load(head_text) or {}
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    return data, body


def parse_edges_from_session(body: str, session_note_path: str) -> tuple[list[AssessmentEdge], list[str], list[str]]:
    edges: list[AssessmentEdge] = []
    parse_errors: list[str] = []
    raw_evidence_texts: list[str] = []

    for line_no, line in enumerate(_extract_concept_map_lines(body), start=1):
        if "->" not in line:
            continue

        stripped = line.strip().lstrip("-* ").strip()
        parts = [part.strip() for part in stripped.split("|") if part.strip()]
        if not parts:
            continue

        triple = parts[0]
        triple_parts = [part.strip() for part in triple.split("->")]
        if len(triple_parts) != 3:
            parse_errors.append(f"line {line_no}: invalid edge format")
            continue

        source_raw, relation_raw, target_raw = triple_parts
        confidence = 3.0
        evidence_ptrs = []

        for segment in parts[1:]:
            key, sep, value = segment.partition(":")
            if not sep:
                continue
            key = key.strip().lower()
            value = value.strip()
            if key in {"evidence_ptr", "evidence", "근거"} and value:
                raw_evidence_texts.append(value)
                evidence_ptrs.append(pointer_from_text(value, fallback_path=session_note_path))
            elif key in {"confidence", "자신감"} and value:
                try:
                    confidence = max(1.0, min(5.0, float(value)))
                except ValueError:
                    pass

        edges.append(
            AssessmentEdge(
                source_canonical_id=source_raw,
                relation_raw=relation_raw,
                relation_norm=normalize_relation(relation_raw),
                target_canonical_id=target_raw,
                evidence_ptrs=evidence_ptrs,
                confidence=confidence,
                is_valid=False,
                issues=[],
            )
        )

    return edges, parse_errors, raw_evidence_texts


def _resolve_edge_concepts(
    edge: AssessmentEdge,
    resolver: EntityResolver,
) -> tuple[ConceptIdentity | None, ConceptIdentity | None]:
    src_identity = resolver.resolve(edge.source_canonical_id, entity_type="concept")
    tgt_identity = resolver.resolve(edge.target_canonical_id, entity_type="concept")

    if src_identity is None:
        edge.issues.append("normalization_failed")
        edge.source_canonical_id = "unknown"
    else:
        edge.source_canonical_id = src_identity.canonical_id

    if tgt_identity is None:
        edge.issues.append("normalization_failed")
        edge.target_canonical_id = "unknown"
    else:
        edge.target_canonical_id = tgt_identity.canonical_id

    if edge.relation_norm == "unknown_relation":
        edge.issues.append("unknown_relation")

    if not edge.evidence_ptrs:
        edge.issues.append("missing_evidence_ptr")

    edge.is_valid = src_identity is not None and tgt_identity is not None and edge.relation_norm != "unknown_relation"
    return src_identity, tgt_identity


def _build_weaknesses(edges: list[AssessmentEdge]) -> list[dict]:
    weaknesses: list[dict] = []
    for edge in edges:
        if not edge.issues and edge.confidence >= 3.0:
            continue

        reason = ",".join(edge.issues) if edge.issues else "low_confidence"
        severity = "high" if any(issue.endswith("failed") for issue in edge.issues) else "medium"
        weaknesses.append(
            {
                "source": edge.source_canonical_id,
                "target": edge.target_canonical_id,
                "reason": reason,
                "severity": severity,
                "confidence": edge.confidence,
            }
        )

    weaknesses.sort(key=lambda item: (item["severity"] != "high", item["confidence"]))
    return weaknesses[:10]


def grade_session_content(
    db: SQLiteDatabase,
    topic: str,
    session_id: str,
    content: str,
    session_note_path: str,
    allow_external: bool = False,
    run_id: str | None = None,
) -> dict:
    now = datetime.now(timezone.utc)
    run_id = str(run_id or f"learn_grade_{uuid4().hex[:12]}")

    frontmatter, body = parse_frontmatter(content)
    target_trunk_ids = frontmatter.get("target_trunk_ids") or []
    if not isinstance(target_trunk_ids, list):
        target_trunk_ids = []
    target_trunk_ids = [str(item).strip() for item in target_trunk_ids if str(item).strip()]

    if not target_trunk_ids:
        existing = db.get_learning_session(session_id)
        if existing:
            raw = existing.get("target_trunk_ids_json")
            if isinstance(raw, list):
                target_trunk_ids = [str(item).strip() for item in raw if str(item).strip()]

    edges, parse_errors, raw_evidence_texts = parse_edges_from_session(body, session_note_path=session_note_path)

    policy = evaluate_policy_for_payload(
        allow_external=allow_external,
        raw_texts=raw_evidence_texts,
        mode="external-allowed" if allow_external else "local-only",
    )
    if not policy.allowed:
        db.append_learning_event(
            event_id=f"evt_{uuid4().hex}",
            event_type="learning.policy.blocked",
            logical_step="grade",
            session_id=session_id,
            run_id=run_id,
            request_id=run_id,
            source="learning",
            payload={
                "topic": topic,
                "reason": policy.blocked_reason,
                "errors": policy.policy_errors,
            },
            policy_class=policy.classification,
        )
        return {
            "schema": GRADE_SCHEMA,
            "runId": run_id,
            "topic": topic,
            "status": "blocked",
            "policy": policy.to_dict(),
            "session": {
                "sessionId": session_id,
                "path": session_note_path,
                "targetTrunkIds": target_trunk_ids,
            },
            "scores": {
                "coverage": 0,
                "edgeAccuracy": 0,
                "explanationQuality": 0,
                "final": 0,
                "totalEdges": len(edges),
                "validEdges": 0,
                "minEdges": 0,
            },
            "gateDecision": {
                "passed": False,
                "status": "blocked",
                "reasons": policy.policy_errors,
            },
            "weaknesses": [{"reason": "policy_blocked", "severity": "high"}],
            "policyErrors": policy.policy_errors,
            "createdAt": now.isoformat(),
            "updatedAt": now.isoformat(),
        }

    resolver = EntityResolver(db)

    used_target_trunks: set[str] = set()
    valid_edges = 0
    evidence_valid_edges = 0

    for edge in edges:
        src_identity, tgt_identity = _resolve_edge_concepts(edge, resolver)

        if src_identity and src_identity.canonical_id in target_trunk_ids:
            used_target_trunks.add(src_identity.canonical_id)
        if tgt_identity and tgt_identity.canonical_id in target_trunk_ids:
            used_target_trunks.add(tgt_identity.canonical_id)

        if edge.is_valid:
            valid_edges += 1
        if edge.evidence_ptrs:
            evidence_valid_edges += 1

    concept_count = len(target_trunk_ids)
    min_edges = max(5, concept_count - 1) if concept_count > 0 else 5
    total_edges = len(edges)

    coverage = len(used_target_trunks) / max(1, concept_count)
    edge_accuracy = (valid_edges + 1) / (total_edges + 2)
    explanation_quality = evidence_valid_edges / max(1, total_edges)
    final = 0.50 * edge_accuracy + 0.30 * coverage + 0.20 * explanation_quality

    score = AssessmentScore(
        coverage=coverage,
        edge_accuracy=edge_accuracy,
        explanation_quality=explanation_quality,
        final=final,
        total_edges=total_edges,
        valid_edges=valid_edges,
        min_edges=min_edges,
    )

    reasons: list[str] = []
    passed = True
    gate_status = "passed"

    if total_edges < min_edges:
        passed = False
        gate_status = "insufficient"
        reasons.append(f"insufficient edges: {total_edges} < {min_edges}")

    if final < 0.75:
        passed = False
        gate_status = "failed" if gate_status == "passed" else gate_status
        reasons.append("final score below threshold")

    if edge_accuracy < 0.70:
        passed = False
        gate_status = "failed" if gate_status == "passed" else gate_status
        reasons.append("edge accuracy below threshold")

    if coverage < 0.60:
        passed = False
        gate_status = "failed" if gate_status == "passed" else gate_status
        reasons.append("coverage below threshold")

    if parse_errors:
        passed = False
        if gate_status == "passed":
            gate_status = "failed"
        reasons.extend(parse_errors)

    decision = ProgressGateDecision(passed=passed, status=gate_status, reasons=reasons)
    weaknesses = _build_weaknesses(edges)

    db.upsert_learning_session(
        session_id=session_id,
        topic_slug=str(frontmatter.get("topic_slug") or frontmatter.get("topic") or topic),
        target_trunk_ids=target_trunk_ids,
        status=gate_status,
    )
    db.replace_learning_session_edges(
        session_id,
        [
            {
                "source_canonical_id": edge.source_canonical_id,
                "relation_norm": edge.relation_norm,
                "target_canonical_id": edge.target_canonical_id,
                "evidence_ptrs": [ptr.to_dict() for ptr in edge.evidence_ptrs],
                "confidence": edge.confidence,
                "is_valid": edge.is_valid,
            }
            for edge in edges
        ],
    )
    db.upsert_learning_progress(
        session_id=session_id,
        topic_slug=str(frontmatter.get("topic_slug") or frontmatter.get("topic") or topic),
        score_final=final,
        score_edge_accuracy=edge_accuracy,
        score_coverage=coverage,
        score_explanation_quality=explanation_quality,
        gate_passed=passed,
        gate_status=gate_status,
        weaknesses=weaknesses,
        details={
            "parseErrors": parse_errors,
            "usedTargetTrunks": sorted(used_target_trunks),
        },
    )

    db.append_learning_event(
        event_id=f"evt_{uuid4().hex}",
        event_type="learning.grade.completed",
        logical_step="grade",
        session_id=session_id,
        run_id=run_id,
        request_id=run_id,
        source="learning",
        payload={
            "final": round(final, 6),
            "passed": passed,
            "gateStatus": gate_status,
            "totalEdges": total_edges,
        },
        policy_class=policy.classification,
    )

    return {
        "schema": GRADE_SCHEMA,
        "runId": run_id,
        "topic": topic,
        "status": "ok" if passed else gate_status,
        "policy": policy.to_dict(),
        "session": {
            "sessionId": session_id,
            "path": session_note_path,
            "targetTrunkIds": target_trunk_ids,
        },
        "targetTrunkIds": target_trunk_ids,
        "scores": score.to_dict(),
        "gateDecision": decision.to_dict(),
        "weaknesses": weaknesses,
        "edges": [edge.to_dict() for edge in edges],
        "policyErrors": policy.policy_errors,
        "createdAt": now.isoformat(),
        "updatedAt": now.isoformat(),
    }
