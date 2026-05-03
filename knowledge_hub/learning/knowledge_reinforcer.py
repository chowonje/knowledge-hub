"""
Knowledge Reinforcement Recommender

지식 공백(gap)을 식별하고, 관련 소스(논문/노트/웹)를 매칭하여
구체적인 학습 보강 액션을 제안합니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from knowledge_hub.ai.rag import RAGSearcher
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.learning.policy import evaluate_policy_for_payload


@dataclass
class SourceSuggestion:
    """보강 소스 제안"""
    source_type: str  # paper, web, vault_note
    source_id: str
    title: str
    relevance_score: float
    snippet: str
    file_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sourceType": self.source_type,
            "sourceId": self.source_id,
            "title": self.title,
            "relevanceScore": round(self.relevance_score, 4),
            "snippet": self.snippet,
            "filePath": self.file_path,
        }


@dataclass
class ReinforcementAction:
    """보강 액션"""
    action_type: str  # read_paper, search_web, fill_concept, add_relation, strengthen_evidence
    priority: float
    target_entity_id: str
    target_entity_name: str
    reason: str
    suggested_sources: list[SourceSuggestion] = field(default_factory=list)
    estimated_impact: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "actionType": self.action_type,
            "priority": round(self.priority, 4),
            "targetEntityId": self.target_entity_id,
            "targetEntityName": self.target_entity_name,
            "reason": self.reason,
            "suggestedSources": [s.to_dict() for s in self.suggested_sources],
            "estimatedImpact": self.estimated_impact,
        }


def recommend_reinforcements(
    db: SQLiteDatabase,
    searcher: RAGSearcher,
    topic: str,
    session_id: str,
    gap_result: dict,
    top_k_per_gap: int = 3,
    allow_external: bool = False,
    run_id: str | None = None,
) -> dict:
    """
    지식 공백 기반 보강 추천

    Args:
        db: SQLite 데이터베이스
        searcher: RAG 검색기
        topic: 학습 주제
        session_id: 세션 ID
        gap_result: gap_analyzer 결과
        top_k_per_gap: 각 gap당 추천할 소스 수
        allow_external: 외부 호출 허용 여부
        run_id: 실행 ID

    Returns:
        {
            "schema": "knowledge-hub.learning.reinforcement.result.v1",
            "runId": str,
            "topic": str,
            "sessionId": str,
            "status": "ok" | "blocked" | "error",
            "actions": [ReinforcementAction],
            "summary": {...},
            "policy": {...}
        }
    """
    now = datetime.now(timezone.utc)
    run_id = run_id or f"reinforce_{uuid4().hex[:12]}"

    # 정책 평가
    policy = evaluate_policy_for_payload(
        allow_external=allow_external,
        raw_texts=[topic],
        mode="external-allowed" if allow_external else "local-only",
    )

    if not policy.allowed:
        return {
            "schema": "knowledge-hub.learning.reinforcement.result.v1",
            "runId": run_id,
            "topic": topic,
            "sessionId": session_id,
            "status": "blocked",
            "actions": [],
            "summary": {},
            "policy": policy.to_dict(),
            "createdAt": now.isoformat(),
        }

    # Gap 추출
    missing_trunks = gap_result.get("missingTrunks", [])
    weak_edges = gap_result.get("weakEdges", [])
    evidence_gaps = gap_result.get("evidenceGaps", [])

    actions: list[ReinforcementAction] = []

    # 1) Missing trunks -> 개념 보강 액션
    for trunk in missing_trunks[:5]:
        canonical_id = trunk.get("canonical_id", "")
        display_name = trunk.get("display_name", canonical_id)
        priority = trunk.get("priority", 0.5)

        # RAG 검색으로 관련 소스 찾기
        search_query = f"{topic} {display_name}"
        sources = _search_sources(searcher, search_query, top_k_per_gap)

        # 예상 영향 계산
        impact = f"coverage +{int(priority * 15)}%, trunk concept 보완"

        action = ReinforcementAction(
            action_type="fill_concept",
            priority=priority,
            target_entity_id=canonical_id,
            target_entity_name=display_name,
            reason=f"Missing trunk concept: {display_name}",
            suggested_sources=sources,
            estimated_impact=impact,
        )
        actions.append(action)

    # 2) Weak edges -> 관계 강화 액션
    for edge in weak_edges[:5]:
        source_id = edge.get("sourceCanonicalId", "")
        target_id = edge.get("targetCanonicalId", "")
        relation = edge.get("relationNorm", "related_to")
        confidence = edge.get("confidence", 0.0)
        issues = edge.get("issues", [])

        priority = edge.get("priority", 0.5)

        # 관계에 대한 검색 쿼리
        search_query = f"{source_id} {relation} {target_id}"
        sources = _search_sources(searcher, search_query, top_k_per_gap)

        impact = f"edge accuracy +{int(priority * 10)}%"

        action = ReinforcementAction(
            action_type="strengthen_evidence" if "missing_evidence_ptr" in issues else "add_relation",
            priority=priority,
            target_entity_id=f"{source_id}_{target_id}",
            target_entity_name=f"{source_id} -> {target_id}",
            reason=f"Weak edge: {', '.join(issues)}",
            suggested_sources=sources,
            estimated_impact=impact,
        )
        actions.append(action)

    # 3) Evidence gaps -> 근거 보강 액션
    for gap in evidence_gaps[:3]:
        source_id = gap.get("sourceCanonicalId", "")
        target_id = gap.get("targetCanonicalId", "")
        priority = gap.get("priority", 0.4)

        search_query = f"{source_id} {target_id} evidence"
        sources = _search_sources(searcher, search_query, top_k_per_gap)

        impact = f"explanation quality +{int(priority * 20)}%"

        action = ReinforcementAction(
            action_type="strengthen_evidence",
            priority=priority,
            target_entity_id=f"{source_id}_{target_id}",
            target_entity_name=f"{source_id} -> {target_id}",
            reason="Missing evidence pointer",
            suggested_sources=sources,
            estimated_impact=impact,
        )
        actions.append(action)

    # 우선순위 정렬
    actions.sort(key=lambda a: a.priority, reverse=True)

    # 요약 통계
    summary = {
        "totalActions": len(actions),
        "actionsByType": _count_by_type(actions),
        "avgPriority": sum(a.priority for a in actions) / len(actions) if actions else 0.0,
        "totalSources": sum(len(a.suggested_sources) for a in actions),
    }

    return {
        "schema": "knowledge-hub.learning.reinforcement.result.v1",
        "runId": run_id,
        "topic": topic,
        "sessionId": session_id,
        "status": "ok",
        "actions": [a.to_dict() for a in actions],
        "summary": summary,
        "policy": policy.to_dict(),
        "createdAt": now.isoformat(),
    }


def _search_sources(
    searcher: RAGSearcher,
    query: str,
    top_k: int
) -> list[SourceSuggestion]:
    """RAG 검색으로 관련 소스 찾기"""
    try:
        results = searcher.search(
            query=query,
            top_k=top_k,
            retrieval_mode="hybrid",
            alpha=0.7,
        )

        sources: list[SourceSuggestion] = []
        for result in results:
            metadata = result.metadata or {}
            source_type_raw = metadata.get("source_type", "vault")

            # source_type 매핑
            if source_type_raw == "paper":
                source_type = "paper"
                source_id = metadata.get("arxiv_id", metadata.get("title", "unknown"))
            elif source_type_raw == "web":
                source_type = "web"
                source_id = metadata.get("url", metadata.get("title", "unknown"))
            else:
                source_type = "vault_note"
                source_id = metadata.get("id", metadata.get("file_path", "unknown"))

            title = metadata.get("title", "Untitled")
            snippet = result.document[:200] + "..." if len(result.document) > 200 else result.document
            file_path = metadata.get("file_path", "")

            source = SourceSuggestion(
                source_type=source_type,
                source_id=source_id,
                title=title,
                relevance_score=result.score,
                snippet=snippet,
                file_path=file_path,
            )
            sources.append(source)

        return sources

    except Exception:
        return []


def _count_by_type(actions: list[ReinforcementAction]) -> dict[str, int]:
    """액션 타입별 개수 집계"""
    counts: dict[str, int] = {}
    for action in actions:
        action_type = action.action_type
        counts[action_type] = counts.get(action_type, 0) + 1
    return counts
