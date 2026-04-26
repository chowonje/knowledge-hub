"""Inspectable memory-form routing for labs surfaces.

This module keeps retrieval path selection separate from memory-form selection.
It does not execute retrieval. It only exposes which stored memory forms should
be preferred first for a given query and which retrieval paths are suitable for
each form.
"""

from __future__ import annotations

from typing import Any

from knowledge_hub.ai.retrieval_fit import classify_query_intent, normalize_source_type


MEMORY_FORMS: tuple[str, ...] = (
    "paper_memory",
    "document_memory",
    "claim_evidence",
    "ontology_cluster",
    "chunk",
)


ALLOWED_PATHS_BY_FORM: dict[str, list[str]] = {
    "paper_memory": ["paper_memory_prefilter", "vector", "lexical"],
    "document_memory": ["document_memory_summary", "vector", "lexical"],
    "claim_evidence": ["claim_compare", "vector", "lexical", "ontology_alias"],
    "ontology_cluster": ["ontology_alias", "graph_signal", "cluster_scope"],
    "chunk": ["vector", "lexical"],
}


def _base_entry(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "priority": 99,
        "why": [],
        "allowedRetrievalPaths": list(ALLOWED_PATHS_BY_FORM.get(name, [])),
        "qualityRisks": [],
        "handoffForms": [],
    }


def _finalize_entries(entries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    items = list(entries.values())
    items.sort(key=lambda item: (int(item.get("priority") or 99), str(item.get("name") or "")))
    for index, item in enumerate(items, start=1):
        item["priority"] = index
        item["why"] = [str(reason).strip() for reason in item.get("why") or [] if str(reason).strip()]
        item["qualityRisks"] = [str(reason).strip() for reason in item.get("qualityRisks") or [] if str(reason).strip()]
        item["handoffForms"] = [str(reason).strip() for reason in item.get("handoffForms") or [] if str(reason).strip()]
    return items


def build_memory_route(*, query: str, source_type: str | None = None) -> dict[str, Any]:
    text = str(query or "").strip()
    normalized_source = normalize_source_type(source_type)
    intent = classify_query_intent(text)

    entries = {name: _base_entry(name) for name in MEMORY_FORMS}

    def boost(name: str, priority: int, *, why: str, risks: list[str] | None = None, handoff: list[str] | None = None) -> None:
        entry = entries[name]
        entry["priority"] = min(int(entry["priority"]), int(priority))
        entry["why"].append(why)
        if risks:
            entry["qualityRisks"].extend(risks)
        if handoff:
            entry["handoffForms"].extend(handoff)

    if intent == "comparison":
        boost(
            "claim_evidence",
            1,
            why="비교 질의는 normalized claim과 evidence를 먼저 봐야 조건 불일치와 충돌을 분리하기 쉽습니다.",
            risks=["claim normalization이 partial이면 coverage가 줄 수 있습니다."],
            handoff=["paper_memory", "document_memory", "chunk"],
        )
        boost(
            "paper_memory",
            2,
            why="비교 대상 논문을 빠르게 대표 카드로 좁히면 논문 간 축 정렬이 쉬워집니다.",
            handoff=["document_memory"],
        )
        boost(
            "document_memory",
            3,
            why="섹션 단위 근거를 봐야 결과와 limitation을 과잉 일반화하지 않습니다.",
            handoff=["chunk"],
        )
    elif intent == "paper_lookup":
        boost(
            "paper_memory",
            1,
            why="논문 조회 질의는 논문 대표 카드로 시작하는 것이 가장 빠릅니다.",
            handoff=["document_memory", "chunk"],
        )
        boost(
            "document_memory",
            2,
            why="대표 카드 뒤에는 섹션 요약으로 problem/method/results를 확인하는 것이 좋습니다.",
            handoff=["chunk"],
        )
    elif intent == "paper_topic":
        boost(
            "paper_memory",
            1,
            why="주제형 논문 질의는 먼저 여러 논문의 대표 카드를 폭넓게 모아 후보군을 만드는 것이 맞습니다.",
            handoff=["document_memory", "chunk", "ontology_cluster"],
        )
        boost(
            "document_memory",
            2,
            why="대표 카드만으로 부족한 경우 섹션 요약과 핵심 결과 단위를 함께 봐야 논문 포함/제외 판단이 쉬워집니다.",
            handoff=["chunk"],
        )
        boost(
            "ontology_cluster",
            3,
            why="근접 개념과 별칭은 topic expansion에 보조 신호로 유용합니다.",
            risks=["개념 연결이 넓어져 주변 논문이 섞일 수 있습니다."],
        )
    elif intent == "evaluation":
        boost(
            "claim_evidence",
            1,
            why="평가 질의는 metric, dataset, comparator를 포함한 결과 단위를 먼저 봐야 합니다.",
            risks=["결과 문장이 표/캡션에 흩어져 있으면 누락될 수 있습니다."],
            handoff=["document_memory", "chunk"],
        )
        boost(
            "document_memory",
            2,
            why="evaluation/result 섹션이 함께 있어야 수치와 조건을 안전하게 읽을 수 있습니다.",
            handoff=["chunk"],
        )
        boost(
            "paper_memory",
            3,
            why="논문 카드로 전체 성격을 먼저 잡으면 결과 해석이 쉬워집니다.",
        )
    elif intent in {"implementation", "howto"}:
        boost(
            "document_memory",
            1,
            why="구현/사용법 질의는 method/approach/system 섹션을 먼저 보는 것이 좋습니다.",
            handoff=["chunk", "paper_memory"],
        )
        boost(
            "chunk",
            2,
            why="세부 절차나 code-like 설명은 원문 청크까지 내려가야 놓치지 않습니다.",
        )
        boost(
            "ontology_cluster",
            3,
            why="관련 개념이나 주변 노트를 넓게 탐색할 때만 보조적으로 유용합니다.",
            risks=["개념 연결이 강해도 실제 절차 설명은 부족할 수 있습니다."],
        )
    elif intent in {"definition", "topic_lookup"}:
        boost(
            "document_memory",
            1,
            why="정의/설명 질의는 document summary와 background를 먼저 보는 것이 읽기 쉽습니다.",
            handoff=["ontology_cluster", "chunk"],
        )
        boost(
            "ontology_cluster",
            2,
            why="관련 개념과 주변 맥락은 ontology/cluster가 더 잘 연결합니다.",
            risks=["관계 연결은 강하지만 그대로 읽기 좋은 설명은 아닐 수 있습니다."],
            handoff=["paper_memory"],
        )
        boost(
            "paper_memory",
            3,
            why="주제가 논문 중심이면 대표 카드가 빠른 압축 뷰를 제공합니다.",
        )

    # source-sensitive bias
    if normalized_source == "paper":
        boost("paper_memory", 1, why="source_type=paper 제약이 있어 논문 카드 우선 접근이 안전합니다.")
        boost("document_memory", 2, why="paper source는 section-aware 요약이 잘 맞습니다.")
    elif normalized_source == "vault":
        boost("document_memory", 1, why="vault source는 note summary/section 기억이 가장 직접적입니다.")
        boost("ontology_cluster", 2, why="vault 간 링크/cluster를 함께 보면 주변 맥락을 좁히기 좋습니다.")
    elif normalized_source == "web":
        boost("document_memory", 1, why="web source도 요약/section 기억이 먼저고 graph는 보조 신호입니다.")

    # universal verifier
    boost(
        "chunk",
        5,
        why="최종 답변 전에는 원문 청크/근거 확인이 가장 안전한 검증 수단입니다.",
        risks=["읽기 품질은 낮지만 provenance 확인에는 가장 직접적입니다."],
    )

    preferred = _finalize_entries(entries)
    preferred_names = [str(item.get("name")) for item in preferred]
    primary = preferred_names[0] if preferred_names else "chunk"
    verifier = "chunk"
    skipped: list[dict[str, Any]] = []

    for form in MEMORY_FORMS:
        if form not in preferred_names:
            skipped.append(
                {
                    "name": form,
                    "reason": "현재 질의 의도와 source 제약 기준으로 우선순위 바깥입니다.",
                }
            )

    recommended_order = "memory_form_first"
    strategy_comparison = [
        {
            "name": "memory_form_first",
            "summary": "먼저 어떤 기억 형태를 쓸지 정하고, 그 뒤에 해당 기억에 맞는 retrieval path를 고릅니다.",
            "pros": [
                "질문 의도와 맞지 않는 검색 신호가 결과를 오염시키는 일을 줄입니다.",
                "claim/document/paper/graph memory를 역할별로 분리해 품질 일관성이 좋아집니다.",
                "최종 evidence 검증 경로를 chunk로 고정하기 쉬워집니다.",
            ],
            "cons": [
                "상위 memory routing 규칙을 유지해야 합니다.",
                "질의 분류가 틀리면 잘못된 memory form을 먼저 볼 수 있습니다.",
            ],
            "whenPreferred": [
                "comparison, evaluation, paper lookup처럼 기대하는 출력 형태가 분명한 질의",
                "claim/evidence와 section summary를 분리하고 싶은 경우",
            ],
            "qualityRisk": "질의 의도 분류가 거칠면 초반 memory choice가 빗나갈 수 있습니다.",
        },
        {
            "name": "retrieval_path_first",
            "summary": "vector/graph/cluster 같은 검색기를 먼저 돌리고, 나온 결과를 나중에 memory form처럼 해석합니다.",
            "pros": [
                "구현이 단순하고 현재 retrieval-first 구조와 잘 맞습니다.",
                "초기 recall을 넓게 확보하기 쉽습니다.",
            ],
            "cons": [
                "검색기 다중화가 곧 memory form 선택으로 오해될 수 있습니다.",
                "figure wrapper나 concept hit 같은 표면 신호가 요약/비교 품질을 오염시키기 쉽습니다.",
                "비교 질의에서 claim form보다 generic chunk가 먼저 떠서 조건 불일치를 놓치기 쉽습니다.",
            ],
            "whenPreferred": [
                "open-ended topic search",
                "아직 질의 의도를 안정적으로 분류하기 어려운 경우",
            ],
            "qualityRisk": "초기 retrieval 실수가 cluster/graph rerank로 약하게 증폭될 수 있습니다.",
        },
    ]

    warnings: list[str] = []
    if intent in {"comparison", "evaluation"}:
        warnings.append("비교/평가 질의는 claim normalization coverage가 낮으면 document/chunk fallback 비중이 커집니다.")
    if not normalized_source:
        warnings.append("source_type 제약이 없어서 memory form 선택이 더 넓고 느슨합니다.")

    return {
        "schema": "knowledge-hub.memory-route.result.v1",
        "status": "ok",
        "contractRole": "read_only_memory_form_route_explainer",
        "retrievalExecuted": False,
        "canonicalWritePerformed": False,
        "query": text,
        "sourceType": normalized_source or "all",
        "queryIntent": intent,
        "recommendedDecisionOrder": recommended_order,
        "route": {
            "primaryForm": primary,
            "verifierForm": verifier,
            "preferredForms": preferred,
            "skippedForms": skipped,
        },
        "strategyComparison": strategy_comparison,
        "warnings": warnings,
    }


__all__ = ["build_memory_route"]
