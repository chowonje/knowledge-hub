"""Bounded multi-step ask planner for labs surfaces."""

from __future__ import annotations

import re
from typing import Any


def _infer_intent(question: str) -> str:
    lowered = str(question or "").lower()
    if any(token in lowered for token in (" vs ", " versus ", "compare", "차이", "비교")):
        return "comparison"
    if any(token in lowered for token in ("implement", "implementation", "구현", "how to")):
        return "implementation"
    if any(token in lowered for token in ("paper", "arxiv", "논문")):
        return "paper_lookup"
    return "definition"


def _dedupe(items: list[str], max_steps: int) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        normalized = re.sub(r"\s+", " ", str(item or "").strip())
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(normalized)
        if len(ordered) >= max_steps:
            break
    return ordered


def _plan_subqueries(question: str, *, max_steps: int) -> tuple[str, list[str]]:
    intent = _infer_intent(question)
    lowered = str(question or "").strip()
    candidates = [lowered]
    if intent == "comparison":
        parts = re.split(r"\b(?:vs|versus|비교|차이)\b", lowered, flags=re.IGNORECASE)
        parts = [part.strip(" ?,.") for part in parts if part.strip(" ?,.")]
        candidates.extend(parts[:2])
        if len(parts) >= 2:
            candidates.append(f"{parts[0]} and {parts[1]} differences")
    elif intent == "implementation":
        candidates.extend([f"{lowered} design", f"{lowered} implementation details"])
    elif intent == "paper_lookup":
        candidates.extend([f"{lowered} contributions", f"{lowered} limitations"])
    else:
        candidates.extend([f"{lowered} definition", f"{lowered} examples"])
    return intent, _dedupe(candidates, max_steps=max(2, min(int(max_steps or 4), 4)))


def run_ask_graph(
    searcher,
    *,
    question: str,
    source: str | None = None,
    mode: str = "hybrid",
    alpha: float = 0.7,
    max_steps: int = 4,
    top_k: int = 5,
    min_score: float = 0.3,
    return_trace: bool = True,
) -> dict[str, Any]:
    intent, subqueries = _plan_subqueries(question, max_steps=max_steps)
    trace: list[dict[str, Any]] = []
    for subquery in subqueries:
        step = searcher.generate_answer(
            subquery,
            top_k=top_k,
            min_score=min_score,
            source_type=source,
            retrieval_mode=mode,
            alpha=alpha,
            allow_external=False,
        )
        trace.append(
            {
                "subquery": subquery,
                "answer": str(step.get("answer") or ""),
                "warnings": list(step.get("warnings") or []),
                "sources": [
                    {
                        "title": str(item.get("title") or ""),
                        "source_type": str(item.get("source_type") or ""),
                        "score": float(item.get("score") or 0.0),
                    }
                    for item in list(step.get("sources") or [])[:5]
                    if isinstance(item, dict)
                ],
            }
        )

    final_result = searcher.generate_answer(
        question,
        top_k=top_k,
        min_score=min_score,
        source_type=source,
        retrieval_mode=mode,
        alpha=alpha,
        allow_external=False,
    )
    return {
        "schema": "knowledge-hub.ask-graph.result.v1",
        "status": "ok",
        "question": question,
        "decomposition": {
            "intent": intent,
            "max_steps": max_steps,
            "subqueries": subqueries,
            "synthesis_strategy": "final-pass-original-question",
        },
        "answer": final_result.get("answer"),
        "sources": final_result.get("sources", []),
        "evidence": final_result.get("evidence", final_result.get("sources", [])),
        "answer_signals": final_result.get("answerSignals", {}),
        "answer_verification": final_result.get("answerVerification", {}),
        "answer_rewrite": final_result.get("answerRewrite", {}),
        "initial_answer_verification": final_result.get("initialAnswerVerification", {}),
        "warnings": final_result.get("warnings", []),
        "trace": trace if return_trace else [],
    }
