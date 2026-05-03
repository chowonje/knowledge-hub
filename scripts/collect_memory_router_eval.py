#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from knowledge_hub.application.context import AppContextFactory


def _read_queries(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".csv":
        rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
        items: list[dict[str, str]] = []
        for row in rows:
            query = str(row.get("query") or "").strip()
            if not query:
                continue
            payload = {str(key): str(value or "").strip() for key, value in row.items()}
            payload["query"] = query
            payload["source"] = str(row.get("source") or "").strip()
            items.append(payload)
        return items
    rows = path.read_text(encoding="utf-8").splitlines()
    return [
        {
            "query": line.strip(),
            "source": "",
        }
        for line in rows
        if line.strip() and not line.strip().startswith("#")
    ]


def _is_temporal_query(query: str) -> bool:
    body = str(query or "").lower()
    markers = (
        "latest",
        "recent",
        "updated",
        "update",
        "changed since",
        "before",
        "after",
        "최근",
        "최신",
        "업데이트",
        "변경",
        "이후",
        "이전",
    )
    return any(token in body for token in markers)


def _csv_bool(value: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"1", "true", "yes", "y"}:
        return "1"
    if raw in {"0", "false", "no", "n"}:
        return "0"
    return ""


class _EvalOnlyLLM:
    def generate(self, prompt: str, context: str = "") -> str:  # noqa: ARG002
        return "EVAL_ONLY"

    def stream_generate(self, prompt: str, context: str = ""):  # noqa: ARG002
        yield "EVAL_ONLY"


def _patch_searcher_for_eval(searcher, *, profile: str) -> None:
    eval_llm = _EvalOnlyLLM()
    searcher._eval_answer_profile = str(profile or "candidate-v6").strip().lower()
    searcher._resolve_llm_for_request = lambda **_kwargs: (  # type: ignore[attr-defined]
        eval_llm,
        {"route": "fixed", "provider": "eval", "model": "stub"},
        [],
    )
    searcher._verify_answer = lambda **_kwargs: {  # type: ignore[attr-defined]
        "status": "skipped",
        "summary": "evaluation collection skips verification",
        "claims": [],
        "supportedClaimCount": 0,
        "unsupportedClaimCount": 0,
        "uncertainClaimCount": 0,
        "conflictMentioned": True,
        "needsCaution": False,
        "warnings": [],
        "route": {"route": "fallback-only", "provider": "", "model": ""},
    }
    searcher._rewrite_answer = lambda **kwargs: (  # type: ignore[attr-defined]
        kwargs["answer"],
        {
            "attempted": False,
            "applied": False,
            "finalAnswerSource": "original",
            "warnings": [],
        },
    )
    searcher._apply_conservative_fallback_if_needed = lambda **kwargs: (  # type: ignore[attr-defined]
        kwargs["answer"],
        kwargs["rewrite_meta"],
        kwargs["verification"],
    )


def _first_source(result: dict[str, Any]) -> dict[str, Any]:
    sources = list(result.get("sources") or [])
    if not sources:
        return {}
    first = sources[0]
    return first if isinstance(first, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect top1 ask-path results for memory-router manual evaluation.")
    parser.add_argument("--config", default=None, help="Optional config path")
    parser.add_argument(
        "--queries",
        default="docs/experiments/memory_router_eval_queries_v1.txt",
        help="Text file with one query per line",
    )
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--top-k", type=int, default=8, help="Ask retrieval top-k")
    parser.add_argument("--source", default=None, help="Optional source filter: paper, vault, web, concept")
    parser.add_argument(
        "--mode",
        default="hybrid",
        choices=["semantic", "keyword", "hybrid"],
        help="Retrieval mode",
    )
    parser.add_argument("--alpha", type=float, default=0.7, help="Hybrid alpha")
    parser.add_argument(
        "--memory-route-mode",
        default="off",
        choices=["off", "prefilter"],
        help="Ask memory route mode",
    )
    parser.add_argument(
        "--profile",
        default="candidate-v6",
        choices=["off-control", "on-control", "candidate-v3", "candidate-v4", "candidate-v5", "candidate-v6"],
        help="Eval-only profile: off-control disables memory routing, on-control keeps routing but uses stricter answerability gate.",
    )
    args = parser.parse_args()

    queries_path = Path(args.queries).expanduser()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    factory = AppContextFactory(config_path=args.config)
    searcher = factory.get_searcher()
    profile = str(args.profile or "candidate-v6").strip().lower()
    _patch_searcher_for_eval(searcher, profile=profile)
    effective_memory_route_mode = str(args.memory_route_mode)
    if profile == "off-control":
        effective_memory_route_mode = "off"
    elif profile in {"on-control", "candidate-v3", "candidate-v4", "candidate-v5", "candidate-v6"}:
        effective_memory_route_mode = "prefilter"

    rows: list[dict[str, Any]] = []
    for item in _read_queries(queries_path):
        query = str(item.get("query") or "").strip()
        source_type = str(item.get("source") or args.source or "").strip() or None
        result = searcher.generate_answer(
            query,
            top_k=max(1, int(args.top_k)),
            source_type=source_type,
            retrieval_mode=str(args.mode),
            alpha=float(args.alpha),
            allow_external=False,
            memory_route_mode=effective_memory_route_mode,
            paper_memory_mode=effective_memory_route_mode if str(source_type or "").strip().lower() == "paper" else "off",
        )
        top1 = _first_source(result)
        memory_route = dict(result.get("memoryRoute") or {})
        memory_prefilter = dict(result.get("memoryPrefilter") or {})
        evidence_packet = dict(result.get("evidencePacket") or {})
        validation = dict(evidence_packet.get("validation") or {})
        context_budget = dict(result.get("contextBudget") or {})
        temporal_query = _csv_bool(str(item.get("temporal_query") or ""))
        if not temporal_query:
            temporal_query = "1" if _is_temporal_query(query) else "0"
        rows.append(
            {
                "query": query,
                "source": str(source_type or ""),
                "query_type": str(item.get("query_type") or ""),
                "expected_primary_source": str(item.get("expected_primary_source") or ""),
                "expected_answer_style": str(item.get("expected_answer_style") or ""),
                "difficulty": str(item.get("difficulty") or ""),
                "rank": 1,
                "label": "",
                "no_result": "1" if str(result.get("status") or "") == "no_result" or not top1 else "0",
                "temporal_query": temporal_query,
                "wrong_era": "",
                "top1_title": str(top1.get("title") or ""),
                "top1_source_type": str(top1.get("source_type") or ""),
                "top1_quality_flag": str(top1.get("quality_flag") or ""),
                "top1_citation": str(top1.get("citation_label") or ""),
                "top1_excerpt": str(top1.get("excerpt") or ""),
                "answer_status": str(result.get("status") or ""),
                "eval_profile": profile,
                "memory_route_requested": str(memory_route.get("requestedMode") or ""),
                "memory_route_applied": str(bool(memory_route.get("applied"))).lower(),
                "memory_prefilter_reason": str(memory_prefilter.get("reason") or ""),
                "gating_decision": str(memory_prefilter.get("gatingDecision") or ""),
                "memory_confidence": str(memory_prefilter.get("memoryConfidence") or ""),
                "chunk_expansion_triggered": str(bool(memory_prefilter.get("chunkExpansionTriggered"))).lower(),
                "chunk_expansion_reason": str(memory_prefilter.get("chunkExpansionReason") or ""),
                "verifier_budget_used": str(memory_prefilter.get("verifierBudgetUsed") or ""),
                "stale_memory_signals": " | ".join(str(item) for item in list(memory_prefilter.get("staleMemorySignals") or [])),
                "temporal_route_applied": str(bool(memory_prefilter.get("temporalRouteApplied"))).lower(),
                "source_scope_enforced": str(bool(result.get("sourceScopeEnforced"))).lower(),
                "mixed_fallback_used": str(bool(result.get("mixedFallbackUsed"))).lower(),
                "matched_memory_ids": " | ".join(str(item) for item in list(memory_prefilter.get("matchedMemoryIds") or [])),
                "matched_document_ids": " | ".join(str(item) for item in list(memory_prefilter.get("matchedDocumentIds") or [])),
                "memory_relations_used": " | ".join(
                    str(item.get("relation_id") or item) for item in list(result.get("memoryRelationsUsed") or [])
                ),
                "temporal_signals": str(result.get("temporalSignals") or {}),
                "answerable": str(bool(evidence_packet.get("answerable"))).lower(),
                "answerable_decision_reason": str(evidence_packet.get("answerableDecisionReason") or ""),
                "top1_substantive": str(bool(validation.get("top1Substantive"))).lower(),
                "top1_reselected": str(bool(validation.get("top1Reselected"))).lower(),
                "top1_rejected_reason": str(validation.get("top1RejectedReason") or ""),
                "substantive_evidence_count": str(validation.get("substantiveEvidenceCount") or ""),
                "direct_answer_evidence_count": str(validation.get("directAnswerEvidenceCount") or ""),
                "non_substantive_evidence_count": str(validation.get("nonSubstantiveEvidenceCount") or ""),
                "temporal_grounded_count": str(validation.get("temporalGroundedCount") or ""),
                "semantic_family_count": str(result.get("semanticFamilyCount") or ""),
                "collapsed_related_evidence_count": str(len(list(result.get("collapsedRelatedEvidence") or []))),
                "context_token_budget": str((result.get("retrievalPlan") or {}).get("tokenBudget") or ""),
                "final_packed_token_estimate": str(context_budget.get("finalPackedTokenEstimate") or ""),
                "memory_context_token_estimate": str(context_budget.get("memoryContextTokenEstimate") or ""),
                "raw_chunk_token_estimate": str(context_budget.get("rawChunkTokenEstimate") or ""),
                "dedup_saved_tokens": str(context_budget.get("dedupSavedTokens") or ""),
                "gating_saved_tokens": str(context_budget.get("gatingSavedTokens") or ""),
                "verifier_added_tokens": str(context_budget.get("verifierAddedTokens") or ""),
                "insufficient_reasons": " | ".join(str(item) for item in list(evidence_packet.get("insufficientEvidenceReasons") or [])),
                "pred_label": "",
                "pred_wrong_era": "",
                "pred_should_abstain": "",
                "pred_confidence": "",
                "pred_reason": "",
                "notes": "",
            }
        )

    fieldnames = [
        "query",
        "source",
        "query_type",
        "expected_primary_source",
        "expected_answer_style",
        "difficulty",
        "rank",
        "label",
        "no_result",
        "temporal_query",
        "wrong_era",
        "top1_title",
        "top1_source_type",
        "top1_quality_flag",
        "top1_citation",
        "top1_excerpt",
        "answer_status",
        "eval_profile",
        "memory_route_requested",
        "memory_route_applied",
        "memory_prefilter_reason",
        "gating_decision",
        "memory_confidence",
        "chunk_expansion_triggered",
        "chunk_expansion_reason",
        "verifier_budget_used",
        "stale_memory_signals",
        "temporal_route_applied",
        "source_scope_enforced",
        "mixed_fallback_used",
        "matched_memory_ids",
        "matched_document_ids",
        "memory_relations_used",
        "temporal_signals",
        "answerable",
        "answerable_decision_reason",
        "top1_substantive",
        "top1_reselected",
        "top1_rejected_reason",
        "substantive_evidence_count",
        "direct_answer_evidence_count",
        "non_substantive_evidence_count",
        "temporal_grounded_count",
        "semantic_family_count",
        "collapsed_related_evidence_count",
        "context_token_budget",
        "final_packed_token_estimate",
        "memory_context_token_estimate",
        "raw_chunk_token_estimate",
        "dedup_saved_tokens",
        "gating_saved_tokens",
        "verifier_added_tokens",
        "insufficient_reasons",
        "pred_label",
        "pred_wrong_era",
        "pred_should_abstain",
        "pred_confidence",
        "pred_reason",
        "notes",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(
        f"Wrote memory-router eval sheet: {out_path} "
        f"({len(rows)} queries, mode={effective_memory_route_mode}, retrieval={args.mode}, profile={profile})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
