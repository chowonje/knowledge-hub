from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any

from knowledge_hub.learning.policy import evaluate_policy_for_payload
from knowledge_hub.providers.registry import get_provider_info


LOCAL_PROVIDER_NAMES = {"ollama", "pplx-local", "pplx_st", "pplx-st"}
EXTERNAL_PROVIDER_NAMES = {"openai", "anthropic", "google", "openai-compat"}

JUDGE_BACKEND = "rule_llm_v1"
DEFAULT_PASS_THRESHOLD = 0.62
DEFAULT_STRONG_KEEP_THRESHOLD = 0.78
DEFAULT_CANDIDATE_MULTIPLIER = 3


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[A-Za-z0-9가-힣_+#.-]+", str(text or "").lower()) if token}


def _safe_ratio(matched: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(matched) / float(total)


def _provider_is_local(config) -> bool:
    provider = str(getattr(config, "summarization_provider", "") or "").strip().lower()
    if provider in LOCAL_PROVIDER_NAMES:
        return True
    if provider in EXTERNAL_PROVIDER_NAMES:
        return False
    info = get_provider_info(provider)
    return bool(info and info.is_local)


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        loaded = json.loads(text)
        return loaded if isinstance(loaded, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        loaded = json.loads(match.group(0))
        return loaded if isinstance(loaded, dict) else None
    except Exception:
        return None


class PaperJudgeService:
    def __init__(
        self,
        config,
        *,
        llm: Any | None = None,
        allow_external: bool = False,
        pass_threshold: float = DEFAULT_PASS_THRESHOLD,
        strong_keep_threshold: float = DEFAULT_STRONG_KEEP_THRESHOLD,
    ):
        self.config = config
        self.llm = llm
        self.allow_external = bool(allow_external)
        self.pass_threshold = _clamp01(pass_threshold or DEFAULT_PASS_THRESHOLD)
        self.strong_keep_threshold = _clamp01(strong_keep_threshold or DEFAULT_STRONG_KEEP_THRESHOLD)

    def score_with_rules(self, paper, *, topic: str, user_goal: str = "") -> dict[str, Any]:
        topic_tokens = _tokenize(f"{topic} {user_goal}")
        title_tokens = _tokenize(getattr(paper, "title", ""))
        abstract_tokens = _tokenize(getattr(paper, "abstract", ""))
        field_tokens = _tokenize(" ".join(getattr(paper, "fields_of_study", []) or []))
        combined_tokens = title_tokens | abstract_tokens | field_tokens

        title_overlap = _safe_ratio(len(topic_tokens & title_tokens), len(topic_tokens))
        abstract_overlap = _safe_ratio(len(topic_tokens & abstract_tokens), len(topic_tokens))
        field_overlap = _safe_ratio(len(topic_tokens & field_tokens), len(topic_tokens))
        relevance_score = _clamp01((0.55 * title_overlap) + (0.30 * abstract_overlap) + (0.15 * field_overlap))

        current_year = datetime.now(timezone.utc).year
        paper_year = int(getattr(paper, "year", 0) or 0)
        year_delta = max(0, current_year - paper_year) if paper_year else 10
        novelty_score = _clamp01(1.0 - min(1.0, float(year_delta) / 8.0))

        citations = max(0, int(getattr(paper, "citation_count", 0) or 0))
        citation_signal_score = _clamp01(math.log1p(citations) / math.log1p(500.0))

        abstract_text = str(getattr(paper, "abstract", "") or "").strip()
        abstract_length_score = _clamp01(min(len(abstract_text), 1200) / 1200.0)
        read_value_score = _clamp01((0.60 * relevance_score) + (0.25 * abstract_length_score) + (0.15 * citation_signal_score))

        reasons: list[str] = []
        if title_overlap > 0:
            reasons.append("topic tokens overlap with the paper title")
        if field_overlap > 0:
            reasons.append("field-of-study tags overlap with the requested topic")
        if paper_year and year_delta <= 2:
            reasons.append("paper is recent enough to be novel")
        if citations >= 20:
            reasons.append("citation signal suggests the paper is already visible")
        if abstract_length_score >= 0.45:
            reasons.append("abstract is rich enough to estimate practical reading value")

        return {
            "dimension_scores": {
                "relevance_score": round(relevance_score, 6),
                "novelty_score": round(novelty_score, 6),
                "read_value_score": round(read_value_score, 6),
                "citation_signal_score": round(citation_signal_score, 6),
            },
            "top_reasons": reasons[:4],
        }

    def score_with_llm(self, paper, *, topic: str, user_goal: str = "") -> dict[str, Any]:
        provider = str(getattr(self.config, "summarization_provider", "") or "").strip().lower()
        is_local = _provider_is_local(self.config)
        policy = evaluate_policy_for_payload(
            allow_external=bool(self.allow_external),
            raw_texts=[
                str(topic or ""),
                str(user_goal or ""),
                str(getattr(paper, "title", "") or ""),
                str(getattr(paper, "abstract", "") or ""),
                " ".join(getattr(paper, "fields_of_study", []) or []),
            ],
            mode="paper-judge",
        )
        warnings = list(policy.warnings or [])
        if not is_local and not self.allow_external:
            warnings.append("llm judge skipped: external not allowed")
            return {
                "used": False,
                "degraded": True,
                "warnings": warnings,
                "policy": policy,
                "dimension_scores": {},
                "top_reasons": [],
            }
        if self.llm is None:
            warnings.append("llm judge skipped: llm unavailable")
            return {
                "used": False,
                "degraded": True,
                "warnings": warnings,
                "policy": policy,
                "dimension_scores": {},
                "top_reasons": [],
            }

        prompt = (
            "You are a paper judge for a personal research workflow.\n"
            "Rate whether this paper is worth reading now for the stated topic.\n"
            "Return JSON only with keys: read_value_score, top_reasons.\n"
            "read_value_score must be a float between 0 and 1.\n"
            "top_reasons must be a short array of 2 to 4 strings.\n\n"
            f"Topic: {topic}\n"
            f"User goal: {user_goal}\n"
            f"Title: {getattr(paper, 'title', '')}\n"
            f"Fields: {', '.join(getattr(paper, 'fields_of_study', []) or [])}\n"
            f"Year: {getattr(paper, 'year', 0)}\n"
            f"Citations: {getattr(paper, 'citation_count', 0)}\n"
            f"Abstract: {getattr(paper, 'abstract', '')}\n"
        )

        try:
            raw = self.llm.generate(prompt, context="")
        except Exception as error:
            warnings.append(f"llm judge failed: {error}")
            return {
                "used": False,
                "degraded": True,
                "warnings": warnings,
                "policy": policy,
                "dimension_scores": {},
                "top_reasons": [],
            }

        payload = _extract_json_object(raw) or {}
        read_value_score = _clamp01(payload.get("read_value_score"))
        reasons = [str(item).strip() for item in list(payload.get("top_reasons") or []) if str(item).strip()]
        if not reasons and read_value_score > 0:
            reasons = ["llm judged the paper worth reading for this topic"]

        return {
            "used": True,
            "degraded": False,
            "warnings": warnings,
            "policy": policy,
            "dimension_scores": {
                "read_value_score": round(read_value_score, 6),
            },
            "top_reasons": reasons[:4],
            "raw": raw,
            "provider": provider,
        }

    def merge_scores(
        self,
        paper,
        *,
        topic: str,
        user_goal: str = "",
    ) -> dict[str, Any]:
        rule = self.score_with_rules(paper, topic=topic, user_goal=user_goal)
        llm = self.score_with_llm(paper, topic=topic, user_goal=user_goal)

        rule_scores = dict(rule.get("dimension_scores") or {})
        llm_scores = dict(llm.get("dimension_scores") or {})
        merged_dimension_scores = {
            "relevance_score": _clamp01(rule_scores.get("relevance_score")),
            "novelty_score": _clamp01(rule_scores.get("novelty_score")),
            "citation_signal_score": _clamp01(rule_scores.get("citation_signal_score")),
            "read_value_score": _clamp01(
                (0.35 * _clamp01(rule_scores.get("read_value_score")))
                + (0.65 * _clamp01(llm_scores.get("read_value_score")))
            )
            if llm.get("used")
            else _clamp01(rule_scores.get("read_value_score")),
        }
        total_score = _clamp01(
            (0.45 * merged_dimension_scores["relevance_score"])
            + (0.20 * merged_dimension_scores["novelty_score"])
            + (0.25 * merged_dimension_scores["read_value_score"])
            + (0.10 * merged_dimension_scores["citation_signal_score"])
        )
        decision = "keep" if total_score >= self.pass_threshold else "skip"

        reasons: list[str] = []
        for item in list(rule.get("top_reasons") or []) + list(llm.get("top_reasons") or []):
            text = str(item).strip()
            if text and text not in reasons:
                reasons.append(text)

        policy = llm.get("policy")
        provider = str(getattr(self.config, "summarization_provider", "") or "").strip().lower()
        return {
            "paper_id": str(getattr(paper, "arxiv_id", "") or ""),
            "arxiv_id": str(getattr(paper, "arxiv_id", "") or ""),
            "title": str(getattr(paper, "title", "") or ""),
            "decision": decision,
            "total_score": round(total_score, 6),
            "dimension_scores": {key: round(_clamp01(value), 6) for key, value in merged_dimension_scores.items()},
            "top_reasons": reasons[:4],
            "backend": JUDGE_BACKEND,
            "degraded": bool(llm.get("degraded")),
            "policy": {
                "allowed": bool(getattr(policy, "allowed", True)),
                "classification": str(getattr(policy, "classification", "P2") or "P2"),
                "rule": str(getattr(policy, "rule", "local_only_no_external") or "local_only_no_external"),
                "trace_id": str(getattr(policy, "trace_id", "") or ""),
                "blocked_reason": getattr(policy, "blocked_reason", None),
                "warnings": list(getattr(policy, "warnings", []) or []),
            },
            "provider": provider,
            "strong_keep": total_score >= self.strong_keep_threshold,
            "llm_used": bool(llm.get("used")),
            "warnings": list(llm.get("warnings") or []),
        }

    def evaluate_candidates(
        self,
        papers: list[Any],
        *,
        topic: str,
        threshold: float | None = None,
        user_goal: str = "",
    ) -> dict[str, Any]:
        resolved_threshold = _clamp01(threshold if threshold is not None else self.pass_threshold)
        items = [self.merge_scores(paper, topic=topic, user_goal=user_goal) for paper in papers]
        items.sort(key=lambda item: (float(item.get("total_score", 0.0) or 0.0), str(item.get("title") or "")), reverse=True)
        selected = [item for item in items if float(item.get("total_score", 0.0) or 0.0) >= resolved_threshold]
        warnings: list[str] = []
        degraded = False
        for item in items:
            degraded = degraded or bool(item.get("degraded"))
            for warning in list(item.get("warnings") or []):
                if warning not in warnings:
                    warnings.append(warning)
        return {
            "schema": "knowledge-hub.paper.judge.result.v1",
            "status": "ok",
            "topic": topic,
            "backend": JUDGE_BACKEND,
            "threshold": round(resolved_threshold, 6),
            "candidateCount": len(items),
            "selectedCount": len(selected),
            "degraded": degraded,
            "warnings": warnings,
            "items": items,
        }

    def select_candidates(
        self,
        papers: list[Any],
        *,
        topic: str,
        threshold: float | None = None,
        top_k: int | None = None,
        user_goal: str = "",
    ) -> tuple[list[Any], dict[str, Any]]:
        payload = self.evaluate_candidates(papers, topic=topic, threshold=threshold, user_goal=user_goal)
        item_by_id = {str(item.get("paper_id") or ""): item for item in payload.get("items") or []}
        resolved_top_k = max(1, int(top_k or len(papers) or 1))
        selected_items = [item for item in payload.get("items") or [] if str(item.get("decision")) == "keep"][:resolved_top_k]
        selected_ids = {str(item.get("paper_id") or "") for item in selected_items}
        selected = [paper for paper in papers if str(getattr(paper, "arxiv_id", "") or "") in selected_ids]
        payload["selectedCount"] = len(selected)
        return selected, payload
