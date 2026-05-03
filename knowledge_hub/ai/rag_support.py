from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from knowledge_hub.core.keywords import STOP_ENGLISH, STOP_KOREAN
from knowledge_hub.core.models import SearchResult
from knowledge_hub.core.sanitizer import redact_p0
from knowledge_hub.ai.retrieval_fit import normalize_source_type as _shared_normalize_source_type


def normalize_source_type(source_type: str | None) -> str | None:
    normalized = _shared_normalize_source_type(source_type)
    return normalized or None


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def tokenize(text: str) -> set[str]:
    body = (text or "").lower()
    en = [t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9]+", body) if len(t) >= 3 and t not in STOP_ENGLISH]
    ko = [t for t in re.findall(r"[가-힣]{2,}", body) if t not in STOP_KOREAN]
    return set(en + ko)


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1.0, len(left | right))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        if parsed < 0:
            return 0.0
        if parsed > 1:
            return 1.0
        return parsed
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def retrieval_sort_score(result: SearchResult) -> float:
    extras = dict(getattr(result, "lexical_extras", {}) or {})
    try:
        return float(extras.get("retrieval_sort_score"))
    except Exception:
        return safe_float(getattr(result, "score", 0.0), 0.0)


def retrieval_sort_key(result: SearchResult, *, prefer_lexical: bool = False) -> tuple[float, float, float, float]:
    if prefer_lexical:
        return (
            retrieval_sort_score(result),
            safe_float(getattr(result, "score", 0.0), 0.0),
            safe_float(getattr(result, "lexical_score", 0.0), 0.0),
            safe_float(getattr(result, "semantic_score", 0.0), 0.0),
        )
    return (
        retrieval_sort_score(result),
        safe_float(getattr(result, "score", 0.0), 0.0),
        safe_float(getattr(result, "semantic_score", 0.0), 0.0),
        safe_float(getattr(result, "lexical_score", 0.0), 0.0),
    )


def quality_priority(flag: str) -> int:
    return {
        "ok": 0,
        "needs_review": 1,
        "unscored": 2,
        "reject": 3,
    }.get(str(flag or "unscored").strip().lower(), 2)


def result_id(result: SearchResult) -> str:
    if result.document_id:
        return result.document_id
    title = str(result.metadata.get("title", "")).strip()
    file_path = str(result.metadata.get("file_path", "")).strip()
    chunk = str(result.metadata.get("chunk_index", "0"))
    if file_path:
        return f"{file_path}#{chunk}#{title}".strip("#")
    if title:
        return f"{title}#{chunk}"
    return f"{result.distance:.4f}-{len(result.document)}"


def note_id_for_result(result: SearchResult) -> str:
    metadata = result.metadata or {}
    candidates = [
        str(metadata.get("note_id") or "").strip(),
        str(metadata.get("file_path") or "").strip(),
        str(metadata.get("document_id") or result.document_id or "").strip(),
        str(metadata.get("parent_id") or "").strip(),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        token = candidate
        if "::section:" in token:
            token = token.split("::section:", 1)[0]
        if "#chunk:" in token:
            token = token.split("#chunk:", 1)[0]
        if token:
            return token
    return ""


def source_label_for_result(result: SearchResult) -> str:
    return str(normalize_source_type((result.metadata or {}).get("source_type", "")) or "").strip().lower()


def json_load_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value or "{}")
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def extract_json_payload(text: str) -> dict[str, Any]:
    body = str(text or "").strip()
    if not body:
        return {}
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", body, re.IGNORECASE)
    if fenced:
        body = fenced.group(1).strip()
    try:
        parsed = json.loads(body)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    start = body.find("{")
    end = body.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(body[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def truncate_text(text: str, limit: int = 280) -> str:
    body = clean_text(text)
    if len(body) <= limit:
        return body
    return body[: max(0, limit - 3)].rstrip() + "..."


def query_hash(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def query_digest(text: str, limit: int = 80) -> str:
    sanitized = redact_p0(str(text or ""))
    sanitized = truncate_text(sanitized, limit=limit)
    return sanitized or ""


def merge_top_signal_items(existing: list[dict[str, Any]] | None, additions: dict[str, float], limit: int = 3) -> list[dict[str, Any]]:
    merged: dict[str, float] = {}
    for item in existing or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        try:
            merged[name] = round(float(item.get("value") or 0.0), 6)
        except Exception:
            continue
    for name, value in additions.items():
        token = str(name or "").strip()
        if not token:
            continue
        try:
            parsed = float(value)
        except Exception:
            continue
        if abs(parsed) <= 0.000001:
            continue
        merged[token] = round(parsed, 6)
    ranked = [{"name": name, "value": value} for name, value in merged.items()]
    ranked.sort(key=lambda item: abs(float(item["value"])), reverse=True)
    return ranked[: max(1, int(limit))]


def result_paper_id(result: SearchResult) -> str:
    metadata = dict(getattr(result, "metadata", {}) or {})
    for key in ("arxiv_id", "paper_id"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    return ""


def merge_source_filter(base_filter: dict[str, Any] | None, source_type: str) -> dict[str, Any] | None:
    merged = dict(base_filter or {})
    current = normalize_source_type(merged.get("source_type"))
    if current and current != source_type:
        return None
    merged["source_type"] = source_type
    return merged


def supplemental_source_types_for_intent(intent: str) -> list[str]:
    if intent in {"definition", "comparison", "evaluation", "paper_lookup", "paper_topic", "topic_lookup"}:
        return ["paper", "concept"]
    if intent in {"implementation", "howto"}:
        return ["paper"]
    return []


def route_summary(route: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(route or {})
    return {
        "route": str(payload.get("route") or ""),
        "provider": str(payload.get("provider") or ""),
        "model": str(payload.get("model") or ""),
    }


def merge_search_results(results: list[SearchResult], *, top_k: int) -> list[SearchResult]:
    merged: dict[str, SearchResult] = {}
    for item in results:
        key = result_id(item)
        existing = merged.get(key)
        if existing is None:
            merged[key] = item
            continue
        if retrieval_sort_key(item) > retrieval_sort_key(existing):
            merged[key] = item
    ranked = list(merged.values())
    ranked.sort(key=retrieval_sort_key, reverse=True)
    return ranked[: max(1, int(top_k))]


def record_answer_log(
    recorder: Any,
    *,
    query: str,
    payload: dict[str, Any],
    source_type: str | None,
    retrieval_mode: str,
    allow_external: bool,
) -> None:
    if not callable(recorder):
        return
    verification = dict(payload.get("answerVerification") or payload.get("answer_verification") or {})
    rewrite = dict(payload.get("answerRewrite") or payload.get("answer_rewrite") or {})
    router = dict(payload.get("router") or {})
    warnings = [str(item).strip() for item in (payload.get("warnings") or []) if str(item).strip()]
    status = str(payload.get("status") or "").strip() or "ok"
    if not payload.get("sources") and not payload.get("evidence") and status == "ok":
        status = "no_result"
    try:
        recorder(
            query_hash=query_hash(query),
            query_digest=query_digest(query),
            source_type=str(normalize_source_type(source_type) or "all"),
            retrieval_mode=str(retrieval_mode or "hybrid"),
            allow_external=bool(allow_external),
            result_status=status,
            verification_status=str(verification.get("status") or ""),
            needs_caution=bool(verification.get("needsCaution")),
            supported_claim_count=safe_int(verification.get("supportedClaimCount")),
            uncertain_claim_count=safe_int(verification.get("uncertainClaimCount")),
            unsupported_claim_count=safe_int(verification.get("unsupportedClaimCount")),
            conflict_mentioned=bool(verification.get("conflictMentioned")),
            rewrite_attempted=bool(rewrite.get("attempted")),
            rewrite_applied=bool(rewrite.get("applied")),
            final_answer_source=str(rewrite.get("finalAnswerSource") or "original"),
            warning_count=len(warnings),
            source_count=len(payload.get("sources") or []),
            evidence_count=len(payload.get("evidence") or []),
            answer_route=route_summary(router.get("selected")),
            verification_route=route_summary(verification.get("route")),
            rewrite_route=route_summary(rewrite.get("route")),
            warnings=warnings[:20],
        )
    except Exception:
        return


def build_answer_prompt(*, query: str, answer_signals: dict[str, Any]) -> str:
    quality_counts = dict(answer_signals.get("quality_counts") or {})
    representative_paper = dict(answer_signals.get("representative_paper") or {})
    concept_core = dict(answer_signals.get("concept_core_evidence") or {})
    answer_mode = clean_text(str(answer_signals.get("answer_mode") or ""))
    beginner_mode = answer_mode.endswith("_beginner") or bool(
        re.search(r"\b(simple|easy|beginner|for beginners|intuition)\b|쉽게|입문|초심자|직관", str(query or ""), re.IGNORECASE)
    )
    if bool(answer_signals.get("paper_definition_mode")) and representative_paper:
        representative_title = clean_text(str(representative_paper.get("title") or "대표 논문"))
        representative_paper_id = clean_text(str(representative_paper.get("paperId") or ""))
        representative_label = clean_text(str(representative_paper.get("citationLabel") or ""))
        supporting_paper_count = safe_int(answer_signals.get("supporting_paper_count"), 0)
        beginner_rules = (
            "- 쉬운 설명 요청이므로 전문 용어를 바로 풀어쓰고, 짧은 직관이나 비유를 1개만 사용한다.\n"
            "- 초심자가 떠올리기 쉬운 이미지로 설명하고, 수식이나 불필요한 역사 설명은 줄인다.\n"
            "- 기술 용어를 쓰면 같은 문장에서 쉬운 말로 바로 풀어쓴다.\n"
            "- 첫 두 단락에서는 논문명, 연도, 벤치마크보다 직관 설명에 집중한다.\n"
            "- GPU나 구현 세부는 핵심 원리를 설명한 뒤 대표 사례 단락에서만 다룬다.\n"
            if beginner_mode
            else ""
        )
        anchor_desc = representative_title
        if representative_paper_id:
            anchor_desc = f"{anchor_desc} ({representative_paper_id})"
        elif representative_label:
            anchor_desc = f"{anchor_desc} [{representative_label}]"
        return (
            "당신은 knowledge-hub의 근거 기반 한국어 답변기입니다.\n"
            f"질문: {query}\n"
            "답변 모드: paper_definition_anchor\n"
            f"대표 논문: {anchor_desc}\n"
            f"보조 논문 수: {supporting_paper_count}\n\n"
            "답변 규칙:\n"
            "- 먼저 질문의 개념 자체를 일반 개념으로 정의한다.\n"
            "- 답변 순서는 '한줄 정의 -> 작동 원리 -> 왜 중요한지 -> 대표 사례'를 지킨다.\n"
            "- 대표 논문은 개념의 본문 주인공이 아니라 대표 사례 또는 전환점으로만 소개한다.\n"
            "- 여러 논문의 공통분모를 섞어 뭉뚱그린 일반론으로 시작하지 않는다.\n"
            "- 첫 단락에서는 개념 정의와 핵심 메커니즘 2~3개만 설명한다.\n"
            "- 작동 원리에는 개념의 핵심 메커니즘만 넣고, 역사적 성공 요인이나 구현 세부는 뒤로 미룬다.\n"
            "- 왜 중요한지 한 단락으로 짧게 설명한 뒤, 대표 논문은 예시나 상징적 사례로 연결한다.\n"
            "- 대표 논문에 없는 요소를 핵심 아이디어처럼 섞어 단정하지 않는다.\n"
            "- 가능하면 '한줄 정의', '작동 원리', '왜 중요한지', '대표 사례', '한계/주의'가 드러나게 짧게 쓴다.\n"
            f"{beginner_rules}"
            "- 근거 없는 추론을 하지 말고, 답을 모르면 모른다고 말한다.\n\n"
            "근거 요약:\n"
            f"- ok={quality_counts.get('ok', 0)}, needs_review={quality_counts.get('needs_review', 0)}, "
            f"reject={quality_counts.get('reject', 0)}, unscored={quality_counts.get('unscored', 0)}\n"
            f"- concept_core={clean_text(str(concept_core.get('title') or representative_title))}\n"
            f"- representative_paper={anchor_desc}\n"
            f"- supporting_papers={supporting_paper_count}\n"
            f"- audience={'beginner' if beginner_mode else 'general'}\n"
            f"- caution_required={bool(answer_signals.get('caution_required'))}\n"
        )
    return (
        "당신은 knowledge-hub의 근거 기반 한국어 답변기입니다.\n"
        f"질문: {query}\n\n"
        "답변 규칙:\n"
        "- quality_flag=ok, source_trust_score가 높은 근거를 먼저 사용한다.\n"
        "- needs_review/reject/unscored 근거는 보조적 근거로만 사용하고, 핵심 결론을 단정하지 않는다.\n"
        "- contradiction_penalty가 큰 근거나 상충 belief가 있으면 견해 충돌이나 불확실성을 명시한다.\n"
        "- reference_tier=specialist 또는 glossary/standard/background_reference는 배경 정의나 표준 설명의 우선 근거로 본다.\n"
        "- 근거 없는 추론을 하지 말고, 답을 모르면 모른다고 말한다.\n"
        "- 가능하면 '핵심 답변', '근거', '불확실성/한계'가 드러나게 짧게 쓴다.\n\n"
        "근거 요약:\n"
        f"- ok={quality_counts.get('ok', 0)}, needs_review={quality_counts.get('needs_review', 0)}, "
        f"reject={quality_counts.get('reject', 0)}, unscored={quality_counts.get('unscored', 0)}\n"
        f"- specialist_references={int(answer_signals.get('specialist_reference_count') or 0)}\n"
        f"- contradictory_sources={int(answer_signals.get('contradictory_source_count') or 0)}\n"
        f"- contradicting_beliefs={int(answer_signals.get('contradicting_belief_count') or 0)}\n"
        f"- caution_required={bool(answer_signals.get('caution_required'))}\n"
    )


def build_claim_native_prompt(*, query: str, answer_provenance: str) -> str:
    return (
        "당신은 knowledge-hub의 claim-first 근거 기반 한국어 답변기입니다.\n"
        f"질문: {query}\n"
        f"answer_provenance={answer_provenance}\n\n"
        "규칙:\n"
        "- 반드시 Verified Claims를 1차 근거로 사용한다.\n"
        "- Alignment Groups가 있으면 같은 frame 안에서만 비교한다.\n"
        "- Conflicts가 있으면 한쪽으로 평탄화하지 말고 출처와 함께 충돌을 드러낸다.\n"
        "- Scope Warnings와 Abstention Conditions가 있으면 답변에 그대로 반영한다.\n"
        "- low-confidence fallback claim만 있을 때는 단정하지 말고 예비적 근거라고 명시한다.\n"
        "- Verified Claims에 없는 내용은 추론으로 보강하지 않는다.\n"
        "- 가능하면 '핵심 답변', '근거', '불확실성/한계'가 드러나게 짧게 쓴다.\n"
    )


def build_section_native_prompt(*, query: str, answer_provenance: str) -> str:
    return (
        "당신은 knowledge-hub의 section-first 근거 기반 한국어 답변기입니다.\n"
        f"질문: {query}\n"
        f"answer_provenance={answer_provenance}\n\n"
        "규칙:\n"
        "- 반드시 Selected Sections를 1차 이해 근거로 사용한다.\n"
        "- problem, method, results, limitations의 역할을 구분해서 설명한다.\n"
        "- Missing Sections나 Supplemental Retrieval Context가 있으면 불확실성/한계에 반영한다.\n"
        "- 외부 배경지식을 paper-native section 요약과 섞어 사실처럼 단정하지 않는다.\n"
        "- 선택된 section에 없는 내용은 추론으로 보강하지 않는다.\n"
        "- 가능하면 '핵심 답변', '근거', '불확실성/한계'가 드러나게 짧게 쓴다.\n"
    )


def build_claim_native_context(
    *,
    claim_cards: list[dict[str, Any]],
    claim_alignment: list[dict[str, Any]],
    claim_verification: list[dict[str, Any]],
    comparison_verification: dict[str, Any] | None = None,
    scope_warnings: list[str] | None = None,
    abstention_conditions: list[str] | None = None,
    supplemental_context: str = "",
) -> str:
    verification_by_id = {
        clean_text(item.get("claimCardId")): dict(item)
        for item in claim_verification or []
        if clean_text(item.get("claimCardId"))
    }
    claim_blocks: list[str] = []
    for index, claim in enumerate(claim_cards, 1):
        claim_id = clean_text(claim.get("claimCardId") or claim.get("claim_card_id"))
        verification = verification_by_id.get(claim_id, {})
        anchors = [clean_text(item) for item in list(claim.get("anchorExcerpts") or []) if clean_text(item)]
        claim_blocks.append(
            "\n".join(
                [
                    f"Claim {index} [{clean_text(claim.get('sourceKind') or claim.get('source_kind'))}:{clean_text(claim.get('sourceId') or claim.get('source_id'))}]",
                    f"  Text: {clean_text(claim.get('summaryText') or claim.get('summary_text') or claim.get('claimText') or claim.get('claim_text'))}",
                    (
                        "  Frame: "
                        f"task={clean_text(claim.get('taskCanonical') or claim.get('task'))}, "
                        f"dataset={clean_text(claim.get('datasetCanonical') or claim.get('dataset'))}, "
                        f"metric={clean_text(claim.get('metricCanonical') or claim.get('metric'))}, "
                        f"comparator={clean_text(claim.get('comparatorCanonical') or claim.get('comparator'))}"
                    ),
                    f"  Status: {clean_text(claim.get('status'))}, origin={clean_text(claim.get('origin'))}, trust={clean_text(claim.get('trustLevel') or claim.get('trust_level'))}",
                    f"  Conditions: {clean_text(claim.get('conditionText') or claim.get('condition_text')) or '(none)'}",
                    f"  Scope: {clean_text(claim.get('scopeText') or claim.get('scope_text')) or '(none)'}",
                    f"  Limitation: {clean_text(claim.get('negativeScopeText') or claim.get('negative_scope_text') or claim.get('limitationText') or claim.get('limitation_text')) or '(none)'}",
                    f"  Verification: {clean_text(verification.get('status')) or 'unknown'}",
                    f"  Evidence anchor: {anchors[0] if anchors else '(none)'}",
                ]
            )
        )
    alignment_blocks: list[str] = []
    for item in claim_alignment or []:
        frame = dict(item.get("canonicalFrame") or item.get("frame") or {})
        alignment_blocks.append(
            "\n".join(
                [
                    (
                        "Group "
                        f"{clean_text(item.get('groupKey')) or '(unnamed)'}: "
                        f"task={clean_text(frame.get('task'))}, "
                        f"dataset={clean_text(frame.get('dataset'))}, "
                        f"metric={clean_text(frame.get('metric'))}"
                    ),
                    f"  Claims: {', '.join(clean_text(v) for v in list(item.get('claimCardIds') or []) if clean_text(v)) or '(none)'}",
                    f"  Value spread: {json.dumps(item.get('valueSpread') or {}, ensure_ascii=False)}",
                    f"  Condition: {clean_text(item.get('conditionText')) or '(none)'}",
                ]
            )
        )
    conflict_lines = []
    for item in list((comparison_verification or {}).get("disagreements") or []):
        conflict_lines.append(
            f"- {clean_text(item.get('groupKey'))}: {', '.join(clean_text(v) for v in list(item.get('claimCardIds') or []) if clean_text(v))}"
        )
    scope_lines = [f"- {clean_text(item)}" for item in (scope_warnings or []) if clean_text(item)]
    abstention_lines = [f"- {clean_text(item)}" for item in (abstention_conditions or []) if clean_text(item)]
    parts = [
        "=== Verified Claims ===",
        "\n\n".join(claim_blocks) or "(none)",
        "=== Alignment Groups ===",
        "\n\n".join(alignment_blocks) or "(none)",
        "=== Conflicts ===",
        "\n".join(conflict_lines) or "(none)",
        "=== Scope Warnings ===",
        "\n".join(scope_lines) or "(none)",
        "=== Abstention Conditions ===",
        "\n".join(abstention_lines) or "(none)",
    ]
    if clean_text(supplemental_context):
        parts.extend(["=== Supplemental Retrieval Context ===", clean_text(supplemental_context)])
    return "\n\n".join(parts)


def build_section_native_context(
    *,
    section_cards: list[dict[str, Any]],
    section_coverage: dict[str, Any] | None = None,
    supplemental_context: str = "",
) -> str:
    blocks: list[str] = []
    for index, section in enumerate(section_cards, 1):
        key_points = [clean_text(item) for item in list(section.get("keyPoints") or section.get("key_points") or []) if clean_text(item)]
        scope_notes = [clean_text(item) for item in list(section.get("scopeNotes") or section.get("scope_notes") or []) if clean_text(item)]
        blocks.append(
            "\n".join(
                [
                    f"Section {index} [{clean_text(section.get('role')) or 'other'}:{clean_text(section.get('paperId') or section.get('paper_id'))}]",
                    f"  Title: {clean_text(section.get('title')) or '(untitled)'}",
                    f"  Path: {clean_text(section.get('sectionPath') or section.get('section_path')) or '(none)'}",
                    f"  Summary: {clean_text(section.get('contextualSummary') or section.get('contextual_summary')) or '(none)'}",
                    f"  Evidence excerpt: {clean_text(section.get('sourceExcerpt') or section.get('source_excerpt')) or '(none)'}",
                    f"  Thesis: {clean_text(section.get('documentThesis') or section.get('document_thesis')) or '(none)'}",
                    f"  Key points: {', '.join(key_points) or '(none)'}",
                    f"  Scope notes: {', '.join(scope_notes) or '(none)'}",
                    f"  Confidence: {safe_float(section.get('confidence'), 0.0):.3f}",
                ]
            )
        )
    coverage = dict(section_coverage or {})
    missing_roles = [clean_text(item) for item in list(coverage.get("missingRoles") or []) if clean_text(item)]
    weak_roles = [clean_text(item) for item in list(coverage.get("weakRoles") or []) if clean_text(item)]
    parts = [
        "=== Selected Sections ===",
        "\n\n".join(blocks) or "(none)",
        "=== Section Coverage ===",
        "\n".join(
            [
                f"- selected_roles={', '.join(clean_text(item) for item in list(coverage.get('selectedRoles') or []) if clean_text(item)) or '(none)'}",
                f"- missing_roles={', '.join(missing_roles) or '(none)'}",
                f"- weak_roles={', '.join(weak_roles) or '(none)'}",
                f"- status={clean_text(coverage.get('status')) or 'unknown'}",
            ]
        ),
    ]
    if clean_text(supplemental_context):
        parts.extend(["=== Supplemental Retrieval Context ===", clean_text(supplemental_context)])
    return "\n\n".join(parts)


def _extract_markdown_section(text: str, headings: tuple[str, ...]) -> str:
    body = str(text or "")
    if not body.strip():
        return ""
    escaped = "|".join(re.escape(item) for item in headings if clean_text(item))
    if not escaped:
        return ""
    pattern = re.compile(
        rf"(?ims)^\s*##+\s*(?:{escaped})\s*$\n+(.*?)(?=^\s*##+\s+|\Z)"
    )
    match = pattern.search(body)
    if not match:
        return ""
    return clean_text(match.group(1))


def _bounded_context_line(text: str, *, limit: int = 320) -> str:
    compact = clean_text(text)
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def build_paper_definition_context(
    *,
    query: str,
    filtered: list[SearchResult],
    evidence: list[dict[str, Any]],
    answer_signals: dict[str, Any],
    claim_context: str = "",
) -> str:
    representative = dict(answer_signals.get("representative_paper") or {})
    concept_core = dict(answer_signals.get("concept_core_evidence") or {})
    answer_mode = clean_text(str(answer_signals.get("answer_mode") or ""))
    beginner_mode = answer_mode.endswith("_beginner") or bool(
        re.search(r"\b(simple|easy|beginner|for beginners|intuition)\b|쉽게|입문|초심자|직관", str(query or ""), re.IGNORECASE)
    )
    representative_paper_id = clean_text(representative.get("paperId"))
    if not filtered or not evidence or not representative_paper_id:
        return clean_text(claim_context)

    evidence_by_paper_id = {
        clean_text(item.get("arxiv_id") or item.get("paper_id")): item
        for item in evidence
        if clean_text(item.get("arxiv_id") or item.get("paper_id"))
    }
    representative_result = next((item for item in filtered if clean_text((item.metadata or {}).get("arxiv_id") or (item.metadata or {}).get("paper_id")) == representative_paper_id), filtered[0])
    representative_document = str(getattr(representative_result, "document", "") or "")
    representative_evidence = evidence_by_paper_id.get(representative_paper_id, evidence[0] if evidence else {})
    concept_definition = _extract_markdown_section(representative_document, ("한줄 요약", "요약"))
    concept_mechanism = _extract_markdown_section(representative_document, ("핵심 아이디어", "Core Idea"))
    concept_method = _extract_markdown_section(representative_document, ("방법", "Method"))
    representative_fallback = clean_text(
        representative_evidence.get("excerpt")
        or representative_evidence.get("document")
        or representative_document
    )
    concept_summary = _bounded_context_line(
        concept_definition or clean_text(str(concept_core.get("summary") or "")) or representative_fallback,
        limit=320,
    )
    mechanism_summary = _bounded_context_line(concept_mechanism or concept_method or representative_fallback, limit=360)
    importance_lines = [
        f"- 이 질문은 '{clean_text(str(concept_core.get('title') or representative.get('title') or '핵심 개념'))}' 자체를 설명하는 과제다.",
        "- 대표 논문은 개념의 대표 사례나 전환점으로만 사용하고, 개념 자체와 동일시하지 않는다.",
        f"- 보조 논문 수: {safe_int(answer_signals.get('supporting_paper_count'), 0)}",
    ]
    parts = [
        "=== Paper Definition Task ===",
        f"Question: {clean_text(query)}",
        "Instruction: Explain the concept first. Use the representative paper only as a later example or turning point.",
        f"Audience: {'beginner' if beginner_mode else 'general'}",
        "=== Concept Core Evidence ===",
        f"concept_title={clean_text(str(concept_core.get('title') or representative.get('title') or '핵심 개념'))}",
        f"one_line_definition={concept_summary}",
        f"mechanism_summary={mechanism_summary}",
        "=== Why It Matters ===",
        "\n".join(importance_lines),
    ]
    if beginner_mode:
        parts.append("intuition_hint=Use one simple intuition or everyday metaphor and avoid jargon-heavy wording.")
    for index, result in enumerate(filtered[:2], 1):
        metadata = dict(result.metadata or {})
        paper_id = clean_text(metadata.get("arxiv_id") or metadata.get("paper_id"))
        evidence_item = evidence_by_paper_id.get(paper_id, evidence[index - 1] if index - 1 < len(evidence) else {})
        title = clean_text(metadata.get("title") or evidence_item.get("title") or "Untitled")
        document = str(getattr(result, "document", "") or "")
        role = "Representative Paper Example" if paper_id == representative_paper_id else "Support Paper Example"
        one_line = _extract_markdown_section(document, ("한줄 요약", "요약"))
        core_idea = _extract_markdown_section(document, ("핵심 아이디어", "Core Idea"))
        method = _extract_markdown_section(document, ("방법", "Method"))
        fallback = clean_text(
            evidence_item.get("excerpt")
            or evidence_item.get("document")
            or document
        )
        block = [
            f"=== {role} {index} ===",
            f"title={title}",
            f"paper_id={paper_id or '-'}",
            f"citation={clean_text(evidence_item.get('citation_label')) or '-'}",
            f"one_line_summary={_bounded_context_line(one_line or fallback, limit=320)}",
        ]
        if core_idea:
            block.append(f"core_idea={_bounded_context_line(core_idea, limit=420)}")
        if role == "Representative Paper Example" and method:
            block.append(f"method_hint={_bounded_context_line(method, limit=280)}")
        elif role != "Representative Paper Example":
            support_note = core_idea or one_line or method or fallback
            block.append(f"support_note={_bounded_context_line(support_note, limit=220)}")
        parts.append("\n".join(block))
    if clean_text(claim_context):
        parts.extend(["=== Claim Context ===", clean_text(claim_context)])
    return "\n\n".join(parts)


def build_answer_context(
    *,
    filtered: list[SearchResult],
    parent_ctx_by_result: dict[str, dict[str, Any]],
) -> str:
    context_parts: list[str] = []
    for i, result in enumerate(filtered, 1):
        parent_ctx = parent_ctx_by_result.get(result_id(result), {})
        title = result.metadata.get("title", "Untitled")
        file_path = result.metadata.get("file_path", "")
        src = result.metadata.get("source_type", "")
        section = result.metadata.get("section_title", "")
        summary = result.metadata.get("contextual_summary", "")
        parent_label = str(parent_ctx.get("parent_label", "")).strip()
        parent_span = str(parent_ctx.get("chunk_span", "")).strip()
        parent_id = str(parent_ctx.get("parent_id", "")).strip()
        ranking = dict((result.lexical_extras or {}).get("ranking_signals") or {})
        quality_flag = str((result.lexical_extras or {}).get("quality_flag") or "unscored")
        source_trust = safe_float((result.lexical_extras or {}).get("source_trust_score"), 0.0)
        reference_role = str((result.lexical_extras or {}).get("reference_role") or "")
        reference_tier = str((result.lexical_extras or {}).get("reference_tier") or "")

        header = [f"문서 {i}: {title}", f"[{src}]", f"file={file_path}"]
        if section:
            header.append(f"section={section}")
        if summary:
            header.append(f"summary={summary}")
        if parent_label:
            header.append(f"parent={parent_label}")
        if parent_span:
            header.append(f"chunks={parent_span}")
        if parent_id:
            header.append(f"parent_id={parent_id}")

        context_parts.append(
            f"{' '.join(header)}\n"
            f"시맨틱 점수: {result.semantic_score:.3f}, "
            f"키워드 점수: {result.lexical_score:.3f}, 최종: {result.score:.3f}\n"
            f"품질 신호: quality={quality_flag}, trust={source_trust:.3f}, "
            f"reference_role={reference_role or '-'}, reference_tier={reference_tier or '-'}, "
            f"contradiction_penalty={safe_float(ranking.get('contradiction_penalty'), 0.0):.3f}\n\n"
            f"{parent_ctx.get('text', result.document)}"
        )
    return "\n\n---\n\n".join(context_parts)


def answer_mentions_conflict(answer: str) -> bool:
    body = str(answer or "").lower()
    markers = (
        "불확실",
        "한계",
        "주의",
        "상충",
        "충돌",
        "모순",
        "해석이 갈릴",
        "단정하기 어렵",
        "may",
        "might",
        "uncertain",
        "conflict",
        "contradict",
    )
    return any(marker in body for marker in markers)


def split_answer_claims(answer: str) -> list[str]:
    raw = str(answer or "").strip()
    if not raw:
        return []
    normalized = re.sub(r"(?m)^\s{0,3}#+\s*", "", raw)
    normalized = re.sub(r"(?m)^\s*[-*]\s+", "", normalized)
    chunks = re.split(r"(?:\n{2,}|(?<=[.!?다요])\s+)", normalized)
    claims: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        cleaned = clean_text(chunk)
        if len(cleaned) < 12:
            continue
        if cleaned.endswith(":"):
            continue
        token = cleaned.lower()
        if token in seen:
            continue
        seen.add(token)
        claims.append(cleaned)
    return claims[:8]


def verification_route_stub(*, reason: str, route: str = "fallback-only") -> dict[str, Any]:
    return {
        "route": route,
        "provider": "",
        "model": "",
        "reasons": [reason],
        "fallbackUsed": route == "fallback-only",
    }


def rewrite_route_stub(*, reason: str, route: str = "fallback-only") -> dict[str, Any]:
    return {
        "route": route,
        "provider": "",
        "model": "",
        "reasons": [reason],
        "fallbackUsed": route == "fallback-only",
    }


def build_answer_verification_context(
    *,
    evidence: list[dict[str, Any]],
    answer_signals: dict[str, Any],
    contradicting_beliefs: list[dict[str, Any]],
) -> str:
    sections: list[str] = []
    sections.append("검증용 근거 번들:")
    for index, item in enumerate(evidence[:6], 1):
        sections.append(
            f"[근거 {index}] title={item.get('title', '')} "
            f"source_type={item.get('source_type', '')} "
            f"quality={item.get('quality_flag', 'unscored')} "
            f"trust={safe_float(item.get('source_trust_score'), 0.0):.3f} "
            f"reference_role={item.get('reference_role', '') or '-'} "
            f"reference_tier={item.get('reference_tier', '') or '-'}\n"
            f"excerpt={truncate_text(str(item.get('excerpt') or ''), 320)}"
        )
    contradiction_summary = [
        truncate_text(str(item.get("statement") or item.get("summary") or ""), 180)
        for item in contradicting_beliefs[:5]
        if str(item.get("statement") or item.get("summary") or "").strip()
    ]
    sections.append(
        "답변 신호:\n"
        f"- contradictory_sources={int(answer_signals.get('contradictory_source_count') or 0)}\n"
        f"- contradicting_beliefs={int(answer_signals.get('contradicting_belief_count') or 0)}\n"
        f"- caution_required={bool(answer_signals.get('caution_required'))}\n"
        f"- strongest_quality={answer_signals.get('strongest_quality_flag', 'unscored')}\n"
        f"- contradiction_summaries={contradiction_summary or []}"
    )
    return "\n\n".join(sections)


def build_answer_verification_prompt(*, query: str, answer: str) -> str:
    return (
        "당신은 knowledge-hub의 답변 검증기입니다.\n"
        "해야 할 일은 세 가지뿐입니다.\n"
        "1. 답변을 구체적 claim 단위로 분해한다.\n"
        "2. 제공된 근거에만 의존해서 각 claim을 supported|uncertain|unsupported 로 판정한다.\n"
        "3. 근거 충돌 신호가 있는데 답변이 이를 언급했는지 확인한다.\n\n"
        "판정 규칙:\n"
        "- supported: 제공된 근거 중 최소 하나가 claim을 직접 지지한다.\n"
        "- uncertain: 근거가 부분적이거나 해석 여지가 있다.\n"
        "- unsupported: 답변이 근거에 없는 구체적 사실을 새로 도입한다.\n"
        "- 근거 충돌/상충 신호가 있는데 답변이 불확실성이나 주의점을 말하지 않으면 conflictMentioned=false 로 둔다.\n"
        "- 근거가 부족하면 보수적으로 uncertain 으로 둔다.\n"
        "- 출력은 JSON 객체만 반환한다. 설명 문장은 JSON 내부 summary/reason에만 쓴다.\n\n"
        f"질문: {query}\n"
        f"답변:\n{answer}\n\n"
        "JSON schema:\n"
        "{\n"
        '  "claims": [{"claim": "...", "verdict": "supported|uncertain|unsupported", "evidenceTitles": ["..."], "reason": "..."}],\n'
        '  "conflictMentioned": true,\n'
        '  "needsCaution": false,\n'
        '  "summary": "짧은 한국어 요약"\n'
        "}"
    )


def default_answer_rewrite(
    *,
    answer: str,
    route: dict[str, Any] | None = None,
    summary: str = "",
) -> dict[str, Any]:
    return {
        "attempted": False,
        "applied": False,
        "triggeredBy": [],
        "attemptCount": 0,
        "summary": summary or "재작성 트리거가 없어 원본 답변을 유지했습니다.",
        "originalAnswer": answer,
        "finalAnswerSource": "original",
        "route": dict(route or rewrite_route_stub(reason="not_triggered", route="fallback-only")),
        "warnings": [],
    }


def should_rewrite_answer(verification: dict[str, Any]) -> list[str]:
    triggers: list[str] = []
    if int(verification.get("unsupportedClaimCount") or 0) > 0:
        triggers.append("unsupported_claim")
    if int(verification.get("claimUnsupportedCount") or 0) > 0:
        triggers.append("claim_unsupported")
    if bool(verification.get("needsCaution")) and not bool(verification.get("conflictMentioned")):
        triggers.append("missing_conflict_language")
    return list(dict.fromkeys(triggers))


def build_answer_rewrite_context(
    *,
    evidence: list[dict[str, Any]],
    answer_signals: dict[str, Any],
    contradicting_beliefs: list[dict[str, Any]],
    verification: dict[str, Any],
) -> str:
    evidence_context = build_answer_verification_context(
        evidence=evidence,
        answer_signals=answer_signals,
        contradicting_beliefs=contradicting_beliefs,
    )
    claim_lines = []
    for item in list(verification.get("claims") or [])[:8]:
        if not isinstance(item, dict):
            continue
        claim = truncate_text(str(item.get("claim") or ""), 180)
        verdict = str(item.get("verdict") or "").strip().lower()
        reason = truncate_text(str(item.get("reason") or ""), 160)
        if claim:
            claim_lines.append(f"- {verdict}: {claim} ({reason})")
    return (
        f"{evidence_context}\n\n"
        "기존 답변 검증 요약:\n"
        f"- status={verification.get('status', 'unknown')}\n"
        f"- unsupported={int(verification.get('unsupportedClaimCount') or 0)}\n"
        f"- uncertain={int(verification.get('uncertainClaimCount') or 0)}\n"
        f"- conflictMentioned={bool(verification.get('conflictMentioned'))}\n"
        f"- needsCaution={bool(verification.get('needsCaution'))}\n"
        f"- claims={claim_lines or []}"
    )


def build_answer_rewrite_prompt(
    *,
    query: str,
    original_answer: str,
    verification: dict[str, Any],
    triggered_by: list[str],
) -> str:
    return (
        "당신은 knowledge-hub의 근거 기반 답변 교정기입니다.\n"
        "목표는 기존 답변의 supported 부분은 최대한 보존하면서, 누락된 주의 문구와 citation 정렬만 교정하는 것입니다.\n\n"
        "교정 규칙:\n"
        "- unsupported claim을 새 표현으로 고쳐 쓰지 않는다. 근거가 부족하면 보수적 fallback이 담당한다.\n"
        "- conflictMentioned=false 이면 상충 가능성/불확실성/주의점을 반드시 명시한다.\n"
        "- 제공된 근거를 벗어나는 새로운 사실을 추가하지 않는다.\n"
        "- supported 내용은 가능한 한 유지한다.\n"
        "- 길이는 원본 답변과 비슷하거나 더 짧게 유지한다.\n"
        "- 출력은 교정된 최종 답변 텍스트만 반환한다.\n\n"
        f"질문: {query}\n"
        f"재작성 트리거: {', '.join(triggered_by) or 'none'}\n"
        f"현재 검증 요약: status={verification.get('status', 'unknown')}, "
        f"unsupported={int(verification.get('unsupportedClaimCount') or 0)}, "
        f"uncertain={int(verification.get('uncertainClaimCount') or 0)}, "
        f"conflictMentioned={bool(verification.get('conflictMentioned'))}\n\n"
        f"원본 답변:\n{original_answer}"
    )


def build_conservative_answer(
    *,
    verification: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> str:
    claims = [item for item in list(verification.get("claims") or []) if isinstance(item, dict)]
    supported = [clean_text(str(item.get("claim") or "")) for item in claims if str(item.get("verdict") or "").strip().lower() == "supported"]
    uncertain = [clean_text(str(item.get("claim") or "")) for item in claims if str(item.get("verdict") or "").strip().lower() == "uncertain"]
    evidence_titles = []
    seen_titles: set[str] = set()
    for item in evidence[:4]:
        title = str(item.get("title") or "").strip()
        if title and title not in seen_titles:
            seen_titles.add(title)
            evidence_titles.append(title)

    lines: list[str] = ["제공된 근거만으로는 질문에 대해 단정적인 결론을 내리기 어렵습니다."]
    if supported:
        lines.append("현재 근거에서 직접 확인되는 점은 다음과 같습니다:")
        for claim in supported[:3]:
            lines.append(f"- {claim}")
    if uncertain:
        lines.append("다만 다음 내용은 부분적 근거만 있어 보수적으로 해석해야 합니다:")
        for claim in uncertain[:2]:
            lines.append(f"- {claim}")
    elif not supported:
        lines.append("현재 근거에서는 직접 지지되는 핵심 claim을 충분히 확인하지 못했습니다.")
    if evidence_titles:
        lines.append(f"주요 근거: {', '.join(evidence_titles)}")
    return "\n".join(lines).strip()


def build_answer_generation_fallback(
    *,
    query: str,
    error: Exception,
    stage: str,
    evidence: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    routing_meta: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    title_lines: list[str] = []
    for item in evidence[:3]:
        title = clean_text(str(item.get("title") or ""))
        excerpt = truncate_text(str(item.get("excerpt") or item.get("document") or ""), 140)
        if title and excerpt:
            title_lines.append(f"- {title}: {excerpt}")
        elif title:
            title_lines.append(f"- {title}")

    answer_lines = ["로컬 답변 생성이 지연되어 검색된 근거만 보수적으로 정리합니다."]
    if title_lines:
        answer_lines.append("현재 상위 근거:")
        answer_lines.extend(title_lines)
    elif citations:
        answer_lines.append("현재 상위 근거:")
        answer_lines.extend(f"- {str(item.get('title') or '').strip()}" for item in citations[:3] if str(item.get("title") or "").strip())
    else:
        answer_lines.append(f"질문 `{query}` 에 대해 활용 가능한 근거가 충분하지 않아 보수적으로 응답을 제한합니다.")

    answer = "\n".join(answer_lines).strip()
    error_type = type(error).__name__
    error_message = truncate_text(str(error) or error_type, 180)
    generation_meta = {
        "status": "fallback",
        "stage": stage,
        "fallbackUsed": True,
        "errorType": error_type,
        "errorMessage": error_message,
        "route": dict(routing_meta or {}),
    }
    verification = {
        "status": "skipped",
        "summary": "LLM answer generation failed; returned conservative evidence-grounded fallback.",
        "claims": [],
        "supportedClaimCount": 0,
        "unsupportedClaimCount": 0,
        "uncertainClaimCount": 0,
        "conflictMentioned": True,
        "needsCaution": False,
        "warnings": [f"answer generation fallback applied: {stage}:{error_type}"],
    }
    rewrite_meta = default_answer_rewrite(
        answer=answer,
        route=routing_meta,
        summary="LLM answer generation failed; emitted conservative evidence-grounded fallback.",
    )
    rewrite_meta["finalAnswerSource"] = "generation_fallback"
    warnings = [f"answer generation fallback applied: {stage}:{error_type}"]
    return answer, generation_meta, verification, rewrite_meta, warnings
