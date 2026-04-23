"""Markdown note templates for Korean note materialization."""

from __future__ import annotations

import re
from typing import Any

import yaml

from knowledge_hub.notes.composer import compose_concept_note_payload, compose_source_note_payload

from knowledge_hub.notes.source_profile import filter_low_signal_evidence


VISIBLE_FRONTMATTER_KEYS = ("type", "status", "title", "updated")
OPTIONAL_VISIBLE_FRONTMATTER_KEYS = ("knowledge_label",)

_SOURCE_NOTE_LAYOUTS: dict[str, dict[str, Any]] = {
    "survey_taxonomy": {
        "thesis_label": "읽기 프레임",
        "claim_heading": "### 분류 축과 핵심 주장",
        "claim_fallback": "이 survey가 어떤 분류 축과 비교 기준을 제안하는지 추가 근거가 더 필요합니다.",
        "sections": (
            {
                "title": "### 분류/비교 프레임",
                "keys": ("contributions",),
                "fallback": "분류 체계와 비교 프레임은 원문과 대표 발췌를 함께 봐야 선명해집니다.",
            },
            {
                "title": "### 검토 범위와 비교 기준",
                "keys": ("methodology",),
                "fallback": "어떤 문헌과 기준으로 비교했는지 원문 범위를 확인해야 합니다.",
            },
            {
                "title": "### 핵심 근거와 연구 공백",
                "keys": ("results_or_findings", "key_results"),
                "fallback": "연구 공백이나 비교 결과를 드러내는 근거가 아직 부족합니다.",
            },
            {
                "title": "### 읽을 가치 / 해석 포인트",
                "keys": ("insights",),
                "fallback": "후속 논문을 읽기 전에 상위 프레임을 잡는 용도로 활용할 수 있습니다.",
            },
        ),
        "limitation_heading": "### 적용 범위와 놓친 지점",
        "limitation_fallback": "survey의 포함 범위와 제외된 영역을 원문 기준으로 확인해야 합니다.",
        "excerpt_title": "## 분류 근거 발췌",
    },
    "benchmark": {
        "thesis_label": "평가 질문",
        "claim_heading": "### 측정 대상과 핵심 주장",
        "claim_fallback": "무엇을 측정했고 어떤 격차를 드러내는지 핵심 주장을 더 확인해야 합니다.",
        "sections": (
            {
                "title": "### 벤치마크가 드러내는 것",
                "keys": ("contributions",),
                "fallback": "이 benchmark가 드러내려는 능력 범위를 먼저 확인해야 합니다.",
            },
            {
                "title": "### 평가 셋업",
                "keys": ("methodology",),
                "fallback": "데이터셋, 태스크, 지표를 포함한 평가 셋업이 더 필요합니다.",
            },
            {
                "title": "### 주요 점수/격차",
                "keys": ("results_or_findings", "key_results"),
                "fallback": "성능 수치나 사람-모델 격차를 드러내는 결과를 더 확인해야 합니다.",
            },
            {
                "title": "### 해석 포인트",
                "keys": ("insights",),
                "fallback": "점수보다 측정 의도와 실패 양상을 해석하는 데 활용해야 합니다.",
            },
        ),
        "limitation_heading": "### 측정 한계와 해석 주의점",
        "limitation_fallback": "측정 범위, 데이터 편향, 해석 한계를 원문과 함께 읽어야 합니다.",
        "excerpt_title": "## 평가 근거 발췌",
    },
    "system_card_safety_report": {
        "thesis_label": "안전 판단",
        "claim_heading": "### 위험·완화 핵심 판단",
        "claim_fallback": "공개된 위험, 완화 조치, 배포 범위를 더 확인해야 합니다.",
        "sections": (
            {
                "title": "### 공개된 capability/배포 범위",
                "keys": ("contributions",),
                "fallback": "어떤 capability와 배포 범위를 전제로 한 문서인지 더 확인해야 합니다.",
            },
            {
                "title": "### 평가 및 mitigation 근거",
                "keys": ("methodology",),
                "fallback": "평가 절차와 mitigation 근거가 더 필요합니다.",
            },
            {
                "title": "### 관측된 위험과 운영 신호",
                "keys": ("results_or_findings", "key_results"),
                "fallback": "위험 시나리오와 운영 신호를 보여주는 근거가 더 필요합니다.",
            },
            {
                "title": "### 운영 시사점",
                "keys": ("insights",),
                "fallback": "운영 정책과 모니터링 기준을 연결해서 읽어야 합니다.",
            },
        ),
        "limitation_heading": "### 남는 위험과 운영 제약",
        "limitation_fallback": "문서가 다루지 못한 위험과 운영 제약을 따로 검토해야 합니다.",
        "excerpt_title": "## 안전 근거 발췌",
    },
    "blog_tutorial": {
        "thesis_label": "실무 초점",
        "claim_heading": "### 따라갈 핵심 포인트",
        "claim_fallback": "이 튜토리얼이 실제로 무엇을 빠르게 익히게 해주는지 더 확인해야 합니다.",
        "sections": (
            {
                "title": "### 이 튜토리얼이 바로 주는 것",
                "keys": ("contributions",),
                "fallback": "빠르게 얻는 실무 가치와 적용 범위를 더 확인해야 합니다.",
            },
            {
                "title": "### 구현 절차",
                "keys": ("methodology",),
                "fallback": "구현 순서와 필요한 전제 단계를 원문에서 확인해야 합니다.",
            },
            {
                "title": "### 예제에서 확인한 포인트",
                "keys": ("results_or_findings", "key_results"),
                "fallback": "예제나 데모에서 확인되는 핵심 포인트를 더 확보해야 합니다.",
            },
            {
                "title": "### 바로 활용할 포인트",
                "keys": ("insights",),
                "fallback": "실무에 바로 옮길 수 있는 포인트를 더 정리해야 합니다.",
            },
        ),
        "limitation_heading": "### 전제 조건과 실무 주의점",
        "limitation_fallback": "환경 설정, 생략된 단계, 운영 제약을 따로 확인해야 합니다.",
        "excerpt_title": "## 절차/예제 발췌",
    },
    "method_paper": {
        "thesis_label": "핵심 논지",
        "claim_heading": "### 핵심 주장",
        "claim_fallback": "문서의 핵심 주장을 추가 근거와 함께 확인해야 합니다.",
        "sections": (
            {
                "title": "### 핵심 기여 (Contributions)",
                "keys": ("contributions",),
                "fallback": "문서의 기여를 뒷받침하는 근거가 더 필요합니다.",
            },
            {
                "title": "### 방법론 (Methodology)",
                "keys": ("methodology",),
                "fallback": "접근 방식과 실험 절차를 더 확인해야 합니다.",
            },
            {
                "title": "### 주요 결과/발견",
                "keys": ("results_or_findings", "key_results"),
                "fallback": "결과와 발견을 뒷받침하는 근거가 더 필요합니다.",
            },
            {
                "title": "### 읽을 가치 / 시사점",
                "keys": ("insights",),
                "fallback": "후속 구현이나 개념 연결 관점의 시사점을 더 확인해야 합니다.",
            },
        ),
        "limitation_heading": "### 한계 및 향후 과제",
        "limitation_fallback": "적용 범위와 실패 조건은 원문과 대표 근거를 함께 확인해야 합니다.",
        "excerpt_title": "## 원문 핵심 발췌",
    },
}

_CONCEPT_NOTE_LAYOUTS: dict[str, dict[str, str]] = {
    "generic": {
        "core_heading": "### 핵심 정의",
        "relation_heading": "### 주요 관계",
        "claim_heading": "### 대표 근거",
        "support_heading": "### 관련 문서",
        "source_heading": "### 관련 소스",
        "related_heading": "### 연결 개념",
    },
    "model": {
        "core_heading": "### 핵심 구조",
        "relation_heading": "### 어떤 구성요소/시스템과 연결되나",
        "claim_heading": "### 대표 설계 근거",
        "support_heading": "### 대표 문서",
        "source_heading": "### 관련 소스",
        "related_heading": "### 연결 개념",
    },
    "method": {
        "core_heading": "### 핵심 절차",
        "relation_heading": "### 적용 단계와 연결 요소",
        "claim_heading": "### 대표 적용 근거",
        "support_heading": "### 대표 문서",
        "source_heading": "### 관련 소스",
        "related_heading": "### 연결 개념",
    },
    "metric": {
        "core_heading": "### 무엇을 측정하나",
        "relation_heading": "### 해석에 같이 봐야 할 요소",
        "claim_heading": "### 대표 측정 근거",
        "support_heading": "### 대표 문서",
        "source_heading": "### 관련 소스",
        "related_heading": "### 연결 개념",
    },
    "task": {
        "core_heading": "### 무엇을 해결하나",
        "relation_heading": "### 연관 시스템/평가 요소",
        "claim_heading": "### 대표 문제 근거",
        "support_heading": "### 대표 문서",
        "source_heading": "### 관련 소스",
        "related_heading": "### 연결 개념",
    },
    "benchmark": {
        "core_heading": "### 무엇을 드러내나",
        "relation_heading": "### 측정 구성요소",
        "claim_heading": "### 대표 평가 근거",
        "support_heading": "### 대표 문서",
        "source_heading": "### 관련 소스",
        "related_heading": "### 연결 개념",
    },
    "safety_risk": {
        "core_heading": "### 어떤 위험인가",
        "relation_heading": "### 연관 시스템/완화 요소",
        "claim_heading": "### 대표 위험 근거",
        "support_heading": "### 대표 문서",
        "source_heading": "### 관련 소스",
        "related_heading": "### 연결 개념",
    },
}


def safe_file_name(name: str, fallback: str = "untitled", max_length: int = 120) -> str:
    token = str(name or "").strip()
    if not token:
        return fallback
    token = re.sub(r"[\\/:*?\"<>|]+", "-", token)
    token = re.sub(r"\s+", " ", token).strip()
    if len(token) > max_length:
        token = token[:max_length].rstrip(" .-_")
    return token or fallback



def slugify_title(name: str, fallback: str = "untitled", max_length: int = 96) -> str:
    token = safe_file_name(name, fallback=fallback, max_length=max_length).lower()
    token = re.sub(r"[^a-z0-9가-힣]+", "-", token)
    token = re.sub(r"-+", "-", token).strip("-")
    if len(token) > max_length:
        token = token[:max_length].rstrip("-")
    return token or fallback



def yaml_frontmatter(payload: dict[str, Any]) -> str:
    body = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False).strip()
    return f"---\n{body}\n---\n"



def split_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    text = str(content or "")
    if not text.startswith("---\n"):
        return {}, text
    marker = "\n---\n"
    end = text.find(marker, 4)
    if end == -1:
        return {}, text
    raw_frontmatter = text[4:end]
    body = text[end + len(marker):]
    try:
        data = yaml.safe_load(raw_frontmatter) or {}
    except Exception:
        data = {}
    return data if isinstance(data, dict) else {}, body



def update_frontmatter(content: str, updates: dict[str, Any]) -> str:
    current, body = split_frontmatter(content)
    current.update(updates or {})
    return yaml_frontmatter(current) + body.lstrip("\n")



def replace_frontmatter(content: str, frontmatter: dict[str, Any]) -> str:
    _, body = split_frontmatter(content)
    return yaml_frontmatter(frontmatter) + body.lstrip("\n")



def filter_visible_frontmatter(frontmatter: dict[str, Any], *, extra_keys: tuple[str, ...] = ()) -> dict[str, Any]:
    keys = tuple(VISIBLE_FRONTMATTER_KEYS) + tuple(
        key for key in extra_keys if key and key not in VISIBLE_FRONTMATTER_KEYS
    )
    payload = {key: frontmatter.get(key, "") for key in keys}
    return {key: value for key, value in payload.items() if value not in (None, "")}



def build_visible_frontmatter(
    *,
    note_type: str,
    status: str,
    title: str,
    updated: str,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "type": str(note_type or "").strip(),
        "status": str(status or "").strip(),
        "title": str(title or "").strip(),
        "updated": str(updated or "").strip(),
    }
    extras = {
        str(key): value
        for key, value in (extra_fields or {}).items()
        if str(key).strip() in OPTIONAL_VISIBLE_FRONTMATTER_KEYS
    }
    payload.update(extras)
    return filter_visible_frontmatter(payload, extra_keys=tuple(extras.keys()))



def upsert_marked_section(content: str, key: str, body: str) -> str:
    start = f"<!-- SECTION:{key}:start -->"
    end = f"<!-- SECTION:{key}:end -->"
    block = f"{start}\n{body.rstrip()}\n{end}"
    pattern = re.compile(rf"{re.escape(start)}.*?{re.escape(end)}", re.DOTALL)
    if pattern.search(content):
        return pattern.sub(block, content)
    if content and not content.endswith("\n"):
        content += "\n"
    return f"{content}\n{block}\n"



def _as_bullets(values: list[str], *, fallback: str) -> list[str]:
    items = [str(value).strip() for value in values if str(value or "").strip()]
    if not items:
        items = [fallback]
    bullets: list[str] = []
    for item in items:
        bullets.append(item if item.startswith("- ") else f"- {item}")
    return bullets



def _summary_sentences(summary: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+|\n+", str(summary or "").strip())
    return [token.strip() for token in raw if token.strip()]


def _normalize_summary_text(text: Any) -> str:
    raw = str(text or "").replace("\r\n", "\n").strip()
    if not raw:
        return ""
    lines: list[str] = []
    pending_blank = False
    for line in raw.splitlines():
        token = line.strip()
        if not token:
            if lines:
                pending_blank = True
            continue
        if re.match(r"^#{1,6}\s+", token):
            continue
        token = re.sub(r"^\*\*([^*]+?)\*\*:?\s*", r"\1", token).strip()
        token = token.rstrip(":").strip()
        if re.match(r"^(관련 문서|관련 문서 및 지지 근거|관련 문서 및 인용|핵심 기술 용어)\s*:?\s*$", token):
            continue
        if token in {"-", "*"}:
            continue
        if pending_blank and lines:
            lines.append("")
        pending_blank = False
        lines.append(token)
    return "\n".join(lines).strip()


def _summary_line(text: Any, fallback: str) -> str:
    normalized = re.sub(r"\s+", " ", _normalize_summary_text(text))
    if not normalized:
        return fallback
    head = re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)[0].strip()
    candidate = head or normalized
    if len(candidate) > 180:
        candidate = candidate[:177].rstrip() + "..."
    return candidate


def _source_layout(document_type: str) -> dict[str, Any]:
    token = str(document_type or "").strip() or "method_paper"
    return _SOURCE_NOTE_LAYOUTS.get(token, _SOURCE_NOTE_LAYOUTS["method_paper"])


def _concept_layout(concept_type: str) -> dict[str, str]:
    token = str(concept_type or "").strip() or "generic"
    return _CONCEPT_NOTE_LAYOUTS.get(token, _CONCEPT_NOTE_LAYOUTS["generic"])


def _collect_source_items(payload: dict[str, Any], keys: tuple[str, ...], *, limit: int) -> list[str]:
    values: list[str] = []
    for key in keys:
        raw = payload.get(key)
        if isinstance(raw, list):
            values.extend(str(item).strip() for item in raw if str(item).strip())
        elif str(raw or "").strip():
            values.append(str(raw).strip())
    return filter_low_signal_evidence(values, limit=limit)


def _source_section_lines(
    payload: dict[str, Any],
    *,
    keys: tuple[str, ...],
    fallback: str,
    limit: int = 4,
) -> list[str]:
    items = _collect_source_items(payload, keys, limit=limit)
    return _as_bullets(items, fallback=fallback)


def _source_core_concept_lines(payload: dict[str, Any]) -> list[str]:
    explicit = [str(item).strip() for item in (payload.get("core_concepts") or []) if str(item).strip()]
    if explicit:
        return _as_bullets(explicit[:8], fallback="핵심 용어 없음")
    entity_lines = [str(item).lstrip("- ").strip() for item in (payload.get("entity_lines") or []) if str(item).strip()]
    return _as_bullets(entity_lines[:8], fallback="핵심 용어 없음")



def render_source_note(payload: dict[str, Any]) -> str:
    payload = compose_source_note_payload(payload)
    frontmatter_payload = payload.get("frontmatter") if isinstance(payload.get("frontmatter"), dict) else {}
    frontmatter = yaml_frontmatter(
        filter_visible_frontmatter(
            frontmatter_payload,
            extra_keys=tuple(key for key in OPTIONAL_VISIBLE_FRONTMATTER_KEYS if key in frontmatter_payload),
        )
    )
    related_concepts = _as_bullets(payload.get("related_concepts") or [], fallback="관련 개념 없음")
    source_items = [payload.get("source_url") or ""]
    source_items.extend(payload.get("representative_sources") or [])
    sources = _as_bullets(source_items, fallback="원문 URL 없음")
    summary_body = _normalize_summary_text(payload.get("core_summary") or payload.get("summary_ko"))
    summary_line = str(payload.get("summary_line_ko") or "").strip() or _summary_line(summary_body, "핵심 요약 없음")
    document_type = str(payload.get("document_type") or "").strip()
    thesis = str(payload.get("thesis") or "").strip()
    layout = _source_layout(document_type)
    claim_lines = _source_section_lines(
        payload,
        keys=("top_claims", "claim_lines"),
        fallback=str(layout["claim_fallback"]),
        limit=4,
    )
    limitation_lines = _source_section_lines(
        payload,
        keys=("limitations",),
        fallback=str(layout["limitation_fallback"]),
        limit=4,
    )

    excerpt_sections: list[str] = []
    ko_excerpts = [str(item).strip() for item in (payload.get("key_excerpts_ko") or []) if str(item).strip()]
    en_excerpts = [str(item).strip() for item in (payload.get("key_excerpts_en") or []) if str(item).strip()]
    for index, original in enumerate(en_excerpts[:2], start=1):
        ko_line = ko_excerpts[index - 1] if index - 1 < len(ko_excerpts) else "핵심 발췌 요약 없음"
        excerpt_sections.extend(
            [
                f"### 근거 {index}",
                f"- 한국어 요약: {ko_line.lstrip('- ').strip()}",
                "> 원문 발췌",
                f"> {original}",
                "",
            ]
        )
    if not excerpt_sections:
        excerpt_sections = ["- 발췌 근거 없음", ""]

    section_blocks: list[str] = []
    for section in layout["sections"]:
        section_blocks.extend(
            [
                str(section["title"]),
                *_source_section_lines(
                    payload,
                    keys=tuple(section["keys"]),
                    fallback=str(section["fallback"]),
                    limit=4,
                ),
                "",
            ]
        )

    body = [
        f"# {payload.get('title_ko') or payload.get('title_en')}",
        "",
        "## 요약",
        "",
        f"- 문서 타입: `{document_type}`" if document_type else "",
        f"- {layout['thesis_label']}: {thesis}" if thesis else "",
        "",
        "### 한줄 요약",
        summary_line,
        "",
        str(layout["claim_heading"]),
        *claim_lines,
        "",
        *section_blocks,
        str(layout["limitation_heading"]),
        *limitation_lines,
        "",
        "### 핵심 용어",
        *_source_core_concept_lines(payload),
        "",
        str(layout["excerpt_title"]),
        *excerpt_sections,
        "## 관련 개념",
        *related_concepts,
        "",
        "## 출처",
        *sources,
        "",
    ]
    return frontmatter + "\n".join(body).rstrip() + "\n"



def build_concept_sections(payload: dict[str, Any]) -> dict[str, str]:
    payload = compose_concept_note_payload(payload)
    layout = _concept_layout(str(payload.get("concept_type") or "generic"))
    core_summary = _normalize_summary_text(payload.get("core_summary") or payload.get("summary_ko"))
    summary_lines = [
        "## KHUB Auto Summary",
        core_summary,
        "",
        layout["claim_heading"],
        *(_as_bullets(payload.get("claim_lines") or [], fallback="대표 주장 없음")),
        "",
        layout["relation_heading"],
        *(_as_bullets(payload.get("relation_lines") or [], fallback="자동 추출 관계 없음")),
    ]
    evidence_lines = [
        "## KHUB Evidence",
        layout["support_heading"],
        *(_as_bullets(payload.get("support_lines") or [], fallback="근거 문서 없음")),
        "",
        "### 핵심 근거 (한국어)",
        *(_as_bullets(payload.get("key_excerpts_ko") or [], fallback="핵심 근거 없음")),
    ]
    related_lines = [
        "## KHUB Related Sources",
        layout["source_heading"],
        *(_as_bullets(payload.get("related_sources") or [], fallback="관련 소스 없음")),
    ]
    return {
        "khub-auto-summary": "\n".join(summary_lines).strip(),
        "khub-evidence": "\n".join(evidence_lines).strip(),
        "khub-related-sources": "\n".join(related_lines).strip(),
    }



def render_concept_note(payload: dict[str, Any]) -> str:
    payload = compose_concept_note_payload(payload)
    frontmatter_payload = payload.get("frontmatter") if isinstance(payload.get("frontmatter"), dict) else {}
    frontmatter = yaml_frontmatter(
        filter_visible_frontmatter(
            frontmatter_payload,
            extra_keys=tuple(key for key in OPTIONAL_VISIBLE_FRONTMATTER_KEYS if key in frontmatter_payload),
        )
    )
    layout = _concept_layout(str(payload.get("concept_type") or "generic"))
    related_concepts = _as_bullets(payload.get("related_concepts") or [], fallback="관련 개념 없음")
    support_lines = _as_bullets(payload.get("support_lines") or [], fallback="근거 문서 없음")
    related_sources = _as_bullets(payload.get("related_sources") or [], fallback="관련 소스 없음")
    relation_lines = _as_bullets(payload.get("relation_lines") or [], fallback="자동 추출 관계 없음")
    claim_lines = _as_bullets(payload.get("claim_lines") or [], fallback="대표 주장 없음")
    excerpt_lines = _as_bullets(payload.get("key_excerpts_ko") or [], fallback="핵심 근거 없음")
    core_summary = _normalize_summary_text(payload.get("core_summary") or payload.get("summary_ko"))
    summary_line = str(payload.get("summary_line_ko") or "").strip() or _summary_line(core_summary, "핵심 요약 없음")

    why_it_matters = _as_bullets(
        payload.get("why_it_matters") or [
            "여러 문서에서 반복적으로 등장하며 다른 개념과의 연결 강도가 높은 개념입니다.",
            "학습/검색/설계 맥락에서 재사용 가능한 기준 개념으로 활용할 수 있습니다.",
        ],
        fallback="학습/검색/설계 맥락에서 재사용 가능한 기준 개념으로 활용할 수 있습니다.",
    )

    body = [
        f"# {payload.get('title_ko') or payload.get('title')}",
        "",
        "## 요약",
        "",
        "### 한줄 요약",
        summary_line,
        "",
        layout["core_heading"],
        core_summary or "핵심 정의 없음",
        "",
        "### 왜 중요한가",
        *why_it_matters,
        "",
        layout["relation_heading"],
        *relation_lines,
        "",
        layout["claim_heading"],
        *claim_lines,
        "",
        *excerpt_lines,
        "",
        layout["support_heading"],
        *support_lines,
        "",
        layout["source_heading"],
        *related_sources,
        "",
        layout["related_heading"],
        *related_concepts,
        "",
    ]
    return frontmatter + "\n".join(body).rstrip() + "\n"



def merge_manual_concept_note(existing_content: str, payload: dict[str, Any]) -> str:
    sections = build_concept_sections(payload)
    updated = existing_content
    for key, body in sections.items():
        updated = upsert_marked_section(updated, key, body)
    return updated if updated.endswith("\n") else f"{updated}\n"
