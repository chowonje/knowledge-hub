"""Support helpers for canonical paper CLI commands."""

from __future__ import annotations

import json as _json
import re
from pathlib import Path
from typing import Any

from knowledge_hub.knowledge.ai_taxonomy import classify_ai_concept, merge_ai_classification_properties
from knowledge_hub.interfaces.cli.commands.paper_shared_runtime import MAX_SUMMARIZE_CHARS
from knowledge_hub.vault.concepts import iter_concept_note_paths

_FILELIKE_SUFFIX_RE = re.compile(r"\.(?:md|markdown|txt|json|ya?ml|toml|csv|pdf|docx?|py|ipynb|js|ts|tsx)$", re.IGNORECASE)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_ACRONYM_STOP_WORDS = {"a", "an", "and", "for", "in", "of", "on", "the", "to", "via", "with"}
_SENTENCE_LIKE_TOKEN_RE = re.compile(
    r"\b(?:across|advanced|advancing|affordable|analyzing|analysing|beyond|can|designing|efficient|evaluating|improving|revisiting|scaling|towards?|understanding)\b",
    re.IGNORECASE,
)
_GENERIC_CONCEPT_BLACKLIST = {
    "agents",
    "benchmark",
    "benchmarks",
    "standard",
    "standards",
    "survey",
    "surveys",
}
_PROTECTED_HEURISTIC_NAMES = {
    "AMA-Bench",
    "AgentDyn",
    "FIRE-Bench",
    "Gaia2",
    "LoCoMo",
    "Locomo-Plus",
    "Magma",
    "MemoryArena",
    "ProjDevBench",
}
_CURATED_CANONICAL_ALIASES = {
    "AI Agent": {"AI Agents"},
    "AI Coding Agent": {"AI Coding Agents"},
    "AI Scientist": {"AI scientists"},
    "Agent Benchmark": {"agent benchmarks"},
    "Agent Memory": {"agent memory", "Agent Memory"},
    "Coding Agent": {"Coding Agents", "Coding Agents?"},
    "Dense representation": {"Dense Representations"},
    "Evaluation metric": {"Evaluation Metrics"},
    "Foundation Model": {"Foundation Models"},
    "Language Model": {"Language Models"},
    "Large Language Model": {"Large Language Models", "LLMs"},
    "Large Reasoning Model": {"Large Reasoning Models"},
    "LLM Agent": {"LLM Agents", "LLM-based agents", "LLM- Agents"},
    "Memory-Augmented Generation": {"memory-augmented generation"},
    "Mistral 7B": {"mistral 7b", "Mistral 7b"},
    "Neural Network": {"Neural Networks"},
    "Object-centric representation": {"Object-Centric Representations"},
    "Open X-Embodiment": {"open x-embodiment"},
    "Prompt Injection": {"prompt injection attacks", "prompt injection attack"},
    "Reinforcement Learning": {"reinforcement learning"},
    "Retrieval-Augmented Generation": {"RAG", "retrieval-augmented generation"},
    "Segment Anything": {"segment anything"},
    "Small Language Model": {"Small Language Models"},
    "Table Reasoning": {"table reasoning"},
    "Vision-Language-Action Model": {"Vision-Language-Action Models"},
}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _clean_list(values: list[str], *, limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        token = _clean_text(raw)
        if not token:
            continue
        lowered = token.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(token)
        if limit is not None and len(result) >= limit:
            break
    return result


def _is_acronym_like(name: Any) -> bool:
    token = _clean_text(name)
    if " " in token or not token:
        return False
    compact = re.sub(r"[^A-Za-z0-9]", "", token)
    if len(compact) < 2 or len(compact) > 8:
        return False
    letters = [char for char in compact if char.isalpha()]
    if not letters:
        return False
    if all(char.isupper() for char in letters):
        return True
    if compact.endswith("s") and len(compact) > 2:
        stem = compact[:-1]
        stem_letters = [char for char in stem if char.isalpha()]
        return bool(stem_letters) and all(char.isupper() for char in stem_letters)
    return False


def _is_noise_concept_name(name: Any) -> bool:
    token = _clean_text(name).strip(" -:;,.")
    if not token:
        return True
    if "/" in token or "\\" in token:
        return True
    lowered = token.casefold()
    if _FILELIKE_SUFFIX_RE.search(lowered):
        return True
    if any(marker in token for marker in ("$", "\\", "{", "}", "<", ">", "=")):
        return True
    if "_" in token:
        return True
    if not re.search(r"[A-Za-z가-힣]", token):
        return True
    if token.casefold() in _GENERIC_CONCEPT_BLACKLIST:
        return True
    if len(re.findall(r"[A-Za-z0-9가-힣]+", token)) >= 3 and _SENTENCE_LIKE_TOKEN_RE.search(token):
        return True
    return False


def _is_protected_proper_noun_concept(name: Any) -> bool:
    token = _clean_text(name).strip(" -:;,.")
    if not token:
        return False
    if token in _PROTECTED_HEURISTIC_NAMES:
        return True
    if token.endswith("-Bench") or token.endswith("Bench"):
        return True
    if re.search(r"[A-Za-z]+\d", token):
        return True
    if re.search(r"[A-Z][a-z]+[A-Z][A-Za-z0-9-]*", token):
        return True
    return False


def _singularize_token(token: str) -> str:
    word = token.casefold()
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _concept_signature(name: Any) -> str:
    token = _clean_text(name)
    words = re.findall(r"[A-Za-z0-9가-힣]+", token.casefold())
    normalized = [_singularize_token(word) for word in words if word]
    return " ".join(normalized)


def _concept_acronym(name: Any) -> str:
    token = _clean_text(name)
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]*", token)
    if len(words) < 2:
        return ""
    initials = [word[0].upper() for word in words if word.casefold() not in _ACRONYM_STOP_WORDS]
    if len(initials) < 2:
        return ""
    return "".join(initials)


def _pick_canonical_name(names: list[str]) -> str:
    candidates = _clean_list(names)
    if not candidates:
        return ""
    return sorted(
        candidates,
        key=lambda item: (
            _is_acronym_like(item),
            -len(re.findall(r"[A-Za-z0-9가-힣]+", item)),
            len(item),
            item.casefold(),
        ),
    )[0]


def _dedupe_synonym_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, list[str]] = {}
    alias_to_canonical: dict[str, str] = {}
    for group in groups:
        canonical = _clean_text(group.get("canonical"))
        aliases = _clean_list(list(group.get("aliases") or []))
        if not canonical or not aliases:
            continue
        canonical_key = canonical.casefold()
        bucket = merged.setdefault(canonical_key, [canonical])
        for alias in aliases:
            alias_key = alias.casefold()
            if alias_key == canonical_key:
                continue
            existing = alias_to_canonical.get(alias_key)
            if existing and existing != canonical_key:
                continue
            alias_to_canonical[alias_key] = canonical_key
            if alias not in bucket[1:]:
                bucket.append(alias)
    result: list[dict[str, Any]] = []
    for values in merged.values():
        canonical = values[0]
        aliases = _clean_list(values[1:])
        if aliases:
            result.append({"canonical": canonical, "aliases": aliases})
    return result


def _detect_local_synonym_groups(concept_names: list[str]) -> tuple[list[dict[str, Any]], set[str]]:
    groups: list[dict[str, Any]] = []
    grouped_names: set[str] = set()

    present_by_casefold = {name.casefold(): name for name in concept_names}
    for canonical, aliases in _CURATED_CANONICAL_ALIASES.items():
        matched_aliases: list[str] = []
        for alias in aliases:
            resolved = present_by_casefold.get(alias.casefold())
            if resolved:
                matched_aliases.append(resolved)
        if not matched_aliases:
            continue
        groups.append({"canonical": canonical, "aliases": _clean_list(matched_aliases)})
        grouped_names.update(matched_aliases)

    by_signature: dict[str, list[str]] = {}
    for name in concept_names:
        if name in grouped_names:
            continue
        signature = _concept_signature(name)
        if not signature:
            continue
        by_signature.setdefault(signature, []).append(name)
    for names in by_signature.values():
        unique_names = _clean_list(names)
        if len(unique_names) < 2:
            continue
        canonical = _pick_canonical_name(unique_names)
        aliases = [name for name in unique_names if name.casefold() != canonical.casefold()]
        if aliases:
            groups.append({"canonical": canonical, "aliases": aliases})
            grouped_names.update(unique_names)

    remaining = [name for name in concept_names if name not in grouped_names]
    full_names_by_acronym: dict[str, list[str]] = {}
    acronym_names: dict[str, list[str]] = {}
    for name in remaining:
        if _is_acronym_like(name):
            compact = re.sub(r"[^A-Za-z0-9]", "", name).upper().rstrip("S")
            if compact:
                acronym_names.setdefault(compact, []).append(name)
            continue
        acronym = _concept_acronym(name)
        if acronym:
            full_names_by_acronym.setdefault(acronym, []).append(name)

    for acronym, aliases in acronym_names.items():
        full_names = _clean_list(full_names_by_acronym.get(acronym, []))
        if len(full_names) != 1:
            continue
        names = _clean_list(full_names + aliases)
        if len(names) < 2:
            continue
        canonical = _pick_canonical_name(names)
        alias_names = [name for name in names if name.casefold() != canonical.casefold()]
        if alias_names:
            groups.append({"canonical": canonical, "aliases": alias_names})
            grouped_names.update(names)

    return _dedupe_synonym_groups(groups), grouped_names


def _extract_json_candidates(raw: Any) -> list[str]:
    body = str(raw or "").strip()
    if not body:
        return []
    body = re.sub(r"<think>[\s\S]*?</think>", " ", body, flags=re.IGNORECASE).strip()
    candidates = [body]
    for match in _JSON_FENCE_RE.finditer(body):
        token = match.group(1).strip()
        if token:
            candidates.append(token)
    array_start = body.find("[")
    array_end = body.rfind("]")
    if array_start >= 0 and array_end > array_start:
        candidates.append(body[array_start: array_end + 1])
    obj_start = body.find("{")
    obj_end = body.rfind("}")
    if obj_start >= 0 and obj_end > obj_start:
        candidates.append(body[obj_start: obj_end + 1])
    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        token = str(candidate or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result


def _resolve_known_concept_name(raw: Any, known_names: list[str]) -> str:
    token = _clean_text(raw)
    if not token:
        return ""
    for name in known_names:
        if name.casefold() == token.casefold():
            return name
    return ""


def _trusted_synonym_pair(canonical: Any, alias: Any) -> bool:
    canonical_name = _clean_text(canonical)
    alias_name = _clean_text(alias)
    if not canonical_name or not alias_name:
        return False
    if canonical_name.casefold() == alias_name.casefold():
        return False
    if _is_protected_proper_noun_concept(canonical_name) or _is_protected_proper_noun_concept(alias_name):
        return False
    if _concept_signature(canonical_name) == _concept_signature(alias_name):
        return True

    canonical_acronym = re.sub(r"[^A-Za-z0-9]", "", canonical_name).upper().rstrip("S")
    alias_acronym = re.sub(r"[^A-Za-z0-9]", "", alias_name).upper().rstrip("S")
    if canonical_acronym and alias_acronym and canonical_acronym == alias_acronym:
        return True

    if _is_acronym_like(canonical_name) and _concept_acronym(alias_name) == canonical_acronym:
        return True
    if _is_acronym_like(alias_name) and _concept_acronym(canonical_name) == alias_acronym:
        return True

    return False


def _filter_llm_synonym_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for group in groups:
        canonical = _clean_text(group.get("canonical"))
        if not canonical:
            continue
        aliases = [
            alias
            for alias in _clean_list(list(group.get("aliases") or []))
            if _trusted_synonym_pair(canonical, alias)
        ]
        if aliases:
            filtered.append({"canonical": canonical, "aliases": aliases})
    return filtered


def _coerce_synonym_groups(raw: Any, known_names: list[str]) -> list[dict[str, Any]]:
    parsed_payload: Any = None
    for candidate in _extract_json_candidates(raw):
        try:
            parsed_payload = _json.loads(candidate)
        except Exception:
            continue
        if parsed_payload is not None:
            break
    if parsed_payload is None:
        return []

    group_items: list[Any] = []
    if isinstance(parsed_payload, list):
        group_items = list(parsed_payload)
    elif isinstance(parsed_payload, dict):
        for key in ("groups", "synonyms", "results", "items", "data"):
            value = parsed_payload.get(key)
            if isinstance(value, list):
                group_items = list(value)
                break
    if not group_items:
        return []

    normalized_groups: list[dict[str, Any]] = []
    for item in group_items:
        if not isinstance(item, dict):
            continue
        canonical = _resolve_known_concept_name(item.get("canonical"), known_names)
        if not canonical:
            continue
        aliases = _clean_list(
            [
                _resolve_known_concept_name(alias, known_names)
                for alias in list(item.get("aliases") or [])
            ]
        )
        aliases = [alias for alias in aliases if alias.casefold() != canonical.casefold()]
        if not aliases:
            continue
        normalized_groups.append({"canonical": canonical, "aliases": aliases})
    return _dedupe_synonym_groups(normalized_groups)


def upsert_ai_concept(
    sqlite_db,
    *,
    entity_id: str,
    canonical_name: str,
    source: str,
    description: str = "",
    title: str = "",
    tags: list[str] | None = None,
    related_names: list[str] | None = None,
    relation_predicates: list[str] | None = None,
):
    resolver = getattr(sqlite_db, "resolve_entity", None)
    if callable(resolver):
        resolved = resolver(canonical_name, entity_type="concept") or {}
        resolved_entity_id = _clean_text(resolved.get("entity_id"))
        if resolved_entity_id:
            entity_id = resolved_entity_id

    existing = sqlite_db.get_ontology_entity(entity_id) or {}
    current_properties = existing.get("properties") if isinstance(existing.get("properties"), dict) else {}
    effective_description = str(description or existing.get("description", "") or "")
    effective_confidence = float(existing.get("confidence", 1.0) or 1.0)
    classification = classify_ai_concept(
        canonical_name=canonical_name,
        title=title,
        tags=[str(item).strip() for item in (tags or []) if str(item).strip()],
        related_names=[str(item).strip() for item in (related_names or []) if str(item).strip()],
        relation_predicates=[str(item).strip() for item in (relation_predicates or []) if str(item).strip()],
        source_type="paper",
    )
    merged_properties, _changed = merge_ai_classification_properties(current_properties, classification)
    sqlite_db.upsert_ontology_entity(
        entity_id=entity_id,
        entity_type="concept",
        canonical_name=canonical_name,
        description=effective_description,
        properties=merged_properties,
        confidence=effective_confidence,
        source=source,
    )


def resolve_routed_llm(
    config,
    *,
    task_type: str,
    allow_external: bool,
    llm_mode: str = "auto",
    query: str = "",
    context: str = "",
    source_count: int = 0,
    provider_override: str | None = None,
    model_override: str | None = None,
    timeout_sec: int | None = None,
):
    from knowledge_hub.interfaces.cli.commands import paper_shared_runtime as _paper_runtime

    return _paper_runtime._resolve_routed_llm(
        config,
        task_type=task_type,
        allow_external=allow_external,
        llm_mode=llm_mode,
        query=query,
        context=context,
        source_count=source_count,
        provider_override=provider_override,
        model_override=model_override,
        timeout_sec=timeout_sec,
    )


def fallback_to_mini_llm(
    config,
    *,
    task_type: str,
    allow_external: bool,
    query: str,
    context: str,
):
    if not allow_external:
        return None, None, []
    return resolve_routed_llm(
        config,
        task_type=task_type,
        allow_external=True,
        llm_mode="mini",
        query=query,
        context=context,
        source_count=1,
    )


def assess_summary_quality(notes: str) -> dict:
    if not notes or len(notes.strip()) < 30:
        return {"score": 0, "label": "없음", "color": "red", "reasons": ["요약 없음"]}

    reasons = []
    score = 0
    text = notes.strip()

    if len(text) >= 500:
        score += 30
    elif len(text) >= 200:
        score += 15
    else:
        reasons.append(f"너무 짧음({len(text)}자)")

    structured_markers = ["### 한줄 요약", "### 핵심 기여", "### 방법론", "### 주요 실험", "### 한계"]
    found_sections = sum(1 for marker in structured_markers if marker in text)
    if found_sections >= 4:
        score += 40
    elif found_sections >= 2:
        score += 20
    else:
        reasons.append("구조화 부족")

    if any(char.isdigit() for char in text) and any(keyword in text for keyword in ["%", "정확도", "성능", "BLEU", "F1", "점", "배"]):
        score += 15
    else:
        reasons.append("구체적 수치 부족")

    if any(keyword in text for keyword in ["제안", "기여", "새로", "기존", "향상", "개선"]):
        score += 15
    else:
        reasons.append("핵심 기여 불명확")

    if text.startswith("citations:") or text.startswith("citation"):
        return {"score": 5, "label": "미요약", "color": "red", "reasons": ["초기 메모만 존재"]}

    if score >= 80:
        label, color = "우수", "green"
    elif score >= 50:
        label, color = "보통", "yellow"
    elif score >= 20:
        label, color = "미흡", "bright_red"
    else:
        label, color = "형편없음", "red"

    return {"score": score, "label": label, "color": color, "reasons": reasons}


def extract_summary_text(content: str, title: str, sqlite_db) -> str:
    placeholder = "요약본/번역본이 아직 등록되지 않았습니다"

    for heading in ["## 요약", "# 📌 한줄 요약", "## 초록"]:
        if heading in content:
            section = content.split(heading, 1)[1]
            next_heading = re.search(r"\n#{1,3} ", section)
            if next_heading:
                section = section[: next_heading.start()]
            section = section.strip()
            if section and placeholder not in section and len(section) > 20:
                return section[:3000]

    arxiv_match = re.search(r'arxiv_id:\s*"?([0-9]+\.[0-9]+)"?', content)
    if arxiv_match:
        aid = arxiv_match.group(1)
        paper = sqlite_db.get_paper(aid)
        if paper:
            notes = paper.get("notes", "")
            if notes and len(notes) > 30:
                return f"제목: {paper.get('title', title)}\n분야: {paper.get('field', '')}\n{notes}"[:3000]

    return f"제목: {title}"


def extract_keywords_with_evidence(llm, title: str, text: str, sqlite_db=None) -> list[dict]:
    prompt = (
        "You extract 5-10 core academic concepts from AI/ML papers. "
        "For each concept, provide a short evidence sentence from the text that "
        "shows why this concept is relevant to this paper, plus a confidence score.\n\n"
        "Return ONLY valid JSON: [{\"concept\": \"Name\", \"evidence\": \"sentence\", \"confidence\": 0.9}, ...]\n\n"
        "Rules:\n"
        "- Use SINGULAR form (e.g. 'Neural Network' not 'Neural Networks')\n"
        "- Use full names, not abbreviations\n"
        "- Use standard academic terms\n"
        "- confidence: 0.5-1.0 based on how central the concept is to this paper\n"
        "- evidence: 1 sentence from the text, or a brief paraphrase if exact quote unavailable\n\n"
        f"Paper: {title}\n\n{text[:2500]}"
    )
    raw = llm.generate(prompt).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    items = _json.loads(raw)
    if not isinstance(items, list):
        return []

    results = []
    seen = set()
    for item in items:
        if not isinstance(item, dict) or "concept" not in item:
            continue
        name = str(item["concept"]).strip()
        if not name or len(name) <= 1:
            continue
        if sqlite_db:
            canonical = sqlite_db.resolve_entity(name, entity_type="concept")
            if canonical:
                name = str(canonical.get("canonical_name", name))
        token = name.lower()
        if token in seen:
            continue
        seen.add(token)
        results.append(
            {
                "concept": name,
                "evidence": str(item.get("evidence", ""))[:500],
                "confidence": min(1.0, max(0.0, float(item.get("confidence", 0.7)))),
            }
        )
    return results


def extract_keywords_openai(llm, title: str, text: str, sqlite_db=None) -> list[str]:
    prompt = (
        "You extract 5-10 core academic concepts/keywords from AI/ML papers. "
        "Return ONLY a JSON array of English concept names. "
        "Use standard academic terms (e.g. 'Transformer', 'Attention Mechanism', "
        "'Reinforcement Learning', 'Knowledge Distillation'). "
        "Always use SINGULAR form (e.g. 'Neural Network' not 'Neural Networks'). "
        "Use full names, not abbreviations (e.g. 'Large Language Model' not 'LLM'). "
        "Do NOT include LaTeX commands, paper-specific names, or generic terms like 'AI' or 'deep learning' unless central.\n\n"
        f"Paper: {title}\n\n{text[:2500]}"
    )
    raw = llm.generate(prompt).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    keywords = _json.loads(raw)
    if not isinstance(keywords, list):
        return []

    result = []
    seen = set()
    for keyword in keywords:
        name = str(keyword).strip()
        if not name or len(name) <= 1:
            continue
        if sqlite_db:
            canonical = sqlite_db.resolve_entity(name, entity_type="concept")
            if canonical:
                name = str(canonical.get("canonical_name", name))
        token = name.lower()
        if token in seen:
            continue
        seen.add(token)
        result.append(name)
    return result


def update_note_concepts(content: str, concepts: list[str]) -> str:
    concept_lines = "# 🧩 내가 배워야 할 개념\n- [[00_Concept_Index]]\n"
    for concept in concepts:
        concept_lines += f"- [[{concept}]]\n"

    placeholder = "요약본/번역본이 아직 등록되지 않았습니다"
    if placeholder in content:
        cleaned = content
        for line in content.split("\n"):
            if placeholder in line:
                cleaned = cleaned.replace(line, "")
                break
        for line in cleaned.split("\n"):
            if "sync-keywords" in line:
                cleaned = cleaned.replace(line, "")
                break
        content = cleaned.rstrip() + "\n\n"

    if "내가 배워야 할 개념" in content:
        pattern = re.compile(r"(#[#\s]*🧩?\s*내가 배워야 할 개념.*?\n)((?:- \[\[.*?\]\]\n)*)", re.MULTILINE)
        if pattern.search(content):
            content = pattern.sub(concept_lines, content)
        else:
            content = content.rstrip() + "\n\n" + concept_lines
    elif "핵심 키워드:" in content:
        kw_line = re.search(r"핵심 키워드:.*\n", content)
        if kw_line:
            content = content[: kw_line.start()] + concept_lines + content[kw_line.end() :]
    else:
        content = content.rstrip() + "\n\n" + concept_lines

    return content


def regenerate_concept_index(index_path: Path, all_concepts: dict[str, list[str]]):
    sorted_concepts = sorted(all_concepts.items(), key=lambda item: -len(item[1]))

    lines = [
        "---",
        "title: 00_Concept_Index",
        "---",
        "",
        "# AI Papers Concept Index",
        "",
        "이 폴더 내 요약 노트에서 추출된 개념 링크 목록",
        "",
        "## 개념",
    ]
    for concept, papers in sorted_concepts:
        lines.append(f"- [[{concept}]] ({len(papers)})")

    lines.append("")
    index_path.write_text("\n".join(lines), encoding="utf-8")


def batch_describe_concepts(llm, batch: list[str], all_concepts: list[str]) -> dict:
    concept_list_str = ", ".join(all_concepts[:200])
    prompt = (
        "You are an AI/ML concept expert. For each concept, provide:\n"
        "1. A concise Korean description (1-2 sentences) explaining what it is\n"
        "2. 3-5 related concepts from the provided concept list\n\n"
        "Return ONLY valid JSON: {\"ConceptName\": {\"description\": \"한국어 설명\", \"related\": [\"Related1\", \"Related2\", ...]}, ...}\n"
        "Pick related concepts ONLY from the provided list. Be precise and educational.\n\n"
        f"Concepts to describe:\n{_json.dumps(batch, ensure_ascii=False)}\n\n"
        f"Available concepts for relations:\n{concept_list_str}"
    )
    raw = llm.generate(prompt).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return _json.loads(raw)


def build_concept_note(name: str, description: str, related: list[str], papers: list[str]) -> str:
    lines = [
        "---",
        "type: concept",
        f'title: "{name}"',
        "---",
        "",
        f"# {name}",
        "",
        description,
        "",
    ]

    if related:
        lines.append("## 관련 개념")
        for related_name in related:
            lines.append(f"- [[{related_name}]]")
        lines.append("")

    if papers:
        lines.append("## 관련 논문")
        for paper in papers:
            lines.append(f"- [[{paper}]]")
        lines.append("")

    lines.append("*[[00_Concept_Index|← 개념 목록으로]]*")
    lines.append("")
    return "\n".join(lines)


def rebuild_concept_index_with_relations(papers_dir: Path, concepts_dir: Path, concept_papers: dict[str, list[str]]):
    sorted_concepts = sorted(concept_papers.items(), key=lambda item: -len(item[1]))
    has_note = {item.stem for item in iter_concept_note_paths(concepts_dir)}

    lines = [
        "---",
        "title: 00_Concept_Index",
        "---",
        "",
        "# AI Papers Concept Index",
        "",
        "이 폴더 내 요약 노트에서 추출된 개념 링크 목록",
        f"총 **{len(sorted_concepts)}개** 개념 | **{len(has_note)}개** 설명 노트 생성됨",
        "",
    ]

    freq_groups = {"## 핵심 개념 (3회 이상)": [], "## 주요 개념 (2회)": [], "## 기타 개념 (1회)": []}
    for concept, papers in sorted_concepts:
        count = len(papers)
        status = "📝" if concept in has_note else "📌"
        entry = f"- {status} [[{concept}]] ({count}편)"
        if count >= 3:
            freq_groups["## 핵심 개념 (3회 이상)"].append(entry)
        elif count == 2:
            freq_groups["## 주요 개념 (2회)"].append(entry)
        else:
            freq_groups["## 기타 개념 (1회)"].append(entry)

    for heading, entries in freq_groups.items():
        if entries:
            lines.append(heading)
            lines.extend(entries)
            lines.append("")

    lines.append("")
    (papers_dir / "00_Concept_Index.md").write_text("\n".join(lines), encoding="utf-8")


def concept_id(name: str) -> str:
    return re.sub(r"\s+", "_", name.strip()).lower()


def detect_synonym_groups(llm, concept_names: list[str]) -> list[dict]:
    cleaned_names = _clean_list([name for name in concept_names if not _is_noise_concept_name(name)])
    if len(cleaned_names) < 2:
        return []

    local_groups, _grouped_names = _detect_local_synonym_groups(cleaned_names)
    llm_groups: list[dict[str, Any]] = []
    if len(cleaned_names) >= 2:
        prompt = (
            "You are an AI/ML terminology expert. Given a list of concept names, "
            "find groups of synonyms, abbreviations, plural/singular variants, or "
            "near-duplicates that should be merged into a single canonical concept.\n\n"
            "Rules:\n"
            "- Only group terms that truly refer to the SAME concept\n"
            "- Do NOT merge parent-child (e.g. 'Reinforcement Learning' and 'Multi-Agent RL' are different)\n"
            "- Do NOT merge benchmark names, project names, dataset names, or model names just because they look related\n"
            "- Prefer singular form as canonical\n"
            "- Prefer full name over abbreviation as canonical\n"
            "- Return ONLY a JSON array of {\"canonical\": \"...\", \"aliases\": [\"...\"]}\n"
            "- Skip concepts with no duplicates\n\n"
            + _json.dumps(cleaned_names, ensure_ascii=False)
        )
        raw = llm.generate(prompt).strip()
        llm_groups = _filter_llm_synonym_groups(_coerce_synonym_groups(raw, cleaned_names))
    return _dedupe_synonym_groups(local_groups + llm_groups)


def merge_obsidian_concept(papers_dir: Path, concepts_dir: Path, alias: str, canonical: str) -> int:
    safe_alias = re.sub(r'[\\/:*?"<>|]', "", alias).strip()
    safe_canonical = re.sub(r'[\\/:*?"<>|]', "", canonical).strip()
    alias_path = concepts_dir / f"{safe_alias}.md"
    canonical_path = concepts_dir / f"{safe_canonical}.md"

    if not alias_path.exists():
        return 0

    alias_content = alias_path.read_text(encoding="utf-8")
    alias_papers = set(re.findall(r"\[\[([^\]]+)\]\]", alias_content))

    if canonical_path.exists():
        canonical_content = canonical_path.read_text(encoding="utf-8")
        canonical_papers = set(re.findall(r"\[\[([^\]]+)\]\]", canonical_content))
        new_papers = alias_papers - canonical_papers - {canonical, "00_Concept_Index"}

        if new_papers and "## 관련 논문" in canonical_content:
            insert_point = canonical_content.index("## 관련 논문") + len("## 관련 논문")
            next_newline = canonical_content.index("\n", insert_point)
            extra = "\n".join(f"- [[{paper}]]" for paper in sorted(new_papers))
            canonical_content = canonical_content[:next_newline] + "\n" + extra + canonical_content[next_newline:]
            canonical_path.write_text(canonical_content, encoding="utf-8")

    alias_path.unlink()
    return 1


def replace_in_paper_notes(papers_dir: Path, old_name: str, new_name: str):
    old_link = f"[[{old_name}]]"
    new_link = f"[[{new_name}]]"
    for md_path in papers_dir.glob("*.md"):
        if md_path.name == "00_Concept_Index.md":
            continue
        content = md_path.read_text(encoding="utf-8")
        if old_link in content:
            md_path.write_text(content.replace(old_link, new_link), encoding="utf-8")


def assess_vault_note_quality(content: str) -> dict:
    placeholder = "아직 등록되지 않았습니다"

    if placeholder in content:
        return {"score": 0, "label": "플레이스홀더", "color": "red", "reason": "플레이스홀더만 존재"}

    summary_text = ""
    for heading in ["## 요약", "# 📌 한줄 요약"]:
        if heading in content:
            section = content.split(heading, 1)[1]
            next_heading = re.search(r"\n#{1,2} [^#]", section)
            if next_heading:
                section = section[: next_heading.start()]
            summary_text = section.strip()
            break

    if not summary_text or len(summary_text) < 50:
        return {"score": 0, "label": "요약없음", "color": "red", "reason": "요약 섹션 없음/부족"}

    has_garbled = bool(re.search(r"\\hline|\\begin\{tabular\}|\\end\{tabular\}|& &|\\\\", summary_text))
    if has_garbled:
        return {"score": 10, "label": "깨짐", "color": "red", "reason": "LaTeX 잔해 포함"}

    structured_markers = ["### 한줄 요약", "### 핵심 기여", "### 방법론"]
    has_structure = sum(1 for marker in structured_markers if marker in summary_text) >= 2
    if has_structure and len(summary_text) >= 500:
        return {"score": 90, "label": "우수", "color": "green", "reason": ""}

    title_match = re.search(r'title:\s*"?(.+?)"?\s*$', content, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else ""
    if title and len(summary_text) > 100 and not has_structure:
        title_words = set(title.lower().split()[:5])
        summary_lower = summary_text.lower()
        significant_words = [word for word in title_words if len(word) > 3]
        if significant_words:
            overlap = sum(1 for word in significant_words if word in summary_lower)
            if overlap == 0:
                return {"score": 15, "label": "엉뚱함", "color": "bright_red", "reason": "요약이 논문 제목과 무관"}

    if len(summary_text) >= 300:
        return {"score": 70, "label": "보통", "color": "yellow", "reason": "구조화 부족"}
    if len(summary_text) >= 100:
        return {"score": 40, "label": "미흡", "color": "bright_red", "reason": "짧음"}
    return {"score": 20, "label": "부실", "color": "red", "reason": "매우 짧음"}


def collect_vault_note_text(md_path: Path, papers_dir: Path) -> str:
    stem = md_path.stem

    txt_candidates = list(papers_dir.glob(f"{stem}*.txt"))
    if not txt_candidates:
        short = stem[:40]
        txt_candidates = [path for path in papers_dir.glob("*.txt") if short in path.stem]
    for txt in txt_candidates:
        text = txt.read_text(encoding="utf-8")
        if len(text) > 200:
            return text[:MAX_SUMMARIZE_CHARS]

    content = md_path.read_text(encoding="utf-8")

    arxiv_match = re.search(r"source:\s*(\d{4}\.\d{4,5})", content)
    if not arxiv_match:
        arxiv_match = re.search(r'arxiv_id:\s*"?(\d{4}\.\d{4,5})"?', content)
    if arxiv_match:
        aid = arxiv_match.group(1)
        for txt in papers_dir.glob(f"*{aid}*.txt"):
            text = txt.read_text(encoding="utf-8")
            if len(text) > 200:
                return text[:MAX_SUMMARIZE_CHARS]

    for heading in ["## 초록", "## Abstract", "## 초록 (한국어)"]:
        if heading in content:
            section = content.split(heading, 1)[1]
            next_heading = re.search(r"\n#{1,2} [^#]", section)
            if next_heading:
                section = section[: next_heading.start()]
            section = section.strip()
            if len(section) > 100:
                return f"제목: {stem}\n\n{section}"

    body = re.sub(r"^---.*?---\s*", "", content, flags=re.DOTALL)
    body = re.sub(r"\[\[.*?\]\]", "", body)
    body = body.strip()
    if len(body) > 100:
        return f"제목: {stem}\n\n{body[:MAX_SUMMARIZE_CHARS]}"

    return f"제목: {stem}"


def update_vault_note_summary(md_path: Path, summary: str):
    content = md_path.read_text(encoding="utf-8")

    placeholder = "아직 등록되지 않았습니다"
    if placeholder in content:
        for line in content.split("\n"):
            if placeholder in line:
                content = content.replace(line, "")
                break
        for line in content.split("\n"):
            if "sync-keywords" in line:
                content = content.replace(line, "")
                break

    if "## 요약" in content:
        lines = content.split("\n")
        start = None
        end = None
        for index, line in enumerate(lines):
            if line.strip() == "## 요약":
                start = index
            elif start is not None and re.match(r"^#{1,2} [^#]", line) and index > start:
                end = index
                break
        if start is not None:
            if end is None:
                end = len(lines)
            content = "\n".join(lines[:start] + ["## 요약", "", summary, ""] + lines[end:])
    elif "# 📌 한줄 요약" in content:
        lines = content.split("\n")
        start = None
        end = None
        for index, line in enumerate(lines):
            if "# 📌 한줄 요약" in line:
                start = index
            elif start is not None and re.match(r"^#{1,2} [^#]", line) and index > start + 1:
                end = index
                break
        if start is not None:
            if end is None:
                end = len(lines)
            content = "\n".join(lines[:start] + ["## 요약", "", summary, ""] + lines[end:])
    else:
        fm_end = content.find("---", content.find("---") + 3)
        if fm_end > 0:
            insert_at = fm_end + 3
            content = content[:insert_at] + "\n\n## 요약\n\n" + summary + "\n" + content[insert_at:]
        else:
            content = "## 요약\n\n" + summary + "\n\n" + content

    content = re.sub(r"\n{4,}", "\n\n\n", content)
    md_path.write_text(content, encoding="utf-8")


__all__ = [
    "assess_summary_quality",
    "assess_vault_note_quality",
    "batch_describe_concepts",
    "build_concept_note",
    "collect_vault_note_text",
    "concept_id",
    "detect_synonym_groups",
    "extract_keywords_openai",
    "extract_keywords_with_evidence",
    "extract_summary_text",
    "fallback_to_mini_llm",
    "merge_obsidian_concept",
    "rebuild_concept_index_with_relations",
    "regenerate_concept_index",
    "replace_in_paper_notes",
    "resolve_routed_llm",
    "update_note_concepts",
    "update_vault_note_summary",
    "upsert_ai_concept",
]
