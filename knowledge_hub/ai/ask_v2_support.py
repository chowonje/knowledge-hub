from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from knowledge_hub.ai.retrieval_fit import normalize_source_type
from knowledge_hub.core.chunking import snippet_for_path
from knowledge_hub.domain.vault_knowledge.scope import (
    vault_scope_from_filter as _domain_vault_scope_from_filter,
    vault_scope_from_query as _domain_vault_scope_from_query,
)


_TEMPORAL_RE = re.compile(r"\b(latest|recent|updated|newest|before|after|since)\b|최근|최신|업데이트|이전|이후", re.IGNORECASE)
_HARD_TEMPORAL_RE = re.compile(r"\b(latest|updated|newest|before|after|since)\b|최신|업데이트|이전|이후", re.IGNORECASE)
_COMPARE_RE = re.compile(r"\b(compare|comparison|versus|vs|difference)\b|비교|차이", re.IGNORECASE)
_RELATION_RE = re.compile(r"\b(related|relationship|connected|link|dependency|depends on)\b|관계|연결|의존", re.IGNORECASE)
_DEFINITION_RE = re.compile(
    r"\b(what is|define|definition|meaning of|concept|core idea|main idea|principle|intuition)\b|정의|개념|무엇|뭐야|뭐지|뭔지|핵심\s*아이디어|원리|직관",
    re.IGNORECASE,
)
_EXPLAIN_RE = re.compile(r"\b(explain|explainer)\b|설명", re.IGNORECASE)
_EVAL_RE = re.compile(r"\b(result|benchmark|metric|evaluate|evaluation|performance|accuracy)\b|결과|평가|성능|지표", re.IGNORECASE)
_IMPL_RE = re.compile(r"\b(implement|implementation|architecture|pipeline|how works|how to build)\b|구현|아키텍처|파이프라인", re.IGNORECASE)
_ARXIV_RE = re.compile(r"\b\d{4}\.\d{4,5}\b")
_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
_PY_SYMBOL_RE = re.compile(r"^\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
_JS_SYMBOL_RE = re.compile(r"^\s*(?:export\s+)?(?:async\s+function|function|class|const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)", re.MULTILINE)
_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([A-Za-z0-9_./-]+)\s+import|import\s+([A-Za-z0-9_.,\s/-]+)|const\s+.*=\s+require\(['\"]([^'\"]+)['\"]\)|import\s+.*from\s+['\"]([^'\"]+)['\"])",
    re.MULTILINE,
)
_CALL_SKIP = {"if", "for", "while", "return", "print", "len", "range", "class", "def", "await"}
_SYMBOL_LINE_RE = re.compile(r"^\s*(?:async\s+def|def|class|export\s+class|export\s+function|function|const|let|var)\b", re.IGNORECASE)
_IMPORT_LINE_RE = re.compile(r"^\s*(?:from\s+\S+\s+import|import\s+\S+|const\s+.*=\s+require\(|import\s+.*from\s+['\"])", re.IGNORECASE)
_TEST_PATH_RE = re.compile(r"(^|/)(tests?|spec)(/|$)|(^|/)test_[^/]+|[._-](test|spec)\.", re.IGNORECASE)
_DOC_PATH_RE = re.compile(r"(^|/)(docs?)(/|$)|(^|/)readme(?:\.[a-z0-9]+)?$", re.IGNORECASE)
_ADAPTER_PATH_RE = re.compile(r"(^|/)(adapters?|integrations?|gateways?|clients?|controllers?|handlers?|routers?|routes?)(/|$)", re.IGNORECASE)
_ENTRYPOINT_PATH_RE = re.compile(r"(^|/)(__main__|main|app|server|cli|manage|index)\.[a-z0-9]+$", re.IGNORECASE)
_CONFIG_PATH_RE = re.compile(r"(^|/)(settings|config|pyproject|package|docker-compose|compose|vite|webpack|tsconfig|jest)\b|(^|/)(dockerfile)$", re.IGNORECASE)
_OVERVIEW_QUERY_RE = re.compile(r"\b(readme|docs?|documentation|overview|guide|setup|install|explain|summary)\b|문서|가이드|개요|설명|요약", re.IGNORECASE)
_DEBUG_QUERY_RE = re.compile(r"\b(test|tests|spec|debug|bug|error|failing|failure|pytest|unit test|integration test)\b|테스트|디버그|버그|오류|실패", re.IGNORECASE)
_ARCH_QUERY_RE = re.compile(r"\b(architecture|architectural|pipeline|flow|entrypoint|integration|boundary|module|service)\b|아키텍처|파이프라인|흐름|구조|엔트리|통합", re.IGNORECASE)
_NOISE_FILE_NAMES = {"readme.md", "readme", "license", "changelog.md"}
_COMMON_INTERNAL_IMPORTS = {
    "app",
    "apps",
    "client",
    "clients",
    "config",
    "configs",
    "controller",
    "controllers",
    "core",
    "gateway",
    "gateways",
    "handler",
    "handlers",
    "helper",
    "helpers",
    "lib",
    "libs",
    "model",
    "models",
    "module",
    "modules",
    "route",
    "router",
    "routes",
    "service",
    "services",
    "src",
    "test",
    "tests",
    "util",
    "utils",
}
_PROJECT_ALLOWED_SUFFIXES = {".json", ".js", ".jsx", ".md", ".py", ".toml", ".ts", ".tsx", ".yaml", ".yml"}
_PROJECT_EXCLUDED_DIRS = {".git", ".mypy_cache", ".next", ".pytest_cache", ".venv", "build", "dist", "node_modules", "venv"}
_PROJECT_EXCLUDED_PARTS = {"__pycache__", ".ipynb_checkpoints", "build", "dist", "node_modules", "site-packages", "dist-packages", "vendor", "third_party"}


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def query_terms(query: str) -> list[str]:
    return [part.casefold() for part in re.split(r"[^0-9A-Za-z가-힣._/-]+", str(query or "").strip()) if part.strip()]


def clean_lines(values: list[Any], *, limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        token = clean_text(raw)
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


def resolved_source_ids(
    frame_payload: dict[str, Any] | None,
    query_plan_payload: dict[str, Any] | None,
) -> list[str]:
    frame = dict(frame_payload or {})
    query_plan = dict(query_plan_payload or {})
    values = (
        list(frame.get("resolved_source_ids") or [])
        or list(query_plan.get("resolvedSourceIds") or [])
        or list(query_plan.get("resolvedPaperIds") or [])
        or list(query_plan.get("resolved_paper_ids") or [])
    )
    return clean_lines(list(values))


def build_ranked_forms(
    values: list[Any],
    *,
    limit: int,
    exclude_casefold: set[str] | None = None,
) -> list[str]:
    blocked = {clean_text(item).casefold() for item in list(exclude_casefold or set()) if clean_text(item)}
    result: list[str] = []
    for raw in values:
        token = clean_text(raw)
        if not token:
            continue
        if token.casefold() in blocked:
            continue
        if any(existing.casefold() == token.casefold() for existing in result):
            continue
        result.append(token)
        if len(result) >= limit:
            break
    return result


def first_nonempty_search(
    *,
    forms: list[str] | None,
    fallback_query: str,
    search_fn,
) -> list[Any]:
    for form in list(forms or []) or [fallback_query]:
        results = list(search_fn(form) or [])
        if results:
            return results
    return []


def accumulate_search_results(
    *,
    forms: list[str] | None,
    fallback_query: str,
    search_fn,
) -> list[Any]:
    result: list[Any] = []
    for form in list(forms or []) or [fallback_query]:
        result.extend(list(search_fn(form) or []))
    return result


def build_paper_selection_inputs(
    *,
    query: str,
    frame_payload: dict[str, Any] | None,
    query_plan_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    frame = dict(frame_payload or {})
    query_plan = dict(query_plan_payload or {})
    return {
        "family": clean_text(frame.get("family") or query_plan.get("family") or ""),
        "resolved_paper_ids": resolved_source_ids(frame, query_plan),
        "lookup_forms": build_ranked_forms(
            [query, *list(frame.get("expanded_terms") or query_plan.get("expandedTerms") or query_plan.get("expanded_terms") or [])],
            limit=4,
        ),
    }


def build_web_selection_inputs(
    *,
    query: str,
    frame_payload: dict[str, Any] | None,
    query_plan_payload: dict[str, Any] | None,
    metadata_filter: dict[str, Any] | None,
) -> dict[str, Any]:
    frame = dict(frame_payload or {})
    query_plan = dict(query_plan_payload or {})
    effective_metadata_filter = dict(frame.get("metadata_filter") or metadata_filter or {})
    family = clean_text(frame.get("family") or query_plan.get("family") or "")
    media_platform = clean_text(effective_metadata_filter.get("media_platform"))
    resolved_ids = resolved_source_ids(frame, query_plan)
    if media_platform.casefold() == "youtube":
        search_forms = build_ranked_forms([query, *list(frame.get("expanded_terms") or [])], limit=3)
    elif family == "reference_explainer":
        search_forms = build_ranked_forms(
            [*list(frame.get("expanded_terms") or []), query],
            limit=3,
            exclude_casefold={"guide", "reference", "overview", "설명", "정의"},
        )
    elif family == "temporal_update":
        search_forms = build_ranked_forms([query, *list(frame.get("expanded_terms") or [])], limit=3)
    else:
        search_forms = [clean_text(query)] if clean_text(query) else []
    resolved_doc_ids = [
        item
        for item in resolved_ids
        if item.startswith("web_") or item == clean_text(effective_metadata_filter.get("document_id"))
    ]
    return {
        "effective_metadata_filter": effective_metadata_filter,
        "family": family,
        "media_platform": media_platform,
        "resolved_source_ids": resolved_ids,
        "resolved_urls": [item for item in resolved_ids if item.startswith("http://") or item.startswith("https://")],
        "resolved_doc_ids": resolved_doc_ids,
        "search_forms": search_forms,
        "document_scope": resolved_doc_ids
        or [clean_text(effective_metadata_filter.get("document_id")) for _ in [0] if clean_text(effective_metadata_filter.get("document_id"))],
        "prefer_materialized_fallback": family in {"reference_explainer", "temporal_update", "relation_explainer"},
    }


def build_vault_selection_inputs(
    *,
    query: str,
    frame_payload: dict[str, Any] | None,
    query_plan_payload: dict[str, Any] | None,
    metadata_filter: dict[str, Any] | None,
) -> dict[str, Any]:
    frame = dict(frame_payload or {})
    query_plan = dict(query_plan_payload or {})
    effective_metadata_filter = dict(frame.get("metadata_filter") or metadata_filter or {})
    family = clean_text(frame.get("family") or "")
    resolved_ids = resolved_source_ids(frame, query_plan)
    resolved_note_ids = [
        item
        for item in resolved_ids
        if item.startswith("vault:") or item == clean_text(effective_metadata_filter.get("note_id"))
    ]
    resolved_file_paths = [item for item in resolved_ids if item.endswith(".md") or item.endswith(".markdown")]
    if family == "vault_explainer":
        search_forms = build_ranked_forms(
            [*list(frame.get("expanded_terms") or []), query],
            limit=3,
            exclude_casefold={"overview", "summary", "설명", "정의", "요약"},
        )
    else:
        search_forms = build_ranked_forms([query, *list(frame.get("expanded_terms") or [])], limit=3)
    return {
        "effective_metadata_filter": effective_metadata_filter,
        "family": family,
        "resolved_source_ids": resolved_ids,
        "resolved_note_ids": resolved_note_ids,
        "resolved_file_paths": resolved_file_paths,
        "scoped_note_id": clean_text(effective_metadata_filter.get("note_id")) or next(iter(resolved_note_ids), ""),
        "scoped_file_path": clean_text(effective_metadata_filter.get("file_path")) or next(iter(resolved_file_paths), ""),
        "search_forms": search_forms,
    }


def text_overlap(query: str, *parts: Any) -> float:
    haystack = " ".join(clean_text(part) for part in parts).casefold()
    if not haystack:
        return 0.0
    score = 0.0
    for term in query_terms(query):
        if term in haystack:
            score += 1.0
    joined = " ".join(query_terms(query))
    if joined and joined in haystack:
        score += 2.0
    return score


def slot_coverage(card: dict[str, Any]) -> dict[str, str]:
    return dict(card.get("slot_coverage") or {})


def stable_score(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def parse_timestamp(value: Any) -> datetime | None:
    token = clean_text(value)
    if not token:
        return None
    try:
        if token.endswith("Z"):
            return datetime.fromisoformat(token.replace("Z", "+00:00")).astimezone(timezone.utc)
        parsed = datetime.fromisoformat(token)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def has_newer_upstream(card_updated_at: Any, *upstream_updated_at: Any) -> bool:
    baseline = parse_timestamp(card_updated_at)
    if baseline is None:
        return False
    for candidate in upstream_updated_at:
        parsed = parse_timestamp(candidate)
        if parsed is not None and parsed > baseline:
            return True
    return False


def parse_note_metadata(row: dict[str, Any] | None) -> dict[str, Any]:
    raw = (row or {}).get("metadata")
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(raw or "{}")
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def snippet_hash(*parts: Any) -> str:
    payload = "||".join(clean_text(part) for part in parts if clean_text(part))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20] if payload else ""


def stable_anchor_id(card_id: str, role: str, label: str, section_path: str, excerpt: str) -> str:
    digest = hashlib.sha1(f"{card_id}|{role}|{label}|{section_path}|{excerpt}".encode("utf-8")).hexdigest()[:16]
    return f"anchor:{digest}"


def trim_text(value: Any, *, max_chars: int = 240) -> str:
    token = clean_text(value)
    if len(token) <= max_chars:
        return token
    return token[: max(0, max_chars - 3)].rstrip() + "..."


def classify_project_query_profile(query: str) -> dict[str, bool]:
    token = clean_text(query)
    return {
        "architecture": bool(_ARCH_QUERY_RE.search(token)),
        "debug": bool(_DEBUG_QUERY_RE.search(token)),
        "overview": bool(_OVERVIEW_QUERY_RE.search(token)),
    }


def detect_project_file_role(relative_path: str) -> str:
    lowered = clean_text(relative_path).casefold()
    if not lowered:
        return "module"
    if _TEST_PATH_RE.search(lowered):
        return "test"
    if _DOC_PATH_RE.search(lowered):
        return "docs"
    if _ENTRYPOINT_PATH_RE.search(lowered):
        return "entrypoint"
    if _CONFIG_PATH_RE.search(lowered):
        return "config"
    if _ADAPTER_PATH_RE.search(lowered):
        return "adapter"
    return "module"


def _snippet_lines(text: str, *, limit: int | None = None) -> list[str]:
    lines = [clean_text(line) for line in str(text or "").splitlines() if clean_text(line)]
    return lines[:limit] if limit is not None else lines


def _symbol_definition_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if _SYMBOL_LINE_RE.match(line)]


def _call_lines(lines: list[str]) -> list[str]:
    result: list[str] = []
    for line in lines:
        if _SYMBOL_LINE_RE.match(line) or _IMPORT_LINE_RE.match(line):
            continue
        if _CALL_RE.search(line):
            result.append(line)
    return result


def _collect_import_tokens(text: str) -> list[str]:
    imports: list[str] = []
    for groups in _IMPORT_RE.findall(text):
        token = next((clean_text(value) for value in groups if clean_text(value)), "")
        if not token:
            continue
        parts = [clean_text(part) for part in re.split(r"[,\s]+", token) if clean_text(part)]
        imports.extend(parts or [token])
    return clean_lines(imports, limit=10)


def _classify_import_token(token: str, *, relative_path: str) -> str:
    lowered = clean_text(token).casefold()
    if not lowered:
        return "external"
    if lowered.startswith(".") or lowered.startswith("/") or "/" in lowered:
        return "internal"
    first_segment = lowered.split(".")[0]
    rel_parts = {part.casefold() for part in relative_path.split("/") if clean_text(part)}
    if first_segment in _COMMON_INTERNAL_IMPORTS or first_segment in rel_parts:
        return "internal"
    return "external"


def _build_integration_boundary(imports: list[str], *, relative_path: str) -> str:
    internal = clean_lines([token for token in imports if _classify_import_token(token, relative_path=relative_path) == "internal"], limit=4)
    external = clean_lines([token for token in imports if _classify_import_token(token, relative_path=relative_path) == "external"], limit=4)
    parts: list[str] = []
    if internal:
        parts.append(f"internal: {', '.join(internal)}")
    if external:
        parts.append(f"external: {', '.join(external)}")
    return trim_text(" | ".join(parts), max_chars=260)


def extract_repo_structure(snippet: str, *, relative_path: str = "") -> dict[str, str]:
    text = str(snippet or "")
    lines = _snippet_lines(text)
    symbols = clean_lines(_PY_SYMBOL_RE.findall(text) + _JS_SYMBOL_RE.findall(text), limit=8)
    calls = clean_lines([name for name in _CALL_RE.findall(text) if name not in _CALL_SKIP], limit=10)
    imports = _collect_import_tokens(text)
    symbol_lines = _symbol_definition_lines(lines)
    call_lines = _call_lines(lines)
    module_excerpt = trim_text(" ".join(lines[:6]), max_chars=280)
    symbol_owner_core = trim_text(", ".join(symbols), max_chars=180)
    if symbols and calls:
        call_flow_core = trim_text(", ".join(f"{symbols[0]} -> {call}" for call in calls[:5]), max_chars=200)
    else:
        call_flow_core = trim_text(", ".join(calls), max_chars=180)
    integration_boundary_core = _build_integration_boundary(imports, relative_path=relative_path)
    return {
        "module_excerpt": module_excerpt,
        "symbol_owner_core": symbol_owner_core,
        "call_flow_core": call_flow_core,
        "integration_boundary_core": integration_boundary_core,
        "symbol_owner_excerpt": trim_text(" ".join(symbol_lines[:4]) or symbol_owner_core, max_chars=260),
        "call_flow_excerpt": trim_text(" ".join(call_lines[:4]) or call_flow_core, max_chars=260),
        "integration_boundary_excerpt": trim_text(" ".join([line for line in lines if _IMPORT_LINE_RE.match(line)][:4]) or integration_boundary_core, max_chars=260),
    }


def paper_scope_from_filter(metadata_filter: dict[str, Any] | None) -> str:
    scoped = dict(metadata_filter or {})
    return clean_text(scoped.get("arxiv_id") or scoped.get("paper_id"))


def paper_scope_from_query(query: str) -> str:
    match = _ARXIV_RE.search(str(query or ""))
    return clean_text(match.group(0) if match else "")


def web_scope_from_filter(metadata_filter: dict[str, Any] | None) -> str:
    scoped = dict(metadata_filter or {})
    return clean_text(scoped.get("canonical_url") or scoped.get("url") or scoped.get("source_url"))


def web_scope_from_query(query: str) -> str:
    match = _URL_RE.search(str(query or ""))
    return clean_text(match.group(0) if match else "")


def vault_scope_from_filter(metadata_filter: dict[str, Any] | None) -> str:
    return _domain_vault_scope_from_filter(metadata_filter)


def vault_scope_from_query(query: str) -> str:
    return _domain_vault_scope_from_query(query)


def repo_scope_from_filter(metadata_filter: dict[str, Any] | None) -> str:
    scoped = dict(metadata_filter or {})
    return clean_text(scoped.get("repo_path") or scoped.get("workspace_path"))


def source_kind(source_type: str | None, metadata_filter: dict[str, Any] | None = None) -> str:
    normalized = normalize_source_type(source_type)
    if normalized in {"project", "repo"}:
        return "project"
    if normalized in {"paper", "web", "vault"}:
        return normalized
    if paper_scope_from_filter(metadata_filter):
        return "paper"
    if web_scope_from_filter(metadata_filter):
        return "web"
    if vault_scope_from_filter(metadata_filter):
        return "vault"
    if repo_scope_from_filter(metadata_filter):
        return "project"
    return ""


def classify_intent(query: str, metadata_filter: dict[str, Any] | None = None) -> str:
    body = str(query or "")
    if paper_scope_from_filter(metadata_filter) or paper_scope_from_query(query):
        return "paper_lookup"
    temporal_match = bool(_TEMPORAL_RE.search(body))
    hard_temporal_match = bool(_HARD_TEMPORAL_RE.search(body))
    if temporal_match and not hard_temporal_match and _EVAL_RE.search(body):
        return "evaluation"
    if temporal_match:
        return "temporal"
    if _COMPARE_RE.search(body):
        return "comparison"
    if _RELATION_RE.search(body):
        return "relation"
    if _EVAL_RE.search(body):
        return "evaluation"
    if _IMPL_RE.search(body):
        return "implementation"
    if _DEFINITION_RE.search(body) or _EXPLAIN_RE.search(body):
        return "definition"
    return "paper_summary"


def route_mode(*, source_kind_value: str, intent: str, matched_entities: list[dict[str, Any]], query: str) -> str:
    if source_kind_value in {"vault", "project"}:
        return "card-first"
    short_query = len(query_terms(query)) <= 3
    ambiguous = short_query and len(matched_entities) > 1
    if intent in {"definition", "comparison", "relation"} or ambiguous:
        return "ontology-first"
    return "card-first"


def should_attempt_claim_cards(
    *,
    route: "AskV2Route",
    section_first_requested: bool,
    section_allowed: bool,
    has_section_anchors: bool,
) -> bool:
    if route.intent in {"comparison", "evaluation"}:
        return True
    if route.source_kind == "paper" and not section_first_requested:
        return True
    if has_section_anchors:
        return False
    if route.source_kind == "project":
        return True
    if route.source_kind == "web" and route.intent == "relation":
        return True
    if section_first_requested and not section_allowed:
        return True
    return False


@dataclass
class AskV2Route:
    source_kind: str
    intent: str
    mode: str
    matched_entities: list[dict[str, Any]]
    entity_ids: list[str]


def build_project_cards(*, workspace_files: list[dict[str, Any]], repo_path: str) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for item in workspace_files:
        relative_path = clean_text(item.get("relative_path"))
        raw_snippet = str(item.get("snippet") or "")
        snippet = clean_text(raw_snippet)
        if not relative_path or not snippet:
            continue
        card_id = f"repo-card-v2:{clean_text(item.get('path'))}"
        file_role_core = detect_project_file_role(relative_path)
        structure = extract_repo_structure(raw_snippet, relative_path=relative_path)
        module_core = structure.get("module_excerpt") or trim_text(snippet, max_chars=280)
        symbol_owner_core = structure.get("symbol_owner_core") or ""
        call_flow_core = structure.get("call_flow_core") or ""
        integration_boundary_core = clean_text(item.get("reason")) or structure.get("integration_boundary_core") or ""
        weak_slots: list[str] = []
        anchors: list[dict[str, Any]] = []
        for role, excerpt in [
            ("module", structure.get("module_excerpt") or module_core),
            ("symbol_owner", structure.get("symbol_owner_excerpt") or symbol_owner_core),
            ("call_flow", structure.get("call_flow_excerpt") or call_flow_core),
            ("integration_boundary", structure.get("integration_boundary_excerpt") or integration_boundary_core),
        ]:
            token = clean_text(excerpt)
            if not token:
                weak_slots.append(role)
                continue
            anchors.append(
                {
                    "anchor_id": stable_anchor_id(card_id, role, relative_path, f"{relative_path}::{role}", token),
                    "card_id": card_id,
                    "claim_id": "",
                    "document_id": clean_text(item.get("path")),
                    "unit_id": relative_path,
                    "title": relative_path,
                    "source_type": "project",
                    "section_path": f"{relative_path}::{role}",
                    "span_locator": f"{relative_path}::{role}",
                    "snippet_hash": snippet_hash(relative_path, role, token),
                    "evidence_role": role,
                    "excerpt": token,
                    "score": 0.93 if role == "module" else 0.86,
                    "file_path": clean_text(item.get("path")),
                }
            )
        slot_coverage = {
            "moduleCore": "complete",
            "symbolOwnerCore": "complete" if symbol_owner_core else "missing",
            "callFlowCore": "complete" if call_flow_core else "missing",
            "integrationBoundaryCore": "complete" if integration_boundary_core else "missing",
            "fileRoleCore": "complete" if file_role_core else "missing",
        }
        signal_count = sum(1 for key, value in slot_coverage.items() if key != "moduleCore" and value == "complete")
        quality_flag = "ok" if signal_count >= 2 or (file_role_core in {"entrypoint", "adapter", "config"} and signal_count >= 1) else "needs_review"
        cards.append(
            {
                "card_id": card_id,
                "source_kind": "project",
                "repo_path": repo_path,
                "relative_path": relative_path,
                "title": relative_path,
                "module_core": module_core,
                "symbol_owner_core": symbol_owner_core,
                "call_flow_core": call_flow_core,
                "integration_boundary_core": integration_boundary_core,
                "file_role_core": file_role_core,
                "search_text": clean_text(
                    f"{relative_path} {item.get('reason')} {file_role_core} {module_core} {symbol_owner_core} {call_flow_core} {integration_boundary_core}"
                ),
                "quality_flag": quality_flag,
                "slot_coverage": slot_coverage,
                "diagnostics": {
                    "fileRole": file_role_core,
                    "signalCount": signal_count,
                    "weakSlots": sorted(set(weak_slots + [key for key, value in slot_coverage.items() if value != "complete"])),
                },
                "anchors": anchors,
            }
        )
    return cards


def collect_project_fallback_files(
    *,
    repo_path: str,
    query: str,
    existing_relative_paths: set[str],
    limit: int = 8,
    max_excerpt_chars: int = 1200,
) -> list[dict[str, Any]]:
    root = Path(repo_path).expanduser().resolve()
    if not root.exists():
        return []
    selected: list[tuple[float, str, Path, str]] = []
    profile = classify_project_query_profile(query)
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name
            for name in dirnames
            if name.lower() not in _PROJECT_EXCLUDED_DIRS
            and not any(part.lower() in _PROJECT_EXCLUDED_PARTS for part in (Path(current_root) / name).parts)
        ]
        base = Path(current_root)
        for filename in filenames:
            path = base / filename
            if path.suffix.lower() not in _PROJECT_ALLOWED_SUFFIXES:
                continue
            rel_path = path.relative_to(root).as_posix()
            if rel_path in existing_relative_paths:
                continue
            if any(part.lower() in _PROJECT_EXCLUDED_PARTS for part in path.parts):
                continue
            snippet = snippet_for_path(path, max_chars=max_excerpt_chars)
            if not clean_text(snippet):
                continue
            role = detect_project_file_role(rel_path)
            overlap = text_overlap(query, rel_path, snippet)
            if profile.get("architecture") and role in {"entrypoint", "adapter", "module", "config"}:
                overlap += 1.0
            if profile.get("debug") and role == "test":
                overlap += 1.0
            if profile.get("overview") and role == "docs":
                overlap += 0.75
            if overlap <= 0:
                continue
            selected.append((overlap, rel_path, path, snippet))
    selected.sort(key=lambda item: (-item[0], item[1]))
    return [
        {
            "path": str(path),
            "relative_path": rel_path,
            "role": "workspace_file",
            "reason": f"project fallback overlap: {score:.2f}",
            "snippet": snippet,
            "source_type": "project",
            "normalized_source_type": "project",
            "ephemeral": True,
            "title": rel_path,
        }
        for score, rel_path, path, snippet in selected[: max(1, int(limit))]
    ]


__all__ = [
    "AskV2Route",
    "_NUMBER_RE",
    "_TEMPORAL_RE",
    "build_project_cards",
    "classify_intent",
    "classify_project_query_profile",
    "clean_text",
    "clean_lines",
    "collect_project_fallback_files",
    "detect_project_file_role",
    "has_newer_upstream",
    "paper_scope_from_filter",
    "paper_scope_from_query",
    "parse_note_metadata",
    "query_terms",
    "repo_scope_from_filter",
    "route_mode",
    "slot_coverage",
    "snippet_hash",
    "source_kind",
    "stable_anchor_id",
    "stable_score",
    "text_overlap",
    "vault_scope_from_query",
    "web_scope_from_filter",
    "web_scope_from_query",
]
