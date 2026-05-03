from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from knowledge_hub.document_memory import DocumentMemoryBuilder
from knowledge_hub.document_memory.extraction import DocumentMemorySchemaExtractor
from knowledge_hub.papers.memory_runtime import build_paper_memory_builder
from knowledge_hub.learning.task_router import get_llm_for_task
from knowledge_hub.vault.card_v2_builder import VaultCardV2Builder
from knowledge_hub.vault.concepts import find_concept_note_path, iter_concept_note_paths, normalize_concept_wikilink_target
from knowledge_hub.vault.parser import ObsidianParser

DEFAULT_CORE_MATH_CONCEPTS = [
    "Vector",
    "Matrix",
    "Inner Product",
    "Gradient",
    "Hessian",
    "Expectation",
    "Conditional Probability",
    "Bayes Theorem",
    "Cross-Entropy",
    "KL Divergence",
    "Convex Set and Convex Function",
    "Gradient Descent and SGD",
]

_TITLE_SEARCH_MATCH_THRESHOLD = 30.0
_TITLE_TABLE_FALLBACK_THRESHOLD = 60.0

_TITLE_TOKEN_EXPANSIONS = {
    "rl": "reinforcement learning",
    "cnn": "convolutional neural network",
    "cnns": "convolutional neural networks",
}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in items:
        token = _clean_text(raw)
        lowered = token.casefold()
        if not token or lowered in seen:
            continue
        seen.add(lowered)
        result.append(token)
    return result


def _resolve_vault_root(config: Any) -> Path:
    token = _clean_text(getattr(config, "vault_path", "") or "")
    if not token:
        raise ValueError("vault_path is not configured")
    root = Path(token).expanduser().resolve()
    if not root.exists():
        raise ValueError(f"vault_path does not exist: {root}")
    return root


def _resolve_concepts_dir(config: Any) -> Path:
    vault_root = _resolve_vault_root(config)
    configured = _clean_text(getattr(config, "obsidian_concepts_folder", "") or "")
    if configured:
        candidate = (vault_root / configured).resolve()
        if candidate.exists():
            return candidate
    candidates = [
        vault_root / "AI" / "AI_Papers" / "Concepts",
        vault_root / "Papers" / "Concepts",
        vault_root / "Projects" / "AI" / "AI_Papers" / "Concepts",
        vault_root / "Concepts",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ValueError(f"concepts directory not found under vault: {vault_root}")


def _resolve_math_concepts_dir(config: Any) -> Path:
    concepts_dir = _resolve_concepts_dir(config)
    if concepts_dir.name == "Math":
        return concepts_dir
    math_dir = concepts_dir / "Math"
    if math_dir.exists():
        return math_dir
    raise ValueError(f"math concepts directory not found under concepts root: {concepts_dir}")


def _resolve_papers_dir(config: Any) -> Path:
    vault_root = _resolve_vault_root(config)
    candidates = [
        vault_root / "AI" / "AI_Papers" / "Papers",
        vault_root / "Papers",
        vault_root / "Projects" / "AI" / "AI_Papers" / "Papers",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ValueError(f"papers directory not found under vault: {vault_root}")


def _normalize_link_key(value: Any) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    token = token.split("|", 1)[0].split("#", 1)[0].split("^", 1)[0].strip()
    token = token.replace("\\", "/").strip().strip("/")
    if token.lower().endswith(".md"):
        token = token[:-3]
    return token.casefold()


def _link_lookup(sqlite_db, note_id: str, file_path: str, title: str) -> dict[str, str]:
    mapping: dict[str, set[str]] = {}

    def register(key: str, target_note_id: str) -> None:
        token = _normalize_link_key(key)
        if not token or not target_note_id:
            return
        mapping.setdefault(token, set()).add(target_note_id)

    for row in list(sqlite_db.list_notes(source_type="vault", limit=1_000_000) or []):
        target_id = _clean_text(row.get("id"))
        target_path = _clean_text(row.get("file_path"))
        target_title = _clean_text(row.get("title"))
        if not target_id:
            continue
        register(target_id, target_id)
        register(target_path, target_id)
        register(Path(target_path).stem, target_id)
        register(target_title, target_id)

    register(note_id, note_id)
    register(file_path, note_id)
    register(Path(file_path).stem, note_id)
    register(title, note_id)

    return {
        key: next(iter(values))
        for key, values in mapping.items()
        if len(values) == 1
    }


def _resolve_links(links: list[str], lookup: dict[str, str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in list(links or []):
        token = lookup.get(_normalize_link_key(raw), "")
        if not token or token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result


def _bridge_note_paths(papers_dir: Path) -> list[Path]:
    root = Path(papers_dir)
    if not root.exists():
        return []
    return sorted(root.glob("Math Bridge - *.md"), key=lambda path: path.name.casefold())


def _parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    body = str(content or "")
    if not body.startswith("---\n"):
        return {}, body
    end = body.find("\n---\n", 4)
    if end < 0:
        return {}, body
    raw = body[4:end]
    fields: dict[str, str] = {}
    for line in raw.splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        fields[_clean_text(key)] = str(value or "").strip()
    return fields, body[end + 5 :]


def _extract_note_concepts(content: str) -> list[str]:
    body = str(content or "")
    sections: list[str] = []
    for heading in ("내가 배워야 할 개념", "관련 개념"):
        marker = body.find(heading)
        if marker < 0:
            continue
        section = body[marker:]
        next_heading = re.search(r"\n#{1,3}\s+", section[1:])
        if next_heading:
            section = section[: next_heading.start() + 1]
        sections.append(section)

    concepts: list[str] = []
    seen: set[str] = set()
    for section in sections:
        for raw in re.findall(r"\[\[([^\]]+)\]\]", section):
            name = normalize_concept_wikilink_target(raw)
            lowered = name.casefold()
            if (
                not name
                or lowered == "00_concept_index"
                or lowered == "00_concept_index.md"
            ):
                continue
            if lowered in seen:
                continue
            seen.add(lowered)
            concepts.append(name)
    return concepts


def _extract_primary_paper_link(content: str) -> str:
    body = str(content or "")
    marker = body.find("## 논문")
    if marker < 0:
        return ""
    section = body[marker:]
    next_heading = re.search(r"\n#{1,3}\s+", section[1:])
    if next_heading:
        section = section[: next_heading.start() + 1]
    for raw in re.findall(r"\[\[([^\]]+)\]\]", section):
        token = str(raw or "").split("|", 1)[0].split("#", 1)[0].strip()
        if token:
            return Path(token).stem
    return ""


def _default_concept_entity_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(name or "").strip().lower()).strip("_")
    return f"concept_{slug}" if slug else "concept_math"


def _normalize_title_key(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()
    if not normalized:
        return ""
    tokens: list[str] = []
    for token in normalized.split():
        replacement = _TITLE_TOKEN_EXPANSIONS.get(token)
        if replacement:
            tokens.extend(replacement.split())
            continue
        tokens.append(token)
    return " ".join(tokens).strip()


def _title_query_variants(title: str) -> list[str]:
    token = _clean_text(title)
    if not token:
        return []
    variants = [token]
    if ": " in token:
        variants.append(token.replace(": ", " "))
    if " A Survey" in token and ": A Survey" not in token:
        variants.append(token.replace(" A Survey", ": A Survey"))
    first, sep, rest = token.partition(" ")
    if sep and ":" not in first:
        variants.append(f"{first}: {rest}")
    return _dedupe(variants)


def _title_match_score(query: str, candidate: str) -> float:
    query_clean = _normalize_title_key(query)
    candidate_clean = _normalize_title_key(candidate)
    if not query_clean or not candidate_clean:
        return 0.0
    if query_clean == candidate_clean:
        return 100.0
    score = 0.0
    if query_clean in candidate_clean or candidate_clean in query_clean:
        score += 35.0
    query_tokens = {token for token in query_clean.split() if token}
    candidate_tokens = {token for token in candidate_clean.split() if token}
    if query_tokens and candidate_tokens:
        overlap = len(query_tokens & candidate_tokens) / max(1, len(query_tokens))
        score += overlap * 50.0
    if candidate.startswith(query.split(" ", 1)[0]):
        score += 10.0
    return score


def _best_title_match_from_rows(
    rows: list[dict[str, Any]],
    *,
    title_candidates: list[str],
) -> tuple[float, dict[str, Any]]:
    best_score = 0.0
    best_match: dict[str, Any] = {}
    for row in rows:
        candidate_title = _clean_text(row.get("title"))
        if not candidate_title:
            continue
        for title in title_candidates:
            score = _title_match_score(title, candidate_title)
            if score <= best_score:
                continue
            best_score = score
            best_match = dict(row)
    return best_score, best_match


class MathConceptMemoryService:
    def __init__(self, sqlite_db, *, config: Any):
        self.sqlite_db = sqlite_db
        self.config = config

    def math_dir(self) -> Path:
        return _resolve_math_concepts_dir(self.config)

    def _math_prefix(self) -> str:
        vault_root = _resolve_vault_root(self.config)
        return self.math_dir().relative_to(vault_root).as_posix().rstrip("/") + "/"

    def papers_dir(self) -> Path:
        return _resolve_papers_dir(self.config)

    def list_math_note_paths(self) -> list[Path]:
        return iter_concept_note_paths(self.math_dir())

    def resolve_target_paths(
        self,
        *,
        concepts: list[str] | None = None,
        all_math: bool = False,
    ) -> tuple[list[Path], list[str]]:
        math_dir = self.math_dir()
        requested = _dedupe(list(concepts or []))
        if all_math:
            return self.list_math_note_paths(), []
        if not requested:
            requested = list(DEFAULT_CORE_MATH_CONCEPTS)

        resolved: list[Path] = []
        missing: list[str] = []
        for concept in requested:
            path = find_concept_note_path(math_dir, concept)
            if path is None:
                missing.append(concept)
                continue
            resolved.append(path)
        unique_paths: list[Path] = []
        seen: set[str] = set()
        for path in resolved:
            token = str(path.resolve())
            if token in seen:
                continue
            seen.add(token)
            unique_paths.append(path)
        return unique_paths, missing

    def ensure_note_registered(self, note_path: Path) -> dict[str, Any]:
        vault_root = _resolve_vault_root(self.config)
        path = Path(note_path).expanduser().resolve()
        if vault_root not in [path, *path.parents]:
            raise ValueError(f"note is outside configured vault: {path}")
        parser = ObsidianParser(str(vault_root), exclude_folders=list(getattr(self.config, "vault_excludes", []) or []))
        document = parser.parse_file(path)
        if document is None:
            raise ValueError(f"note content is empty: {path}")
        note_id = _clean_text(document.file_path)
        self.sqlite_db.upsert_note(
            note_id=note_id,
            title=_clean_text(document.title) or path.stem,
            content=str(document.content or ""),
            file_path=note_id,
            source_type=str(document.source_type.value),
            metadata=dict(document.metadata or {}),
        )
        self.sqlite_db.replace_note_tags(note_id, list(document.tags or []))
        lookup = _link_lookup(self.sqlite_db, note_id, note_id, document.title)
        resolved_links = _resolve_links(list(document.links or []), lookup)
        self.sqlite_db.replace_links_for_source(note_id, resolved_links, "wiki_link")
        return {
            "note_id": note_id,
            "title": _clean_text(document.title) or path.stem,
            "file_path": note_id,
        }

    def _document_memory_builder(
        self,
        *,
        allow_external: bool,
        llm_mode: str,
        note_paths: list[Path],
    ) -> tuple[DocumentMemoryBuilder, dict[str, Any], list[str]]:
        context = "\n".join(
            path.relative_to(_resolve_vault_root(self.config)).as_posix()
            for path in note_paths[:24]
        )
        llm, decision, warnings = get_llm_for_task(
            self.config,
            task_type="materialization_concept_enrichment",
            allow_external=bool(allow_external),
            query="Build math concept document memory cards from vault concept notes.",
            context=context,
            source_count=len(note_paths),
            force_route=str(llm_mode or "auto"),
        )
        if llm is None:
            return DocumentMemoryBuilder(self.sqlite_db, config=self.config), decision.to_dict(), list(warnings)
        extractor = DocumentMemorySchemaExtractor(llm, model=decision.model)
        return (
            DocumentMemoryBuilder(
                self.sqlite_db,
                config=self.config,
                schema_extractor=extractor,
                extraction_mode="schema",
            ),
            decision.to_dict(),
            list(warnings),
        )

    def build(
        self,
        *,
        concepts: list[str] | None = None,
        all_math: bool = False,
        allow_external: bool = False,
        llm_mode: str = "auto",
    ) -> dict[str, Any]:
        note_paths, missing = self.resolve_target_paths(concepts=concepts, all_math=all_math)
        builder, route, warnings = self._document_memory_builder(
            allow_external=allow_external,
            llm_mode=llm_mode,
            note_paths=note_paths,
        )
        card_builder = VaultCardV2Builder(self.sqlite_db)
        items: list[dict[str, Any]] = []
        for note_path in note_paths:
            registered = self.ensure_note_registered(note_path)
            note_id = registered["note_id"]
            builder.build_and_store_note(note_id=note_id)
            diagnostics = builder.get_last_extraction_diagnostics(note_id)
            card = card_builder.build_and_store(note_id=note_id)
            items.append(
                {
                    "concept": Path(note_path).stem,
                    "noteId": note_id,
                    "title": _clean_text(card.get("title") or registered["title"]),
                    "filePath": _clean_text(card.get("file_path") or registered["file_path"]),
                    "cardId": _clean_text(card.get("card_id")),
                    "qualityFlag": _clean_text(card.get("quality_flag") or "unscored"),
                    "claimRefCount": len(list(card.get("claim_refs") or [])),
                    "anchorCount": len(list(card.get("anchors") or [])),
                    "claimCardCount": len(list(card.get("claim_cards") or [])),
                    "documentMemory": {
                        "mode": _clean_text(diagnostics.get("mode") or "deterministic"),
                        "attempted": bool(diagnostics.get("attempted")),
                        "applied": bool(diagnostics.get("applied")),
                        "fallbackUsed": bool(diagnostics.get("fallbackUsed")),
                        "extractorModel": _clean_text(diagnostics.get("extractorModel")),
                        "coverageStatus": _clean_text(diagnostics.get("coverageStatus") or ""),
                        "warnings": list(diagnostics.get("warnings") or []),
                    },
                }
            )
        return {
            "status": "ok" if not missing else "partial",
            "count": len(items),
            "items": items,
            "missingConcepts": missing,
            "route": route,
            "warnings": warnings,
            "defaultCoreConcepts": list(DEFAULT_CORE_MATH_CONCEPTS),
        }

    def _math_note_ids(self) -> list[str]:
        prefix = self._math_prefix()
        rows = self.sqlite_db.conn.execute(
            """
            SELECT id
            FROM notes
            WHERE source_type = 'vault' AND file_path LIKE ?
            ORDER BY updated_at DESC
            """,
            (f"{prefix}%",),
        ).fetchall()
        return [_clean_text(row["id"]) for row in rows if _clean_text(row["id"])]

    def show(self, *, concept: str) -> dict[str, Any] | None:
        note_paths, missing = self.resolve_target_paths(concepts=[concept], all_math=False)
        if missing or not note_paths:
            return None
        note_id = note_paths[0].relative_to(_resolve_vault_root(self.config)).as_posix()
        card = self.sqlite_db.get_vault_card_v2(note_id)
        if not card:
            return None
        return {
            **dict(card),
            "claim_refs": self.sqlite_db.list_vault_card_claim_refs_v2(card_id=str(card.get("card_id") or "")),
            "anchors": self.sqlite_db.list_vault_evidence_anchors_v2(card_id=str(card.get("card_id") or "")),
        }

    def search(self, *, query: str, limit: int = 10) -> list[dict[str, Any]]:
        note_ids = self._math_note_ids()
        if not note_ids:
            return []
        return list(self.sqlite_db.search_vault_cards_v2(str(query or "").strip(), limit=max(1, int(limit)), note_ids=note_ids) or [])

    def list_bridge_note_paths(self) -> list[Path]:
        return _bridge_note_paths(self.papers_dir())

    def _resolve_bridge_paper(self, *, frontmatter: dict[str, str], body: str, note_path: Path) -> dict[str, Any]:
        explicit_id = _clean_text(frontmatter.get("paper_id"))
        if explicit_id:
            paper = self.sqlite_db.get_paper(explicit_id) or {}
            if paper:
                return dict(paper)

        title_candidates = _dedupe(
            [
                _clean_text(frontmatter.get("paper_title")),
                _clean_text(_extract_primary_paper_link(body)),
                _clean_text(note_path.stem.removeprefix("Math Bridge - ")),
            ]
        )
        if not title_candidates:
            return {}

        for title in title_candidates:
            row = self.sqlite_db.conn.execute(
                """
                SELECT *
                FROM papers
                WHERE lower(trim(title)) = ?
                ORDER BY importance DESC, year DESC
                LIMIT 1
                """,
                (title.casefold(),),
            ).fetchone()
            if row:
                return dict(row)

        best_score = 0.0
        best_match: dict[str, Any] = {}
        for title in title_candidates:
            for query in _title_query_variants(title):
                score, match = _best_title_match_from_rows(
                    list(self.sqlite_db.search_papers(query, limit=10) or []),
                    title_candidates=[title],
                )
                if score > best_score:
                    best_score = score
                    best_match = match
        if best_score >= _TITLE_SEARCH_MATCH_THRESHOLD:
            return best_match

        rows = self.sqlite_db.conn.execute(
            """
            SELECT *
            FROM papers
            ORDER BY importance DESC, year DESC, title ASC
            """
        ).fetchall()
        best_score, best_match = _best_title_match_from_rows(
            [dict(row) for row in rows],
            title_candidates=title_candidates,
        )
        if best_score >= _TITLE_TABLE_FALLBACK_THRESHOLD:
            return best_match
        return {}

    def _ensure_concept_entity(self, concept_name: str) -> tuple[str, str]:
        token = _clean_text(concept_name)
        if not token:
            return "", ""
        resolver = getattr(self.sqlite_db, "resolve_entity", None)
        if callable(resolver):
            resolved = resolver(token, entity_type="concept") or {}
            entity_id = _clean_text(resolved.get("entity_id"))
            canonical_name = _clean_text(resolved.get("canonical_name"))
            if entity_id:
                return entity_id, canonical_name or token
        entity_id = _default_concept_entity_id(token)
        self.sqlite_db.upsert_ontology_entity(
            entity_id=entity_id,
            entity_type="concept",
            canonical_name=token,
            description="",
            properties={"domains": ["math"], "source_surface": "vault_math_bridge"},
            confidence=1.0,
            source="vault_math_bridge_sync",
        )
        return entity_id, token

    def sync_bridge_papers(
        self,
        *,
        apply: bool = False,
        rebuild_paper_memory: bool = False,
        ensure_math_cards: bool = True,
        allow_external: bool = False,
        llm_mode: str = "auto",
    ) -> dict[str, Any]:
        vault_root = _resolve_vault_root(self.config)
        bridge_paths = self.list_bridge_note_paths()
        scanned: list[dict[str, Any]] = []
        requested_concepts: list[str] = []

        for note_path in bridge_paths:
            content = note_path.read_text(encoding="utf-8")
            frontmatter, body = _parse_frontmatter(content)
            concepts = _extract_note_concepts(body)
            requested_concepts.extend(concepts)
            paper = self._resolve_bridge_paper(frontmatter=frontmatter, body=body, note_path=note_path)
            relative_path = note_path.relative_to(vault_root).as_posix()
            scanned.append(
                {
                    "bridgeNote": relative_path,
                    "bridgePaperId": _clean_text(frontmatter.get("paper_id")),
                    "paperTitle": _clean_text(frontmatter.get("paper_title")) or _clean_text(_extract_primary_paper_link(body)),
                    "resolvedPaperId": _clean_text(paper.get("arxiv_id")),
                    "resolvedPaperTitle": _clean_text(paper.get("title")),
                    "concepts": list(concepts),
                    "status": "ready" if paper and concepts else ("missing_paper" if not paper else "missing_concepts"),
                }
            )

        unique_concepts = _dedupe(requested_concepts)
        math_card_payload: dict[str, Any] | None = None
        if apply and ensure_math_cards and unique_concepts:
            math_card_payload = self.build(
                concepts=unique_concepts,
                all_math=False,
                allow_external=bool(allow_external),
                llm_mode=str(llm_mode or "auto"),
            )

        applied_relations = 0
        rebuilt_cards: list[dict[str, str]] = []
        for item in scanned:
            paper_id = _clean_text(item.get("resolvedPaperId"))
            bridge_note = _clean_text(item.get("bridgeNote"))
            concept_names = list(item.get("concepts") or [])
            if not apply or not paper_id or not concept_names:
                continue

            for concept_name in concept_names:
                concept_id, canonical_name = self._ensure_concept_entity(concept_name)
                if not concept_id:
                    continue
                evidence = json.dumps(
                    {
                        "note_id": bridge_note,
                        "paper_title": _clean_text(item.get("resolvedPaperTitle") or item.get("paperTitle")),
                        "concept_name": canonical_name,
                    },
                    ensure_ascii=False,
                )
                self.sqlite_db.add_relation(
                    "paper",
                    paper_id,
                    "uses",
                    "concept",
                    concept_id,
                    evidence_text=evidence,
                    confidence=0.92,
                )
                applied_relations += 1

            if rebuild_paper_memory:
                builder = build_paper_memory_builder(
                    self.sqlite_db,
                    config=self.config,
                    allow_external=bool(allow_external),
                    llm_mode=str(llm_mode or "auto"),
                    query=_clean_text(item.get("resolvedPaperTitle") or item.get("paperTitle")),
                    context="\n".join(concept_names[:12]),
                    source_count=len(concept_names),
                )
                builder.build_and_store(paper_id=paper_id)
                rebuilt_cards.append({"paperId": paper_id, "title": _clean_text(item.get("resolvedPaperTitle") or item.get("paperTitle"))})

        matched = [item for item in scanned if item.get("resolvedPaperId")]
        missing_papers = [item for item in scanned if item.get("status") == "missing_paper"]
        missing_concepts = [item for item in scanned if item.get("status") == "missing_concepts"]
        return {
            "status": "ok" if not missing_papers and not missing_concepts else "partial",
            "apply": bool(apply),
            "bridgeCount": len(scanned),
            "matchedPaperCount": len(matched),
            "missingPaperCount": len(missing_papers),
            "missingConceptCount": len(missing_concepts),
            "uniqueConceptCount": len(unique_concepts),
            "relationCount": applied_relations,
            "rebuiltPaperMemoryCount": len(rebuilt_cards),
            "mathCardBuild": math_card_payload or {},
            "items": scanned,
            "rebuiltPaperMemory": rebuilt_cards,
        }


__all__ = ["DEFAULT_CORE_MATH_CONCEPTS", "MathConceptMemoryService"]
