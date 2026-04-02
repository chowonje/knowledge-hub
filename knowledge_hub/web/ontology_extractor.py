"""Web-to-ontology extraction pipeline (rule-first, optional external refinement)."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from knowledge_hub.infrastructure.config import Config
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.core.keywords import extract_keywords_from_text
from knowledge_hub.core.sanitizer import detect_p0, redact_payload
from knowledge_hub.core.schema_validator import annotate_schema_errors
from knowledge_hub.learning.resolver import EntityResolver, normalize_term
from knowledge_hub.knowledge.ai_taxonomy import classify_ai_concept, merge_ai_classification_properties
from knowledge_hub.papers.claim_extractor import estimate_evidence_quality, score_claim_with_breakdown
from knowledge_hub.web.claim_extractor import extract_web_claim_candidates

log = logging.getLogger("khub.web.ontology_extractor")

RELATION_ENUM = {
    "causes",
    "enables",
    "part_of",
    "contrasts",
    "example_of",
    "requires",
    "improves",
    "related_to",
    "unknown_relation",
}

RELATION_PATTERNS: list[tuple[re.Pattern[str], str, float]] = [
    (re.compile(r"\b(cause|causes|caused|because|leads to|trigger)\b|원인|유발"), "causes", 0.95),
    (re.compile(r"\b(enable|enables|allow|allows|facilitate)\b|가능하게|지원"), "enables", 0.9),
    (re.compile(r"\b(part of|component of|consist of|composed of)\b|구성|일부"), "part_of", 0.9),
    (re.compile(r"\b(unlike|whereas|however|in contrast)\b|반면|대조"), "contrasts", 0.9),
    (re.compile(r"\b(such as|for example|e g)\b|예를 들어|예시"), "example_of", 0.85),
    (re.compile(r"\b(require|requires|need|needs|prerequisite)\b|필요|요구"), "requires", 0.9),
    (re.compile(r"\b(improve|improves|improved|boost|enhance)\b|향상|개선"), "improves", 0.9),
]


def _slugify(value: str) -> str:
    token = normalize_term(value).replace(" ", "-")
    token = re.sub(r"-+", "-", token).strip("-")
    return token or "untitled-topic"


def _snippet_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _new_concept_id(candidate: str) -> str:
    digest = hashlib.sha1(normalize_term(candidate).encode("utf-8")).hexdigest()[:12]
    return f"webc_{digest}"


def _new_claim_id(
    note_id: str,
    subject_entity_id: str,
    predicate: str,
    object_entity_id: str = "",
    object_literal: str = "",
    claim_text: str = "",
) -> str:
    digest = hashlib.sha1(
        f"{note_id}|{subject_entity_id}|{predicate}|{object_entity_id}|{object_literal}|{claim_text}".encode("utf-8")
    ).hexdigest()[:16]
    return f"claim:web:{digest}"


def _split_sentences(text: str) -> list[str]:
    lines = re.split(r"[.!?\n]+", text or "")
    return [line.strip() for line in lines if line and line.strip()]


def _frequency_score(normalized_text: str, candidate: str) -> float:
    token = normalize_term(candidate)
    if not token:
        return 0.0
    raw_hits = normalized_text.count(token)
    length_boost = min(0.2, len(token) / 40.0)
    return max(0.0, min(1.0, min(1.0, raw_hits / 4.0) + length_boost))


def _cooccur_score(mentions: int) -> float:
    return max(0.0, min(1.0, 0.25 + min(mentions, 4) * 0.18))


def _relation_from_sentence(sentence: str) -> tuple[str, float]:
    normalized = normalize_term(sentence)
    if not normalized:
        return "unknown_relation", 0.0
    for pattern, relation, score in RELATION_PATTERNS:
        if pattern.search(normalized):
            return relation, score
    return "related_to", 0.6


@dataclass
class _EntityCandidate:
    """통합 엔티티 후보 (concept, person, organization 등)"""
    note_id: str
    source_url: str
    original_term: str
    canonical_id: str
    display_name: str
    entity_type: str = "concept"  # concept, person, organization, event
    aliases: list[str] = field(default_factory=list)
    confidence: float = 0.0
    resolve_confidence: float = 0.0
    resolve_method: str = ""
    evidence_ptrs: list[dict[str, str]] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class _RelationCandidate:
    note_id: str
    source_url: str
    source_canonical_id: str
    relation_norm: str
    target_canonical_id: str
    confidence: float
    evidence_ptrs: list[dict[str, str]]
    reason: dict[str, Any]


class WebOntologyExtractor:
    def __init__(self, db: SQLiteDatabase, config: Config | None = None):
        self.db = db
        self.config = config
        self.resolver = EntityResolver(db)

    def _is_predicate_approved(self, predicate_id: str) -> bool:
        token = str(predicate_id or "").strip()
        if not token:
            return False
        row = self.db.get_predicate(token)
        if not row:
            return False
        return str(row.get("status", "")).strip() in {"core", "approved_ext"}

    def _relation_endpoint_context(
        self,
        *,
        source_canonical_id: str,
        target_canonical_id: str,
        reason: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        payload = dict(reason or {})
        source_type = str(payload.get("source_type", "concept") or "concept").strip().lower() or "concept"
        target_type = str(payload.get("target_type", "concept") or "concept").strip().lower() or "concept"
        source_id = str(payload.get("source_id", source_canonical_id) or source_canonical_id).strip()
        target_id = str(payload.get("target_id", target_canonical_id) or target_canonical_id).strip()
        return {
            "source_type": source_type,
            "source_id": source_id,
            "target_type": target_type,
            "target_id": target_id,
        }

    def _relation_reason_payload(
        self,
        *,
        reason: dict[str, Any] | None,
        context: dict[str, str],
        blocked_by: str = "",
        semantic_validation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = dict(reason or {})
        payload["kind"] = "relation"
        payload["source_type"] = context["source_type"]
        payload["source_id"] = context["source_id"]
        payload["target_type"] = context["target_type"]
        payload["target_id"] = context["target_id"]
        if blocked_by:
            payload["blocked_by"] = blocked_by
        if semantic_validation:
            payload["semantic_validation"] = semantic_validation
        return payload

    def _validate_minimum_relation_semantics(
        self,
        predicate_id: str,
        *,
        source_type: str,
        target_type: str,
    ) -> dict[str, Any]:
        return self.db.ontology_store.validate_minimum_predicate_usage(
            predicate_id,
            source_type,
            target_type,
        )

    def _update_pending_reason(self, pending_id: int, reason: dict[str, Any]) -> None:
        self.db.ontology_store.update_ontology_pending_reason(int(pending_id), reason)

    def _sanitize_keyword(self, value: str) -> str:
        token = (value or "").strip()
        if not token or detect_p0(token):
            return ""
        return token

    def _make_pointer(self, note_path: str, snippet_seed: str, source_url: str = "") -> dict[str, str]:
        pointer = {
            "type": "note",
            "path": note_path or "",
            "heading": "",
            "block_id": "",
            "snippet_hash": _snippet_hash(snippet_seed),
        }
        if source_url:
            pointer["source_url"] = source_url
        return pointer

    def _llm_refine_unknown_terms(
        self,
        unknown_terms: list[str],
        topic: str,
        note_title: str,
        source_url: str,
        allow_external: bool,
    ) -> tuple[dict[str, tuple[str, float]], bool]:
        """Return candidate -> (canonical_name, confidence)."""
        if not allow_external or not unknown_terms:
            return {}, False
        if self.config is None:
            return {}, False

        sanitized_terms = [term for term in unknown_terms if term and not detect_p0(term)]
        if not sanitized_terms:
            return {}, False

        payload = redact_payload({
            "topic": topic,
            "title": note_title[:120],
            "source_url": source_url[:240],
            "candidate_terms": sanitized_terms[:20],
        })

        try:
            from knowledge_hub.infrastructure.providers import get_llm

            provider = self.config.summarization_provider
            provider_cfg = self.config.get_provider_config(provider)
            llm = get_llm(provider, model=self.config.summarization_model, **provider_cfg)
            prompt = (
                "Return strict JSON only. "
                "Map each candidate term to a canonical concept name and confidence.\n"
                "Schema: {\"mappings\":[{\"candidate\":\"...\",\"canonical\":\"...\",\"confidence\":0.0}]}\n"
                f"Input: {json.dumps(payload, ensure_ascii=False)}"
            )
            raw = llm.generate(prompt=prompt, max_tokens=350)
            parsed = json.loads(raw)
            mappings = parsed.get("mappings", []) if isinstance(parsed, dict) else []
            result: dict[str, tuple[str, float]] = {}
            for item in mappings:
                if not isinstance(item, dict):
                    continue
                candidate = str(item.get("candidate", "")).strip()
                canonical = str(item.get("canonical", "")).strip()
                if not candidate or not canonical:
                    continue
                try:
                    confidence = float(item.get("confidence", 0.0))
                except Exception:
                    confidence = 0.0
                result[candidate] = (canonical, max(0.0, min(1.0, confidence)))
            return result, True
        except Exception:
            return {}, False

    def _accept_or_pending(
        self,
        value: float,
        accept_threshold: float,
        pending_threshold: float,
    ) -> str:
        if value >= accept_threshold:
            return "accepted"
        if value >= pending_threshold:
            return "pending"
        return "dropped"

    @staticmethod
    def _apply_schema_annotation(payload: dict[str, Any], schema_id: str) -> dict[str, Any]:
        try:
            annotate_schema_errors(payload, schema_id)
        except Exception as error:
            log.warning("schema annotation failed (%s): %s", schema_id, error)
        return payload

    def _upsert_concept_if_needed(
        self,
        concept_id: str,
        display_name: str,
        *,
        note_title: str = "",
        source_url: str = "",
        tags: list[str] | None = None,
        aliases: list[str] | None = None,
        related_names: list[str] | None = None,
        relation_predicates: list[str] | None = None,
    ) -> None:
        existing = self.db.get_ontology_entity(concept_id)
        existing_properties = (
            existing.get("properties")
            if isinstance(existing, dict) and isinstance(existing.get("properties"), dict)
            else {}
        )
        parsed_url = urlparse(source_url) if source_url else None
        domain = str(parsed_url.netloc or "").strip().lower() if parsed_url else ""
        classification = classify_ai_concept(
            canonical_name=display_name.strip(),
            aliases=[str(item).strip() for item in (aliases or []) if str(item).strip()],
            title=note_title,
            domain=domain,
            tags=[str(item).strip() for item in (tags or []) if str(item).strip()],
            related_names=[str(item).strip() for item in (related_names or []) if str(item).strip()],
            relation_predicates=[str(item).strip() for item in (relation_predicates or []) if str(item).strip()],
            source_type="web",
        )
        merged_properties, properties_changed = merge_ai_classification_properties(existing_properties, classification)
        if existing:
            if str(existing.get("canonical_name", "")).strip() != display_name.strip() or properties_changed:
                self.db.upsert_ontology_entity(
                    entity_id=concept_id,
                    entity_type="concept",
                    canonical_name=display_name.strip(),
                    description=str(existing.get("description", "") or ""),
                    properties=merged_properties,
                    confidence=float(existing.get("confidence", 1.0) or 1.0),
                    source=str(existing.get("source", "web_ingest") or "web_ingest"),
                )
            return

        existing_by_name = self.db.resolve_entity(display_name.strip(), entity_type="concept")
        if existing_by_name:
            resolved_id = str(existing_by_name.get("entity_id") or "").strip()
            if resolved_id and resolved_id != concept_id:
                self._upsert_concept_if_needed(
                    resolved_id,
                    display_name,
                    note_title=note_title,
                    source_url=source_url,
                    tags=tags,
                    aliases=aliases,
                    related_names=related_names,
                    relation_predicates=relation_predicates,
                )
            return

        self.db.upsert_ontology_entity(
            entity_id=concept_id,
            entity_type="concept",
            canonical_name=display_name.strip(),
            properties=merged_properties,
            source="web_ingest",
        )

    def _track_alias(self, concept_id: str, alias: str) -> bool:
        alias = (alias or "").strip()
        if not alias:
            return False
        existing_aliases = self.db.get_entity_aliases(concept_id)
        if alias in existing_aliases:
            return False
        self.db.add_entity_alias(alias, concept_id)
        return True

    def _resolve_claim_entity(
        self,
        raw_value: str,
        related_entities: list[dict[str, Any]],
    ) -> tuple[str, float, str]:
        token = str(raw_value or "").strip()
        if not token:
            return "", 0.0, ""

        normalized = normalize_term(token)
        for entity in related_entities:
            entity_id = str(entity.get("canonical_id") or entity.get("entity_id") or "").strip()
            display_name = str(entity.get("display_name") or "").strip()
            aliases = [str(alias).strip() for alias in (entity.get("aliases") or []) if str(alias).strip()]
            names = [display_name, *aliases]
            if entity_id == token:
                return entity_id, float(entity.get("confidence") or 0.8), display_name or token
            if any(normalize_term(name) == normalized for name in names if name):
                return entity_id, float(entity.get("confidence") or 0.8), display_name or token

        resolved = self.resolver.resolve(token, entity_type="concept")
        if resolved:
            return (
                str(resolved.canonical_id),
                float(resolved.resolve_confidence or 0.6),
                str(resolved.display_name or token),
            )
        return "", 0.0, token

    def _extract_claims_for_note(
        self,
        *,
        topic: str,
        run_id: str,
        note_id: str,
        note_title: str,
        source_url: str,
        note_path: str,
        content: str,
        related_entities: list[dict[str, Any]],
        allow_external: bool,
        claim_accept_threshold: float = 0.82,
        claim_pending_threshold: float = 0.65,
    ) -> tuple[list[dict[str, Any]], list[int], int, list[str]]:
        if not related_entities:
            return [], [], 0, []

        note_row = self.db.get_note(note_id) or {}
        note_metadata = note_row.get("metadata") if isinstance(note_row.get("metadata"), dict) else {}
        record_id = str(note_metadata.get("record_id") or "").strip()
        source_item_id = str(note_metadata.get("source_item_id") or "").strip()
        warnings: list[str] = []
        accepted_claims: list[dict[str, Any]] = []
        pending_claim_ids: list[int] = []
        dropped_claims = 0

        candidates, route_warnings = extract_web_claim_candidates(
            config=self.config,
            title=note_title,
            text=content,
            source_metadata={
                "source_url": source_url,
                "note_id": note_id,
                "record_id": record_id,
            },
            related_entities=related_entities,
            allow_external=allow_external,
            max_claims=6,
            max_input_chars=3500,
        )
        warnings.extend(route_warnings)

        seen_claims: set[str] = set()
        for candidate in candidates:
            subject_entity_id, subject_conf, _subject_name = self._resolve_claim_entity(candidate.subject, related_entities)
            if not subject_entity_id:
                dropped_claims += 1
                continue
            object_entity_id, object_conf, object_display = self._resolve_claim_entity(candidate.object_value, related_entities)
            object_literal = None
            if not object_entity_id:
                object_literal = str(candidate.object_value or "").strip() or object_display or None
            resolve_conf = subject_conf
            if object_entity_id:
                resolve_conf = max(0.0, min(1.0, (subject_conf + object_conf) / 2.0))
            evidence_quality = estimate_evidence_quality(candidate.evidence)
            claim_score, score_breakdown = score_claim_with_breakdown(
                candidate.llm_confidence,
                resolve_conf,
                evidence_quality,
                claim_text=candidate.claim_text,
                evidence=candidate.evidence,
                subject=candidate.subject,
                predicate=str(candidate.predicate or "").strip().lower(),
                object_value=str(candidate.object_value or object_literal or object_display or ""),
            )
            claim_id = _new_claim_id(
                note_id=note_id,
                subject_entity_id=subject_entity_id,
                predicate=str(candidate.predicate or "").strip().lower(),
                object_entity_id=object_entity_id,
                object_literal=str(object_literal or ""),
                claim_text=candidate.claim_text,
            )
            if claim_id in seen_claims:
                continue
            seen_claims.add(claim_id)
            evidence_ptr = self._make_pointer(
                note_path=note_path,
                snippet_seed=f"{note_id}:{candidate.claim_text}:{run_id}",
                source_url=source_url,
            )
            evidence_ptr["note_id"] = note_id
            if record_id:
                evidence_ptr["record_id"] = record_id
            if source_item_id:
                evidence_ptr["source_item_id"] = source_item_id
            bucket = self._accept_or_pending(
                claim_score,
                accept_threshold=claim_accept_threshold,
                pending_threshold=claim_pending_threshold,
            )
            evidence_ptr["claim_decision"] = bucket
            evidence_ptr["score_breakdown"] = score_breakdown
            if bucket == "accepted":
                self.db.upsert_claim(
                    claim_id=claim_id,
                    claim_text=candidate.claim_text,
                    subject_entity_id=subject_entity_id,
                    predicate=str(candidate.predicate or "").strip().lower(),
                    object_entity_id=object_entity_id or None,
                    object_literal=object_literal,
                    confidence=claim_score,
                    evidence_ptrs=[evidence_ptr],
                    source="web_claim_extractor",
                )
                accepted_claims.append(
                    {
                        "claimId": claim_id,
                        "claimText": candidate.claim_text,
                        "subjectEntityId": subject_entity_id,
                        "predicate": str(candidate.predicate or "").strip().lower(),
                        "objectEntityId": object_entity_id or "",
                        "objectLiteral": object_literal or "",
                        "confidence": round(claim_score, 6),
                    }
                )
                continue

            if bucket == "pending":
                pending_id = self.db.add_ontology_pending(
                    pending_type="claim",
                    run_id=run_id,
                    topic_slug=topic,
                    note_id=note_id,
                    source_url=source_url,
                    source_entity_id=subject_entity_id,
                    predicate_id=str(candidate.predicate or "").strip().lower(),
                    target_entity_id=object_entity_id or "",
                    confidence=claim_score,
                    evidence_ptrs=[evidence_ptr],
                    reason={
                        "kind": "claim",
                        "claim_id": claim_id,
                        "claim_text": candidate.claim_text,
                        "subject_entity_id": subject_entity_id,
                        "predicate": str(candidate.predicate or "").strip().lower(),
                        "object_entity_id": object_entity_id or "",
                        "object_literal": object_literal or "",
                        "evidence_quality": evidence_quality,
                        "resolve_confidence": resolve_conf,
                        "entity_resolution_confidence": score_breakdown.get("entity_resolution_confidence", resolve_conf),
                        "generic_claim_penalty": score_breakdown.get("generic_claim_penalty", 0.0),
                        "contradiction_hint": score_breakdown.get("contradiction_hint", 0.0),
                        "score_breakdown": score_breakdown,
                    },
                    status="pending",
                )
                if pending_id:
                    pending_claim_ids.append(int(pending_id))
                continue

            dropped_claims += 1

        return accepted_claims, pending_claim_ids, dropped_claims, warnings

    def extract_claims_from_notes(
        self,
        *,
        topic: str,
        note_rows: list[dict[str, Any]],
        run_id: str | None = None,
        allow_external: bool = False,
    ) -> dict[str, Any]:
        topic_slug = _slugify(topic)
        run_id = run_id or f"web_claim_{uuid4().hex[:12]}"
        accepted_claims: list[dict[str, Any]] = []
        pending_ids: list[int] = []
        pending_samples: list[dict[str, Any]] = []
        dropped_claims = 0
        warnings: list[str] = []
        note_summaries: list[dict[str, Any]] = []

        for note in note_rows:
            note_id = str(note.get("note_id", "")).strip()
            source_url = str(note.get("url", "")).strip()
            note_title = str(note.get("title", "")).strip()
            content = str(note.get("content", "") or "")
            note_path = str(note.get("file_path", "")).strip()
            if not note_id or not content.strip():
                continue

            relations = self.db.get_relations("note", note_id)
            concept_ids: set[str] = set()
            for relation in relations:
                source_type = str(relation.get("source_type") or "").strip()
                target_type = str(relation.get("target_type") or "").strip()
                source_id = str(relation.get("source_id") or relation.get("source_entity_id") or "").strip()
                target_id = str(relation.get("target_id") or relation.get("target_entity_id") or "").strip()
                if source_type == "note" and source_id == note_id and target_type == "concept" and target_id:
                    concept_ids.add(target_id)
                if target_type == "note" and target_id == note_id and source_type == "concept" and source_id:
                    concept_ids.add(source_id)

            related_entities = []
            for concept_id in sorted(concept_ids):
                entity = self.db.get_ontology_entity(concept_id) or {}
                related_entities.append(
                    {
                        "canonical_id": concept_id,
                        "display_name": str(entity.get("canonical_name") or concept_id),
                        "aliases": self.db.get_entity_aliases(concept_id),
                        "confidence": float(entity.get("confidence", 0.0) or 0.0),
                    }
                )

            claim_rows, claim_pending_ids, note_dropped_claims, claim_warnings = self._extract_claims_for_note(
                topic=topic_slug,
                run_id=run_id,
                note_id=note_id,
                note_title=note_title,
                source_url=source_url,
                note_path=note_path,
                content=content,
                related_entities=related_entities,
                allow_external=allow_external,
            )
            accepted_claims.extend(claim_rows)
            dropped_claims += int(note_dropped_claims)
            warnings.extend(claim_warnings)
            note_pending = 0
            for pending_id in claim_pending_ids:
                pending_ids.append(int(pending_id))
                note_pending += 1
                if len(pending_samples) < 50:
                    pending_samples.append(
                        {
                            "id": int(pending_id),
                            "kind": "claim",
                            "noteId": note_id,
                        }
                    )

            note_summaries.append(
                {
                    "noteId": note_id,
                    "url": source_url,
                    "claimsAccepted": len(claim_rows),
                    "pendingCount": note_pending,
                }
            )

        return {
            "runId": run_id,
            "topic": topic,
            "topicSlug": topic_slug,
            "claimsAccepted": len(accepted_claims),
            "pendingCount": len(pending_ids),
            "droppedClaims": dropped_claims,
            "acceptedClaims": accepted_claims[:200],
            "pendingSamples": pending_samples,
            "noteSummaries": note_summaries,
            "warnings": warnings,
            "policy": {
                "allowExternal": bool(allow_external),
                "externalUsed": bool(allow_external),
            },
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "updatedAt": datetime.now(timezone.utc).isoformat(),
        }

    def extract_from_notes(
        self,
        topic: str,
        note_rows: list[dict[str, Any]],
        run_id: str | None = None,
        allow_external: bool = False,
        concept_threshold: float = 0.78,
        relation_threshold: float = 0.75,
        concept_pending_threshold: float = 0.60,
        relation_pending_threshold: float = 0.55,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        run_id = run_id or f"web_ont_{uuid4().hex[:12]}"
        topic_slug = _slugify(topic)

        accepted_concepts: dict[str, dict[str, Any]] = {}
        accepted_concept_conf: dict[str, float] = {}
        accepted_relations: list[dict[str, Any]] = []
        accepted_claims: list[dict[str, Any]] = []
        pending_ids: set[int] = set()
        pending_samples: list[dict[str, Any]] = []
        note_summaries: list[dict[str, Any]] = []
        aliases_added = 0
        dropped_concepts = 0
        dropped_relations = 0
        dropped_claims = 0
        external_used = False
        warnings: list[str] = []

        for note in note_rows:
            note_id = str(note.get("note_id", "")).strip()
            source_url = str(note.get("url", "")).strip()
            note_title = str(note.get("title", "")).strip()
            content = str(note.get("content", "") or "")
            note_path = str(note.get("file_path", "")).strip()
            normalized_text = normalize_term(content)
            if not note_id or not content.strip():
                continue

            keywords = extract_keywords_from_text(f"{note_title}\n{content}", max_keywords=18)
            keywords = [self._sanitize_keyword(token) for token in keywords]
            keywords = [token for token in keywords if token]
            # preserve order while deduplicating
            unique_keywords: list[str] = []
            seen_tokens: set[str] = set()
            for token in keywords:
                normalized_token = normalize_term(token)
                if not normalized_token or normalized_token in seen_tokens:
                    continue
                seen_tokens.add(normalized_token)
                unique_keywords.append(token)

            unresolved_terms: list[str] = []
            keyword_to_resolved: dict[str, tuple[str, float, str, str]] = {}
            for keyword in unique_keywords:
                resolved = self.resolver.resolve(keyword, entity_type="concept")
                if resolved:
                    keyword_to_resolved[keyword] = (
                        resolved.canonical_id,
                        float(resolved.resolve_confidence),
                        resolved.display_name,
                        resolved.resolve_method,
                    )
                else:
                    unresolved_terms.append(keyword)

            llm_mappings: dict[str, tuple[str, float]] = {}
            used_external = False
            if allow_external and unresolved_terms:
                llm_mappings, used_external = self._llm_refine_unknown_terms(
                    unknown_terms=unresolved_terms,
                    topic=topic,
                    note_title=note_title,
                    source_url=source_url,
                    allow_external=allow_external,
                )
            external_used = external_used or used_external

            note_mentions: dict[str, set[str]] = {}
            note_concepts_accepted = 0
            note_relations_accepted = 0
            note_claims_accepted = 0
            note_pending = 0

            for keyword in unique_keywords:
                resolved = keyword_to_resolved.get(keyword)
                resolve_conf = 0.55
                resolve_method = "unknown"
                concept_id = _new_concept_id(keyword)
                display_name = keyword.strip()
                aliases = [keyword.strip()]

                if resolved:
                    concept_id, resolve_conf, display_name, resolve_method = resolved
                elif keyword in llm_mappings:
                    mapped_name, mapped_conf = llm_mappings[keyword]
                    display_name = mapped_name
                    resolve_conf = max(0.55, mapped_conf)
                    resolve_method = "llm-sanitized"
                    existing = self.db.resolve_entity(mapped_name, entity_type="concept")
                    if existing and existing.get("entity_id"):
                        concept_id = str(existing.get("entity_id"))
                    else:
                        concept_id = _new_concept_id(mapped_name)

                freq_score = _frequency_score(normalized_text, keyword)
                source_score = 1.0
                confidence = 0.5 * resolve_conf + 0.3 * freq_score + 0.2 * source_score
                confidence = max(0.0, min(1.0, confidence))
                ptr = self._make_pointer(note_path=note_path, snippet_seed=f"{keyword}:{note_id}", source_url=source_url)
                concept = _EntityCandidate(
                    note_id=note_id,
                    source_url=source_url,
                    original_term=keyword,
                    canonical_id=concept_id,
                    display_name=display_name,
                    entity_type="concept",
                    aliases=aliases,
                    confidence=confidence,
                    resolve_confidence=resolve_conf,
                    resolve_method=resolve_method,
                    evidence_ptrs=[ptr],
                )
                bucket = self._accept_or_pending(
                    confidence,
                    accept_threshold=concept_threshold,
                    pending_threshold=concept_pending_threshold,
                )

                if bucket == "accepted":
                    self._upsert_concept_if_needed(
                        concept.canonical_id,
                        concept.display_name,
                        note_title=note_title,
                        source_url=source_url,
                        tags=[topic],
                        aliases=concept.aliases,
                    )
                    if normalize_term(concept.original_term) != normalize_term(concept.display_name):
                        if self._track_alias(concept.canonical_id, concept.original_term):
                            aliases_added += 1

                    self.db.add_relation(
                        source_type="note",
                        source_id=note_id,
                        relation="mentions",
                        target_type="concept",
                        target_id=concept.canonical_id,
                        evidence_text=json.dumps(
                            {
                                "source": "web",
                                "run_id": run_id,
                                "relation_norm": "mentions",
                                "evidence_ptrs": concept.evidence_ptrs,
                            },
                            ensure_ascii=False,
                        ),
                        confidence=confidence,
                    )
                    accepted = accepted_concepts.setdefault(
                        concept.canonical_id,
                        {
                            "canonical_id": concept.canonical_id,
                            "display_name": concept.display_name,
                            "confidence": confidence,
                            "resolveMethod": concept.resolve_method,
                            "evidencePtrs": concept.evidence_ptrs,
                        },
                    )
                    if confidence > float(accepted.get("confidence", 0.0)):
                        accepted["confidence"] = confidence
                        accepted["evidencePtrs"] = concept.evidence_ptrs
                    accepted_concept_conf[concept.canonical_id] = max(
                        float(accepted_concept_conf.get(concept.canonical_id, 0.0)),
                        confidence,
                    )
                    mentions = note_mentions.setdefault(concept.canonical_id, set())
                    mentions.add(concept.original_term)
                    mentions.add(concept.display_name)
                    for alias in self.db.get_entity_aliases(concept.canonical_id):
                        mentions.add(alias)
                    note_concepts_accepted += 1
                    continue

                if bucket == "pending":
                    pending_id = self.db.add_web_ontology_pending(
                        run_id=run_id,
                        topic_slug=topic_slug,
                        note_id=note_id,
                        source_url=source_url,
                        source_canonical_id=concept.canonical_id,
                        relation_norm="concept_candidate",
                        target_canonical_id=concept.canonical_id,
                        confidence=confidence,
                        evidence_ptrs=concept.evidence_ptrs,
                        reason={
                            "kind": "concept",
                            "display_name": concept.display_name,
                            "original_term": concept.original_term,
                            "aliases": concept.aliases,
                            "resolve_method": concept.resolve_method,
                        },
                        status="pending",
                    )
                    if pending_id:
                        pending_ids.add(pending_id)
                        note_pending += 1
                        if len(pending_samples) < 20:
                            pending_samples.append(
                                {
                                    "id": pending_id,
                                    "kind": "concept",
                                    "display_name": concept.display_name,
                                    "confidence": round(confidence, 4),
                                }
                            )
                    continue

                dropped_concepts += 1

            sentence_relations = self._extract_relations_for_note(
                note_id=note_id,
                source_url=source_url,
                note_path=note_path,
                content=content,
                note_mentions=note_mentions,
                concept_conf=accepted_concept_conf,
                run_id=run_id,
            )

            relation_seen: set[tuple[str, str, str]] = set()
            accepted_relation_keys: set[tuple[str, str, str]] = set()
            for relation in sentence_relations:
                key = (
                    relation.source_canonical_id,
                    relation.relation_norm,
                    relation.target_canonical_id,
                )
                if key in relation_seen:
                    continue
                relation_seen.add(key)

                bucket = self._accept_or_pending(
                    relation.confidence,
                    accept_threshold=relation_threshold,
                    pending_threshold=relation_pending_threshold,
                )
                relation_context = self._relation_endpoint_context(
                    source_canonical_id=relation.source_canonical_id,
                    target_canonical_id=relation.target_canonical_id,
                    reason=relation.reason,
                )
                if bucket == "accepted":
                    if not self._is_predicate_approved(relation.relation_norm):
                        ext_pending_id = self.db.add_ontology_pending(
                            pending_type="predicate_ext",
                            run_id=run_id,
                            topic_slug=topic_slug,
                            note_id=note_id,
                            source_url=source_url,
                            source_entity_id=relation.source_canonical_id,
                            predicate_id=relation.relation_norm,
                            target_entity_id=relation.target_canonical_id,
                            confidence=relation.confidence,
                            evidence_ptrs=relation.evidence_ptrs,
                            reason={
                                "kind": "predicate_ext",
                                "predicate_id": relation.relation_norm,
                                "description": f"web relation candidate: {relation.relation_norm}",
                                "note_id": note_id,
                                "run_id": run_id,
                            },
                            status="pending",
                        )
                        relation_pending_id = self.db.add_ontology_pending(
                            pending_type="relation",
                            run_id=run_id,
                            topic_slug=topic_slug,
                            note_id=note_id,
                            source_url=source_url,
                            source_entity_id=relation.source_canonical_id,
                            predicate_id=relation.relation_norm,
                            target_entity_id=relation.target_canonical_id,
                            confidence=relation.confidence,
                            evidence_ptrs=relation.evidence_ptrs,
                            reason={
                                "kind": "relation",
                                "blocked_by": "predicate_not_approved",
                                **self._relation_reason_payload(
                                    reason=relation.reason,
                                    context=relation_context,
                                ),
                            },
                            status="pending",
                        )
                        for pending_id in [ext_pending_id, relation_pending_id]:
                            if pending_id:
                                pending_ids.add(int(pending_id))
                                note_pending += 1
                                if len(pending_samples) < 20:
                                    pending_samples.append(
                                        {
                                            "id": int(pending_id),
                                            "kind": "predicate_ext"
                                            if pending_id == ext_pending_id
                                            else "relation",
                                            "relation_norm": relation.relation_norm,
                                            "confidence": round(relation.confidence, 4),
                                        }
                                    )
                        continue

                    validation = self._validate_minimum_relation_semantics(
                        relation.relation_norm,
                        source_type=relation_context["source_type"],
                        target_type=relation_context["target_type"],
                    )
                    if validation:
                        pending_id = self.db.add_web_ontology_pending(
                            run_id=run_id,
                            topic_slug=topic_slug,
                            note_id=note_id,
                            source_url=source_url,
                            source_canonical_id=relation.source_canonical_id,
                            relation_norm=relation.relation_norm,
                            target_canonical_id=relation.target_canonical_id,
                            confidence=relation.confidence,
                            evidence_ptrs=relation.evidence_ptrs,
                            reason=self._relation_reason_payload(
                                reason=relation.reason,
                                context=relation_context,
                                blocked_by="predicate_semantics",
                                semantic_validation=validation,
                            ),
                            status="pending",
                        )
                        if pending_id:
                            pending_ids.add(int(pending_id))
                            note_pending += 1
                            if len(pending_samples) < 20:
                                pending_samples.append(
                                    {
                                        "id": int(pending_id),
                                        "kind": "relation",
                                        "relation_norm": relation.relation_norm,
                                        "confidence": round(relation.confidence, 4),
                                    }
                                )
                        continue

                    self.db.add_relation(
                        source_type=relation_context["source_type"],
                        source_id=relation_context["source_id"],
                        relation=relation.relation_norm,
                        target_type=relation_context["target_type"],
                        target_id=relation_context["target_id"],
                        evidence_text=json.dumps(
                            {
                                "source": "web",
                                "run_id": run_id,
                                "relation_norm": relation.relation_norm,
                                "evidence_ptrs": relation.evidence_ptrs,
                                "reason": relation.reason,
                            },
                            ensure_ascii=False,
                        ),
                        confidence=relation.confidence,
                    )
                    note_relations_accepted += 1
                    accepted_relations.append(
                        {
                            "source_canonical_id": relation.source_canonical_id,
                            "relation_norm": relation.relation_norm,
                            "target_canonical_id": relation.target_canonical_id,
                            "confidence": round(relation.confidence, 6),
                            "evidence_ptrs": relation.evidence_ptrs,
                        }
                    )
                    accepted_relation_keys.add(key)
                    continue

                if bucket == "pending":
                    pending_id = self.db.add_web_ontology_pending(
                        run_id=run_id,
                        topic_slug=topic_slug,
                        note_id=note_id,
                        source_url=source_url,
                        source_canonical_id=relation.source_canonical_id,
                        relation_norm=relation.relation_norm,
                        target_canonical_id=relation.target_canonical_id,
                        confidence=relation.confidence,
                        evidence_ptrs=relation.evidence_ptrs,
                        reason=self._relation_reason_payload(
                            reason=relation.reason,
                            context=relation_context,
                        ),
                        status="pending",
                    )
                    if pending_id:
                        pending_ids.add(pending_id)
                        note_pending += 1
                        if len(pending_samples) < 20:
                            pending_samples.append(
                                {
                                    "id": pending_id,
                                    "kind": "relation",
                                    "relation_norm": relation.relation_norm,
                                    "confidence": round(relation.confidence, 4),
                                }
                            )
                    continue

                dropped_relations += 1

            note_related_entities = [
                {
                    "canonical_id": concept_id,
                    "display_name": item.get("display_name"),
                    "aliases": self.db.get_entity_aliases(concept_id),
                    "confidence": accepted_concept_conf.get(concept_id, item.get("confidence", 0.0)),
                }
                for concept_id, item in accepted_concepts.items()
                if concept_id in note_mentions
            ]
            claim_rows, claim_pending_ids, note_dropped_claims, claim_warnings = self._extract_claims_for_note(
                topic=topic_slug,
                run_id=run_id,
                note_id=note_id,
                note_title=note_title,
                source_url=source_url,
                note_path=note_path,
                content=content,
                related_entities=note_related_entities,
                allow_external=allow_external,
            )
            accepted_claims.extend(claim_rows)
            note_claims_accepted = len(claim_rows)
            dropped_claims += int(note_dropped_claims)
            warnings.extend(claim_warnings)
            note_relation_predicates: dict[str, list[str]] = {}
            note_related_names: dict[str, list[str]] = {}
            note_related_name_map = {
                str(item.get("canonical_id") or ""): str(item.get("display_name") or "")
                for item in note_related_entities
                if str(item.get("canonical_id") or "").strip()
            }
            for relation in sentence_relations:
                key = (
                    relation.source_canonical_id,
                    relation.relation_norm,
                    relation.target_canonical_id,
                )
                if key not in accepted_relation_keys:
                    continue
                for concept_id, neighbor_id in (
                    (relation.source_canonical_id, relation.target_canonical_id),
                    (relation.target_canonical_id, relation.source_canonical_id),
                ):
                    if concept_id not in note_mentions:
                        continue
                    note_relation_predicates.setdefault(concept_id, []).append(relation.relation_norm)
                    neighbor_name = note_related_name_map.get(neighbor_id, "")
                    if neighbor_name:
                        note_related_names.setdefault(concept_id, []).append(neighbor_name)
            for concept_id in note_mentions:
                entity = accepted_concepts.get(concept_id) or {}
                self._upsert_concept_if_needed(
                    concept_id,
                    str(entity.get("display_name") or concept_id),
                    note_title=note_title,
                    source_url=source_url,
                    tags=[topic],
                    aliases=sorted(note_mentions.get(concept_id, set())),
                    related_names=note_related_names.get(concept_id, []),
                    relation_predicates=note_relation_predicates.get(concept_id, []),
                )
            for pending_id in claim_pending_ids:
                pending_ids.add(int(pending_id))
                note_pending += 1
                if len(pending_samples) < 20:
                    pending_samples.append(
                        {
                            "id": int(pending_id),
                            "kind": "claim",
                            "noteId": note_id,
                        }
                    )

            note_summaries.append(
                {
                    "noteId": note_id,
                    "url": source_url,
                    "conceptsAccepted": note_concepts_accepted,
                    "relationsAccepted": note_relations_accepted,
                    "claimsAccepted": note_claims_accepted,
                    "pendingCount": note_pending,
                }
            )

        return {
            "runId": run_id,
            "topic": topic,
            "topicSlug": topic_slug,
            "createdAt": now,
            "updatedAt": datetime.now(timezone.utc).isoformat(),
            "conceptsAccepted": len(accepted_concepts),
            "relationsAccepted": len(accepted_relations),
            "claimsAccepted": len(accepted_claims),
            "pendingCount": len(pending_ids),
            "aliasesAdded": aliases_added,
            "droppedConcepts": dropped_concepts,
            "droppedRelations": dropped_relations,
            "droppedClaims": dropped_claims,
            "acceptedConcepts": sorted(
                accepted_concepts.values(),
                key=lambda item: float(item.get("confidence", 0.0)),
                reverse=True,
            )[:100],
            "acceptedRelations": accepted_relations[:200],
            "acceptedClaims": accepted_claims[:200],
            "pendingSamples": pending_samples,
            "noteSummaries": note_summaries,
            "policy": {
                "allowExternal": bool(allow_external),
                "externalUsed": external_used,
            },
            "warnings": warnings,
        }

    def _extract_relations_for_note(
        self,
        note_id: str,
        source_url: str,
        note_path: str,
        content: str,
        note_mentions: dict[str, set[str]],
        concept_conf: dict[str, float],
        run_id: str,
    ) -> list[_RelationCandidate]:
        results: list[_RelationCandidate] = []
        sentences = _split_sentences(content)
        for sentence in sentences:
            normalized_sentence = normalize_term(sentence)
            if not normalized_sentence:
                continue

            mention_positions: list[tuple[str, int]] = []
            for concept_id, mentions in note_mentions.items():
                best_pos = None
                for mention in mentions:
                    needle = normalize_term(mention)
                    if not needle:
                        continue
                    pos = normalized_sentence.find(needle)
                    if pos >= 0 and (best_pos is None or pos < best_pos):
                        best_pos = pos
                if best_pos is not None:
                    mention_positions.append((concept_id, best_pos))

            if len(mention_positions) < 2:
                continue

            mention_positions.sort(key=lambda item: item[1])
            ordered_concepts = [item[0] for item in mention_positions]

            relation_norm, pattern_score = _relation_from_sentence(sentence)
            if relation_norm not in RELATION_ENUM:
                relation_norm = "unknown_relation"
            for idx in range(len(ordered_concepts) - 1):
                source_id = ordered_concepts[idx]
                target_id = ordered_concepts[idx + 1]
                if source_id == target_id:
                    continue
                if relation_norm == "related_to" and source_id > target_id:
                    source_id, target_id = target_id, source_id

                endpoint_conf = (
                    float(concept_conf.get(source_id, 0.5))
                    + float(concept_conf.get(target_id, 0.5))
                ) / 2.0
                confidence = (
                    0.4 * pattern_score
                    + 0.3 * _cooccur_score(len(ordered_concepts))
                    + 0.3 * endpoint_conf
                )
                confidence = max(0.0, min(1.0, confidence))
                pointer = self._make_pointer(
                    note_path=note_path,
                    snippet_seed=f"{note_id}:{sentence}:{run_id}",
                    source_url=source_url,
                )
                results.append(
                    _RelationCandidate(
                        note_id=note_id,
                        source_url=source_url,
                        source_canonical_id=source_id,
                        relation_norm=relation_norm,
                        target_canonical_id=target_id,
                        confidence=confidence,
                        evidence_ptrs=[pointer],
                        reason={
                            "note_id": note_id,
                            "pattern_score": pattern_score,
                            "cooccur_score": _cooccur_score(len(ordered_concepts)),
                            "endpoint_conf": endpoint_conf,
                        },
                    )
                )
        return results

    def list_pending(self, topic: str = "", limit: int = 50) -> dict[str, Any]:
        topic_slug = _slugify(topic) if topic else ""
        items = self.db.list_web_ontology_pending(topic_slug=topic_slug or None, status="pending", limit=limit)
        payload = {
            "schema": "knowledge-hub.crawl.pending.list.result.v1",
            "topic": topic,
            "topicSlug": topic_slug or None,
            "status": "ok",
            "count": len(items),
            "items": items,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "updatedAt": datetime.now(timezone.utc).isoformat(),
        }
        return self._apply_schema_annotation(payload, "knowledge-hub.crawl.pending.list.result.v1")

    def apply_pending(self, pending_id: int) -> dict[str, Any]:
        row = self.db.get_web_ontology_pending(int(pending_id))
        if not row:
            return self._apply_schema_annotation(
                {
                "schema": "knowledge-hub.crawl.pending.apply.result.v1",
                "status": "error",
                "error": f"pending id not found: {pending_id}",
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "updatedAt": datetime.now(timezone.utc).isoformat(),
                },
                "knowledge-hub.crawl.pending.apply.result.v1",
            )
        if row.get("status") != "pending":
            return self._apply_schema_annotation(
                {
                "schema": "knowledge-hub.crawl.pending.apply.result.v1",
                "status": "error",
                "error": f"pending id is already {row.get('status')}",
                "item": row,
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "updatedAt": datetime.now(timezone.utc).isoformat(),
                },
                "knowledge-hub.crawl.pending.apply.result.v1",
            )

        reason = row.get("reason_json") if isinstance(row.get("reason_json"), dict) else {}
        kind = str(reason.get("kind", "relation"))
        applied = False

        if kind == "concept":
            concept_id = str(row.get("source_canonical_id", "")).strip()
            display_name = str(reason.get("display_name") or reason.get("original_term") or concept_id).strip()
            if concept_id and display_name:
                self._upsert_concept_if_needed(concept_id, display_name)
                for alias in reason.get("aliases", []):
                    self._track_alias(concept_id, str(alias))
                self.db.add_relation(
                    source_type="note",
                    source_id=str(row.get("note_id", "")),
                    relation="mentions",
                    target_type="concept",
                    target_id=concept_id,
                    evidence_text=json.dumps(
                        {
                            "source": "web",
                            "run_id": row.get("run_id"),
                            "relation_norm": "mentions",
                            "evidence_ptrs": row.get("evidence_ptrs_json") or [],
                        },
                        ensure_ascii=False,
                    ),
                    confidence=float(row.get("confidence", 0.0)),
                )
                applied = True
        else:
            relation_norm = str(row.get("relation_norm", "related_to"))
            if not self._is_predicate_approved(relation_norm):
                return self._apply_schema_annotation(
                    {
                        "schema": "knowledge-hub.crawl.pending.apply.result.v1",
                        "status": "error",
                        "error": f"predicate not approved: {relation_norm}",
                        "item": row,
                        "createdAt": datetime.now(timezone.utc).isoformat(),
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                    },
                    "knowledge-hub.crawl.pending.apply.result.v1",
                )
            relation_context = self._relation_endpoint_context(
                source_canonical_id=str(row.get("source_canonical_id", "")).strip(),
                target_canonical_id=str(row.get("target_canonical_id", "")).strip(),
                reason=reason,
            )
            validation = self._validate_minimum_relation_semantics(
                relation_norm,
                source_type=relation_context["source_type"],
                target_type=relation_context["target_type"],
            )
            if validation:
                reason_payload = self._relation_reason_payload(
                    reason=reason,
                    context=relation_context,
                    blocked_by="predicate_semantics",
                    semantic_validation=validation,
                )
                self._update_pending_reason(int(pending_id), reason_payload)
                self.db.update_web_ontology_pending_status(int(pending_id), "rejected")
                updated = self.db.get_web_ontology_pending(int(pending_id))
                issue_text = "; ".join(validation.get("issues", []))
                return self._apply_schema_annotation(
                    {
                        "schema": "knowledge-hub.crawl.pending.apply.result.v1",
                        "status": "error",
                        "error": f"predicate semantics rejected: {relation_norm} ({issue_text})",
                        "item": updated,
                        "createdAt": datetime.now(timezone.utc).isoformat(),
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                    },
                    "knowledge-hub.crawl.pending.apply.result.v1",
                )
            source_id = relation_context["source_id"]
            target_id = relation_context["target_id"]
            if source_id and target_id:
                if not self.db.get_ontology_entity(source_id):
                    self.db.upsert_ontology_entity(
                        entity_id=source_id,
                        entity_type=relation_context["source_type"],
                        canonical_name=source_id,
                        source="web_pending_apply",
                    )
                if not self.db.get_ontology_entity(target_id):
                    self.db.upsert_ontology_entity(
                        entity_id=target_id,
                        entity_type=relation_context["target_type"],
                        canonical_name=target_id,
                        source="web_pending_apply",
                    )
                self.db.add_relation(
                    source_type=relation_context["source_type"],
                    source_id=source_id,
                    relation=relation_norm,
                    target_type=relation_context["target_type"],
                    target_id=target_id,
                    evidence_text=json.dumps(
                        {
                            "source": "web",
                            "run_id": row.get("run_id"),
                            "relation_norm": relation_norm,
                            "evidence_ptrs": row.get("evidence_ptrs_json") or [],
                            "reason": reason,
                        },
                        ensure_ascii=False,
                    ),
                    confidence=float(row.get("confidence", 0.0)),
                )
                applied = True

        if not applied:
            return self._apply_schema_annotation(
                {
                "schema": "knowledge-hub.crawl.pending.apply.result.v1",
                "status": "error",
                "error": f"unable to apply pending id: {pending_id}",
                "item": row,
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "updatedAt": datetime.now(timezone.utc).isoformat(),
                },
                "knowledge-hub.crawl.pending.apply.result.v1",
            )

        self.db.update_web_ontology_pending_status(int(pending_id), "approved")
        updated = self.db.get_web_ontology_pending(int(pending_id))
        return self._apply_schema_annotation(
            {
            "schema": "knowledge-hub.crawl.pending.apply.result.v1",
            "status": "ok",
            "item": updated,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "updatedAt": datetime.now(timezone.utc).isoformat(),
            },
            "knowledge-hub.crawl.pending.apply.result.v1",
        )

    def reject_pending(self, pending_id: int) -> dict[str, Any]:
        row = self.db.get_web_ontology_pending(int(pending_id))
        if not row:
            return self._apply_schema_annotation(
                {
                "schema": "knowledge-hub.crawl.pending.reject.result.v1",
                "status": "error",
                "error": f"pending id not found: {pending_id}",
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "updatedAt": datetime.now(timezone.utc).isoformat(),
                },
                "knowledge-hub.crawl.pending.reject.result.v1",
            )
        self.db.update_web_ontology_pending_status(int(pending_id), "rejected")
        updated = self.db.get_web_ontology_pending(int(pending_id))
        return self._apply_schema_annotation(
            {
            "schema": "knowledge-hub.crawl.pending.reject.result.v1",
            "status": "ok",
            "item": updated,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "updatedAt": datetime.now(timezone.utc).isoformat(),
            },
            "knowledge-hub.crawl.pending.reject.result.v1",
        )
