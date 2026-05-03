"""Concept identity resolution for Learning Coach."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.learning.models import ConceptIdentity


def normalize_term(value: str) -> str:
    lowered = (value or "").strip().lower()
    lowered = re.sub(r"[_\-]+", " ", lowered)
    lowered = re.sub(r"[^a-z0-9가-힣\s]", "", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


_COMMON_CONCEPT_QUERY_ALIASES = {
    "cnn": "convolutional neural network",
    "cnns": "convolutional neural networks",
}


def _expand_concept_query_alias(term: str, *, entity_type: str | None) -> str:
    if str(entity_type or "").strip().lower() != "concept":
        return term
    normalized = normalize_term(term)
    alias = _COMMON_CONCEPT_QUERY_ALIASES.get(normalized)
    return str(alias or term or "").strip()


@dataclass
class _ConceptCandidate:
    concept_id: str
    canonical_name: str
    aliases: list[str]
    normalized: str
    normalized_aliases: list[str]


class ConceptResolver:
    """Deprecated: Use EntityResolver.

    This wrapper exists only for backward compatibility and delegates all
    resolution to EntityResolver with ``entity_type="concept"``.
    """

    def __init__(self, db: SQLiteDatabase, fuzzy_threshold: float = 0.88):
        self.db = db
        self.fuzzy_threshold = max(0.0, min(1.0, fuzzy_threshold))
        # Backward-compatible wrapper: concept resolution delegates to EntityResolver.
        self._entity_resolver = EntityResolver(db, fuzzy_threshold=self.fuzzy_threshold)

    def resolve(self, raw_name: str) -> ConceptIdentity | None:
        return self._entity_resolver.resolve(raw_name, entity_type="concept")


@dataclass
class _EntityCandidate:
    """통합 엔티티 후보"""
    entity_id: str
    entity_type: str
    canonical_name: str
    aliases: list[str]
    normalized: str
    normalized_aliases: list[str]


class EntityResolver:
    """
    통합 온톨로지 엔티티 해석기
    
    ConceptResolver를 확장하여 concept, person, paper, organization 등
    모든 entity type을 처리할 수 있습니다.
    """
    
    def __init__(self, db: SQLiteDatabase, fuzzy_threshold: float = 0.88):
        self.db = db
        self.fuzzy_threshold = max(0.0, min(1.0, fuzzy_threshold))
        self._candidates: dict[str, list[_EntityCandidate]] = {}
        self._load_all_entities()
    
    def _load_all_entities(self):
        """모든 entity type의 엔티티를 로드"""
        # ontology_entities 테이블에서 로드
        entities = self.db.list_ontology_entities(limit=5000)
        
        for entity in entities:
            entity_id = str(entity.get("entity_id", "")).strip()
            entity_type = str(entity.get("entity_type", "concept")).strip()
            name = str(entity.get("canonical_name", "")).strip()
            
            if not entity_id or not name:
                continue
            
            aliases = self.db.get_entity_aliases(entity_id)
            
            candidate = _EntityCandidate(
                entity_id=entity_id,
                entity_type=entity_type,
                canonical_name=name,
                aliases=aliases,
                normalized=normalize_term(name),
                normalized_aliases=[normalize_term(alias) for alias in aliases if normalize_term(alias)],
            )
            
            if entity_type not in self._candidates:
                self._candidates[entity_type] = []
            self._candidates[entity_type].append(candidate)
    
    def resolve(
        self, 
        raw_name: str, 
        entity_type: str | None = None
    ) -> ConceptIdentity | None:
        """
        이름/별칭으로 엔티티 해석
        
        Args:
            raw_name: 해석할 이름
            entity_type: 필터링할 엔티티 타입 (None이면 모든 타입 검색)
        """
        term = (raw_name or "").strip()
        if not term:
            return None
        term = _expand_concept_query_alias(term, entity_type=entity_type)
        
        normalized = normalize_term(term)
        
        # 검색할 후보 목록 결정
        if entity_type and entity_type in self._candidates:
            search_candidates = self._candidates[entity_type]
        else:
            # 모든 타입 검색
            search_candidates = []
            for candidates_list in self._candidates.values():
                search_candidates.extend(candidates_list)
        
        # 1) exact canonical/id
        for item in search_candidates:
            if term == item.canonical_name or term == item.entity_id:
                return ConceptIdentity(
                    canonical_id=item.entity_id,
                    display_name=item.canonical_name,
                    aliases=item.aliases,
                    resolve_confidence=1.0,
                    resolve_method="exact",
                )
        
        # 2) alias table exact
        for item in search_candidates:
            if term in item.aliases:
                return ConceptIdentity(
                    canonical_id=item.entity_id,
                    display_name=item.canonical_name,
                    aliases=item.aliases,
                    resolve_confidence=0.98,
                    resolve_method="alias",
                )
        
        # 3) normalized string exact
        for item in search_candidates:
            if normalized and (normalized == item.normalized or normalized in item.normalized_aliases):
                return ConceptIdentity(
                    canonical_id=item.entity_id,
                    display_name=item.canonical_name,
                    aliases=item.aliases,
                    resolve_confidence=0.95,
                    resolve_method="normalized",
                )
        
        # 4) fuzzy
        best: tuple[float, _EntityCandidate] | None = None
        for item in search_candidates:
            values = [item.normalized, *item.normalized_aliases]
            score = max(
                (SequenceMatcher(None, normalized, value).ratio() for value in values if value),
                default=0.0
            )
            if best is None or score > best[0]:
                best = (score, item)
        
        if best and best[0] >= self.fuzzy_threshold:
            return ConceptIdentity(
                canonical_id=best[1].entity_id,
                display_name=best[1].canonical_name,
                aliases=best[1].aliases,
                resolve_confidence=float(best[0]),
                resolve_method="fuzzy",
            )
        
        return None
    
    def resolve_concept(self, raw_name: str) -> ConceptIdentity | None:
        """concept 타입만 검색하는 편의 메서드 (하위 호환성)"""
        return self.resolve(raw_name, entity_type="concept")
