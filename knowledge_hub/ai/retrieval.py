from __future__ import annotations

from knowledge_hub.ai.rag_support import (
    clean_text as _clean_text,
    note_id_for_result as _note_id_for_result,
    safe_float as _safe_float,
    safe_int as _safe_int,
    tokenize as _tokenize,
)
from knowledge_hub.core.models import SearchResult


def apply_feature_boosts(searcher, results: list[SearchResult]) -> list[SearchResult]:
    if not searcher.sqlite_db:
        return results
    snapshot_finder = getattr(searcher.sqlite_db, "find_source_feature_snapshot", None)
    for item in results:
        metadata = item.metadata or {}
        snapshot = {}
        if callable(snapshot_finder):
            found = snapshot_finder(
                note_id=_note_id_for_result(item),
                record_id=str(metadata.get("record_id") or "").strip(),
                canonical_url=str(metadata.get("url") or metadata.get("canonical_url") or "").strip(),
                source_item_id=str(metadata.get("source_item_id") or "").strip(),
            )
            if isinstance(found, dict):
                snapshot = found
        importance = _safe_float(snapshot.get("importance_score"), 0.0)
        freshness = _safe_float(snapshot.get("freshness_score"), 0.0)
        claim_density = _safe_float(snapshot.get("claim_density"), 0.0)
        support_doc_count = _safe_int(snapshot.get("support_doc_count"), 0)
        normalized_support_doc_count = min(max(support_doc_count, 0) / 10.0, 1.0)
        contradiction = _safe_float(snapshot.get("contradiction_score"), 0.0)
        feature_boost = (
            (0.15 * importance)
            + (0.12 * freshness)
            + (0.08 * min(1.0, claim_density))
            + (0.05 * normalized_support_doc_count)
        )
        ranking = searcher._build_retrieval_ranking_signals(item, snapshot)
        item.score = max(
            0.0,
            min(
                1.0,
                _safe_float(item.score, 0.0)
                + feature_boost
                + _safe_float(ranking.get("quality_boost"), 0.0)
                + _safe_float(ranking.get("source_trust_boost"), 0.0)
                + _safe_float(ranking.get("reference_prior_boost"), 0.0)
                - _safe_float(ranking.get("contradiction_penalty"), 0.0),
            ),
        )
        extras = dict(item.lexical_extras or {})
        extras.setdefault("feature_importance", round(importance, 6))
        extras.setdefault("feature_freshness", round(freshness, 6))
        extras.setdefault("feature_claim_density", round(claim_density, 6))
        extras.setdefault("feature_support_doc_count", support_doc_count)
        extras.setdefault("feature_normalized_support_doc_count", round(normalized_support_doc_count, 6))
        extras.setdefault("feature_contradiction", round(contradiction, 6))
        extras.setdefault("feature_boost", round(feature_boost, 6))
        extras.setdefault("feature_penalty", round(_safe_float(ranking.get("contradiction_penalty"), 0.0), 6))
        extras.setdefault("quality_flag", ranking["quality_flag"])
        extras.setdefault("quality_boost", ranking["quality_boost"])
        extras.setdefault("source_trust_score", ranking["source_trust_score"])
        extras.setdefault("source_trust_boost", ranking["source_trust_boost"])
        extras.setdefault("reference_role", ranking["reference_role"])
        extras.setdefault("reference_tier", ranking["reference_tier"])
        extras.setdefault("reference_prior_boost", ranking["reference_prior_boost"])
        extras.setdefault("retrieval_adjusted_score", round(item.score, 6))
        extras.setdefault(
            "ranking_signals",
            {
                "quality_flag": ranking["quality_flag"],
                "quality_boost": ranking["quality_boost"],
                "source_trust_score": ranking["source_trust_score"],
                "source_trust_boost": ranking["source_trust_boost"],
                "reference_role": ranking["reference_role"],
                "reference_tier": ranking["reference_tier"],
                "reference_prior_boost": ranking["reference_prior_boost"],
                "contradiction_penalty": ranking["contradiction_penalty"],
                "feature_boost": round(feature_boost, 6),
                "retrieval_adjusted_score": round(item.score, 6),
            },
        )
        item.lexical_extras = extras
    return results


def expand_query_with_ontology(searcher, query: str, max_related: int = 5) -> list[str]:
    if not searcher.sqlite_db:
        return [query]
    try:
        from knowledge_hub.learning.resolver import EntityResolver

        resolver = EntityResolver(searcher.sqlite_db)
        tokens = _tokenize(query)
        related_terms = set()
        for token in tokens:
            identity = resolver.resolve(token)
            if identity:
                entity_id = str(identity.canonical_id)
                entity_type = "concept"

                entity_row = searcher.sqlite_db.get_ontology_entity(entity_id)
                if entity_row:
                    entity_type = str(entity_row.get("entity_type", "concept") or "concept")

                relations = searcher.sqlite_db.get_relations(entity_type, entity_id)
                for rel in relations[:max_related]:
                    source_type = str(rel.get("source_type", "") or "")
                    source_id = str(rel.get("source_id", "") or "")
                    target_type = str(rel.get("target_type", "") or "")
                    target_id = str(rel.get("target_id", "") or "")

                    neighbor_type = ""
                    neighbor_id = ""
                    if source_type == entity_type and source_id == entity_id:
                        neighbor_type = target_type
                        neighbor_id = target_id
                    elif target_type == entity_type and target_id == entity_id:
                        neighbor_type = source_type
                        neighbor_id = source_id

                    if not neighbor_id:
                        continue
                    if neighbor_id == entity_id and neighbor_type == entity_type:
                        continue

                    neighbor_name = ""
                    neighbor_entity = searcher.sqlite_db.get_ontology_entity(neighbor_id)
                    if neighbor_entity:
                        neighbor_name = str(neighbor_entity.get("canonical_name", "")).strip()

                    if neighbor_name:
                        related_terms.add(neighbor_name)

        expanded_queries = [query]
        if related_terms:
            related_str = " ".join(list(related_terms)[:max_related])
            expanded_queries.append(f"{query} {related_str}")
        return expanded_queries
    except Exception:
        return [query]


def semantic_search(searcher, query_embedding: list[float], top_k: int, filter_dict=None) -> list[SearchResult]:
    if not query_embedding:
        return []

    results = searcher.database.search(
        query_embedding=query_embedding,
        top_k=top_k,
        filter_dict=filter_dict,
    )

    if not results.get("documents") or not results["documents"][0]:
        return []

    documents = results["documents"][0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0] if results.get("ids") else []

    out: list[SearchResult] = []
    total = len(documents)
    for i in range(total):
        doc = documents[i]
        meta = metadatas[i] if i < len(metadatas) else {}
        distance = distances[i] if i < len(distances) else 1.0
        score = max(0.0, min(1.0, 1.0 - float(distance)))
        out.append(
            SearchResult(
                document=doc or "",
                metadata=meta or {},
                distance=float(distance),
                score=score,
                semantic_score=score,
                lexical_score=0.0,
                retrieval_mode="semantic",
                lexical_extras={"distance_rank": i + 1},
                document_id=str(ids[i]) if i < len(ids) else "",
            )
        )
    return out


def lexical_search(searcher, query: str, top_k: int, filter_dict=None) -> list[SearchResult]:
    if not query.strip():
        return []

    query_text = _clean_text(query)
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return []

    lexical_search_fn = getattr(searcher.database, "lexical_search", None)
    if callable(lexical_search_fn):
        try:
            fts_hits = lexical_search_fn(query=query_text, top_k=top_k, filter_dict=filter_dict)
        except TypeError:
            fts_hits = lexical_search_fn(query_text, top_k=top_k, filter_dict=filter_dict)
        except Exception:
            fts_hits = []
        if fts_hits:
            out: list[SearchResult] = []
            for rank_index, item in enumerate(fts_hits, start=1):
                score = max(0.0, min(1.0, float(item.get("score", 0.0))))
                out.append(
                    SearchResult(
                        document=str(item.get("document", "") or ""),
                        metadata=item.get("metadata", {}) or {},
                        distance=1.0,
                        score=score,
                        semantic_score=0.0,
                        lexical_score=score,
                        retrieval_mode="keyword",
                        lexical_extras={
                            "query": query_text,
                            "fts_rank": float(item.get("rank", 0.0) or 0.0),
                            "rank_position": rank_index,
                        },
                        document_id=str(item.get("id", "") or ""),
                    )
                )
            return out
        return []

    results = searcher.database.get_documents(
        filter_dict=filter_dict,
        limit=max(top_k, 200),
        include_ids=True,
        include_documents=True,
        include_metadatas=True,
    )
    docs = results.get("documents", [])
    metas = results.get("metadatas", [])
    ids = results.get("ids", [])

    if not docs:
        return []

    scored: list[tuple[float, SearchResult]] = []
    query_phrase = query_text.lower()

    for idx, document in enumerate(docs):
        metadata = metas[idx] if idx < len(metas) else {}
        source_text = _clean_text(
            f"{metadata.get('title', '')} "
            f"{metadata.get('section_title', '')} "
            f"{metadata.get('section_path', '')} "
            f"{metadata.get('contextual_summary', '')} "
            f"{metadata.get('keywords', '')} "
            f"{metadata.get('field', '')}"
        )
        candidate_tokens = _tokenize(f"{document} {source_text}")
        matched = query_tokens & candidate_tokens
        match_ratio = len(matched) / max(1, len(query_tokens))
        term_density = len(query_tokens) / max(1, len(candidate_tokens))

        phrase_score = 0.0
        phrase_text = _clean_text(document).lower()
        if query_phrase and query_phrase in phrase_text:
            phrase_score = 1.0
        elif metadata.get("contextual_summary"):
            summary = str(metadata.get("contextual_summary", "")).lower()
            if query_phrase and query_phrase in summary:
                phrase_score = 0.8

        score = 0.58 * match_ratio + 0.24 * phrase_score + 0.18 * term_density
        if metadata.get("section_title") or metadata.get("contextual_summary"):
            score = min(1.0, score + 0.05)

        score = max(0.0, min(1.0, score))
        if score < 0.05:
            continue

        lexical = SearchResult(
            document=document or "",
            metadata=metadata or {},
            distance=1.0,
            score=score,
            semantic_score=0.0,
            lexical_score=score,
            retrieval_mode="keyword",
            lexical_extras={
                "query": query_text,
                "query_terms": sorted(query_tokens),
                "matched_terms": sorted(matched),
                "match_ratio": match_ratio,
                "term_density": term_density,
            },
            document_id=str(ids[idx]) if idx < len(ids) else "",
        )
        scored.append((score, lexical))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [item for _, item in scored[:top_k]]
