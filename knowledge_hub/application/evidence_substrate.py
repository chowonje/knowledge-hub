from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from knowledge_hub.ai.compare_packet import build_compare_packet_from_runtime, build_compare_packet_from_sources
from knowledge_hub.application.evidence_registry import build_lineage, resolve_registry_lookup
from knowledge_hub.infrastructure.persistence.vector import inspect_vector_store


SUBSTRATE_CONTRACT_SCHEMA = "knowledge-hub.evidence-substrate.contract.v1"
INSPECT_RESULT_SCHEMA = "knowledge-hub.inspect.result.v1"
COMPARE_RESULT_SCHEMA = "knowledge-hub.compare.result.v1"
TRACE_RESULT_SCHEMA = "knowledge-hub.answer-trace.result.v1"


def build_substrate_contract() -> dict[str, Any]:
    """Return the public contract that ties local knowledge to Codex/MCP use."""

    return {
        "schema": SUBSTRATE_CONTRACT_SCHEMA,
        "status": "ok",
        "primaryWorkflow": "Codex context + evidence-backed local knowledge",
        "canonicalFlow": [
            "SourceLedgerRecord",
            "PreparedSourceRecord",
            "RetrievalUnit",
            "EvidencePacket",
            "ResearchContext/ComparePacket",
            "AnswerTrace",
        ],
        "sourceBoundary": {
            "persistentKnowledge": ["paper", "web", "vault_note", "curated_local_doc"],
            "ephemeralContext": ["current_repo", "current_task", "diff", "conversation_snippet"],
            "rule": "Persistent sources may be indexed; ephemeral context must stay bounded and inspectable unless explicitly promoted.",
        },
        "indexContract": {
            "khubIndexBuilds": ["lexical_index", "vector_index", "metadata_index"],
            "khubIndexDoesNotBuildByDefault": ["claim_cards", "evidence_links", "answer_traces"],
            "embeddingPolicy": "Embeddings are derived retrieval artifacts, not the source of truth.",
        },
        "publicFacades": {
            "inspect": "Inspect corpus/source/index/chunk state without mutating stores.",
            "compare": "Run a comparison-shaped ask facade and expose compare/evidence diagnostics.",
            "trace": "Trace an answer payload back to citations, evidence packet contracts, and sources.",
        },
        "mcpResources": [
            "khub://corpus/status",
            "khub://source/{source_id}",
            "khub://chunk/{chunk_id}",
            "khub://packet/{packet_id}",
            "khub://context/{context_pack_id}",
        ],
        "registryContract": {
            "authority": "Derived packet/context/trace lookup registry; source ledger and source-backed chunks remain factual authority.",
            "creation": "Explicit registry writes may persist EvidencePacket, ContextPack, and AnswerTrace lookup records.",
            "lookup": "khub://packet/{id} and khub://context/{id} resolve only when a registry record exists.",
            "staleness": "Records carry source refs and source revision hashes; mismatches are stale, not silently trusted.",
            "deletion": "Deleting a registry record removes only the lookup projection, never source records or indexes.",
        },
        "promotionCriteria": [
            "Improves Codex/MCP search, comparison, citation, or verification behavior.",
            "Keeps user-facing complexity small.",
            "Has a focused CLI/MCP/evidence regression check.",
        ],
    }


def _config_value(config: Any, name: str, default: str = "") -> str:
    value = getattr(config, name, default)
    if callable(value):
        try:
            value = value()
        except TypeError:
            value = default
    return str(value or default)


def _sqlite_stats(sqlite_db: Any) -> dict[str, Any]:
    if sqlite_db is None:
        return {"available": False, "reason": "sqlite_unavailable"}
    get_stats = getattr(sqlite_db, "get_stats", None)
    if callable(get_stats):
        try:
            stats = get_stats()
            return {"available": True, "stats": dict(stats or {})}
        except Exception as error:  # pragma: no cover - defensive for live stores
            return {"available": False, "reason": str(error)}
    return {"available": False, "reason": "get_stats_unavailable"}


def _lexical_db_path(config: Any) -> Path:
    vector_root = Path(_config_value(config, "vector_db_path", "~/.khub/chroma_db")).expanduser()
    return vector_root / "_lexical.sqlite3"


def _load_lexical_rows(config: Any, identifier: str, *, limit: int = 20) -> list[dict[str, Any]]:
    token = str(identifier or "").strip()
    if not token:
        return []
    db_path = _lexical_db_path(config)
    if not db_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    query = """
        SELECT f.doc_id, f.title, f.document, m.metadata_json
        FROM lexical_documents_fts AS f
        JOIN lexical_documents_meta AS m ON m.doc_id = f.doc_id
        WHERE f.doc_id = ?
           OR f.doc_id LIKE ?
           OR m.metadata_json LIKE ?
        LIMIT ?
    """
    like_token = f"%{token}%"
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        for row in conn.execute(query, (token, like_token, like_token, max(1, int(limit)))):
            metadata: dict[str, Any] = {}
            try:
                metadata = dict(json.loads(str(row["metadata_json"] or "{}")) or {})
            except json.JSONDecodeError:
                metadata = {}
            rows.append(
                {
                    "chunkId": str(row["doc_id"] or ""),
                    "title": str(row["title"] or metadata.get("title") or ""),
                    "documentPreview": str(row["document"] or "")[:1200],
                    "metadata": metadata,
                    "uri": f"khub://chunk/{row['doc_id']}",
                }
            )
    except sqlite3.Error:
        return []
    finally:
        try:
            conn.close()  # type: ignore[name-defined]
        except Exception:
            pass
    return rows


def build_inspect_payload(
    *,
    config: Any,
    sqlite_db: Any = None,
    target: str = "corpus",
    identifier: str = "",
    limit: int = 20,
) -> dict[str, Any]:
    target = str(target or "corpus").strip().lower()
    identifier = str(identifier or "").strip()
    collection_name = _config_value(config, "collection_name", "knowledge_hub")
    vector_db_path = _config_value(config, "vector_db_path", "~/.khub/chroma_db")
    sqlite_path = _config_value(config, "sqlite_path", "~/.khub/knowledge.db")
    vector_status = inspect_vector_store(vector_db_path, collection_name)

    payload: dict[str, Any] = {
        "schema": INSPECT_RESULT_SCHEMA,
        "status": "ok",
        "target": target,
        "identifier": identifier,
        "contract": build_substrate_contract(),
        "stores": {
            "sqlite": {
                "path": sqlite_path,
                **_sqlite_stats(sqlite_db),
            },
            "vector": vector_status,
            "lexical": {
                "path": str(_lexical_db_path(config)),
                "available": _lexical_db_path(config).exists(),
            },
        },
        "persistentEphemeralBoundary": build_substrate_contract()["sourceBoundary"],
    }
    if target in {"source", "chunk"}:
        rows = _load_lexical_rows(config, identifier, limit=limit)
        payload["matches"] = rows
        payload["matchCount"] = len(rows)
        if not rows:
            payload["status"] = "not_found"
            payload["warnings"] = [f"{target} identifier not found in lexical metadata"]
    elif target == "index":
        payload["indexContract"] = build_substrate_contract()["indexContract"]
    elif target == "contract":
        payload["contract"] = build_substrate_contract()
    elif target in {"packet", "context", "trace"}:
        if sqlite_db is None:
            lookup = {
                "status": "not_found",
                "reason": "sqlite unavailable for registry lookup",
                "registryRecord": {},
            }
        else:
            lookup = resolve_registry_lookup(sqlite_db, target, identifier)
        payload["registryLookup"] = lookup
        payload["matchCount"] = 1 if lookup.get("status") in {"ok", "stale"} else 0
        if lookup.get("status") not in {"ok", "stale"}:
            payload["status"] = "not_found"
            payload["warnings"] = [str(lookup.get("reason") or f"{target} registry record not found")]
    return payload


def build_trace_payload(answer_payload: dict[str, Any], *, question: str = "") -> dict[str, Any]:
    payload = dict(answer_payload or {})
    evidence_contract = dict(payload.get("evidencePacketContract") or {})
    answer_contract = dict(payload.get("answerContract") or {})
    compare_contract = dict(payload.get("comparePacketContract") or {})
    citations = list(payload.get("citations") or [])
    sources = list(payload.get("sources") or payload.get("evidence") or [])
    spans = list(evidence_contract.get("spans") or [])
    gaps = list(evidence_contract.get("gaps") or [])
    warnings = [str(item) for item in list(payload.get("warnings") or [])]

    status = "ok" if citations and spans else "insufficient_evidence"
    if not citations:
        warnings.append("trace has no citation entries")
    if not spans:
        warnings.append("trace has no citation-grade evidence spans")

    return {
        "schema": TRACE_RESULT_SCHEMA,
        "status": status,
        "question": str(question or payload.get("question") or ""),
        "answerId": str(answer_contract.get("answer_id") or answer_contract.get("answerId") or ""),
        "evidencePacketId": str(
            evidence_contract.get("packet_id")
            or evidence_contract.get("packetId")
            or answer_contract.get("evidence_packet_id")
            or answer_contract.get("evidencePacketId")
            or ""
        ),
        "answer": str(payload.get("answer") or ""),
        "citations": citations,
        "sources": sources,
        "evidenceSpans": spans,
        "lineage": build_lineage(
            {
                "citations": citations,
                "sources": sources,
                "evidenceSpans": spans,
                "comparePacket": compare_contract,
            }
        ),
        "gaps": gaps,
        "comparePacket": compare_contract,
        "warnings": warnings,
        "nextSuggestedActions": [
            {"tool": "khub_inspect", "args": {"target": "source"}},
            {"tool": "khub_context", "args": {"question": str(question or payload.get("question") or "")}},
        ],
    }


def build_compare_payload(answer_payload: dict[str, Any], *, query: str) -> dict[str, Any]:
    trace = build_trace_payload(answer_payload, question=query)
    compare_contract = dict(answer_payload.get("comparePacketContract") or {})
    evidence_contract = dict(answer_payload.get("evidencePacketContract") or {})
    answer_contract = dict(answer_payload.get("answerContract") or {})
    warnings = list(trace["warnings"])
    if not compare_contract:
        v2_payload = dict(answer_payload.get("v2") or {})
        runtime_compare = build_compare_packet_from_runtime(
            query=query,
            source_type=str(answer_payload.get("sourceType") or answer_payload.get("source_type") or ""),
            family=str(answer_payload.get("paperFamily") or ""),
            runtime_execution=dict(v2_payload.get("runtimeExecution") or {}),
            query_frame=dict(answer_payload.get("queryFrame") or {}),
            claim_cards=list(v2_payload.get("claimCards") or answer_payload.get("claimCards") or []),
            claim_alignment=dict(v2_payload.get("claimAlignment") or answer_payload.get("claimAlignment") or {}),
            paper_knowledge_slots=list(v2_payload.get("paperKnowledgeSlots") or []),
            evidence_policy=dict(answer_payload.get("evidencePolicy") or {}),
            comparison_verification=dict(v2_payload.get("comparisonVerification") or {}),
        )
        if runtime_compare:
            compare_contract = runtime_compare
            warnings.append("compare packet built from ask-v2 paper knowledge slots")
    enriched_compare = build_compare_packet_from_sources(
        query=query,
        sources=list(trace.get("sources") or []),
        citations=list(trace.get("citations") or []),
        strict_spans=list(evidence_contract.get("spans") or []),
        strict_citations=list(answer_contract.get("citations") or []),
        existing_packet=compare_contract or None,
        policy=dict(answer_payload.get("evidencePolicy") or {}),
    )
    if enriched_compare:
        if compare_contract:
            old_span_count = int(((compare_contract.get("coverage") or {}).get("supportingSpanCount")) or 0)
            new_span_count = int(((enriched_compare.get("coverage") or {}).get("supportingSpanCount")) or 0)
            if new_span_count > old_span_count:
                warnings.append("compare packet enriched with retrieved source spans")
        else:
            warnings.append("compare packet built from retrieved source spans because claim-aligned dimensions were unavailable")
        compare_contract = enriched_compare
        trace["comparePacket"] = compare_contract
        trace["lineage"] = build_lineage(
            {
                "citations": trace.get("citations") or [],
                "sources": trace.get("sources") or [],
                "evidenceSpans": trace.get("evidenceSpans") or [],
                "comparePacket": compare_contract,
            }
        )
    status = "ok"
    if not compare_contract:
        status = "insufficient_compare_contract"
    if trace["status"] == "insufficient_evidence":
        status = "insufficient_evidence"
    return {
        "schema": COMPARE_RESULT_SCHEMA,
        "status": status,
        "query": str(query or ""),
        "answer": str(answer_payload.get("answer") or ""),
        "comparePacket": compare_contract,
        "trace": trace,
        "citations": trace["citations"],
        "sources": trace["sources"],
        "warnings": warnings,
        "nextSuggestedActions": [
            {"tool": "khub_trace", "args": {"question": str(query or "")}},
            {"tool": "khub_inspect", "args": {"target": "corpus"}},
        ],
    }
