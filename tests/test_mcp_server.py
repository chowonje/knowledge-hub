from __future__ import annotations

import asyncio
import importlib
import json
from types import SimpleNamespace

import pytest
from knowledge_hub.core.schema_validator import validate_payload


def _import_mcp_server():
    try:
        return importlib.import_module("knowledge_hub.interfaces.mcp.server")
    except SystemExit:
        pytest.skip("mcp dependency is unavailable in test environment")


def _import_tool_specs():
    try:
        return importlib.import_module("knowledge_hub.mcp.tool_specs")
    except SystemExit:
        pytest.skip("mcp dependency is unavailable in test environment")


def _decode_response(contents):
    assert contents
    text = contents[0].text
    return json.loads(text)


def _enable_labs_profile(monkeypatch):
    monkeypatch.setenv("KHUB_MCP_PROFILE", "labs")


class _FakeSearcher:
    def __init__(self):
        self.database = SimpleNamespace(get_stats=lambda: {"total_documents": 3, "collection_name": "knowledge_hub"})
        self.llm = SimpleNamespace(generate=lambda prompt, context="": f"generated:{prompt[:24]}::{len(context)}")

    def search(self, query, top_k=5, source_type="all", retrieval_mode="hybrid", alpha=0.7, expand_parent_context=True):
        return [
            SimpleNamespace(
                metadata={
                    "title": "RAG Note",
                    "source_type": "note",
                    "resolved_parent_id": "p1",
                    "resolved_parent_label": "Section A",
                    "resolved_parent_chunk_span": "0-2",
                },
                score=0.92,
                semantic_score=0.9,
                lexical_score=0.8,
                retrieval_mode="hybrid",
                lexical_extras={
                    "quality_flag": "ok",
                    "source_trust_score": 0.94,
                    "reference_role": "glossary_reference",
                    "reference_tier": "specialist",
                    "ranking_signals": {
                        "quality_flag": "ok",
                        "reference_role": "glossary_reference",
                        "reference_tier": "specialist",
                    },
                },
                document="retrieval augmented generation",
            )
        ]

    def generate_answer(
        self,
        question,
        top_k=5,
        min_score=0.3,
        source_type="all",
        retrieval_mode="hybrid",
        alpha=0.7,
        allow_external=False,
        paper_memory_mode="off",
        metadata_filter=None,
    ):
        _ = (question, top_k, min_score, source_type, retrieval_mode, alpha, allow_external)
        return {
            "answer": "RAG 재작성 답변",
            "warnings": ["answer verification caution: unsupported=0 uncertain=1 conflict_mentioned=False"],
            "answerSignals": {
                "total_sources": 1,
                "quality_counts": {"ok": 1, "needs_review": 0, "reject": 0, "unscored": 0},
                "preferred_sources": 1,
                "specialist_reference_count": 1,
                "high_trust_source_count": 1,
                "contradictory_source_count": 0,
                "contradicting_belief_count": 0,
                "strongest_quality_flag": "ok",
                "caution_required": False,
            },
            "answerVerification": {
                "status": "verified",
                "supportedClaimCount": 1,
                "unsupportedClaimCount": 0,
                "uncertainClaimCount": 0,
                "conflictMentioned": True,
                "needsCaution": False,
                "summary": "핵심 답변이 근거와 직접 연결됩니다.",
                "warnings": [],
                "claims": [
                    {
                        "claim": "RAG는 검색된 문서를 바탕으로 답변을 생성한다.",
                        "verdict": "supported",
                        "evidenceTitles": ["RAG Note"],
                        "reason": "source excerpt가 답변을 직접 지지합니다.",
                    }
                ],
                "route": {"route": "strong", "provider": "openai", "model": "gpt-5.4"},
            },
            "initialAnswerVerification": {
                "status": "caution",
                "supportedClaimCount": 0,
                "unsupportedClaimCount": 1,
                "uncertainClaimCount": 0,
                "conflictMentioned": True,
                "needsCaution": True,
                "summary": "원본 답변에는 unsupported claim이 있었습니다.",
                "warnings": [],
                "claims": [],
                "route": {"route": "strong", "provider": "openai", "model": "gpt-5.4"},
            },
            "answerRewrite": {
                "attempted": True,
                "applied": True,
                "triggeredBy": ["unsupported_claim"],
                "attemptCount": 1,
                "summary": "근거 밖 주장을 제거하기 위해 답변을 1회 재작성했습니다.",
                "originalAnswer": "원본 답변",
                "finalAnswerSource": "rewritten",
                "route": {"route": "strong", "provider": "openai", "model": "gpt-5.4"},
                "warnings": [],
            },
            "sources": [
                {
                    "title": "RAG Note",
                    "source_type": "note",
                    "score": 0.92,
                    "semantic_score": 0.9,
                    "lexical_score": 0.8,
                    "retrieval_mode": "hybrid",
                    "parent_id": "p1",
                    "parent_label": "Section A",
                    "parent_chunk_span": "0-2",
                    "quality_flag": "ok",
                    "source_trust_score": 0.94,
                    "reference_role": "glossary_reference",
                    "reference_tier": "specialist",
                    "ranking_signals": {
                        "quality_flag": "ok",
                        "reference_role": "glossary_reference",
                        "reference_tier": "specialist",
                    },
                }
            ],
            "paperMemoryPrefilter": {
                "requestedMode": paper_memory_mode,
                "effectiveMode": "compat" if paper_memory_mode == "prefilter" else paper_memory_mode,
                "modeAliasApplied": paper_memory_mode == "prefilter",
                "applied": source_type == "paper" and paper_memory_mode == "prefilter",
                "fallbackUsed": False,
                "matchedPaperIds": [str((metadata_filter or {}).get("arxiv_id") or "2501.00001")] if source_type == "paper" and paper_memory_mode == "prefilter" else [],
                "matchedMemoryIds": ["paper-memory:2501.00001:test"] if source_type == "paper" and paper_memory_mode == "prefilter" else [],
                "reason": "matched_cards" if source_type == "paper" and paper_memory_mode == "prefilter" else "disabled",
            },
        }

    def build_ops_report(self, limit=100, days=7):
        _ = (limit, days)
        return {
            "schema": "knowledge-hub.rag.report.result.v1",
            "status": "ok",
            "days": 7,
            "limit": 100,
            "counts": {
                "total": 3,
                "needsCaution": 1,
                "rewriteAttempted": 1,
                "rewriteApplied": 1,
                "conservativeFallback": 0,
                "unsupportedClaimLogs": 1,
            },
            "verification": {"verified": 2, "caution": 1, "failed": 0, "skipped": 0, "unknown": 0},
            "rates": {
                "needsCautionRate": 0.3333,
                "rewriteAttemptedRate": 0.3333,
                "rewriteAppliedRate": 0.3333,
                "conservativeFallbackRate": 0.0,
                "unsupportedClaimRate": 0.3333,
            },
            "topWarningPatterns": [{"warning": "answer verification caution", "count": 1}],
            "alerts": [
                {
                    "code": "rag_verification_failed_or_skipped",
                    "severity": "warning",
                    "scope": "rag",
                    "summary": "verification failed/skipped 사례가 1건 있습니다.",
                    "metric": "verification.failed+skipped",
                    "observed": 1,
                    "threshold": "> 0",
                }
            ],
            "recommendedActions": [
                {
                    "actionType": "inspect_verification_routes",
                    "scope": "rag",
                    "summary": "verification route와 policy/config 경로를 점검하세요.",
                    "reasonCodes": ["rag_verification_failed_or_skipped"],
                    "command": "khub",
                    "args": ["config", "list"],
                }
            ],
            "samples": [
                {
                    "id": 1,
                    "createdAt": "2026-03-16T10:00:00Z",
                    "queryHash": "abc",
                    "queryDigest": "RAG란 무엇인가...",
                    "resultStatus": "ok",
                    "verificationStatus": "caution",
                    "needsCaution": True,
                    "unsupportedClaimCount": 1,
                    "rewriteApplied": True,
                    "finalAnswerSource": "rewritten",
                    "warningCount": 1,
                }
            ],
            "warnings": [],
        }


class _FakeLearningCoachService:
    def __init__(self, _config):
        pass

    def start_or_resume_topic(self, **kwargs):
        return {
            "schema": "knowledge-hub.learning.start-resume.result.v1",
            "status": "ok",
            "topic": kwargs.get("topic", ""),
            "topicSlug": "rag",
            "session_id": "session-rag",
            "sessionId": "session-rag",
            "recent_state_summary": {"gateStatus": "draft"},
            "weak_areas": [{"conceptId": "c1", "displayName": "Concept 1", "status": "unknown"}],
            "next_review_targets": ["Concept 1"],
            "recommended_next_step": "ask-question",
            "runId": "learn_start_1",
            "createdAt": "2026-03-25T00:00:00Z",
            "updatedAt": "2026-03-25T00:00:00Z",
        }

    def get_session_state(self, **kwargs):
        topic = kwargs.get("topic") or "rag"
        return {
            "schema": "knowledge-hub.learning.session-state.result.v1",
            "status": "ok",
            "topic": topic,
            "topicSlug": "rag",
            "session": {"sessionId": "session-rag", "targetTrunkIds": ["c1"], "status": "draft"},
            "progress": {"gateStatus": "draft", "gatePassed": False, "scoreFinal": 0.0},
            "summary": {"checkpointSummary": "Need work on Concept 1"},
            "conceptStates": [{"conceptId": "c1", "displayName": "Concept 1", "status": "unknown"}],
            "quizHistory": [{"runId": "quiz1", "score": 50.0}],
            "weakAreas": [{"conceptId": "c1", "displayName": "Concept 1", "status": "unknown"}],
            "nextReviewTargets": ["Concept 1"],
            "recentEvents": [],
            "runId": "learn_state_1",
            "createdAt": "2026-03-25T00:00:00Z",
            "updatedAt": "2026-03-25T00:00:00Z",
        }

    def explain_topic(self, **kwargs):
        return {
            "schema": "knowledge-hub.learning.explain.result.v1",
            "status": "ok",
            "topic": kwargs.get("topic", ""),
            "session_id": kwargs.get("session_id") or "session-rag",
            "sessionId": kwargs.get("session_id") or "session-rag",
            "question": kwargs.get("question", ""),
            "answer": "RAG는 검색된 문맥을 사용해 답변을 만든다.",
            "evidence_refs": ["RAG Note"],
            "evidenceRefs": ["RAG Note"],
            "supplemental_model_knowledge": [],
            "supplementalModelKnowledge": [],
            "followup_checks": ["RAG의 핵심 단계를 말해보세요."],
            "followupChecks": ["RAG의 핵심 단계를 말해보세요."],
            "stateSummary": {"checkpointSummary": "Need work on Concept 1"},
            "runId": "learn_explain_1",
            "createdAt": "2026-03-25T00:00:00Z",
            "updatedAt": "2026-03-25T00:00:00Z",
        }

    def checkpoint(self, **kwargs):
        return {
            "schema": "knowledge-hub.learning.checkpoint.result.v1",
            "status": "ok",
            "saved": True,
            "topic": kwargs.get("topic", ""),
            "session_id": kwargs.get("session_id", ""),
            "sessionId": kwargs.get("session_id", ""),
            "updated_concepts": [{"conceptId": "c1", "displayName": "Concept 1", "status": "unknown"}],
            "updatedConcepts": [{"conceptId": "c1", "displayName": "Concept 1", "status": "unknown"}],
            "weak_areas": [{"conceptId": "c1", "displayName": "Concept 1", "status": "unknown"}],
            "weakAreas": [{"conceptId": "c1", "displayName": "Concept 1", "status": "unknown"}],
            "next_review_targets": ["Concept 1"],
            "nextReviewTargets": ["Concept 1"],
            "summary": kwargs.get("summary", ""),
            "stateSummary": {"checkpointSummary": kwargs.get("summary", "")},
            "runId": "learn_checkpoint_1",
            "createdAt": "2026-03-25T00:00:00Z",
            "updatedAt": "2026-03-25T00:00:00Z",
        }

    def map(self, **kwargs):
        return {
            "schema": "knowledge-hub.learning.map.result.v1",
            "status": "ok",
            "topic": kwargs.get("topic", ""),
            "trunks": [{"canonical_id": "c1", "display_name": "Concept 1"}],
            "branches": [{"canonical_id": "c2", "display_name": "Concept 2", "parentTrunkIds": ["c1"]}],
        }

    def reinforce(self, **kwargs):
        return {
            "schema": "knowledge-hub.learning.reinforce.result.v1",
            "status": "ok",
            "topic": kwargs.get("topic", ""),
            "sessionId": kwargs.get("session_id", ""),
            "actions": [{"actionType": "read", "targetEntityName": "Concept 1"}],
        }


class _FakeWebIngestService:
    last_crawl_kwargs = None

    def __init__(self, _config):
        pass

    def crawl_and_ingest(self, **kwargs):
        type(self).last_crawl_kwargs = dict(kwargs)
        count = len(kwargs.get("urls") or [])
        return {
            "schema": "knowledge-hub.crawl.ingest.result.v1",
            "status": "ok",
            "runId": "crawl_ingest_1",
            "requested": count,
            "crawled": count,
            "stored": count,
            "indexedChunks": count,
            "failed": [],
            "engine": kwargs.get("engine", "auto"),
            "topic": kwargs.get("topic", ""),
            "warnings": ["youtube_caption_unavailable"] if kwargs.get("input_source") == "youtube" else [],
            "indexDiagnostics": {
                "requested": bool(kwargs.get("index", True)),
                "attempted": bool(kwargs.get("index", True)),
                "autoRetryEligible": kwargs.get("index_autofix_mode") == "youtube_single_retry",
                "autoRetryAttempted": kwargs.get("index_autofix_mode") == "youtube_single_retry",
                "initialIndexedChunks": 0 if kwargs.get("input_source") == "youtube" else count,
                "finalIndexedChunks": count,
                "status": "retry_succeeded" if kwargs.get("input_source") == "youtube" else "indexed",
                "reason": "index retry succeeded" if kwargs.get("input_source") == "youtube" else "indexing completed",
                "warnings": ["youtube_index_retry_succeeded"] if kwargs.get("input_source") == "youtube" else [],
            },
            "ontology": {
                "runId": "crawl_ingest_1",
                "conceptsAccepted": 0,
                "relationsAccepted": 0,
                "pendingCount": 0,
                "aliasesAdded": 0,
            },
            "writebackPaths": [],
            "createdAt": "2026-03-25T00:00:00Z",
            "updatedAt": "2026-03-25T00:00:00Z",
        }


class _FakeSQLiteDB:
    def __init__(self):
        self.document_memory_units = {}
        self.paper_memory_cards = {}
        self.ops_actions = [
            {
                "action_id": "ops_action_1",
                "scope": "rag",
                "action_type": "inspect_rag_samples",
                "status": "pending",
                "target_kind": "rag_window",
                "target_key": "window:days=7;limit=100",
                "summary": "rag sample을 점검하세요.",
                "reason_codes_json": ["rag_high_caution_rate"],
                "command": "khub",
                "args_json": ["rag-report", "--days", "7", "--limit", "100"],
                "alert_json": [],
                "seen_count": 1,
                "first_seen_at": "2026-03-16T10:00:00Z",
                "last_seen_at": "2026-03-16T10:00:00Z",
                "acked_at": "",
                "acked_by": "",
                "resolved_at": "",
                "resolved_by": "",
                "note": "",
            }
        ]
        self.ops_action_receipts = []

    def get_stats(self):
        return {"notes": 4, "papers": 2, "tags": 7, "links": 3}

    def get_paper(self, paper_id):
        if paper_id not in {"2501.00001", "2501.00002"}:
            return None
        return {
            "arxiv_id": paper_id,
            "title": "RAG for Agents",
            "authors": "A. Researcher",
            "year": 2025,
            "field": "Computer Science",
            "notes": "paper memory fallback notes",
            "translated_path": "",
            "text_path": "",
        }

    def get_note(self, note_id):
        if note_id == "paper:2501.00001":
            return {
                "id": note_id,
                "title": "[논문] RAG for Agents",
                "content": "# RAG for Agents\n\n## 요약\n\nRAG memory card summary.\n",
                "metadata": "{\"arxiv_id\": \"2501.00001\", \"quality_flag\": \"ok\"}",
                "source_type": "paper",
            }
        return None

    def list_claims_by_note(self, note_id, limit=50):  # noqa: ANN001
        _ = limit
        if note_id == "paper:2501.00001":
            return [
                {
                    "claim_id": "claim_1",
                    "claim_text": "The paper improves retrieval-grounded responses.",
                    "confidence": 0.91,
                }
            ]
        return []

    def list_claims_by_entity(self, entity_id, limit=50):  # noqa: ANN001
        _ = limit
        if entity_id == "paper:2501.00001":
            return [
                {
                    "claim_id": "claim_1",
                    "claim_text": "The paper improves retrieval-grounded responses.",
                    "confidence": 0.91,
                }
            ]
        return []

    def get_claim(self, claim_id):
        if claim_id == "claim_1":
            return {
                "claim_id": "claim_1",
                "claim_text": "The paper improves retrieval-grounded responses.",
                "confidence": 0.91,
            }
        return None

    def get_paper_concepts(self, paper_id):
        if paper_id == "2501.00001":
            return [{"canonical_name": "RAG", "entity_id": "concept_rag"}]
        return []

    def list_document_memory_units(self, document_id, limit=200):  # noqa: ANN001
        return list(self.document_memory_units.get(str(document_id), []))[:limit]

    def replace_document_memory_units(self, *, document_id, units):  # noqa: ANN001
        stored = [dict(unit) for unit in units]
        self.document_memory_units[str(document_id)] = stored
        return stored

    def get_document_memory_summary(self, document_id):  # noqa: ANN001
        for unit in self.document_memory_units.get(str(document_id), []):
            if str(unit.get("unit_type") or "") == "document_summary":
                return dict(unit)
        return None

    def upsert_paper_memory_card(self, *, card):
        payload = dict(card)
        payload.setdefault("created_at", "2026-03-17T00:00:00Z")
        payload.setdefault("updated_at", "2026-03-17T00:00:00Z")
        self.paper_memory_cards[payload["paper_id"]] = payload
        return payload

    def get_paper_memory_card(self, paper_id):
        return self.paper_memory_cards.get(paper_id)

    def search_paper_memory_cards(self, query, limit=10):  # noqa: ANN001
        items = list(self.paper_memory_cards.values())
        query_text = str(query).lower()
        matched = [
            item
            for item in items
            if query_text in str(item.get("search_text", "")).lower()
            or query_text in str(item.get("title", "")).lower()
        ]
        return matched[:limit]

    def list_ops_actions(self, status=None, scope=None, limit=50):
        items = list(self.ops_actions)
        if status:
            items = [item for item in items if item["status"] == status]
        if scope:
            items = [item for item in items if item["scope"] == scope]
        return items[:limit]

    def count_ops_actions(self):
        counts = {"pending": 0, "acked": 0, "resolved": 0, "total": 0}
        for item in self.ops_actions:
            counts[item["status"]] += 1
        counts["total"] = counts["pending"] + counts["acked"] + counts["resolved"]
        return counts

    def set_ops_action_status(self, action_id, *, status, actor="", note="", changed_at=None):
        _ = changed_at
        for item in self.ops_actions:
            if item["action_id"] != action_id:
                continue
            item["status"] = status
            if status == "acked":
                item["acked_by"] = actor
                item["acked_at"] = "2026-03-16T11:00:00Z"
            if status == "resolved":
                item["resolved_by"] = actor
                item["resolved_at"] = "2026-03-16T12:00:00Z"
            item["note"] = note
            return item
        return None

    def get_ops_action(self, action_id):
        for item in self.ops_actions:
            if item["action_id"] == action_id:
                return item
        return None

    def create_ops_action_receipt(self, **kwargs):
        receipt = {
            "receipt_id": f"ops_receipt_{len(self.ops_action_receipts) + 1}",
            "action_id": kwargs.get("action_id", ""),
            "executed_at": kwargs.get("executed_at", "2026-03-16T10:30:00Z"),
            "mode": kwargs.get("mode", "async"),
            "status": kwargs.get("status", "started"),
            "runner": kwargs.get("runner", "mcp"),
            "command": kwargs.get("command", ""),
            "args_json": list(kwargs.get("args", [])),
            "mcp_job_id": kwargs.get("mcp_job_id", ""),
            "result_summary": kwargs.get("result_summary", ""),
            "error_summary": kwargs.get("error_summary", ""),
            "artifact_json": dict(kwargs.get("artifact", {})),
            "actor": kwargs.get("actor", ""),
            "updated_at": kwargs.get("executed_at", "2026-03-16T10:30:00Z"),
        }
        self.ops_action_receipts.insert(0, receipt)
        return receipt

    def update_ops_action_receipt(self, receipt_id, **kwargs):
        for item in self.ops_action_receipts:
            if item["receipt_id"] != receipt_id:
                continue
            if "status" in kwargs:
                item["status"] = kwargs["status"]
            if "mcp_job_id" in kwargs:
                item["mcp_job_id"] = kwargs["mcp_job_id"]
            if "result_summary" in kwargs:
                item["result_summary"] = kwargs["result_summary"]
            if "error_summary" in kwargs:
                item["error_summary"] = kwargs["error_summary"]
            if "artifact" in kwargs:
                item["artifact_json"] = dict(kwargs["artifact"])
            if "actor" in kwargs:
                item["actor"] = kwargs["actor"]
            if "updated_at" in kwargs:
                item["updated_at"] = kwargs["updated_at"]
            return item
        return None

    def get_ops_action_receipt(self, receipt_id):
        for item in self.ops_action_receipts:
            if item["receipt_id"] == receipt_id:
                return item
        return None

    def list_ops_action_receipts(self, *, action_id, limit=20):
        return [item for item in self.ops_action_receipts if item["action_id"] == action_id][:limit]

    def get_latest_ops_action_receipt(self, action_id):
        items = self.list_ops_action_receipts(action_id=action_id, limit=1)
        return items[0] if items else None

    def list_mcp_jobs(self, status=None, tool=None, limit=20):
        _ = (status, tool, limit)
        return [
            {
                "job_id": "job_1",
                "tool": "crawl_web_ingest",
                "status": "queued",
                "classification": "P2",
                "progress": 0,
                "run_time_ms": None,
                "error": None,
                "request_json": {"topic": "rag"},
                "request_echo_json": {"topic": "rag"},
                "created_at": "2026-03-01T00:00:00Z",
                "updated_at": "2026-03-01T00:00:00Z",
                "source_refs_json": [],
                "policy_result": "allowed",
                "finished_at": None,
                "started_at": None,
            }
        ]

    def search_papers(self, query):
        _ = query
        return [
            {"arxiv_id": "2501.00001", "title": "RAG for Agents"},
            {"arxiv_id": "2501.00002", "title": "Structured Retrieval"},
        ]

    def list_foundry_sync_conflicts(self, status="pending", connector_id=None, source_filter=None, limit=50):
        _ = (status, connector_id, source_filter, limit)
        return [
            {
                "id": 1,
                "status": "pending",
                "entity_id": "concept_rag",
                "connector_id": "knowledge-hub-web",
                "source_filter": "web",
            }
        ]

    def update_foundry_sync_conflict_status(self, conflict_id, status, reviewer, resolution_note):
        _ = (conflict_id, status, reviewer, resolution_note)
        return True

    def get_foundry_sync_conflict(self, conflict_id):
        return {"id": conflict_id, "status": "approved"}

    def list_entity_merge_proposals(self, topic_slug=None, status="pending", limit=50):
        _ = (topic_slug, status, limit)
        return [
            {
                "id": 7,
                "topic_slug": "rag",
                "source_entity_id": "concept_rag_duplicate",
                "target_entity_id": "concept_rag",
                "confidence": 0.93,
                "match_method": "alias_exact",
                "status": status,
            }
        ]

    def apply_entity_merge_proposal(self, proposal_id):
        _ = proposal_id
        return True

    def reject_entity_merge_proposal(self, proposal_id):
        _ = proposal_id
        return True


class _FakeKoNoteMaterializer:
    def __init__(self, _config, sqlite_db=None):
        self.sqlite_db = sqlite_db

    def status(self, *, run_id):
        return {
            "schema": "knowledge-hub.ko-note.status.result.v1",
            "status": "completed",
            "runId": run_id,
            "counts": {"staged": 1, "source": 1, "concept": 0, "total": 1},
            "paths": {"stagingRoot": "/tmp/staging"},
            "items": [
                {
                    "id": 1,
                    "itemType": "source",
                    "status": "staged",
                    "qualityFlag": "needs_review",
                    "reviewQueue": True,
                    "reviewReasons": ["핵심 주장 섹션이 약합니다."],
                    "reviewPatchHints": ["핵심 주장 bullet을 보강하세요."],
                    "remediationAttemptCount": 1,
                    "remediationLastStatus": "remediated",
                    "remediationLastImproved": True,
                    "remediationStrategy": "section",
                    "remediationTargetSectionCount": 2,
                    "remediationPatchedSectionCount": 2,
                    "remediationPatchedLineCount": 3,
                    "remediationPreservedLineCount": 4,
                    "remediationRecommendedStrategy": "",
                    "approvalMode": "",
                    "approvalBy": "",
                    "approvalAt": "",
                    "approvalPolicyVersion": "",
                }
            ],
            "reviewQueue": {"source": {"total": 1}, "concept": {"total": 0}, "combined": {"total": 1}},
            "ts": "2026-03-06T00:00:00Z",
        }

    def report(self, *, run_id, recent_runs=10):
        _ = recent_runs
        return {
            "schema": "knowledge-hub.ko-note.report.result.v1",
            "status": "ok",
            "runId": run_id,
            "run": {
                "runId": run_id,
                "status": "completed",
                "sourceGenerated": 1,
                "conceptGenerated": 1,
                "counts": {"staged": 1, "approved": 1, "applied": 0, "rejected": 0, "source": 1, "concept": 1, "total": 2},
                "quality": {
                    "concept": {"ok": 1, "needs_review": 0, "reject": 0, "unscored": 0, "total": 1},
                    "source": {"ok": 0, "needs_review": 1, "reject": 0, "unscored": 0, "total": 1},
                    "combined": {"ok": 1, "needs_review": 1, "reject": 0, "unscored": 0, "total": 2},
                },
                "reviewQueue": {"source": {"total": 1}, "concept": {"total": 0}, "combined": {"total": 1}},
                "remediation": {"attempted": 1, "improved": 1, "unchanged": 0, "failed": 0, "regressed": 0, "recommendedFull": 0},
                "autoApproved": {"source": 0, "concept": 1, "total": 1},
                "approvedFromReview": {"source": 1, "concept": 0, "total": 1},
                "warnings": [],
            },
            "recentRuns": [
                {
                    "runId": run_id,
                    "status": "completed",
                    "sourceGenerated": 1,
                    "conceptGenerated": 1,
                    "approvedCount": 1,
                    "rejectedCount": 0,
                    "autoApproved": 1,
                    "approvedFromReview": 1,
                    "reviewQueued": 1,
                    "updatedAt": "2026-03-06T00:00:00Z",
                    "warnings": [],
                }
            ],
            "recentSummary": {
                "totalRuns": 1,
                "sourceGenerated": 1,
                "conceptGenerated": 1,
                "approved": 1,
                "rejected": 0,
                "autoApproved": 1,
                "approvedFromReview": 1,
                "reviewQueued": 1,
            },
            "alerts": [
                {
                    "code": "ko_note_review_queue_pending",
                    "severity": "warning",
                    "scope": "ko_note",
                    "summary": "review queue에 1개 항목이 남아 있습니다.",
                    "metric": "reviewQueue.combined.total",
                    "observed": 1,
                    "threshold": "> 0",
                }
            ],
            "recommendedActions": [
                {
                    "actionType": "inspect_review_queue",
                    "scope": "ko_note",
                    "summary": "review queue 항목을 먼저 확인하세요.",
                    "reasonCodes": ["ko_note_review_queue_pending"],
                    "command": "khub",
                    "args": ["labs", "crawl", "ko-note-review-list", "--run-id", run_id],
                }
            ],
            "warnings": [],
            "ts": "2026-03-06T00:00:00Z",
        }

    def review_list(self, *, run_id, item_type="all", quality_flag="all", limit=50):
        _ = (item_type, quality_flag, limit)
        return {
            "schema": "knowledge-hub.ko-note.review.list.result.v1",
            "status": "ok",
            "runId": run_id,
            "itemType": item_type,
            "qualityFlag": quality_flag,
            "counts": {"total": 1, "source": 1, "concept": 0, "needs_review": 1, "reject": 0, "unscored": 0},
            "items": [
                {
                    "id": 1,
                    "itemType": "source",
                    "status": "staged",
                    "qualityFlag": "needs_review",
                    "reviewQueue": True,
                    "reviewReasons": ["핵심 주장 섹션이 약합니다."],
                    "reviewPatchHints": ["핵심 주장 bullet을 보강하세요."],
                    "remediationAttemptCount": 1,
                    "remediationLastStatus": "remediated",
                    "remediationLastImproved": True,
                    "remediationStrategy": "section",
                    "remediationTargetSectionCount": 2,
                    "remediationPatchedSectionCount": 2,
                    "remediationPatchedLineCount": 3,
                    "remediationPreservedLineCount": 4,
                    "remediationRecommendedStrategy": "",
                    "approvalMode": "",
                    "approvalBy": "",
                    "approvalAt": "",
                    "approvalPolicyVersion": "",
                }
            ],
            "ts": "2026-03-06T00:00:00Z",
        }

    def review_approve(self, *, item_id, reviewer="mcp-user", note=""):
        _ = note
        return {
            "schema": "knowledge-hub.ko-note.review.result.v1",
            "status": "ok",
            "itemId": item_id,
            "runId": "ko_note_run_1",
            "itemType": "source",
            "decision": "approved",
            "qualityFlag": "needs_review",
            "review": {"decision": {"status": "approved", "reviewer": reviewer}},
            "ts": "2026-03-06T00:00:00Z",
        }

    def review_reject(self, *, item_id, reviewer="mcp-user", note=""):
        _ = note
        return {
            "schema": "knowledge-hub.ko-note.review.result.v1",
            "status": "ok",
            "itemId": item_id,
            "runId": "ko_note_run_1",
            "itemType": "source",
            "decision": "rejected",
            "qualityFlag": "reject",
            "review": {"decision": {"status": "rejected", "reviewer": reviewer}},
            "ts": "2026-03-06T00:00:00Z",
        }


class _FakeKoNoteEnricher:
    def __init__(self, _config, sqlite_db=None):
        self.sqlite_db = sqlite_db

    def remediate(self, *, run_id, item_type="all", quality_flag="all", item_id=0, limit=50, strategy="section", allow_external=False, llm_mode="auto"):
        _ = (limit, allow_external, llm_mode)
        return {
            "schema": "knowledge-hub.ko-note.remediate.result.v1",
            "status": "ok",
            "runId": "ko_note_remediate_1",
            "sourceRunId": run_id,
            "itemType": item_type,
            "qualityFlag": quality_flag,
            "itemId": item_id,
            "strategy": strategy,
            "attempted": 1,
            "remediated": 1,
            "improved": 1,
            "unchanged": 0,
            "failed": 0,
            "items": [{
                "id": 1,
                "itemType": "source",
                "status": "staged",
                "beforeQualityFlag": "needs_review",
                "afterQualityFlag": "ok",
                "improved": True,
                "changed": True,
                "strategy": strategy,
                "targetSections": ["top_claims", "limitations"],
                "patchedSections": ["top_claims", "limitations"],
                "preservedSectionsCount": 12,
                "patchedLineCount": 3,
                "preservedLineCount": 4,
                "recommendedStrategy": "",
            }],
            "warnings": [],
            "ts": "2026-03-06T00:00:00Z",
        }


def _setup_fakes(module):
    module.SERVER_STATE.config = object()
    module.SERVER_STATE.sqlite_db = _FakeSQLiteDB()
    module.SERVER_STATE.searcher = _FakeSearcher()
    module.SERVER_STATE.searcher.sqlite_db = module.SERVER_STATE.sqlite_db
    module.SERVER_STATE.LearningCoachService = _FakeLearningCoachService
    module.SERVER_STATE.WebIngestService = _FakeWebIngestService
    module.SERVER_STATE.KoNoteMaterializer = _FakeKoNoteMaterializer
    module.SERVER_STATE.KoNoteEnricher = _FakeKoNoteEnricher
    module.SERVER_STATE._run_async_tool = None
    module.SERVER_STATE.initialize = lambda: None
    module.SERVER_STATE.initialize_core_only = lambda: None
    import knowledge_hub.mcp.handlers.crawl as crawl_handlers
    import knowledge_hub.mcp.handlers.search as search_handlers

    crawl_handlers.build_ko_note_report = (
        lambda sqlite_db, *, run_id, recent_runs: _FakeKoNoteMaterializer(module.SERVER_STATE.config, sqlite_db=sqlite_db).report(
            run_id=run_id,
            recent_runs=recent_runs,
        )
    )
    search_handlers.build_rag_ops_report = (
        lambda sqlite_db, *, limit, days: module.SERVER_STATE.searcher.build_ops_report(limit=limit, days=days)
    )


def test_tool_specs_accept_memory_mode_contract():
    tool_specs = _import_tool_specs()
    tools = tool_specs.build_tools(profile="default")
    ask_tool = next(tool for tool in tools if tool.name == "ask_knowledge")
    paper_lookup_tool = next(tool for tool in tools if tool.name == "paper_lookup_and_summarize")

    assert ask_tool.inputSchema["properties"]["memory_route_mode"]["enum"] == ["off", "compat", "on", "prefilter"]
    assert "ask retrieval memory prefilter/prior mode" in ask_tool.inputSchema["properties"]["memory_route_mode"]["description"]
    assert ask_tool.inputSchema["properties"]["paper_memory_mode"]["enum"] == ["off", "compat", "on", "prefilter"]
    assert "paper-source memory prefilter mode" in ask_tool.inputSchema["properties"]["paper_memory_mode"]["description"]
    assert paper_lookup_tool.inputSchema["properties"]["memory_route_mode"]["enum"] == ["off", "compat", "on", "prefilter"]
    assert paper_lookup_tool.inputSchema["properties"]["paper_memory_mode"]["enum"] == ["off", "compat", "on", "prefilter"]


def test_search_knowledge_requires_query_and_returns_result_shape():
    module = _import_mcp_server()
    _setup_fakes(module)

    failed = _decode_response(asyncio.run(module.call_tool("search_knowledge", {})))
    assert failed["status"] == "failed"
    assert "query" in failed["payload"]["error"]

    ok = _decode_response(asyncio.run(module.call_tool("search_knowledge", {"query": "rag"})))
    assert ok["status"] == "ok"
    assert ok["payload"]["result_count"] == 1
    assert isinstance(ok["payload"]["results"], list)
    assert ok["payload"]["results"][0]["title"] == "RAG Note"
    assert ok["payload"]["results"][0]["quality_flag"] == "ok"
    assert ok["payload"]["results"][0]["reference_tier"] == "specialist"
    assert ok["payload"]["results"][0]["normalized_source_type"] == "vault"
    assert isinstance(ok["payload"]["results"][0]["top_ranking_signals"], list)
    assert ok["payload"]["runtimeDiagnostics"]["schema"] == "knowledge-hub.runtime.diagnostics.v1"


def test_build_task_context_returns_schema_valid_payload(tmp_path):
    module = _import_mcp_server()
    _setup_fakes(module)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "AGENTS.md").write_text("- Preserve boundaries\n", encoding="utf-8")
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "agent.ts").write_text("export const enabled = true;\n", encoding="utf-8")

    ok = _decode_response(
        asyncio.run(
            module.call_tool(
                "build_task_context",
                {"goal": "Implement src/agent.ts update", "repo_path": str(repo)},
            )
        )
    )
    assert ok["status"] == "ok"
    assert ok["payload"]["schema"] == "knowledge-hub.task-context.result.v1"
    assert ok["payload"]["mode"] == "coding"
    assert ok["payload"]["workspace_files"][0]["source_type"] == "project"
    assert ok["payload"]["gateway"]["surface"] == "task_context"
    assert ok["payload"]["gateway"]["mode"] == "context"
    assert ok["payload"]["runtimeDiagnostics"]["schema"] == "knowledge-hub.runtime.diagnostics.v1"


def test_learn_map_requires_topic_and_returns_trunks_branches(monkeypatch):
    _enable_labs_profile(monkeypatch)
    module = _import_mcp_server()
    _setup_fakes(module)

    failed = _decode_response(asyncio.run(module.call_tool("learn_map", {})))
    assert failed["status"] == "failed"
    assert "topic" in failed["payload"]["error"]

    ok = _decode_response(asyncio.run(module.call_tool("learn_map", {"topic": "rag"})))
    assert ok["status"] == "ok"
    assert isinstance(ok["payload"]["trunks"], list)
    assert isinstance(ok["payload"]["branches"], list)
    assert ok["payload"]["trunks"][0]["canonical_id"] == "c1"


def test_learning_start_resume_explain_and_checkpoint_tools(monkeypatch):
    _enable_labs_profile(monkeypatch)
    module = _import_mcp_server()
    _setup_fakes(module)

    start = _decode_response(asyncio.run(module.call_tool("learning_start_or_resume_topic", {"topic": "rag"})))
    assert start["status"] == "ok"
    assert start["payload"]["session_id"] == "session-rag"

    state = _decode_response(asyncio.run(module.call_tool("learning_get_session_state", {"topic": "rag"})))
    assert state["status"] == "ok"
    assert state["payload"]["conceptStates"][0]["conceptId"] == "c1"

    explain = _decode_response(
        asyncio.run(module.call_tool("learning_explain_topic", {"topic": "rag", "question": "RAG란?"}))
    )
    assert explain["status"] == "ok"
    assert explain["payload"]["evidence_refs"] == ["RAG Note"]

    checkpoint = _decode_response(
        asyncio.run(
            module.call_tool(
                "learning_checkpoint",
                {"topic": "rag", "session_id": "session-rag", "summary": "Need work", "unknown_items": ["Concept 1"]},
            )
        )
    )
    assert checkpoint["status"] == "ok"
    assert checkpoint["payload"]["saved"] is True


def test_learn_reinforce_requires_topic_and_session_id(monkeypatch):
    _enable_labs_profile(monkeypatch)
    module = _import_mcp_server()
    _setup_fakes(module)

    failed_topic = _decode_response(asyncio.run(module.call_tool("learn_reinforce", {"session_id": "s1"})))
    assert failed_topic["status"] == "failed"
    assert "topic" in failed_topic["payload"]["error"]

    failed_session = _decode_response(asyncio.run(module.call_tool("learn_reinforce", {"topic": "rag"})))
    assert failed_session["status"] == "failed"
    assert "session_id" in failed_session["payload"]["error"]

    ok = _decode_response(
        asyncio.run(module.call_tool("learn_reinforce", {"topic": "rag", "session_id": "s1"}))
    )
    assert ok["status"] == "ok"
    assert isinstance(ok["payload"]["actions"], list)


def test_get_hub_stats_without_arguments():
    module = _import_mcp_server()
    _setup_fakes(module)

    result = _decode_response(asyncio.run(module.call_tool("get_hub_stats", {})))
    assert result["status"] == "ok"
    payload = result["payload"]
    assert payload["notes"] == 4
    assert payload["papers"] == 2
    assert payload["vector_documents"] == 3


def test_mcp_job_list_response_shape():
    module = _import_mcp_server()
    _setup_fakes(module)

    result = _decode_response(asyncio.run(module.call_tool("mcp_job_list", {"limit": 5})))
    assert result["status"] == "ok"
    jobs = result["payload"]["jobs"]
    assert isinstance(jobs, list)
    assert jobs
    assert {"jobId", "tool", "status", "classification", "request", "requestEcho"} <= set(jobs[0].keys())


def test_ask_knowledge_requires_question_and_returns_answer_shape():
    module = _import_mcp_server()
    _setup_fakes(module)

    failed = _decode_response(asyncio.run(module.call_tool("ask_knowledge", {})))
    assert failed["status"] == "failed"
    assert "question" in failed["payload"]["error"]

    ok = _decode_response(asyncio.run(module.call_tool("ask_knowledge", {"question": "RAG란?"})))
    assert ok["status"] == "ok"
    assert ok["payload"]["answer"] == "RAG 재작성 답변"
    assert isinstance(ok["payload"]["sources"], list)
    assert ok["payload"]["sources"][0]["title"] == "RAG Note"
    assert ok["payload"]["sources"][0]["quality_flag"] == "ok"
    assert ok["payload"]["sources"][0]["ranking_signals"]["reference_role"] == "glossary_reference"
    assert ok["payload"]["sources"][0]["normalized_source_type"] == "vault"
    assert isinstance(ok["payload"]["sources"][0]["top_ranking_signals"], list)
    assert ok["payload"]["answer_signals"]["preferred_sources"] == 1
    assert ok["payload"]["answer_verification"]["status"] == "verified"
    assert ok["payload"]["answer_verification"]["claims"][0]["verdict"] == "supported"
    assert ok["payload"]["answer_rewrite"]["applied"] is True
    assert ok["payload"]["initial_answer_verification"]["unsupportedClaimCount"] == 1
    assert ok["payload"]["warnings"][0].startswith("answer verification caution:")

    paper_prefilter = _decode_response(
        asyncio.run(
            module.call_tool(
                "ask_knowledge",
                {"question": "RAG란?", "source": "paper", "paper_memory_mode": "prefilter"},
            )
        )
    )
    assert paper_prefilter["status"] == "ok"
    assert paper_prefilter["payload"]["paper_memory_prefilter"]["applied"] is True
    assert paper_prefilter["payload"]["paper_memory_prefilter"]["matchedPaperIds"] == ["2501.00001"]


def test_ask_knowledge_is_local_only_and_exposes_memory_contract(monkeypatch):
    module = _import_mcp_server()
    _setup_fakes(module)

    import knowledge_hub.mcp.handlers.search as search_handlers

    captured: dict[str, object] = {}

    def _fake_generate(searcher, question, **kwargs):  # noqa: ANN001
        _ = (searcher, question)
        captured.update(kwargs)
        return {
            "answer": "contract answer",
            "sources": [],
            "citations": [],
            "warnings": [],
        }

    monkeypatch.setattr(search_handlers, "_generate_answer_compat", _fake_generate)

    ok = _decode_response(
        asyncio.run(
            module.call_tool(
                "ask_knowledge",
                {
                    "question": "RAG란?",
                    "source": "paper",
                    "memory_route_mode": "prefilter",
                    "paper_memory_mode": "prefilter",
                },
            )
        )
    )

    assert ok["status"] == "ok"
    assert captured["allow_external"] is False
    assert captured["memory_route_mode"] == "prefilter"
    assert captured["paper_memory_mode"] == "prefilter"
    assert ok["payload"]["allowExternal"] is False
    assert ok["payload"]["allow_external"] is False
    assert ok["payload"]["externalPolicy"]["policyMode"] == "local-only"
    assert ok["payload"]["externalPolicy"]["decisionSource"] == "mcp_default_local_only"
    assert ok["payload"]["evidencePacketContract"] == {}
    assert ok["payload"]["answerContract"] == {}
    assert ok["payload"]["verificationVerdict"] == {}
    assert ok["payload"]["memory_route"]["contractRole"] == "ask_retrieval_memory_prefilter"
    assert ok["payload"]["memory_route"]["requestedMode"] == "prefilter"
    assert ok["payload"]["memory_route"]["effectiveMode"] == "compat"
    assert ok["payload"]["memory_route"]["aliasDeprecated"] is True
    assert ok["payload"]["memory_prefilter"]["contractRole"] == "retrieval_memory_prefilter"
    assert ok["payload"]["paper_memory_prefilter"]["contractRole"] == "paper_source_memory_prefilter"


def test_ko_note_review_tools_are_exposed_and_return_payloads(monkeypatch):
    _enable_labs_profile(monkeypatch)
    module = _import_mcp_server()
    _setup_fakes(module)

    status = _decode_response(asyncio.run(module.call_tool("ko_note_status", {"run_id": "ko_note_run_1"})))
    assert status["status"] == "ok"
    assert status["payload"]["reviewQueue"]["combined"]["total"] == 1

    review_list = _decode_response(asyncio.run(module.call_tool("ko_note_review_list", {"run_id": "ko_note_run_1"})))
    assert review_list["status"] == "ok"
    assert review_list["payload"]["items"][0]["remediationAttemptCount"] == 1

    approved = _decode_response(asyncio.run(module.call_tool("ko_note_review_approve", {"item_id": 1})))
    assert approved["status"] == "ok"
    assert approved["payload"]["decision"] == "approved"

    rejected = _decode_response(asyncio.run(module.call_tool("ko_note_review_reject", {"item_id": 1})))
    assert rejected["status"] == "ok"
    assert rejected["payload"]["decision"] == "rejected"

    async def _fake_run_async_tool(name, request_echo, sync_job, started_at=None):  # noqa: ANN001
        _ = (name, request_echo, started_at)
        payload = await sync_job()
        return "job_ko_note_1", {"payload": payload}

    monkeypatch.setattr(module.SERVER_STATE, "_run_async_tool", _fake_run_async_tool, raising=False)
    remediated = _decode_response(asyncio.run(module.call_tool("ko_note_remediate", {"run_id": "ko_note_run_1", "strategy": "section"})))
    assert remediated["status"] == "queued"
    assert remediated["jobId"] == "job_ko_note_1"
    assert remediated["payload"]["attempted"] == 1
    assert remediated["payload"]["strategy"] == "section"


def test_ops_report_tools_are_exposed_and_return_payloads(monkeypatch):
    _enable_labs_profile(monkeypatch)
    module = _import_mcp_server()
    _setup_fakes(module)

    ko_note_report = _decode_response(asyncio.run(module.call_tool("ko_note_report", {"run_id": "ko_note_run_1"})))
    assert ko_note_report["status"] == "ok"
    assert ko_note_report["payload"]["run"]["autoApproved"]["concept"] == 1
    assert ko_note_report["payload"]["alerts"][0]["code"] == "ko_note_review_queue_pending"

    rag_report = _decode_response(asyncio.run(module.call_tool("rag_report", {"limit": 10, "days": 7})))
    assert rag_report["status"] == "ok"
    assert rag_report["payload"]["counts"]["needsCaution"] == 1
    assert rag_report["payload"]["samples"][0]["finalAnswerSource"] == "rewritten"
    assert rag_report["payload"]["recommendedActions"][0]["actionType"] == "inspect_verification_routes"


def test_search_papers_requires_query_and_returns_items(monkeypatch):
    _enable_labs_profile(monkeypatch)
    module = _import_mcp_server()
    _setup_fakes(module)

    failed = _decode_response(asyncio.run(module.call_tool("search_papers", {})))
    assert failed["status"] == "failed"
    assert "query" in failed["payload"]["error"]

    ok = _decode_response(asyncio.run(module.call_tool("search_papers", {"query": "rag"})))
    assert ok["status"] == "ok"
    assert ok["payload"]["paper_count"] == 2
    assert isinstance(ok["payload"]["items"], list)
    assert ok["payload"]["items"][0]["arxiv_id"] == "2501.00001"


def test_paper_memory_tools_are_exposed_and_return_payloads(monkeypatch):
    _enable_labs_profile(monkeypatch)
    module = _import_mcp_server()
    _setup_fakes(module)

    built = _decode_response(asyncio.run(module.call_tool("build_paper_memory", {"paper_id": "2501.00001"})))
    assert built["status"] == "ok"
    assert built["payload"]["schema"] == "knowledge-hub.paper-memory.build.result.v1"
    assert built["payload"]["items"][0]["paperId"] == "2501.00001"
    assert "paper" not in built["payload"]["items"][0]
    assert validate_payload(built["payload"], built["payload"]["schema"], strict=True).ok
    units = module.SERVER_STATE.sqlite_db.list_document_memory_units("paper:2501.00001", limit=20)
    assert units
    assert any(unit.get("unit_type") == "document_summary" for unit in units)

    shown = _decode_response(asyncio.run(module.call_tool("get_paper_memory_card", {"paper_id": "2501.00001"})))
    assert shown["status"] == "ok"
    assert shown["payload"]["item"]["paperId"] == "2501.00001"
    assert shown["payload"]["item"]["paper"]["paperId"] == "2501.00001"
    assert validate_payload(shown["payload"], shown["payload"]["schema"], strict=True).ok

    searched = _decode_response(asyncio.run(module.call_tool("search_paper_memory", {"query": "RAG memory"})))
    assert searched["status"] == "ok"
    assert searched["payload"]["count"] == 1
    assert searched["payload"]["items"][0]["claimRefs"] == ["claim_1"]
    assert searched["payload"]["items"][0]["claims"][0]["claimId"] == "claim_1"
    assert validate_payload(searched["payload"], searched["payload"]["schema"], strict=True).ok


def test_paper_memory_tools_handle_missing_and_empty_paths(monkeypatch):
    _enable_labs_profile(monkeypatch)
    module = _import_mcp_server()
    _setup_fakes(module)

    missing = _decode_response(asyncio.run(module.call_tool("get_paper_memory_card", {"paper_id": "missing"})))
    assert missing["status"] == "failed"
    assert missing["payload"]["item"] == {}
    assert validate_payload(missing["payload"], missing["payload"]["schema"], strict=True).ok

    empty = _decode_response(asyncio.run(module.call_tool("search_paper_memory", {"query": "no matches here"})))
    assert empty["status"] == "ok"
    assert empty["payload"]["count"] == 0
    assert empty["payload"]["items"] == []
    assert validate_payload(empty["payload"], empty["payload"]["schema"], strict=True).ok

    failed_build = _decode_response(asyncio.run(module.call_tool("build_paper_memory", {"paper_id": "missing"})))
    assert failed_build["status"] == "failed"
    assert "paper not found" in failed_build["payload"]["error"]


def test_paper_lookup_and_summarize_requires_identifier_or_query(monkeypatch):
    module = _import_mcp_server()
    _setup_fakes(module)

    failed = _decode_response(asyncio.run(module.call_tool("paper_lookup_and_summarize", {})))
    assert failed["status"] == "failed"
    assert "paper_id 또는 query" in failed["payload"]["error"]


def test_paper_lookup_and_summarize_by_query_returns_paper_and_summary(monkeypatch):
    module = _import_mcp_server()
    _setup_fakes(module)

    import knowledge_hub.papers.discoverer as discoverer

    monkeypatch.setattr(
        discoverer,
        "get_paper_detail",
        lambda paper_id: {
            "paper_id": paper_id,
            "arxiv_id": paper_id,
            "title": "RAG for Agents",
            "authors": ["A. Researcher"],
            "year": 2025,
            "abstract": "Test abstract",
            "citation_count": 12,
            "fields_of_study": ["Computer Science"],
        },
    )

    ok = _decode_response(
        asyncio.run(
            module.call_tool(
                "paper_lookup_and_summarize",
                {"query": "RAG for Agents", "paper_memory_mode": "prefilter"},
            )
        )
    )
    assert ok["status"] == "ok"
    assert ok["payload"]["matched"] is True
    assert ok["payload"]["paper"]["arxiv_id"] == "2501.00001"
    assert ok["payload"]["paper"]["title"] == "RAG for Agents"
    assert ok["payload"]["summary"]["answer"] == "RAG 재작성 답변"
    assert ok["payload"]["summary"]["external_calls"] is False
    assert ok["payload"]["summary"]["paper_memory_prefilter"]["applied"] is True


def test_paper_lookup_and_summarize_scopes_answer_generation_to_selected_paper(monkeypatch):
    module = _import_mcp_server()
    _setup_fakes(module)

    import knowledge_hub.papers.discoverer as discoverer

    monkeypatch.setattr(
        discoverer,
        "get_paper_detail",
        lambda paper_id: {
            "paper_id": paper_id,
            "arxiv_id": paper_id,
            "title": "RAG for Agents",
        },
    )

    captured: dict[str, object] = {}
    original_generate_answer = module.SERVER_STATE.searcher.generate_answer

    def _wrapped_generate_answer(*args, **kwargs):
        captured["metadata_filter"] = dict(kwargs.get("metadata_filter") or {})
        return original_generate_answer(*args, **kwargs)

    monkeypatch.setattr(module.SERVER_STATE.searcher, "generate_answer", _wrapped_generate_answer)

    ok = _decode_response(
        asyncio.run(
            module.call_tool(
                "paper_lookup_and_summarize",
                {"query": "RAG for Agents", "paper_memory_mode": "prefilter"},
            )
        )
    )

    assert ok["status"] == "ok"
    assert captured["metadata_filter"] == {"source_type": "paper", "arxiv_id": "2501.00001"}
    assert ok["payload"]["summary"]["paper_memory_prefilter"]["matchedPaperIds"] == ["2501.00001"]


def test_default_tool_profile_exposes_public_retrieval_core(monkeypatch):
    monkeypatch.delenv("KHUB_MCP_PROFILE", raising=False)
    module = _import_mcp_server()

    tools = asyncio.run(module.list_tools())
    names = {tool.name for tool in tools}
    assert names == {
        "search_knowledge",
        "ask_knowledge",
        "build_task_context",
        "discover_and_ingest",
        "get_paper_detail",
        "paper_lookup_and_summarize",
        "get_hub_stats",
        "mcp_job_status",
        "mcp_job_list",
        "mcp_job_cancel",
    }


def test_default_tool_profile_blocks_hidden_direct_calls(monkeypatch):
    monkeypatch.delenv("KHUB_MCP_PROFILE", raising=False)
    module = _import_mcp_server()
    _setup_fakes(module)

    blocked = _decode_response(asyncio.run(module.call_tool("run_agentic_query", {"goal": "RAG 비교"})))
    assert blocked["status"] == "blocked"
    assert blocked["payload"]["profile"] == "default"
    assert "KHUB_MCP_PROFILE=labs" in blocked["payload"]["hint"]


def test_foundry_conflict_list_response_shape(monkeypatch):
    _enable_labs_profile(monkeypatch)
    module = _import_mcp_server()
    _setup_fakes(module)

    ok = _decode_response(asyncio.run(module.call_tool("foundry_conflict_list", {"limit": 10})))
    assert ok["status"] == "ok"
    assert ok["payload"]["schema"] == "knowledge-hub.foundry.conflict.list.result.v1"


def test_entity_merge_tools_response_shape(monkeypatch):
    _enable_labs_profile(monkeypatch)
    module = _import_mcp_server()
    _setup_fakes(module)

    listed = _decode_response(asyncio.run(module.call_tool("entity_merge_list", {"topic": "rag", "limit": 10})))
    assert listed["status"] == "ok"
    assert listed["payload"]["schema"] == "knowledge-hub.entity.merge.list.result.v1"
    assert listed["payload"]["count"] == 1
    assert listed["payload"]["items"][0]["target_entity_id"] == "concept_rag"

    applied = _decode_response(asyncio.run(module.call_tool("entity_merge_apply", {"id": 7})))
    assert applied["status"] == "ok"
    assert applied["payload"]["applied"] is True

    rejected = _decode_response(asyncio.run(module.call_tool("entity_merge_reject", {"id": 7})))
    assert rejected["status"] == "ok"
    assert rejected["payload"]["rejected"] is True


def test_list_tools_contains_core_contracts(monkeypatch):
    monkeypatch.setenv("KHUB_MCP_PROFILE", "default")
    module = _import_mcp_server()

    tools = asyncio.run(module.list_tools())
    names = {tool.name for tool in tools}
    assert "search_knowledge" in names
    assert "ask_knowledge" in names
    assert "build_task_context" in names
    assert "discover_and_ingest" in names
    assert "get_paper_detail" in names
    assert "paper_lookup_and_summarize" in names
    assert "get_hub_stats" in names
    assert "run_agentic_query" not in names
    assert "learning_start_or_resume_topic" not in names
    assert "learning_get_session_state" not in names
    assert "learning_explain_topic" not in names
    assert "learning_checkpoint" not in names
    assert "crawl_web_ingest" not in names
    assert "crawl_youtube_ingest" not in names
    assert "build_paper_memory" not in names
    assert "get_paper_memory_card" not in names
    assert "search_paper_memory" not in names
    assert "search_papers" not in names
    assert "get_paper_citations" not in names
    assert "learn_map" not in names
    assert "belief_list" not in names
    assert "ontology_profile_list" not in names
    assert "ko_note_status" not in names
    assert "ops_action_list" not in names
    assert "rag_report" not in names
    assert "learn_reinforce" not in names
    assert "mcp_job_list" in names
    assert "mcp_job_status" in names
    assert "mcp_job_cancel" in names
    assert "crawl_pipeline_run" not in names
    assert "transform_run" not in names
    assert "ask_graph" not in names
    assert "notebook_workbench_chat" not in names


def test_list_tools_includes_labs_profile(monkeypatch):
    monkeypatch.setenv("KHUB_MCP_PROFILE", "labs")
    module = _import_mcp_server()

    tools = asyncio.run(module.list_tools())
    names = {tool.name for tool in tools}
    assert "search_knowledge" in names
    assert "ask_knowledge" in names
    assert "build_task_context" in names
    assert "run_agentic_query" in names
    assert "learning_start_or_resume_topic" in names
    assert "learning_get_session_state" in names
    assert "learning_explain_topic" in names
    assert "learning_checkpoint" in names
    assert "build_paper_memory" in names
    assert "get_paper_memory_card" in names
    assert "search_paper_memory" in names
    assert "search_papers" in names
    assert "learn_map" in names
    assert "belief_list" in names
    assert "ontology_profile_list" in names
    assert "ko_note_status" in names
    assert "ops_action_list" in names
    assert "rag_report" in names
    assert "mcp_job_list" in names
    assert "crawl_web_ingest" in names
    assert "crawl_pipeline_run" in names
    assert "crawl_youtube_ingest" in names
    assert "transform_run" in names
    assert "ask_graph" in names
    assert "notebook_workbench_chat" in names


def test_all_tool_profile_allows_hidden_direct_calls(monkeypatch):
    monkeypatch.setenv("KHUB_MCP_PROFILE", "all")
    module = _import_mcp_server()
    _setup_fakes(module)

    tools = asyncio.run(module.list_tools())
    names = {tool.name for tool in tools}
    assert "run_agentic_query" in names

    failed = _decode_response(asyncio.run(module.call_tool("run_agentic_query", {})))
    assert failed["status"] == "failed"
    assert "goal" in failed["payload"]["error"]


def test_crawl_youtube_ingest_returns_schema_backed_payload(monkeypatch):
    _enable_labs_profile(monkeypatch)
    module = _import_mcp_server()
    _setup_fakes(module)
    _FakeWebIngestService.last_crawl_kwargs = None

    async def _run_async_tool(tool=None, request_echo=None, sync_job=None, **kwargs):
        _ = (tool, request_echo, kwargs)
        payload = await sync_job()
        return "job_1", {"payload": payload}

    module.SERVER_STATE._run_async_tool = _run_async_tool

    ok = _decode_response(
        asyncio.run(
            module.call_tool(
                "crawl_youtube_ingest",
                {"urls": ["https://youtu.be/abc123xyz89"], "topic": "agents", "transcript_language": "ko"},
            )
        )
    )

    assert ok["status"] == "queued"
    assert ok["payload"]["schema"] == "knowledge-hub.crawl.ingest.result.v1"
    assert ok["payload"]["indexDiagnostics"]["status"] == "retry_succeeded"
    assert _FakeWebIngestService.last_crawl_kwargs["input_source"] == "youtube"
    assert _FakeWebIngestService.last_crawl_kwargs["engine"] == "youtube"
    assert _FakeWebIngestService.last_crawl_kwargs["transcript_language"] == "ko"
    assert _FakeWebIngestService.last_crawl_kwargs["index_autofix_mode"] == "youtube_single_retry"


def test_transform_and_workbench_tools_return_schema_backed_payloads(monkeypatch):
    _enable_labs_profile(monkeypatch)
    module = _import_mcp_server()
    _setup_fakes(module)

    listed = _decode_response(asyncio.run(module.call_tool("transform_list", {})))
    assert listed["status"] == "ok"
    assert listed["payload"]["schema"] == "knowledge-hub.transform.list.result.v1"

    preview = _decode_response(
        asyncio.run(
            module.call_tool(
                "transform_preview",
                {"transformation_id": "technical_summary", "query": "RAG retrieval"},
            )
        )
    )
    assert preview["status"] == "ok"
    assert preview["payload"]["schema"] == "knowledge-hub.transform.preview.result.v1"

    ask_graph = _decode_response(asyncio.run(module.call_tool("ask_graph", {"question": "RAG란?"})))
    assert ask_graph["status"] == "ok"
    assert ask_graph["payload"]["schema"] == "knowledge-hub.ask-graph.result.v1"
    assert ask_graph["payload"]["trace"]

    workbench = _decode_response(
        asyncio.run(module.call_tool("notebook_workbench_chat", {"message": "Summarize retrieval"}))
    )
    assert workbench["status"] == "ok"
    assert workbench["payload"]["schema"] == "knowledge-hub.workbench.chat.result.v1"


def test_ops_action_tools_response_shape(monkeypatch):
    _enable_labs_profile(monkeypatch)
    module = _import_mcp_server()
    _setup_fakes(module)

    listed = _decode_response(asyncio.run(module.call_tool("ops_action_list", {"limit": 10})))
    assert listed["status"] == "ok"
    assert listed["payload"]["schema"] == "knowledge-hub.ops.action.list.result.v1"
    assert listed["payload"]["count"] == 1

    acked = _decode_response(asyncio.run(module.call_tool("ops_action_ack", {"action_id": "ops_action_1", "actor": "mcp-user"})))
    assert acked["status"] == "ok"
    assert acked["payload"]["decision"] == "acked"
    assert acked["payload"]["item"]["ackedBy"] == "mcp-user"

    resolved = _decode_response(asyncio.run(module.call_tool("ops_action_resolve", {"action_id": "ops_action_1", "actor": "mcp-user"})))
    assert resolved["status"] == "ok"
    assert resolved["payload"]["decision"] == "resolved"
    assert resolved["payload"]["item"]["resolvedBy"] == "mcp-user"


def test_ops_action_execute_and_receipts_tools_response_shape(monkeypatch):
    _enable_labs_profile(monkeypatch)
    module = _import_mcp_server()
    _setup_fakes(module)

    async def _fake_run_async_tool(name, request_echo, sync_job, started_at=None, **kwargs):  # noqa: ANN001
        _ = (name, request_echo, started_at)
        if callable(kwargs.get("on_queued")):
            kwargs["on_queued"]("job_ops_action_1", "2026-03-16T10:30:00Z")
        return "job_ops_action_1", {"payload": {"message": "queued"}}

    monkeypatch.setattr(module.SERVER_STATE, "_run_async_tool", _fake_run_async_tool, raising=False)
    queued = _decode_response(
        asyncio.run(module.call_tool("ops_action_execute", {"action_id": "ops_action_1", "actor": "mcp-user"}))
    )
    assert queued["status"] == "queued"
    assert queued["jobId"] == "job_ops_action_1"
    assert queued["payload"]["schema"] == "knowledge-hub.ops.action.execute.result.v1"
    assert queued["payload"]["receipt"]["status"] == "started"

    listed = _decode_response(asyncio.run(module.call_tool("ops_action_receipts", {"action_id": "ops_action_1", "limit": 10})))
    assert listed["status"] == "ok"
    assert listed["payload"]["schema"] == "knowledge-hub.ops.action.receipts.result.v1"
    assert listed["payload"]["count"] == 1


def test_ops_action_execute_auto_resolves_agent_writeback_request_after_success(monkeypatch):
    _enable_labs_profile(monkeypatch)
    module = _import_mcp_server()
    _setup_fakes(module)
    from knowledge_hub.mcp.handlers import jobs as jobs_handler

    module.SERVER_STATE.sqlite_db.ops_actions.append(
        {
            "action_id": "ops_action_agent_1",
            "scope": "agent",
            "action_type": "agent_repo_writeback_request",
            "status": "acked",
            "target_kind": "repo_goal",
            "target_key": "repo_goal:/tmp/repo:abcd1234",
            "summary": "Agent repo-local writeback request: Apply repo-local patch",
            "reason_codes_json": ["agent_gateway_repo_local_writeback_request"],
            "command": "khub",
            "args_json": [
                "agent",
                "run",
                "--goal",
                "Apply repo-local patch",
                "--repo-path",
                "/tmp/repo",
                "--role",
                "planner",
                "--orchestrator-mode",
                "adaptive",
                "--max-rounds",
                "3",
                "--max-workspace-files",
                "8",
            ],
            "alert_json": [],
            "action_json": {
                "goal": "Apply repo-local patch",
                "repoPath": "/tmp/repo",
                "role": "planner",
                "orchestratorMode": "adaptive",
                "maxRounds": 3,
                "maxWorkspaceFiles": 8,
                "requestKind": "repo_local_writeback",
                "gatewayVersion": "v2",
            },
            "seen_count": 1,
            "first_seen_at": "2026-03-16T12:45:00Z",
            "last_seen_at": "2026-03-16T12:45:00Z",
            "acked_at": "2026-03-16T12:46:00Z",
            "acked_by": "mcp-user",
            "resolved_at": "",
            "resolved_by": "",
            "note": "",
        }
    )

    class _FakeExecutor:
        def execute_sync(self, *, action_item, khub=None, config_path=None):  # noqa: ANN001
            _ = (khub, config_path)
            assert action_item["action_type"] == "agent_repo_writeback_request"
            return {
                "status": "ok",
                "command": "khub",
                "args": [*action_item["args_json"], "--json"],
                "resultSummary": "knowledge-hub.foundry.agent.run.result.v1 status=completed stage=DONE writeback=ok applied repo-local writeback",
                "errorSummary": "",
                "artifact": {
                    "schema": "knowledge-hub.foundry.agent.run.result.v1",
                    "status": "completed",
                    "stage": "DONE",
                    "goal": "Apply repo-local patch",
                    "writeback": {"ok": True, "detail": "applied repo-local writeback"},
                    "dryRun": False,
                },
                "warnings": [],
            }

    monkeypatch.setattr(jobs_handler, "OpsActionExecutor", _FakeExecutor)

    async def _fake_run_async_tool(name, request_echo, sync_job, started_at=None, **kwargs):  # noqa: ANN001
        _ = (name, request_echo, started_at)
        if callable(kwargs.get("on_queued")):
            kwargs["on_queued"]("job_ops_action_agent_1", "2026-03-16T10:30:00Z")
        payload = await sync_job()
        if callable(kwargs.get("on_finished")):
            kwargs["on_finished"](
                "job_ops_action_agent_1",
                module.MCP_TOOL_STATUS_OK,
                payload,
                None,
                "2026-03-16T10:30:00Z",
                "2026-03-16T10:31:00Z",
            )
        return "job_ops_action_agent_1", {"payload": payload}

    monkeypatch.setattr(module.SERVER_STATE, "_run_async_tool", _fake_run_async_tool, raising=False)
    queued = _decode_response(
        asyncio.run(module.call_tool("ops_action_execute", {"action_id": "ops_action_agent_1", "actor": "mcp-user"}))
    )
    assert queued["status"] == "queued"

    action = module.SERVER_STATE.sqlite_db.get_ops_action("ops_action_agent_1")
    assert action["status"] == "resolved"
    assert action["resolved_by"] == "mcp-user"

    receipts = module.SERVER_STATE.sqlite_db.list_ops_action_receipts(action_id="ops_action_agent_1", limit=10)
    assert receipts
    assert receipts[0]["status"] == "succeeded"


def test_run_agentic_query_requires_goal_and_returns_queued(monkeypatch):
    _enable_labs_profile(monkeypatch)
    module = _import_mcp_server()
    _setup_fakes(module)

    failed = _decode_response(asyncio.run(module.call_tool("run_agentic_query", {})))
    assert failed["status"] == "failed"
    assert "goal" in failed["payload"]["error"]

    async def _fake_run_async_tool(name, request_echo, sync_job, started_at=None):  # noqa: ANN001
        _ = (name, request_echo, sync_job, started_at)
        return "job_test_001", {"payload": {"message": "queued"}}

    monkeypatch.setattr(module.SERVER_STATE, "_run_async_tool", _fake_run_async_tool, raising=False)
    queued = _decode_response(asyncio.run(module.call_tool("run_agentic_query", {"goal": "RAG 비교"})))
    assert queued["status"] == "queued"
    assert queued["jobId"] == "job_test_001"


def test_unknown_tool_returns_failed_envelope():
    module = _import_mcp_server()
    _setup_fakes(module)

    failed = _decode_response(asyncio.run(module.call_tool("unknown_tool_xyz", {})))
    assert failed["status"] == "failed"
    assert "알 수 없는 도구" in failed["payload"]["error"]
