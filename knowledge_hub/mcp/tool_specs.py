from __future__ import annotations

import os

from mcp.types import Tool

from knowledge_hub.application.mcp.responses import DEFAULT_TOOL_NAMES


def resolve_tool_profile(profile: str | None = None) -> str:
    resolved = (profile or os.getenv("KHUB_MCP_PROFILE") or "default").strip().lower()
    if resolved not in {"default", "labs", "all"}:
        return "default"
    return resolved


def _resolve_tool_profile(profile: str | None = None) -> str:
    return resolve_tool_profile(profile)


def _filter_tools(tools: list[Tool], profile: str) -> list[Tool]:
    if profile == "default":
        return [tool for tool in tools if tool.name in DEFAULT_TOOL_NAMES]
    return tools


def build_tools(profile: str | None = None) -> list[Tool]:
    tools = [
        Tool(
            name="build_paper_memory",
            description="기존 paper/source-note/claim/ontology 산출물로 paper memory card를 빌드",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "arXiv 또는 paper id"},
                },
                "required": ["paper_id"],
            },
        ),
        Tool(
            name="get_paper_memory_card",
            description="paper memory card 조회",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "arXiv 또는 paper id"},
                },
                "required": ["paper_id"],
            },
        ),
        Tool(
            name="search_paper_memory",
            description="paper memory card 검색",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "검색어"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="search_knowledge",
            description="통합 지식 검색 (Obsidian 노트 + 논문 + 웹 문서에서 의미론적 유사도 검색)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "검색 질문"},
                    "top_k": {"type": "integer", "description": "결과 수 (기본: 5)", "default": 5},
                    "source": {"type": "string", "enum": ["all", "note", "paper", "web"], "description": "소스 필터"},
                    "mode": {
                        "type": "string",
                        "description": "검색 모드: semantic, keyword, hybrid",
                        "enum": ["semantic", "keyword", "hybrid"],
                        "default": "hybrid",
                    },
                    "alpha": {
                        "type": "number",
                        "description": "hybrid에서 semantic 가중치 (0~1)",
                        "default": 0.7,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="ask_knowledge",
            description="통합 지식 기반 RAG 답변 생성 (노트+논문+웹에서 검색 후 LLM이 답변)",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "질문"},
                    "top_k": {"type": "integer", "description": "참고 문서 수", "default": 5},
                    "source": {"type": "string", "enum": ["all", "note", "paper", "web"], "description": "소스 필터"},
                    "mode": {
                        "type": "string",
                        "description": "검색 모드: semantic, keyword, hybrid",
                        "enum": ["semantic", "keyword", "hybrid"],
                        "default": "hybrid",
                    },
                    "alpha": {
                        "type": "number",
                        "description": "hybrid에서 semantic 가중치 (0~1)",
                        "default": 0.7,
                    },
                    "memory_route_mode": {
                        "type": "string",
                        "description": "ask retrieval memory prefilter/prior mode: off, compat, on (prefilter는 deprecated compat alias)",
                        "enum": ["off", "compat", "on", "prefilter"],
                        "default": "off"
                    },
                    "paper_memory_mode": {
                        "type": "string",
                        "description": "paper-source memory prefilter mode: off, compat, on (prefilter는 deprecated compat alias)",
                        "enum": ["off", "compat", "on", "prefilter"],
                        "default": "off"
                    },
                    "min_score": {"type": "number", "description": "최소 유사도 (0~1)", "default": 0.3},
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="build_task_context",
            description="질의 목표에 맞춰 Obsidian/논문/웹 지식과 현재 repo 컨텍스트를 읽기 전용으로 조합",
            inputSchema={
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "작업 목표 또는 질문"},
                    "repo_path": {"type": "string", "description": "현재 작업 repo 경로. 기본값은 caller cwd"},
                    "include_workspace": {"type": "boolean", "default": True},
                    "include_vault": {"type": "boolean", "default": True},
                    "include_papers": {"type": "boolean", "default": True},
                    "include_web": {"type": "boolean", "default": True},
                    "max_workspace_files": {"type": "integer", "default": 8},
                    "max_knowledge_hits": {"type": "integer", "default": 5}
                },
                "required": ["goal"],
            },
        ),
        Tool(
            name="transform_list",
            description="사용 가능한 bounded transformation 목록 조회",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="transform_preview",
            description="bounded transformation 실행 전 source selection과 prompt를 미리보기",
            inputSchema={
                "type": "object",
                "properties": {
                    "transformation_id": {"type": "string"},
                    "query": {"type": "string"},
                    "repo_path": {"type": "string"},
                    "include_workspace": {"type": "boolean", "default": False},
                    "include_vault": {"type": "boolean", "default": True},
                    "include_papers": {"type": "boolean", "default": True},
                    "include_web": {"type": "boolean", "default": True},
                    "max_sources": {"type": "integer", "default": 6},
                },
                "required": ["transformation_id", "query"],
            },
        ),
        Tool(
            name="transform_run",
            description="bounded transformation 실행",
            inputSchema={
                "type": "object",
                "properties": {
                    "transformation_id": {"type": "string"},
                    "query": {"type": "string"},
                    "repo_path": {"type": "string"},
                    "include_workspace": {"type": "boolean", "default": False},
                    "include_vault": {"type": "boolean", "default": True},
                    "include_papers": {"type": "boolean", "default": True},
                    "include_web": {"type": "boolean", "default": True},
                    "max_sources": {"type": "integer", "default": 6},
                    "dry_run": {"type": "boolean", "default": False},
                },
                "required": ["transformation_id", "query"],
            },
        ),
        Tool(
            name="ask_graph",
            description="bounded multi-step ask planner with trace",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "source": {"type": "string", "enum": ["all", "note", "paper", "web"]},
                    "mode": {"type": "string", "enum": ["semantic", "keyword", "hybrid"], "default": "hybrid"},
                    "alpha": {"type": "number", "default": 0.7},
                    "max_steps": {"type": "integer", "default": 4},
                    "top_k": {"type": "integer", "default": 5},
                    "return_trace": {"type": "boolean", "default": True},
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="notebook_workbench_search",
            description="bounded source set 안에서 local workbench search 수행",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "query": {"type": "string"},
                    "selected_source_ids": {"type": "array", "items": {"type": "string"}},
                    "selected_source_context_modes": {
                        "type": "object",
                        "additionalProperties": {"type": "string", "enum": ["full", "summary", "excluded"]},
                    },
                    "include_vault": {"type": "boolean", "default": True},
                    "include_papers": {"type": "boolean", "default": True},
                    "include_web": {"type": "boolean", "default": True},
                    "top_k": {"type": "integer", "default": 5},
                    "mode": {"type": "string", "enum": ["semantic", "keyword", "hybrid"], "default": "hybrid"},
                    "alpha": {"type": "number", "default": 0.7}
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="notebook_workbench_chat",
            description="bounded source set 안에서 local workbench chat 수행",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "message": {"type": "string"},
                    "intent": {"type": "string", "enum": ["summary", "qa", "compare"], "default": "qa"},
                    "selected_source_ids": {"type": "array", "items": {"type": "string"}},
                    "selected_source_context_modes": {
                        "type": "object",
                        "additionalProperties": {"type": "string", "enum": ["full", "summary", "excluded"]},
                    },
                    "include_vault": {"type": "boolean", "default": True},
                    "include_papers": {"type": "boolean", "default": True},
                    "include_web": {"type": "boolean", "default": True},
                    "top_k": {"type": "integer", "default": 5},
                    "mode": {"type": "string", "enum": ["semantic", "keyword", "hybrid"], "default": "hybrid"},
                    "alpha": {"type": "number", "default": 0.7}
                },
                "required": ["message"],
            },
        ),
        Tool(
            name="ko_note_status",
            description="ko-note run 상태와 review/remediation 요약 조회",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {"type": "string", "description": "ko-note run id"}
                },
                "required": ["run_id"],
            },
        ),
        Tool(
            name="ko_note_report",
            description="ko-note run 상세 리포트와 최근 run 요약 조회",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {"type": "string", "description": "ko-note run id"},
                    "recent_runs": {"type": "integer", "default": 10}
                },
                "required": ["run_id"],
            },
        ),
        Tool(
            name="ko_note_review_list",
            description="review queue에 있는 ko-note 항목 조회",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {"type": "string", "description": "ko-note run id"},
                    "item_type": {
                        "type": "string",
                        "enum": ["source", "concept", "all"],
                        "default": "all",
                    },
                    "quality_flag": {
                        "type": "string",
                        "enum": ["needs_review", "reject", "unscored", "all"],
                        "default": "all",
                    },
                    "limit": {"type": "integer", "default": 50},
                },
                "required": ["run_id"],
            },
        ),
        Tool(
            name="ko_note_review_approve",
            description="staged ko-note item을 approved로 전환",
            inputSchema={
                "type": "object",
                "properties": {
                    "item_id": {"type": "integer"},
                    "reviewer": {"type": "string", "default": "mcp-user"},
                    "note": {"type": "string", "default": ""},
                },
                "required": ["item_id"],
            },
        ),
        Tool(
            name="ko_note_review_reject",
            description="staged ko-note item을 rejected로 전환",
            inputSchema={
                "type": "object",
                "properties": {
                    "item_id": {"type": "integer"},
                    "reviewer": {"type": "string", "default": "mcp-user"},
                    "note": {"type": "string", "default": ""},
                },
                "required": ["item_id"],
            },
        ),
        Tool(
            name="ko_note_remediate",
            description="review queue의 staged ko-note를 재-enrich하여 품질을 다시 계산",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {"type": "string", "description": "ko-note run id"},
                    "item_type": {
                        "type": "string",
                        "enum": ["source", "concept", "all"],
                        "default": "all",
                    },
                    "quality_flag": {
                        "type": "string",
                        "enum": ["needs_review", "reject", "unscored", "all"],
                        "default": "all",
                    },
                    "item_id": {"type": "integer", "default": 0},
                    "limit": {"type": "integer", "default": 50},
                    "strategy": {
                        "type": "string",
                        "enum": ["section", "full"],
                        "default": "section",
                    },
                    "allow_external": {"type": "boolean", "default": False},
                    "llm_mode": {
                        "type": "string",
                        "enum": ["auto", "local", "mini", "strong", "fallback-only"],
                        "default": "auto",
                    }
                },
                "required": ["run_id"],
            },
        ),
        Tool(
            name="rag_report",
            description="최근 RAG answer verification/rewrite 운영 리포트 조회",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 100},
                    "days": {"type": "integer", "default": 7}
                },
            },
        ),
        Tool(
            name="ops_action_list",
            description="ops action queue 조회",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["pending", "acked", "resolved", "all"], "default": "pending"},
                    "scope": {"type": "string", "enum": ["ko_note", "rag", "all"], "default": "all"},
                    "limit": {"type": "integer", "default": 50}
                },
            },
        ),
        Tool(
            name="ops_action_ack",
            description="ops action queue item을 acked로 전환",
            inputSchema={
                "type": "object",
                "properties": {
                    "action_id": {"type": "string"},
                    "actor": {"type": "string", "default": "mcp-user"},
                    "note": {"type": "string", "default": ""}
                },
                "required": ["action_id"],
            },
        ),
        Tool(
            name="ops_action_execute",
            description="safe ops action queue item을 실행하고 receipt를 기록",
            inputSchema={
                "type": "object",
                "properties": {
                    "action_id": {"type": "string"},
                    "actor": {"type": "string", "default": "mcp-user"},
                },
                "required": ["action_id"],
            },
        ),
        Tool(
            name="ops_action_receipts",
            description="ops action execution receipt 조회",
            inputSchema={
                "type": "object",
                "properties": {
                    "action_id": {"type": "string"},
                    "limit": {"type": "integer", "default": 20}
                },
                "required": ["action_id"],
            },
        ),
        Tool(
            name="ops_action_resolve",
            description="ops action queue item을 resolved로 전환",
            inputSchema={
                "type": "object",
                "properties": {
                    "action_id": {"type": "string"},
                    "actor": {"type": "string", "default": "mcp-user"},
                    "note": {"type": "string", "default": ""}
                },
                "required": ["action_id"],
            },
        ),
        Tool(
            name="run_agentic_query",
            description="Plan/Act/Verify 기반 에이전트형 질의 실행. --dump_json이면 foundry-core 출력 스키마로 반환",
            inputSchema={
                "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "에이전트 목표 질의"},
                "max_rounds": {"type": "integer", "description": "최대 라운드", "default": 2},
                "role": {"type": "string", "description": "에이전트 역할", "default": "planner"},
                "orchestratorMode": {"type": "string", "description": "조율 모드", "enum": ["single-pass", "adaptive", "strict"], "default": "adaptive"},
                "reportPath": {"type": "string", "description": "실행 리포트 저장 경로"},
                "repo_path": {"type": "string", "description": "현재 작업 repo 경로"},
                "include_workspace": {"type": "boolean", "description": "coding/debug/design 요청에서 repo context 포함 여부"},
                "max_workspace_files": {"type": "integer", "description": "task context에 포함할 최대 repo 파일 수", "default": 8},
                "dry_run": {"type": "boolean", "description": "검증만 수행", "default": False},
                "dump_json": {"type": "boolean", "description": "JSON 로그 반환", "default": False},
                "compact": {"type": "boolean", "description": "텍스트 모드에서 요약 출력", "default": False},
            },
            "required": ["goal"],
            },
        ),
        Tool(
            name="ontology_profile_list",
            description="사용 가능한 ontology profile 목록 조회",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="ontology_profile_show",
            description="특정 ontology profile 또는 현재 compiled profile 조회",
            inputSchema={
                "type": "object",
                "properties": {
                    "profile_id": {"type": "string"},
                    "compiled": {"type": "boolean", "default": False},
                },
            },
        ),
        Tool(
            name="ontology_profile_activate",
            description="core/domain/personal ontology profile 활성화",
            inputSchema={
                "type": "object",
                "properties": {
                    "profile_id": {"type": "string"},
                    "kind": {"type": "string", "enum": ["core", "domain", "personal"]},
                },
                "required": ["profile_id", "kind"],
            },
        ),
        Tool(
            name="ontology_profile_import",
            description="외부 ontology profile 파일을 사용자 profile 저장소로 import",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_path": {"type": "string"},
                    "profile_id": {"type": "string"},
                    "kind": {"type": "string", "enum": ["core", "domain", "personal"], "default": "personal"},
                },
                "required": ["source_path"],
            },
        ),
        Tool(
            name="ontology_profile_export",
            description="ontology profile 또는 compiled active profile을 파일로 export",
            inputSchema={
                "type": "object",
                "properties": {
                    "profile_id": {"type": "string"},
                    "destination": {"type": "string"},
                    "compiled": {"type": "boolean", "default": False},
                },
                "required": ["profile_id", "destination"],
            },
        ),
        Tool(
            name="ontology_proposal_submit",
            description="ontology profile proposal 등록",
            inputSchema={
                "type": "object",
                "properties": {
                    "proposal_type": {"type": "string", "enum": ["entity_type", "predicate", "profile_patch"]},
                    "target_profile": {"type": "string", "default": "personal"},
                    "payload": {"type": "object"},
                    "source": {"type": "string", "default": "user"},
                },
                "required": ["proposal_type", "payload"],
            },
        ),
        Tool(
            name="ontology_proposal_list",
            description="ontology profile proposal 목록 조회",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "proposal_type": {"type": "string", "enum": ["entity_type", "predicate", "profile_patch"]},
                    "target_profile": {"type": "string"},
                    "limit": {"type": "integer", "default": 100},
                },
            },
        ),
        Tool(
            name="ontology_proposal_apply",
            description="ontology profile proposal 승인 적용",
            inputSchema={
                "type": "object",
                "properties": {"proposal_id": {"type": "integer"}},
                "required": ["proposal_id"],
            },
        ),
        Tool(
            name="ontology_proposal_reject",
            description="ontology profile proposal 거절",
            inputSchema={
                "type": "object",
                "properties": {
                    "proposal_id": {"type": "integer"},
                    "reason": {"type": "string", "default": ""},
                },
                "required": ["proposal_id"],
            },
        ),
        Tool(
            name="belief_list",
            description="belief ledger 목록 조회",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "scope": {"type": "string"},
                    "limit": {"type": "integer", "default": 100},
                },
            },
        ),
        Tool(
            name="belief_show",
            description="belief 상세 조회",
            inputSchema={
                "type": "object",
                "properties": {"belief_id": {"type": "string"}},
                "required": ["belief_id"],
            },
        ),
        Tool(
            name="belief_upsert",
            description="belief 생성/갱신",
            inputSchema={
                "type": "object",
                "properties": {
                    "belief_id": {"type": "string"},
                    "statement": {"type": "string"},
                    "scope": {"type": "string", "default": "global"},
                    "status": {"type": "string", "default": "proposed"},
                    "confidence": {"type": "number", "default": 0.5},
                    "derived_from_claim_ids": {"type": "array", "items": {"type": "string"}},
                    "support_ids": {"type": "array", "items": {"type": "string"}},
                    "contradiction_ids": {"type": "array", "items": {"type": "string"}},
                    "last_validated_at": {"type": "string"},
                    "review_due_at": {"type": "string"},
                },
                "required": ["statement"],
            },
        ),
        Tool(
            name="belief_review",
            description="belief 상태 검토/전환",
            inputSchema={
                "type": "object",
                "properties": {
                    "belief_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["proposed", "reviewed", "trusted", "stale", "rejected"]},
                    "last_validated_at": {"type": "string"},
                    "review_due_at": {"type": "string"},
                },
                "required": ["belief_id", "status"],
            },
        ),
        Tool(
            name="decision_create",
            description="decision 생성/갱신",
            inputSchema={
                "type": "object",
                "properties": {
                    "decision_id": {"type": "string"},
                    "title": {"type": "string"},
                    "summary": {"type": "string", "default": ""},
                    "related_belief_ids": {"type": "array", "items": {"type": "string"}},
                    "chosen_option": {"type": "string", "default": ""},
                    "status": {"type": "string", "default": "open"},
                    "review_due_at": {"type": "string"},
                },
                "required": ["title"],
            },
        ),
        Tool(
            name="decision_list",
            description="decision 목록 조회",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "limit": {"type": "integer", "default": 100},
                },
            },
        ),
        Tool(
            name="decision_review",
            description="decision 상태 검토/전환",
            inputSchema={
                "type": "object",
                "properties": {
                    "decision_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["open", "committed", "reviewed", "closed"]},
                    "review_due_at": {"type": "string"},
                },
                "required": ["decision_id", "status"],
            },
        ),
        Tool(
            name="outcome_record",
            description="decision outcome 기록",
            inputSchema={
                "type": "object",
                "properties": {
                    "outcome_id": {"type": "string"},
                    "decision_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["observed", "confirmed", "invalidated"], "default": "observed"},
                    "summary": {"type": "string"},
                    "recorded_at": {"type": "string"},
                },
                "required": ["decision_id", "summary"],
            },
        ),
        Tool(
            name="outcome_show",
            description="decision outcome 목록 조회",
            inputSchema={
                "type": "object",
                "properties": {
                    "decision_id": {"type": "string"},
                    "limit": {"type": "integer", "default": 50},
                },
                "required": ["decision_id"],
            },
        ),
        Tool(
            name="search_papers",
            description="논문 메타데이터 검색 (제목, 저자, 분야)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "검색어"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="index_paper_keywords",
            description="번역본 텍스트 기반 핵심 키워드 추출 후 paper/keyword note 및 링크를 생성/업데이트",
            inputSchema={
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string", "description": "단일 논문 arXiv ID (생략 시 번역된 모든 논문 처리)"},
                    "top_k": {"type": "integer", "description": "추출할 최대 핵심 키워드 수", "default": 12},
                    "max_links_per_keyword": {"type": "integer", "description": "키워드당 연결할 최대 관련 노트 수", "default": 5},
                    "dry_run": {"type": "boolean", "description": "실제 저장 없이 후보만 산출", "default": False},
                },
            },
        ),
        Tool(
            name="crawl_web_ingest",
            description="crawl4ai(또는 기본 크롤러)로 웹 문서를 수집해 notes/vector DB에 적재",
            inputSchema={
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "수집할 URL 배열",
                    },
                    "topic": {"type": "string", "description": "학습 주제 라벨"},
                    "engine": {"type": "string", "enum": ["auto", "crawl4ai", "basic"], "default": "auto"},
                    "timeout": {"type": "integer", "default": 15},
                    "delay": {"type": "number", "default": 0.5},
                    "index": {"type": "boolean", "default": True},
                    "extract_concepts": {"type": "boolean", "default": True},
                    "allow_external": {"type": "boolean", "default": False},
                    "writeback": {"type": "boolean", "default": False},
                    "concept_threshold": {"type": "number", "default": 0.78},
                    "relation_threshold": {"type": "number", "default": 0.75},
                    "emit_ontology_graph": {"type": "boolean", "default": False},
                    "ontology_ttl_path": {"type": "string", "description": "ttl 저장 경로"},
                    "validate_ontology_graph": {"type": "boolean", "default": False},
                    "learn_map": {"type": "boolean", "default": False},
                    "compact": {"type": "boolean", "default": False},
                },
                "required": ["urls"],
            },
        ),
        Tool(
            name="crawl_youtube_ingest",
            description="YouTube URL을 caption-first/local-ASR-fallback으로 수집해 notes/vector DB에 적재",
            inputSchema={
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "수집할 YouTube URL 배열",
                    },
                    "topic": {"type": "string", "description": "학습 주제 라벨"},
                    "timeout": {"type": "integer", "default": 30},
                    "delay": {"type": "number", "default": 0.0},
                    "index": {"type": "boolean", "default": True},
                    "extract_concepts": {"type": "boolean", "default": True},
                    "allow_external": {"type": "boolean", "default": False},
                    "writeback": {"type": "boolean", "default": False},
                    "transcript_language": {"type": "string", "description": "선호 자막/전사 언어 코드"},
                    "asr_model": {"type": "string", "default": "tiny"},
                    "compact": {"type": "boolean", "default": False},
                },
                "required": ["urls"],
            },
        ),
        Tool(
            name="crawl_pending_list",
            description="웹 온톨로지 pending 큐 조회",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "limit": {"type": "integer", "default": 50},
                    "compact": {"type": "boolean", "default": False},
                },
            },
        ),
        Tool(
            name="crawl_pending_apply",
            description="웹 온톨로지 pending 항목 승인 적용",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "pending item id"},
                    "compact": {"type": "boolean", "default": False},
                },
                "required": ["id"],
            },
        ),
        Tool(
            name="crawl_pending_reject",
            description="웹 온톨로지 pending 항목 거절",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "pending item id"},
                    "compact": {"type": "boolean", "default": False},
                },
                "required": ["id"],
            },
        ),
        Tool(
            name="crawl_pipeline_run",
            description="대용량 crawl 파이프라인 실행(raw/normalized/indexed + checkpoint)",
            inputSchema={
                "type": "object",
                "properties": {
                    "urls": {"type": "array", "items": {"type": "string"}},
                    "topic": {"type": "string"},
                    "source": {"type": "string", "default": "web"},
                    "profile": {"type": "string", "enum": ["safe", "balanced", "fast"], "default": "safe"},
                    "source_policy": {"type": "string", "enum": ["fixed", "hybrid", "keyword"], "default": "hybrid"},
                    "limit": {"type": "integer", "default": 0},
                    "engine": {"type": "string", "enum": ["auto", "crawl4ai", "basic"], "default": "auto"},
                    "timeout": {"type": "integer", "default": 15},
                    "delay": {"type": "number", "default": 0.5},
                    "index": {"type": "boolean", "default": True},
                    "extract_concepts": {"type": "boolean", "default": True},
                    "allow_external": {"type": "boolean", "default": False}
                },
                "required": ["urls"]
            }
        ),
        Tool(
            name="crawl_pipeline_resume",
            description="중단된 crawl 파이프라인 재개",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "profile": {"type": "string", "enum": ["safe", "balanced", "fast"]},
                    "source_policy": {"type": "string", "enum": ["fixed", "hybrid", "keyword"]},
                    "limit": {"type": "integer", "default": 0},
                    "engine": {"type": "string", "enum": ["auto", "crawl4ai", "basic"], "default": "auto"},
                    "timeout": {"type": "integer", "default": 15},
                    "delay": {"type": "number", "default": 0.5},
                    "index": {"type": "boolean", "default": True},
                    "extract_concepts": {"type": "boolean", "default": True},
                    "allow_external": {"type": "boolean", "default": False}
                },
                "required": ["job_id"]
            }
        ),
        Tool(
            name="crawl_pipeline_status",
            description="crawl 파이프라인 상태/체크포인트 조회",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"}
                },
                "required": ["job_id"]
            }
        ),
        Tool(
            name="crawl_domain_policy_list",
            description="도메인 승인 정책 목록 조회",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["approved", "pending", "rejected"]},
                    "limit": {"type": "integer", "default": 200}
                }
            }
        ),
        Tool(
            name="crawl_domain_policy_apply",
            description="도메인 승인(approved) 적용",
            inputSchema={
                "type": "object",
                "properties": {
                    "domain": {"type": "string"},
                    "reason": {"type": "string", "default": ""}
                },
                "required": ["domain"]
            }
        ),
        Tool(
            name="crawl_domain_policy_reject",
            description="도메인 수집 거절(rejected) 적용",
            inputSchema={
                "type": "object",
                "properties": {
                    "domain": {"type": "string"},
                    "reason": {"type": "string", "default": ""}
                },
                "required": ["domain"]
            }
        ),
        Tool(
            name="crawl_pipeline_benchmark",
            description="crawl 파이프라인 성능 스모크 실행",
            inputSchema={
                "type": "object",
                "properties": {
                    "urls": {"type": "array", "items": {"type": "string"}},
                    "sample": {"type": "integer", "default": 20},
                    "topic": {"type": "string"},
                    "profile": {"type": "string", "enum": ["safe", "balanced", "fast"], "default": "safe"},
                    "source_policy": {"type": "string", "enum": ["fixed", "hybrid", "keyword"], "default": "hybrid"},
                    "engine": {"type": "string", "enum": ["auto", "crawl4ai", "basic"], "default": "auto"}
                },
                "required": ["urls"]
            }
        ),
        Tool(
            name="foundry_conflict_list",
            description="Foundry dual-write 충돌 pending 큐 조회",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["pending", "approved", "rejected"],
                        "default": "pending",
                    },
                    "connector_id": {"type": "string"},
                    "source_filter": {"type": "string", "enum": ["all", "note", "paper", "web", "expense", "sleep", "schedule", "behavior"]},
                    "limit": {"type": "integer", "default": 50},
                    "compact": {"type": "boolean", "default": False},
                },
            },
        ),
        Tool(
            name="foundry_conflict_apply",
            description="Foundry dual-write 충돌 승인(apply)",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "reviewer": {"type": "string", "default": "mcp"},
                    "note": {"type": "string"},
                    "compact": {"type": "boolean", "default": False},
                },
                "required": ["id"],
            },
        ),
        Tool(
            name="foundry_conflict_reject",
            description="Foundry dual-write 충돌 거절(reject)",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "reviewer": {"type": "string", "default": "mcp"},
                    "note": {"type": "string"},
                    "compact": {"type": "boolean", "default": False},
                },
                "required": ["id"],
            },
        ),
        Tool(
            name="entity_merge_list",
            description="Entity resolution merge proposal pending 큐 조회",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "approved", "rejected"],
                        "default": "pending",
                    },
                    "limit": {"type": "integer", "default": 50},
                    "compact": {"type": "boolean", "default": False},
                },
            },
        ),
        Tool(
            name="entity_merge_apply",
            description="Entity resolution merge proposal 승인(apply)",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "compact": {"type": "boolean", "default": False},
                },
                "required": ["id"],
            },
        ),
        Tool(
            name="entity_merge_reject",
            description="Entity resolution merge proposal 거절(reject)",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "compact": {"type": "boolean", "default": False},
                },
                "required": ["id"],
            },
        ),
        Tool(
            name="learning_start_or_resume_topic",
            description="학습 topic 기준 최근 세션을 자동 이어서 시작하거나 새 세션 생성",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "학습 주제"},
                    "force_new_session": {"type": "boolean", "default": False},
                    "source": {"type": "string", "enum": ["all", "note", "paper", "web"], "default": "all"},
                    "days": {"type": "integer", "default": 180},
                    "top_k": {"type": "integer", "default": 12},
                    "concept_count": {"type": "integer", "default": 6},
                    "compact": {"type": "boolean", "default": False},
                },
                "required": ["topic"],
            },
        ),
        Tool(
            name="learning_get_session_state",
            description="학습 세션 상태, 퀴즈 이력, 약점, 다음 복습 대상을 조회",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "session_id": {"type": "string"},
                    "compact": {"type": "boolean", "default": False},
                },
            },
        ),
        Tool(
            name="learning_explain_topic",
            description="근거 우선 설명을 생성하고 부족할 때만 모델 보완 설명을 구분해 반환",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "question": {"type": "string"},
                    "session_id": {"type": "string"},
                    "source": {"type": "string", "enum": ["all", "note", "paper", "web"], "default": "all"},
                    "top_k": {"type": "integer", "default": 5},
                    "min_score": {"type": "number", "default": 0.3},
                    "compact": {"type": "boolean", "default": False},
                },
                "required": ["topic", "question"],
            },
        ),
        Tool(
            name="learning_checkpoint",
            description="평가 외 수동 체크포인트를 저장하고 known/shaky/unknown 상태를 갱신",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "session_id": {"type": "string"},
                    "summary": {"type": "string"},
                    "known_items": {"type": "array", "items": {"oneOf": [{"type": "string"}, {"type": "object"}]}},
                    "shaky_items": {"type": "array", "items": {"oneOf": [{"type": "string"}, {"type": "object"}]}},
                    "unknown_items": {"type": "array", "items": {"oneOf": [{"type": "string"}, {"type": "object"}]}},
                    "misconceptions": {"type": "array", "items": {"oneOf": [{"type": "string"}, {"type": "object"}]}},
                    "writeback": {"type": "boolean", "default": False},
                    "compact": {"type": "boolean", "default": False},
                },
                "required": ["topic", "session_id", "summary"],
            },
        ),
        Tool(
            name="learn_map",
            description="Learning Coach: topic 기준 trunk/branch map 생성",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "학습 주제"},
                    "source": {"type": "string", "enum": ["all", "note", "paper", "web"], "default": "all"},
                    "days": {"type": "integer", "default": 180},
                    "top_k": {"type": "integer", "default": 12},
                    "dry_run": {"type": "boolean", "default": False},
                    "writeback": {"type": "boolean", "default": False},
                    "canvas": {"type": "boolean", "default": False},
                    "allow_external": {"type": "boolean", "default": False},
                    "compact": {"type": "boolean", "default": False},
                },
                "required": ["topic"],
            },
        ),
        Tool(
            name="learn_assess_template",
            description="Learning Coach: 세션 템플릿 생성 및 target_trunk_ids 고정",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "session_id": {"type": "string"},
                    "concept_count": {"type": "integer", "default": 6},
                    "source": {"type": "string", "enum": ["all", "note", "paper", "web"], "default": "all"},
                    "days": {"type": "integer", "default": 180},
                    "top_k": {"type": "integer", "default": 12},
                    "dry_run": {"type": "boolean", "default": False},
                    "writeback": {"type": "boolean", "default": False},
                    "allow_external": {"type": "boolean", "default": False},
                    "compact": {"type": "boolean", "default": False},
                },
                "required": ["topic", "session_id"],
            },
        ),
        Tool(
            name="learn_grade",
            description="Learning Coach: 세션 concept map 채점 및 게이트 판정",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "session_id": {"type": "string"},
                    "dry_run": {"type": "boolean", "default": False},
                    "writeback": {"type": "boolean", "default": False},
                    "allow_external": {"type": "boolean", "default": False},
                    "compact": {"type": "boolean", "default": False},
                },
                "required": ["topic", "session_id"],
            },
        ),
        Tool(
            name="learn_next",
            description="Learning Coach: gate 결과 기반 다음 branch/remediation 추천",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "session_id": {"type": "string"},
                    "source": {"type": "string", "enum": ["all", "note", "paper", "web"], "default": "all"},
                    "days": {"type": "integer", "default": 180},
                    "top_k": {"type": "integer", "default": 12},
                    "dry_run": {"type": "boolean", "default": False},
                    "writeback": {"type": "boolean", "default": False},
                    "allow_external": {"type": "boolean", "default": False},
                    "compact": {"type": "boolean", "default": False},
                },
                "required": ["topic", "session_id"],
            },
        ),
        Tool(
            name="learn_run",
            description="Learning Coach: map -> assess-template -> grade -> next 오케스트레이션",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "session_id": {"type": "string"},
                    "source": {"type": "string", "enum": ["all", "note", "paper", "web"], "default": "all"},
                    "days": {"type": "integer", "default": 180},
                    "top_k": {"type": "integer", "default": 12},
                    "concept_count": {"type": "integer", "default": 6},
                    "auto_next": {"type": "boolean", "default": False},
                    "dry_run": {"type": "boolean", "default": False},
                    "writeback": {"type": "boolean", "default": False},
                    "canvas": {"type": "boolean", "default": False},
                    "allow_external": {"type": "boolean", "default": False},
                    "compact": {"type": "boolean", "default": False},
                },
                "required": ["topic", "session_id"],
            },
        ),
        Tool(
            name="learn_analyze_gaps",
            description="Learning Coach: 세션/온톨로지 기준 누락 개념/약한 엣지/근거 부족 탐지",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "session_id": {"type": "string"},
                    "source": {"type": "string", "enum": ["all", "note", "paper", "web"], "default": "all"},
                    "days": {"type": "integer", "default": 180},
                    "top_k": {"type": "integer", "default": 12},
                    "dry_run": {"type": "boolean", "default": False},
                    "writeback": {"type": "boolean", "default": False},
                    "allow_external": {"type": "boolean", "default": False},
                    "compact": {"type": "boolean", "default": False},
                },
                "required": ["topic", "session_id"],
            },
        ),
        Tool(
            name="learn_generate_quiz",
            description="Learning Coach: 혼합형 퀴즈 생성",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "session_id": {"type": "string"},
                    "source": {"type": "string", "enum": ["all", "note", "paper", "web"], "default": "all"},
                    "days": {"type": "integer", "default": 180},
                    "top_k": {"type": "integer", "default": 12},
                    "mix": {"type": "string", "enum": ["mixed", "mcq", "essay"], "default": "mixed"},
                    "question_count": {"type": "integer", "default": 6},
                    "dry_run": {"type": "boolean", "default": False},
                    "writeback": {"type": "boolean", "default": False},
                    "allow_external": {"type": "boolean", "default": False},
                    "compact": {"type": "boolean", "default": False},
                },
                "required": ["topic", "session_id"],
            },
        ),
        Tool(
            name="learn_grade_quiz",
            description="Learning Coach: 퀴즈 답안 채점 및 약점 피드백",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "session_id": {"type": "string"},
                    "answers": {"type": "array", "items": {"type": "object"}},
                    "source": {"type": "string", "enum": ["all", "note", "paper", "web"], "default": "all"},
                    "days": {"type": "integer", "default": 180},
                    "top_k": {"type": "integer", "default": 12},
                    "dry_run": {"type": "boolean", "default": False},
                    "writeback": {"type": "boolean", "default": False},
                    "allow_external": {"type": "boolean", "default": False},
                    "compact": {"type": "boolean", "default": False},
                },
                "required": ["topic", "session_id"],
            },
        ),
        Tool(
            name="learn_reinforce",
            description="Learning Coach: gap 분석 기반 지식 보강 추천 (소스 매칭 + 우선순위)",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "session_id": {"type": "string"},
                    "source": {"type": "string", "enum": ["all", "note", "paper", "web"], "default": "all"},
                    "days": {"type": "integer", "default": 180},
                    "top_k": {"type": "integer", "default": 12},
                    "top_k_per_gap": {"type": "integer", "default": 3, "description": "각 gap당 추천 소스 수"},
                    "dry_run": {"type": "boolean", "default": False},
                    "writeback": {"type": "boolean", "default": False},
                    "allow_external": {"type": "boolean", "default": False},
                    "compact": {"type": "boolean", "default": False},
                },
                "required": ["topic", "session_id"],
            },
        ),
        Tool(
            name="learn_suggest_patch",
            description="Learning Coach: 부족 영역 보완 초안(diff 텍스트) 제안 생성",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "session_id": {"type": "string"},
                    "source": {"type": "string", "enum": ["all", "note", "paper", "web"], "default": "all"},
                    "days": {"type": "integer", "default": 180},
                    "top_k": {"type": "integer", "default": 12},
                    "dry_run": {"type": "boolean", "default": False},
                    "writeback": {"type": "boolean", "default": False},
                    "allow_external": {"type": "boolean", "default": False},
                    "compact": {"type": "boolean", "default": False},
                },
                "required": ["topic", "session_id"],
            },
        ),
        Tool(
            name="learn_graph_build",
            description="온톨로지 기반 학습 그래프 후보 생성 및 pending 큐 적재",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "top_k": {"type": "integer", "default": 12},
                    "allow_external": {"type": "boolean", "default": False},
                },
                "required": ["topic"],
            },
        ),
        Tool(
            name="learn_graph_pending_list",
            description="학습 그래프 pending 항목 목록 조회",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "item_type": {"type": "string", "enum": ["all", "edge", "path", "difficulty", "resource_link"], "default": "all"},
                    "limit": {"type": "integer", "default": 100},
                },
            },
        ),
        Tool(
            name="learn_graph_pending_apply",
            description="학습 그래프 pending 항목 승인 적용",
            inputSchema={
                "type": "object",
                "properties": {
                    "pending_id": {"type": "integer"},
                },
                "required": ["pending_id"],
            },
        ),
        Tool(
            name="learn_graph_pending_reject",
            description="학습 그래프 pending 항목 거절",
            inputSchema={
                "type": "object",
                "properties": {
                    "pending_id": {"type": "integer"},
                },
                "required": ["pending_id"],
            },
        ),
        Tool(
            name="learn_path_generate",
            description="승인된 학습 그래프 엣지 기반 학습 경로 생성",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "approved_only": {"type": "boolean", "default": True},
                    "writeback": {"type": "boolean", "default": False},
                },
                "required": ["topic"],
            },
        ),
        Tool(
            name="run_learning_pipeline",
            description="학습 파이프라인 오케스트레이션: learn_map -> assess-template -> grade -> next",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "학습 주제"},
                    "session_id": {"type": "string", "description": "세션 ID"},
                    "source": {"type": "string", "enum": ["all", "note", "paper", "web"], "default": "all"},
                    "days": {"type": "integer", "default": 180},
                    "top_k": {"type": "integer", "default": 12},
                    "concept_count": {"type": "integer", "default": 6},
                    "auto_next": {"type": "boolean", "default": False},
                    "dry_run": {"type": "boolean", "default": False},
                    "writeback": {"type": "boolean", "default": False},
                    "canvas": {"type": "boolean", "default": False},
                    "allow_external": {"type": "boolean", "default": False},
                    "compact": {"type": "boolean", "default": False},
                },
                "required": ["topic", "session_id"],
            },
        ),
        Tool(
            name="run_paper_ingest_flow",
            description="논문 발굴-다운로드-요약-인덱싱 흐름을 비동기로 실행",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "검색할 주제"},
                    "max_papers": {"type": "integer", "description": "수집할 최대 논문 수 (기본: 5)", "default": 5},
                    "year_start": {"type": "integer", "description": "검색 시작 연도 (예: 2024)"},
                    "min_citations": {"type": "integer", "description": "최소 인용수 필터", "default": 0},
                    "sort_by": {"type": "string", "enum": ["relevance", "citationCount"], "default": "relevance"},
                    "create_obsidian_note": {"type": "boolean", "default": True},
                    "generate_summary": {"type": "boolean", "default": True},
                    "dry_run": {"type": "boolean", "default": False},
                },
                "required": ["topic"],
            },
        ),
        Tool(
            name="get_hub_stats",
            description="Knowledge Hub 통계 정보 (노트, 논문, 태그, 벡터 DB 현황)",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="search_authors",
            description="저자 이름/소속으로 검색 (Semantic Scholar)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "저자 이름 또는 소속"},
                    "limit": {"type": "integer", "description": "최대 결과 수", "default": 10},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_author_papers",
            description="특정 저자의 논문 목록 조회",
            inputSchema={
                "type": "object",
                "properties": {
                    "author_id": {"type": "string", "description": "Semantic Scholar 저자 ID"},
                    "limit": {"type": "integer", "description": "최대 논문 수", "default": 20},
                },
                "required": ["author_id"],
            },
        ),
        Tool(
            name="get_paper_detail",
            description="논문 상세 정보 조회 (abstract, 메타데이터, 인용수 등). arXiv ID, DOI, SS ID 사용 가능",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "논문 ID (arXiv ID, DOI, Semantic Scholar ID)"},
                },
                "required": ["paper_id"],
            },
        ),
        Tool(
            name="paper_lookup_and_summarize",
            description="논문 식별자 또는 검색어로 논문을 찾고, 메타데이터와 로컬 RAG 기반 요약을 함께 반환",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "논문 ID (arXiv ID, DOI, Semantic Scholar ID)"},
                    "query": {"type": "string", "description": "논문 제목 또는 검색어"},
                    "top_k": {"type": "integer", "description": "요약 생성 시 참고 문서 수", "default": 5},
                    "min_score": {"type": "number", "description": "최소 유사도 (0~1)", "default": 0.3},
                    "mode": {
                        "type": "string",
                        "description": "검색 모드: semantic, keyword, hybrid",
                        "enum": ["semantic", "keyword", "hybrid"],
                        "default": "hybrid",
                    },
                    "alpha": {
                        "type": "number",
                        "description": "hybrid에서 semantic 가중치 (0~1)",
                        "default": 0.7,
                    },
                    "memory_route_mode": {
                        "type": "string",
                        "description": "ask retrieval memory prefilter/prior mode: off, compat, on (prefilter는 deprecated compat alias)",
                        "enum": ["off", "compat", "on", "prefilter"],
                        "default": "off"
                    },
                    "paper_memory_mode": {
                        "type": "string",
                        "description": "paper-source memory prefilter mode: off, compat, on (prefilter는 deprecated compat alias)",
                        "enum": ["off", "compat", "on", "prefilter"],
                        "default": "off"
                    },
                },
            },
        ),
        Tool(
            name="paper_topic_synthesize",
            description="로컬 논문 코퍼스에서 주제형 다논문 후보를 모으고 선택/요약하는 labs 전용 합성 경로",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "주제형 논문 질문"},
                    "source_mode": {
                        "type": "string",
                        "enum": ["local", "discover", "hybrid"],
                        "default": "local",
                    },
                    "candidate_limit": {"type": "integer", "default": 12},
                    "selected_limit": {"type": "integer", "default": 6},
                    "top_k": {"type": "integer", "default": 8},
                    "mode": {
                        "type": "string",
                        "description": "검색 모드: semantic, keyword, hybrid",
                        "enum": ["semantic", "keyword", "hybrid"],
                        "default": "hybrid",
                    },
                    "alpha": {"type": "number", "default": 0.7},
                    "allow_external": {"type": "boolean", "default": False},
                    "llm_mode": {
                        "type": "string",
                        "enum": ["auto", "local", "mini", "strong"],
                        "default": "auto",
                    },
                    "provider": {"type": "string"},
                    "model": {"type": "string"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_paper_citations",
            description="이 논문을 인용한 논문들 조회 (피인용)",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "논문 ID"},
                    "limit": {"type": "integer", "description": "최대 결과 수", "default": 20},
                },
                "required": ["paper_id"],
            },
        ),
        Tool(
            name="get_paper_references",
            description="이 논문이 참고한 논문들 조회 (참고문헌)",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "논문 ID"},
                    "limit": {"type": "integer", "description": "최대 결과 수", "default": 20},
                },
                "required": ["paper_id"],
            },
        ),
        Tool(
            name="analyze_citation_network",
            description="인용 네트워크 분석 — 피인용/참고문헌/연도 분포/분야 통계",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "논문 ID"},
                    "depth": {"type": "integer", "description": "분석 깊이 (1 또는 2)", "default": 1},
                    "citations_limit": {"type": "integer", "description": "피인용 최대 수", "default": 10},
                    "references_limit": {"type": "integer", "description": "참고문헌 최대 수", "default": 10},
                },
                "required": ["paper_id"],
            },
        ),
        Tool(
            name="batch_paper_lookup",
            description="복수 논문 일괄 조회 (arXiv ID, DOI 등)",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "논문 ID 배열",
                    },
                },
                "required": ["paper_ids"],
            },
        ),
        Tool(
            name="discover_and_ingest",
            description="AI 논문 자동 발견 → 벡터DB 중복 체크 → 다운로드 → 요약 → 인덱싱 → Obsidian 연결 (전체 파이프라인)",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "검색할 주제 (예: 'large language model', 'AI agent', 'RAG')"},
                    "max_papers": {"type": "integer", "description": "수집할 최대 논문 수 (기본: 5)", "default": 5},
                    "year_start": {"type": "integer", "description": "검색 시작 연도 (예: 2024)"},
                    "min_citations": {"type": "integer", "description": "최소 인용수 필터", "default": 0},
                    "sort_by": {
                        "type": "string",
                        "description": "정렬 기준",
                        "enum": ["relevance", "citationCount"],
                        "default": "relevance",
                    },
                    "create_obsidian_note": {"type": "boolean", "description": "Obsidian vault에 요약 노트 생성", "default": True},
                    "generate_summary": {"type": "boolean", "description": "LLM으로 한국어 요약 생성", "default": True},
                    "judge_enabled": {"type": "boolean", "description": "optional paper discovery filter 사용", "default": False},
                    "judge_threshold": {"type": "number", "description": "optional judge keep threshold", "default": 0.62},
                    "judge_candidates": {"type": "integer", "description": "optional judge 평가 후보 수 (기본: max_papers*3)"},
                    "allow_external": {"type": "boolean", "description": "optional judge에서 외부 LLM 사용 허용", "default": False},
                    "dry_run": {"type": "boolean", "default": False},
                },
                "required": ["topic"],
            },
        ),
        Tool(
            name="check_paper_duplicate",
            description="특정 arXiv 논문이 이미 벡터DB/SQLite에 존재하는지 확인",
            inputSchema={
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string", "description": "확인할 arXiv ID"},
                },
                "required": ["arxiv_id"],
            },
        ),
        Tool(
            name="mcp_job_status",
            description="비동기 MCP 작업 상태 조회",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "jobId"},
                    "include_payload": {"type": "boolean", "default": True},
                    "compact": {"type": "boolean", "default": False},
                },
                "required": ["job_id"],
            },
        ),
        Tool(
            name="mcp_job_list",
            description="비동기 MCP 작업 목록 조회",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["queued", "running", "done", "failed", "blocked", "expired"]},
                    "tool": {"type": "string"},
                    "limit": {"type": "integer", "default": 20},
                    "compact": {"type": "boolean", "default": False},
                },
            },
        ),
        Tool(
            name="mcp_job_cancel",
            description="비동기 작업 취소 요청(queued/running -> expired)",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "jobId"},
                    "compact": {"type": "boolean", "default": False},
                },
                "required": ["job_id"],
            },
        ),
    ]
    return _filter_tools(tools, _resolve_tool_profile(profile))
