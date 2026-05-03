from __future__ import annotations

from mcp.types import Tool


def build_tools(profile: str | None = None) -> list[Tool]:
    _ = profile
    return [
        Tool(
            name="search_knowledge",
            description="통합 지식 검색",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "검색 질문"},
                    "top_k": {"type": "integer", "default": 5},
                    "source": {"type": "string", "enum": ["all", "note", "paper", "web"]},
                    "mode": {"type": "string", "enum": ["semantic", "keyword", "hybrid"], "default": "hybrid"},
                    "alpha": {"type": "number", "default": 0.7},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="ask_knowledge",
            description="통합 지식 기반 RAG 답변 생성",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "질문"},
                    "top_k": {"type": "integer", "default": 5},
                    "source": {"type": "string", "enum": ["all", "note", "paper", "web"]},
                    "mode": {"type": "string", "enum": ["semantic", "keyword", "hybrid"], "default": "hybrid"},
                    "alpha": {"type": "number", "default": 0.7},
                    "min_score": {"type": "number", "default": 0.3},
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="build_task_context",
            description="질의 목표에 맞춰 지식과 현재 repo 컨텍스트를 읽기 전용으로 조합",
            inputSchema={
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "작업 목표 또는 질문"},
                    "repo_path": {"type": "string", "description": "현재 작업 repo 경로"},
                    "include_workspace": {"type": "boolean", "default": True},
                    "include_vault": {"type": "boolean", "default": True},
                    "include_papers": {"type": "boolean", "default": True},
                    "include_web": {"type": "boolean", "default": True},
                    "max_workspace_files": {"type": "integer", "default": 8},
                    "max_knowledge_hits": {"type": "integer", "default": 5},
                },
                "required": ["goal"],
            },
        ),
        Tool(
            name="get_hub_stats",
            description="현재 knowledge-hub corpus 통계 조회",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]
