#!/usr/bin/env python3
"""
Knowledge Hub MCP Server

Cursor에서 통합 지식 검색 및 RAG를 사용할 수 있게 합니다.

Cursor 설정:
{
  "knowledge-hub": {
    "command": "khub-mcp",
    "env": {}
  }
}
"""

import sys
import asyncio
import json
import subprocess
import shutil
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

project_root = Path(__file__).resolve().parent.parent

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print("MCP 라이브러리 필요: pip install mcp", file=sys.stderr)
    sys.exit(1)

from knowledge_hub.core.config import Config
from knowledge_hub.core.database import VectorDatabase, SQLiteDatabase
from knowledge_hub.providers.registry import get_llm, get_embedder
from knowledge_hub.ai.rag import RAGSearcher

searcher = None
sqlite_db = None
config = None
DEFAULT_FOUNDRY_RETRY_ATTEMPTS = 2


def _foundry_retry_count() -> int:
    raw = os.getenv("KHUB_FOUNDRY_RETRY_ATTEMPTS", str(DEFAULT_FOUNDRY_RETRY_ATTEMPTS))
    try:
        value = int((raw or "").strip())
        if value < 1:
            return 1
        return min(value, 8)
    except Exception:
        return DEFAULT_FOUNDRY_RETRY_ATTEMPTS


def _normalize_classification(value: object | None) -> str:
    normalized = str(value or "P2").strip().upper()
    if normalized in {"P0", "P1", "P2", "P3"}:
        return normalized
    return "P2"


def _evaluate_policy_gate(artifact: object | None) -> tuple[bool, list[str]]:
    classification = "P2"
    if isinstance(artifact, dict):
        classification = _normalize_classification(artifact.get("classification"))
    if classification == "P0":
        return False, ["policy denied: P0 artifact blocked by local policy"]
    return True, []


def _transition_code(item: dict[str, object], stage_fallback: str = "PLAN", status_fallback: str = "STEP") -> str:
    stage = str(item.get("stage", stage_fallback)).strip().upper() or stage_fallback
    raw_status = str(item.get("status", item.get("action", item.get("step", status_fallback)))).strip().upper() or status_fallback

    if stage == "PLAN":
        if raw_status in {"PLAN", "SKIP"}:
            status = raw_status
        else:
            status = "PLAN"
    elif stage == "ACT":
        if raw_status in {"TOOL", "WRITE", "READ", "SEARCH", "ASK"}:
            status = raw_status
        else:
            status = "TOOL"
    elif stage == "VERIFY":
        if raw_status in {"PASS", "OK"}:
            status = "PASS"
        elif raw_status in {"FAIL", "ERROR", "DENY", "DENIED", "BLOCKED", "BLOCK"}:
            status = "BLOCK" if raw_status in {"DENY", "DENIED", "BLOCKED", "BLOCK"} else "FAIL"
        elif raw_status in {"DONE", "COMPLETE", "COMPLETED"}:
            status = "PASS"
        else:
            status = raw_status or "PASS"
    elif stage == "WRITEBACK":
        status = "DONE" if raw_status in {"DONE", "OK", "SUCCESS", "SUCCEEDED"} else raw_status
    else:
        status = raw_status

    return f"{stage}.{status}"

def _run_foundry_agent_goal(
    goal: str,
    max_rounds: int,
    dry_run: bool,
    dump_json: bool = False,
    role: str | None = None,
    report_path: str | None = None,
    orchestrator_mode: str | None = None,
) -> tuple[str | None, str | None]:
    script = project_root / "foundry-core" / "src" / "cli-agent.ts"
    if not script.exists():
        return None, "foundry-core cli-agent.ts not found"

    base_args = [
        str(project_root),
        "python",
        "run",
        "--goal",
        goal,
        "--max-rounds",
        str(max_rounds),
        "--role",
        (role or "planner"),
        "--orchestrator-mode",
        (orchestrator_mode or "adaptive"),
    ]
    if report_path:
        base_args.extend(["--report-path", report_path])
    if dry_run:
        base_args.append("--dry-run")
    if dump_json:
        base_args.append("--dump-json")

    dist_script = project_root / "foundry-core" / "dist" / "cli-agent.js"
    max_attempts = _foundry_retry_count()
    candidates = [
        ["node", str(script), *base_args],
        ["node", str(dist_script), *base_args] if dist_script.exists() else None,
        ["tsx", str(script), *base_args] if shutil.which("tsx") else None,
        ["ts-node", str(script), *base_args] if shutil.which("ts-node") else None,
        ["npx", "tsx", str(script), *base_args] if shutil.which("npx") else None,
    ]
    last_error = "foundry-core execution failed"

    for _ in range(max_attempts):
        for command in candidates:
            if not command:
                continue
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=90,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip(), None
                if result.stderr.strip():
                    last_error = f"foundry bridge error: {result.stderr.strip().splitlines()[-1]}"
                elif result.returncode != 0:
                    last_error = f"foundry bridge failed with code {result.returncode}"
            except FileNotFoundError:
                continue
            except Exception as error:
                last_error = f"foundry bridge exception: {error}"

    return None, last_error


def _coerce_foundry_payload(raw: str) -> dict[str, object]:
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {"status": "ok", "payload_type": "raw-text", "data": data}
    except Exception:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        for line in reversed(lines):
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    return data
            except Exception:
                continue
        return {
            "status": "ok",
            "payload_type": "raw-text",
            "data": raw.strip()[:4000],
        }


def _write_agent_run_report(payload: dict[str, object], report_path: str | None, source: str) -> None:
    if not report_path:
        return
    try:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema": "knowledge-hub.foundry.agent.run.report.v1",
                    "generatedAt": datetime.utcnow().isoformat(),
                    "source": source,
                    "run": payload,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as error:
        print(f"agent run report write failed: {error}", file=sys.stderr)


def _normalize_foundry_payload(payload: dict[str, object], goal: str, max_rounds: int, dry_run: bool) -> dict[str, object]:
    now = datetime.utcnow().isoformat()
    run_id = str(payload.get("runId") or payload.get("run_id") or f"foundry_{int(datetime.utcnow().timestamp())}")
    raw_status = str(payload.get("status", "")).upper().strip()
    if raw_status in {"RUNNING", "COMPLETED", "BLOCKED", "FAILED"}:
        status = raw_status.lower()
    elif raw_status in {"DONE", "OK", "SUCCESS", "DRY_RUN_OK", "VERIFY_OK"}:
        status = "completed"
    elif raw_status.startswith("DRY_RUN"):
        status = "blocked" if dry_run else "failed"
    elif "FAIL" in raw_status or "ERROR" in raw_status:
        status = "failed"
    elif not raw_status:
        status = "running" if not dry_run else "blocked"
    elif dry_run:
        status = "blocked"
    else:
        status = "completed"

    stage = str(payload.get("stage", "DONE" if status in {"completed", "blocked", "failed"} else "PLAN")).upper()

    source = str(payload.get("source", "knowledge-hub/mcp_server.fallback"))

    plan = payload.get("plan")
    if not isinstance(plan, list):
        trace_like = payload.get("trace")
        if isinstance(trace_like, list):
            plan = [str(step.get("step", step.get("action", ""))) for step in trace_like if isinstance(step, dict) and (step.get("step") or step.get("action"))]
        else:
            plan = []
    plan = [str(v) for v in plan if v]

    raw_transitions = payload.get("transitions")
    if not isinstance(raw_transitions, list):
        raw_transitions = payload.get("trace", [])

    base_source = str(payload.get("source", source))
    transitions: list[dict[str, object]] = []
    if isinstance(raw_transitions, list):
        for item in raw_transitions:
            if not isinstance(item, dict):
                continue
            status = item.get("status", item.get("action", item.get("step", "STEP")))
            message = item.get("message", item.get("action", item.get("step", "")))
            transitions.append({
                "stage": str(item.get("stage", "PLAN")).upper(),
                "status": str(status).upper(),
                "message": str(message),
                "tool": item.get("tool"),
                "step": item.get("step"),
                "action": item.get("action"),
                "source": item.get("source", base_source),
                "code": str(item.get("code", _transition_code(item))),
                "at": str(item.get("at", now)),
            })
    if not transitions:
        transitions = [{
            "stage": "PLAN",
            "status": "SKIP",
            "message": "no transitions",
            "tool": "knowledge-hub-fallback",
            "source": source,
            "code": "PLAN.SKIP",
            "at": now,
        }]

    verify_raw = payload.get("verify")
    is_completed = status == "completed"
    if isinstance(verify_raw, dict):
        verify = {
            "allowed": bool(verify_raw.get("allowed", is_completed)),
            "schemaValid": bool(verify_raw.get("schemaValid", verify_raw.get("schema_valid", is_completed))),
            "policyAllowed": bool(verify_raw.get("policyAllowed", verify_raw.get("policy_allowed", is_completed))),
            "schemaErrors": verify_raw.get("schemaErrors", verify_raw.get("errors", [])) or [],
        }
    else:
        verify = {
            "allowed": is_completed,
            "schemaValid": is_completed,
            "policyAllowed": is_completed,
            "schemaErrors": payload.get("errors", []),
        }
    if not is_completed:
        verify["allowed"] = False
        verify["schemaValid"] = False

    writeback_raw = payload.get("writeback")
    if isinstance(writeback_raw, dict):
        writeback = {
            "ok": bool(writeback_raw.get("ok", status in {"completed", "blocked"})),
            "detail": str(writeback_raw.get("detail", "")),
        }
    else:
        writeback = {
            "ok": bool(status in {"completed", "blocked"}),
            "detail": str(payload.get("writeback", "") if not isinstance(payload.get("writeback"), dict) else ""),
        }

    artifact = payload.get("artifact")
    if artifact is None:
        artifact = None
    elif not isinstance(artifact, dict):
        artifact = {"jsonContent": artifact, "classification": "P2", "generatedAt": now}

    policy_allowed, policy_errors = _evaluate_policy_gate(artifact)
    if not policy_allowed:
        status = "blocked"
        stage = "VERIFY"
        artifact = {
            "id": artifact.get("id") if isinstance(artifact, dict) else None,
            "jsonContent": "[REDACTED_BY_POLICY]",
            "classification": "P0",
            "generatedAt": now,
        } if artifact is not None else None

    merged_errors = list(verify.get("schemaErrors") or [])
    if policy_errors:
        merged_errors.extend(policy_errors)

    verify["policyAllowed"] = bool(policy_allowed)
    verify["allowed"] = bool(verify.get("allowed", status == "completed")) and policy_allowed
    verify["schemaValid"] = bool(verify.get("schemaValid", status == "completed")) and policy_allowed
    verify["schemaErrors"] = merged_errors

    if policy_errors:
        writeback["ok"] = False
        writeback["detail"] = "policy gate blocked"

    return {
        "schema": "knowledge-hub.foundry.agent.run.result.v1",
        "source": source,
        "runId": run_id,
        "status": status,
        "goal": str(payload.get("goal", goal)),
        "stage": stage,
        "tool": payload.get("tool"),
        "plan": plan,
        "transitions": transitions,
        "verify": verify,
        "writeback": writeback,
        "artifact": artifact,
        "createdAt": str(payload.get("createdAt", now)),
        "updatedAt": str(payload.get("updatedAt", now)),
        "dryRun": bool(payload.get("dryRun", payload.get("dry_run", dry_run))),
        "maxRounds": int(payload.get("maxRounds", max_rounds)),
    }


def _format_agent_result_text(payload: dict[str, object], compact: bool = False) -> str:
    transitions = payload.get("transitions", [])
    lines = [
        f"[runId] {payload.get('runId')}",
        f"[status] {payload.get('status')}",
        f"[stage] {payload.get('stage')}",
        f"[goal] {payload.get('goal')}",
    ]

    plan = payload.get("plan") or []
    if isinstance(plan, list) and plan:
        lines.append(f"[plan] {' -> '.join([str(s) for s in plan])}")

    verify = payload.get("verify")
    if isinstance(verify, dict):
        if not bool(verify.get("allowed", False)):
            lines.append(f"[verify] blocked: {verify.get('schemaErrors', [])}")
        else:
            lines.append("[verify] allowed")

    if isinstance(transitions, list) and transitions:
        lines.append("[trace]")
        limit = 3 if compact else 20
        for item in transitions[:limit]:
            if not isinstance(item, dict):
                continue
            at = item.get("at", "")
            stage = item.get("stage", "")
            status = item.get("status", "")
            message = item.get("message", item.get("action", item.get("step", "")))
            tool = item.get("tool")
            step = item.get("step")
            tool_text = f" [tool={tool}]" if tool else ""
            step_text = f" [step={step}]" if step else ""
            lines.append(f"- {at} {stage}.{status}:{tool_text}{step_text} {message}")

    artifact = payload.get("artifact")
    if isinstance(artifact, dict):
        content = artifact.get("jsonContent", "")
        content_preview = str(content)[:1200 if not compact else 300]
        if content_preview:
            lines.append(f"[artifact] {content_preview}")
        if compact:
            errors = []
            if isinstance(payload.get("verify"), dict):
                errors = payload["verify"].get("schemaErrors", [])
            if errors:
                lines.append(f"[errors] {'; '.join([str(e) for e in errors])}")
    return "\n".join(lines)


def _build_fallback_agent_payload(
    goal: str,
    max_rounds: int,
    dry_run: bool,
    plan: list[str],
    artifact,
    verify_ok: bool,
    errors: list[str],
    trace: list[dict],
    role: str | None = None,
    orchestrator_mode: str | None = None,
    playbook: dict[str, object] | None = None,
) -> str:
    now = datetime.utcnow().isoformat()
    transitions = []
    for item in trace:
        normalized_item = {
            "stage": str(item.get("stage", "PLAN")).upper(),
            "status": str(item.get("status", item.get("action", item.get("step", "STEP"))).upper()).strip() or "STEP",
            "message": str(item.get("message", item.get("action", item.get("step", "")))),
            "tool": str(item.get("tool", "knowledge-hub-fallback")),
            "step": item.get("step"),
            "action": item.get("action"),
            "at": now,
        }
        code = _transition_code(normalized_item)
        normalized_item["code"] = code
        normalized_item["status"] = code.split(".", 1)[-1]
        normalized_item["stage"] = code.split(".", 1)[0]
        transitions.append(normalized_item)
    if not transitions:
        transitions = [{
            "stage": "PLAN",
            "status": "SKIP",
            "message": "fallback-no-steps",
            "tool": "knowledge-hub-fallback",
            "code": "PLAN.SKIP",
            "at": now,
        }]
    artifact_payload = {
        "id": f"fallback_artifact_{int(datetime.utcnow().timestamp())}",
        "jsonContent": artifact,
        "classification": "P2",
        "generatedAt": now,
    }
    policy_allowed, policy_errors = _evaluate_policy_gate(artifact_payload)
    status = "completed" if verify_ok and policy_allowed else "blocked" if (dry_run or not policy_allowed) else "failed"
    if not policy_allowed:
        errors.extend(policy_errors)
        artifact_payload["jsonContent"] = "[REDACTED_BY_POLICY]"
        artifact_payload["classification"] = "P0"
    stage = "VERIFY" if not policy_allowed else ("DONE" if verify_ok else "FAILED")
    return json.dumps(
        {
            "schema": "knowledge-hub.foundry.agent.run.result.v1",
            "source": "knowledge-hub/mcp_server.fallback",
            "runId": f"fallback_{int(datetime.utcnow().timestamp())}",
            "goal": goal,
            "role": role or "planner",
            "orchestratorMode": orchestrator_mode or "adaptive",
            "maxRounds": max_rounds,
            "status": status,
            "plan": plan,
            "stage": stage,
            "playbook": playbook,
            "transitions": transitions,
            "verify": {
                "allowed": verify_ok and policy_allowed and not errors,
                "schemaValid": verify_ok and policy_allowed and not errors,
                "policyAllowed": policy_allowed,
                "schemaErrors": errors,
            },
            "artifact": artifact_payload,
            "writeback": {
                "ok": bool(verify_ok),
                "detail": "fallback writeback skipped" if dry_run else "fallback writeback completed",
            },
            "createdAt": now,
            "updatedAt": now,
            "dryRun": dry_run,
        },
        ensure_ascii=False,
    )


def initialize():
    global searcher, sqlite_db, config

    config_path = str(project_root / "config.yaml")
    if not Path(config_path).exists():
        config_path = None
    config = Config(config_path)

    embed_cfg = config.get_provider_config(config.embedding_provider)
    embedder = get_embedder(config.embedding_provider, model=config.embedding_model, **embed_cfg)

    summ_cfg = config.get_provider_config(config.summarization_provider)
    llm = get_llm(config.summarization_provider, model=config.summarization_model, **summ_cfg)

    vector_db = VectorDatabase(config.vector_db_path, config.collection_name)
    sqlite_db = SQLiteDatabase(config.sqlite_path)
    searcher = RAGSearcher(embedder, vector_db, llm)


app = Server("knowledge-hub")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_knowledge",
            description="통합 지식 검색 (Obsidian 노트 + 논문 + 웹 문서에서 의미론적 유사도 검색)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "검색 질문"},
                    "top_k": {"type": "integer", "description": "결과 수 (기본: 5)", "default": 5},
                    "source": {"type": "string", "description": "소스 필터: vault, paper, web (생략시 전체)"},
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
                    "min_score": {"type": "number", "description": "최소 유사도 (0~1)", "default": 0.3},
                },
                "required": ["question"],
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
                "dry_run": {"type": "boolean", "description": "검증만 수행", "default": False},
                "dump_json": {"type": "boolean", "description": "JSON 로그 반환", "default": False},
                "compact": {"type": "boolean", "description": "텍스트 모드에서 요약 출력", "default": False},
            },
            "required": ["goal"],
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
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent]:
    global searcher, sqlite_db

    if searcher is None:
        try:
            initialize()
        except Exception as e:
            return [TextContent(type="text", text=f"초기화 실패: {e}\nOllama 실행 확인: ollama serve")]

    try:
        if name == "search_knowledge":
            query = arguments.get("query")
            top_k = arguments.get("top_k", 5)
            source = arguments.get("source")
            results = searcher.search(query, top_k=top_k, source_type=source)

            if not results:
                return [TextContent(type="text", text="검색 결과가 없습니다.")]

            response = f"'{query}' 검색 결과 ({len(results)}개):\n\n"
            for i, r in enumerate(results, 1):
                title = r.metadata.get("title", "Untitled")
                src = r.metadata.get("source_type", "")
                response += f"{i}. **{title}** [{src}] (유사도: {r.score:.3f})\n"
                response += f"   {r.document[:200]}{'...' if len(r.document) > 200 else ''}\n\n"
            return [TextContent(type="text", text=response)]

        elif name == "ask_knowledge":
            question = arguments.get("question")
            top_k = arguments.get("top_k", 5)
            min_score = arguments.get("min_score", 0.3)
            result = searcher.generate_answer(question, top_k=top_k, min_score=min_score)

            response = f"**질문:** {question}\n\n**답변:**\n{result['answer']}\n\n"
            if result["sources"]:
                response += "**참고 문서:**\n"
                for i, s in enumerate(result["sources"], 1):
                    response += f"{i}. {s['title']} [{s['source_type']}] (유사도: {s['score']:.3f})\n"
            return [TextContent(type="text", text=response)]

        elif name == "run_agentic_query":
            goal = arguments.get("goal")
            if not goal:
                return [TextContent(type="text", text="goal이 필요합니다.")]

            max_rounds = int(arguments.get("max_rounds", 2))
            role = str(arguments.get("role", "planner"))
            report_path = arguments.get("reportPath") if isinstance(arguments.get("reportPath"), str) else None
            orchestrator_mode = arguments.get("orchestratorMode") if isinstance(arguments.get("orchestratorMode"), str) else None
            dry_run = bool(arguments.get("dry_run", False))
            dump_json = bool(arguments.get("dump_json", False))
            compact = bool(arguments.get("compact", False))

            role = role.strip().lower()
            if role not in {"planner", "researcher", "analyst", "summarizer", "auditor", "coach"}:
                role = "planner"
            orchestrator_mode = (orchestrator_mode or "adaptive").strip().lower()
            if orchestrator_mode not in {"single-pass", "adaptive", "strict"}:
                orchestrator_mode = "adaptive"

            delegated, delegated_err = _run_foundry_agent_goal(
                goal=goal,
                max_rounds=max_rounds,
                role=role,
                report_path=report_path,
                orchestrator_mode=orchestrator_mode,
                dry_run=dry_run,
                dump_json=dump_json,
            )
            if delegated:
                payload = _coerce_foundry_payload(delegated)
                payload = _normalize_foundry_payload(payload, goal=goal, max_rounds=max_rounds, dry_run=dry_run)
                payload["source"] = "foundry-core/cli-agent"
                if report_path:
                    _write_agent_run_report(payload, report_path, "foundry-core/cli-agent")
                if dump_json:
                    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]
                return [TextContent(type="text", text=_format_agent_result_text(payload, compact=compact))]
            if delegated_err:
                # foundry bridge 실패: 내부 폴백 추적 정보에 남김
                errors = [f"foundry bridge unavailable: {delegated_err}"]
            else:
                errors = []

            trace = []
            fallback_tokens = str(goal).lower()
            include_search = (
                role in {"planner", "researcher", "analyst", "coach"}
                or any(k in fallback_tokens for k in ["비교", "compare", "차이", "대조", "찾아", "검색", "search", "목록", "리스트", "추천"])
                or orchestrator_mode == "strict"
            )
            if role == "summarizer":
                include_search = False
            plan = ["search_knowledge", "ask_knowledge"] if include_search else ["ask_knowledge"]
            playbook = {
                "schema": "knowledge-hub.agent.playbook.v1",
                "source": "knowledge-hub/mcp_server.fallback",
                "goal": goal,
                "role": role,
                "orchestratorMode": orchestrator_mode,
                "maxRounds": max_rounds,
                "plan": plan,
            }
            artifact = None
            verify_ok = True

            for step in plan:
                trace.append({"stage": "PLAN", "step": step})
                if step == "search_knowledge":
                    search_results = searcher.search(goal, top_k=5)
                    artifact = [
                        {"title": r.metadata.get("title", "Untitled"), "score": r.score, "source_type": r.metadata.get("source_type", "")}
                        for r in search_results
                    ]
                    trace.append({"stage": "ACT", "step": step, "count": len(artifact)})
                else:
                    answer = searcher.generate_answer(goal, top_k=5)
                    artifact = answer
                    trace.append({"stage": "ACT", "step": step, "has_answer": bool(answer.get("answer"))})

                if not artifact:
                    verify_ok = False
                    errors.append(f"{step} produced empty artifact")

            if dump_json:
                fallback = _build_fallback_agent_payload(
                    goal=goal,
                    max_rounds=max_rounds,
                    dry_run=dry_run,
                    plan=plan,
                    artifact=artifact,
                    verify_ok=verify_ok,
                    errors=errors,
                    trace=trace,
                    role=role,
                    orchestrator_mode=orchestrator_mode,
                    playbook=playbook,
                )
                payload = _normalize_foundry_payload(
                    _coerce_foundry_payload(fallback),
                    goal=goal,
                    max_rounds=max_rounds,
                    dry_run=dry_run,
                )
                if report_path:
                    _write_agent_run_report(payload, report_path, "knowledge-hub/mcp_server.fallback")
                return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

            fallback = _build_fallback_agent_payload(
                goal=goal,
                max_rounds=max_rounds,
                dry_run=dry_run,
                plan=plan,
                artifact=artifact,
                verify_ok=verify_ok,
                errors=errors,
                trace=trace,
                role=role,
                orchestrator_mode=orchestrator_mode,
                playbook=playbook,
            )
            payload = _normalize_foundry_payload(
                _coerce_foundry_payload(fallback),
                goal=goal,
                max_rounds=max_rounds,
                dry_run=dry_run,
            )
            if report_path:
                _write_agent_run_report(payload, report_path, "knowledge-hub/mcp_server.fallback")
            return [TextContent(type="text", text=_format_agent_result_text(payload, compact=compact))]

        elif name == "search_papers":
            query = arguments.get("query")
            papers = sqlite_db.search_papers(query)
            if not papers:
                return [TextContent(type="text", text="논문을 찾을 수 없습니다.")]

            response = f"논문 검색 '{query}' ({len(papers)}개):\n\n"
            for p in papers:
                status = []
                if p.get("pdf_path"):
                    status.append("PDF")
                if p.get("translated_path"):
                    status.append("번역됨")
                response += f"- **{p['title']}** ({p.get('year', '?')}) [{p.get('field', '')}] {' | '.join(status)}\n"
                response += f"  arXiv: {p['arxiv_id']} | 중요도: {p.get('importance', '?')}\n\n"
            return [TextContent(type="text", text=response)]

        elif name == "index_paper_keywords":
            arxiv_id = arguments.get("arxiv_id")
            if arxiv_id:
                arxiv_id = str(arxiv_id).strip() or None
            try:
                top_k = int(arguments.get("top_k", 12))
            except Exception:
                top_k = 12
            try:
                max_links_per_keyword = int(arguments.get("max_links_per_keyword", 5))
            except Exception:
                max_links_per_keyword = 5
            dry_run = bool(arguments.get("dry_run", False))
            if top_k < 1:
                top_k = 1
            if max_links_per_keyword < 0:
                max_links_per_keyword = 0

            from knowledge_hub.papers.manager import PaperManager
            manager = PaperManager(
                config=config,
                vector_db=searcher.database,
                sqlite_db=sqlite_db,
                embedder=searcher.embedder,
            )
            result = manager.sync_translated_keywords(
                arxiv_id=arxiv_id,
                top_k=top_k,
                max_links_per_keyword=max_links_per_keyword,
                dry_run=dry_run,
            )
            payload = {
                "status": "ok",
                "mode": result["mode"],
                "processed": result["processed"],
                "updated": result["updated"],
                "skipped": result["skipped"],
                "target": "all" if arxiv_id is None else arxiv_id,
                "items": result["items"],
            }
            return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))]

        elif name == "get_hub_stats":
            sql_stats = sqlite_db.get_stats()
            vec_stats = searcher.database.get_stats()

            response = (
                f"**Knowledge Hub 통계**\n\n"
                f"- 노트: {sql_stats['notes']}개\n"
                f"- 논문: {sql_stats['papers']}개\n"
                f"- 태그: {sql_stats['tags']}개\n"
                f"- 링크: {sql_stats['links']}개\n"
                f"- 벡터 문서: {vec_stats['total_documents']}개\n"
                f"- 컬렉션: {vec_stats['collection_name']}\n"
            )
            return [TextContent(type="text", text=response)]

        elif name == "search_authors":
            from knowledge_hub.papers.discoverer import search_authors as _search_authors
            query = arguments.get("query", "")
            limit = int(arguments.get("limit", 10))
            authors = _search_authors(query, limit=limit)
            if not authors:
                return [TextContent(type="text", text="검색 결과가 없습니다.")]
            lines = [f"저자 검색: '{query}' ({len(authors)}명)\n"]
            for i, a in enumerate(authors, 1):
                affil = ", ".join(a.affiliations[:2]) if a.affiliations else "-"
                lines.append(
                    f"{i}. **{a.name}** (ID: {a.author_id})\n"
                    f"   소속: {affil} | 논문: {a.paper_count:,} | 인용: {a.citation_count:,} | h-index: {a.h_index}\n"
                )
            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "get_author_papers":
            from knowledge_hub.papers.discoverer import get_author_papers as _get_author_papers
            author_id = arguments.get("author_id", "")
            limit = int(arguments.get("limit", 20))
            author, papers = _get_author_papers(author_id, limit=limit)
            lines = []
            if author:
                lines.append(f"**{author.name}** — 논문: {author.paper_count:,} | 인용: {author.citation_count:,} | h-index: {author.h_index}\n")
            if not papers:
                lines.append("논문이 없습니다.")
            else:
                for i, p in enumerate(papers, 1):
                    fields = ", ".join(p.fields_of_study[:2]) if p.fields_of_study else ""
                    lines.append(f"{i}. **{p.title}** ({p.year}) — 인용: {p.citation_count:,} [{fields}] arXiv: {p.arxiv_id or '-'}")
            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "get_paper_detail":
            from knowledge_hub.papers.discoverer import get_paper_detail as _get_paper_detail
            paper_id = arguments.get("paper_id", "")
            data = _get_paper_detail(paper_id)
            if not data:
                return [TextContent(type="text", text=f"논문 '{paper_id}'를 찾을 수 없습니다.")]
            return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))]

        elif name == "get_paper_citations":
            from knowledge_hub.papers.discoverer import get_paper_citations as _get_citations
            paper_id = arguments.get("paper_id", "")
            limit = int(arguments.get("limit", 20))
            _, papers = _get_citations(paper_id, limit=limit)
            if not papers:
                return [TextContent(type="text", text="피인용 논문이 없습니다.")]
            lines = [f"'{paper_id}' 피인용 논문 ({len(papers)}편):\n"]
            for i, p in enumerate(papers, 1):
                lines.append(f"{i}. **{p.title}** ({p.year}) — 인용: {p.citation_count:,} | arXiv: {p.arxiv_id or '-'}")
            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "get_paper_references":
            from knowledge_hub.papers.discoverer import get_paper_references as _get_refs
            paper_id = arguments.get("paper_id", "")
            limit = int(arguments.get("limit", 20))
            _, papers = _get_refs(paper_id, limit=limit)
            if not papers:
                return [TextContent(type="text", text="참고문헌이 없습니다.")]
            lines = [f"'{paper_id}' 참고문헌 ({len(papers)}편):\n"]
            for i, p in enumerate(papers, 1):
                lines.append(f"{i}. **{p.title}** ({p.year}) — 인용: {p.citation_count:,} | arXiv: {p.arxiv_id or '-'}")
            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "analyze_citation_network":
            from knowledge_hub.papers.discoverer import analyze_citation_network as _analyze
            paper_id = arguments.get("paper_id", "")
            depth = int(arguments.get("depth", 1))
            cit_limit = int(arguments.get("citations_limit", 10))
            ref_limit = int(arguments.get("references_limit", 10))
            result = _analyze(paper_id, depth=depth, citations_limit=cit_limit, references_limit=ref_limit)
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

        elif name == "batch_paper_lookup":
            from knowledge_hub.papers.discoverer import get_papers_batch as _batch
            paper_ids = arguments.get("paper_ids", [])
            if not paper_ids:
                return [TextContent(type="text", text="paper_ids가 필요합니다.")]
            results = _batch(paper_ids)
            return [TextContent(type="text", text=json.dumps(results, ensure_ascii=False, indent=2))]

        elif name == "discover_and_ingest":
            topic = arguments.get("topic")
            if not topic:
                return [TextContent(type="text", text="topic이 필요합니다.")]

            max_papers = int(arguments.get("max_papers", 5))
            year_start = arguments.get("year_start")
            if year_start is not None:
                year_start = int(year_start)
            min_citations = int(arguments.get("min_citations", 0))
            sort_by = arguments.get("sort_by", "relevance")
            create_obsidian = bool(arguments.get("create_obsidian_note", True))
            gen_summary = bool(arguments.get("generate_summary", True))

            from knowledge_hub.papers.manager import PaperManager

            manager = PaperManager(
                config=config,
                vector_db=searcher.database,
                sqlite_db=sqlite_db,
                embedder=searcher.embedder,
            )
            llm_instance = searcher.llm if gen_summary else None

            result = manager.discover_and_ingest(
                topic=topic,
                max_papers=max_papers,
                year_start=year_start,
                min_citations=min_citations,
                sort_by=sort_by,
                create_obsidian_note=create_obsidian,
                generate_summary=gen_summary,
                llm=llm_instance,
            )

            lines = [f"**논문 자동 수집 결과** ({topic})\n"]
            lines.append(f"{result['message']}\n")

            if result["ingested"]:
                lines.append("### 수집된 논문\n")
                for i, p in enumerate(result["ingested"], 1):
                    lines.append(
                        f"{i}. **{p['title']}** ({p['year']})\n"
                        f"   arXiv: {p['arxiv_id']} | 인용: {p['citations']} | 분야: {', '.join(p['fields'])}\n"
                    )
                    if p.get("summary"):
                        lines.append(f"   > {p['summary'][:200]}...\n")

            if result.get("obsidian_notes_created"):
                lines.append(f"\n### Obsidian 노트 생성: {len(result['obsidian_notes_created'])}개")
                for path in result["obsidian_notes_created"]:
                    lines.append(f"  - {Path(path).name}")

            if result["failed"]:
                lines.append("\n### 실패")
                for f in result["failed"]:
                    lines.append(f"  - {f['arxiv_id']}: {f['error']}")

            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "check_paper_duplicate":
            arxiv_id = arguments.get("arxiv_id", "").strip()
            if not arxiv_id:
                return [TextContent(type="text", text="arxiv_id가 필요합니다.")]

            from knowledge_hub.papers.manager import PaperManager
            manager = PaperManager(
                config=config,
                vector_db=searcher.database,
                sqlite_db=sqlite_db,
                embedder=searcher.embedder,
            )
            is_dup, reason = manager.is_duplicate(arxiv_id)

            if is_dup:
                paper_info = sqlite_db.get_paper(arxiv_id)
                title = paper_info["title"] if paper_info else "N/A"
                response = f"**중복 발견** ({reason})\n\narXiv: {arxiv_id}\n제목: {title}"
            else:
                response = f"**중복 없음** - {arxiv_id}는 아직 수집되지 않았습니다."

            return [TextContent(type="text", text=response)]

        else:
            return [TextContent(type="text", text=f"알 수 없는 도구: {name}")]

    except Exception as e:
        import traceback
        return [TextContent(type="text", text=f"오류: {e}\n{traceback.format_exc()}")]


async def _async_main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main():
    """Entry point for khub-mcp command."""
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
