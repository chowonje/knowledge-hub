"""rag_answer_logs must record which runtime (ask_v2 vs legacy) served each answer."""

from types import SimpleNamespace

import knowledge_hub.ai.answer_orchestrator as answer_orchestrator_module
from knowledge_hub.ai.answer_orchestrator import AnswerOrchestrator
from knowledge_hub.ai.rag_answer_runtime import AnswerRuntimeExecution, RAGAnswerRuntime
from knowledge_hub.ai.rag_support import record_answer_log


def _capture_recorder(captured):
    def recorder(**kwargs):
        captured.update(kwargs)

    return recorder


def _base_payload(**extra):
    payload = {
        "answer": "x",
        "answerVerification": {"status": "verified"},
        "router": {"selected": {"route": "local", "provider": "ollama", "model": "m"}},
        "sources": [{"title": "t"}],
        "evidence": [{"title": "t"}],
        "warnings": [],
    }
    payload.update(extra)
    return payload


def test_record_answer_log_includes_runtime_markers():
    captured: dict = {}
    payload = _base_payload(
        runtimeExecution={"used": "ask_v2", "fallbackReason": "", "askV2HardGate": True}
    )
    record_answer_log(
        _capture_recorder(captured),
        query="q",
        payload=payload,
        source_type="paper",
        retrieval_mode="hybrid",
        allow_external=False,
    )
    route = dict(captured.get("answer_route") or {})
    assert route.get("runtimeUsed") == "ask_v2"
    assert route.get("askV2HardGate") is True
    assert route.get("runtimeFallbackReason") == ""


def test_record_answer_log_without_marker_keeps_legacy_shape():
    captured: dict = {}
    record_answer_log(
        _capture_recorder(captured),
        query="q",
        payload=_base_payload(),
        source_type="paper",
        retrieval_mode="hybrid",
        allow_external=False,
    )
    route = dict(captured.get("answer_route") or {})
    assert "runtimeUsed" not in route


def test_generate_via_orchestrator_passes_runtime_execution(monkeypatch):
    captured: dict = {}

    class _FakeOrchestrator:
        def __init__(self, searcher):
            pass

        def generate(self, **kwargs):
            captured.update(kwargs)
            return {"answer": "x"}

    monkeypatch.setattr(answer_orchestrator_module, "AnswerOrchestrator", _FakeOrchestrator)
    runtime = RAGAnswerRuntime(searcher=SimpleNamespace())
    pipeline_result = SimpleNamespace(
        v2_diagnostics={
            "runtimeExecution": {
                "used": "legacy",
                "fallbackReason": "ask_v2_not_used",
                "sectionDecision": "skipped",
                "sectionBlockReason": "",
            }
        }
    )
    execution = AnswerRuntimeExecution(pipeline_result=pipeline_result, evidence_packet=None)
    request = RAGAnswerRuntime.build_request(query="q", source_type="paper")
    runtime._generate_via_orchestrator(request=request, execution=execution)
    runtime_execution = dict(captured.get("runtime_execution") or {})
    assert runtime_execution.get("used") == "legacy"
    assert runtime_execution.get("fallbackReason") == "ask_v2_not_used"


def test_orchestrator_record_attaches_runtime_meta_to_payload_and_log():
    captured: dict = {}
    searcher = SimpleNamespace(
        sqlite_db=SimpleNamespace(add_rag_answer_log=_capture_recorder(captured))
    )
    orchestrator = AnswerOrchestrator(searcher)
    orchestrator._runtime_execution_meta = {
        "used": "legacy",
        "fallbackReason": "ask_v2_not_used",
        "askV2HardGate": False,
    }
    payload = _base_payload()
    orchestrator._record_answer_log(
        query="q",
        payload=payload,
        source_type="paper",
        retrieval_mode="hybrid",
        allow_external=False,
    )
    assert payload.get("runtimeExecution", {}).get("used") == "legacy"
    route = dict(captured.get("answer_route") or {})
    assert route.get("runtimeUsed") == "legacy"
    assert route.get("runtimeFallbackReason") == "ask_v2_not_used"
