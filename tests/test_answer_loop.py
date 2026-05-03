from __future__ import annotations

import csv
import json
from pathlib import Path

from knowledge_hub.application import answer_loop as loop
from knowledge_hub.core.models import SearchResult
from knowledge_hub.core.schema_validator import validate_payload


class _StubSearcher:
    def __init__(self):
        self.config = type("Cfg", (), {})()
        self.sqlite_db = None

    def search(self, query, **kwargs):  # noqa: ANN001
        _ = (query, kwargs)
        return [
            SearchResult(
                document="Frozen evidence excerpt about grounded answer generation.",
                metadata={
                    "title": "Grounded Retrieval Note",
                    "source_type": "vault",
                    "file_path": "vault/grounded.md",
                },
                distance=0.1,
                score=0.91,
                semantic_score=0.88,
                lexical_score=0.55,
                retrieval_mode="hybrid",
                lexical_extras={},
                document_id="doc-1",
            )
        ]


class _StubLLM:
    def __init__(self, response: str):
        self.response = response
        self.last_policy = {}

    def generate(self, prompt, context="", max_tokens=None):  # noqa: ANN001
        _ = (prompt, context, max_tokens)
        return self.response


class _StubFactory:
    def __init__(self, *, llm_response: str = ""):
        self._searcher = _StubSearcher()
        self._llm_response = llm_response or "This answer stays grounded in the provided evidence."
        self.config = type("Cfg", (), {"get_nested": staticmethod(lambda *args, default=None: default)})()

    def get_searcher(self):
        return self._searcher

    def build_llm(self, provider, model=None):  # noqa: ANN001
        _ = (provider, model)
        return _StubLLM(self._llm_response)


def _write_queries(path: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query",
                "source",
                "query_type",
                "expected_primary_source",
                "expected_answer_style",
                "difficulty",
                "review_bucket",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "query": "How should grounded answers cite evidence?",
                "source": "vault",
                "query_type": "explanation",
                "expected_primary_source": "vault",
                "expected_answer_style": "grounded concise answer",
                "difficulty": "medium",
                "review_bucket": "groundedness",
            }
        )
    return path


def _write_implementation_queries(path: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query",
                "source",
                "query_type",
                "expected_primary_source",
                "expected_answer_style",
                "difficulty",
                "review_bucket",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "query": "repo 구조에서 entrypoint와 adapter 경계를 어떻게 찾는지 설명해줘",
                "source": "project",
                "query_type": "implementation",
                "expected_primary_source": "project",
                "expected_answer_style": "implementation_steps",
                "difficulty": "hard",
                "review_bucket": "project_architecture_boundary",
            }
        )
    return path


def test_build_answer_eval_packet_is_schema_valid():
    searcher = _StubSearcher()
    packet = loop.build_answer_eval_packet(
        searcher,
        {
            "query": "How should grounded answers cite evidence?",
            "source": "vault",
            "query_type": "explanation",
            "expected_primary_source": "vault",
            "expected_answer_style": "grounded concise answer",
        },
        packet_ref="packet-0001",
    )

    assert packet["packetRef"] == "packet-0001"
    assert packet["retrieved_sources"][0]["title"] == "Grounded Retrieval Note"
    assert validate_payload(packet, packet["schema"], strict=True).ok


def test_build_answer_eval_packet_uses_retrieval_pipeline_context_for_paper(monkeypatch):
    searcher = _StubSearcher()

    class _FakeFrame:
        def to_query_plan_dict(self):
            return {
                "family": "concept_explainer",
                "resolved_paper_ids": ["1706.03762"],
            }

    observed = {}

    class _FakePipeline:
        def __init__(self, inner_searcher):  # noqa: ANN001
            observed["searcher"] = inner_searcher

        def execute(self, **kwargs):  # noqa: ANN001
            observed["kwargs"] = kwargs
            return type(
                "Result",
                (),
                {
                    "results": [
                        SearchResult(
                            document="Attention Is All You Need summary chunk.",
                            metadata={
                                "title": "Attention Is All You Need",
                                "source_type": "paper",
                                "arxiv_id": "1706.03762",
                            },
                            distance=0.01,
                            score=0.99,
                            semantic_score=0.97,
                            lexical_score=0.91,
                            retrieval_mode="hybrid",
                            lexical_extras={},
                            document_id="paper-1706.03762-0",
                        )
                    ]
                },
            )()

    fake_domain_pack = type(
        "DomainPack",
        (),
        {
            "normalize": staticmethod(lambda *args, **kwargs: _FakeFrame()),
        },
    )()

    monkeypatch.setattr(loop, "RetrievalPipelineService", _FakePipeline)
    monkeypatch.setattr(loop, "get_domain_pack", lambda **kwargs: fake_domain_pack)

    packet = loop.build_answer_eval_packet(
        searcher,
        {
            "query": "Transformer의 핵심 아이디어를 비전공자도 이해할 수 있게 설명해줘",
            "source": "paper",
            "query_type": "definition",
            "expected_primary_source": "paper",
            "expected_answer_style": "beginner_explainer",
        },
        packet_ref="packet-paper-1",
    )

    assert observed["searcher"] is searcher
    assert observed["kwargs"]["query_frame"] is not None
    assert observed["kwargs"]["query_plan"]["resolved_paper_ids"] == ["1706.03762"]
    assert packet["retrieved_sources"][0]["title"] == "Attention Is All You Need"
    assert packet["retrieved_sources"][0]["source_ref"] == "https://arxiv.org/abs/1706.03762"


def test_build_answer_eval_packet_adds_axis_first_compare_guidance(monkeypatch):
    searcher = _StubSearcher()

    class _FakeFrame:
        def to_dict(self):
            return {
                "family": "paper_compare",
                "resolved_source_ids": ["alexnet-2012", "2010.11929"],
            }

        def to_query_plan_dict(self):
            return {
                "family": "paper_compare",
                "resolved_paper_ids": ["alexnet-2012", "2010.11929"],
            }

    class _FakePipeline:
        def __init__(self, inner_searcher):  # noqa: ANN001
            self.inner_searcher = inner_searcher

        def execute(self, **kwargs):  # noqa: ANN001
            _ = kwargs
            return type(
                "Result",
                (),
                {
                    "results": [
                        SearchResult(
                            document="AlexNet launched large-scale CNN vision classification.",
                            metadata={
                                "title": "ImageNet Classification with Deep Convolutional Neural Networks",
                                "source_type": "paper",
                                "paper_id": "alexnet-2012",
                            },
                            distance=0.01,
                            score=0.98,
                            semantic_score=0.96,
                            lexical_score=0.91,
                            retrieval_mode="hybrid",
                            lexical_extras={},
                            document_id="paper-alexnet-0",
                        ),
                        SearchResult(
                            document="Vision Transformer applies standard qkv self-attention to image patches.",
                            metadata={
                                "title": "An Image is Worth 16x16 Words",
                                "source_type": "paper",
                                "arxiv_id": "2010.11929",
                            },
                            distance=0.02,
                            score=0.97,
                            semantic_score=0.95,
                            lexical_score=0.9,
                            retrieval_mode="hybrid",
                            lexical_extras={},
                            document_id="paper-vit-0",
                        ),
                        SearchResult(
                            document="DeiT studies data-efficient image transformers with distillation.",
                            metadata={
                                "title": "Training data-efficient image transformers & distillation through attention",
                                "source_type": "paper",
                                "arxiv_id": "2012.12877",
                            },
                            distance=0.03,
                            score=0.9,
                            semantic_score=0.89,
                            lexical_score=0.82,
                            retrieval_mode="hybrid",
                            lexical_extras={},
                            document_id="paper-deit-0",
                        ),
                    ]
                },
            )()

    fake_domain_pack = type(
        "DomainPack",
        (),
        {
            "normalize": staticmethod(lambda *args, **kwargs: _FakeFrame()),
        },
    )()

    monkeypatch.setattr(loop, "RetrievalPipelineService", _FakePipeline)
    monkeypatch.setattr(loop, "get_domain_pack", lambda **kwargs: fake_domain_pack)

    packet = loop.build_answer_eval_packet(
        searcher,
        {
            "query": "CNN이랑 ViT를 논문 관점에서 비교해서 핵심 차이와 각각 잘하는 상황을 설명해줘",
            "source": "paper",
            "query_type": "comparison",
            "expected_primary_source": "paper",
            "expected_answer_style": "beginner_compare",
        },
        packet_ref="packet-compare-1",
    )

    guidance = dict(packet["guidance"])
    assert guidance["comparisonMode"] == "axis_first"
    assert guidance["comparisonTargets"] == [
        "ImageNet Classification with Deep Convolutional Neural Networks",
        "An Image is Worth 16x16 Words",
    ]
    assert "inductive bias and locality" in guidance["comparisonAxes"]
    assert len(guidance["axisEvidenceMatrix"]) == len(guidance["comparisonAxes"])
    assert guidance["axisEvidenceMatrix"][0]["axis"] == "core structure"
    assert guidance["axisEvidenceMatrix"][0]["coverage"] == "supported"
    assert guidance["axisEvidenceMatrix"][0]["perTarget"][0]["status"] == "direct"
    assert guidance["axisEvidenceMatrix"][0]["perTarget"][1]["status"] == "direct"
    data_axis = next(item for item in guidance["axisEvidenceMatrix"] if item["axis"] == "data and pretraining requirements")
    assert data_axis["coverage"] == "partial"
    assert data_axis["sharedSupport"][0]["title"] == "Training data-efficient image transformers & distillation through attention"
    assert guidance["responseSections"] == [
        "one-line difference",
        "axis comparison",
        "where each works better",
        "limits of current evidence",
    ]
    assert guidance["evidenceAuditRequired"] is True
    assert guidance["coreDifferenceEvidenceKinds"] == [
        "target_anchor",
        "direct_comparative_evidence",
        "background_evidence",
    ]
    assert packet["retrieved_sources"][0]["role"] == "target_anchor"
    assert packet["retrieved_sources"][0]["evidence_kind"] == "target_anchor"
    assert packet["retrieved_sources"][1]["role"] == "target_anchor"
    assert packet["retrieved_sources"][1]["evidence_kind"] == "target_anchor"
    assert packet["retrieved_sources"][2]["role"] == "supporting_evidence"
    assert packet["retrieved_sources"][2]["evidence_kind"] == "background_evidence"


def test_build_answer_eval_packet_uses_project_workspace_fallback_when_retrieval_empty(monkeypatch):
    searcher = _StubSearcher()

    class _EmptyPipeline:
        def __init__(self, inner_searcher):  # noqa: ANN001
            self.inner_searcher = inner_searcher

        def execute(self, **kwargs):  # noqa: ANN001
            _ = kwargs
            return type("Result", (), {"results": []})()

    monkeypatch.setattr(loop, "RetrievalPipelineService", _EmptyPipeline)
    monkeypatch.setattr(
        loop,
        "build_context_pack",
        lambda *args, **kwargs: {
            "workspace_sources": [
                {
                    "title": "README.md",
                    "source_type": "project",
                    "relative_path": "README.md",
                    "path": "/repo/README.md",
                    "snippet": "Canonical entrypoint and adapter boundary guide.",
                    "reason": "priority project document: README.md",
                },
                {
                    "title": "docs/PROJECT_STATE.md",
                    "source_type": "project",
                    "relative_path": "docs/PROJECT_STATE.md",
                    "path": "/repo/docs/PROJECT_STATE.md",
                    "snippet": "interfaces.* are canonical entrypoints; adapters stay secondary.",
                    "reason": "priority project document: docs/PROJECT_STATE.md",
                },
            ],
            "warnings": ["workspace snippets truncated: kept 2 files out of 7 candidates"],
        },
    )

    packet = loop.build_answer_eval_packet(
        searcher,
        {
            "query": "repo 구조에서 entrypoint와 adapter 경계를 어떻게 찾는지 설명해줘",
            "source": "project",
            "query_type": "implementation",
            "expected_primary_source": "project",
            "expected_answer_style": "implementation_steps",
        },
        packet_ref="packet-project-1",
        repo_path="/repo",
    )

    assert [item["title"] for item in packet["retrieved_sources"]] == ["README.md", "docs/PROJECT_STATE.md"]
    assert packet["retrieved_sources"][0]["source_type"] == "project"
    assert packet["retrieved_sources"][0]["source_ref"] == "README.md"
    assert packet["retrieved_sources"][0]["retrieval_mode"] == "context_pack"
    assert "project workspace fallback used after persistent retrieval returned no matching evidence" in packet["warnings"]


def test_collect_answer_loop_passes_repo_path_to_packet_builder(monkeypatch, tmp_path: Path):
    queries_path = _write_queries(tmp_path / "queries.csv")
    factory = _StubFactory()
    observed = {}

    def _fake_packet(searcher, query_row, **kwargs):  # noqa: ANN001
        _ = (searcher, query_row)
        observed["repo_path"] = kwargs.get("repo_path")
        return {
            "schema": loop.ANSWER_EVAL_PACKET_SCHEMA,
            "packetRef": "packet-0001",
            "question": "How should grounded answers cite evidence?",
            "source": "vault",
            "query_type": "explanation",
            "expected_primary_source": "vault",
            "expected_answer_style": "grounded concise answer",
            "difficulty": "medium",
            "review_bucket": "groundedness",
            "retrieved_sources": [],
            "warnings": [],
            "guidance": {"abstainPreferred": False},
            "runtimeDiagnostics": {"summary": "ok"},
        }

    monkeypatch.setattr(loop, "build_answer_eval_packet", _fake_packet)

    loop.collect_answer_loop(
        factory=factory,
        request=loop.CollectRequest(
            queries_path=str(queries_path),
            out_dir=str(tmp_path / "out"),
            answer_backends=(loop.ANSWER_BACKEND_OPENAI_GPT5_MINI,),
            repo_path="/repo",
        ),
    )

    assert observed["repo_path"] == "/repo"


def test_answer_prompt_uses_compare_axis_rules():
    packet = {
        "question": "CNN vs ViT",
        "expected_primary_source": "paper",
        "expected_answer_style": "beginner_compare",
        "guidance": {
            "abstainPreferred": False,
            "comparisonMode": "axis_first",
            "comparisonTaskHint": "Do not answer as a separate summary of model A and model B. Compare them on shared axes first.",
            "comparisonAxes": [
                "core structure",
                "inductive bias and locality",
                "data and pretraining requirements",
            ],
            "comparisonTargets": ["CNN", "ViT"],
            "responseSections": ["one-line difference", "axis comparison", "limits of current evidence"],
            "coreDifferenceEvidenceKinds": ["target_anchor", "background_evidence"],
            "exampleOnlyEvidenceKinds": ["task_specific_example", "weak_indirect_evidence"],
            "axisEvidenceMatrix": [
                {
                    "axis": "core structure",
                    "coverage": "supported",
                    "perTarget": [
                        {"target": "CNN", "status": "direct", "sourceTitle": "AlexNet"},
                        {"target": "ViT", "status": "direct", "sourceTitle": "ViT"},
                    ],
                    "sharedSupport": [],
                }
            ],
        },
    }

    prompt = loop._answer_prompt(packet)

    assert "Compare targets: CNN, ViT." in prompt
    assert "Fixed comparison axes: core structure, inductive bias and locality, data and pretraining requirements." in prompt
    assert "Do not answer as a separate summary of model A and model B." in prompt
    assert "Match the language and register of the user's question" in prompt
    assert "Axis evidence matrix:" in prompt
    assert "- core structure => supported; CNN: direct via AlexNet; ViT: direct via ViT" in prompt
    assert "Use only these evidence kinds for the main difference claim: target_anchor, background_evidence." in prompt
    assert "Treat these kinds as example-only support, not as the main difference claim: task_specific_example, weak_indirect_evidence." in prompt


def test_evidence_kind_marks_task_specific_examples_for_compare_queries():
    kind = loop._evidence_kind(
        query_row={
            "query_type": "comparison",
            "expected_primary_source": "paper",
        },
        metadata={
            "title": "You Only Look Once: Unified, Real-Time Object Detection",
            "source_type": "paper",
        },
        document="A real-time object detection architecture for detection tasks.",
        role="supporting_evidence",
    )

    assert kind == "task_specific_example"


def test_judge_prompt_mentions_system_layers():
    prompt = loop._judge_prompt(
        {
            "packet": {
                "question": "CNN vs ViT",
                "expected_primary_source": "paper",
                "expected_answer_style": "beginner_compare",
            },
            "result": {
                "backend": "codex_mcp",
                "model": "",
                "status": "ok",
                "answer_text": "comparison answer",
            },
        }
    )

    assert "query understanding, retrieval fit, evidence grounding, answer assembly, reasoning quality, and overclaim risk" in prompt
    assert "dominant failure layer: MODEL, RETRIEVAL, ASSEMBLY, or PROMPT" in prompt


def test_packet_context_rounds_score_to_avoid_long_numeric_spans():
    context = loop._packet_context(
        {
            "retrieved_sources": [
                {
                    "rank": 1,
                    "role": "target_anchor",
                    "evidence_kind": "target_anchor",
                    "title": "AlexNet",
                    "source_type": "paper",
                    "source_ref": "paper:alexnet-2012",
                    "score": 0.6728188544273376,
                    "excerpt": "example excerpt",
                }
            ]
        }
    )

    assert "score=0.6728" in context
    assert "6728188544273376" not in context


def test_verify_compare_answer_accepts_low_coverage_axis_when_answer_marks_insufficient():
    verification = loop._verify_compare_answer(
        {
            "retrieved_sources": [{"title": "BERT"}, {"title": "GPT-4"}],
            "guidance": {
                "axisEvidenceMatrix": [
                    {
                        "axis": "compute and resolution scaling",
                        "coverage": "partial",
                        "perTarget": [],
                        "sharedSupport": [],
                    }
                ]
            },
            "warnings": [],
        },
        {
            "answer_text": "compute와 resolution scaling: 이 축은 현재 근거가 부족해서 단정할 수 없습니다.",
        },
    )

    assert verification["status"] == "axis_matrix_checked"
    assert verification["unsupportedCount"] == 0


def test_verify_compare_answer_flags_low_coverage_axis_when_answer_overclaims():
    verification = loop._verify_compare_answer(
        {
            "retrieved_sources": [{"title": "BERT"}, {"title": "GPT-4"}],
            "guidance": {
                "axisEvidenceMatrix": [
                    {
                        "axis": "compute and resolution scaling",
                        "coverage": "partial",
                        "perTarget": [],
                        "sharedSupport": [],
                    }
                ]
            },
            "warnings": [],
        },
        {
            "answer_text": "compute와 resolution scaling: GPT가 항상 더 잘 확장됩니다.",
        },
    )

    assert verification["status"] == "axis_gap_detected"
    assert verification["unsupportedCount"] == 1
    assert "compute and resolution scaling" in verification["summary"]
    assert verification["unsupportedSegments"][0]["axis"] == "compute and resolution scaling"
    assert "GPT가 항상 더 잘 확장됩니다" in verification["unsupportedSegments"][0]["segment"]


def test_answer_segments_split_sentences_and_tables():
    segments = loop._answer_segments(
        "첫 문장입니다. 둘째 문장입니다.\n| 축 | 설명 |\n|---|---|\n| compute | 항상 더 좋다 |\n마지막 문장입니다."
    )

    assert "첫 문장입니다." in segments
    assert "둘째 문장입니다." in segments
    assert "| 축 | 설명 |" in segments
    assert "| compute | 항상 더 좋다 |" in segments
    assert "마지막 문장입니다." in segments


def test_serialize_collect_row_uses_compare_verification():
    row = loop._serialize_collect_row(
        {
            "query": "BERT vs GPT",
            "source": "paper",
            "query_type": "comparison",
            "expected_primary_source": "paper",
            "expected_answer_style": "beginner_compare",
            "difficulty": "medium",
            "review_bucket": "source_fit",
        },
        {
            "packetRef": "packet-0001",
            "retrieved_sources": [{"title": "BERT paper", "source_ref": "paper:bert"}],
            "guidance": {
                "axisEvidenceMatrix": [
                    {
                        "axis": "compute and resolution scaling",
                        "coverage": "partial",
                        "perTarget": [],
                        "sharedSupport": [],
                    }
                ]
            },
            "warnings": [],
            "runtimeDiagnostics": {"summary": "ok"},
        },
        {
            "status": "ok",
            "answer_text": "compute와 resolution scaling: GPT가 항상 더 잘 확장됩니다.",
            "backend": "codex_mcp",
            "model": "gpt-5.4",
            "latency_ms": 123,
            "warnings": [],
        },
        top_k=8,
        retrieval_mode="hybrid",
    )

    assert row["verification_status"] == "axis_gap_detected"
    assert row["unsupported_claim_count"] == "1"


def test_collect_and_judge_answer_loop_preserves_final_columns(tmp_path: Path):
    factory = _StubFactory(
        llm_response='{"pred_label":"good","pred_groundedness":"good","pred_usefulness":"good","pred_readability":"good","pred_source_accuracy":"good","pred_should_abstain":"0","pred_confidence":"0.9","pred_reason":"well grounded"}'
    )
    queries = _write_queries(tmp_path / "queries.csv")
    collect_payload = loop.collect_answer_loop(
        factory=factory,
        request=loop.CollectRequest(
            queries_path=str(queries),
            out_dir=str(tmp_path / "out"),
            answer_backends=(loop.ANSWER_BACKEND_OPENAI_GPT5_MINI,),
            repo_path=str(tmp_path),
        ),
    )
    judge_payload = loop.judge_answer_loop(
        factory=factory,
        collect_manifest_path=str((collect_payload.get("artifactPaths") or {}).get("manifestPath")),
        judge_model="gpt-5",
    )

    judged_rows = list(csv.DictReader(Path((judge_payload["artifactPaths"] or {})["judgedCsvPath"]).open("r", encoding="utf-8")))
    assert judged_rows[0]["answer_backend"] == "openai_gpt5_mini"
    assert judged_rows[0]["pred_label"] == "good"
    assert judged_rows[0]["judge_model"] == "gpt-5"
    assert judged_rows[0]["final_label"] == ""
    assert validate_payload(judge_payload, judge_payload["schema"], strict=True).ok


def test_run_codex_tool_sync_uses_exec_transport_by_default(tmp_path: Path, monkeypatch):
    observed = {}

    def _fake_run(**kwargs):  # noqa: ANN003
        observed["prompt"] = kwargs["prompt"]
        observed["cwd"] = kwargs["cwd"]
        observed["model"] = kwargs["model"]
        observed["task_type"] = kwargs["task_type"]
        return {
            "isError": False,
            "threadId": "thread-123",
            "content": "codex answer",
            "structuredContent": {"transport": "exec"},
        }

    monkeypatch.setattr(loop, "_run_codex_tool_sync_impl", _fake_run)

    payload = loop._run_codex_tool_sync(
        config=_StubFactory().config,
        prompt="hello from codex",
        cwd=str(tmp_path),
        sandbox="read-only",
        approval_policy="never",
        model="gpt-5.4",
    )

    assert observed["prompt"] == "hello from codex"
    assert observed["cwd"] == str(tmp_path)
    assert observed["model"] == "gpt-5.4"
    assert observed["task_type"] == "rag_answer"
    assert payload["isError"] is False
    assert payload["threadId"] == "thread-123"
    assert payload["content"] == "codex answer"
    assert payload["structuredContent"]["transport"] == "exec"


def test_sanitize_answer_text_removes_bracketed_reference_markers():
    raw = "핵심은 attention입니다 [1].\n\n이 방식은 병렬화에 유리합니다 (Source 1)."
    assert loop._sanitize_answer_text(raw) == "핵심은 attention입니다.\n\n이 방식은 병렬화에 유리합니다."


def test_summary_and_failure_buckets(tmp_path: Path):
    judged_csv = tmp_path / "answer_loop_judged.csv"
    with judged_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=loop.ANSWER_LOOP_FIELDNAMES)
        writer.writeheader()
        writer.writerow(
            {
                "query": "Q1",
                "expected_answer_style": "abstain if weak",
                "answer_status": "ok",
                "answer_backend": "openai_gpt5_mini",
                "packet_ref": "packet-1",
                "pred_label": "partial",
                "pred_groundedness": "bad",
                "pred_usefulness": "partial",
                "pred_readability": "good",
                "pred_source_accuracy": "bad",
                "pred_should_abstain": "0",
                "pred_reason": "weak grounding",
            }
        )
    judge_manifest = {
        "schema": loop.ANSWER_LOOP_JUDGE_SCHEMA,
        "status": "ok",
        "judgeProvider": "openai",
        "judgeModel": "gpt-5",
        "rowCount": 1,
        "artifactPaths": {
            "judgedCsvPath": str(judged_csv),
            "judgeManifestPath": str(tmp_path / "answer_loop_judge_manifest.json"),
        },
    }
    judge_manifest_path = tmp_path / "answer_loop_judge_manifest.json"
    judge_manifest_path.write_text(json.dumps(judge_manifest), encoding="utf-8")

    summary = loop.summarize_answer_loop(judge_manifest_path=str(judge_manifest_path))

    assert summary["overall"]["predLabelScore"] == 0.5
    assert summary["failureBucketCounts"]["groundedness_failure"] == 1
    assert summary["failureBucketCounts"]["source_accuracy_failure"] == 1
    assert summary["failureBucketCounts"]["abstention_failure"] == 1
    assert validate_payload(summary, summary["schema"], strict=True).ok


def test_autofix_blocks_on_dirty_tree(tmp_path: Path, monkeypatch):
    judged_csv = tmp_path / "answer_loop_judged.csv"
    with judged_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=loop.ANSWER_LOOP_FIELDNAMES)
        writer.writeheader()
    judge_manifest_path = tmp_path / "answer_loop_judge_manifest.json"
    judge_manifest_path.write_text(
        json.dumps(
            {
                "schema": loop.ANSWER_LOOP_JUDGE_SCHEMA,
                "status": "ok",
                "judgeProvider": "openai",
                "judgeModel": "gpt-5",
                "rowCount": 0,
                "artifactPaths": {"judgedCsvPath": str(judged_csv)},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(loop, "_git_status", lambda repo_path: " M dirty.py\n")

    payload = loop.autofix_answer_loop(
        factory=_StubFactory(),
        judge_manifest_path=str(judge_manifest_path),
        repo_path=str(tmp_path),
        allow_dirty=False,
    )

    assert payload["status"] == "blocked"
    assert payload["reason"] == "dirty_worktree"
    assert validate_payload(payload, payload["schema"], strict=True).ok


def test_load_collect_records_returns_empty_when_collect_manifest_path_is_missing():
    assert loop._load_collect_records({"artifactPaths": {}}) == []


def test_patch_executor_runs_codex_diff_and_verification(monkeypatch):
    codex_calls = []
    git_name_calls = []
    git_patch_calls = []

    def _fake_codex_run(**kwargs):  # noqa: ANN003
        codex_calls.append(dict(kwargs))
        return {
            "isError": False,
            "threadId": "patch-thread",
            "structuredContent": {"warnings": ["minor warning"]},
        }

    def _fake_git_diff_names(repo_path):  # noqa: ANN001
        git_name_calls.append(repo_path)
        if len(git_name_calls) == 1:
            return ["already_changed.py"]
        return ["already_changed.py", "knowledge_hub/application/answer_loop.py"]

    def _fake_git_diff_patch(repo_path):  # noqa: ANN001
        git_patch_calls.append(repo_path)
        return "before" if len(git_patch_calls) == 1 else "after"

    monkeypatch.setattr(loop, "_run_codex_tool_sync", _fake_codex_run)
    monkeypatch.setattr(loop, "_git_diff_names", _fake_git_diff_names)
    monkeypatch.setattr(loop, "_git_diff_patch", _fake_git_diff_patch)
    monkeypatch.setattr(
        loop,
        "_run_targeted_verification",
        lambda repo_path: {"status": "ok", "command": ["pytest"], "returncode": 0, "outputPreview": "ok"},  # noqa: ARG005
    )

    payload = loop._build_patch_executor(factory=_StubFactory()).run(
        prompt="patch prompt",
        repo_path="/repo",
        patch_model="gpt-5.4",
    )

    assert codex_calls[0]["cwd"] == "/repo"
    assert codex_calls[0]["model"] == "gpt-5.4"
    assert payload["status"] == "ok"
    assert payload["changedFiles"] == ["knowledge_hub/application/answer_loop.py"]
    assert payload["warnings"] == ["minor warning"]
    assert payload["verification"]["status"] == "ok"


def test_autofix_routes_patch_execution_through_patch_executor(tmp_path: Path, monkeypatch):
    judged_csv = tmp_path / "answer_loop_judged.csv"
    with judged_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=loop.ANSWER_LOOP_FIELDNAMES)
        writer.writeheader()
        writer.writerow(
            {
                "query": "How should grounded answers cite evidence?",
                "answer_backend": "openai_gpt5_mini",
                "packet_ref": "packet-1",
                "pred_label": "partial",
                "pred_groundedness": "partial",
                "pred_usefulness": "good",
                "pred_readability": "good",
                "pred_source_accuracy": "good",
                "pred_should_abstain": "0",
                "pred_reason": "needs tighter citation grounding",
                "expected_answer_style": "grounded concise answer",
            }
        )
    judge_manifest_path = tmp_path / "answer_loop_judge_manifest.json"
    judge_manifest_path.write_text(
        json.dumps(
            {
                "schema": loop.ANSWER_LOOP_JUDGE_SCHEMA,
                "status": "ok",
                "judgeProvider": "openai",
                "judgeModel": "gpt-5",
                "rowCount": 1,
                "artifactPaths": {"judgedCsvPath": str(judged_csv)},
            }
        ),
        encoding="utf-8",
    )

    observed = {}

    class _StubPatchExecutor:
        def run(self, *, prompt, repo_path, patch_model):  # noqa: ANN001
            observed["prompt"] = prompt
            observed["repo_path"] = repo_path
            observed["patch_model"] = patch_model
            return {
                "status": "ok",
                "reason": "",
                "changedFiles": ["knowledge_hub/application/answer_loop.py"],
                "warnings": ["stub-warning"],
                "backendTrace": {"threadId": "patch-thread"},
                "verification": {"status": "ok"},
            }

    monkeypatch.setattr(
        loop,
        "summarize_answer_loop",
        lambda **kwargs: {  # noqa: ARG005
            "schema": loop.ANSWER_LOOP_SUMMARY_SCHEMA,
            "status": "ok",
            "rowCount": 1,
            "overall": {},
            "backends": {},
            "failureBucketCounts": {bucket: 0 for bucket in loop.FAILURE_BUCKETS},
            "failureCardCount": 1,
        },
    )
    monkeypatch.setattr(loop, "_build_patch_executor", lambda **kwargs: _StubPatchExecutor())  # noqa: ARG005

    payload = loop.autofix_answer_loop(
        factory=_StubFactory(),
        judge_manifest_path=str(judge_manifest_path),
        repo_path=str(tmp_path),
        patch_model="gpt-5.4",
    )

    assert "needs tighter citation grounding" in observed["prompt"]
    assert observed["repo_path"] == str(tmp_path)
    assert observed["patch_model"] == "gpt-5.4"
    assert payload["status"] == "ok"
    assert payload["changedFiles"] == ["knowledge_hub/application/answer_loop.py"]
    assert payload["warnings"] == ["stub-warning"]
    assert payload["backendTrace"]["threadId"] == "patch-thread"
    assert validate_payload(payload, payload["schema"], strict=True).ok


def test_post_collect_executor_uses_direct_application_path(monkeypatch):
    observed = {}

    def _fake_judge(**kwargs):  # noqa: ANN003
        observed["judge"] = dict(kwargs)
        return {
            "schema": loop.ANSWER_LOOP_JUDGE_SCHEMA,
            "status": "ok",
            "judgeProvider": "openai",
            "judgeModel": "gpt-5",
            "rowCount": 1,
            "artifactPaths": {"judgeManifestPath": "/tmp/judge-manifest.json"},
        }

    def _fake_summarize(**kwargs):  # noqa: ANN003
        observed["summarize"] = dict(kwargs)
        return {
            "schema": loop.ANSWER_LOOP_SUMMARY_SCHEMA,
            "status": "ok",
            "rowCount": 1,
            "overall": {},
            "backends": {},
            "failureBucketCounts": {bucket: 0 for bucket in loop.FAILURE_BUCKETS},
            "failureCardCount": 0,
        }

    monkeypatch.setattr(loop, "judge_answer_loop", _fake_judge)
    monkeypatch.setattr(loop, "summarize_answer_loop", _fake_summarize)

    executor = loop._build_post_collect_executor(factory=_StubFactory(), repo_path="/repo")
    judge_payload, summary_payload = executor.run(
        collect_manifest_path="/tmp/collect-manifest.json",
        judge_model="gpt-5",
    )

    assert executor.config_path is None
    assert observed["judge"]["collect_manifest_path"] == "/tmp/collect-manifest.json"
    assert observed["judge"]["judge_model"] == "gpt-5"
    assert observed["summarize"]["judge_manifest_path"] == "/tmp/judge-manifest.json"
    assert judge_payload["schema"] == loop.ANSWER_LOOP_JUDGE_SCHEMA
    assert summary_payload["schema"] == loop.ANSWER_LOOP_SUMMARY_SCHEMA


def test_post_collect_executor_uses_cli_commands_with_canonical_labs_prefix(monkeypatch):
    calls = []

    class _ConfigFactory(_StubFactory):
        def __init__(self):
            super().__init__()
            self._config_path = "/tmp/khub-config.yaml"

    def _fake_cli_json(*, repo_path, config_path, args):  # noqa: ANN003
        calls.append(
            {
                "repo_path": repo_path,
                "config_path": config_path,
                "args": list(args),
            }
        )
        if args[:4] == ["labs", "eval", "answer-loop", "judge"]:
            return {
                "schema": loop.ANSWER_LOOP_JUDGE_SCHEMA,
                "status": "ok",
                "judgeProvider": "openai",
                "judgeModel": "gpt-5",
                "rowCount": 1,
                "artifactPaths": {"judgeManifestPath": "/tmp/judge-manifest.json"},
            }
        if args[:4] == ["labs", "eval", "answer-loop", "summarize"]:
            return {
                "schema": loop.ANSWER_LOOP_SUMMARY_SCHEMA,
                "status": "ok",
                "rowCount": 1,
                "overall": {},
                "backends": {},
                "failureBucketCounts": {bucket: 0 for bucket in loop.FAILURE_BUCKETS},
                "failureCardCount": 0,
            }
        raise AssertionError(args)

    monkeypatch.setattr(loop, "_run_cli_json_command", _fake_cli_json)

    executor = loop._build_post_collect_executor(factory=_ConfigFactory(), repo_path="/repo")
    judge_payload, summary_payload = executor.run(
        collect_manifest_path="/tmp/collect-manifest.json",
        judge_model="gpt-5",
    )

    assert executor.config_path == "/tmp/khub-config.yaml"
    assert calls[0]["args"][:4] == ["labs", "eval", "answer-loop", "judge"]
    assert calls[1]["args"][:4] == ["labs", "eval", "answer-loop", "summarize"]
    assert judge_payload["schema"] == loop.ANSWER_LOOP_JUDGE_SCHEMA
    assert summary_payload["schema"] == loop.ANSWER_LOOP_SUMMARY_SCHEMA


def test_post_collect_executor_uses_direct_autofix_path(monkeypatch):
    observed = {}

    def _fake_autofix(**kwargs):  # noqa: ANN003
        observed.update(dict(kwargs))
        return {
            "schema": loop.ANSWER_LOOP_AUTOFIX_SCHEMA,
            "status": "ok",
            "repoPath": kwargs["repo_path"],
            "changedFiles": [],
            "warnings": [],
            "artifactPaths": {},
            "backendTrace": {},
            "verification": {},
        }

    monkeypatch.setattr(loop, "autofix_answer_loop", _fake_autofix)

    executor = loop._build_post_collect_executor(factory=_StubFactory(), repo_path="/repo")
    payload = executor.autofix(
        judge_manifest_path="/tmp/judge-manifest.json",
        allow_dirty=True,
        patch_model="gpt-5.4",
    )

    assert observed["judge_manifest_path"] == "/tmp/judge-manifest.json"
    assert observed["repo_path"] == "/repo"
    assert observed["allow_dirty"] is True
    assert observed["patch_model"] == "gpt-5.4"
    assert payload["schema"] == loop.ANSWER_LOOP_AUTOFIX_SCHEMA


def test_post_collect_executor_uses_cli_autofix_with_canonical_labs_prefix(monkeypatch):
    calls = []

    class _ConfigFactory(_StubFactory):
        def __init__(self):
            super().__init__()
            self._config_path = "/tmp/khub-config.yaml"

    def _fake_cli_json(*, repo_path, config_path, args):  # noqa: ANN003
        calls.append(
            {
                "repo_path": repo_path,
                "config_path": config_path,
                "args": list(args),
            }
        )
        return {
            "schema": loop.ANSWER_LOOP_AUTOFIX_SCHEMA,
            "status": "ok",
            "repoPath": repo_path,
            "changedFiles": [],
            "warnings": [],
            "artifactPaths": {},
            "backendTrace": {},
            "verification": {},
        }

    monkeypatch.setattr(loop, "_run_cli_json_command", _fake_cli_json)

    executor = loop._build_post_collect_executor(factory=_ConfigFactory(), repo_path="/repo")
    payload = executor.autofix(
        judge_manifest_path="/tmp/judge-manifest.json",
        allow_dirty=True,
        patch_model="gpt-5.4",
    )

    assert calls[0]["config_path"] == "/tmp/khub-config.yaml"
    assert calls[0]["args"][:4] == ["labs", "eval", "answer-loop", "autofix"]
    assert "--judge-manifest" in calls[0]["args"]
    assert "--repo-path" in calls[0]["args"]
    assert "--allow-dirty" in calls[0]["args"]
    assert "--patch-model" in calls[0]["args"]
    assert payload["schema"] == loop.ANSWER_LOOP_AUTOFIX_SCHEMA


def test_run_collect_via_cli_uses_canonical_labs_eval_prefix(tmp_path: Path, monkeypatch):
    observed = {}

    class _ConfigFactory(_StubFactory):
        def __init__(self):
            super().__init__()
            self._config_path = "/tmp/khub-config.yaml"

    def _fake_cli_json(*, repo_path, config_path, args):  # noqa: ANN003
        observed["repo_path"] = repo_path
        observed["config_path"] = config_path
        observed["args"] = list(args)
        return {
            "schema": loop.ANSWER_LOOP_COLLECT_SCHEMA,
            "status": "ok",
            "rowCount": 1,
            "packetCount": 1,
            "artifactPaths": {"manifestPath": str(tmp_path / "collect-manifest.json")},
        }

    monkeypatch.setattr(loop, "_run_cli_json_command", _fake_cli_json)

    payload = loop._run_collect_via_cli(
        factory=_ConfigFactory(),
        request=loop.CollectRequest(
            queries_path=str(_write_queries(tmp_path / "queries.csv")),
            out_dir=str(tmp_path / "out"),
            answer_backends=(loop.ANSWER_BACKEND_OPENAI_GPT5_MINI,),
            repo_path=str(tmp_path),
        ),
        repo_path=str(tmp_path),
    )

    assert observed["config_path"] == "/tmp/khub-config.yaml"
    assert observed["args"][:4] == ["labs", "eval", "answer-loop", "collect"]
    assert payload["schema"] == loop.ANSWER_LOOP_COLLECT_SCHEMA


def test_run_answer_loop_stops_on_no_improvement(tmp_path: Path, monkeypatch):
    collect_calls = []

    def _fake_collect(*, factory, request):  # noqa: ANN001
        collect_calls.append(request.out_dir)
        manifest_path = Path(request.out_dir) / "answer_loop_collect_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": loop.ANSWER_LOOP_COLLECT_SCHEMA,
            "status": "ok",
            "request": request.to_dict(),
            "rowCount": 1,
            "packetCount": 1,
            "artifactPaths": {"manifestPath": str(manifest_path), "csvPath": str(manifest_path.with_suffix(".csv")), "recordsPath": str(manifest_path.with_suffix(".jsonl"))},
        }
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    summary_values = iter(
        [
            {
                "schema": loop.ANSWER_LOOP_SUMMARY_SCHEMA,
                "status": "ok",
                "rowCount": 1,
                "overall": {
                    "predLabelScore": 0.70,
                    "predGroundednessScore": 0.70,
                    "predSourceAccuracyScore": 0.70,
                    "abstainAgreement": 0.0,
                },
                "backends": {},
                "failureBucketCounts": {bucket: 0 for bucket in loop.FAILURE_BUCKETS},
                "failureCardCount": 0,
            },
            {
                "schema": loop.ANSWER_LOOP_SUMMARY_SCHEMA,
                "status": "ok",
                "rowCount": 1,
                "overall": {
                    "predLabelScore": 0.71,
                    "predGroundednessScore": 0.70,
                    "predSourceAccuracyScore": 0.70,
                    "abstainAgreement": 0.0,
                },
                "backends": {},
                "failureBucketCounts": {bucket: 0 for bucket in loop.FAILURE_BUCKETS},
                "failureCardCount": 0,
            },
        ]
    )

    monkeypatch.setattr(loop, "collect_answer_loop", _fake_collect)
    monkeypatch.setattr(
        loop,
        "judge_answer_loop",
        lambda **kwargs: {  # noqa: ARG005
            "schema": loop.ANSWER_LOOP_JUDGE_SCHEMA,
            "status": "ok",
            "judgeProvider": "openai",
            "judgeModel": "gpt-5",
            "rowCount": 1,
            "artifactPaths": {"judgeManifestPath": str(tmp_path / f"judge-{len(collect_calls)}.json")},
        },
    )
    monkeypatch.setattr(loop, "summarize_answer_loop", lambda **kwargs: next(summary_values))  # noqa: ARG005
    monkeypatch.setattr(loop, "autofix_answer_loop", lambda **kwargs: {"schema": loop.ANSWER_LOOP_AUTOFIX_SCHEMA, "status": "ok"})  # noqa: ARG005

    payload = loop.run_answer_loop(
        factory=_StubFactory(),
        request=loop.CollectRequest(
            queries_path=str(_write_queries(tmp_path / "queries.csv")),
            out_dir=str(tmp_path / "run"),
            answer_backends=(loop.ANSWER_BACKEND_OPENAI_GPT5_MINI,),
            repo_path=str(tmp_path),
        ),
        judge_model="gpt-5",
        max_attempts=3,
        repo_path=str(tmp_path),
    )

    assert payload["attemptCount"] == 2
    assert payload["stoppedReason"] == "no_improvement"
    assert payload["bestAttempt"] == 1
    assert validate_payload(payload, payload["schema"], strict=True).ok


def test_optimize_answer_loop_is_codex_only_and_schema_valid(tmp_path: Path, monkeypatch):
    class _NoLLMFactory(_StubFactory):
        def build_llm(self, provider, model=None):  # noqa: ANN001
            raise AssertionError(f"optimize should not call build_llm: {provider} {model}")

    def _fake_codex_run(**kwargs):  # noqa: ANN003
        prompt = str(kwargs.get("prompt") or "")
        if "Return exactly one JSON object" in prompt:
            return {
                "isError": False,
                "threadId": "judge-thread",
                "content": '{"pred_label":"good","pred_groundedness":"good","pred_usefulness":"good","pred_readability":"good","pred_source_accuracy":"good","pred_should_abstain":"0","pred_confidence":"0.9","pred_reason":"grounded"}',
                "structuredContent": {"warnings": []},
            }
        if "You are revising a user-facing answer" in prompt:
            return {
                "isError": False,
                "threadId": "revise-thread",
                "content": "Revised grounded answer that stays within the packet evidence.",
                "structuredContent": {"warnings": []},
            }
        return {
            "isError": False,
            "threadId": "answer-thread",
            "content": "Baseline grounded answer that stays within the packet evidence.",
            "structuredContent": {"warnings": []},
        }

    monkeypatch.setattr(loop, "_run_codex_tool_sync", _fake_codex_run)

    payload = loop.optimize_answer_loop(
        factory=_NoLLMFactory(),
        request=loop.CollectRequest(
            queries_path=str(_write_queries(tmp_path / "queries.csv")),
            out_dir=str(tmp_path / "opt"),
            repo_path=str(tmp_path),
        ),
        repo_path=str(tmp_path),
        generator_model="gpt-5.4",
        judge_model="gpt-5.4",
        daily_token_budget_estimate=100000,
        max_rounds=1,
    )

    assert payload["schema"] == loop.ANSWER_LOOP_OPTIMIZE_SCHEMA
    assert payload["generatorModel"] == "gpt-5.4"
    assert payload["judgeModel"] == "gpt-5.4"
    assert payload["estimatedTokens"]["judge"] > 0
    assert validate_payload(payload, payload["schema"], strict=True).ok


def test_optimize_answer_loop_stops_when_judge_budget_is_exhausted(tmp_path: Path, monkeypatch):
    def _fake_codex_run(**kwargs):  # noqa: ANN003
        prompt = str(kwargs.get("prompt") or "")
        if "Return exactly one JSON object" in prompt:
            return {
                "isError": False,
                "threadId": "judge-thread",
                "content": '{"pred_label":"partial","pred_groundedness":"partial","pred_usefulness":"partial","pred_readability":"good","pred_source_accuracy":"partial","pred_should_abstain":"0","pred_confidence":"0.5","pred_reason":"needs work"}',
                "structuredContent": {"warnings": []},
            }
        return {
            "isError": False,
            "threadId": "answer-thread",
            "content": "Baseline grounded answer.",
            "structuredContent": {"warnings": []},
        }

    monkeypatch.setattr(loop, "_run_codex_tool_sync", _fake_codex_run)

    payload = loop.optimize_answer_loop(
        factory=_StubFactory(),
        request=loop.CollectRequest(
            queries_path=str(_write_queries(tmp_path / "queries.csv")),
            out_dir=str(tmp_path / "opt-budget"),
            repo_path=str(tmp_path),
        ),
        repo_path=str(tmp_path),
        generator_model="gpt-5.4",
        judge_model="gpt-5.4",
        daily_token_budget_estimate=1000,
        judge_budget_ratio=0.10,
    )

    assert payload["stopReason"] == "judge_budget_exhausted"
    assert Path(payload["artifactPaths"]["bestAnswersPath"]).exists()
    assert Path(payload["artifactPaths"]["resultPath"]).exists()


def test_optimize_answer_loop_does_not_starve_operational_judge_due_to_holdout_reserve(tmp_path: Path, monkeypatch):
    original_prompt = loop._codex_judge_prompt

    def _huge_holdout_prompt(record, *, variant="operational"):  # noqa: ANN001
        prompt = original_prompt(record, variant=variant)
        if variant == "holdout":
            return prompt + "\n" + ("reserve " * 5000)
        return prompt

    def _fake_codex_run(**kwargs):  # noqa: ANN003
        prompt = str(kwargs.get("prompt") or "")
        if "Return exactly one JSON object" in prompt:
            return {
                "isError": False,
                "threadId": "judge-thread",
                "content": '{"pred_label":"good","pred_groundedness":"good","pred_usefulness":"good","pred_readability":"good","pred_source_accuracy":"good","pred_should_abstain":"0","pred_confidence":"0.9","pred_reason":"good"}',
                "structuredContent": {"warnings": []},
            }
        return {
            "isError": False,
            "threadId": "answer-thread",
            "content": "A grounded baseline answer.",
            "structuredContent": {"warnings": []},
        }

    monkeypatch.setattr(loop, "_codex_judge_prompt", _huge_holdout_prompt)
    monkeypatch.setattr(loop, "_run_codex_tool_sync", _fake_codex_run)

    payload = loop.optimize_answer_loop(
        factory=_StubFactory(),
        request=loop.CollectRequest(
            queries_path=str(_write_queries(tmp_path / "queries.csv")),
            out_dir=str(tmp_path / "opt-holdout-reserve"),
            repo_path=str(tmp_path),
        ),
        repo_path=str(tmp_path),
        generator_model="gpt-5.4",
        judge_model="gpt-5.4",
        daily_token_budget_estimate=5000,
        judge_budget_ratio=0.10,
    )

    assert payload["estimatedTokens"]["judge"] > 0
    assert payload["bestCandidates"]


def test_optimize_answer_loop_stops_when_total_budget_is_exhausted(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        loop,
        "_run_codex_tool_sync",
        lambda **kwargs: {  # noqa: ARG005
            "isError": False,
            "threadId": "answer-thread",
            "content": "Budget limited answer.",
            "structuredContent": {"warnings": []},
        },
    )

    payload = loop.optimize_answer_loop(
        factory=_StubFactory(),
        request=loop.CollectRequest(
            queries_path=str(_write_queries(tmp_path / "queries.csv")),
            out_dir=str(tmp_path / "opt-total-budget"),
            repo_path=str(tmp_path),
        ),
        repo_path=str(tmp_path),
        generator_model="gpt-5.4",
        judge_model="gpt-5.4",
        daily_token_budget_estimate=20,
    )

    assert payload["stopReason"] == "total_budget_exhausted"
    assert Path(payload["artifactPaths"]["candidateLogPath"]).exists()


def test_optimize_answer_loop_reuses_frozen_packets(monkeypatch, tmp_path: Path):
    build_calls = []

    original = loop.build_answer_eval_packet

    def _tracked_packet(*args, **kwargs):  # noqa: ANN003
        build_calls.append(kwargs.get("packet_ref"))
        return original(*args, **kwargs)

    def _fake_codex_run(**kwargs):  # noqa: ANN003
        prompt = str(kwargs.get("prompt") or "")
        if "Return exactly one JSON object" in prompt:
            return {
                "isError": False,
                "threadId": "judge-thread",
                "content": '{"pred_label":"partial","pred_groundedness":"partial","pred_usefulness":"good","pred_readability":"good","pred_source_accuracy":"partial","pred_should_abstain":"0","pred_confidence":"0.5","pred_reason":"needs revision"}',
                "structuredContent": {"warnings": []},
            }
        if "You are revising a user-facing answer" in prompt:
            return {
                "isError": False,
                "threadId": "revision-thread",
                "content": "Revised answer.",
                "structuredContent": {"warnings": []},
            }
        return {
            "isError": False,
            "threadId": "answer-thread",
            "content": "Baseline answer.",
            "structuredContent": {"warnings": []},
        }

    monkeypatch.setattr(loop, "build_answer_eval_packet", _tracked_packet)
    monkeypatch.setattr(loop, "_run_codex_tool_sync", _fake_codex_run)

    payload = loop.optimize_answer_loop(
        factory=_StubFactory(),
        request=loop.CollectRequest(
            queries_path=str(_write_queries(tmp_path / "queries.csv")),
            out_dir=str(tmp_path / "opt-frozen"),
            repo_path=str(tmp_path),
        ),
        repo_path=str(tmp_path),
        generator_model="gpt-5.4",
        judge_model="gpt-5.4",
        max_rounds=2,
        daily_token_budget_estimate=100000,
    )

    assert build_calls == ["packet-0001"]
    candidate_rows = [json.loads(line) for line in Path(payload["artifactPaths"]["candidateLogPath"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert {row["packetRef"] for row in candidate_rows} == {"packet-0001"}


def test_optimization_revision_prompt_keeps_private_rubric_hidden():
    candidate = {
        "packet": {
            "question": "How should grounded answers cite evidence?",
            "expected_answer_style": "grounded concise answer",
            "retrieved_sources": [{"title": "Grounded Retrieval Note", "source_type": "vault", "excerpt": "Use the provided evidence only."}],
            "guidance": {"abstainPreferred": False},
        },
        "answerText": "Draft answer",
        "judge": {
            "pred_label": "partial",
            "pred_groundedness": "bad",
            "pred_usefulness": "partial",
            "pred_readability": "good",
            "pred_source_accuracy": "bad",
            "pred_should_abstain": "0",
        },
    }

    prompt = loop._optimization_revision_prompt(candidate, candidate_index=1, candidate_count=2)
    operational = loop._codex_judge_prompt({"packet": candidate["packet"], "result": {"answer_text": "Draft answer"}}, variant="operational")
    holdout = loop._codex_judge_prompt({"packet": candidate["packet"], "result": {"answer_text": "Draft answer"}}, variant="holdout")

    assert "Return exactly one JSON object" not in prompt
    assert "final private audit judge" not in prompt
    assert "retrieval fit, evidence grounding, answer assembly" not in prompt
    assert "iterative optimization" in operational
    assert "final private audit judge" in holdout


def test_answer_prompt_adds_implementation_style_navigation_guidance():
    prompt = loop._answer_prompt(
        {
            "question": "repo 구조에서 entrypoint와 adapter 경계를 어떻게 찾는지 설명해줘",
            "expected_primary_source": "project",
            "expected_answer_style": "implementation_steps",
            "guidance": {"abstainPreferred": False},
            "retrieved_sources": [],
        }
    )

    assert "concrete step-by-step repo-navigation procedure" in prompt
    assert "next inspection step rather than a confirmed fact" in prompt


def test_candidate_successful_requires_usefulness_for_implementation_steps():
    candidate = {
        "packet": {
            "expected_answer_style": "implementation_steps",
            "guidance": {"abstainPreferred": False},
        },
        "judge": {
            "pred_label": "good",
            "pred_groundedness": "good",
            "pred_source_accuracy": "good",
            "pred_usefulness": "partial",
            "pred_readability": "good",
            "pred_should_abstain": "0",
        },
    }

    assert loop._candidate_successful(candidate, judge_key="judge") is False


def test_candidate_replacement_respects_metric_priority():
    packet = {"expected_answer_style": "grounded concise answer", "guidance": {"abstainPreferred": False}}
    incumbent = {
        "packet": packet,
        "judge": {
            "pred_label": "good",
            "pred_groundedness": "good",
            "pred_source_accuracy": "good",
            "pred_usefulness": "partial",
            "pred_readability": "partial",
            "pred_should_abstain": "0",
        },
    }
    challenger = {
        "packet": packet,
        "judge": {
            "pred_label": "good",
            "pred_groundedness": "partial",
            "pred_source_accuracy": "good",
            "pred_usefulness": "good",
            "pred_readability": "good",
            "pred_should_abstain": "0",
        },
    }

    assert loop._candidate_better(challenger, incumbent, judge_key="judge") is False


def test_anti_gaming_penalties_flag_rubric_copy_and_unsupported_security_claim():
    payload, flags = loop._apply_anti_gaming_penalties(
        judge_payload={
            "pred_label": "good",
            "pred_groundedness": "good",
            "pred_usefulness": "good",
            "pred_readability": "good",
            "pred_source_accuracy": "good",
            "pred_should_abstain": "0",
            "pred_reason": "fine",
        },
        answer_text="pred_label 기준으로 보면 good 입니다. 이 답변은 정보 누출과 보안 정책 위반 위험도 있습니다.",
        packet={
            "question": "How should grounded answers cite evidence?",
            "retrieved_sources": [{"title": "Grounded Retrieval Note", "excerpt": "Use the evidence only."}],
        },
    )

    assert "rubric_copy" in flags
    assert "unsupported_security_privacy_claim" in flags
    assert payload["pred_label"] == "bad"
    assert payload["pred_source_accuracy"] == "bad"


def test_optimize_answer_loop_revises_implementation_style_answers(tmp_path: Path, monkeypatch):
    def _fake_packet(searcher, query_row, *, packet_ref, top_k=8, repo_path="", **kwargs):  # noqa: ANN001
        _ = (searcher, top_k, repo_path, kwargs)
        return {
            "schema": loop.ANSWER_EVAL_PACKET_SCHEMA,
            "packetRef": packet_ref,
            "question": query_row["query"],
            "source": query_row["source"],
            "query_type": query_row["query_type"],
            "expected_primary_source": query_row["expected_primary_source"],
            "expected_answer_style": query_row["expected_answer_style"],
            "difficulty": query_row["difficulty"],
            "review_bucket": query_row["review_bucket"],
            "retrieved_sources": [
                {
                    "rank": 1,
                    "title": "Repo CLI Guide",
                    "source_type": "project",
                    "source_ref": "docs/guides/cli-commands.md",
                    "paper_id": "",
                    "role": "entrypoint",
                    "evidence_kind": "supporting_evidence",
                    "score": 0.91,
                    "semantic_score": 0.88,
                    "lexical_score": 0.62,
                    "retrieval_mode": "hybrid",
                    "excerpt": "Start with knowledge_hub/cli/, knowledge_hub/mcp/, and docs/guides/cli-commands.md to trace entrypoints and adapter boundaries.",
                }
            ],
            "warnings": [],
            "guidance": {"abstainPreferred": False},
            "runtimeDiagnostics": {"summary": "ok"},
        }

    def _fake_codex_run(**kwargs):  # noqa: ANN003
        prompt = str(kwargs.get("prompt") or "")
        if "Return exactly one JSON object" in prompt:
            if "Answer text:\n1. docs/guides/cli-commands.md에서 surface를 확인한다." in prompt:
                return {
                    "isError": False,
                    "threadId": "judge-thread-good",
                    "content": '{"pred_label":"good","pred_groundedness":"good","pred_usefulness":"good","pred_readability":"good","pred_source_accuracy":"good","pred_should_abstain":"0","pred_confidence":"0.9","pred_reason":"grounded repo-navigation steps"}',
                    "structuredContent": {"warnings": []},
                }
            return {
                "isError": False,
                "threadId": "judge-thread-partial",
                "content": '{"pred_label":"partial","pred_groundedness":"good","pred_usefulness":"partial","pred_readability":"good","pred_source_accuracy":"good","pred_should_abstain":"0","pred_confidence":"0.7","pred_reason":"needs more concrete repo-navigation steps"}',
                "structuredContent": {"warnings": []},
            }
        if "You are revising a user-facing answer" in prompt:
            assert "repo-navigation steps weak" in prompt
            return {
                "isError": False,
                "threadId": "revision-thread",
                "content": "1. docs/guides/cli-commands.md에서 surface를 확인한다.\n2. knowledge_hub/cli/와 knowledge_hub/mcp/에서 entrypoint를 추적한다.\n3. packet에 없는 정확한 심볼은 다음 inspection step으로 표시한다.",
                "structuredContent": {"warnings": []},
            }
        return {
            "isError": False,
            "threadId": "answer-thread",
            "content": "문서와 구조를 먼저 본다.",
            "structuredContent": {"warnings": []},
        }

    monkeypatch.setattr(loop, "build_answer_eval_packet", _fake_packet)
    monkeypatch.setattr(loop, "_run_codex_tool_sync", _fake_codex_run)

    payload = loop.optimize_answer_loop(
        factory=_StubFactory(),
        request=loop.CollectRequest(
            queries_path=str(_write_implementation_queries(tmp_path / "implementation.csv")),
            out_dir=str(tmp_path / "opt-implementation"),
            repo_path=str(tmp_path),
        ),
        repo_path=str(tmp_path),
        generator_model="gpt-5.4",
        judge_model="gpt-5.4",
        candidate_count=1,
        max_rounds=1,
        daily_token_budget_estimate=100000,
    )

    assert payload["roundCount"] == 1
    assert payload["bestCandidates"][0]["candidateId"] == "packet-0001-r01-c01-revision"
    best_answers = Path(payload["artifactPaths"]["bestAnswersPath"]).read_text(encoding="utf-8")
    assert "knowledge_hub/cli/" in best_answers
