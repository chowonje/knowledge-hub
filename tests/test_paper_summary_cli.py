from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from knowledge_hub.core.sanitizer import classify_payload_level
from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.document_memory import builder as document_memory_builder_module
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.interfaces.cli.commands.paper_summary_cmd import paper_summary_group
from knowledge_hub.learning.task_router import TaskRouteDecision
from knowledge_hub.papers import raw_summary as raw_summary_module
from knowledge_hub.papers import structured_summary as structured_summary_module
from knowledge_hub.papers.mineru_adapter import MinerUParseResult
from knowledge_hub.papers.opendataloader_adapter import OpenDataLoaderParseResult
from knowledge_hub.papers.pymupdf_adapter import PyMuPDFParseResult
from knowledge_hub.papers.structured_summary import StructuredPaperSummaryService


class _StubConfig:
    translation_provider = ""
    translation_model = ""
    summarization_provider = ""
    summarization_model = ""

    def __init__(self, *, papers_dir: str = "", data: dict | None = None):
        self._papers_dir = papers_dir
        self._data = dict(data or {})

    def get_nested(self, *args, default=None):  # noqa: ANN002, ANN003
        node: object = self._data
        for key in args:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node

    def get_provider_config(self, provider):  # noqa: ANN001
        _ = provider
        return {}

    @property
    def papers_dir(self) -> str:
        return self._papers_dir


class _StubKhub:
    def __init__(self, db: SQLiteDatabase, *, papers_dir: str, config_data: dict | None = None):
        self._db = db
        self.config = _StubConfig(papers_dir=papers_dir, data=config_data)

    def sqlite_db(self):
        return self._db


def _seed(db: SQLiteDatabase):
    db.upsert_paper(
        {
            "arxiv_id": "2603.13017",
            "title": "Personalized Agent Memory",
            "authors": "A. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": (
                "### 한줄 요약\n\n기존 요약 경로 샘플.\n\n"
                "### 핵심 기여\n\n- 세션 압축\n\n"
                "### 방법론\n\n- 메모리 카드\n"
            ),
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
        }
    )
    db.upsert_note(
        note_id="paper:2603.13017",
        title="[논문] Personalized Agent Memory",
        content=(
            "# Personalized Agent Memory\n\n"
            "## Abstract\n\n"
            "This paper compresses long-running agent sessions into searchable memory cards.\n\n"
            "## Method\n\n"
            "The system builds memory cards, retrieves them for future tasks, and reranks them with context signals.\n\n"
            "## Results\n\n"
            "On a developer-agent benchmark, success rate improves by 8.2 points over a baseline memory buffer.\n\n"
            "## Limitations\n\n"
            "Performance drops when retrieval recall is low or when domain shift changes the task distribution.\n"
        ),
        source_type="paper",
        metadata={"arxiv_id": "2603.13017"},
    )
    db.upsert_ontology_entity(
        entity_id="paper:2603.13017",
        entity_type="paper",
        canonical_name="Personalized Agent Memory",
        source="test",
    )
    db.upsert_claim(
        claim_id="claim:2603.13017:1",
        claim_text="Success rate improves by 8.2 points on a developer-agent benchmark over a baseline memory buffer.",
        subject_entity_id="paper:2603.13017",
        predicate="improves",
        object_entity_id=None,
        object_literal="baseline memory buffer",
        confidence=0.91,
        evidence_ptrs=[{"note_id": "paper:2603.13017", "claim_decision": "accepted"}],
        source="test",
    )
    db.upsert_claim(
        claim_id="claim:2603.13017:2",
        claim_text="Performance drops when retrieval recall is low under domain shift.",
        subject_entity_id="paper:2603.13017",
        predicate="limits",
        object_entity_id=None,
        object_literal="domain shift",
        confidence=0.81,
        evidence_ptrs=[{"note_id": "paper:2603.13017", "claim_decision": "accepted"}],
        source="test",
    )


class _FakeLLM:
    def generate(self, prompt: str, context: str = "", max_tokens: int | None = None) -> str:  # noqa: ARG002
        return json.dumps(
            {
                "oneLine": "장기 에이전트 세션을 메모리 카드로 압축해 재사용하는 구조를 제안한다.",
                "problem": "긴 세션에서 과거 문맥을 다시 찾기 어렵다는 문제를 다룬다.",
                "coreIdea": "세션을 구조화된 메모리 카드로 저장하고 후속 작업에서 검색해 활용한다.",
                "methodSteps": [
                    "세션에서 중요한 이벤트를 카드로 압축한다.",
                    "새 작업에서 관련 카드를 검색하고 재랭킹한다.",
                ],
                "keyResults": [
                    "developer-agent benchmark에서 baseline memory buffer 대비 성공률이 8.2포인트 향상된다."
                ],
                "limitations": [
                    "retrieval recall이 낮거나 domain shift가 크면 성능이 떨어진다."
                ],
                "whenItMatters": "긴 세션을 유지하는 코딩/작업형 에이전트에서 특히 중요하다.",
                "whatIsNew": "대화 로그가 아니라 검색 가능한 memory card 단위를 중심으로 설계한다.",
                "confidenceNotes": ["results와 limitations는 document-memory와 claim hints를 함께 참고했다."],
            },
            ensure_ascii=False,
        )


def test_collect_paper_text_prefers_pdf_text_over_refusal_notes(tmp_path, monkeypatch):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")
    paper = {
        "arxiv_id": "2603.13018",
        "title": "Raw PDF Runtime Paper",
        "notes": "원문을 보내주시면 요약하겠습니다.",
        "pdf_path": str(pdf_path),
        "text_path": "",
        "translated_path": "",
    }
    monkeypatch.setattr(
        raw_summary_module,
        "extract_pdf_text_excerpt",
        lambda *args, **kwargs: "This PDF includes the real abstract, method, and result details." * 4,
    )

    text = raw_summary_module.collect_paper_text(paper)

    assert "real abstract, method, and result details" in text
    assert "원문을 보내주시면" not in text


def test_collect_paper_text_strips_latex_preamble_before_abstract(tmp_path):
    text_path = tmp_path / "paper.txt"
    text_path.write_text(
        "\\documentclass{article}\n"
        "\\usepackage{foo}\n"
        "\\title{Zep}\n"
        "\\author{Example}\n"
        "\\begin{abstract} We introduce Zep, a temporal knowledge graph memory layer for agents. "
        "It improves memory retrieval and long-horizon QA. \\end{abstract}\n"
        "\\section{Introduction} Agents need evolving memory.\n",
        encoding="utf-8",
    )
    paper = {
        "arxiv_id": "2501.13956",
        "title": "Zep",
        "notes": "citations: 0",
        "pdf_path": "",
        "text_path": str(text_path),
        "translated_path": "",
    }

    text = raw_summary_module.collect_paper_text(paper)

    assert "We introduce Zep" in text
    assert "\\documentclass" not in text
    assert "citations: 0" not in text


def test_structured_summary_normalize_prefers_grounded_fallback_over_noisy_llm_values():
    llm_value = {
        "oneLine": "General reasoning represents a long-standing and formidable challenge in artificial intelligence.",
        "problem": "[2501.12948 > Page 1] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning DeepSeek-AI research@deepseek.com Abstract General reasoning represents a long-standing challenge.",
        "coreIdea": "of Lagrange multipliers. Let me set up the Lagrangian.",
        "methodSteps": [
            "[2501.12948 > Page 52] 60 70 80 90 100 score 81.9 77.6 76.3 87.4 78.9 87.6 language = Danish 85.4 77.4 73.0 85.0 71.7 88.0 language = Ukrainian",
        ],
        "keyResults": [
            "[2501.12948 > Page 75] Table 27 | The CLUEWSC benchmark prompt is reproduced in full together with many example answers and long evaluation instructions.",
        ],
        "limitations": [
            "[2501.12948 > Page 60] from a fundamental limitation: in majority voting, samples are generated independently rather than building upon each other.",
        ],
        "whenItMatters": "[2501.12948 > Page 75] Table 27 | The CLUEWSC benchmark prompt is reproduced in full.",
        "whatIsNew": "of Lagrange multipliers. Let me set up the Lagrangian.",
        "confidenceNotes": [],
    }
    fallback = {
        "oneLine": "DeepSeek-R1 uses reinforcement learning to strengthen reasoning behavior in large language models.",
        "problem": "The paper asks how to induce strong general reasoning in LLMs without relying only on supervised reasoning traces.",
        "coreIdea": "It trains reasoning-oriented models with RL, emphasizing GRPO-style optimization and staged distillation into deployable checkpoints.",
        "methodSteps": [
            "Apply reinforcement learning to encourage longer reasoning traces and self-verification.",
            "Distill the stronger reasoning behavior into deployable models after the RL stage.",
        ],
        "keyResults": [
            "DeepSeek-R1 improves reasoning benchmark performance while remaining competitive on broader assistant-style evaluations.",
        ],
        "limitations": [
            "The paper still leaves efficiency, controllability, and broader robustness tradeoffs as open issues.",
        ],
        "whenItMatters": "It matters when a model must solve reasoning-heavy tasks without depending only on prompt tricks.",
        "whatIsNew": "The paper centers reinforcement-learning-first reasoning training rather than only prompt engineering or supervised CoT data.",
        "confidenceNotes": ["fallback"],
    }

    summary = structured_summary_module._normalize_summary(
        llm_value,
        fallback=fallback,
        title="DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning",
    )

    assert summary["problem"] == fallback["problem"]
    assert summary["coreIdea"] == fallback["coreIdea"]
    assert summary["keyResults"][0] == fallback["keyResults"][0]
    assert summary["whatIsNew"] == fallback["whatIsNew"]


def test_structured_summary_fallback_prefers_paper_memory_over_noisy_units():
    service = StructuredPaperSummaryService.__new__(StructuredPaperSummaryService)
    service.config = _StubConfig(papers_dir="")
    document = {
        "documentTitle": "MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training",
        "summary": {
            "documentThesis": "This work studies how to build performant multimodal large language models."
        },
    }
    bundles = {
        "problem": [
            {"contextualSummary": "[2403.09611 > Page 32] 32 B. McKinzie et al."},
        ],
        "method": [
            {"contextualSummary": "[2403.09611 > Page 9] TextCore 0-shot 4-shot 8-shot 20 40 60 80 49.6 39.3 43.8 45 51.7 35.9 58 61.1 52.2 33.4 58.7 62.2"},
        ],
        "results": [
            {"contextualSummary": "[2403.09611 > Page 41] Acknowledgements The authors would like to thank many collaborators and reviewers."},
            {"contextualSummary": "TextCaps across all model sizes and comparable to Flamingo-3B at small scales for most benchmarks."},
        ],
        "limitations": [
            {"contextualSummary": "[2403.09611 > Page 5] Decoder Only LLM \"This Walnut and Blue Cheese Stuffed Mushrooms recipe is sponsored by Fisher Nuts.\""},
        ],
    }
    paper_memory = {
        "paperCore": "MM1 studies how to build performant multimodal large language models and analyzes which architecture and data choices matter most.",
        "problemContext": "Multimodal LLM training recipes vary widely, so the paper isolates which components and data mixtures materially affect performance.",
        "methodCore": "The paper scales a multimodal pretraining recipe across dense and MoE MM1 models while varying connector design, image resolution, and data mixtures.",
        "evidenceCore": "Scaling the recipe yields state-of-the-art pretraining metrics and competitive supervised fine-tuning results on established multimodal benchmarks.",
        "limitations": "The paper focuses on recipe analysis and benchmark performance, leaving deployment efficiency and broader robustness tradeoffs less resolved.",
    }

    fallback = StructuredPaperSummaryService._fallback_summary(
        service,
        document,
        bundles,
        paper_memory,
        paper_id="2403.09611",
        parser_used="pymupdf",
    )

    assert fallback["oneLine"] == paper_memory["paperCore"]
    assert fallback["problem"] == paper_memory["problemContext"]
    assert fallback["coreIdea"] in {paper_memory["methodCore"], paper_memory["paperCore"]}
    assert fallback["keyResults"]
    assert "Acknowledgements" not in " ".join(fallback["keyResults"])
    assert fallback["limitations"][0] == paper_memory["limitations"]


def test_structured_summary_fallback_uses_parsed_markdown_slots_when_available(tmp_path):
    papers_dir = tmp_path / "papers"
    parsed_dir = papers_dir / "parsed" / "2501.12948"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.joinpath("document.md").write_text(
        "# DeepSeek-R1\n\n"
        "## Abstract\n\n"
        "Here we show that the reasoning abilities of LLMs can be incentivized through pure reinforcement learning, "
        "obviating the need for human-labeled reasoning trajectories.\n\n"
        "## Introduction\n\n"
        "We aim to explore whether LLMs can develop reasoning abilities through self-evolution in an RL framework.\n\n"
        "## DeepSeek-R1-Zero\n\n"
        "We build upon DeepSeek-V3-Base and employ Group Relative Policy Optimization (GRPO) as our RL framework. "
        "The reward signal is based on correctness of final predictions, and we bypass supervised fine-tuning before RL.\n\n"
        "## Results\n\n"
        "The trained model achieves superior performance on mathematics, coding competitions, and STEM benchmarks.\n\n"
        "## Limitations\n\n"
        "DeepSeek-R1-Zero still suffers from poor readability and language mixing, which motivates the later multi-stage pipeline.\n",
        encoding="utf-8",
    )

    service = StructuredPaperSummaryService.__new__(StructuredPaperSummaryService)
    service.config = _StubConfig(papers_dir=str(papers_dir))
    fallback = StructuredPaperSummaryService._fallback_summary(
        service,
        {"documentTitle": "DeepSeek-R1", "summary": {"documentThesis": "generic thesis"}},
        {"problem": [], "method": [], "results": [], "limitations": []},
        {
            "paperCore": "General reasoning represents a long-standing challenge.",
            "problemContext": "DeepSeek-R1 title + abstract fragment",
            "methodCore": "of Lagrange multipliers. Let me set up the Lagrangian.",
            "evidenceCore": "Table 3 | Experimental results at each stage of DeepSeek-R1.",
            "limitations": "",
        },
        paper_id="2501.12948",
        parser_used="pymupdf",
    )

    assert "pure reinforcement learning" in fallback["oneLine"].lower()
    assert "group relative policy optimization" in fallback["coreIdea"].lower() or "grpo" in fallback["coreIdea"].lower()
    assert fallback["keyResults"][0].startswith("The trained model achieves superior performance")
    assert "language mixing" in " ".join(fallback["limitations"]).lower()


def test_parsed_markdown_slots_prefer_specific_method_window_over_generic_abstract(tmp_path):
    papers_dir = tmp_path / "papers"
    parsed_dir = papers_dir / "parsed" / "2501.12948"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.joinpath("document.md").write_text(
        "# DeepSeek-R1\n\n"
        "## Abstract\n\n"
        "General reasoning represents a long-standing challenge in AI and recent large language models show progress.\n\n"
        "## Introduction\n\n"
        "We aim to reduce dependence on human-annotated reasoning traces.\n\n"
        "## DeepSeek-R1-Zero\n\n"
        "Specifically, we build upon DeepSeek-V3-Base and employ Group Relative Policy Optimization (GRPO) as our RL framework. "
        "The reward signal is based only on final-answer correctness, and we bypass supervised fine-tuning before RL training.\n\n"
        "## DeepSeek-R1\n\n"
        "We then introduce a multi-stage learning framework that integrates rejection sampling, reinforcement learning, and supervised fine-tuning.\n",
        encoding="utf-8",
    )

    slots = structured_summary_module._parsed_markdown_slots(_StubConfig(papers_dir=str(papers_dir)), paper_id="2501.12948")

    method_core = str(slots.get("method_core") or "").lower()

    assert "general reasoning represents" not in method_core
    assert "group relative policy optimization" in method_core or "multi-stage learning framework" in method_core


def test_parsed_markdown_slots_prefer_goal_style_problem_window_over_generic_background(tmp_path):
    papers_dir = tmp_path / "papers"
    parsed_dir = papers_dir / "parsed" / "2501.12948"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.joinpath("document.md").write_text(
        "# DeepSeek-R1\n\n"
        "## Abstract\n\n"
        "General reasoning represents a long-standing challenge in AI and recent large language models show progress.\n\n"
        "## Introduction\n\n"
        "To tackle these issues, we aim to explore the potential of LLMs for developing reasoning abilities through self-evolution in an RL framework, "
        "with minimal reliance on human labeling efforts.\n",
        encoding="utf-8",
    )

    slots = structured_summary_module._parsed_markdown_slots(_StubConfig(papers_dir=str(papers_dir)), paper_id="2501.12948")

    assert "aim to explore" in str(slots.get("problem_context") or "").lower()
    assert "general reasoning represents" not in str(slots.get("problem_context") or "").lower()


def test_prepare_outbound_summary_inputs_keeps_local_context_unchanged():
    prompt, context, changed = structured_summary_module._prepare_outbound_summary_inputs(
        provider="ollama",
        prompt="summarize",
        context="author@example.com should remain local-only",
    )

    assert prompt == "summarize"
    assert context == "author@example.com should remain local-only"
    assert changed is False


def test_normalize_summary_treats_korean_architecture_line_as_method_idea():
    summary = structured_summary_module._normalize_summary(
        {
            "oneLine": "디코더 기반 LLM에 이미지 인코더와 VL 커넥터를 결합한 구조를 두고 아키텍처 요소와 데이터 혼합 비율을 비교한다.",
            "coreIdea": "In this work, we discuss building performant Multimodal Large Language Models (MLLMs).",
            "whatIsNew": "In this work, we discuss building performant Multimodal Large Language Models (MLLMs).",
            "methodSteps": [],
            "keyResults": [],
            "limitations": [],
            "confidenceNotes": [],
        },
        fallback={
            "oneLine": "",
            "problem": "",
            "coreIdea": "",
            "methodSteps": [],
            "keyResults": [],
            "limitations": [],
            "whenItMatters": "",
            "whatIsNew": "",
            "confidenceNotes": [],
        },
        title="MM1",
    )

    assert "이미지 인코더" in summary["coreIdea"]


def test_best_summary_list_filters_intro_and_author_stubs_from_key_results():
    results = structured_summary_module._best_summary_list(
        field="keyResults",
        title="MM1",
        candidates=[
            "1 Introduction In recent years, the research community has achieved impressive progress in language modeling and image understanding.",
            "[2403.09611 > Page 6] 6 B. McKinzie et al.",
            "[2403.09611 > Page 13] TextCaps across all model sizes and comparable to Flamingo-3B at small scales for most benchmarks.",
        ],
        limit=4,
    )

    assert results == [
        "[2403.09611 > Page 13] TextCaps across all model sizes and comparable to Flamingo-3B at small scales for most benchmarks."
    ]


def test_normalize_summary_prefers_concise_method_idea_over_long_noisy_mix():
    summary = structured_summary_module._normalize_summary(
        {
            "oneLine": "DeepSeek-R1은 인간이 만든 추론 예시에 덜 의존하고 강화학습으로 추론 능력을 키우는 방법이다.",
            "coreIdea": "강화학습을 통해 추론 능력을 키운다. General reasoning represents a long-standing and formidable challenge in artificial intelligence. maximize cumulative rewards, PPO's approach penalizes the cumulative KL divergence, which may implicitly penalize the length of the response and thereby prevent the model's response length from increasing.",
            "whatIsNew": "General reasoning represents a long-standing and formidable challenge in artificial intelligence. maximize cumulative rewards, PPO's approach penalizes the cumulative KL divergence, which may implicitly penalize the length of the response.",
            "methodSteps": [],
            "keyResults": [],
            "limitations": [],
            "confidenceNotes": [],
        },
        fallback={
            "oneLine": "fallback one line",
            "problem": "",
            "coreIdea": "",
            "methodSteps": [],
            "keyResults": [],
            "limitations": [],
            "whenItMatters": "",
            "whatIsNew": "",
            "confidenceNotes": [],
        },
        title="DeepSeek-R1",
    )

    assert summary["coreIdea"] == "DeepSeek-R1은 인간이 만든 추론 예시에 덜 의존하고 강화학습으로 추론 능력을 키우는 방법이다."
    assert summary["whatIsNew"] == "DeepSeek-R1은 인간이 만든 추론 예시에 덜 의존하고 강화학습으로 추론 능력을 키우는 방법이다."


def test_best_summary_list_drops_bare_limitation_stub():
    limitations = structured_summary_module._best_summary_list(
        field="limitations",
        title="DeepSeek-R1",
        candidates=[
            "limitations.",
            "인간 priors의 한계가 남아 있으며, 일부 단계에서 인간 주석자가 추론 흔적을 자연스러운 대화 스타일로 변환한다.",
        ],
        limit=3,
    )

    assert limitations == [
        "인간 priors의 한계가 남아 있으며, 일부 단계에서 인간 주석자가 추론 흔적을 자연스러운 대화 스타일로 변환한다."
    ]


def test_best_summary_list_filters_author_contribution_stub_and_prefers_hangul():
    values = structured_summary_module._best_summary_list(
        field="methodSteps",
        candidates=[
            "Hongyu Hè: Co-implemented VL connector, assisted with experimentation. Max Schwarzer: Implemented support for pre-training on packed image-text pairs.",
            "이미지 인코더와 VL 커넥터를 결합한 디코더 기반 멀티모달 구조를 구성한다.",
        ],
        prefer_hangul=True,
        limit=4,
    )

    assert values == ["이미지 인코더와 VL 커넥터를 결합한 디코더 기반 멀티모달 구조를 구성한다."]


def test_normalize_summary_when_it_matters_falls_back_from_caption_stub():
    summary = structured_summary_module._normalize_summary(
        {
            "oneLine": "강화학습으로 언어모델의 추론 능력을 키우는 접근이다.",
            "problem": "",
            "coreIdea": "강화학습으로 언어모델의 추론 능력을 키우는 접근이다.",
            "methodSteps": [],
            "keyResults": [],
            "limitations": [],
            "whenItMatters": "Table 3 | Experimental results at each stage of DeepSeek-R1.",
            "whatIsNew": "",
            "confidenceNotes": [],
        },
        fallback={
            "oneLine": "강화학습으로 언어모델의 추론 능력을 키우는 접근이다.",
            "problem": "",
            "coreIdea": "강화학습으로 언어모델의 추론 능력을 키우는 접근이다.",
            "methodSteps": [],
            "keyResults": [],
            "limitations": [],
            "whenItMatters": "",
            "whatIsNew": "",
            "confidenceNotes": [],
        },
    )

    assert summary["whenItMatters"] == "강화학습으로 언어모델의 추론 능력을 키우는 접근이다."


def test_normalize_summary_filters_raw_english_spillover_from_core_and_method():
    summary = structured_summary_module._normalize_summary(
        {
            "oneLine": "GPT-4를 멀티모달 모델로 개발하고 시험형 벤치마크와 안전 평가를 함께 수행한다.",
            "problem": "",
            "coreIdea": "Deep reinforcement learning from human preferences. Advances in Neural Information Processing Systems, 30, 2017.",
            "methodSteps": [
                "Deep reinforcement learning from human preferences. Advances in Neural Information Processing Systems, 30, 2017.",
                "GPT-4를 텍스트와 이미지 입력을 받는 멀티모달 모델로 개발하고 내부 평가와 적대적 테스트를 함께 수행한다.",
            ],
            "keyResults": [],
            "limitations": [],
            "whenItMatters": "",
            "whatIsNew": "Deep reinforcement learning from human preferences. Advances in Neural Information Processing Systems, 30, 2017.",
            "confidenceNotes": [],
        },
        fallback={
            "oneLine": "GPT-4를 멀티모달 모델로 개발하고 시험형 벤치마크와 안전 평가를 함께 수행한다.",
            "problem": "",
            "coreIdea": "GPT-4를 멀티모달 모델로 개발하고 시험형 벤치마크와 안전 평가를 함께 수행한다.",
            "methodSteps": [
                "GPT-4를 텍스트와 이미지 입력을 받는 멀티모달 모델로 개발하고 내부 평가와 적대적 테스트를 함께 수행한다."
            ],
            "keyResults": [],
            "limitations": [],
            "whenItMatters": "",
            "whatIsNew": "GPT-4를 멀티모달 모델로 개발하고 시험형 벤치마크와 안전 평가를 함께 수행한다.",
            "confidenceNotes": [],
        },
        title="GPT-4 Technical Report",
    )

    assert summary["coreIdea"] == "GPT-4를 멀티모달 모델로 개발하고 시험형 벤치마크와 안전 평가를 함께 수행한다."
    assert summary["whatIsNew"] == "GPT-4를 멀티모달 모델로 개발하고 시험형 벤치마크와 안전 평가를 함께 수행한다."
    assert summary["methodSteps"][0] == "GPT-4를 텍스트와 이미지 입력을 받는 멀티모달 모델로 개발하고 내부 평가와 적대적 테스트를 함께 수행한다."


def test_structured_summary_llm_summary_redacts_p0_context_for_external_routes(monkeypatch):
    service = StructuredPaperSummaryService.__new__(StructuredPaperSummaryService)
    service.config = _StubConfig()
    seen: dict[str, object] = {}

    class _FakeExternalLLM:
        provider = "openai"

        def generate(self, prompt: str, context: str = "", max_tokens: int | None = None) -> str:  # noqa: ARG002
            seen["prompt"] = prompt
            seen["context"] = context
            seen["classification"] = classify_payload_level([prompt, context])
            return json.dumps({"oneLine": "clean summary"}, ensure_ascii=False)

    monkeypatch.setattr(
        structured_summary_module,
        "get_llm_for_task",
        lambda *args, **kwargs: (  # noqa: ARG005
            _FakeExternalLLM(),
            TaskRouteDecision(
                task_type="rag_answer",
                route="strong",
                provider="openai",
                model="gpt-5.4",
                timeout_sec=90,
                fallback_chain=["strong", "fallback-only"],
                reasons=["test"],
                allow_external_effective=True,
                complexity_score=0,
                policy_mode="external-allowed",
            ),
            [],
        ),
    )

    parsed, decision, warnings, fallback_used = StructuredPaperSummaryService._llm_summary(
        service,
        title="MM1",
        context=(
            "author@example.com\n"
            "96 98 100 102 104 103.1 99.9 99.5 99.5 97.4 97.3 99.6 100\n"
            "real method description remains."
        ),
        quick=False,
        allow_external=True,
        llm_mode="strong",
    )

    assert fallback_used is False
    assert parsed["oneLine"] == "clean summary"
    assert decision["provider"] == "openai"
    assert seen["classification"] != "P0"
    assert "author@example.com" not in str(seen["context"])
    assert "96 98 100 102" not in str(seen["context"])
    assert "외부 요약 호출 전에 P0 패턴을 redaction했습니다." in warnings
    assert "real method description remains." in str(seen["context"])


def test_prepare_outbound_summary_inputs_drops_instruction_prompt_noise_for_external_routes():
    prompt, context, changed = structured_summary_module._prepare_outbound_summary_inputs(
        provider="openai",
        prompt="summarize",
        context=(
            "TITLE: Self-Instruct\n"
            "DOCUMENT_THESIS: ACL 2023 SELF-INSTRUCT yizhongw@cs.washington.edu Abstract Large instruction-tuned language models depend on human-written instructions. "
            "Instruction: Generate a random password with at least 6 characters.\n"
            "METHOD_UNITS:\n"
            "- Task: Compose an email and send it to your friend.\n"
            "- The method bootstraps instructions, filters low-quality generations, and instruction-tunes the original model.\n"
            "- Experiments 187 118 68 31 25 64 59 80 84 66 1 31 30 54 49 44 74 83 112\n"
        ),
    )

    assert changed is True
    assert "author@example.com" not in context
    assert "password" not in context.casefold()
    assert "compose an email" not in context.casefold()
    assert "bootstraps instructions" in context
    assert classify_payload_level([prompt, context]) != "P0"


def test_paper_summary_cli_build_and_show_json(tmp_path, monkeypatch):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed(db)
    khub = _StubKhub(db, papers_dir=str(tmp_path / "papers"))
    runner = CliRunner()

    monkeypatch.setattr(
        structured_summary_module,
        "get_llm_for_task",
        lambda *args, **kwargs: (_FakeLLM(), type("Decision", (), {"to_dict": lambda self: {"provider": "test", "model": "fake", "route": "local", "timeoutSec": 45}})(), []),  # noqa: ARG005,E501
    )

    built = runner.invoke(
        paper_summary_group,
        ["build", "--paper-id", "2603.13017", "--json"],
        obj={"khub": khub},
    )
    assert built.exit_code == 0
    build_payload = json.loads(built.output)
    assert build_payload["schema"] == "knowledge-hub.paper-summary.build.result.v1"
    assert build_payload["status"] in {"ok", "partial"}
    assert build_payload["parserUsed"] == "raw"
    assert build_payload["parserAttempted"] == "pymupdf,mineru,opendataloader,raw"
    assert build_payload["llmRoute"] == "local"
    assert build_payload["contextStats"]["problemUnits"] >= 1
    assert build_payload["contextStats"]["paperMemoryAvailable"] is True
    assert build_payload["memoryRoute"]["decisionOrder"] == "memory_form_first"
    assert build_payload["readingCore"]["decisionOrder"] == "memory_form_first"
    assert build_payload["readingCore"]["housesUsed"]["readingHouse"] is True
    assert build_payload["readingCore"]["housesUsed"]["evidenceHouse"] is True
    assert build_payload["readingCore"]["housesUsed"]["contextHouse"] is False
    assert build_payload["readingCore"]["primaryHouseByField"]["problem"] == "reading_house"
    assert build_payload["readingCore"]["primaryHouseByField"]["keyResults"] == "evidence_house"
    assert build_payload["readingCore"]["supplementalContextUsed"] is False
    assert build_payload["readingCore"]["depthPlan"]["fieldDepths"]
    assert any(item["field"] == "keyResults" and item["depth"] == "deep" for item in build_payload["readingCore"]["depthPlan"]["fieldDepths"])
    assert any(item["name"] == "context_house" for item in build_payload["readingCore"]["disabledHouses"])
    assert build_payload["memoryRoute"]["fieldRoutes"][0]["field"] == "oneLine"
    assert build_payload["memoryRoute"]["fieldRoutes"][0]["primaryForm"] == "paper_memory"
    assert build_payload["memoryRoute"]["fieldRoutes"][0]["depth"] == "shallow"
    assert any(item["field"] == "problem" and item["primaryForm"] == "document_memory" for item in build_payload["memoryRoute"]["fieldRoutes"])
    assert any(item["field"] == "keyResults" and item["primaryForm"] == "claim_evidence" for item in build_payload["memoryRoute"]["fieldRoutes"])
    assert any(item["field"] == "limitations" and item["verifierForm"] == "chunk" for item in build_payload["memoryRoute"]["fieldRoutes"])
    assert any(item["name"] == "ontology_cluster" for item in build_payload["memoryRoute"]["disabledForms"])
    assert build_payload["evidenceSummaries"]["keyResults"]["summaryLines"]
    assert build_payload["summary"]["oneLine"]
    assert build_payload["summary"]["methodSteps"]
    assert build_payload["evidenceMap"]
    assert Path(build_payload["artifactPaths"]["evidenceSummariesPath"]).exists()
    assert Path(build_payload["artifactPaths"]["summaryJsonPath"]).exists()
    assert Path(build_payload["artifactPaths"]["summaryMdPath"]).exists()
    assert validate_payload(build_payload, build_payload["schema"], strict=True).ok

    shown = runner.invoke(
        paper_summary_group,
        ["show", "--paper-id", "2603.13017", "--json"],
        obj={"khub": khub},
    )
    assert shown.exit_code == 0
    show_payload = json.loads(shown.output)
    assert show_payload["schema"] == "knowledge-hub.paper-summary.card.result.v1"
    assert show_payload["paperId"] == "2603.13017"
    assert show_payload["summary"]["limitations"]
    assert show_payload["memoryRoute"]["decisionOrder"] == "memory_form_first"
    assert show_payload["readingCore"]["primaryHouseByField"]["methodSteps"] == "reading_house"
    assert validate_payload(show_payload, show_payload["schema"], strict=True).ok


def test_paper_summary_cli_build_defaults_to_configured_provider(monkeypatch, tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed(db)
    khub = _StubKhub(
        db,
        papers_dir=str(tmp_path / "papers"),
        config_data={"summarization": {"provider": "openai", "model": "gpt-5-mini"}},
    )
    runner = CliRunner()
    seen: dict[str, object] = {}

    def _fake_build(self, **kwargs):  # noqa: ANN001, ANN003
        seen.update(kwargs)
        return {
            "schema": "knowledge-hub.paper-summary.build.result.v1",
            "status": "ok",
            "paperId": "2603.13017",
            "paperTitle": "Personalized Agent Memory",
            "summary": {"oneLine": "ok"},
            "warnings": [],
            "artifactPaths": {},
        }

    monkeypatch.setattr(StructuredPaperSummaryService, "build", _fake_build)

    result = runner.invoke(
        paper_summary_group,
        ["build", "--paper-id", "2603.13017", "--json"],
        obj={"khub": khub},
    )

    assert result.exit_code == 0
    assert json.loads(result.output)["status"] == "ok"
    assert seen["allow_external"] is True
    assert seen["provider_override"] == "openai"
    assert seen["model_override"] == "gpt-5-mini"


def test_paper_summary_cli_blocks_when_opendataloader_missing(tmp_path, monkeypatch):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed(db)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
    db.conn.execute("UPDATE papers SET pdf_path = ? WHERE arxiv_id = ?", (str(pdf_path), "2603.13017"))
    db.conn.commit()

    def _raise_missing(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("opendataloader-pdf is not installed; install it to use --paper-parser opendataloader")

    monkeypatch.setattr(document_memory_builder_module.OpenDataLoaderPDFAdapter, "ensure_artifacts", _raise_missing)

    khub = _StubKhub(db, papers_dir=str(tmp_path / "papers"))
    runner = CliRunner()
    result = runner.invoke(
        paper_summary_group,
        ["build", "--paper-id", "2603.13017", "--paper-parser", "opendataloader", "--json"],
        obj={"khub": khub},
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "blocked"
    assert payload["parserUsed"] == "opendataloader"
    assert payload["parserAttempted"] == "opendataloader"
    assert payload["fallbackUsed"] is True
    assert "not installed" in payload["warnings"][0]
    assert payload["readingCore"]["housesUsed"]["readingHouse"] is False
    assert any(item["name"] == "reading_house" for item in payload["readingCore"]["disabledHouses"])


def test_paper_summary_cli_auto_prefers_mineru_when_available(tmp_path, monkeypatch):
    from knowledge_hub.papers import structured_summary as structured_summary_module

    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed(db)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
    db.conn.execute("UPDATE papers SET pdf_path = ? WHERE arxiv_id = ?", (str(pdf_path), "2603.13017"))
    db.conn.commit()

    class _FakeMinerUAdapter:
        def __init__(self, *, papers_dir: str):
            self.papers_dir = Path(papers_dir)

        def ensure_artifacts(self, *, paper_id: str, pdf_path: str, refresh: bool = False) -> MinerUParseResult:  # noqa: ARG002
            artifact_dir = self.papers_dir / "parsed" / str(paper_id)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            markdown_path = artifact_dir / "document.md"
            json_path = artifact_dir / "document.json"
            manifest_path = artifact_dir / "manifest.json"
            markdown_text = (
                "# Personalized Agent Memory\n\n"
                "## Abstract\n\nIt compresses sessions into cards.\n\n"
                "## Method\n\nIt retrieves cards and reranks them.\n\n"
                "## Results\n\nSuccess rate improves by 8.2 points.\n"
            )
            payload = {
                "markdown_text": markdown_text,
                "elements": [
                    {"type": "heading", "text": "Abstract", "page": 1, "bbox": [0, 0, 100, 20], "heading_path": ["Personalized Agent Memory", "Abstract"], "reading_order": 1},
                    {"type": "paragraph", "text": "It compresses sessions into cards.", "page": 1, "bbox": [0, 20, 180, 60], "heading_path": ["Personalized Agent Memory", "Abstract"], "reading_order": 2},
                ],
                "parser_meta": {"parser": "mineru", "mode": "local", "version": "test", "source_pdf": str(pdf_path)},
            }
            markdown_path.write_text(markdown_text, encoding="utf-8")
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest_path.write_text(json.dumps(payload["parser_meta"], ensure_ascii=False, indent=2), encoding="utf-8")
            return MinerUParseResult(
                markdown_text=markdown_text,
                elements=list(payload["elements"]),
                parser_meta=dict(payload["parser_meta"]),
                artifact_dir=str(artifact_dir),
                markdown_path=str(markdown_path),
                json_path=str(json_path),
                manifest_path=str(manifest_path),
            )

    def _pymupdf_missing(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("PyMuPDF is not installed; install it to use --paper-parser pymupdf")

    monkeypatch.setattr(document_memory_builder_module.PyMuPDFAdapter, "ensure_artifacts", _pymupdf_missing)
    monkeypatch.setattr(document_memory_builder_module, "MinerUPDFAdapter", _FakeMinerUAdapter)
    monkeypatch.setattr(
        structured_summary_module,
        "get_llm_for_task",
        lambda *args, **kwargs: (_FakeLLM(), type("Decision", (), {"to_dict": lambda self: {"provider": "test", "model": "fake", "route": "mini", "timeoutSec": 45}})(), []),  # noqa: ARG005,E501
    )

    khub = _StubKhub(db, papers_dir=str(tmp_path / "papers"))
    runner = CliRunner()
    result = runner.invoke(
        paper_summary_group,
        ["build", "--paper-id", "2603.13017", "--paper-parser", "auto", "--json"],
        obj={"khub": khub},
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["parserUsed"] == "mineru"
    assert payload["documentMemoryDiagnostics"]["parserUsed"] == "mineru"
    assert payload["documentMemoryDiagnostics"]["parseArtifactPath"]
    assert validate_payload(payload, payload["schema"], strict=True).ok


def test_paper_summary_cli_auto_prefers_pymupdf_when_available(tmp_path, monkeypatch):
    from knowledge_hub.papers import structured_summary as structured_summary_module

    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed(db)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
    db.conn.execute("UPDATE papers SET pdf_path = ? WHERE arxiv_id = ?", (str(pdf_path), "2603.13017"))
    db.conn.commit()

    class _FakePyMuPDFAdapter:
        def __init__(self, *, papers_dir: str):
            self.papers_dir = Path(papers_dir)

        def ensure_artifacts(self, *, paper_id: str, pdf_path: str, refresh: bool = False) -> PyMuPDFParseResult:  # noqa: ARG002
            artifact_dir = self.papers_dir / "parsed" / str(paper_id)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            markdown_path = artifact_dir / "document.md"
            json_path = artifact_dir / "document.json"
            manifest_path = artifact_dir / "manifest.json"
            markdown_text = (
                "# Personalized Agent Memory\n\n"
                "## Page 1\n\nIt compresses sessions into cards.\n\n"
                "## Page 2\n\nIt reranks retrieved cards.\n"
            )
            payload = {
                "markdown_text": markdown_text,
                "elements": [
                    {"type": "paragraph", "text": "It compresses sessions into cards.", "page": 1, "heading_path": ["Page 1"], "reading_order": 0},
                    {"type": "paragraph", "text": "It reranks retrieved cards.", "page": 2, "heading_path": ["Page 2"], "reading_order": 1},
                ],
                "parser_meta": {"parser": "pymupdf", "mode": "local", "version": "test", "source_pdf": str(pdf_path)},
            }
            markdown_path.write_text(markdown_text, encoding="utf-8")
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "paper_id": paper_id,
                        "parser_meta": payload["parser_meta"],
                        "markdown_path": str(markdown_path),
                        "json_path": str(json_path),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return PyMuPDFParseResult(
                markdown_text=markdown_text,
                elements=list(payload["elements"]),
                parser_meta=dict(payload["parser_meta"]),
                artifact_dir=str(artifact_dir),
                markdown_path=str(markdown_path),
                json_path=str(json_path),
                manifest_path=str(manifest_path),
            )

    monkeypatch.setattr(document_memory_builder_module, "PyMuPDFAdapter", _FakePyMuPDFAdapter)
    monkeypatch.setattr(
        structured_summary_module,
        "get_llm_for_task",
        lambda *args, **kwargs: (_FakeLLM(), type("Decision", (), {"to_dict": lambda self: {"provider": "test", "model": "fake", "route": "mini", "timeoutSec": 45}})(), []),  # noqa: ARG005,E501
    )

    khub = _StubKhub(db, papers_dir=str(tmp_path / "papers"))
    runner = CliRunner()
    result = runner.invoke(
        paper_summary_group,
        ["build", "--paper-id", "2603.13017", "--paper-parser", "auto", "--json"],
        obj={"khub": khub},
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["parserUsed"] == "pymupdf"
    assert payload["documentMemoryDiagnostics"]["parserUsed"] == "pymupdf"
    assert payload["documentMemoryDiagnostics"]["parseArtifactPath"]


def test_paper_summary_cli_filters_refusal_like_fallback_summary(tmp_path, monkeypatch):
    from knowledge_hub.papers import structured_summary as structured_summary_module

    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    db.upsert_paper(
        {
            "arxiv_id": "2603.13099",
            "title": "Fallback Refusal Filter",
            "authors": "J. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": "논문 원문(PDF)이 필요합니다. 현재 제공된 정보만으로는 요약할 수 없습니다.",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
        }
    )
    khub = _StubKhub(db, papers_dir=str(tmp_path / "papers"))
    runner = CliRunner()

    class _NoLLMDecision:
        def to_dict(self):
            return {"provider": "", "model": "", "route": "fallback-only", "timeoutSec": 45}

    monkeypatch.setattr(
        structured_summary_module,
        "get_llm_for_task",
        lambda *args, **kwargs: (None, _NoLLMDecision(), []),  # noqa: ARG005
    )

    result = runner.invoke(
        paper_summary_group,
        ["build", "--paper-id", "2603.13099", "--json"],
        obj={"khub": khub},
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "원문(PDF)이 필요합니다" not in payload["summary"]["oneLine"]
    assert payload["summary"]["oneLine"]


def test_paper_summary_cli_uses_opendataloader_provenance_when_available(tmp_path, monkeypatch):
    from knowledge_hub.papers import structured_summary as structured_summary_module

    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed(db)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
    db.conn.execute("UPDATE papers SET pdf_path = ? WHERE arxiv_id = ?", (str(pdf_path), "2603.13017"))
    db.conn.commit()

    class _FakeAdapter:
        def __init__(self, *, papers_dir: str):
            self.papers_dir = Path(papers_dir)

        def ensure_artifacts(self, *, paper_id: str, pdf_path: str, refresh: bool = False, parser_options: dict | None = None) -> OpenDataLoaderParseResult:  # noqa: ARG002
            artifact_dir = self.papers_dir / "parsed" / str(paper_id)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            markdown_path = artifact_dir / "document.md"
            json_path = artifact_dir / "document.json"
            manifest_path = artifact_dir / "manifest.json"
            markdown_text = (
                "# Personalized Agent Memory\n\n"
                "## Abstract\n\nIt compresses sessions into searchable memory cards.\n\n"
                "## Results\n\nSuccess rate improves by 8.2 points.\n"
            )
            payload = {
                "markdown_text": markdown_text,
                "elements": [
                    {"type": "heading", "text": "Abstract", "page": 1, "bbox": [0, 0, 100, 20], "heading_path": ["Personalized Agent Memory", "Abstract"]},
                    {"type": "paragraph", "text": "It compresses sessions into searchable memory cards.", "page": 1, "bbox": [0, 20, 180, 50], "heading_path": ["Personalized Agent Memory", "Abstract"]},
                    {"type": "table", "text": "Success rate improves by 8.2 points.", "page": 2, "bbox": [0, 0, 200, 40], "heading_path": ["Personalized Agent Memory", "Results"]},
                ],
                "parser_meta": {
                    "parser": "opendataloader",
                    "mode": "local",
                    "version": "test",
                    "source_pdf": str(pdf_path),
                    "convert_options": dict(parser_options or {}),
                },
            }
            markdown_path.write_text(markdown_text, encoding="utf-8")
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "paper_id": paper_id,
                        "parser_meta": payload["parser_meta"],
                        "convert_options": payload["parser_meta"]["convert_options"],
                        "source_pdf": str(pdf_path),
                        "markdown_path": str(markdown_path),
                        "json_path": str(json_path),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return OpenDataLoaderParseResult(
                markdown_text=markdown_text,
                elements=list(payload["elements"]),
                parser_meta=dict(payload["parser_meta"]),
                artifact_dir=str(artifact_dir),
                markdown_path=str(markdown_path),
                json_path=str(json_path),
                manifest_path=str(manifest_path),
            )

    monkeypatch.setattr(document_memory_builder_module, "OpenDataLoaderPDFAdapter", _FakeAdapter)
    monkeypatch.setattr(
        structured_summary_module,
        "get_llm_for_task",
        lambda *args, **kwargs: (_FakeLLM(), type("Decision", (), {"to_dict": lambda self: {"provider": "test", "model": "fake", "route": "local", "timeoutSec": 45}})(), []),  # noqa: ARG005,E501
    )

    khub = _StubKhub(db, papers_dir=str(tmp_path / "papers"))
    runner = CliRunner()
    result = runner.invoke(
        paper_summary_group,
        ["build", "--paper-id", "2603.13017", "--paper-parser", "opendataloader", "--json"],
        obj={"khub": khub},
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] in {"ok", "partial"}
    assert payload["parserUsed"] == "opendataloader"
    assert payload["parserAttempted"] == "opendataloader"
    assert payload["documentMemoryDiagnostics"]["structuredSectionsDetected"] >= 1
    assert payload["documentMemoryDiagnostics"]["elementsImported"] == 3
    assert any(item["field"] == "problem" for item in payload["memoryRoute"]["fieldRoutes"])
    assert any(item["name"] == "ontology_cluster" for item in payload["memoryRoute"]["disabledForms"])
    assert payload["readingCore"]["housesUsed"]["readingHouse"] is True
    assert payload["readingCore"]["housesUsed"]["evidenceHouse"] is True
    assert any(item["name"] == "context_house" for item in payload["readingCore"]["disabledHouses"])
    assert any(item.get("page") == 1 or item.get("page") == 2 for item in payload["evidenceMap"])


def test_paper_summary_cli_opendataloader_config_reaches_adapter(tmp_path, monkeypatch):
    from knowledge_hub.papers import structured_summary as structured_summary_module

    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed(db)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
    db.conn.execute("UPDATE papers SET pdf_path = ? WHERE arxiv_id = ?", (str(pdf_path), "2603.13017"))
    db.conn.commit()

    class _FakeAdapter:
        last_options: dict[str, object] | None = None

        def __init__(self, *, papers_dir: str):
            self.papers_dir = Path(papers_dir)

        def ensure_artifacts(self, *, paper_id: str, pdf_path: str, refresh: bool = False, parser_options: dict | None = None) -> OpenDataLoaderParseResult:  # noqa: ARG002
            type(self).last_options = dict(parser_options or {})
            artifact_dir = self.papers_dir / "parsed" / str(paper_id)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            markdown_path = artifact_dir / "document.md"
            json_path = artifact_dir / "document.json"
            manifest_path = artifact_dir / "manifest.json"
            markdown_text = "# Personalized Agent Memory\n\n## Abstract\n\nCards.\n"
            payload = {
                "markdown_text": markdown_text,
                "elements": [{"type": "paragraph", "text": "Cards.", "page": 1, "bbox": [0, 0, 100, 20], "heading_path": ["Abstract"]}],
                "parser_meta": {
                    "parser": "opendataloader",
                    "mode": "local",
                    "version": "test",
                    "source_pdf": str(pdf_path),
                    "convert_options": dict(parser_options or {}),
                },
            }
            markdown_path.write_text(markdown_text, encoding="utf-8")
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "paper_id": paper_id,
                        "parser_meta": payload["parser_meta"],
                        "convert_options": payload["parser_meta"]["convert_options"],
                        "source_pdf": str(pdf_path),
                        "markdown_path": str(markdown_path),
                        "json_path": str(json_path),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return OpenDataLoaderParseResult(
                markdown_text=markdown_text,
                elements=list(payload["elements"]),
                parser_meta=dict(payload["parser_meta"]),
                artifact_dir=str(artifact_dir),
                markdown_path=str(markdown_path),
                json_path=str(json_path),
                manifest_path=str(manifest_path),
            )

    monkeypatch.setattr(document_memory_builder_module, "OpenDataLoaderPDFAdapter", _FakeAdapter)
    monkeypatch.setattr(
        structured_summary_module,
        "get_llm_for_task",
        lambda *args, **kwargs: (_FakeLLM(), type("Decision", (), {"to_dict": lambda self: {"provider": "test", "model": "fake", "route": "local", "timeoutSec": 45}})(), []),  # noqa: ARG005,E501
    )

    khub = _StubKhub(
        db,
        papers_dir=str(tmp_path / "papers"),
        config_data={
            "paper": {
                "summary": {
                    "opendataloader": {
                        "reading_order": "xycut",
                        "use_struct_tree": True,
                        "table_method": "cluster",
                    }
                }
            }
        },
    )
    runner = CliRunner()
    result = runner.invoke(
        paper_summary_group,
        ["build", "--paper-id", "2603.13017", "--paper-parser", "opendataloader", "--json"],
        obj={"khub": khub},
    )
    assert result.exit_code == 0
    assert _FakeAdapter.last_options == {
        "reading_order": "xycut",
        "use_struct_tree": True,
        "table_method": "cluster",
    }


def test_paper_summary_cli_opendataloader_cli_override_beats_config(tmp_path, monkeypatch):
    from knowledge_hub.papers import structured_summary as structured_summary_module

    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed(db)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
    db.conn.execute("UPDATE papers SET pdf_path = ? WHERE arxiv_id = ?", (str(pdf_path), "2603.13017"))
    db.conn.commit()

    class _FakeAdapter:
        last_options: dict[str, object] | None = None

        def __init__(self, *, papers_dir: str):
            self.papers_dir = Path(papers_dir)

        def ensure_artifacts(self, *, paper_id: str, pdf_path: str, refresh: bool = False, parser_options: dict | None = None) -> OpenDataLoaderParseResult:  # noqa: ARG002
            type(self).last_options = dict(parser_options or {})
            artifact_dir = self.papers_dir / "parsed" / str(paper_id)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            markdown_path = artifact_dir / "document.md"
            json_path = artifact_dir / "document.json"
            manifest_path = artifact_dir / "manifest.json"
            markdown_text = "# Personalized Agent Memory\n\n## Abstract\n\nCards.\n"
            payload = {
                "markdown_text": markdown_text,
                "elements": [{"type": "paragraph", "text": "Cards.", "page": 1, "bbox": [0, 0, 100, 20], "heading_path": ["Abstract"]}],
                "parser_meta": {
                    "parser": "opendataloader",
                    "mode": "local",
                    "version": "test",
                    "source_pdf": str(pdf_path),
                    "convert_options": dict(parser_options or {}),
                },
            }
            markdown_path.write_text(markdown_text, encoding="utf-8")
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "paper_id": paper_id,
                        "parser_meta": payload["parser_meta"],
                        "convert_options": payload["parser_meta"]["convert_options"],
                        "source_pdf": str(pdf_path),
                        "markdown_path": str(markdown_path),
                        "json_path": str(json_path),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return OpenDataLoaderParseResult(
                markdown_text=markdown_text,
                elements=list(payload["elements"]),
                parser_meta=dict(payload["parser_meta"]),
                artifact_dir=str(artifact_dir),
                markdown_path=str(markdown_path),
                json_path=str(json_path),
                manifest_path=str(manifest_path),
            )

    monkeypatch.setattr(document_memory_builder_module, "OpenDataLoaderPDFAdapter", _FakeAdapter)
    monkeypatch.setattr(
        structured_summary_module,
        "get_llm_for_task",
        lambda *args, **kwargs: (_FakeLLM(), type("Decision", (), {"to_dict": lambda self: {"provider": "test", "model": "fake", "route": "local", "timeoutSec": 45}})(), []),  # noqa: ARG005,E501
    )

    khub = _StubKhub(
        db,
        papers_dir=str(tmp_path / "papers"),
        config_data={
            "paper": {
                "summary": {
                    "opendataloader": {
                        "reading_order": "xycut",
                        "use_struct_tree": False,
                        "table_method": "cluster",
                    }
                }
            }
        },
    )
    runner = CliRunner()
    result = runner.invoke(
        paper_summary_group,
        [
            "build",
            "--paper-id",
            "2603.13017",
            "--paper-parser",
            "opendataloader",
            "--odl-reading-order",
            "off",
            "--odl-use-struct-tree",
            "--json",
        ],
        obj={"khub": khub},
    )
    assert result.exit_code == 0
    assert _FakeAdapter.last_options == {
        "reading_order": "off",
        "use_struct_tree": True,
        "table_method": "cluster",
    }


def test_paper_summary_cli_auto_applies_opendataloader_options_when_selected(tmp_path, monkeypatch):
    from knowledge_hub.papers import structured_summary as structured_summary_module

    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed(db)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
    db.conn.execute("UPDATE papers SET pdf_path = ? WHERE arxiv_id = ?", (str(pdf_path), "2603.13017"))
    db.conn.commit()

    def _mineru_missing(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("mineru is not installed; install it to use --paper-parser mineru")

    def _pymupdf_missing(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("PyMuPDF is not installed; install it to use --paper-parser pymupdf")

    class _FakeAdapter:
        last_options: dict[str, object] | None = None

        def __init__(self, *, papers_dir: str):
            self.papers_dir = Path(papers_dir)

        def ensure_artifacts(self, *, paper_id: str, pdf_path: str, refresh: bool = False, parser_options: dict | None = None) -> OpenDataLoaderParseResult:  # noqa: ARG002
            type(self).last_options = dict(parser_options or {})
            artifact_dir = self.papers_dir / "parsed" / str(paper_id)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            markdown_path = artifact_dir / "document.md"
            json_path = artifact_dir / "document.json"
            manifest_path = artifact_dir / "manifest.json"
            markdown_text = "# Personalized Agent Memory\n\n## Abstract\n\nCards.\n"
            payload = {
                "markdown_text": markdown_text,
                "elements": [{"type": "paragraph", "text": "Cards.", "page": 1, "bbox": [0, 0, 100, 20], "heading_path": ["Abstract"]}],
                "parser_meta": {
                    "parser": "opendataloader",
                    "mode": "local",
                    "version": "test",
                    "source_pdf": str(pdf_path),
                    "convert_options": dict(parser_options or {}),
                },
            }
            markdown_path.write_text(markdown_text, encoding="utf-8")
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "paper_id": paper_id,
                        "parser_meta": payload["parser_meta"],
                        "convert_options": payload["parser_meta"]["convert_options"],
                        "source_pdf": str(pdf_path),
                        "markdown_path": str(markdown_path),
                        "json_path": str(json_path),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return OpenDataLoaderParseResult(
                markdown_text=markdown_text,
                elements=list(payload["elements"]),
                parser_meta=dict(payload["parser_meta"]),
                artifact_dir=str(artifact_dir),
                markdown_path=str(markdown_path),
                json_path=str(json_path),
                manifest_path=str(manifest_path),
            )

    monkeypatch.setattr(document_memory_builder_module.PyMuPDFAdapter, "ensure_artifacts", _pymupdf_missing)
    monkeypatch.setattr(document_memory_builder_module.MinerUPDFAdapter, "ensure_artifacts", _mineru_missing)
    monkeypatch.setattr(document_memory_builder_module, "OpenDataLoaderPDFAdapter", _FakeAdapter)
    monkeypatch.setattr(
        structured_summary_module,
        "get_llm_for_task",
        lambda *args, **kwargs: (_FakeLLM(), type("Decision", (), {"to_dict": lambda self: {"provider": "test", "model": "fake", "route": "local", "timeoutSec": 45}})(), []),  # noqa: ARG005,E501
    )

    khub = _StubKhub(
        db,
        papers_dir=str(tmp_path / "papers"),
        config_data={
            "paper": {
                "summary": {
                    "opendataloader": {
                        "reading_order": "xycut",
                        "use_struct_tree": False,
                        "table_method": "cluster",
                    }
                }
            }
        },
    )
    runner = CliRunner()
    result = runner.invoke(
        paper_summary_group,
        ["build", "--paper-id", "2603.13017", "--paper-parser", "auto", "--json"],
        obj={"khub": khub},
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["parserUsed"] == "opendataloader"
    assert _FakeAdapter.last_options == {
        "reading_order": "xycut",
        "table_method": "cluster",
    }
