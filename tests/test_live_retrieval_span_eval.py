from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path("eval/knowledgeos/scripts/check_live_retrieval_span_eval.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("check_live_retrieval_span_eval", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _result(*, document_id: str, source_type: str, text: str, title: str = ""):
    return SimpleNamespace(
        document_id=document_id,
        document=text,
        distance=0.2,
        score=0.8,
        semantic_score=0.7,
        lexical_score=0.9,
        metadata={
            "source_id": document_id,
            "source_type": source_type,
            "title": title,
        },
    )


class _FakeSearcher:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return list(self.mapping.get(query, []))


def test_live_retrieval_span_eval_passes_expected_source_and_terms():
    module = _load_module()
    searcher = _FakeSearcher(
        {
            "transformer attention": [
                _result(
                    document_id="paper:1706.03762#0",
                    source_type="paper",
                    title="Attention Is All You Need",
                    text="Transformer self attention sequence modeling evidence.",
                )
            ]
        }
    )
    payload = module.run_live_retrieval_span_eval(
        searcher=searcher,
        cases=[
            {
                "case_id": "paper_attention",
                "query": "transformer attention",
                "source_type": "paper",
                "expected_source_ids": ["paper:1706.03762#0"],
                "expected_text_terms": ["transformer", "attention"],
                "min_rank": 1,
            }
        ],
        cases_path=Path("cases.json"),
        top_k=5,
        retrieval_mode="keyword",
        alpha=0.7,
        use_ontology_expansion=False,
        min_cases=1,
        min_source_hit_rate=1.0,
        min_term_overlap_ratio=1.0,
        fail_on_insufficient=True,
    )

    assert payload["status"] == "ok"
    assert payload["sourceHitAtKRate"] == 1.0
    assert payload["termOverlapPassRate"] == 1.0
    assert searcher.calls[0]["source_type"] == "paper"


def test_live_retrieval_span_eval_fails_wrong_source():
    module = _load_module()
    searcher = _FakeSearcher(
        {
            "transformer attention": [
                _result(document_id="paper:wrong#0", source_type="paper", text="Transformer attention."),
            ]
        }
    )
    payload = module.run_live_retrieval_span_eval(
        searcher=searcher,
        cases=[
            {
                "case_id": "paper_attention",
                "query": "transformer attention",
                "source_type": "paper",
                "expected_source_ids": ["paper:1706.03762#0"],
                "expected_text_terms": ["transformer", "attention"],
                "min_rank": 1,
            }
        ],
        cases_path=Path("cases.json"),
        top_k=5,
        retrieval_mode="keyword",
        alpha=0.7,
        use_ontology_expansion=False,
        min_cases=1,
        min_source_hit_rate=1.0,
        min_term_overlap_ratio=1.0,
        fail_on_insufficient=True,
    )

    assert payload["status"] == "failed"
    assert payload["cases"][0]["errors"] == ["expected_source_not_found"]


def test_live_retrieval_span_eval_keeps_non_evidence_signal_out_of_citation_cases():
    module = _load_module()
    result = _result(
        document_id="learning_edge:rag:prereq#0",
        source_type="learning_edge",
        text="A prerequisite learning edge links retrieval to generation.",
    )
    signal_case = {
        "case_id": "learning_edge_signal",
        "query": "rag prerequisite",
        "source_type": "learning_edge",
        "expected_source_ids": ["learning_edge:rag:prereq#0"],
        "expected_text_terms": ["retrieval", "generation"],
        "expected_evidence_role": "retrieval_signal_only",
        "must_abstain": True,
    }
    citation_case = dict(signal_case, expected_evidence_role="citation", must_abstain=False)

    signal_result = module.evaluate_case(signal_case, [result], default_top_k=5)
    citation_result = module.evaluate_case(citation_case, [result], default_top_k=5)

    assert signal_result["status"] == "pass"
    assert signal_result["matchedSourceNonEvidence"] is True
    assert citation_result["status"] == "fail"
    assert "matched_source_is_non_evidence" in citation_result["errors"]


def test_live_retrieval_span_eval_skips_missing_local_cases_without_fail_flag():
    module = _load_module()
    payload = module.run_live_retrieval_span_eval(
        searcher=_FakeSearcher({}),
        cases=[],
        cases_path=Path("missing.json"),
        top_k=5,
        retrieval_mode="keyword",
        alpha=0.7,
        use_ontology_expansion=False,
        min_cases=1,
        min_source_hit_rate=1.0,
        min_term_overlap_ratio=1.0,
        fail_on_insufficient=False,
    )

    assert payload["status"] == "skipped"
    assert payload["errors"] == ["insufficient_evaluable_cases:0/1"]


def test_reranker_ab_eval_reports_promotion_candidate_when_rank_improves():
    module = _load_module()
    wrong = _result(document_id="paper:wrong#0", source_type="paper", text="Transformer attention distractor.")
    expected = _result(
        document_id="paper:1706.03762#0",
        source_type="paper",
        title="Attention Is All You Need",
        text="Transformer self attention sequence modeling evidence.",
    )
    baseline = _FakeSearcher({"transformer attention": [wrong, expected]})
    reranker = _FakeSearcher({"transformer attention": [expected, wrong]})

    payload = module.run_reranker_ab_eval(
        baseline_searcher=baseline,
        reranker_searcher=reranker,
        cases=[
            {
                "case_id": "paper_attention",
                "query": "transformer attention",
                "source_type": "paper",
                "expected_source_ids": ["paper:1706.03762#0"],
                "expected_text_terms": ["transformer", "attention"],
                "min_rank": 1,
            }
        ],
        cases_path=Path("cases.json"),
        top_k=5,
        retrieval_mode="keyword",
        alpha=0.7,
        use_ontology_expansion=False,
        min_cases=1,
        min_source_hit_rate=0.0,
        min_term_overlap_ratio=1.0,
        fail_on_insufficient=True,
    )

    assert payload["schema"] == module.AB_SCHEMA
    assert payload["status"] == "ok"
    assert payload["recommendation"] == "promote_candidate"
    assert payload["deltas"]["failedCaseCount"] == -1
    assert payload["deltas"]["sourceHitWithinMinRankRate"] == 1.0
