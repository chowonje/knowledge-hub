from __future__ import annotations

from types import SimpleNamespace

from knowledge_hub.application.transformations import list_transformations, preview_transformation, run_transformation
from knowledge_hub.application import notebook_workbench as workbench_module
from knowledge_hub.application.notebook_workbench import notebook_workbench_chat, notebook_workbench_search


class _FakeSearcher:
    def __init__(self):
        self.llm = SimpleNamespace(generate=lambda prompt, context="": f"generated:{prompt[:24]}::{len(context)}")
        self.calls = []

    def search(self, query, **kwargs):  # noqa: ANN001
        self.calls.append({"query": query, **kwargs})
        metadata_filter = dict(kwargs.get("metadata_filter") or {})
        parent_id = str(metadata_filter.get("parent_id") or "")
        if parent_id == "note:1":
            return [
                SimpleNamespace(
                    metadata={
                        "title": "Vault Note",
                        "source_type": "note",
                        "resolved_parent_id": "note:1",
                        "parent_id": "note:1",
                        "file_path": "notes/vault.md",
                    },
                    score=0.95,
                    retrieval_mode="hybrid",
                    lexical_extras={"top_ranking_signals": ["exact_title_overlap"]},
                    document=f"{query} vault scoped evidence",
                    document_id="vault-doc-1",
                ),
                SimpleNamespace(
                    metadata={
                        "title": "Paper Note",
                        "source_type": "paper",
                        "resolved_parent_id": "paper:1",
                        "parent_id": "paper:1",
                        "arxiv_id": "2501.00001",
                    },
                    score=0.99,
                    retrieval_mode="hybrid",
                    lexical_extras={"top_ranking_signals": ["leak"]},
                    document=f"{query} leaked paper evidence",
                    document_id="paper-doc-1",
                ),
            ]
        if parent_id == "paper:1":
            return [
                SimpleNamespace(
                    metadata={
                        "title": "Paper Note",
                        "source_type": "paper",
                        "resolved_parent_id": "paper:1",
                        "parent_id": "paper:1",
                        "arxiv_id": "2501.00001",
                    },
                    score=0.88,
                    retrieval_mode="hybrid",
                    lexical_extras={"top_ranking_signals": ["paper_prior"]},
                    document=f"{query} paper scoped evidence",
                    document_id="paper-doc-1",
                )
            ]
        return [
            SimpleNamespace(
                metadata={
                    "title": "Vault Note",
                    "source_type": "note",
                    "resolved_parent_id": "note:1",
                    "parent_id": "note:1",
                    "file_path": "notes/vault.md",
                },
                score=0.91,
                retrieval_mode="hybrid",
                lexical_extras={"top_ranking_signals": ["seed_vault"]},
                document=f"{query} vault evidence",
                document_id="vault-doc-1",
            ),
            SimpleNamespace(
                metadata={
                    "title": "Paper Note",
                    "source_type": "paper",
                    "resolved_parent_id": "paper:1",
                    "parent_id": "paper:1",
                    "arxiv_id": "2501.00001",
                },
                score=0.82,
                retrieval_mode="hybrid",
                lexical_extras={"top_ranking_signals": ["seed_paper"]},
                document=f"{query} paper evidence",
                document_id="paper-doc-1",
            ),
        ]


class _FakeConfig:
    summarization_provider = "ollama"


class _ClaimSQLite:
    def get_note(self, note_id):  # noqa: ANN001
        if note_id == "note:1":
            return {"id": "note:1"}
        return None

    def list_notes(self, limit=50, offset=0, source_type=None, para_category=None):  # noqa: ANN001
        _ = (limit, offset, source_type, para_category)
        return []

    def list_claims_by_note(self, note_id, limit=100):  # noqa: ANN001
        _ = limit
        if note_id == "note:1":
            return [{"claim_id": "claim:vault", "confidence": 0.9}]
        return []

    def list_claims_by_record(self, record_id, limit=100):  # noqa: ANN001
        _ = limit
        if record_id == "2501.00001":
            return [{"claim_id": "claim:paper", "confidence": 0.88}]
        return []

    def list_claim_normalizations(self, claim_ids=None, status=None, limit=100, **kwargs):  # noqa: ANN001
        _ = (status, limit, kwargs)
        rows = []
        for claim_id in claim_ids or []:
            if claim_id == "claim:vault":
                rows.append(
                    {
                        "claim_id": claim_id,
                        "dataset": "HotpotQA",
                        "metric": "F1",
                        "comparator": "baseline",
                        "result_direction": "better",
                        "result_value_numeric": 82.1,
                    }
                )
            if claim_id == "claim:paper":
                rows.append(
                    {
                        "claim_id": claim_id,
                        "dataset": "MMLU",
                        "metric": "Accuracy",
                        "comparator": "prior-work",
                        "result_direction": "better",
                        "result_value_numeric": 71.5,
                    }
                )
        return rows


def test_list_transformations_exposes_seed_definitions():
    payload = list_transformations()

    assert payload["status"] == "ok"
    ids = {item["id"] for item in payload["items"]}
    assert {"technical_summary", "paper_brief", "module_brief", "compare_sources"} <= ids


def test_preview_and_run_transformation_dry_run_uses_context_pack():
    searcher = _FakeSearcher()
    preview = preview_transformation(
        searcher,
        sqlite_db=None,
        transformation_id="technical_summary",
        query="RAG retrieval",
        include_workspace=False,
        max_sources=3,
    )
    result = run_transformation(
        searcher,
        sqlite_db=None,
        llm=searcher.llm,
        config=_FakeConfig(),
        transformation_id="technical_summary",
        query="RAG retrieval",
        dry_run=True,
    )

    assert preview["status"] == "ok"
    assert preview["selected_sources"]
    assert result["status"] == "ok"
    assert result["dry_run"] is True
    assert result["output"] == ""


def test_workbench_search_and_chat_stay_bounded_to_selected_scope():
    searcher = _FakeSearcher()
    search_payload = notebook_workbench_search(
        searcher,
        sqlite_db=None,
        topic="RAG",
        query="retrieval",
        include_vault=True,
        include_papers=True,
        include_web=False,
        top_k=2,
    )
    selected_id = search_payload["results"][0]["local_source_id"]
    searcher.calls.clear()
    chat_payload = notebook_workbench_chat(
        searcher,
        sqlite_db=None,
        config=_FakeConfig(),
        topic="RAG",
        message="Summarize retrieval",
        selected_source_ids=[selected_id],
        include_vault=True,
        include_papers=True,
        include_web=False,
        top_k=2,
    )

    assert search_payload["status"] == "ok"
    assert search_payload["selected_scope_count"] >= 1
    assert search_payload["retrieval_strategy"] == "scoped_search"
    assert search_payload["filters_applied"] >= 1
    assert search_payload["fallback_source_count"] == 0
    assert search_payload["scope_diagnostics"][0]["scoped"] is True
    assert search_payload["scope_diagnostics"][0]["filter_key"] == "parent_id"
    assert search_payload["results"][0]["local_source_id"] == selected_id
    assert search_payload["results"][0]["top_ranking_signals"] == ["exact_title_overlap"]
    assert chat_payload["status"] == "ok"
    assert chat_payload["selected_scope_count"] == 1
    assert chat_payload["retrieval_strategy"] == "scoped_search"
    assert chat_payload["scope_diagnostics"][0]["matched_results"] == 1
    assert chat_payload["sources"][0]["local_source_id"] == selected_id
    assert chat_payload["answer"].startswith("generated:")
    scoped_filters = [call.get("metadata_filter") for call in searcher.calls if call.get("metadata_filter")]
    assert {"parent_id": "note:1"} in scoped_filters
    assert {"parent_id": "paper:1"} not in scoped_filters


def test_workbench_payloads_include_claim_signals():
    searcher = _FakeSearcher()
    sqlite_db = _ClaimSQLite()
    search_payload = notebook_workbench_search(
        searcher,
        sqlite_db=sqlite_db,
        topic="RAG",
        query="retrieval",
        include_vault=True,
        include_papers=True,
        include_web=False,
        top_k=2,
    )
    chat_payload = notebook_workbench_chat(
        searcher,
        sqlite_db=sqlite_db,
        config=_FakeConfig(),
        topic="RAG",
        message="Summarize retrieval",
        include_vault=True,
        include_papers=True,
        include_web=False,
        top_k=2,
    )

    assert search_payload["claimSignals"]["resultsWithClaims"] == 2
    assert search_payload["claim_signals"]["totalNormalizedClaimCount"] == 2
    assert search_payload["results"][0]["claimSignals"]["claimCount"] == 1
    assert search_payload["results"][1]["claim_signals"]["topDatasets"][0]["name"] == "MMLU"
    assert chat_payload["claimSignals"]["resultsWithClaims"] == 2
    assert chat_payload["sources"][0]["claim_signals"]["topMetrics"][0]["name"] == "F1"


def test_workbench_search_and_chat_apply_source_context_modes():
    searcher = _FakeSearcher()
    context_modes = {"note:1": "summary", "paper:1": "excluded"}

    search_payload = notebook_workbench_search(
        searcher,
        sqlite_db=None,
        topic="RAG",
        query="retrieval",
        selected_source_ids=["note:1", "paper:1"],
        selected_source_context_modes=context_modes,
        include_vault=True,
        include_papers=True,
        include_web=False,
        top_k=2,
    )
    searcher.calls.clear()
    chat_payload = notebook_workbench_chat(
        searcher,
        sqlite_db=None,
        config=_FakeConfig(),
        topic="RAG",
        message="Summarize retrieval",
        selected_source_ids=["note:1", "paper:1"],
        selected_source_context_modes=context_modes,
        include_vault=True,
        include_papers=True,
        include_web=False,
        top_k=2,
    )

    assert search_payload["status"] == "ok"
    assert search_payload["selected_scope_count"] == 1
    assert search_payload["excluded_scope_count"] == 1
    assert search_payload["results"][0]["local_source_id"] == "note:1"
    assert search_payload["results"][0]["context_mode"] == "summary"
    assert search_payload["scope_diagnostics"][0]["context_mode"] == "summary"
    assert search_payload["excluded_scope_diagnostics"][0]["context_mode"] == "excluded"
    assert any("excluded by context mode" in warning for warning in search_payload["warnings"])
    assert chat_payload["status"] == "ok"
    assert chat_payload["selected_scope_count"] == 1
    assert chat_payload["excluded_scope_count"] == 1
    assert chat_payload["sources"][0]["context_mode"] == "summary"
    assert chat_payload["scope_diagnostics"][0]["context_mode"] == "summary"
    assert chat_payload["excluded_scope_diagnostics"][0]["context_mode"] == "excluded"
    assert any("excluded by context mode" in warning for warning in chat_payload["warnings"])
    scoped_filters = [call.get("metadata_filter") for call in searcher.calls if call.get("metadata_filter")]
    assert {"parent_id": "note:1"} in scoped_filters
    assert {"parent_id": "paper:1"} not in scoped_filters


def test_workbench_search_drops_out_of_scope_results_even_if_searcher_leaks():
    searcher = _FakeSearcher()

    payload = notebook_workbench_search(
        searcher,
        sqlite_db=None,
        topic="RAG",
        query="retrieval",
        selected_source_ids=["note:1"],
        include_vault=True,
        include_papers=True,
        include_web=False,
        top_k=2,
    )

    assert payload["status"] == "ok"
    assert payload["selected_scope_count"] == 1
    assert [item["local_source_id"] for item in payload["results"]] == ["note:1"]
    assert payload["scope_diagnostics"][0]["matched_results"] == 1


def test_workbench_search_reports_missing_filter_fallback(monkeypatch):
    searcher = _FakeSearcher()

    monkeypatch.setattr(
        workbench_module,
        "build_context_pack",
        lambda *args, **kwargs: {
            "persistent_sources": [
                {
                    "local_source_id": "summary:rag",
                    "title": "Topic Summary",
                    "normalized_source_type": "summary",
                    "content": "RAG topic summary",
                    "metadata": {},
                }
            ],
            "warnings": [],
        },
    )

    payload = notebook_workbench_search(
        searcher,
        sqlite_db=None,
        topic="RAG",
        query="retrieval",
        selected_source_ids=["summary:rag"],
        include_vault=True,
        include_papers=True,
        include_web=False,
        top_k=2,
    )

    assert payload["status"] == "ok"
    assert payload["retrieval_strategy"] == "bounded_local_fallback"
    assert payload["fallback_source_count"] == 1
    assert payload["scope_diagnostics"][0]["fallback_reason"] == "missing_stable_filter"


def test_workbench_search_prefers_stable_scope_id_and_reports_zero_match(monkeypatch):
    class _LeakSearcher(_FakeSearcher):
        def search(self, query, **kwargs):  # noqa: ANN001
            self.calls.append({"query": query, **kwargs})
            metadata_filter = dict(kwargs.get("metadata_filter") or {})
            assert metadata_filter == {"stable_scope_id": "vault:notes/rag.md::section:RAG Foundations"}
            return [
                SimpleNamespace(
                    metadata={
                        "title": "Vault Note",
                        "source_type": "vault",
                        "resolved_parent_id": "vault:notes/other.md::section:Other",
                        "parent_id": "vault:notes/other.md::section:Other",
                        "stable_scope_id": "vault:notes/other.md::section:Other",
                        "file_path": "notes/other.md",
                    },
                    score=0.92,
                    retrieval_mode="hybrid",
                    lexical_extras={"top_ranking_signals": ["leak"]},
                    document=f"{query} leaked evidence",
                    document_id="vault-doc-2",
                )
            ]

    monkeypatch.setattr(
        workbench_module,
        "build_context_pack",
        lambda *args, **kwargs: {
            "persistent_sources": [
                {
                    "local_source_id": "summary:rag",
                    "title": "Topic Summary",
                    "normalized_source_type": "vault",
                    "content": "RAG topic summary",
                    "metadata": {
                        "scope_level": "section",
                        "stable_scope_id": "vault:notes/rag.md::section:RAG Foundations",
                        "document_scope_id": "vault:notes/rag.md",
                        "section_scope_id": "vault:notes/rag.md::section:RAG Foundations",
                        "file_path": "notes/rag.md",
                    },
                }
            ],
            "warnings": [],
        },
    )

    payload = notebook_workbench_search(
        _LeakSearcher(),
        sqlite_db=None,
        topic="RAG",
        query="retrieval",
        selected_source_ids=["summary:rag"],
        include_vault=True,
        include_papers=False,
        include_web=False,
        top_k=2,
    )

    assert payload["retrieval_strategy"] == "bounded_local_fallback"
    assert payload["fallback_source_count"] == 0
    assert payload["scope_diagnostics"][0]["filter_key"] == "stable_scope_id"
    assert payload["scope_diagnostics"][0]["fallback_reason"] == "zero_match"
    assert payload["scope_diagnostics"][0]["stable_scope_id"] == "vault:notes/rag.md::section:RAG Foundations"
    assert payload["scope_diagnostics"][0]["matched_results"] == 0
