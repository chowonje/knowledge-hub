from __future__ import annotations

from knowledge_hub.ai.ask_v2_support import (
    AskV2Route,
    accumulate_search_results,
    build_paper_selection_inputs,
    build_ranked_forms,
    build_vault_selection_inputs,
    build_web_selection_inputs,
    first_nonempty_search,
    resolved_source_ids,
    should_attempt_claim_cards,
    vault_scope_from_query,
)


def test_resolved_source_ids_prefers_frame_payload_then_query_plan_keys():
    assert resolved_source_ids({"resolved_source_ids": ["paper-1", "paper-2"]}, {"resolvedSourceIds": ["ignored"]}) == [
        "paper-1",
        "paper-2",
    ]
    assert resolved_source_ids({}, {"resolvedSourceIds": ["web-1", "web-2"]}) == ["web-1", "web-2"]
    assert resolved_source_ids({}, {"resolvedPaperIds": ["paper-a"]}) == ["paper-a"]
    assert resolved_source_ids({}, {"resolved_paper_ids": ["paper-b"]}) == ["paper-b"]


def test_build_ranked_forms_dedupes_excludes_and_respects_limit():
    assert build_ranked_forms(
        ["Guide", "memory", "MEMORY", "reference", "policy", "extra"],
        limit=3,
        exclude_casefold={"guide", "reference"},
    ) == ["memory", "policy", "extra"]


def test_build_paper_selection_inputs_collects_family_ids_and_lookup_forms():
    payload = build_paper_selection_inputs(
        query="rag overview",
        frame_payload={"family": "paper_lookup", "expanded_terms": ["RAG", "overview"]},
        query_plan_payload={"resolvedPaperIds": ["2005.11401"]},
    )

    assert payload["family"] == "paper_lookup"
    assert payload["resolved_paper_ids"] == ["2005.11401"]
    assert payload["lookup_forms"] == ["rag overview", "RAG", "overview"]


def test_build_web_selection_inputs_builds_scope_and_search_forms():
    payload = build_web_selection_inputs(
        query="latest memory guide",
        frame_payload={"family": "temporal_update", "expanded_terms": ["memory", "guide"]},
        query_plan_payload={"resolvedSourceIds": ["https://example.com", "web_123"]},
        metadata_filter={"media_platform": "youtube", "document_id": "web_123"},
    )

    assert payload["media_platform"] == "youtube"
    assert payload["resolved_urls"] == ["https://example.com"]
    assert payload["resolved_doc_ids"] == ["web_123"]
    assert payload["search_forms"] == ["latest memory guide", "memory", "guide"]
    assert payload["document_scope"] == ["web_123"]


def test_build_vault_selection_inputs_builds_scope_and_filtered_forms():
    payload = build_vault_selection_inputs(
        query="요약",
        frame_payload={
            "family": "vault_explainer",
            "expanded_terms": ["Memory Design Note", "요약", "overview"],
        },
        query_plan_payload={"resolvedSourceIds": ["vault:memory-design", "vault/Memory Design Note.md"]},
        metadata_filter={"file_path": "vault/Memory Design Note.md"},
    )

    assert payload["resolved_note_ids"] == ["vault:memory-design"]
    assert payload["resolved_file_paths"] == ["vault/Memory Design Note.md"]
    assert payload["scoped_note_id"] == "vault:memory-design"
    assert payload["scoped_file_path"] == "vault/Memory Design Note.md"
    assert payload["search_forms"] == ["Memory Design Note"]


def test_vault_scope_from_query_extracts_explicit_note_id():
    assert vault_scope_from_query("vault:missing-note 파일의 핵심 내용을 보여줘") == "vault:missing-note"


def test_vault_scope_from_query_extracts_root_markdown_path_with_spaces():
    assert vault_scope_from_query("Please summarize Project Brief.md") == "Project Brief.md"


def test_first_nonempty_search_returns_first_nonempty_result_in_order():
    seen: list[str] = []

    def _search(form: str):
        seen.append(form)
        return [] if form == "first" else [{"id": form}]

    result = first_nonempty_search(forms=["first", "second", "third"], fallback_query="fallback", search_fn=_search)

    assert result == [{"id": "second"}]
    assert seen == ["first", "second"]


def test_accumulate_search_results_uses_fallback_query_when_forms_empty():
    seen: list[str] = []

    def _search(form: str):
        seen.append(form)
        return [{"id": form}]

    result = accumulate_search_results(forms=[], fallback_query="fallback", search_fn=_search)

    assert result == [{"id": "fallback"}]
    assert seen == ["fallback"]


def test_should_attempt_claim_cards_prefers_compare_eval_relation_and_non_section_paper_fallback():
    compare_route = AskV2Route(source_kind="paper", intent="comparison", mode="ontology-first", matched_entities=[], entity_ids=[])
    paper_lookup_route = AskV2Route(source_kind="paper", intent="paper_lookup", mode="card-first", matched_entities=[], entity_ids=[])
    web_relation_route = AskV2Route(source_kind="web", intent="relation", mode="ontology-first", matched_entities=[], entity_ids=[])
    vault_route = AskV2Route(source_kind="vault", intent="definition", mode="card-first", matched_entities=[], entity_ids=[])

    assert should_attempt_claim_cards(
        route=compare_route,
        section_first_requested=False,
        section_allowed=False,
        has_section_anchors=False,
    )
    assert should_attempt_claim_cards(
        route=web_relation_route,
        section_first_requested=False,
        section_allowed=False,
        has_section_anchors=False,
    )
    assert should_attempt_claim_cards(
        route=paper_lookup_route,
        section_first_requested=False,
        section_allowed=False,
        has_section_anchors=True,
    )
    assert not should_attempt_claim_cards(
        route=vault_route,
        section_first_requested=False,
        section_allowed=False,
        has_section_anchors=False,
    )
