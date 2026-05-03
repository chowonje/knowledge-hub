from __future__ import annotations

import pytest

from knowledge_hub.ai.ask_v2_card_selectors import AskV2CardSelectorRegistry, _BaseCardSelector
from knowledge_hub.ai.ask_v2_support import AskV2Route


class _SentinelSelector:
    def __init__(self):
        self.requests = []

    def select(self, request):  # noqa: ANN001
        self.requests.append(request)
        return [{"card_id": "sentinel"}]


class _VaultSearchShouldNotRun:
    def search_vault_cards_v2(self, *args, **kwargs):  # noqa: ANN002, ANN003
        _ = (args, kwargs)
        raise AssertionError("missing scoped vault paths must not fall back to broad vault card search")


class _VaultScopedMissService:
    sqlite_db = _VaultSearchShouldNotRun()

    def __init__(self):
        self.scoped_requests = []

    def _resolve_vault_cards(self, **kwargs):  # noqa: ANN003
        self.scoped_requests.append(dict(kwargs))
        return []

    def _fallback_vault_note_ids(self, **kwargs):  # noqa: ANN003
        _ = kwargs
        raise AssertionError("missing scoped vault paths must not fall back to note-id search")

    def _dedupe_and_score(self, *args, **kwargs):  # noqa: ANN002, ANN003
        _ = (args, kwargs)
        raise AssertionError("missing scoped vault paths must return before scoring")


def test_card_selector_registry_dispatches_request_to_source_selector():
    registry = AskV2CardSelectorRegistry(object(), fallback_error=RuntimeError)
    sentinel = _SentinelSelector()
    registry._selectors["paper"] = sentinel
    route = AskV2Route(source_kind="paper", intent="paper_lookup", mode="card-first", matched_entities=[], entity_ids=[])

    selected = registry.select(
        source_kind="paper",
        query="flashattention 논문 설명",
        route=route,
        limit=2,
        metadata_filter={"source_type": "paper"},
        query_plan={"family": "paper_lookup"},
        query_frame={"family": "paper_lookup"},
    )

    assert selected == [{"card_id": "sentinel"}]
    assert len(sentinel.requests) == 1
    assert sentinel.requests[0].query == "flashattention 논문 설명"
    assert sentinel.requests[0].route == route
    assert sentinel.requests[0].limit == 2


def test_vault_selector_stops_on_missing_explicit_file_scope():
    service = _VaultScopedMissService()
    registry = AskV2CardSelectorRegistry(service, fallback_error=RuntimeError)
    route = AskV2Route(source_kind="vault", intent="note_lookup", mode="card-first", matched_entities=[], entity_ids=[])

    selected = registry.select(
        source_kind="vault",
        query="Missing/NoSuchNote-999.md 파일의 핵심 내용을 보여줘",
        route=route,
        limit=3,
        metadata_filter={
            "source_type": "vault",
            "file_path": "Missing/NoSuchNote-999.md",
            "note_scope_required": True,
            "vault_scope_missing": True,
        },
        query_frame={
            "source_type": "vault",
            "family": "note_lookup",
            "answer_mode": "abstain",
            "resolved_source_ids": ["Missing/NoSuchNote-999.md"],
            "metadata_filter": {
                "source_type": "vault",
                "file_path": "Missing/NoSuchNote-999.md",
                "note_scope_required": True,
                "vault_scope_missing": True,
            },
        },
    )

    assert selected == []
    assert service.scoped_requests == [
        {
            "note_ids": None,
            "file_paths": ["Missing/NoSuchNote-999.md"],
        }
    ]


def test_vault_selector_stops_on_missing_explicit_note_id_scope():
    service = _VaultScopedMissService()
    registry = AskV2CardSelectorRegistry(service, fallback_error=RuntimeError)
    route = AskV2Route(source_kind="vault", intent="note_lookup", mode="card-first", matched_entities=[], entity_ids=[])

    selected = registry.select(
        source_kind="vault",
        query="vault:missing-note 파일의 핵심 내용을 보여줘",
        route=route,
        limit=3,
        metadata_filter={"source_type": "vault"},
        query_frame={
            "source_type": "vault",
            "family": "note_lookup",
            "answer_mode": "abstain",
            "resolved_source_ids": ["vault:missing-note"],
        },
    )

    assert selected == []
    assert service.scoped_requests == [
        {
            "note_ids": ["vault:missing-note"],
            "file_paths": None,
        }
    ]


def test_card_selector_registry_rejects_unknown_source_kind():
    registry = AskV2CardSelectorRegistry(object(), fallback_error=RuntimeError)
    route = AskV2Route(source_kind="web", intent="definition", mode="card-first", matched_entities=[], entity_ids=[])

    with pytest.raises(ValueError):
        registry.select(
            source_kind="project",
            query="repo overview",
            route=route,
            limit=1,
            metadata_filter=None,
        )


def test_base_card_selector_merge_fallback_cards_replaces_when_empty():
    merged = _BaseCardSelector._merge_fallback_cards(
        [],
        [{"card_id": "fallback"}],
        replace_when_empty=True,
    )

    assert merged == [{"card_id": "fallback"}]


def test_base_card_selector_merge_fallback_cards_prepends_when_requested():
    merged = _BaseCardSelector._merge_fallback_cards(
        [{"card_id": "existing"}],
        [{"card_id": "fallback"}],
        prepend=True,
    )

    assert merged == [{"card_id": "fallback"}, {"card_id": "existing"}]
