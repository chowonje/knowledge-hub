"""khub ask must default to local-only generation unless config opts in explicitly (ADR 2026-06-11)."""

from types import SimpleNamespace

from knowledge_hub.interfaces.cli.commands.search_cmd import _ask_allow_external_default


class _FakeConfig:
    def __init__(self, data):
        self._data = data

    def get_nested(self, *keys, default=None):
        node = self._data
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node


def _ctx(config):
    return SimpleNamespace(config=config)


def test_cloud_summarization_provider_no_longer_implies_external():
    config = _FakeConfig({"summarization": {"provider": "openai"}})
    assert _ask_allow_external_default(_ctx(config), searcher=None) is False


def test_explicit_config_opt_in_enables_external_default():
    config = _FakeConfig({"answer": {"allow_external_default": True}})
    assert _ask_allow_external_default(_ctx(config), searcher=None) is True


def test_missing_config_defaults_to_local_only():
    assert _ask_allow_external_default(_ctx(None), searcher=None) is False
