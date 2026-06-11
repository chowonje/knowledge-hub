"""continuous-sync must stay staged-only by default and never auto-apply unapproved ko-notes."""

from types import SimpleNamespace

from knowledge_hub.interfaces.cli.commands import crawl_support
from knowledge_hub.interfaces.cli.commands.crawl_cmd import crawl_group


class _FakeService:
    def run_pipeline(self, **kwargs):
        return {"jobId": "job-1"}


class _FakeMaterializer:
    instances: list["_FakeMaterializer"] = []

    def __init__(self, config):
        self.apply_calls: list[dict] = []
        _FakeMaterializer.instances.append(self)

    def generate_for_job(self, **kwargs):
        return {"runId": "run-1"}

    def apply(self, **kwargs):
        self.apply_calls.append(dict(kwargs))
        return {"status": "ok"}


def _run_sync(monkeypatch, tmp_path, *, apply_notes: bool) -> dict:
    url_file = tmp_path / "urls.txt"
    url_file.write_text("https://example.com/a\n", encoding="utf-8")
    monkeypatch.setattr(crawl_support, "validate_cli_payload", lambda *args, **kwargs: None)
    _FakeMaterializer.instances = []
    khub = SimpleNamespace(config=None, web_ingest_service=lambda: _FakeService())
    return crawl_support.sync_watchlist_payload(
        khub=khub,
        build_payload={"txtPath": str(url_file)},
        topic="continuous-latest",
        source="web",
        profile="safe",
        source_policy="fixed",
        engine="auto",
        timeout=15,
        delay=0.0,
        index=False,
        extract_concepts=False,
        materialize=True,
        apply_notes=apply_notes,
        max_source_notes=1,
        max_concept_notes=1,
        allow_external=False,
        llm_mode="auto",
        materializer_factory=_FakeMaterializer,
    )


def test_continuous_sync_cli_apply_default_is_staged_only():
    command = crawl_group.commands["continuous-sync"]
    param = next(item for item in command.params if item.name == "apply_notes")
    assert param.default is False, "continuous-sync must not write to the vault by default"


def test_sync_watchlist_apply_only_applies_approved_items(monkeypatch, tmp_path):
    _run_sync(monkeypatch, tmp_path, apply_notes=True)
    materializer = _FakeMaterializer.instances[0]
    assert materializer.apply_calls, "explicit --apply should still apply"
    assert materializer.apply_calls[0].get("only_approved") is True, (
        "sync apply must never push unapproved ko-notes into the vault"
    )


def test_sync_watchlist_default_performs_no_vault_apply(monkeypatch, tmp_path):
    payload = _run_sync(monkeypatch, tmp_path, apply_notes=False)
    materializer = _FakeMaterializer.instances[0]
    assert materializer.apply_calls == []
    assert payload["apply"] == {}
