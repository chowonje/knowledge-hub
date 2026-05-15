from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.interfaces.cli.commands import doctor_cmd as doctor_module

setup_module = None


class _FakeKhub:
    def __init__(self, config):
        self.config = config

    def searcher(self):
        raise AssertionError("doctor must not initialize the searcher")


def _ok_install_environment_check() -> dict[str, object]:
    return {
        "area": "install environment",
        "status": "ok",
        "summary": "khub 실행 경로와 imported package 경로가 일치합니다.",
        "detail": "khub=/tmp/bin/khub; package=/repo/knowledge_hub/__init__.py; cli=0.1.5; editable=/repo",
        "fixCommand": "",
        "recommendedActions": [],
        "diagnostics": {
            "khubExecutable": "/tmp/bin/khub",
            "pythonExecutable": "/tmp/bin/python",
            "importedPackagePath": "/repo/knowledge_hub/__init__.py",
            "issues": [],
        },
    }


@pytest.fixture(autouse=True)
def _stable_install_environment(monkeypatch):
    monkeypatch.setattr(doctor_module, "_install_environment_check", _ok_install_environment_check)


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        config_path=str(tmp_path / "config.yaml"),
        translation_provider="openai",
        translation_model="gpt-5-nano",
        summarization_provider="openai",
        summarization_model="gpt-5-nano",
        embedding_provider="ollama",
        embedding_model="nomic-embed-text",
        paper_summary_parser="auto",
        papers_dir=str(tmp_path / "papers"),
        sqlite_path=str(tmp_path / "knowledge.db"),
        get_provider_config=lambda provider: {"base_url": "http://localhost:11434"} if provider == "ollama" else {},
        get_nested=lambda *args, default=None: default,
    )


def test_build_doctor_payload_translates_runtime_state(monkeypatch, tmp_path):
    config = _config(tmp_path)
    monkeypatch.setattr(
        doctor_module,
        "build_runtime_diagnostics",
        lambda config, searcher=None, searcher_error="": {
            "providers": [
                {"role": "translation", "provider": "openai", "model": "gpt-5-nano", "installed": True, "requires_api_key": True, "api_key_status": "missing", "degraded": True, "reasons": ["missing_api_key"], "available": False},
                {"role": "summarization", "provider": "openai", "model": "gpt-5-nano", "installed": True, "requires_api_key": True, "api_key_status": "missing", "degraded": True, "reasons": ["missing_api_key"], "available": False},
                {"role": "embedding", "provider": "ollama", "model": "nomic-embed-text", "installed": True, "requires_api_key": False, "api_key_status": "not_required", "degraded": False, "reasons": [], "available": True},
            ],
            "vectorCorpus": {"available": False, "reasons": ["vector_corpus_empty"], "total_documents": 0, "collection_name": "knowledge_hub"},
            "warnings": [],
        },
    )
    monkeypatch.setattr(doctor_module, "_module_available", lambda name: False)
    monkeypatch.setattr(
        doctor_module,
        "parser_runtime_status",
        lambda name: {"available": False, "status": "needs_setup", "detail": f"{name} missing", "fixCommand": "install parser"},
    )
    monkeypatch.setattr(doctor_module, "_ollama_ok", lambda url: False)
    monkeypatch.setattr(doctor_module, "_ollama_available", lambda url: False)
    monkeypatch.setattr(doctor_module.registry, "get_provider_info", lambda name: SimpleNamespace(requires_api_key=True, display_name=name, is_local=(name == "ollama")))

    payload = doctor_module.build_doctor_payload(_FakeKhub(config))
    assert payload["schema"] == "knowledge-hub.doctor.result.v1"
    assert payload["status"] in {"needs_setup", "blocked"}
    assert any(item["area"] == "install environment" for item in payload["checks"])
    assert any(item["area"] == "summary" for item in payload["checks"])
    assert any(item["area"] == "paper parser" for item in payload["checks"])
    assert any(item["area"] == "vector corpus" for item in payload["checks"])
    assert payload["nextActions"]


def test_build_doctor_payload_prioritizes_actionable_ollama_recovery(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config.translation_provider = "ollama"
    config.translation_model = "qwen3:14b"
    config.summarization_provider = "ollama"
    config.summarization_model = "qwen3:14b"
    config.embedding_provider = "ollama"
    config.embedding_model = "nomic-embed-text"
    monkeypatch.setattr(
        doctor_module,
        "build_runtime_diagnostics",
        lambda config, searcher=None, searcher_error="": {
            "providers": [
                {"role": "translation", "provider": "ollama", "model": "qwen3:14b", "installed": True, "requires_api_key": False, "api_key_status": "not_required", "degraded": True, "reasons": ["provider_runtime_unavailable"], "available": False},
                {"role": "summarization", "provider": "ollama", "model": "qwen3:14b", "installed": True, "requires_api_key": False, "api_key_status": "not_required", "degraded": True, "reasons": ["provider_runtime_unavailable"], "available": False},
                {"role": "embedding", "provider": "ollama", "model": "nomic-embed-text", "installed": True, "requires_api_key": False, "api_key_status": "not_required", "degraded": True, "reasons": ["provider_runtime_unavailable"], "available": False},
            ],
            "vectorCorpus": {"available": False, "reasons": ["vector_corpus_empty"], "total_documents": 0, "collection_name": "knowledge_hub"},
            "warnings": ["semantic retrieval degraded: provider_runtime_unavailable"],
        },
    )
    monkeypatch.setattr(
        doctor_module,
        "parser_runtime_status",
        lambda name: {"available": True, "status": "ok", "detail": f"{name} ok", "fixCommand": ""},
    )
    monkeypatch.setattr(doctor_module, "_ollama_ok", lambda url: False)
    monkeypatch.setattr(doctor_module, "_ollama_available", lambda url: False)
    monkeypatch.setattr(doctor_module.registry, "get_provider_info", lambda name: SimpleNamespace(requires_api_key=False, display_name=name, is_local=(name == "ollama")))

    payload = doctor_module.build_doctor_payload(_FakeKhub(config))

    checks = {item["area"]: item for item in payload["checks"]}
    assert checks["Ollama"]["status"] == "blocked"
    assert "local runtime unavailable" in checks["embedding"]["detail"]
    assert payload["nextActions"][:3] == [
        "ollama serve  # start the local runtime at http://localhost:11434",
        "ollama pull qwen3:14b",
        "ollama pull nomic-embed-text",
    ]
    assert payload["nextActions"][3].startswith("python -m knowledge_hub.interfaces.cli.main doctor")


def test_build_doctor_payload_prefers_vector_restore_when_restorable_backup_exists(monkeypatch, tmp_path):
    config = _config(tmp_path)
    monkeypatch.setattr(
        doctor_module,
        "build_runtime_diagnostics",
        lambda config, searcher=None, searcher_error="": {
            "providers": [
                {"role": "translation", "provider": "openai", "model": "gpt-5-nano", "installed": True, "requires_api_key": False, "api_key_status": "not_required", "degraded": False, "reasons": [], "available": True},
                {"role": "summarization", "provider": "openai", "model": "gpt-5-nano", "installed": True, "requires_api_key": False, "api_key_status": "not_required", "degraded": False, "reasons": [], "available": True},
                {"role": "embedding", "provider": "ollama", "model": "nomic-embed-text", "installed": True, "requires_api_key": False, "api_key_status": "not_required", "degraded": False, "reasons": [], "available": True},
            ],
            "vectorCorpus": {
                "available": False,
                "reasons": ["vector_corpus_empty"],
                "total_documents": 0,
                "collection_name": "knowledge_hub",
                "recovery_backup": {
                    "path": str(tmp_path / "vector.corrupt.20260416_150415"),
                    "total_documents": 2436,
                    "restorable": True,
                },
            },
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        doctor_module,
        "parser_runtime_status",
        lambda name: {"available": True, "status": "ok", "detail": f"{name} ok", "fixCommand": ""},
    )
    monkeypatch.setattr(doctor_module, "_ollama_ok", lambda url: True)
    monkeypatch.setattr(doctor_module, "_ollama_available", lambda url: True)
    monkeypatch.setattr(doctor_module.registry, "get_provider_info", lambda name: SimpleNamespace(requires_api_key=False, display_name=name, is_local=(name == "ollama")))

    payload = doctor_module.build_doctor_payload(_FakeKhub(config))

    checks = {item["area"]: item for item in payload["checks"]}
    assert checks["vector corpus"]["status"] == "needs_setup"
    assert checks["vector corpus"]["fixCommand"] == "khub vector-compare --latest-backup"


def test_doctor_cmd_json_outputs_public_shape(monkeypatch, tmp_path):
    config = _config(tmp_path)
    Path(config.config_path).write_text("x: 1\n", encoding="utf-8")
    Path(config.papers_dir).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        doctor_module,
        "build_runtime_diagnostics",
        lambda config, searcher=None, searcher_error="": {
            "providers": [
                {"role": "translation", "provider": "openai", "model": "gpt-5-nano", "installed": True, "requires_api_key": False, "api_key_status": "not_required", "degraded": False, "reasons": [], "available": True},
                {"role": "summarization", "provider": "openai", "model": "gpt-5-nano", "installed": True, "requires_api_key": False, "api_key_status": "not_required", "degraded": False, "reasons": [], "available": True},
                {"role": "embedding", "provider": "ollama", "model": "nomic-embed-text", "installed": True, "requires_api_key": False, "api_key_status": "not_required", "degraded": False, "reasons": [], "available": True},
            ],
            "vectorCorpus": {"available": True, "reasons": [], "total_documents": 5, "collection_name": "knowledge_hub"},
            "warnings": [],
        },
    )
    monkeypatch.setattr(doctor_module, "_module_available", lambda name: True)
    monkeypatch.setattr(
        doctor_module,
        "parser_runtime_status",
        lambda name: {"available": True, "status": "ok", "detail": f"{name} ok", "fixCommand": ""},
    )
    monkeypatch.setattr(doctor_module, "_ollama_ok", lambda url: True)
    monkeypatch.setattr(doctor_module, "_ollama_available", lambda url: True)
    monkeypatch.setattr(doctor_module.registry, "get_provider_info", lambda name: SimpleNamespace(requires_api_key=False, display_name=name, is_local=(name == "ollama")))

    runner = CliRunner()
    result = runner.invoke(doctor_module.doctor_cmd, ["--json"], obj={"khub": _FakeKhub(config)})
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.doctor.result.v1"
    assert payload["status"] == "ok"
    assert isinstance(payload["checks"], list)
    assert isinstance(payload["nextActions"], list)
    assert validate_payload(payload, payload["schema"], strict=True).ok
    install_check = next(item for item in payload["checks"] if item["area"] == "install environment")
    assert install_check["diagnostics"]["khubExecutable"] == "/tmp/bin/khub"


def test_install_environment_check_ok_for_matching_editable_paths(tmp_path):
    repo = tmp_path / "repo"
    package_dir = repo / "knowledge_hub"
    package_file = package_dir / "__init__.py"
    package_dir.mkdir(parents=True)
    package_file.write_text("", encoding="utf-8")
    check = doctor_module._build_install_environment_check(
        khub_executable=str(tmp_path / "bin" / "khub"),
        python_executable=str(tmp_path / "bin" / "python"),
        imported_package_file=str(package_file),
        cli_distribution={
            "name": "knowledge-hub-cli",
            "installed": True,
            "version": "0.1.5",
            "editable": True,
            "editableProjectPath": str(repo),
            "topLevelPackagePath": str(package_dir),
        },
        legacy_distribution={"name": "knowledge-hub", "installed": False, "version": ""},
    )

    assert check["status"] == "ok"
    assert check["diagnostics"]["issues"] == []
    assert check["fixCommand"] == ""


def test_install_environment_check_warns_on_legacy_duplicate(tmp_path):
    repo = tmp_path / "repo"
    package_dir = repo / "knowledge_hub"
    package_file = package_dir / "__init__.py"
    package_dir.mkdir(parents=True)
    package_file.write_text("", encoding="utf-8")
    check = doctor_module._build_install_environment_check(
        khub_executable=str(tmp_path / "bin" / "khub"),
        python_executable=str(tmp_path / "bin" / "python"),
        imported_package_file=str(package_file),
        cli_distribution={
            "name": "knowledge-hub-cli",
            "installed": True,
            "version": "0.1.5",
            "editable": True,
            "editableProjectPath": str(repo),
            "topLevelPackagePath": str(package_dir),
        },
        legacy_distribution={"name": "knowledge-hub", "installed": True, "version": "0.1.0"},
    )

    assert check["status"] == "degraded"
    assert "duplicate_legacy_distribution_detected" in check["diagnostics"]["issues"]
    assert "python -m pip uninstall knowledge-hub" in check["recommendedActions"]


def test_install_environment_check_warns_on_stale_editable_mismatch(tmp_path):
    current_repo = tmp_path / "current"
    stale_repo = tmp_path / "stale"
    imported_package = stale_repo / "knowledge_hub" / "__init__.py"
    imported_package.parent.mkdir(parents=True)
    imported_package.write_text("", encoding="utf-8")
    current_package_dir = current_repo / "knowledge_hub"
    current_package_dir.mkdir(parents=True)
    check = doctor_module._build_install_environment_check(
        khub_executable=str(tmp_path / "bin" / "khub"),
        python_executable=str(tmp_path / "bin" / "python"),
        imported_package_file=str(imported_package),
        cli_distribution={
            "name": "knowledge-hub-cli",
            "installed": True,
            "version": "0.1.5",
            "editable": True,
            "editableProjectPath": str(current_repo),
            "topLevelPackagePath": str(current_package_dir),
        },
        legacy_distribution={"name": "knowledge-hub", "installed": False, "version": ""},
    )

    assert check["status"] == "degraded"
    assert "stale_editable_install_detected" in check["diagnostics"]["issues"]
    assert "python -m pip install -e . --no-deps --force-reinstall" in check["recommendedActions"]


def test_install_environment_check_blocks_missing_executable(tmp_path):
    repo = tmp_path / "repo"
    package_file = repo / "knowledge_hub" / "__init__.py"
    package_file.parent.mkdir(parents=True)
    package_file.write_text("", encoding="utf-8")
    check = doctor_module._build_install_environment_check(
        khub_executable="",
        python_executable=str(tmp_path / "bin" / "python"),
        imported_package_file=str(package_file),
        cli_distribution={
            "name": "knowledge-hub-cli",
            "installed": True,
            "version": "0.1.5",
            "editable": True,
            "editableProjectPath": str(repo),
            "topLevelPackagePath": str(package_file.parent),
        },
        legacy_distribution={"name": "knowledge-hub", "installed": False, "version": ""},
    )

    assert check["status"] == "blocked"
    assert "khub_executable_missing" in check["diagnostics"]["issues"]
    assert check["fixCommand"] == "python -m pip install -e . --no-deps --force-reinstall"


def test_install_environment_recommended_actions_feed_next_actions(tmp_path):
    check = doctor_module._build_install_environment_check(
        khub_executable="",
        python_executable=str(tmp_path / "bin" / "python"),
        imported_package_file=str(tmp_path / "repo" / "knowledge_hub" / "__init__.py"),
        cli_distribution={"name": "knowledge-hub-cli", "installed": False, "version": ""},
        legacy_distribution={"name": "knowledge-hub", "installed": True, "version": "0.1.0"},
    )

    actions = doctor_module._next_actions([check], config=_config(tmp_path))

    assert "python -m pip install -e . --no-deps --force-reinstall" in actions
    assert "python -m pip uninstall knowledge-hub" in actions
    assert 'python -c "import knowledge_hub; print(knowledge_hub.__file__)"' in actions
