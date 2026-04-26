from __future__ import annotations

from pathlib import Path

from knowledge_hub.application.public_release_hygiene import check_public_release_hygiene


def test_public_release_hygiene_passes_for_safe_placeholder_files(tmp_path: Path):
    (tmp_path / "README.md").write_text("Use OPENAI_API_KEY from your environment.\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text("OPENAI_API_KEY=sk-...\n", encoding="utf-8")
    (tmp_path / "config.yaml.example").write_text("summarization:\n  provider: ollama\n", encoding="utf-8")

    payload = check_public_release_hygiene(
        tmp_path,
        tracked_files=["README.md", ".env.example", "config.yaml.example"],
    )

    assert payload["status"] == "ok"
    assert payload["issueCount"] == 0


def test_public_release_hygiene_flags_tracked_local_files_secret_literals_and_absolute_paths(tmp_path: Path):
    (tmp_path / "config.yaml.bak_20260412").write_text("summarization:\n  provider: openai\n", encoding="utf-8")
    fake_secret = "sk-" + "abcdefghijklmnopqrstuvwxyz123456"
    fake_path = "/Users" + "/local-user/private/project/config.yaml"
    (tmp_path / "notes.md").write_text(
        f'value = "{fake_secret}"\npath: `{fake_path}`\nplist: <string>{fake_path}</string>\n',
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("OPENAI_API_KEY=abc\n", encoding="utf-8")

    payload = check_public_release_hygiene(
        tmp_path,
        tracked_files=["config.yaml.bak_20260412", "notes.md", ".env"],
    )

    assert payload["status"] == "failed"
    assert payload["issueCount"] >= 4
    kinds = set(payload["issueCountsByKind"])
    assert "tracked_config_backup" in kinds
    assert "tracked_dotenv" in kinds
    assert "openai_style_key" in kinds
    assert "absolute_user_path" in kinds


def test_public_release_hygiene_flags_runtime_sqlite_and_generated_run_paths(tmp_path: Path):
    tracked = [
        ".khub.sqlite",
        "eval/knowledgeos/runs/run_001/summary.json",
        "eval/knowledgeos/failures/failure_bank.jsonl",
        "runs/ab/khoj/result.json",
    ]
    for rel_path in tracked:
        path = tmp_path / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    payload = check_public_release_hygiene(tmp_path, tracked_files=tracked)

    assert payload["status"] == "failed"
    kinds = payload["issueCountsByKind"]
    assert kinds["tracked_runtime_database"] == 1
    assert kinds["tracked_generated_eval_run"] == 1
    assert kinds["tracked_generated_eval_failure_bank"] == 1
    assert kinds["tracked_generated_ab_run"] == 1
