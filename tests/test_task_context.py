from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from knowledge_hub.application.task_context import build_task_context, classify_task_mode


class _FakeSearcher:
    def __init__(self):
        self.search_calls = []

    def search(self, query, **kwargs):  # noqa: ANN001
        self.search_calls.append((query, kwargs))
        return [
            SimpleNamespace(
                metadata={
                    "title": "Vault Note",
                    "source_type": "note",
                    "file_path": "notes/vault.md",
                    "document_scope_id": "vault:notes/vault.md",
                    "section_scope_id": "vault:notes/vault.md::section:Overview",
                    "stable_scope_id": "vault:notes/vault.md::section:Overview",
                    "scope_level": "section",
                },
                score=0.91,
                document="vault evidence",
            ),
            SimpleNamespace(
                metadata={"title": "Paper Note", "source_type": "paper"},
                score=0.82,
                document="paper evidence",
            ),
            SimpleNamespace(
                metadata={"title": "Web Note", "source_type": "web"},
                score=0.77,
                document="web evidence",
            ),
        ]


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_classify_task_mode_distinguishes_coding_and_knowledge():
    assert classify_task_mode("Implement agent context for cli-agent.ts") == "coding"
    assert classify_task_mode("debug failing retrieval path") == "debug"
    assert classify_task_mode("architecture design for task context") == "design"
    assert classify_task_mode("what is retrieval augmented generation?") == "knowledge"


def test_build_task_context_prioritizes_project_docs_and_respects_excludes(tmp_path):
    repo = tmp_path / "repo"
    _write(repo / "AGENTS.md", "- Preserve boundaries\n- Prefer inspectable outputs\n")
    _write(repo / "README.md", "- Repo overview\n")
    _write(repo / "docs" / "PROJECT_STATE.md", "- Current architecture\n")
    _write(repo / "src" / "agent.ts", "export const taskContext = true;\n")
    _write(repo / "node_modules" / "ignored.ts", "export const ignored = true;\n")

    payload = build_task_context(
        _FakeSearcher(),
        goal="Implement task context in src/agent.ts using project conventions",
        repo_path=str(repo),
        include_workspace=True,
        max_workspace_files=4,
        max_knowledge_hits=3,
    )

    rel_paths = [item["relative_path"] for item in payload["workspace_files"]]
    assert rel_paths[:3] == ["AGENTS.md", "README.md", "docs/PROJECT_STATE.md"]
    assert "node_modules/ignored.ts" not in rel_paths
    assert payload["workspace_files"][-1]["relative_path"] == "src/agent.ts"
    assert payload["workspace_files"][0]["source_type"] == "project"
    assert payload["knowledge_hits"][0]["scope_level"] == "section"
    assert payload["knowledge_hits"][0]["stable_scope_id"] == "vault:notes/vault.md::section:Overview"
    assert payload["knowledge_hits"][0]["document_scope_id"] == "vault:notes/vault.md"
    assert payload["knowledge_hits"][0]["section_scope_id"] == "vault:notes/vault.md::section:Overview"
    assert payload["project_conventions"]
    assert payload["runtimeDiagnostics"]["schema"] == "knowledge-hub.runtime.diagnostics.v1"
    assert "Ephemeral workspace evidence:" in payload["suggested_prompt_context"]


def test_build_task_context_keeps_repo_context_ephemeral(tmp_path):
    repo = tmp_path / "repo"
    _write(repo / "README.md", "repo readme\n")

    class _NoPersistenceSearcher(_FakeSearcher):
        def __init__(self):
            super().__init__()
            self.sqlite_db = SimpleNamespace(
                add_note=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected sqlite write"))
            )
            self.database = SimpleNamespace(
                add_documents=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected vector write"))
            )

    payload = build_task_context(
        _NoPersistenceSearcher(),
        goal="Implement feature from README.md",
        repo_path=str(repo),
        include_workspace=True,
        max_workspace_files=2,
    )

    assert payload["schema"] == "knowledge-hub.task-context.result.v1"
    assert payload["evidenceSummary"]["repoContextEphemeral"] is True
    assert payload["gateway"]["surface"] == "task_context"
    assert payload["gateway"]["mode"] == "context"
    assert payload["gateway"]["executionAllowed"] is False
    assert payload["runtimeDiagnostics"]["schema"] == "knowledge-hub.runtime.diagnostics.v1"
    assert all(item["source_type"] == "project" for item in payload["workspace_files"])


def test_build_task_context_excludes_virtualenv_and_vendor_paths_but_keeps_repo_matches(tmp_path):
    repo = tmp_path / "repo"
    _write(repo / "src" / "rag.py", "def build():\n    return True\n")
    _write(repo / ".venv3108" / "lib" / "python3.10" / "site-packages" / "rag.py", "bad = True\n")
    _write(repo / "vendor" / "generated" / "rag.py", "bad = True\n")
    _write(repo / "build" / "rag.py", "bad = True\n")

    payload = build_task_context(
        _FakeSearcher(),
        goal="Inspect rag.py and explain retrieval flow",
        repo_path=str(repo),
        include_workspace=True,
        max_workspace_files=5,
        max_knowledge_hits=1,
    )

    rel_paths = [item["relative_path"] for item in payload["workspace_files"]]
    assert "src/rag.py" in rel_paths
    assert all(".venv" not in path for path in rel_paths)
    assert all("site-packages" not in path for path in rel_paths)
    assert all("vendor/" not in path for path in rel_paths)
    assert all("build/" not in path for path in rel_paths)
