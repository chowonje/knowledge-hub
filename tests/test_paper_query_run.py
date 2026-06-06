from __future__ import annotations

import hashlib
import json
from pathlib import Path

from click.testing import CliRunner

from knowledge_hub.application.paper_query_export import export_query_embedding_bundle
from knowledge_hub.application.remote_index_contract import read_json_object, read_jsonl_objects, verify_checksums
from knowledge_hub.infrastructure.persistence.vector import VectorDatabase
from knowledge_hub.interfaces.cli.commands.paper_harness_cmd import paper_harness_group
from tests.remote_index_test_support import (
    DeterministicEmbedder,
    NoLocalQwenKhub,
    embedding_vector,
    remote_index_config,
)


def _write_query_embedding_output(run_dir: Path, *, query_text: str = "retrieval planning") -> None:
    output_path = run_dir / "query_embeddings.qwen3-8b.jsonl"
    row = {
        "query_id": "query:0001",
        "query": query_text,
        "query_text_hash": hashlib.sha256(query_text.encode("utf-8")).hexdigest(),
        "embedding_model": "qwen3-embedding:8b",
        "embedding_dim": 4096,
        "embedding": embedding_vector(4096),
    }
    output_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    (run_dir / "checksums.output.sha256").write_text(f"{digest}  {output_path.name}\n", encoding="utf-8")


def _seed_harness_vectors(tmp_path: Path) -> NoLocalQwenKhub:
    config = remote_index_config(tmp_path)
    baseline_embedder = DeterministicEmbedder("nomic-embed-text")
    baseline_db = VectorDatabase(config.vector_db_path, config.collection_name, repair_on_init=False)
    baseline_db.add_documents(
        documents=["Agentic retrieval planning paper baseline evidence chunk."],
        embeddings=[baseline_embedder.embed_text("retrieval planning")],
        metadatas=[{"paper_id": "paper-bge", "chunk_id": "paper-bge:1", "source_type": "paper", "stale": 0}],
        ids=["bge:paper-bge:0001"],
    )
    qwen_db = VectorDatabase(config.vector_db_path, "qwen3_8b_full_candidate", repair_on_init=False)
    qwen_db.add_documents(
        documents=["Qwen8 retrieval planning paper side evidence chunk."],
        embeddings=[embedding_vector(4096)],
        metadatas=[{"paper_id": "paper-qwen", "chunk_id": "paper-qwen:1", "source_type": "paper", "stale": 0}],
        ids=["qwen3_8b_full_candidate:paper-qwen:0001"],
    )
    return NoLocalQwenKhub(config)


def test_query_validate_accepts_completed_oracle_run(tmp_path: Path) -> None:
    # Given: a query-export bundle with a matching Oracle query embedding output.
    run_dir = tmp_path / "query-run"
    export_query_embedding_bundle(queries=("retrieval planning",), out_dir=run_dir)
    _write_query_embedding_output(run_dir)

    # When: the completed query run is validated through the CLI.
    result = CliRunner().invoke(
        paper_harness_group,
        ["query-validate", "--run", str(run_dir), "--json"],
    )

    # Then: the run is ready for paper-harness retrieval.
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ready"
    assert payload["queryIds"] == ["query:0001"]
    assert payload["queryEmbeddingPath"].endswith("query_embeddings.qwen3-8b.jsonl")
    assert payload["blockers"] == []


def test_query_validate_blocks_output_checksum_mismatch(tmp_path: Path) -> None:
    # Given: a query run whose output checksum no longer matches the embedding artifact.
    run_dir = tmp_path / "query-run"
    export_query_embedding_bundle(queries=("retrieval planning",), out_dir=run_dir)
    _write_query_embedding_output(run_dir)
    (run_dir / "query_embeddings.qwen3-8b.jsonl").write_text("tampered\n", encoding="utf-8")

    # When: the completed query run is validated.
    result = CliRunner().invoke(
        paper_harness_group,
        ["query-validate", "--run", str(run_dir), "--json"],
    )

    # Then: the run blocks before retrieve-from-run can use it.
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "blocked"
    assert "checksum_mismatch:query_embeddings.qwen3-8b.jsonl" in payload["blockers"]


def test_retrieve_from_run_uses_validated_query_embedding_output(tmp_path: Path) -> None:
    # Given: a validated query run and local bge/qwen side indexes.
    khub = _seed_harness_vectors(tmp_path)
    run_dir = tmp_path / "query-run"
    export_query_embedding_bundle(queries=("retrieval planning",), out_dir=run_dir)
    _write_query_embedding_output(run_dir)
    assert read_jsonl_objects(run_dir / "queries.jsonl")[0]["query_id"] == "query:0001"

    # When: retrieve-from-run is invoked with the query id.
    result = CliRunner().invoke(
        paper_harness_group,
        ["retrieve-from-run", "--run", str(run_dir), "--query-id", "query:0001", "--top-k", "3", "--json"],
        obj={"khub": khub},
    )

    # Then: it returns an evidence pack with the qwen8 arm enabled.
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["query"] == "retrieval planning"
    assert {arm["name"]: arm["status"] for arm in payload["arms"]}["qwen8"] == "ok"


def test_query_remote_plan_writes_worker_and_operator_commands(tmp_path: Path) -> None:
    # Given: a sanitized query-export bundle ready for Oracle query embedding.
    run_dir = tmp_path / "query-run"
    export_query_embedding_bundle(queries=("retrieval planning",), out_dir=run_dir)

    # When: the paper harness prepares the report-only remote execution plan.
    result = CliRunner().invoke(
        paper_harness_group,
        [
            "query-remote-plan",
            "--run",
            str(run_dir),
            "--remote-host",
            "oracle-hermes",
            "--remote-root",
            "~/knowledgeos-remote-indexing",
            "--session",
            "qwen8_query_test",
            "--json",
        ],
    )

    # Then: it writes a query-only worker and operator commands without invalidating input checksums.
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ready"
    assert payload["targetModel"] == "qwen3-embedding:8b"
    assert payload["queryCount"] == 1
    assert payload["remote"]["host"] == "oracle-hermes"
    assert payload["remote"]["session"] == "qwen8_query_test"
    assert "scp -r" in payload["operatorCommands"]["uploadBundle"]
    assert str(run_dir.resolve()) in payload["operatorCommands"]["uploadBundle"]
    assert "tmux new -d -s qwen8_query_test" in payload["operatorCommands"]["startWorker"]
    assert "query_embeddings.qwen3-8b.jsonl" in payload["operatorCommands"]["fetchOutputs"]
    assert "query-validate" in payload["operatorCommands"]["validateLocal"]
    assert "canonical" not in payload["operatorCommands"]["startWorker"]
    worker_script = run_dir / "run_query_embeddings.py"
    assert worker_script.exists()
    worker_text = worker_script.read_text(encoding="utf-8")
    assert "queries.jsonl" in worker_text
    assert "query_text_hash" in worker_text
    assert "http://127.0.0.1:11434/api/embed" in worker_text
    assert "khub" not in worker_text.lower()
    assert read_json_object(run_dir / "remote_operator_plan.json")["status"] == "ready"
    assert verify_checksums(run_dir) == []


def test_query_remote_plan_blocks_invalid_query_bundle(tmp_path: Path) -> None:
    # Given: a run directory that is missing the query-export manifest.
    run_dir = tmp_path / "missing-query-bundle"
    run_dir.mkdir()

    # When: the remote plan command is invoked.
    result = CliRunner().invoke(
        paper_harness_group,
        ["query-remote-plan", "--run", str(run_dir), "--json"],
    )

    # Then: it fails closed and writes no worker script.
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "blocked"
    assert "input_manifest_missing" in payload["blockers"]
    assert "queries_missing" in payload["blockers"]
    assert not (run_dir / "run_query_embeddings.py").exists()


def test_query_remote_plan_preserves_remote_home_when_shell_expands_local_home(tmp_path: Path) -> None:
    # Given: a query bundle and a remote root value already expanded by the local shell.
    run_dir = tmp_path / "query-run"
    export_query_embedding_bundle(queries=("retrieval planning",), out_dir=run_dir)
    locally_expanded_remote_root = str(Path.home() / "knowledgeos-remote-indexing")

    # When: the remote plan is built from that shell-expanded value.
    result = CliRunner().invoke(
        paper_harness_group,
        [
            "query-remote-plan",
            "--run",
            str(run_dir),
            "--remote-root",
            locally_expanded_remote_root,
            "--json",
        ],
    )

    # Then: the operator commands still target the remote user's home directory.
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["remote"]["root"] == "~/knowledgeos-remote-indexing"
    assert "oracle-hermes:~/knowledgeos-remote-indexing/" in payload["operatorCommands"]["uploadBundle"]


def test_query_remote_plan_operator_commands_use_absolute_local_run_path(tmp_path: Path, monkeypatch) -> None:
    # Given: an operator invokes the plan command with a relative local run path.
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)
    run_dir = Path("query-run")
    export_query_embedding_bundle(queries=("retrieval planning",), out_dir=run_dir)

    # When: the remote plan is built from the relative path.
    result = CliRunner().invoke(
        paper_harness_group,
        ["query-remote-plan", "--run", str(run_dir), "--json"],
    )

    # Then: copy/fetch/validate commands remain usable outside the original shell cwd.
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    absolute_run_dir = str((work_dir / run_dir).resolve())
    assert absolute_run_dir in payload["operatorCommands"]["uploadBundle"]
    assert absolute_run_dir in payload["operatorCommands"]["fetchOutputs"]
    assert f"--run {absolute_run_dir} --json" in payload["operatorCommands"]["validateLocal"]
