from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from knowledge_hub.application.remote_index_contract import read_json_object, read_jsonl_objects, verify_checksums
from knowledge_hub.application.paper_retrieval_harness import retrieve_paper_evidence_pack
from knowledge_hub.infrastructure.persistence.vector import VectorDatabase
from knowledge_hub.interfaces.cli.commands.paper_harness_cmd import paper_harness_group
from tests.remote_index_test_support import (
    DeterministicEmbedder,
    NoLocalQwenKhub,
    embedding_vector,
    remote_index_config,
    write_query_embeddings,
)


def _seed_harness_vectors(tmp_path: Path) -> NoLocalQwenKhub:
    config = remote_index_config(tmp_path)
    query = "agentic retrieval planning"
    baseline_embedder = DeterministicEmbedder("nomic-embed-text")
    baseline_db = VectorDatabase(config.vector_db_path, config.collection_name, repair_on_init=False)
    baseline_db.add_documents(
        documents=["Agentic retrieval planning paper baseline evidence chunk."],
        embeddings=[baseline_embedder.embed_text(query)],
        metadatas=[
            {
                "paper_id": "paper-bge",
                "chunk_id": "paper:paper-bge:chunk:0001",
                "title": "Agentic Retrieval Planning",
                "source_hash": "source-bge",
                "chunk_text_hash": "chunk-bge",
                "source_type": "paper",
                "stale": 0,
            }
        ],
        ids=["bge:paper-bge:0001"],
    )
    qwen_db = VectorDatabase(config.vector_db_path, "qwen3_8b_full_candidate", repair_on_init=False)
    qwen_db.add_documents(
        documents=["Qwen8 agentic retrieval planning paper side evidence chunk."],
        embeddings=[embedding_vector(4096)],
        metadatas=[
            {
                "paper_id": "paper-qwen",
                "chunk_id": "paper:paper-qwen:chunk:0001",
                "title": "Qwen Agentic Retrieval",
                "source_hash": "source-qwen",
                "chunk_text_hash": "chunk-qwen",
                "embedding_model": "qwen3-embedding:8b",
                "embedding_dim": 4096,
                "source_type": "paper",
                "stale": 0,
            }
        ],
        ids=["qwen3_8b_full_candidate:paper:paper-qwen:chunk:0001"],
    )
    return NoLocalQwenKhub(config)


def test_retrieve_paper_evidence_pack_merges_bge_keyword_and_qwen_artifact(tmp_path: Path) -> None:
    # Given: bge canonical vectors, qwen8 side vectors, and a matching qwen query artifact.
    khub = _seed_harness_vectors(tmp_path)
    query_embeddings_path = tmp_path / "query_embeddings.qwen3-8b.jsonl"
    write_query_embeddings(query_embeddings_path, model="qwen3-embedding:8b", dim=4096)

    # When: the paper harness retrieves an evidence pack with all arms enabled.
    payload = retrieve_paper_evidence_pack(
        khub=khub,
        query="retrieval planning",
        use_bge=True,
        use_keyword=True,
        use_qwen8=True,
        qwen_query_embeddings_path=query_embeddings_path,
        qwen_query_id="q1",
        top_k=3,
    )

    # Then: the evidence pack is usable by an LLM and includes the qwen8 side arm.
    assert payload["status"] == "ok"
    assert payload["schema"] == "knowledge-hub.labs.paper-retrieval-harness.v1"
    assert {arm["name"]: arm["status"] for arm in payload["arms"]} == {
        "bge": "ok",
        "keyword": "ok",
        "qwen8": "ok",
    }
    assert any("qwen8" in item["retrievalArms"] for item in payload["evidence"])
    assert all(item["paperId"] and item["snippet"] for item in payload["evidence"])


def test_retrieve_paper_evidence_pack_blocks_qwen_without_query_artifact(tmp_path: Path) -> None:
    # Given: local bge evidence exists, but no qwen query embedding artifact is supplied.
    khub = _seed_harness_vectors(tmp_path)

    # When: qwen8 is requested without the query artifact required for 4096-dim search.
    payload = retrieve_paper_evidence_pack(
        khub=khub,
        query="agentic retrieval planning",
        use_bge=True,
        use_keyword=False,
        use_qwen8=True,
        qwen_query_embeddings_path=None,
        qwen_query_id="",
        top_k=3,
    )

    # Then: qwen8 fails closed while the bge evidence remains available.
    assert payload["status"] == "partial"
    arms = {arm["name"]: arm for arm in payload["arms"]}
    assert arms["bge"]["status"] == "ok"
    assert arms["qwen8"]["status"] == "blocked"
    assert "qwen_query_embedding_required" in arms["qwen8"]["blockers"]
    assert payload["evidence"]


def test_retrieve_paper_evidence_pack_blocks_qwen_query_text_hash_mismatch(tmp_path: Path) -> None:
    # Given: a qwen query artifact generated for a different query text.
    khub = _seed_harness_vectors(tmp_path)
    query_embeddings_path = tmp_path / "query_embeddings.qwen3-8b.jsonl"
    write_query_embeddings(query_embeddings_path, model="qwen3-embedding:8b", dim=4096)

    # When: the paper harness is asked to use that artifact for another query.
    payload = retrieve_paper_evidence_pack(
        khub=khub,
        query="different retrieval question",
        use_bge=True,
        use_keyword=False,
        use_qwen8=True,
        qwen_query_embeddings_path=query_embeddings_path,
        qwen_query_id="q1",
        top_k=3,
    )

    # Then: qwen8 blocks closed instead of searching with a mismatched vector.
    assert payload["status"] == "partial"
    arms = {arm["name"]: arm for arm in payload["arms"]}
    assert arms["qwen8"]["status"] == "blocked"
    assert "query_text_hash_mismatch" in arms["qwen8"]["blockers"]


def test_paper_harness_retrieve_cli_emits_json_evidence_pack(tmp_path: Path) -> None:
    # Given: a seeded harness runtime and a qwen query artifact.
    khub = _seed_harness_vectors(tmp_path)
    query_embeddings_path = tmp_path / "query_embeddings.qwen3-8b.jsonl"
    write_query_embeddings(query_embeddings_path, model="qwen3-embedding:8b", dim=4096)

    # When: the labs paper-harness retrieve command is invoked.
    result = CliRunner().invoke(
        paper_harness_group,
        [
            "retrieve",
            "--query",
            "retrieval planning",
            "--qwen-query-embeddings",
            str(query_embeddings_path),
            "--qwen-query-id",
            "q1",
            "--top-k",
            "3",
            "--json",
        ],
        obj={"khub": khub},
    )

    # Then: the CLI returns the same schema-backed evidence pack.
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["query"] == "retrieval planning"
    assert payload["evidence"]


def test_paper_harness_query_export_writes_sanitized_oracle_bundle(tmp_path: Path) -> None:
    # Given: an arbitrary user question that needs a qwen8 query embedding.
    run_dir = tmp_path / "query-run"

    # When: the labs paper-harness query-export command builds a report-only bundle.
    result = CliRunner().invoke(
        paper_harness_group,
        [
            "query-export",
            "--query",
            "한국어 질문으로 agentic RAG 평가 논문 찾기",
            "--out",
            str(run_dir),
            "--json",
        ],
    )

    # Then: only sanitized query embedding request artifacts are produced.
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ready"
    assert payload["queryCount"] == 1
    assert payload["targetModel"] == "qwen3-embedding:8b"
    manifest = read_json_object(run_dir / "input_manifest.json")
    assert manifest["schema"] == "knowledge-hub.labs.paper-harness.query-embedding-request.v1"
    rows = read_jsonl_objects(run_dir / "queries.jsonl")
    assert rows[0]["query_id"] == "query:0001"
    assert rows[0]["query"] == "한국어 질문으로 agentic RAG 평가 논문 찾기"
    assert rows[0]["embedding_model"] == "qwen3-embedding:8b"
    assert rows[0]["expected_embedding_dim"] == 4096
    assert "Do not scan any vault" in (run_dir / "hermes_prompt.md").read_text(encoding="utf-8")
    assert verify_checksums(run_dir) == []


def test_paper_harness_query_export_blocks_empty_queries(tmp_path: Path) -> None:
    # Given: a query-export request with no non-empty query text.
    run_dir = tmp_path / "empty-query-run"

    # When: the command is invoked with blank query text.
    result = CliRunner().invoke(
        paper_harness_group,
        ["query-export", "--query", "   ", "--out", str(run_dir), "--json"],
    )

    # Then: it blocks without producing an Oracle handoff bundle.
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "blocked"
    assert "query_required" in payload["blockers"]
    assert not (run_dir / "queries.jsonl").exists()
