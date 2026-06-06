from __future__ import annotations

import hashlib
import json
from pathlib import Path

from click.testing import CliRunner

from knowledge_hub.application.remote_index_contract import sha256_text
from knowledge_hub.core.config import Config
from knowledge_hub.core.database import SQLiteDatabase


class NoLocalQwenKhub:
    def __init__(self, config: Config) -> None:
        self.config = config

    def sqlite_db(self) -> SQLiteDatabase:
        return SQLiteDatabase(self.config.sqlite_path)

    def build_embedder(self, provider: str, model: str | None = None) -> DeterministicEmbedder:
        if str(model or "").startswith("qwen3-embedding:"):
            raise AssertionError("qwen embedding must come from artifact, not local Ollama")
        return DeterministicEmbedder(model or "nomic-embed-text")


class DeterministicEmbedder:
    def __init__(self, model: str) -> None:
        self.model = model

    def embed_text(self, text: str) -> list[float]:
        seed = hashlib.sha256(f"{self.model}:{text}".encode("utf-8")).digest()
        return [float(seed[index]) / 255.0 for index in range(8)]


def remote_index_config(tmp_path: Path) -> Config:
    config = Config()
    config.set_nested("storage", "sqlite", str(tmp_path / "knowledge.db"))
    config.set_nested("storage", "papers_dir", str(tmp_path / "papers"))
    config.set_nested("storage", "vector_db", str(tmp_path / "vector_db"))
    config.set_nested("storage", "collection_name", "knowledge_hub")
    config.set_nested("embedding", "provider", "ollama")
    config.set_nested("embedding", "model", "nomic-embed-text")
    return config


def seed_papers(config: Config, count: int = 1) -> None:
    db = SQLiteDatabase(config.sqlite_path)
    try:
        for index in range(count):
            paper_id = f"2601.{index:05d}"
            text_path = Path(config.papers_dir) / f"{paper_id}.txt"
            text_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.write_text(
                f"Paper {paper_id} retrieval planning oracle embedding. " * 40,
                encoding="utf-8",
            )
            db.upsert_paper(
                {
                    "arxiv_id": paper_id,
                    "title": f"Remote Index Paper {index}",
                    "authors": "A",
                    "year": 2026,
                    "field": "AI",
                    "importance": 3,
                    "notes": "seed notes",
                    "pdf_path": None,
                    "text_path": str(text_path),
                    "translated_path": None,
                }
            )
    finally:
        db.close()


def export_run(tmp_path: Path, config: Config) -> Path:
    from knowledge_hub.interfaces.cli.commands.remote_index_cmd import remote_index_group

    run_dir = tmp_path / "run"
    result = CliRunner().invoke(
        remote_index_group,
        ["export", "--paper-id", "2601.00000", "--out", str(run_dir), "--json"],
        obj={"khub": NoLocalQwenKhub(config)},
    )
    assert result.exit_code == 0, result.output
    return run_dir


def embedding_vector(dim: int) -> list[float]:
    return [float(index + 1) for index in range(dim)]


def write_embeddings(run_dir: Path, *, model: str, suffix: str, dim: int) -> None:
    chunks = [json.loads(line) for line in (run_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [
        {
            "chunk_id": chunk["chunk_id"],
            "paper_id": chunk["paper_id"],
            "source_hash": chunk["source_hash"],
            "chunk_text_hash": chunk["chunk_text_hash"],
            "embedding_model": model,
            "embedding_dim": dim,
            "embedding": embedding_vector(dim),
        }
        for chunk in chunks
    ]
    embeddings_path = run_dir / f"embeddings.{suffix}.jsonl"
    embeddings_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(embeddings_path.read_bytes()).hexdigest()
    (run_dir / "checksums.output.sha256").write_text(f"{digest}  {embeddings_path.name}\n", encoding="utf-8")


def write_queries(path: Path) -> None:
    path.write_text(
        "query_id,query,expected_paper_id\nq1,retrieval planning,2601.00000\n",
        encoding="utf-8",
    )


def write_query_embeddings(path: Path, *, model: str, dim: int) -> None:
    row = {
        "query_id": "q1",
        "query_text_hash": sha256_text("retrieval planning"),
        "embedding_model": model,
        "embedding_dim": dim,
        "embedding": embedding_vector(dim),
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
