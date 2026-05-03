"""Canonical vector indexing worker entrypoint."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _auto_load_dotenv() -> None:
    for candidate in [Path.cwd() / ".env", Path(__file__).resolve().parents[3] / ".env"]:
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())
            break


def main():
    _auto_load_dotenv()
    data = json.loads(sys.argv[1])

    from knowledge_hub.application.context import AppContextFactory

    factory = AppContextFactory()
    config = factory.config
    embedder = factory.build_embedder(config.embedding_provider, config.embedding_model)
    vector_db = factory.create_vector_db()

    for paper in data:
        arxiv_id = paper["arxiv_id"]
        title = paper["title"]
        abstract = paper.get("abstract", "")
        summary = paper.get("summary", "")
        field = paper.get("field", "")

        text_parts = [f"Title: {title}"]
        if abstract:
            text_parts.append(f"Abstract: {abstract}")
        if summary:
            text_parts.append(f"Summary: {summary}")
        full_text = "\n\n".join(text_parts)

        chunks = []
        start = 0
        while start < len(full_text):
            end = min(start + 1000, len(full_text))
            chunk = full_text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - 200

        if not chunks:
            print(json.dumps({"arxiv_id": arxiv_id, "chunks": 0, "ok": False, "error": "no chunks"}))
            sys.stdout.flush()
            continue

        try:
            embeddings = embedder.embed_batch(chunks)
            valid = [(text, embedding, index) for index, (text, embedding) in enumerate(zip(chunks, embeddings)) if embedding is not None]
            if valid:
                vector_db.add_documents(
                    documents=[item[0] for item in valid],
                    embeddings=[item[1] for item in valid],
                    metadatas=[
                        {
                            "title": title,
                            "arxiv_id": arxiv_id,
                            "source_type": "paper",
                            "field": field,
                            "chunk_index": item[2],
                        }
                        for item in valid
                    ],
                    ids=[f"paper_{arxiv_id}_{item[2]}" for item in valid],
                )
            print(json.dumps({"arxiv_id": arxiv_id, "chunks": len(valid), "ok": True}))
        except Exception as error:
            print(json.dumps({"arxiv_id": arxiv_id, "chunks": 0, "ok": False, "error": str(error)}))
        sys.stdout.flush()

    print(json.dumps({"done": True}))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
