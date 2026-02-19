"""
벡터 인덱싱 워커 - 별도 프로세스에서 실행
"""
import json
import os
import sys


def main():
    for candidate in [os.path.join(os.getcwd(), ".env"), os.path.join(os.path.dirname(__file__), "..", "..", ".env")]:
        if os.path.exists(candidate):
            for line in open(candidate):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
            break

    data = json.loads(sys.argv[1])

    from knowledge_hub.core.config import Config
    Config.reset()
    config = Config()

    from knowledge_hub.providers.registry import get_embedder
    from knowledge_hub.core.database import VectorDatabase

    embed_cfg = config.get_provider_config(config.embedding_provider)
    embedder = get_embedder(config.embedding_provider, model=config.embedding_model, **embed_cfg)
    vector_db = VectorDatabase(config.vector_db_path, config.collection_name)

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
            valid = [(t, e, i) for i, (t, e) in enumerate(zip(chunks, embeddings)) if e is not None]
            if valid:
                vector_db.add_documents(
                    documents=[v[0] for v in valid],
                    embeddings=[v[1] for v in valid],
                    metadatas=[{
                        "title": title,
                        "arxiv_id": arxiv_id,
                        "source_type": "paper",
                        "field": field,
                        "chunk_index": v[2],
                    } for v in valid],
                    ids=[f"paper_{arxiv_id}_{v[2]}" for v in valid],
                )
            print(json.dumps({"arxiv_id": arxiv_id, "chunks": len(valid), "ok": True}))
        except Exception as e:
            print(json.dumps({"arxiv_id": arxiv_id, "chunks": 0, "ok": False, "error": str(e)}))
        sys.stdout.flush()

    print(json.dumps({"done": True}))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
