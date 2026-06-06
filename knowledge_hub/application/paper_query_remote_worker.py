from __future__ import annotations

from typing import Final

QUERY_REMOTE_WORKER_TEMPLATE: Final = """from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Final, TypeAlias

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

MODEL: Final = "__MODEL__"
OUTPUT_FILE: Final = "__OUTPUT_FILE__"
OLLAMA_EMBED_URL: Final = "http://127.0.0.1:11434/api/embed"
BATCH_SIZE: Final = 8


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError(f"jsonl row is not an object: {path}")
        rows.append(payload)
    return rows


def verify_checksums(root: Path, filename: str = "checksums.sha256") -> list[str]:
    checksum_path = root / filename
    if not checksum_path.exists():
        return ["checksums_missing"]
    blockers: list[str] = []
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, _, relative = line.partition("  ")
        target = root / relative.strip()
        if not digest or not relative.strip():
            blockers.append("checksum_row_invalid")
        elif not target.exists():
            blockers.append(f"checksum_target_missing:{relative.strip()}")
        elif sha256_file(target) != digest.strip():
            blockers.append(f"checksum_mismatch:{relative.strip()}")
    return blockers


def embed_texts(texts: list[str]) -> list[list[float]]:
    body = json.dumps({"model": MODEL, "input": texts, "keep_alive": "30m"}).encode("utf-8")
    request = urllib.request.Request(OLLAMA_EMBED_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.load(response)
    embeddings = payload.get("embeddings") if isinstance(payload, dict) else None
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise RuntimeError("ollama_embedding_count_mismatch")
    return [[float(value) for value in embedding] for embedding in embeddings]


def query_id(row: JsonObject, index: int) -> str:
    raw_query_id = str(row.get("query_id") or row.get("queryId") or "").strip()
    return raw_query_id or f"query:{index:04d}"


def query_text(row: JsonObject) -> str:
    return str(row.get("query") or "").strip()


def query_hash(row: JsonObject) -> str:
    raw_query_hash = str(row.get("query_text_hash") or "").strip()
    return raw_query_hash or sha256_text(query_text(row))


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    rows = read_jsonl(path)
    return {str(row.get("query_id") or row.get("queryId") or "").strip() for row in rows if row.get("query_id")}


def write_output_checksums(root: Path, names: list[str]) -> None:
    lines: list[str] = []
    for name in names:
        path = root / name
        if path.exists():
            lines.append(f"{sha256_file(path)}  {name}")
    (root / "checksums.output.sha256").write_text("\\n".join(lines) + "\\n", encoding="utf-8")


def generate(root: Path) -> JsonObject:
    manifest = json.loads((root / "input_manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("input_manifest_not_object")
    if str(manifest.get("targetModel") or "") != MODEL:
        raise RuntimeError("input_manifest_model_mismatch")
    checksum_blockers = verify_checksums(root)
    if checksum_blockers:
        raise RuntimeError(",".join(checksum_blockers))
    rows = read_jsonl(root / "queries.jsonl")
    output = root / OUTPUT_FILE
    temp = root / f"{OUTPUT_FILE}.tmp"
    completed = completed_ids(output if output.exists() else temp)
    remaining = [(index, row) for index, row in enumerate(rows, start=1) if query_id(row, index) not in completed]
    dimensions: set[int] = set()
    written = len(completed)
    started = time.monotonic()
    target = output if output.exists() else temp
    with target.open("a", encoding="utf-8") as handle:
        for offset in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[offset : offset + BATCH_SIZE]
            embeddings = embed_texts([query_text(row) for _, row in batch])
            for (index, row), embedding in zip(batch, embeddings, strict=True):
                dimensions.add(len(embedding))
                payload = {
                    "query_id": query_id(row, index),
                    "query": query_text(row),
                    "query_text_hash": query_hash(row),
                    "embedding_model": MODEL,
                    "embedding_dim": len(embedding),
                    "embedding": embedding,
                }
                handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\\n")
                written += 1
            print(json.dumps({"phase": "queries", "written": written, "total": len(rows)}), flush=True)
    if temp.exists() and not output.exists() and written == len(rows):
        temp.replace(output)
    return {
        "outputFile": OUTPUT_FILE,
        "inputRows": len(rows),
        "embeddedRows": written,
        "failedRows": len(rows) - written,
        "dimensions": sorted(dimensions),
        "elapsedSeconds": round(time.monotonic() - started, 3),
    }


def main() -> int:
    root = Path.cwd()
    errors: list[str] = []
    result: JsonObject = {}
    started = time.monotonic()
    try:
        result = generate(root)
    except (OSError, RuntimeError, TypeError, ValueError, urllib.error.URLError, json.JSONDecodeError) as error:
        errors.append(f"{type(error).__name__}:{error}")
    status = "ok" if not errors and int(result.get("failedRows") or 0) == 0 else "failed"
    report = {
        "status": status,
        "modelNames": [MODEL],
        "queryCount": int(result.get("inputRows") or 0),
        "embeddedQueryRows": int(result.get("embeddedRows") or 0),
        "failedQueryRows": int(result.get("failedRows") or 0),
        "observedEmbeddingDimensions": list(result.get("dimensions") or []),
        "elapsedSeconds": round(time.monotonic() - started, 3),
        "runtimeErrors": errors,
        "queryEmbedding": result,
    }
    (root / "run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
    (root / "run_report.md").write_text(
        "\\n".join(
            [
                "# Qwen8 Query Embedding Run Report",
                "",
                f"- status: {status}",
                f"- model: {MODEL}",
                f"- query rows: {report['embeddedQueryRows']} / {report['queryCount']}",
                f"- dimensions: {report['observedEmbeddingDimensions']}",
                f"- elapsed seconds: {report['elapsedSeconds']}",
                f"- runtime errors: {errors}",
            ]
        )
        + "\\n",
        encoding="utf-8",
    )
    write_output_checksums(root, [OUTPUT_FILE, "run_report.json", "run_report.md"])
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
"""


def build_query_remote_worker(*, model: str, output_file: str) -> str:
    return QUERY_REMOTE_WORKER_TEMPLATE.replace("__MODEL__", model).replace("__OUTPUT_FILE__", output_file)


__all__ = ["build_query_remote_worker"]
