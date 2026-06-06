from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final, TypeAlias

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

REMOTE_INDEX_SCHEMA: Final = "knowledge-hub.labs.remote-index.run.v1"
TARGET_MODELS: Final = ("qwen3-embedding:4b", "qwen3-embedding:8b")
MODEL_SUFFIXES: Final = {
    "qwen3-embedding:4b": "qwen3-4b",
    "qwen3-embedding:8b": "qwen3-8b",
}

HERMES_PROMPT: Final = """You are the Oracle remote embedding worker for KnowledgeOS.

Mode: report_only remote embedding pilot.

Goal:
Given a sanitized input bundle containing preprocessed paper chunks, generate primary embeddings with:
1. qwen3-embedding:8b

If VM capacity or runtime errors prevent 8B completion, generate fallback embeddings with:
1. qwen3-embedding:4b

Hard boundaries:
- Do not scan any vault.
- Do not read or request private local paths outside the provided bundle.
- Do not run khub commands.
- Do not mutate any canonical SQLite DB, Chroma DB, vector index, repository file, or Obsidian vault.
- Do not download papers or parse PDFs.
- Do not change chunk text, chunk ids, metadata, or source hashes.
- Treat all input chunks as already finalized by local KnowledgeOS.
- Output only derived embedding artifacts, validation logs, and a report.

Input bundle expected:
- input_manifest.json
- chunks.jsonl
- checksums.sha256
- README.md or run_instructions.md

Tasks:
1. Validate that input_manifest.json and chunks.jsonl exist.
2. Verify row count and checksums if checksums.sha256 is present.
3. Confirm Ollama is available.
4. Pull or verify the primary model:
   - qwen3-embedding:8b
5. Generate embeddings for every chunk with the primary model.
6. Write separate outputs:
   - embeddings.qwen3-8b.jsonl
   - embeddings.qwen3-4b.jsonl if fallback was needed
   - run_report.json
   - run_report.md
   - checksums.output.sha256
7. Include in run_report.json:
   - status
   - model names
   - observed embedding dimensions
   - chunk count
   - embedded row count per model
   - failed row count per model
   - elapsed time per model
   - peak disk note if observable
   - any model/runtime errors
8. If 8B fails, keep any successful fallback output and mark failed model clearly.
9. After outputs are complete, do not import them anywhere. The local operator will fetch and review.

Success condition:
- The output directory contains embeddings for qwen3-embedding:8b or a clearly marked qwen3-embedding:4b fallback, a machine-readable run_report.json, and checksums.
"""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: JsonObject) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json_object(path: Path) -> JsonObject:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object at {path}")
    return payload


def read_jsonl_objects(path: Path) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError(f"expected JSON object at {path}:{line_number}")
        rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: list[JsonObject]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def write_checksums(root: Path, filenames: list[str], output_name: str = "checksums.sha256") -> Path:
    lines = [f"{sha256_file(root / filename)}  {filename}" for filename in filenames]
    path = root / output_name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


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
