from __future__ import annotations

from pathlib import Path
from typing import Final

from knowledge_hub.application.paper_query_export import DEFAULT_QUERY_MODEL, QUERY_EXPORT_SCHEMA
from knowledge_hub.application.paper_query_run import query_embedding_output_path
from knowledge_hub.application.paper_query_remote_worker import build_query_remote_worker
from knowledge_hub.application.remote_index_contract import (
    JsonObject,
    read_json_object,
    read_jsonl_objects,
    verify_checksums,
    write_json,
)
from knowledge_hub.application.remote_query_embedding import EXPECTED_QUERY_DIMS

QUERY_REMOTE_PLAN_SCHEMA: Final = "knowledge-hub.labs.paper-harness.query-remote-plan.v1"
WORKER_SCRIPT_NAME: Final = "run_query_embeddings.py"
REMOTE_PLAN_NAME: Final = "remote_operator_plan.json"
WORKER_LOG_NAME: Final = "run_worker.log"


def _safe_token(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value).strip("_")


def _remote_root_value(remote_root: str) -> str:
    value = remote_root.rstrip("/")
    local_home = str(Path.home())
    if value == local_home:
        return "~"
    local_home_prefix = f"{local_home}/"
    if value.startswith(local_home_prefix):
        return f"~/{value.removeprefix(local_home_prefix)}"
    return value


def _operator_commands(
    *,
    run_dir: Path,
    remote_host: str,
    remote_root: str,
    remote_run_dir: str,
    session: str,
    output_file: str,
) -> JsonObject:
    local_run = str(run_dir)
    return {
        "uploadBundle": f"ssh {remote_host} 'mkdir -p {remote_root}' && scp -r {local_run} {remote_host}:{remote_root}/",
        "startWorker": (
            f"ssh {remote_host} 'tmux new -d -s {session} "
            f"\"cd {remote_run_dir} && python3 {WORKER_SCRIPT_NAME} 2>&1 | tee {WORKER_LOG_NAME}\"'"
        ),
        "monitor": (
            f"ssh {remote_host} 'tmux capture-pane -pt {session} -S -80; "
            f"wc -l {remote_run_dir}/{output_file}.tmp {remote_run_dir}/{output_file} 2>/dev/null || true; "
            f"df -h {remote_run_dir}'"
        ),
        "fetchOutputs": (
            f"scp {remote_host}:{remote_run_dir}/{output_file} "
            f"{remote_host}:{remote_run_dir}/run_report.json "
            f"{remote_host}:{remote_run_dir}/run_report.md "
            f"{remote_host}:{remote_run_dir}/checksums.output.sha256 "
            f"{remote_host}:{remote_run_dir}/{WORKER_LOG_NAME} {local_run}/"
        ),
        "validateLocal": (
            "python -m knowledge_hub.interfaces.cli.main labs paper-harness query-validate "
            f"--run {local_run} --json"
        ),
    }


def write_query_remote_plan(
    *,
    run_dir: Path,
    model: str = DEFAULT_QUERY_MODEL,
    remote_host: str = "oracle-hermes",
    remote_root: str = "~/knowledgeos-remote-indexing",
    session: str = "",
) -> JsonObject:
    run_path = run_dir.expanduser().resolve()
    output_path = query_embedding_output_path(run_path, model)
    blockers: list[str] = []
    query_rows: list[JsonObject] = []
    expected_dim = EXPECTED_QUERY_DIMS.get(model)
    if expected_dim is None:
        blockers.append("unsupported_query_embedding_model")
    if not (run_path / "input_manifest.json").exists():
        blockers.append("input_manifest_missing")
    if not (run_path / "queries.jsonl").exists():
        blockers.append("queries_missing")
    if not blockers:
        manifest = read_json_object(run_path / "input_manifest.json")
        if str(manifest.get("schema") or "") != QUERY_EXPORT_SCHEMA:
            blockers.append("input_manifest_schema_mismatch")
        if str(manifest.get("targetModel") or "") != model:
            blockers.append("input_manifest_model_mismatch")
        query_rows = read_jsonl_objects(run_path / "queries.jsonl")
        if not query_rows:
            blockers.append("query_required")
        blockers.extend(verify_checksums(run_path))

    remote_root_value = _remote_root_value(remote_root)
    session_name = _safe_token(session or f"qwen8_query_{run_path.name}") or "qwen8_query_embed"
    remote_run_dir = f"{remote_root_value}/{run_path.name}"
    unique_blockers = sorted(set(blockers))
    payload: JsonObject = {
        "schema": QUERY_REMOTE_PLAN_SCHEMA,
        "status": "blocked" if unique_blockers else "ready",
        "blockers": unique_blockers,
        "runDir": str(run_path),
        "queryCount": len(query_rows),
        "targetModel": model,
        "expectedEmbeddingDim": int(expected_dim or 0),
        "workerScriptPath": str(run_path / WORKER_SCRIPT_NAME),
        "planArtifactPath": str(run_path / REMOTE_PLAN_NAME),
        "expectedOutputFiles": [output_path.name, "run_report.json", "run_report.md", "checksums.output.sha256"],
        "remote": {
            "host": remote_host,
            "root": remote_root_value,
            "runDir": remote_run_dir,
            "session": session_name,
        },
        "operatorCommands": {},
        "canonicalMutationAllowed": False,
    }
    if unique_blockers:
        return payload

    (run_path / WORKER_SCRIPT_NAME).write_text(
        build_query_remote_worker(model=model, output_file=output_path.name),
        encoding="utf-8",
    )
    payload["operatorCommands"] = _operator_commands(
        run_dir=run_path,
        remote_host=remote_host,
        remote_root=remote_root_value,
        remote_run_dir=remote_run_dir,
        session=session_name,
        output_file=output_path.name,
    )
    write_json(run_path / REMOTE_PLAN_NAME, payload)
    return payload


__all__ = ["QUERY_REMOTE_PLAN_SCHEMA", "write_query_remote_plan"]
