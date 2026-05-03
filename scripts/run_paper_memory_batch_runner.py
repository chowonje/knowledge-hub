#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log_line(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{_timestamp()}] {message}\n")


def _batch_done(artifact_dir: Path) -> bool:
    summary_path = artifact_dir / "paper_memory_extraction_summary.json"
    if not summary_path.exists():
        return False
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    section = payload.get("paperMemory") if isinstance(payload, dict) else None
    if isinstance(section, dict):
        return bool(section.get("requested"))
    return bool(payload.get("requested")) if isinstance(payload, dict) else False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paper-memory rebuild batches sequentially with resume-friendly logging.")
    parser.add_argument("--batch-dir", required=True, help="Directory containing batch_*.txt files")
    parser.add_argument("--artifact-root", required=True, help="Directory where per-batch artifacts/logs are written")
    parser.add_argument("--config", required=True, help="Config path for rebuild_memory_stores.py")
    parser.add_argument("--compare-summary", required=True, help="Previous audit summary JSON path")
    parser.add_argument("--model", default="exaone3.5:7.8b", help="Extractor model")
    parser.add_argument("--provider", default="ollama", help="Extractor provider")
    parser.add_argument("--sleep-seconds", type=float, default=2.0, help="Sleep between batches")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    rebuild_script = root / "scripts" / "rebuild_memory_stores.py"
    batch_dir = Path(args.batch_dir).expanduser()
    artifact_root = Path(args.artifact_root).expanduser()
    artifact_root.mkdir(parents=True, exist_ok=True)
    master_log = artifact_root / "master.log"
    runner_summary_path = artifact_root / "runner_summary.json"

    batch_files = sorted(batch_dir.glob("batch_*.txt"))
    if not batch_files:
        raise SystemExit(f"no batch files found in {batch_dir}")

    results: list[dict[str, str | int]] = []
    for batch_file in batch_files:
        batch_name = batch_file.stem
        artifact_dir = artifact_root / batch_name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        run_log = artifact_dir / "run.log"
        if _batch_done(artifact_dir):
            _log_line(master_log, f"SKIP {batch_name} already_complete")
            results.append({"batch": batch_name, "status": "skipped"})
            continue

        _log_line(master_log, f"START {batch_name}")
        cmd = [
            "python",
            str(rebuild_script),
            "--config",
            str(Path(args.config).expanduser()),
            "--targets",
            "paper-memory",
            "--paper-id-file",
            str(batch_file),
            "--paper-memory-extraction-mode",
            "schema",
            "--paper-memory-extractor-provider",
            str(args.provider),
            "--paper-memory-extractor-model",
            str(args.model),
            "--paper-memory-artifact-dir",
            str(artifact_dir),
            "--paper-memory-compare-to-summary",
            str(Path(args.compare_summary).expanduser()),
            "--json",
        ]
        with run_log.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(cmd, cwd=str(root), stdout=handle, stderr=subprocess.STDOUT, text=True)
        _log_line(master_log, f"END {batch_name} status={proc.returncode}")
        results.append({"batch": batch_name, "status": "ok" if proc.returncode == 0 else "failed", "code": proc.returncode})
        runner_summary_path.write_text(json.dumps({"batches": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        if proc.returncode != 0:
            return proc.returncode
        if args.sleep_seconds > 0:
            subprocess.run(["sleep", str(args.sleep_seconds)], check=False)

    runner_summary_path.write_text(json.dumps({"batches": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
