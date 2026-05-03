#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from knowledge_hub.application.context import AppContextFactory
from knowledge_hub.application.paper_source_repairs import run_source_cleanup_queue
from knowledge_hub.papers.source_cleanup import load_source_cleanup_queue


def _load_pass_b_ids(path_value: str | None) -> list[str]:
    token = str(path_value or "").strip()
    if not token:
        return []
    path = Path(token).expanduser()
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve known paper source contamination cases before Pass B rebuilds.")
    parser.add_argument("--config", default=None, help="Optional config path")
    parser.add_argument("--cleanup-queue", required=True, help="paper_memory_source_cleanup_queue.csv path")
    parser.add_argument("--pass-b-id-file", default=None, help="Optional pass-B id file to filter after cleanup")
    parser.add_argument("--artifact-dir", required=True, help="Directory to write cleanup artifacts")
    parser.add_argument("--apply", action="store_true", help="Apply resolved relinks to the sqlite papers store")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Emit JSON payload")
    args = parser.parse_args()

    factory = AppContextFactory(config_path=args.config)
    sqlite_db = factory.get_sqlite_db()
    queue_rows = load_source_cleanup_queue(args.cleanup_queue)
    payload = run_source_cleanup_queue(
        sqlite_db=sqlite_db,
        queue_rows=queue_rows,
        artifact_dir=args.artifact_dir,
        pass_b_ids=_load_pass_b_ids(args.pass_b_id_file),
        apply=bool(args.apply),
    )
    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print("paper source cleanup")
    print(f"- decisions={len(decisions)} applied={apply_summary['applied']} skipped={apply_summary['skipped']}")
    for key, value in artifact_paths.items():
        print(f"- {key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
