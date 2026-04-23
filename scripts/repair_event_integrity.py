#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from knowledge_hub.infrastructure.persistence import SQLiteDatabase


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repair ontology entity/event drift by backfilling missing entity_created events."
    )
    parser.add_argument(
        "--db-path",
        default=str(Path.home() / ".khub" / "knowledge.db"),
        help="SQLite DB path (default: ~/.khub/knowledge.db)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum orphan entities to repair (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="Report orphan entities without writing repair events")
    parser.add_argument("--json", action="store_true", help="Print JSON payload")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    db_path = Path(args.db_path).expanduser()
    db = SQLiteDatabase(str(db_path))
    if not db.event_store:
        raise SystemExit("event store is unavailable for this database")

    before_count = db.event_store.count_entities_without_events()
    before_sample = db.event_store.list_entities_without_events(limit=20)
    payload = {
        "dbPath": str(db_path),
        "beforeCount": before_count,
        "beforeSample": [item["entity_id"] for item in before_sample],
        "dryRun": bool(args.dry_run),
    }
    if args.dry_run:
        payload["repairedCount"] = 0
        payload["remainingCount"] = before_count
    else:
        result = db.event_store.repair_missing_entity_events(
            limit=(args.limit or None),
            run_id="repair_event_integrity",
        )
        payload["repairedCount"] = int(result.get("repaired_count") or 0)
        payload["repairedIds"] = list(result.get("entity_ids") or [])
        payload["remainingCount"] = int(result.get("remaining_count") or 0)
    db.close()

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(
            f"repair-event-integrity before={payload['beforeCount']} "
            f"repaired={payload.get('repairedCount', 0)} remaining={payload['remainingCount']}"
        )
        for entity_id in payload.get("beforeSample", [])[:10]:
            print(f"- {entity_id}")
    return 0 if int(payload.get("remainingCount") or 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
