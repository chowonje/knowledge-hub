#!/usr/bin/env python3
"""Check whether the repo is safe and clean enough for a public snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from knowledge_hub.application.public_release_hygiene import check_public_release_hygiene


def main() -> int:
    parser = argparse.ArgumentParser(description="Check public-release hygiene for the current repo.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]), help="repository root to scan")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    payload = check_public_release_hygiene(Path(args.repo_root))
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("[public release hygiene]")
        print(f"status: {payload['status']}")
        print(f"tracked files: {payload['trackedFileCount']}")
        print(f"issues: {payload['issueCount']}")
        for kind, count in sorted((payload.get("issueCountsByKind") or {}).items()):
            print(f"- {kind}: {count}")
        for item in list(payload.get("issues") or [])[:20]:
            match = f" match={item.get('match')}" if item.get("match") else ""
            print(f"  * {item.get('path')}: {item.get('detail')}{match}")
        for item in list(payload.get("nextActions") or []):
            print(f"next: {item}")
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
