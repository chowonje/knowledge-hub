#!/bin/zsh
set -euo pipefail

REPO_DIR="/Users/won/Desktop/allinone/knowledge-hub"
PYTHON_BIN="/Users/won/.pyenv/versions/3.10.13/bin/python"
LOG_DIR="$HOME/.khub/logs"
LOCK_DIR="$HOME/.khub/locks/continuous_sync.lock"
T9_ROOT="/Volumes/T9"

mkdir -p "$LOG_DIR"

if [ ! -d "$T9_ROOT" ]; then
  print -r -- "$(date -Iseconds) T9 not mounted; skipping continuous-sync" >> "$LOG_DIR/continuous_sync.err.log"
  exit 0
fi

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  print -r -- "$(date -Iseconds) continuous-sync already running; skipping" >> "$LOG_DIR/continuous_sync.err.log"
  exit 0
fi
trap 'rmdir "$LOCK_DIR" >/dev/null 2>&1 || true' EXIT

cd "$REPO_DIR"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.pyenv/bin:$HOME/.pyenv/shims"

if [ -f "$REPO_DIR/.env" ]; then
  set -a
  source "$REPO_DIR/.env"
  set +a
fi

"$PYTHON_BIN" -m knowledge_hub.cli.main crawl continuous-sync \
  --per-source-limit 4 \
  --topic continuous-latest \
  --source-policy fixed \
  --profile safe \
  --index \
  --extract-concepts \
  --materialize \
  --apply \
  --allow-external \
  --llm-mode mini \
  --json >> "$LOG_DIR/continuous_sync.log" 2>> "$LOG_DIR/continuous_sync.err.log"
