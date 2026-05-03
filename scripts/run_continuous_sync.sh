#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPO_DIR="${KHUB_REPO_DIR:-${SCRIPT_DIR:h}}"
PYTHON_BIN="${KHUB_PYTHON_BIN:-python}"
LOG_DIR="${KHUB_LOG_DIR:-$HOME/.khub/logs}"
LOCK_DIR="${KHUB_LOCK_DIR:-$HOME/.khub/locks/continuous_sync.lock}"
T9_ROOT="${KHUB_CONTINUOUS_SYNC_ROOT:-}"
PROFILE="${KHUB_CONTINUOUS_SYNC_PROFILE:-safe}"
PER_SOURCE_LIMIT="${KHUB_CONTINUOUS_SYNC_PER_SOURCE_LIMIT:-4}"
TOPIC="${KHUB_CONTINUOUS_SYNC_TOPIC:-continuous-latest}"
LLM_MODE="${KHUB_CONTINUOUS_SYNC_LLM_MODE:-mini}"

mkdir -p "$LOG_DIR"
mkdir -p "${LOCK_DIR:h}"

if [ -z "$T9_ROOT" ] || [ ! -d "$T9_ROOT" ]; then
  print -r -- "$(date -Iseconds) KHUB_CONTINUOUS_SYNC_ROOT is unset or unavailable; skipping continuous-sync" >> "$LOG_DIR/continuous_sync.err.log"
  exit 0
fi

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  print -r -- "$(date -Iseconds) continuous-sync already running; skipping" >> "$LOG_DIR/continuous_sync.err.log"
  exit 0
fi
trap 'rmdir "$LOCK_DIR" >/dev/null 2>&1 || true' EXIT

cd "$REPO_DIR"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.pyenv/bin:$HOME/.pyenv/shims"

if [ "${KHUB_CONTINUOUS_SYNC_LOAD_ENV:-0}" = "1" ] && [ -f "$REPO_DIR/.env" ]; then
  set -a
  source "$REPO_DIR/.env"
  set +a
fi

args=(
  -m knowledge_hub.interfaces.cli.main crawl continuous-sync
  --per-source-limit "$PER_SOURCE_LIMIT"
  --topic "$TOPIC"
  --source-policy fixed
  --profile "$PROFILE"
  --index
  --extract-concepts
  --materialize
  --llm-mode "$LLM_MODE"
  --json
)

if [ "${KHUB_CONTINUOUS_SYNC_APPLY:-0}" = "1" ]; then
  args+=(--apply)
fi

if [ "${KHUB_CONTINUOUS_SYNC_ALLOW_EXTERNAL:-0}" = "1" ]; then
  args+=(--allow-external)
fi

"$PYTHON_BIN" "${args[@]}" >> "$LOG_DIR/continuous_sync.log" 2>> "$LOG_DIR/continuous_sync.err.log"
