#!/bin/zsh
set -euo pipefail

REPO_DIR="${REPO_DIR:-<repo-root>}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="${LOG_DIR:-$HOME/.khub/logs}"
LOCK_DIR="${LOCK_DIR:-$HOME/.khub/locks/daily_eval_center.lock}"
WAIT_FOR_SOURCE_QUALITY_SECONDS="${WAIT_FOR_SOURCE_QUALITY_SECONDS:-1800}"
SOURCE_QUALITY_WAIT_POLL_SECONDS="${SOURCE_QUALITY_WAIT_POLL_SECONDS:-30}"

mkdir -p "$LOG_DIR"
mkdir -p "${LOCK_DIR:h}"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  print -r -- "$(date -Iseconds) daily-eval-center already running; skipping" >> "$LOG_DIR/daily_eval_center.err.log"
  exit 0
fi
trap 'rmdir "$LOCK_DIR" >/dev/null 2>&1 || true' EXIT

cd "$REPO_DIR"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.pyenv/bin:$HOME/.pyenv/shims"

if env_text=$(cat "$REPO_DIR/.env" 2>/dev/null); then
  while IFS= read -r line; do
    [[ -z "$line" || "$line" == \#* || "$line" != *=* ]] && continue
    key="${line%%=*}"
    value="${line#*=}"
    export "$key=$value"
  done <<< "$env_text"
fi

cmd=(
  "$PYTHON_BIN" "scripts/run_daily_eval_center.py"
  "--json"
  "--skip-if-local-date-already-covered"
  "--wait-for-today-source-quality-seconds" "$WAIT_FOR_SOURCE_QUALITY_SECONDS"
  "--source-quality-wait-poll-seconds" "$SOURCE_QUALITY_WAIT_POLL_SECONDS"
)
"${cmd[@]}" >> "$LOG_DIR/daily_eval_center.log" 2>> "$LOG_DIR/daily_eval_center.err.log"
