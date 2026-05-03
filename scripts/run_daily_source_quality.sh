#!/bin/zsh
set -euo pipefail

REPO_DIR="${REPO_DIR:-<repo-root>}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="${LOG_DIR:-$HOME/.khub/logs}"
LOCK_DIR="${LOCK_DIR:-$HOME/.khub/locks/daily_source_quality.lock}"
ACTOR="${ACTOR:-won}"
WRITEBACK="${WRITEBACK:-0}"
APPLY_WRITEBACK="${APPLY_WRITEBACK:-0}"
INCLUDE_WORKSPACE="${INCLUDE_WORKSPACE:-1}"
ENFORCE_HARD_GATE="${ENFORCE_HARD_GATE:-1}"

mkdir -p "$LOG_DIR"
mkdir -p "${LOCK_DIR:h}"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  print -r -- "$(date -Iseconds) daily-source-quality already running; skipping" >> "$LOG_DIR/daily_source_quality.err.log"
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

cmd=("$PYTHON_BIN" "scripts/run_daily_source_quality.py" "--json" "--skip-if-local-date-already-covered")
if [[ "$ENFORCE_HARD_GATE" == "1" ]]; then
  cmd+=("--enforce-hard-gate")
fi
if [[ "$WRITEBACK" == "1" ]]; then
  cmd+=("--writeback" "--actor" "$ACTOR")
  if [[ "$INCLUDE_WORKSPACE" == "1" ]]; then
    cmd+=("--include-workspace")
  else
    cmd+=("--no-include-workspace")
  fi
  if [[ "$APPLY_WRITEBACK" == "1" ]]; then
    cmd+=("--apply-writeback")
  fi
fi

"${cmd[@]}" >> "$LOG_DIR/daily_source_quality.log" 2>> "$LOG_DIR/daily_source_quality.err.log"
