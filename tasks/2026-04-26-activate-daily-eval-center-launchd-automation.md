# Activate daily Eval Center launchd automation

## Goal

- Activate the daily Eval Center snapshot automation under launchd with the same stable helper-path contract used by daily source-quality.

## Scope

- Patch the Eval Center launchd template/docs to target an installed helper under `~/.khub/bin/`.
- Fix the date-dependent duplicate-day regression test so the skip guard remains stable after `2026-04-26`.
- Install the helper and LaunchAgent locally, then verify one forced run.

## Non-scope

- Turning Eval Center into a new gate or runner for source-quality / answer-loop.
- Failure Bank, EvalCase promotion, or autofix integration.

## Done Condition

- `launchctl print gui/$(id -u)/com.won.knowledge-hub.daily-eval-center` shows the installed helper path and exit code `0`.
- A forced run finishes as either a new snapshot or an expected same-day skip.
- Targeted tests and runner compile checks pass.

## Planned Files

- `scripts/run_daily_eval_center.py`
- `tests/test_daily_eval_center_runner.py`
- `ops/launchd/com.won.knowledge-hub.daily-eval-center.plist`
- `docs/eval_center_daily_automation.md`
- `CHANGELOG.md`
- `docs/PROJECT_STATE.md`
- installed helper: `~/.khub/bin/run_daily_eval_center_launchd.sh`
- installed LaunchAgent: `~/Library/LaunchAgents/com.won.knowledge-hub.daily-eval-center.plist`

## Verification Plan

- `pytest tests/test_eval_center.py tests/test_daily_eval_center_runner.py`
- `python -m py_compile scripts/run_daily_eval_center.py`
- `zsh -n scripts/run_daily_eval_center.sh`
- `zsh -n ~/.khub/bin/run_daily_eval_center_launchd.sh`
- `launchctl print gui/$(id -u)/com.won.knowledge-hub.daily-eval-center`
