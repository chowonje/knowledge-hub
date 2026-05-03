# Add daily Eval Center snapshot automation

## Goal

- Add a once-per-day read-only Eval Center snapshot runner without turning Eval Center into a new execution gate.

## Scope

- `scripts/run_daily_eval_center.py`
- `scripts/run_daily_eval_center.sh`
- `ops/launchd/com.won.knowledge-hub.daily-eval-center.plist`
- docs and protecting tests for the daily snapshot behavior

## Non-scope

- running source-quality, answer-loop, or autofix from the Eval Center runner
- introducing a new hard gate on top of Eval Center warnings
- adding Failure Bank or EvalCase storage in this tranche

## Done Condition

- a daily runner writes dated Eval Center JSON/Markdown snapshots and latest aliases
- duplicate same-day runs can skip by local date
- the runner is covered by targeted tests and documented for local automation

## Planned Files

- `scripts/run_daily_eval_center.py`
- `scripts/run_daily_eval_center.sh`
- `ops/launchd/com.won.knowledge-hub.daily-eval-center.plist`
- `docs/eval_center_daily_automation.md`
- `tests/test_daily_eval_center_runner.py`
- `CHANGELOG.md`
- `docs/PROJECT_STATE.md`

## Verification Plan

- `pytest tests/test_eval_center.py tests/test_daily_eval_center_runner.py`
- `python scripts/run_daily_eval_center.py --repo-root <repo-root> --runs-root ~/.khub/eval/knowledgeos/runs --queries-dir eval/knowledgeos/queries --skip-if-local-date-already-covered --local-timezone Asia/Seoul --json`
