# Add Eval Center wait for source quality freshness

## Goal

- Prevent the daily Eval Center automation from writing a stale source-quality warning when source-quality is still finishing for the same local day.

## Scope

- Add a read-only wait/poll option to `scripts/run_daily_eval_center.py`.
- Pass the wait option from repo-side and installed launchd helpers.
- Document the operator behavior and add regression coverage.

## Non-scope

- Do not make Eval Center run source-quality.
- Do not merge the source-quality and Eval Center pipelines.
- Do not change answer-loop execution or Failure Bank sync behavior.

## Done Condition

- Scheduled Eval Center runs can wait for today's source-quality observation before writing `eval_center_latest.*`.
- If source-quality never becomes fresh before the timeout, Eval Center still writes the existing warning rather than blocking forever.

## Planned Files

- `scripts/run_daily_eval_center.py`
- `scripts/run_daily_eval_center.sh`
- `~/.khub/bin/run_daily_eval_center_launchd.sh`
- `tests/test_daily_eval_center_runner.py`
- `docs/eval_center_daily_automation.md`
- `CHANGELOG.md`
- `docs/PROJECT_STATE.md`

## Verification Plan

- `pytest tests/test_daily_eval_center_runner.py`
- `python -m py_compile scripts/run_daily_eval_center.py`
- Manual JSON run against current local artifacts
