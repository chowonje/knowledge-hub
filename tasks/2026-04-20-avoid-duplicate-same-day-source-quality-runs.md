# Avoid duplicate same-day source-quality runs

## Goal

- Prevent a manual early-day source-quality run from being counted again by the later scheduled launchd run on the same local date.

## Scope

- Add a same-local-date skip guard to the Python daily runner.
- Pass that guard through the repo shell wrapper and installed launchd helper.
- Leave durable records and verify the guard with tests plus a real same-day skip.

## Non-scope

- Change source-quality thresholds or promotion rules.
- Turn scheduled docs/worklog writeback back on.

## Done Condition

- Today's manual run can execute once.
- A second invocation on the same local date skips instead of creating another run.
- The scheduled wrapper uses the same guard by default.

## Planned Files

- `scripts/run_daily_source_quality.py`
- `scripts/run_daily_source_quality.sh`
- `tests/test_daily_source_quality_runner.py`
- `~/.khub/bin/run_daily_source_quality_launchd.sh`
- `CHANGELOG.md`
- `docs/PROJECT_STATE.md`
- `docs/source_quality_daily_automation.md`

## Verification Plan

- Run targeted pytest for the daily runner and adjacent report/writeback tests.
- Run today's daily command once with the skip guard enabled.
- Re-run the same command immediately and confirm it returns `skipped=true`.
