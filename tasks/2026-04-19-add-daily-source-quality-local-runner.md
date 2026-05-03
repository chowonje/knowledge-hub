# Add daily source-quality local runner

## Goal

- Add a local daily runner that refreshes source-quality observation artifacts and can leave the result in `docs/status/` plus `worklog/`.

## Scope

- Add a repo-local Python runner for the daily source-quality sequence.
- Add a launchd-friendly shell wrapper and plist template.
- Add a short operator doc and targeted regression test.
- Update product records for the new local workflow.

## Non-scope

- Change source-quality thresholds or hard-gate semantics.
- Broaden the existing docs-only writeback consumer beyond `docs/status/` and `worklog/`.
- Add a new product CLI or MCP surface for this workflow.

## Done Condition

- One command can rerun battery, trend, readiness, and observation locally.
- The summary exposes `decision`, `blockers`, per-source `route_correctness`, and `vault stale_citation_rate`.
- The runner can optionally reuse the existing docs/status + worklog writeback loop.
- The repo has a launchd template, guide, test, and durable records.

## Planned Files

- `scripts/run_daily_source_quality.py`
- `scripts/run_daily_source_quality.sh`
- `ops/launchd/com.won.knowledge-hub.daily-source-quality.plist`
- `docs/source_quality_daily_automation.md`
- `tests/test_daily_source_quality_runner.py`
- `CHANGELOG.md`
- `docs/PROJECT_STATE.md`

## Verification Plan

- Run targeted pytest for the new runner plus adjacent source-quality and writeback script tests.
- Run the daily runner once in local JSON mode to confirm the end-to-end command chain and summary output.
- Verify the launchd install path by loading the agent, kickstarting it, and checking `launchctl print ... last exit code`.
