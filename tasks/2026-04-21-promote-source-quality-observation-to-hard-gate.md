# Promote source quality observation to hard gate

## Goal

- Promote the source-quality observation verdict from passive reporting into a failing operating gate for the local daily loop.

## Scope

- Add a schema-backed hard-gate checker over the latest source-quality observation report.
- Wire the checker into the daily runner and local launchd wrapper.
- Update the operating docs, project state, changelog, and focused tests.

## Non-scope

- Do not add source-quality to required PR CI yet; remote CI needs a separate persisted run-history decision.
- Do not change the source-quality collectors or route/eval semantics.
- Do not enable scheduled docs writeback.

## Done Condition

- `python scripts/run_daily_source_quality.py --skip-if-local-date-already-covered --enforce-hard-gate --json` exits 0 on the current 7-run ready report.
- The checker exits non-zero on not-ready/blocker/metric-drift payloads.
- Project records state that source-quality is now a local daily hard gate.

## Planned Files

- `eval/knowledgeos/scripts/check_source_quality_hard_gate.py`
- `scripts/run_daily_source_quality.py`
- `scripts/run_daily_source_quality.sh`
- `~/.khub/bin/run_daily_source_quality_launchd.sh`
- `tests/test_source_quality_hard_gate.py`
- `tests/test_daily_source_quality_runner.py`
- `docs/source_quality_daily_automation.md`
- `eval/knowledgeos/README.md`
- `docs/status/2026-04-18-observation-action-rules.md`
- `CHANGELOG.md`
- `docs/PROJECT_STATE.md`

## Verification Plan

- `pytest tests/test_source_quality_hard_gate.py tests/test_daily_source_quality_runner.py tests/test_source_quality_observation_report.py tests/test_source_quality_trend_report.py tests/test_legacy_runtime_readiness_report.py -q`
- `python eval/knowledgeos/scripts/check_source_quality_hard_gate.py --runs-root eval/knowledgeos/runs --json`
- `python scripts/run_daily_source_quality.py --skip-if-local-date-already-covered --local-timezone Asia/Seoul --enforce-hard-gate --json`

## Outcome

- completed: source-quality is now a local daily operating hard gate.
- verified: focused pytest passed (`26 passed`), the live hard-gate checker returned `status=ok`, the same-day daily runner skipped duplicate execution while still returning `hardGate.status=ok`, and launchd kickstart ended with `last exit code = 0`.
- residual risk: remote CI is not yet a source-quality hard gate because it does not currently restore a persistent seven-run history.
