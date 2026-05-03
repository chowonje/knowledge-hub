# Add operator-readable Eval Center daily brief

## Goal

- Make the daily Eval Center snapshot readable as an operator brief with part-level status, findings, and next actions.

## Scope

- Add structured `operatorBrief` data to the Eval Center summary payload.
- Render `eval_center_latest.md` as a human-facing daily brief.
- Keep the automation read-only and avoid docs/status or vault writeback.

## Non-scope

- Running answer-loop or source-quality from Eval Center.
- Failure Bank or EvalCase persistence.
- Automatic docs/status or worklog apply.

## Done Condition

- Latest Eval Center JSON includes part-level brief data.
- Latest Eval Center Markdown shows `Part Status` and `Findings`.
- Targeted eval center tests pass.

## Planned Files

- `knowledge_hub/application/eval_center.py`
- `scripts/run_daily_eval_center.py`
- `docs/schemas/eval-center-summary-result.v1.json`
- `tests/test_eval_center.py`
- `tests/test_daily_eval_center_runner.py`
- `docs/eval_center_daily_automation.md`
- `CHANGELOG.md`
- `docs/PROJECT_STATE.md`

## Verification Plan

- `pytest tests/test_eval_center.py tests/test_daily_eval_center_runner.py`
- `python -m py_compile knowledge_hub/application/eval_center.py scripts/run_daily_eval_center.py`
- `python scripts/run_daily_eval_center.py --json` against the local eval run root

## Outcome

- Added `operatorBrief.sections` and `operatorBrief.findings` to the Eval Center summary payload.
- Updated daily Markdown latest output to show `Part Status`, `Findings`, gaps, recommendations, and warnings.
- Added schema validation before latest artifact writes and made duplicate-day skip ignore malformed latest JSON.
- Added artifact `modifiedAt` visibility so stale answer-loop artifacts are visible in the operator brief.
- Verified the installed launchd job still exits `0` after the changes.
