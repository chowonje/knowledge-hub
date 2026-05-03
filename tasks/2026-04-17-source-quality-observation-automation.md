# Source quality observation automation

## Goal

- Automate the repetitive 1-week source-quality observation loop without changing gate semantics.

## Scope

- Add a read-only observation verdict/report above the existing trend and legacy-readiness reports.
- Wire the existing nightly workflow to refresh the latest trend/readiness/observation reports automatically.
- Publish a concise latest observation summary into the GitHub Actions summary and uploaded artifacts.

## Non-scope

- Changing hard-gate thresholds
- Promoting the nightly to a required PR gate
- Removing `rag_legacy_runtime.py`
- Runtime/search/answer behavior changes

## Done Condition

- A latest observation report exists under `eval/knowledgeos/runs/reports/`.
- The nightly workflow refreshes trend, readiness, and observation reports automatically.
- The nightly workflow publishes a short human-readable observation summary.
- Regression coverage protects the observation verdict logic.

## Planned Files

- `.github/workflows/source-quality-nightly.yml`
- `eval/knowledgeos/scripts/report_source_quality_observation.py`
- `tests/test_source_quality_observation_report.py`
- `CHANGELOG.md`
- `docs/PROJECT_STATE.md`
- `worklog/2026-04-17.md`

## Verification Plan

- `python -m py_compile eval/knowledgeos/scripts/report_source_quality_observation.py tests/test_source_quality_observation_report.py`
- `pytest tests/test_source_quality_observation_report.py tests/test_source_quality_trend_report.py tests/test_legacy_runtime_readiness_report.py tests/test_source_quality_battery.py -q`
- `python eval/knowledgeos/scripts/report_source_quality_trend.py --runs-root eval/knowledgeos/runs --limit 7`
- `python eval/knowledgeos/scripts/report_legacy_runtime_readiness.py --repo-root . --runs-root eval/knowledgeos/runs --limit 7`
- `python eval/knowledgeos/scripts/report_source_quality_observation.py --runs-root eval/knowledgeos/runs --required-runs 7`
