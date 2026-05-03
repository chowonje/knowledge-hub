# Source quality week-1 observation

## Goal

- Run the first one-week observation loop for the new source-quality battery before any hard-gate promotion.
- Keep this as an operations/readiness task, not a runtime-refactor task.

## Scope

- Refresh the latest source-quality trend report from `source_quality_battery_*` runs.
- Record the current baseline for `paper`, `vault`, and `web`.
- Define the daily observation checklist for the next seven nightly runs.
- Define the decision rubric for whether `route_correctness` and `vault_stale_citation_rate` are ready for hard-gate promotion.

## Non-scope

- No runtime behavior changes.
- No classifier or `memory_route_mode` contract changes.
- No shim deletion, property migration, or additional seam hardening.
- No PR-required gate promotion during this task.

## Current baseline

- latest trend report refreshed via:
  - `python eval/knowledgeos/scripts/report_source_quality_trend.py --runs-root eval/knowledgeos/runs --limit 7`
- latest run:
  - `eval/knowledgeos/runs/source_quality_battery_20260417_051744/`
- current trend summary:
  - paper `route_correctness = 1.0`
  - paper `citation_correctness = 1.0`
  - vault `route_correctness = 1.0`
  - vault `stale_citation_rate = 0.0`
  - web `route_correctness = 1.0` with observed range `0.65 .. 1.0`
  - web `recency_violation = 0.0`

## Daily checklist

- Confirm the nightly workflow produced a fresh `source_quality_battery_*` run directory.
- Confirm the trend report refreshed:
  - `eval/knowledgeos/runs/reports/source_quality_trend_latest.json`
  - `eval/knowledgeos/runs/reports/source_quality_trend_latest.md`
- Record whether these hard metrics are stable:
  - paper `route_correctness`
  - vault `route_correctness`
  - vault `stale_citation_rate`
  - web `route_correctness`
- Record whether these soft metrics moved materially:
  - paper citation correctness
  - vault abstention correctness
  - web recency violation

## Escalation rules

- Treat any non-zero vault `stale_citation_rate` as an immediate investigation.
- Treat any `web route_correctness` drop below `1.0` as a classification/eval-alignment investigation first, not an automatic runtime regression.
- Treat a paper or vault `route_correctness` drop as a higher-severity signal than a soft-metric wobble.

## Hard-gate promotion rubric

- Do not promote anything before one week of nightly artifacts exists.
- Candidate first hard gates:
  - per-source `route_correctness`
  - vault `stale_citation_rate`
- Promotion requires:
  - seven consecutive nightly runs present
  - no unexplained hard-metric regressions during the window
  - vault `stale_citation_rate == 0.0` throughout the window
  - `web route_correctness` stable enough that any earlier dip has a documented collector/routing explanation

## Verification

- `python eval/knowledgeos/scripts/report_source_quality_trend.py --runs-root eval/knowledgeos/runs --limit 7`

## Next

- After seven nightly runs, produce a short decision note: keep trend-only vs promote selected hard metrics.
