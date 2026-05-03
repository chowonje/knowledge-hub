# Harden daily eval freshness after source quality failure

## Goal

- Recover the 2026-04-30 daily source-quality failure and prevent Eval Center from reporting stale source-quality as healthy.

## Scope

- Rebuild the local derived BGE/Chroma vector index if needed.
- Add a daily source-quality freshness signal to Eval Center snapshots.
- Keep same-day repair reruns from being blocked by superseded same-day failed source-quality batteries.
- Keep vault ask-v2 empty-card cases inside ask-v2 diagnostics instead of falling back to legacy runtime.

## Non-scope

- Do not promote Eval Center into a new execution gate.
- Do not delete source-quality artifacts or historical failed run directories.
- Do not change default paper/web source-quality semantics.

## Done Condition

- Chroma opens successfully.
- Source-quality rerun produces a fresh same-day battery and hard gate `ok`.
- Eval Center latest snapshot reports `sourceQualityFreshnessStatus=fresh`.
- Targeted tests cover freshness, same-day dedupe, and vault no-card ask-v2 behavior.

## Planned Files

- `knowledge_hub/application/eval_center.py`
- `scripts/run_daily_eval_center.py`
- `knowledge_hub/ai/ask_v2.py`
- `eval/knowledgeos/scripts/report_source_quality_trend.py`
- `eval/knowledgeos/scripts/report_legacy_runtime_readiness.py`
- Targeted tests under `tests/`
- `CHANGELOG.md`
- `docs/PROJECT_STATE.md`

## Verification Plan

- `pytest tests/test_source_quality_trend_report.py tests/test_legacy_runtime_readiness_report.py tests/test_source_quality_observation_report.py tests/test_ask_v2_sources.py::test_generate_answer_vault_no_card_candidates_stays_ask_v2_no_result tests/test_eval_center.py tests/test_daily_eval_center_runner.py`
- Chroma open probe for `~/.khub/chroma_db_bge_m3`.
- `python scripts/run_daily_source_quality.py --repo-root /Users/won/Desktop/allinone/knowledge-hub --runs-root /Users/won/.khub/eval/knowledgeos/runs --local-timezone Asia/Seoul --enforce-hard-gate --json`
- `python scripts/run_daily_eval_center.py --repo-root /Users/won/Desktop/allinone/knowledge-hub --runs-root /Users/won/.khub/eval/knowledgeos/runs --queries-dir eval/knowledgeos/queries --local-timezone Asia/Seoul --json`
