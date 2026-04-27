# Source Quality Daily Status

Date: 2026-04-22

## Objective

Run the daily source-quality observation loop and confirm whether the promoted base hard gate still holds while the newer detail-quality gate continues observation.

## Result

- Daily run completed after repairing the derived local vector index.
- Latest completed run: `~/.khub/eval/knowledgeos/runs/source_quality_battery_20260422_044630`
- Base observation decision: `ready_for_hard_gate_review`
- Hard gate status: `ok`
- Detail observation decision: `not_ready_for_detail_gate_review`

## Incident

The first daily attempt failed before evaluation collection because the active Chroma vector store could not initialize:

- error: `range start index 10 out of range for slice of length 9`
- affected derived store: `~/.khub/chroma_db_bge_m3`
- preserved copy: `~/.khub/chroma_db_bge_m3.pre_rebuild.20260422_134222`

The vector store was rebuilt with `khub index --all`.

Rebuild result:

- status: `ok`
- papers indexed: `387`
- concepts indexed: `66`
- vector documents: `453`
- report: `~/.khub/runs/index-idx_068fe710c5ca.json`

## Verification

- `python -m knowledge_hub.interfaces.cli.main search "alpha retrieval" --source vault --mode keyword --json`
  - runtime diagnostics returned `status=ok`
- `python scripts/run_daily_source_quality.py --skip-if-local-date-already-covered --local-timezone Asia/Seoul --enforce-hard-gate --json`
  - completed successfully
  - `hardGate.status=ok`

## Current Metrics

- paper route correctness: `1.0`
- vault route correctness: `1.0`
- web route correctness: `1.0`
- vault stale citation rate: `0.0`
- paper citation correctness: `1.0`
- vault abstention correctness: `0.0`
- web recency violation: `0.0`

## Remaining Blockers

Detail-gate promotion is still blocked:

- `vault_vault_abstention_correctness_need_7_numeric_points_have_1`
- `vault_vault_abstention_correctness_not_stable`

Interpretation: the base operational hard gate is green, but the detail-quality gate needs more daily observations and the vault abstention behavior is currently failing the candidate threshold.
