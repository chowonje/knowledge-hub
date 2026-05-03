# Add Failure Bank v0 JSONL store

## Goal

- Add Failure Bank v0 as an explicit JSONL store for answer-loop failure cards.

## Scope

- Import answer-loop `answer_loop_failure_cards.jsonl` into stable Failure Bank records.
- Add `khub labs eval failure-bank sync` and `khub labs eval failure-bank list`.
- Surface Failure Bank presence and record counts in Eval Center.
- Keep EvalCase promotion and autofix out of scope.

## Non-scope

- Automatic root-cause classification.
- Automatic EvalCase creation.
- Automatic repair/autofix.
- Daily docs/status writeback.

## Done Condition

- Failure cards can be synced into `~/.khub/eval/knowledgeos/failures/failure_bank.jsonl`.
- The store dedupes repeated imports by stable `failureId`.
- The list command returns schema-valid JSON.
- Eval Center no longer reports `failure_bank` as a missing gap after sync.

## Planned Files

- `knowledge_hub/application/failure_bank.py`
- `knowledge_hub/interfaces/cli/commands/eval_cmd.py`
- `knowledge_hub/application/eval_center.py`
- `docs/schemas/failure-bank-sync-result.v1.json`
- `docs/schemas/failure-bank-list-result.v1.json`
- `knowledge_hub/core/schema_validator.py`
- `tests/test_failure_bank.py`
- `tests/test_eval_cmd.py`
- docs and project records

## Verification Plan

- `pytest tests/test_failure_bank.py tests/test_eval_center.py tests/test_daily_eval_center_runner.py tests/test_eval_cmd.py`
- `python -m py_compile knowledge_hub/application/failure_bank.py knowledge_hub/application/eval_center.py knowledge_hub/interfaces/cli/commands/eval_cmd.py scripts/run_daily_eval_center.py`
- `khub labs eval failure-bank sync --runs-root ~/.khub/eval/knowledgeos/runs --json`
- `khub labs eval failure-bank list --json`

## Outcome

- Synced 5 answer-loop failure cards into `~/.khub/eval/knowledgeos/failures/failure_bank.jsonl`.
- Eval Center latest now shows `Failure Bank` as `ok` with 5 open records.
