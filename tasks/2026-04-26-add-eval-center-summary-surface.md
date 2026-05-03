# Add eval center summary surface

## Goal

- Add the first recommended Eval Center slice as a read-only operator summary under the existing `khub labs eval` surface.

## Scope

- Add a schema-backed `khub labs eval center` command.
- Keep summary logic in `knowledge_hub.application.eval_center`.
- Inventory source-quality reports, answer-loop artifacts, query CSVs, report paths, coverage tests, and known gaps.
- Update CLI docs, project state, changelog, and focused regression tests.

## Non-scope

- No database migration for eval cases.
- No Failure Bank store implementation.
- No new top-level CLI command.
- No collector, model, retrieval, ranking, or answer-loop execution changes.

## Done Condition

- `khub labs eval center --json` emits a valid `knowledge-hub.eval-center.summary.result.v1` payload.
- Missing artifacts are surfaced as warnings rather than hidden assumptions.
- Focused unit/CLI/help/boundary tests cover the new surface.
- Durable records describe the behavior and remaining gaps.

## Planned Files

- `knowledge_hub/application/eval_center.py`
- `knowledge_hub/interfaces/cli/commands/eval_cmd.py`
- `knowledge_hub/core/schema_validator.py`
- `docs/schemas/eval-center-summary-result.v1.json`
- `tests/test_eval_center.py`
- `tests/test_eval_cmd.py`
- `tests/test_interfaces_cli_main.py`
- `tests/test_architecture_boundaries.py`
- `docs/guides/cli-commands.md`
- `docs/PROJECT_STATE.md`
- `CHANGELOG.md`
- `worklog/2026-04-26.md`

## Verification Plan

- `pytest tests/test_eval_center.py tests/test_eval_cmd.py::test_eval_center_json_is_schema_valid tests/test_interfaces_cli_main.py::test_cli_labs_eval_help_exposes_eval_center tests/test_architecture_boundaries.py::test_eval_center_cli_delegates_to_application_module`
- `khub labs eval center --json`
- `git diff --check`
