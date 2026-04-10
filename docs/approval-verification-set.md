# Approval Verification Set

## Goal

- Freeze the smallest credible release/merge gate for the current capture read-model + processor + orphan-operator-loop + OS bridge + authority subset.
- Add a separate, faster local CLI smoke gate for release-trust checks when the full approval set is too wide for a quick sanity pass.

## Release Smoke Gate

```bash
cd /Users/won/Desktop/allinone/knowledge-hub && python scripts/check_release_smoke.py
```

Covers:
- `khub --help`
- `khub setup --quick --non-interactive` in an isolated temp `HOME`
- `khub dinger capture --help`
- `khub --config <temp>/.khub/config.yaml status`
- `khub --config <temp>/.khub/config.yaml doctor --json`
- invalid command exit behavior (`khub definitely-missing`)

Pass contract:
- hosted providers are not required
- missing Ollama is acceptable when the live CLI still returns the intended local contract (`doctor.status in ok|blocked|degraded|needs_setup`)
- empty vector corpus is acceptable when it is surfaced as `needs_setup` or another non-fatal local readiness state
- help surfaces must still expose the expected top-level and nested operator commands
- invalid commands must exit non-zero and still surface a user-facing command error
- the gate fails only when the CLI surface breaks: command exit failure, missing runtime table, invalid doctor JSON/schema, or unexpected readiness status such as `failed`

Notes:
- this is a narrow release-trust smoke gate, not a replacement for the frozen approval verification set below
- prefer this command for fast local sanity checks before spending time on the broader targeted approval-set reruns
- exit code contract: `0` on pass, `1` on smoke failure

## Approval Verification Set

1. Capture read-model + processor + `requeue` CLI

```bash
cd /Users/won/Desktop/allinone/knowledge-hub && python -m pytest -q \
  tests/test_dinger_cmd.py::test_dinger_capture_list_exposes_operator_statuses \
  tests/test_dinger_cmd.py::test_dinger_capture_show_returns_packet_and_runtime_artifacts \
  tests/test_dinger_cmd.py::test_dinger_capture_retry_reprocesses_failed_packet \
  tests/test_dinger_cmd.py::test_dinger_capture_retry_surfaces_processor_failure \
  tests/test_dinger_cmd.py::test_dinger_capture_process_advances_packet_to_filed_with_runtime_artifacts \
  tests/test_dinger_cmd.py::test_dinger_capture_process_links_to_os_idempotently \
  tests/test_dinger_cmd.py::test_dinger_capture_requeue_restores_orphan_packet_and_reenables_retry \
  tests/test_dinger_cmd.py::test_dinger_capture_requeue_fails_for_legacy_orphan_without_snapshot \
  tests/test_dinger_cmd.py::test_dinger_capture_requeue_reports_already_present_packet_stably \
  tests/test_dinger_capture_processor.py
```

Covers:
- operator `list/show/retry/requeue` behavior
- snapshot-backed orphan packet restore and stable `already_present` no-op handling
- temp-runtime `capture-process` filed/link paths and processor safety

2. Capture cleanup operator loop

```bash
cd /Users/won/Desktop/allinone/knowledge-hub && python -m pytest -q \
  tests/test_dinger_capture_cleanup.py
```

Covers:
- dry-run cleanup classification
- apply/delete semantics for stale runtime artifacts and stale claim files
- recoverable orphan keep-for-requeue behavior
- queue-backed runtime safety

Note:
- the helper suite is the blocker gate because it owns the actual cleanup policy contract: what is deletable, what is protected, and what `delete|keep|errors|counts` mean
- the CLI wrapper now also has dedicated targeted regressions in `tests/test_dinger_cmd.py`, but those stay as supporting evidence rather than the frozen blocker because the wrapper is a thin adapter over the helper contract
- This does not mean operators must run cleanup before close-out; it means the cleanup policy contract is frozen through the targeted test surface rather than through a raw `--help` check.

3. Filed Dinger -> OS bridge gate

```bash
cd /Users/won/Desktop/allinone/knowledge-hub && python -m pytest -q \
  tests/test_os_cmd.py::test_os_capture_bridges_dinger_file_results_to_os \
  tests/test_os_cmd.py::test_os_capture_reuses_existing_open_dinger_bridge_item_even_with_summary_drift \
  tests/test_os_cmd.py::test_os_capture_creates_new_item_when_same_dedupe_key_only_matches_resolved_item \
  tests/test_os_cmd.py::test_os_project_evidence_derives_dinger_candidates
```

Covers:
- filed Dinger result -> OS inbox bridge
- open-item reuse vs resolved-item replay semantics
- project evidence candidate derivation from the same Dinger-backed source refs

4. Python authority contract gate

```bash
cd /Users/won/Desktop/allinone/knowledge-hub && python -m pytest -q \
  'tests/test_authority_contract.py::test_bridge_fixtures_validate_against_python_authority_schemas[os-capture-result.v1.fixture.json-knowledge-hub.os.capture.result.v1]' \
  'tests/test_authority_contract.py::test_capture_flow_fixtures_validate_against_docs_helper_envelope[os-capture-result.v1.fixture.json]' \
  tests/test_authority_contract.py::test_capture_flow_docs_helper_pins_stage_policy_and_traceability_progression \
  tests/test_authority_contract.py::test_authority_timeout_fixture_stays_classification_only_and_non_canonical \
  tests/test_authority_contract.py::test_runtime_statuses_stay_command_specific_while_docs_stage_pins_flow_position \
  tests/test_authority_contract.py::test_os_capture_fixture_pins_note_first_dedupe_and_replay_policy \
  tests/test_authority_contract.py::test_dinger_file_schema_rejects_source_ref_without_primary_identifier \
  tests/test_authority_contract.py::test_os_capture_schema_rejects_missing_project_scope_or_invalid_severity \
  tests/test_authority_contract.py::test_temp_runtime_capture_process_smoke_stays_local_and_reuses_existing_open_item
```

5. TypeScript authority contract gate

```bash
cd /Users/won/Desktop/allinone/knowledge-hub/foundry-core && node --import tsx --test tests/authority-contract.test.ts
```

## Excluded From Approval Gate

- `cd /Users/won/Desktop/allinone/knowledge-hub && pytest`
  - too broad for this tranche; full repo green 아님
- `cd /Users/won/Desktop/allinone/knowledge-hub && python -m pytest -q tests/test_os_cmd.py::test_os_capture_bridges_dinger_capture_results_with_trace tests/test_os_cmd.py::test_os_capture_bridges_dinger_file_results_to_os tests/test_os_cmd.py::test_os_capture_reuses_existing_open_dinger_bridge_item_even_with_summary_drift tests/test_os_cmd.py::test_os_capture_creates_new_item_when_same_dedupe_key_only_matches_resolved_item tests/test_os_cmd.py::test_os_project_evidence_derives_dinger_candidates`
  - still outside the frozen approval gate because the tranche gate remains intentionally narrower than the broader OS subset; a later rerun in this worktree showed the raw direct-bridge assertion green again, so this is now a gate-expansion choice rather than a current red blocker
- raw `--help` checks for `capture status` / `capture requeue` / `capture cleanup`
  - useful as surface evidence only; they are not substitutes for the targeted behavioral tests above
- review surface commands
  - current live CLI exposes `os project evidence`, `os evidence review`, and `os inbox triage`
  - evidence list/show/review coverage still stays outside the frozen blocker set because `project evidence` and `evidence review` remain a thin candidate-review slice over the existing review/promotion surfaces, and the current narrow gate keeps task/decision promotion semantics separate
  - only consider promoting the evidence list/show/review slice into the gate after a deliberate expansion decision backed by a narrow targeted subset that proves candidate-disposition semantics without auto-finalizing tasks or decisions
- `cd /Users/won/Desktop/allinone/knowledge-hub && venv/bin/python -m pytest -q tests/test_dinger_cmd.py -k 'dinger_capture_cleanup_wrapper'`
  - excluded from the frozen blocker set for now because it protects the thin wrapper layer rather than the helper-owned cleanup policy; keep it as supporting evidence unless gate expansion is chosen deliberately

## Verification Result

- 2026-04-09 rerun in this worktree:
  - capture read-model + processor + `requeue`: `15 passed in 1.33s`
  - capture cleanup operator loop: `6 passed in 0.90s`
  - filed Dinger -> OS bridge gate: `4 passed in 1.01s`
  - Python authority contract gate: `9 passed in 1.15s`
  - TypeScript authority contract gate: `14 passed / 0 failed`, `duration_ms 395.862458`
- frozen approval-set total:
  - `48 passed / 0 failed` across 5 commands
- excluded-check confirmation:
  - broader OS subset during the initial freeze pass: `1 failed / 4 passed in 0.99s`
  - `python -m knowledge_hub.interfaces.cli.main dinger capture --help` now shows `cleanup|list|requeue|retry|show|status`
  - `python -m knowledge_hub.interfaces.cli.main os --help` shows `capture|decide|decision|evidence|goal|inbox|next|project|task`; `os evidence --help` shows `review`; `os inbox --help` shows `list|resolve|triage`
  - gate note: live evidence list/show/review help is now present, but the frozen blocker set still stops short of that slice
- later follow-up evidence in the same worktree:
  - broader OS subset including the raw direct-bridge path: `5 passed in 1.13s`
  - this removes the earlier stale direct-bridge assertion as a current reproduced red signal, but it still does not automatically widen the frozen approval gate
- cleanup gate decision follow-up in the same worktree:
  - helper-level cleanup blocker rerun: `6 passed, 1 warning in 0.38s`
  - wrapper-level cleanup evidence rerun: `5 passed, 39 deselected, 2 warnings in 0.58s`
  - cleanup-specific combined evidence: `11 passed / 0 failed`
  - decision: keep the frozen gate on `tests/test_dinger_capture_cleanup.py`; do not widen the blocker set to the wrapper suite yet
- non-blocking environment note:
  - pytest emitted `.pytest_cache` write warnings in this worktree, but the targeted gate commands still completed successfully

## Cleanup Wrapper Decision

- current implementation split:
  - helper/runtime policy lives in `knowledge_hub/application/dinger_capture_cleanup.py`
  - CLI wrapper lives in `knowledge_hub/interfaces/cli/commands/dinger_cmd.py`
- helper-owned behavior:
  - classify queue-backed captures vs recoverable orphans vs unrecoverable runtime junk vs stale/active/invalid claims
  - execute delete vs keep decisions and populate schema-backed `delete|keep|errors|counts`
  - guarantee cleanup never mutates canonical surfaces such as queue packets, Dinger pages, or OS items
- wrapper-only behavior:
  - map `--apply` to `dry_run=False`
  - enforce the `--apply` + `--confirm` guard
  - emit a failed JSON envelope on guard failure instead of a raw click exception when `--json` is used
  - render non-JSON summary/hint text for operator use
- recommendation:
  - keep the narrow blocker on the helper-level suite because it protects the cleanup policy itself
  - treat the wrapper-level suite as a secondary, evidence-only slice for now because it mostly verifies thin option/envelope/printing behavior on top of the same helper payload
  - if a later gate expansion is explicitly chosen, promote the existing targeted wrapper subset rather than adding broad `tests/test_dinger_cmd.py`
