# Add artifact acquisition corpus bootstrap

## Goal

- Add an explicit opt-in helper for manifest-backed local paper corpus acquisition without making repair, eval, import, or CI download paper artifacts implicitly.

## Scope

- Add hidden `khub paper corpus-bootstrap` operator command.
- Read the existing corpus manifest and selected artifact/source ids.
- Plan missing artifacts by default.
- Download only when both `--apply` and `--allow-network` are present.
- Verify source-content hash and byte length before writing to `papers_dir`.
- Preserve fail-closed behavior for missing provenance, unsupported URLs, existing mismatches, and downloaded mismatches.
- Emit safe diagnostics and `repair-source` dry-run hints without invoking repair.
- Add focused tests and durable docs/records.

## Non-scope

- No default downloads in `repair-source`, live compare eval, CI, import, or paper add.
- No committed PDFs/full text.
- No SQLite writes, source attach, derivative rebuild, MCP exposure, provider calls, or public help widening.
- No broad alias, public CLI/MCP, or acquisition policy rewrite.

## Done Condition

- Operators can dry-run or explicitly apply selected corpus artifact acquisition.
- Hash or size mismatch cannot produce a green status or overwrite existing mismatched files.
- Repo fixtures remain repo/CI-owned and are not acquired through `papers_dir`.
- Local corpus artifacts do not resolve through repo fixture paths.
- The helper result contract and project records document the boundary.

## Planned Files

- `knowledge_hub/papers/corpus_bootstrap.py`
- `knowledge_hub/interfaces/cli/commands/paper_cmd.py`
- `tests/test_corpus_bootstrap.py`
- `docs/contracts/corpus-bootstrap-result-v1.md`
- `README.md`
- `docs/guides/cli-commands.md`
- `CHANGELOG.md`
- `docs/PROJECT_STATE.md`
- `tasks/2026-05-17-add-artifact-acquisition-corpus-bootstrap.md`
- `reviews/2026-05-17-add-artifact-acquisition-corpus-bootstrap-review.md`
- `worklog/2026-05-17.md`

## Verification Plan

- `pytest tests/test_corpus_bootstrap.py -q`
- `pytest tests/test_paper_source_repairs.py tests/test_live_compare_quality_eval.py tests/test_corpus_bootstrap.py -q`
- `python -m py_compile knowledge_hub/papers/corpus_bootstrap.py knowledge_hub/interfaces/cli/commands/paper_cmd.py`
- `python scripts/check_release_smoke.py`
- `python scripts/check_public_release_hygiene.py`
- `pytest -q`
- `git diff --check`
