# Add paper source artifact corpus policy

## Goal

- Formalize and implement the paper source artifact policy so live compare and repair-source runs distinguish missing local corpus files from evidence failures.

## Scope

- Add a durable ADR for the hybrid source artifact policy.
- Add a metadata-only corpus manifest for current wide live compare paper artifacts.
- Make `paper repair-source` use manifest-backed local artifact checks for source attachment.
- Report `missing_artifact` and `hash_mismatch` diagnostics without DB writes or derivative rebuilds.
- Add live compare declared/evaluable/skipped corpus coverage metrics.
- Derive live compare corpus requirements from expected source ids and the repo-controlled manifest when ignored operator-local cases omit explicit requirements.
- Update operator docs and durable records.

## Non-scope

- No committed PDFs or full paper text artifacts.
- No download/acquisition helper.
- No provider calls, registry writes, MCP changes, or new persistence schema.
- No loosening of strict evidence gates.

## Done Condition

- Missing or hash-mismatched corpus artifacts are visible as structured diagnostics and do not write source paths or rebuild derivatives.
- Live compare reports declared/evaluable/skipped coverage metrics alongside existing safety metrics, and the default gate fails when required corpus coverage is incomplete.
- Missing `corpusRequirements` in ignored live compare cases are generated or validated from the manifest; expected source ids without manifest mappings are reported as missing requirements and fail the gate.
- The manifest and ADR make the git-vs-local-corpus boundary explicit.
- Full pytest no longer fails on an unrelated provider help expectation; the test protects `khub help advanced` as the hidden/operator inventory instead of widening default help.
- Focused tests, `py_compile`, JSON validation, and diff hygiene pass.

## Planned Files

- `docs/adr/2026-05-17-paper-source-artifact-policy.md`
- `eval/knowledgeos/fixtures/corpus_manifest.json`
- `eval/knowledgeos/README.md`
- `eval/knowledgeos/templates/live_compare_quality_eval_cases.template.json`
- `eval/knowledgeos/scripts/check_live_compare_quality_eval.py`
- `knowledge_hub/application/corpus_artifacts.py`
- `knowledge_hub/application/paper_source_repairs.py`
- `tests/test_live_compare_quality_eval.py`
- `tests/test_paper_source_repairs.py`
- `tests/test_provider_custom_surface.py`
- `CHANGELOG.md`
- `docs/PROJECT_STATE.md`

## Verification Plan

- `pytest tests/test_paper_source_repairs.py tests/test_live_compare_quality_eval.py -q`
- `python -m py_compile knowledge_hub/application/corpus_artifacts.py knowledge_hub/application/paper_source_repairs.py eval/knowledgeos/scripts/check_live_compare_quality_eval.py`
- `python -m json.tool eval/knowledgeos/fixtures/corpus_manifest.json`
- `pytest tests/test_provider_custom_surface.py -q`
- `pytest tests/test_provider_custom_surface.py tests/test_paper_source_repairs.py tests/test_live_compare_quality_eval.py -q`
- `pytest -q`
- `python scripts/check_release_smoke.py`
- `python scripts/check_public_release_hygiene.py`
- `git diff --check`
