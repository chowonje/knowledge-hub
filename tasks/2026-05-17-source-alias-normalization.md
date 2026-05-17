# Source alias normalization

## Goal

- Prevent short aliases such as `CNN`, `GPT`, and `RAG` from becoming broad global source identities while preserving explicit compare-context rescues.

## Scope

- Treat ambiguous short aliases as lookup/query context only for single-paper lookup planning.
- Keep representative retrieval expansions so acronym lookup prompts still have useful search terms.
- Preserve curated compare-context source-pair rescues for prompts such as `RAG vs FiD` and `BERT vs GPT`.
- Prevent live compare expected-source coverage from resolving ambiguous short aliases through payload titles alone.
- Add focused regression tests and durable project records.

## Non-scope

- No new source acquisition, corpus policy, public CLI/MCP, provider, storage, SQLite, or index lifecycle changes.
- No changes to evidence, provenance, no-answer, answerability, corpus coverage, or strict span rules.
- No global alias ontology rewrite.

## Done Condition

- Single-paper lookup prompts for `CNN`, `GPT`, and `RAG` do not set exact `resolved_source_ids`.
- Compare prompts still resolve the curated source pairs that rely on those aliases.
- Live compare expected-source coverage reports ambiguous short expected aliases as unresolved unless non-ambiguous metadata supports the match.
- Focused compare/query/eval tests and release checks pass.

## Planned Files

- `knowledge_hub/domain/ai_papers/query_plan.py`
- `knowledge_hub/domain/source_identity.py`
- `tests/test_paper_query_plan.py`
- `tests/test_live_compare_quality_eval.py`
- `CHANGELOG.md`
- `docs/PROJECT_STATE.md`
- `tasks/2026-05-17-source-alias-normalization.md`
- `reviews/2026-05-17-source-alias-normalization-review.md`
- `worklog/2026-05-17.md`

## Verification Plan

- `pytest tests/test_paper_query_plan.py tests/test_live_compare_quality_eval.py -q`
- `pytest tests/test_paper_query_plan.py tests/test_live_compare_quality_eval.py tests/test_compare_packet_contract.py tests/test_paper_ask_v2.py tests/test_retrieval_pipeline_services.py -q`
- `python -m py_compile knowledge_hub/domain/ai_papers/query_plan.py knowledge_hub/domain/source_identity.py eval/knowledgeos/scripts/check_live_compare_quality_eval.py`
- `python scripts/check_release_smoke.py`
- `python scripts/check_public_release_hygiene.py`
- `pytest -q`
- `git diff --check`
