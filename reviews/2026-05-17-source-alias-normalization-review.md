# Review: Source alias normalization

## Findings

- Ambiguous short aliases are now centralized as `CNN`, `GPT`, and `RAG` in the affected source-identity and paper-query planning surfaces.
- Single-paper lookup planning no longer seeds those aliases as exact title/source lookup forms or accepts local exact-title matches for them.
- Representative retrieval expansions remain in `expanded_terms`, so acronym lookup prompts still have useful retrieval hints without exact source binding.
- Compare-context lookup is preserved: curated compare rescues still allow explicit source-pair prompts that rely on `RAG`, `GPT`, or similar aliases.
- Live compare expected-source coverage no longer treats payload title `RAG` as enough to satisfy expected source id `RAG`; the evaluator reports an unresolved expected-source alias instead.

## Risks

- The ambiguous short alias set is intentionally narrow. Other short aliases may still need review if they become overloaded in live evals.
- Some operator-local eval cases that used bare `RAG`, `GPT`, or `CNN` as expected source ids may need to switch to arXiv ids, stable source ids, or full titles.
- This tranche does not rewrite the broader alias ontology or public docs beyond the project-state/changelog record.

## Missing Tests

- Focused tests cover lookup planning for `CNN`, `RAG`, and `GPT`, including local exact-title and card-search paths, and live compare expected-source coverage for an ambiguous short alias.
- Direct source-identity tests cover ambiguous short alias suppression while preserving stable hash and arXiv aliases; live compare tests also cover stable source-id matching when the payload title is an ambiguous short alias.
- Broader focused compare/ask/retrieval regression, changed Python `py_compile`, release smoke, public release hygiene, full pytest, and diff hygiene passed after docs/record updates.
- Operator live compare eval was attempted in this worktree but skipped with `0` declared/evaluated cases because `eval/knowledgeos/queries/live_compare_quality_eval_cases.local.json` is absent, so this tranche should not claim fresh live-corpus eval evidence from that command.
