# Recover strict paper compare provenance

## Goal

- Recover strict paper compare provenance for the remaining wide eval gaps without weakening answerability gates.

## Scope

- Read-only provenance enrichment in ask-v2 for existing local paper source files.
- Local source-path repair for an existing paper row when a canonical source file is already present in the configured paper store.
- Focused regressions for PDF offset recovery, parser-wrapper token matching, and memory-unit locator rejection.
- Focused regressions for Korean summary text not becoming strict original-source evidence and for compare respecting explicit non-strict slot refs.
- Local wide compare eval rerun and durable state notes.

## Non-scope

- No registry writes, provider calls, MCP/public schema changes, or new persistence/schema.
- No promotion of fallback spans, locator-only spans, or `memory-unit:` locators to strict evidence.
- No promotion of Korean summary text as strict original-source evidence when the actual source span is English/original PDF text.

## Done Condition

- `1312.5602` and `2310.11511` recover strict `chars:start-end` source-hash-backed anchors when existing local source files support them.
- `alexnet-2012` recovers only if a canonical local source file can be linked and a stored source span genuinely matches original source text.
- Wide compare eval improves from `12/15` while keeping `expectedNoAnswerPassRate=1.0`, `nonEvidenceLeakCount=0`, and `fallbackOnlyCaseRate=0.0`.

## Planned Files

- `knowledge_hub/ai/ask_v2.py`
- `knowledge_hub/ai/compare_packet.py`
- `knowledge_hub/papers/card_v2_builder.py`
- `knowledge_hub/papers/knowledge_slots.py`
- `knowledge_hub/application/paper_source_repairs.py`
- `tests/test_paper_ask_v2.py`
- `tests/test_compare_packet_contract.py`
- `tests/test_paper_knowledge_slots.py`
- `tests/test_paper_source_repairs.py`
- `CHANGELOG.md`
- `docs/PROJECT_STATE.md`

## Verification Plan

- Focused new regression tests.
- Focused compare/ask/substrate tests.
- 15-case local live compare eval.
- `py_compile`, `git diff --check`, full pytest, release smoke, public hygiene.

## Result

- Local `alexnet-2012` source was first repaired by linking the existing canonical PDF in the configured paper store and rebuilding deterministic document-memory / PaperCardV2 artifacts. A SQLite backup was created first under `~/.khub/`.
- Reproducibility follow-up: a pre-repair database copy failed the wide compare eval on AlexNet strict source coverage; applying the repo-controlled `khub paper repair-source --paper-id alexnet-2012` path against that copied DB restored the canonical source path and produced a `15/15` isolated run without depending on the manually edited operator DB.
- Wide live compare eval is now reproducible after the repair command with `expectedAnswerablePassRate=1.0`, `expectedAnswerableStrictSourceCoverageRate=1.0`, `expectedNoAnswerPassRate=1.0`, `fallbackOnlyCaseRate=0.0`, and `nonEvidenceLeakCount=0`.
- Regression coverage now protects PDF source provenance recovery, Korean summary and same-language paraphrase non-promotion, explicit non-strict slot refs, snippet-hash/source-hash separation, and `memory-unit:` locator non-promotion.
- Strict provenance hardening follow-up: compare packet strict spans now require `sourceContentHash` / `source_content_hash` plus exact `chars:start-end`, ask-v2 claim anchors no longer fill snippet `contentHash` from source hashes, the live compare eval recomputes strict spans from those same source-provenance fields, source aliasing no longer resolves expected sources from snippet hashes, and persisted evidence-registry source revisions no longer fall back to snippet hashes; `contentHash`, `bytes:` locators, bare ranges, `memory-unit:` locators, locator-only anchors, and fallback spans remain non-strict.
