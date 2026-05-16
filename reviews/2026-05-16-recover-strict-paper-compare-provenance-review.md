# Review: Recover strict paper compare provenance

## Findings

- No evidence-gate loosening found in the implemented path: strict spans still require source id, source content hash, and offset-backed `chars:start-end`.
- `memory-unit:` locators remain non-strict even when a hash is present.
- `alexnet-2012` now recovers only after a canonical local PDF is linked and rebuilt through deterministic document-memory / PaperCardV2. The recovered spans use the PDF source hash plus `chars:start-end` offsets.
- Korean summary slot text is not allowed to become strict original-source evidence when the strict anchor quote is an English source span.
- Compare packet span normalization now prefers real `sourceContentHash` over snippet/content hashes when both are present.
- Post-review hardening fixed two evidence-contract risks: source-backed PaperCardV2 anchors now prefer `source_excerpt` when a local source path exists instead of reusing summary/paraphrase text for offset-backed spans, and ask-v2 result metadata no longer falls back from snippet `contentHash` to source-content hash.

## Risks

- PDF offset recovery is based on extracted text while the source hash identifies the PDF file, matching the existing paper source snapshot pattern but still dependent on parser stability.
- The bounded ordered-token matcher can bridge parser-inserted LaTeX wrappers; it is intentionally window-limited to reduce broad accidental matches.
- The AlexNet local data repair is outside git history because it updates the operator SQLite store; a backup was created under `~/.khub/` before mutation.

## Missing Tests

- No missing regression identified for this tranche. Added tests cover AlexNet-style strict provenance recovery, Korean summary and same-language paraphrase non-promotion, explicit non-strict slot refs, snippet-hash/source-hash separation, and memory-unit locator non-promotion.
