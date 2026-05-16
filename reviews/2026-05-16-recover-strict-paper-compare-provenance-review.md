# Review: Recover strict paper compare provenance

## Findings

- No evidence-gate loosening found in the implemented path: strict spans still require source id, source content hash, and offset-backed `chars:start-end`.
- `memory-unit:` locators remain non-strict even when a hash is present.
- `alexnet-2012` now recovers only after a canonical local PDF is linked and rebuilt through deterministic document-memory / PaperCardV2. The recovered spans use the PDF source hash plus `chars:start-end` offsets.
- The AlexNet source attachment is now reproducible through `khub paper repair-source` when the configured paper store contains the exact canonical PDF basename; the 15/15 eval no longer depends on an unrecorded manual SQLite edit.
- Korean summary slot text is not allowed to become strict original-source evidence when the strict anchor quote is an English source span.
- Compare packet span normalization now prefers real `sourceContentHash` over snippet/content hashes when both are present.
- Post-review hardening fixed two evidence-contract risks: source-backed PaperCardV2 anchors now prefer `source_excerpt` when a local source path exists instead of reusing summary/paraphrase text for offset-backed spans, and ask-v2 result metadata no longer falls back from snippet `contentHash` to source-content hash.
- Follow-up hardening closed the remaining compare packet P1 risks: `contentHash` no longer backfills `sourceContentHash`, ask-v2 claim anchors no longer fill snippet `contentHash` from source hashes, strict spans require exact `chars:start-end`, and `bytes:` / bare range locators remain non-strict even with source hashes.
- Post-subagent review hardening closed the downstream gate and save/lookup risks too: the live compare quality eval now recomputes strict span eligibility from `sourceContentHash` plus exact `chars:start-end`, source identity aliasing no longer treats snippet `contentHash` as a source hash alias, and evidence registry source refs/source revision hashes no longer fall back to snippet hashes.

## Risks

- PDF offset recovery is based on extracted text while the source hash identifies the PDF file, matching the existing paper source snapshot pattern but still dependent on parser stability.
- The bounded ordered-token matcher can bridge parser-inserted LaTeX wrappers; it is intentionally window-limited to reduce broad accidental matches.
- The AlexNet repair still depends on the operator having the canonical PDF in the configured paper store; the PDF itself is not committed as a fixture. The DB mutation is performed by repo-controlled repair code, not by treating fallback, locator-only, memory-unit, Korean summary, or paraphrase text as strict evidence.

## Missing Tests

- No missing regression identified for this tranche. Added tests cover AlexNet-style configured source attachment through the repair path, PDF strict provenance recovery, Korean summary and same-language paraphrase non-promotion, explicit non-strict slot refs, snippet-hash/source-hash separation, and memory-unit locator non-promotion.
- Follow-up regressions also cover `contentHash`-only `chars:` spans, `bytes:` locators, bare range locators, source/snippet hash separation, fallback-span non-promotion in compare packets, and ask-v2 non-`chars:` locator non-promotion.
- Live-eval and registry regressions cover legacy `strictSpanBacked` flags without `sourceContentHash`, expected-source alias false positives from snippet `contentHash`, and saved packet source revision false positives from snippet-only hashes.
