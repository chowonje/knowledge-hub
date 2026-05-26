# Visual Retrieval Hint Candidate Store Dry Run

- schema: `knowledge-hub.paper.visual-retrieval-hint-candidate-store-dry-run.v1`
- status: `ready`
- decision: `ready_for_visual_annotation_expansion_pack_design`
- generatedAt: `2026-05-26T13:31:42Z`
- sourceDesignReport: `eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_design.v1.json`
- dryRunRows: `18`
- plannedWriteRows: `18`
- candidateStoreWriteRows: `0`
- indexEligibleRows: `0`
- runtimeVisibleRows: `0`
- privatePathLeakRows: `0`

## Mutation Guarantees

- writes: `report_only`
- candidateStoreWriteRows: `0`
- vectorIndexing: `False`
- indexMutationRows: `0`
- runtimeAnswerVisibleExposureRows: `0`
- databaseMutationRows: `0`
- reindexOrReembedRows: `0`
- vaultScanRows: `0`
- externalDownloadRows: `0`
- strictEvidencePromotionRows: `0`
- answerabilityGateBypassRows: `0`

## Dry Run Rows

| # | paperId | type | page | hintCandidateId | wouldWriteOnApply | recordSha256 |
|---:|---|---|---:|---|---|---|
| 1 | alexnet-2012 | figure_caption_region | 3 | `visual-retrieval-hint:alexnet-2012:figure_caption_region:3:7533b25075b8dd07` | `True` | `sha256:05e1c060675e6d80cc715ae550bcbe05efc7a58457c152488aa16db7f5c14864` |
| 2 | alexnet-2012 | equation_region | 4 | `visual-retrieval-hint:alexnet-2012:equation_region:4:599abe7d6fb2a0e3` | `True` | `sha256:969662dc7126163fab5770d640e982bb265c9c4b18c3c9ec87dc273577ea8ccd` |
| 3 | alexnet-2012 | equation_region | 4 | `visual-retrieval-hint:alexnet-2012:equation_region:4:044989cbca13e1ab` | `True` | `sha256:97e49036c8ebac1e86098e96e4f294b01c484f36014e9f38f2f32f00fc6dd1aa` |
| 4 | alexnet-2012 | figure_caption_region | 5 | `visual-retrieval-hint:alexnet-2012:figure_caption_region:5:a19ab7caaf1ab591` | `True` | `sha256:fedf36beb9f7b283e7327f1869e1c34fd5cc1d09a0f38b1ad2919dbe5b6474f7` |
| 5 | alexnet-2012 | figure_caption_region | 6 | `visual-retrieval-hint:alexnet-2012:figure_caption_region:6:272d65dc6ff69f11` | `True` | `sha256:6f6d3e0181aab04a34d83efe750ab3d55ab86eb128e44259dc90e7ca29729bd8` |
| 6 | alexnet-2012 | equation_region | 6 | `visual-retrieval-hint:alexnet-2012:equation_region:6:5dbeede97c7195d0` | `True` | `sha256:f5a3e0ff543a9348432eeaaeda7e21523ea84c558ab90ccfce69d0df126dcf0a` |
| 7 | alexnet-2012 | table_region | 7 | `visual-retrieval-hint:alexnet-2012:table_region:7:89ae4caae1bab67a` | `True` | `sha256:63e6c09efcf9be2097a7bdbafac23816982ff7763de08bba7642968e7c016a2b` |
| 8 | alexnet-2012 | table_region | 7 | `visual-retrieval-hint:alexnet-2012:table_region:7:5714e8a00c6d7b79` | `True` | `sha256:b3ac4e9870f01d26cb0dbce1fb5d8f6a91b66b8d65fb247e24e55bddad6a0fd5` |
| 9 | alexnet-2012 | figure_caption_region | 8 | `visual-retrieval-hint:alexnet-2012:figure_caption_region:8:c4c1f97a36eecaa9` | `True` | `sha256:b33f5249e98f8bc8fe6d80e08c711549bbaa89e4cda0ca3d73c866547053b5bd` |
| 10 | resnet-2015 | figure_caption_region | 1 | `visual-retrieval-hint:resnet-2015:figure_caption_region:1:a0093ef958135ed2` | `True` | `sha256:e22b80f7ba6059af0dcc995da44352c723ea4eafdc8bb7cdbe5c36b8b94bf991` |
| 11 | resnet-2015 | figure_caption_region | 2 | `visual-retrieval-hint:resnet-2015:figure_caption_region:2:8131d443f220b4e5` | `True` | `sha256:373eb5b922fa1c862124c885603743ca1d68c11e0181a591bdae22357e337f03` |
| 12 | resnet-2015 | equation_region | 3 | `visual-retrieval-hint:resnet-2015:equation_region:3:558ad607d1504b14` | `True` | `sha256:5b69d3531c09c4450bf6039827f737fb8f68f83671ec3db2c21a3a83a5d10c35` |
| 13 | resnet-2015 | equation_region | 3 | `visual-retrieval-hint:resnet-2015:equation_region:3:e6f9e4f13bd33541` | `True` | `sha256:21c425193d9803587279f67306a9a9a06430c0ca668460863684634284d75960` |
| 14 | resnet-2015 | equation_region | 3 | `visual-retrieval-hint:resnet-2015:equation_region:3:63077f0605d9ffe2` | `True` | `sha256:2ad044683876ce3a552d2844a13d6643c423233251af2696a645c4bc2629fc27` |
| 15 | resnet-2015 | figure_caption_region | 4 | `visual-retrieval-hint:resnet-2015:figure_caption_region:4:0667d8cb1903aae8` | `True` | `sha256:8f554eba0d4764a5fde6c4746d18c7c5d63573f84e76dec056d3ff87e0329004` |
| 16 | resnet-2015 | figure_caption_region | 5 | `visual-retrieval-hint:resnet-2015:figure_caption_region:5:92873cb76744c957` | `True` | `sha256:9cdd3d4c3decbb12132f261bca322413e8fbd2c50aef0d73f4ba3d21e5077a89` |
| 17 | resnet-2015 | table_region | 5 | `visual-retrieval-hint:resnet-2015:table_region:5:b78e3e200c882470` | `True` | `sha256:71184048dc7e12fae7a55546f034d64c89befe7e42943ec8c736e5ae22fb7cc3` |
| 18 | resnet-2015 | table_region | 5 | `visual-retrieval-hint:resnet-2015:table_region:5:08af9e1ef54f5dcf` | `True` | `sha256:23c14be42b1b33eda7260caf0fdaca53eea01aa863156d1572becbf62e6f6b50` |

## Warnings

- `This dry-run previews future JSONL records but writes no candidate store.`
- `All rows remain unindexed and not runtime-visible after dry-run.`
- `A separate apply tranche is required before any store mutation, and a later gate is required before indexing.`
