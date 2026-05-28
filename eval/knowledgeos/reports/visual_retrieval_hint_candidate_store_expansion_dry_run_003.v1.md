# Visual Retrieval Hint Candidate Store Expansion Dry Run

- schema: `knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-dry-run.v1`
- status: `ready`
- decision: `ready_for_visual_retrieval_hint_candidate_store_expansion_review`
- generatedAt: `2026-05-27T03:17:05Z`
- sourceDesignReport: `eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_design_003.v1.json`
- dryRunRows: `24`
- plannedWriteRows: `24`
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
| 1 | clip-2021 | table_region | 21 | `visual-retrieval-hint:clip-2021:table_region:21:7383ff52813f3311` | `True` | `sha256:84151a8e1da287a7f2635cd8e075ce227aebc4e3b9ce8d4c586f9d7bded2b311` |
| 2 | mae-2021 | table_region | 7 | `visual-retrieval-hint:mae-2021:table_region:7:60c6c34fe4bf2a08` | `True` | `sha256:6a7c108a11abf7695cbe22448a4ec06711206c7fb9a8a2dd461ad998bb278c9d` |
| 3 | resnet-2015 | table_region | 6 | `visual-retrieval-hint:resnet-2015:table_region:6:8b698e4cb33e4331` | `True` | `sha256:3445f0854ab7600fc547b232ade1138efe08064815dd2ab548a3ee09d67aeecf` |
| 4 | clip-2021 | table_region | 21 | `visual-retrieval-hint:clip-2021:table_region:21:919ea1060820f507` | `True` | `sha256:e7c908133f791e557c084cc2cdc6f0e96a9f2454f222a6532198852755129c6e` |
| 5 | mae-2021 | table_region | 8 | `visual-retrieval-hint:mae-2021:table_region:8:d8bb6136c1efe969` | `True` | `sha256:3636d70e087155999699942f06f2a69768351b90429d2dcb0769a20462295565` |
| 6 | resnet-2015 | table_region | 6 | `visual-retrieval-hint:resnet-2015:table_region:6:2305462f39ba5737` | `True` | `sha256:daba1a554e8b43086937bad0247ce438e67399202aff9db6b4e95205358ee223` |
| 7 | clip-2021 | table_region | 22 | `visual-retrieval-hint:clip-2021:table_region:22:3d7cfa6728d04d79` | `True` | `sha256:c5a1b4a083a75006fcc613306395e5a2a910a02677a387e760db1ea86ef0168b` |
| 8 | mae-2021 | table_region | 8 | `visual-retrieval-hint:mae-2021:table_region:8:2bc01c52e31087bf` | `True` | `sha256:3dbb542b62a54f94e4852ef5fa0792beafa19ed6f739af6c2a166be2b24073e7` |
| 9 | clip-2021 | figure_caption_region | 7 | `visual-retrieval-hint:clip-2021:figure_caption_region:7:1a37a433790b9a22` | `True` | `sha256:b3990d83c3faf6e59b045fac44bd4fcbf3e75c366023ee8ec46ac8d866ed2b48` |
| 10 | mae-2021 | figure_caption_region | 3 | `visual-retrieval-hint:mae-2021:figure_caption_region:3:6e122ed840ea33ae` | `True` | `sha256:fb68b77259a158e9af4bf2a587d0e19283a3cdf8508fec84d655a53bc6457dfe` |
| 11 | resnet-2015 | figure_caption_region | 8 | `visual-retrieval-hint:resnet-2015:figure_caption_region:8:7ec512c0cf518107` | `True` | `sha256:073f0230fd68f0489b07d4b33bb619478df668a37442e1ed1f117df9f0848397` |
| 12 | clip-2021 | figure_caption_region | 8 | `visual-retrieval-hint:clip-2021:figure_caption_region:8:272352414ddb5953` | `True` | `sha256:82dfdfdd41bf30de13918b99b224250b05f524ab378be27bf984e08b7ca4044f` |
| 13 | mae-2021 | figure_caption_region | 4 | `visual-retrieval-hint:mae-2021:figure_caption_region:4:0ecee7372a8c2dd1` | `True` | `sha256:3d2162173a958d53edc89474bdabcbc2bf1b6fd8da479e4ea1c14121aeacc673` |
| 14 | clip-2021 | figure_caption_region | 9 | `visual-retrieval-hint:clip-2021:figure_caption_region:9:056f3fe7e8348200` | `True` | `sha256:afdc880c7e4d491f11ffc135466cdd7f8f67a669fd0b82f7313afb8bb1214e5e` |
| 15 | mae-2021 | figure_caption_region | 6 | `visual-retrieval-hint:mae-2021:figure_caption_region:6:a20fa27dc1268fe7` | `True` | `sha256:16c5c617682cdc6483d11c8139c4c178ba6a3934b0a699bdb3bc4f01555e63d6` |
| 16 | clip-2021 | figure_caption_region | 10 | `visual-retrieval-hint:clip-2021:figure_caption_region:10:8ae776681eff1396` | `True` | `sha256:8b2fdc39599f63e330256ab9a1666e444b9a49fbb1d8edb7bf65d435d69b86c2` |
| 17 | clip-2021 | equation_region | 5 | `visual-retrieval-hint:clip-2021:equation_region:5:a5cfa0b0864a4292` | `True` | `sha256:368bcae46487af2f76a37173849561a69856fff4c0a9c2e90069ffc40bad6ae6` |
| 18 | clip-2021 | equation_region | 5 | `visual-retrieval-hint:clip-2021:equation_region:5:96e864eaac443f0b` | `True` | `sha256:ad58852a526adad64e8665a818a0c832451ae5ba2c72489beaad0911f1de0b9b` |
| 19 | clip-2021 | equation_region | 10 | `visual-retrieval-hint:clip-2021:equation_region:10:f75e7778a2640f77` | `True` | `sha256:5d9bcac117da18ecd35e3be4075a4325817731962bac5d59b43d5fed1a0b4d1d` |
| 20 | clip-2021 | equation_region | 10 | `visual-retrieval-hint:clip-2021:equation_region:10:3c7f38012be09d67` | `True` | `sha256:8d9d67074584f63534bbafe050d3e9796c59950e6e00361afd6965551534e9fc` |
| 21 | clip-2021 | layout_region | 1 | `visual-retrieval-hint:clip-2021:layout_region:1:3764120a5bf66acf` | `True` | `sha256:eb8101d2c322f3597ab8b7261807e6601f9c9fc73b865657adcab56cfca79b65` |
| 22 | mae-2021 | layout_region | 1 | `visual-retrieval-hint:mae-2021:layout_region:1:7e8e4671cdd367fc` | `True` | `sha256:28960dfb9e39c3bb80ff7cd7d1555f5ba1edb0253fddccefb1ab63de7a994132` |
| 23 | alexnet-2012 | layout_region | 1 | `visual-retrieval-hint:alexnet-2012:layout_region:1:330062fb88b99ac7` | `True` | `sha256:ca6d7979c5394662110b9eb9416424339d7cde3a52b5ecba5d0d7ae5f7f762ef` |
| 24 | resnet-2015 | layout_region | 1 | `visual-retrieval-hint:resnet-2015:layout_region:1:0848218a8267509b` | `True` | `sha256:dff3b78a8a082d13b0f781a4b35758b183a3480e425440129579ba739b5f8999` |

## Warnings

- `This dry-run previews future JSONL records but writes no candidate store.`
- `All rows remain unindexed and not runtime-visible after dry-run.`
- `A separate apply tranche is required before any store mutation, and a later gate is required before indexing.`
