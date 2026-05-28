# Visual Retrieval Hint Candidate Store Expansion Dry Run

- schema: `knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-dry-run.v1`
- status: `ready`
- decision: `ready_for_visual_retrieval_hint_candidate_store_expansion_review`
- generatedAt: `2026-05-26T15:56:41Z`
- sourceDesignReport: `eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_design.v1.json`
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
| 1 | clip-2021 | image_region | 2 | `visual-retrieval-hint:clip-2021:image_region:2:e25902a7066b85fb` | `True` | `sha256:76c68823acce59436d4403fd7975db22b00fc29d33f32d5547e44cb43c58f720` |
| 2 | mae-2021 | image_region | 1 | `visual-retrieval-hint:mae-2021:image_region:1:d54434235dc1f73e` | `True` | `sha256:9dee9b8ab72e3639696196cb8cad8f6f776ff657774c359f246d42051ad11d41` |
| 3 | alexnet-2012 | image_region | 6 | `visual-retrieval-hint:alexnet-2012:image_region:6:6bc72fab3308cc56` | `True` | `sha256:2c0235f0384b89bd674d52112bfa9566075190e7d29609b5561ddda49eedd0f0` |
| 4 | clip-2021 | image_region | 2 | `visual-retrieval-hint:clip-2021:image_region:2:0d674a922cfdab51` | `True` | `sha256:e2dcc704b1380b3c04e31cc7bcb21c7b2c3885d37e62a5c50f4c108a54fc0c5d` |
| 5 | mae-2021 | image_region | 1 | `visual-retrieval-hint:mae-2021:image_region:1:e3eb86ab7dd4815d` | `True` | `sha256:96a3a07002d2bcd27826c76222e753d02ed97524a468e3ba1479676d0dbd6ec8` |
| 6 | alexnet-2012 | image_region | 8 | `visual-retrieval-hint:alexnet-2012:image_region:8:aebea8a49393b7e9` | `True` | `sha256:8243f181fe7354b3d21b595dbeb3af06497a0ae1cef7b3ebbb6a14b7746b6fb3` |
| 7 | clip-2021 | image_region | 15 | `visual-retrieval-hint:clip-2021:image_region:15:b2eef0626d7910cf` | `True` | `sha256:075bc49099b5c3fb12af5559666ce33adcd05f64fcd26ba088841efda83ee5ef` |
| 8 | mae-2021 | image_region | 2 | `visual-retrieval-hint:mae-2021:image_region:2:47e41fe986d634ac` | `True` | `sha256:c4b30e0f81960150165087b6085a1fb11f28598271dd503450d31419ce281af1` |
| 9 | clip-2021 | figure_caption_region | 2 | `visual-retrieval-hint:clip-2021:figure_caption_region:2:791bd02c68004e94` | `True` | `sha256:30656d0103b9e1659f248cdcd81f3830e616101aaa0ecc9bc2b170a73545267d` |
| 10 | mae-2021 | figure_caption_region | 1 | `visual-retrieval-hint:mae-2021:figure_caption_region:1:7f1d38a62ebc3ebe` | `True` | `sha256:11bb1d6bb0279e51f9fecdb38a3631653d2bbce2276dbaa357f92c1167e7a127` |
| 11 | resnet-2015 | figure_caption_region | 6 | `visual-retrieval-hint:resnet-2015:figure_caption_region:6:beeb9fb9ad38518d` | `True` | `sha256:0713d78a1378874de15fbe9306acdc253b2df6c3fab73459da4aad96258ec404` |
| 12 | clip-2021 | figure_caption_region | 3 | `visual-retrieval-hint:clip-2021:figure_caption_region:3:7de4aa3a082925a4` | `True` | `sha256:0d9ed835a582e1adf7c2d933347e14b35125e47ebb194c5e7efc80997f32736c` |
| 13 | mae-2021 | figure_caption_region | 2 | `visual-retrieval-hint:mae-2021:figure_caption_region:2:b1c1635f3ea9867d` | `True` | `sha256:d99c2e3b174cfa9312b42459f0ef0926d1dfc8f959dd51c5983277887429021c` |
| 14 | resnet-2015 | figure_caption_region | 8 | `visual-retrieval-hint:resnet-2015:figure_caption_region:8:5983f29673800b6b` | `True` | `sha256:dcacfc2d6ed9c1b4f126481f15e08a0515df449e1cbb73cc5e3d544e1f54b779` |
| 15 | clip-2021 | figure_caption_region | 5 | `visual-retrieval-hint:clip-2021:figure_caption_region:5:aa1447f2ba9526ae` | `True` | `sha256:fbf642b7eba6cacc4311157bbf5057d62e20cbcec60de2aba95a43cb4626d1fe` |
| 16 | mae-2021 | figure_caption_region | 2 | `visual-retrieval-hint:mae-2021:figure_caption_region:2:eea2a43fb3574de9` | `True` | `sha256:685a27a9ed3a1a264f6f0a471516783776a4a23fbc9b8b8fec8b8dc594db5b20` |
| 17 | clip-2021 | table_region | 7 | `visual-retrieval-hint:clip-2021:table_region:7:202c4c0be3960b79` | `True` | `sha256:ee39d8a50e5fcdbc0c8e1a2e8063de5e64f7a67d356b02b3f5f19f742db1ebc9` |
| 18 | mae-2021 | table_region | 5 | `visual-retrieval-hint:mae-2021:table_region:5:626e13443e1ee40c` | `True` | `sha256:b974b7b3b3dafd3568ccdf3ca8dd7c023faa8d31500bd748313228dcc2fae960` |
| 19 | resnet-2015 | table_region | 6 | `visual-retrieval-hint:resnet-2015:table_region:6:7adc39075ec4efe2` | `True` | `sha256:bc27c0f1890f986bf8421777e116c4ebea85a61317a2aeaa817aca4f2418933a` |
| 20 | clip-2021 | table_region | 17 | `visual-retrieval-hint:clip-2021:table_region:17:0c93a8dbedbc0a03` | `True` | `sha256:63948db5743bc588e4e2ca7ebb7eb4689f83a6f0094d948fd33bf7ab17c9c3eb` |
| 21 | mae-2021 | table_region | 5 | `visual-retrieval-hint:mae-2021:table_region:5:eeae9d1ecf9c1122` | `True` | `sha256:f5ee1303034ff35d18bd9731ef1f3b54d7cd67734094c80409021a93ec95dbfe` |
| 22 | clip-2021 | equation_region | 1 | `visual-retrieval-hint:clip-2021:equation_region:1:1578f0d850ce7830` | `True` | `sha256:d34539769032cab9dac22d7234246cf986b38baed2df55134ab6d50598cddf91` |
| 23 | clip-2021 | equation_region | 5 | `visual-retrieval-hint:clip-2021:equation_region:5:a85a9e64dfa04951` | `True` | `sha256:383ae778e8867aebd7715147c7823ececb4a3648e401f2048cd5e251d18cdcfa` |
| 24 | clip-2021 | equation_region | 5 | `visual-retrieval-hint:clip-2021:equation_region:5:2d6d39c3bbc50090` | `True` | `sha256:1c0930cad1fcf5ce7218dbc6439ef3f8550fab7062ca0bc17325ee6e2bfb9527` |

## Warnings

- `This dry-run previews future JSONL records but writes no candidate store.`
- `All rows remain unindexed and not runtime-visible after dry-run.`
- `A separate apply tranche is required before any store mutation, and a later gate is required before indexing.`
