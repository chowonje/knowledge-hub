# Limited Visual Retrieval Hint Candidate Store Allowlist Apply Executor Dry Run

- schema: `knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-allowlist-apply-executor-dry-run.v1`
- status: `ready`
- decision: `ready_for_limited_visual_retrieval_hint_candidate_store_apply_executor_review`
- generatedAt: `2026-05-27T11:11:18Z`
- sourceAllowlistReviewReport: `eval/knowledgeos/reports/limited_visual_retrieval_hint_candidate_store_apply_allowlist_review_005.v1.json`
- sourceAllowlistRows: `125`
- excludedHoldoutRows: `5`
- executorDryRunRows: `125`
- plannedWriteRows: `125`
- candidateStoreWriteRows: `0`
- candidateStoreApplyRows: `0`
- blockedRows: `0`
- privatePathLeakRows: `0`
- schemaViolationCount: `0`

## Mutation Guarantees

- writes: `report_only`
- candidateStoreWriteRows: `0`
- candidateStoreApplyRows: `0`
- vectorIndexing: `False`
- operationalSearchIndexQueryRows: `0`
- answerGenerationRows: `0`
- databaseMutationRows: `0`
- indexMutationRows: `0`
- strictEvidencePromotionRows: `0`
- runtimeAnswerVisibleExposureRows: `0`
- vaultScanRows: `0`
- externalDownloadRows: `0`

## Type Summary

| type | executor dry-run rows | planned writes | jsonl serializable | blocked |
|---|---:|---:|---:|---:|
| equation_region | 25 | 25 | 25 | 0 |
| figure_caption_region | 38 | 38 | 38 | 0 |
| image_region | 16 | 16 | 16 | 0 |
| layout_region | 16 | 16 | 16 | 0 |
| table_region | 30 | 30 | 30 | 0 |

## Sample Executor Dry Run Rows

| # | paperId | type | page | would write on separate apply | sourceCandidateId |
|---:|---|---|---:|---|---|
| 1 | alexnet-2012 | figure_caption_region | 3 | `True` | `visual-layout:alexnet-2012:figure_caption_region:3:5e4d346d4c7b2c53` |
| 2 | alexnet-2012 | equation_region | 4 | `True` | `visual-layout:alexnet-2012:equation_region:4:f9d505b3e6aabed6` |
| 3 | alexnet-2012 | equation_region | 4 | `True` | `visual-layout:alexnet-2012:equation_region:4:79cf21420d190e08` |
| 4 | alexnet-2012 | figure_caption_region | 5 | `True` | `visual-layout:alexnet-2012:figure_caption_region:5:03c450b40ade0263` |
| 5 | alexnet-2012 | figure_caption_region | 6 | `True` | `visual-layout:alexnet-2012:figure_caption_region:6:f8996246c800f4f1` |
| 6 | alexnet-2012 | equation_region | 6 | `True` | `visual-layout:alexnet-2012:equation_region:6:81ead85a04bce51a` |
| 7 | alexnet-2012 | table_region | 7 | `True` | `visual-layout:alexnet-2012:table_region:7:0777b7d11b932456` |
| 8 | alexnet-2012 | table_region | 7 | `True` | `visual-layout:alexnet-2012:table_region:7:b9f742e1da360378` |
| 9 | alexnet-2012 | figure_caption_region | 8 | `True` | `visual-layout:alexnet-2012:figure_caption_region:8:889b5135e8e9ab9e` |
| 10 | resnet-2015 | figure_caption_region | 1 | `True` | `visual-layout:resnet-2015:figure_caption_region:1:aba2bf94ef1625a2` |
| 11 | resnet-2015 | figure_caption_region | 2 | `True` | `visual-layout:resnet-2015:figure_caption_region:2:9687efdba9452012` |
| 12 | resnet-2015 | equation_region | 3 | `True` | `visual-layout:resnet-2015:equation_region:3:0e80570b407b6793` |
| 13 | resnet-2015 | equation_region | 3 | `True` | `visual-layout:resnet-2015:equation_region:3:69248b1db8c80503` |
| 14 | resnet-2015 | equation_region | 3 | `True` | `visual-layout:resnet-2015:equation_region:3:7068ad040e2d0243` |
| 15 | resnet-2015 | figure_caption_region | 4 | `True` | `visual-layout:resnet-2015:figure_caption_region:4:0b3754b79959d64e` |
| 16 | resnet-2015 | figure_caption_region | 5 | `True` | `visual-layout:resnet-2015:figure_caption_region:5:666c24606607fbbf` |
| 17 | resnet-2015 | table_region | 5 | `True` | `visual-layout:resnet-2015:table_region:5:749ede3c6c93e4fa` |
| 18 | resnet-2015 | table_region | 5 | `True` | `visual-layout:resnet-2015:table_region:5:6f67c711d387611a` |
| 19 | clip-2021 | image_region | 2 | `True` | `visual-layout:clip-2021:image_region:2:2616032b2a9d352d` |
| 20 | mae-2021 | image_region | 1 | `True` | `visual-layout:mae-2021:image_region:1:258a60cc216d8130` |
| 21 | alexnet-2012 | image_region | 6 | `True` | `visual-layout:alexnet-2012:image_region:6:508eb98021de0fd7` |
| 22 | clip-2021 | image_region | 2 | `True` | `visual-layout:clip-2021:image_region:2:2642c8db8bd29d0a` |
| 23 | mae-2021 | image_region | 1 | `True` | `visual-layout:mae-2021:image_region:1:40b27e8b4073d98f` |
| 24 | alexnet-2012 | image_region | 8 | `True` | `visual-layout:alexnet-2012:image_region:8:219e7594a10f3a3f` |
| 25 | clip-2021 | image_region | 15 | `True` | `visual-layout:clip-2021:image_region:15:1da1f1cf8ce81bd5` |
| 26 | mae-2021 | image_region | 2 | `True` | `visual-layout:mae-2021:image_region:2:0b66280633a636cc` |
| 27 | clip-2021 | figure_caption_region | 2 | `True` | `visual-layout:clip-2021:figure_caption_region:2:b354160446b3a89c` |
| 28 | mae-2021 | figure_caption_region | 1 | `True` | `visual-layout:mae-2021:figure_caption_region:1:f3477cc8ce7ec8b1` |
| 29 | resnet-2015 | figure_caption_region | 6 | `True` | `visual-layout:resnet-2015:figure_caption_region:6:6b3dd4d05dc69888` |
| 30 | clip-2021 | figure_caption_region | 3 | `True` | `visual-layout:clip-2021:figure_caption_region:3:fd2db71adcd98ec4` |
| 31 | clip-2021 | figure_caption_region | 5 | `True` | `visual-layout:clip-2021:figure_caption_region:5:87777b9ff7911bbe` |
| 32 | mae-2021 | figure_caption_region | 2 | `True` | `visual-layout:mae-2021:figure_caption_region:2:b35cefa766216e3f` |

## Excluded Holdouts

- `visual-retrieval-hint:mae-2021:figure_caption_region:2:b1c1635f3ea9867d`
- `visual-retrieval-hint:resnet-2015:figure_caption_region:8:5983f29673800b6b`
- `visual-retrieval-hint:mae-2021:table_region:8:8a7d640f19165642`
- `visual-retrieval-hint:clip-2021:table_region:22:567a15ee4200c9cb`
- `visual-retrieval-hint:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:table_region:6:db1aff0a9626c727`

## Warnings

- `This dry-run reconstructs allowlisted future JSONL records but writes no candidate store.`
- `The five holdout rows remain excluded from this executor dry-run.`
- `A separate explicit apply gate is still required before any store mutation, and a later gate is required before indexing.`
