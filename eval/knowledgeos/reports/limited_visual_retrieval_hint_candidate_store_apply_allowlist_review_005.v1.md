# Limited Visual Retrieval Hint Candidate Store Apply Allowlist Review

- schema: `knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-apply-allowlist-review.v1`
- status: `ready`
- decision: `ready_for_limited_visual_retrieval_hint_candidate_store_allowlist_apply_executor_dry_run`
- generatedAt: `2026-05-27T10:50:51Z`
- reviewRows: `130`
- allowlistRows: `125`
- holdoutRows: `5`
- applyExecutorDryRunCandidateRows: `125`
- plannedSeparateApplyWriteRows: `125`
- candidateStoreWriteRows: `0`
- blockedRows: `0`

## Mutation Guarantees

- writes: `report_only`
- candidateStoreWriteRows: `0`
- vectorIndexing: `False`
- operationalSearchIndexQueryRows: `0`
- answerGenerationRows: `0`
- databaseMutationRows: `0`
- indexMutationRows: `0`
- strictEvidencePromotionRows: `0`
- runtimeAnswerVisibleExposureRows: `0`

## Type Summary

| type | rows | allowlist | holdout | blocked | improved queries | strong-lift queries |
|---|---:|---:|---:|---:|---:|---:|
| equation_region | 25 | 25 | 0 | 0 | 46 | 32 |
| figure_caption_region | 40 | 38 | 2 | 0 | 29 | 5 |
| image_region | 16 | 16 | 0 | 0 | 28 | 21 |
| layout_region | 16 | 16 | 0 | 0 | 23 | 9 |
| table_region | 33 | 30 | 3 | 0 | 25 | 11 |

## Sample Review Rows

| # | paperId | type | status | improved queries | strong lift | sourceCandidateId |
|---:|---|---|---|---:|---:|---|
| 1 | alexnet-2012 | figure_caption_region | allowlisted_for_apply_executor_dry_run | 1 | 0 | `visual-layout:alexnet-2012:figure_caption_region:3:5e4d346d4c7b2c53` |
| 2 | alexnet-2012 | equation_region | allowlisted_for_apply_executor_dry_run | 2 | 1 | `visual-layout:alexnet-2012:equation_region:4:f9d505b3e6aabed6` |
| 3 | alexnet-2012 | equation_region | allowlisted_for_apply_executor_dry_run | 2 | 2 | `visual-layout:alexnet-2012:equation_region:4:79cf21420d190e08` |
| 4 | alexnet-2012 | figure_caption_region | allowlisted_for_apply_executor_dry_run | 0 | 0 | `visual-layout:alexnet-2012:figure_caption_region:5:03c450b40ade0263` |
| 5 | alexnet-2012 | figure_caption_region | allowlisted_for_apply_executor_dry_run | 0 | 0 | `visual-layout:alexnet-2012:figure_caption_region:6:f8996246c800f4f1` |
| 6 | alexnet-2012 | equation_region | allowlisted_for_apply_executor_dry_run | 2 | 1 | `visual-layout:alexnet-2012:equation_region:6:81ead85a04bce51a` |
| 7 | alexnet-2012 | table_region | allowlisted_for_apply_executor_dry_run | 0 | 0 | `visual-layout:alexnet-2012:table_region:7:0777b7d11b932456` |
| 8 | alexnet-2012 | table_region | allowlisted_for_apply_executor_dry_run | 1 | 1 | `visual-layout:alexnet-2012:table_region:7:b9f742e1da360378` |
| 9 | alexnet-2012 | figure_caption_region | allowlisted_for_apply_executor_dry_run | 0 | 0 | `visual-layout:alexnet-2012:figure_caption_region:8:889b5135e8e9ab9e` |
| 10 | resnet-2015 | figure_caption_region | allowlisted_for_apply_executor_dry_run | 1 | 0 | `visual-layout:resnet-2015:figure_caption_region:1:aba2bf94ef1625a2` |
| 11 | resnet-2015 | figure_caption_region | allowlisted_for_apply_executor_dry_run | 2 | 0 | `visual-layout:resnet-2015:figure_caption_region:2:9687efdba9452012` |
| 12 | resnet-2015 | equation_region | allowlisted_for_apply_executor_dry_run | 2 | 1 | `visual-layout:resnet-2015:equation_region:3:0e80570b407b6793` |
| 13 | resnet-2015 | equation_region | allowlisted_for_apply_executor_dry_run | 2 | 1 | `visual-layout:resnet-2015:equation_region:3:69248b1db8c80503` |
| 14 | resnet-2015 | equation_region | allowlisted_for_apply_executor_dry_run | 2 | 1 | `visual-layout:resnet-2015:equation_region:3:7068ad040e2d0243` |
| 15 | resnet-2015 | figure_caption_region | allowlisted_for_apply_executor_dry_run | 1 | 0 | `visual-layout:resnet-2015:figure_caption_region:4:0b3754b79959d64e` |
| 16 | resnet-2015 | figure_caption_region | allowlisted_for_apply_executor_dry_run | 2 | 1 | `visual-layout:resnet-2015:figure_caption_region:5:666c24606607fbbf` |
| 17 | resnet-2015 | table_region | allowlisted_for_apply_executor_dry_run | 1 | 1 | `visual-layout:resnet-2015:table_region:5:749ede3c6c93e4fa` |
| 18 | resnet-2015 | table_region | allowlisted_for_apply_executor_dry_run | 1 | 0 | `visual-layout:resnet-2015:table_region:5:6f67c711d387611a` |
| 19 | clip-2021 | image_region | allowlisted_for_apply_executor_dry_run | 2 | 2 | `visual-layout:clip-2021:image_region:2:2616032b2a9d352d` |
| 20 | mae-2021 | image_region | allowlisted_for_apply_executor_dry_run | 2 | 2 | `visual-layout:mae-2021:image_region:1:258a60cc216d8130` |
| 21 | alexnet-2012 | image_region | allowlisted_for_apply_executor_dry_run | 2 | 2 | `visual-layout:alexnet-2012:image_region:6:508eb98021de0fd7` |
| 22 | clip-2021 | image_region | allowlisted_for_apply_executor_dry_run | 2 | 2 | `visual-layout:clip-2021:image_region:2:2642c8db8bd29d0a` |
| 23 | mae-2021 | image_region | allowlisted_for_apply_executor_dry_run | 2 | 2 | `visual-layout:mae-2021:image_region:1:40b27e8b4073d98f` |
| 24 | alexnet-2012 | image_region | allowlisted_for_apply_executor_dry_run | 1 | 0 | `visual-layout:alexnet-2012:image_region:8:219e7594a10f3a3f` |

## Holdouts

- `visual-retrieval-hint:mae-2021:figure_caption_region:2:b1c1635f3ea9867d`
- `visual-retrieval-hint:resnet-2015:figure_caption_region:8:5983f29673800b6b`
- `visual-retrieval-hint:mae-2021:table_region:8:8a7d640f19165642`
- `visual-retrieval-hint:clip-2021:table_region:22:567a15ee4200c9cb`
- `visual-retrieval-hint:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:table_region:6:db1aff0a9626c727`

## Warnings

- `Only allowlisted rows may feed a future apply-executor dry-run.`
- `Holdout rows remain excluded until a later search-gate review, recrop, or reannotation tranche.`
- `This report does not authorize or perform candidate-store writes, vector indexing, or runtime exposure.`
