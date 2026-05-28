# Visual Retrieval Hint Search Eval

- schema: `knowledge-hub.paper.visual-retrieval-hint-search-eval.v1`
- status: `ready`
- decision: `ready_for_limited_visual_retrieval_hint_candidate_store_apply_design`
- generatedAt: `2026-05-27T07:48:48Z`
- inputHintRows: `90`
- queryRows: `180`
- textOnlyHitAt5Rows: `141`
- augmentedHitAt5Rows: `178`
- rankImprovedRows: `89`
- rankRegressedRows: `1`
- textOnlyMrr: `0.611073`
- augmentedMrr: `0.913426`

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

| type | queryRows | textHit@5 | augmentedHit@5 | improved | regressed |
|---|---:|---:|---:|---:|---:|
| equation_region | 34 | 21 | 34 | 30 | 0 |
| figure_caption_region | 64 | 60 | 63 | 21 | 1 |
| image_region | 16 | 4 | 16 | 15 | 0 |
| layout_region | 16 | 12 | 16 | 7 | 0 |
| table_region | 50 | 44 | 49 | 16 | 0 |

## Sample Query Rows

| # | kind | paperId | type | textRank | augmentedRank | delta | sourceCandidateId |
|---:|---|---|---|---:|---:|---:|---|
| 1 | natural_lookup | alexnet-2012 | equation_region | None | 1 | 467 | `visual-layout:alexnet-2012:equation_region:4:79cf21420d190e08` |
| 2 | natural_lookup | clip-2021 | equation_region | None | 1 | 467 | `visual-layout:clip-2021:equation_region:19:95c566a959a29f80` |
| 3 | keyword_lookup | clip-2021 | image_region | None | 1 | 467 | `visual-layout:clip-2021:image_region:2:2616032b2a9d352d` |
| 4 | keyword_lookup | clip-2021 | image_region | None | 1 | 467 | `visual-layout:clip-2021:image_region:2:2642c8db8bd29d0a` |
| 5 | natural_lookup | clip-2021 | image_region | None | 1 | 467 | `visual-layout:clip-2021:image_region:2:2642c8db8bd29d0a` |
| 6 | keyword_lookup | alexnet-2012 | image_region | None | 2 | 466 | `visual-layout:alexnet-2012:image_region:6:508eb98021de0fd7` |
| 7 | natural_lookup | alexnet-2012 | image_region | None | 2 | 466 | `visual-layout:alexnet-2012:image_region:6:508eb98021de0fd7` |
| 8 | natural_lookup | clip-2021 | equation_region | None | 2 | 466 | `visual-layout:clip-2021:equation_region:5:1c4738912cb4bdf9` |
| 9 | natural_lookup | clip-2021 | image_region | None | 2 | 466 | `visual-layout:clip-2021:image_region:2:2616032b2a9d352d` |
| 10 | natural_lookup | clip-2021 | equation_region | None | 3 | 465 | `visual-layout:clip-2021:equation_region:5:7b890343228eec5b` |
| 11 | natural_lookup | resnet-2015 | figure_caption_region | None | 6 | 462 | `visual-layout:resnet-2015:figure_caption_region:8:5cf998f6a7a3739f` |
| 12 | natural_lookup | mae-2021 | image_region | 285 | 2 | 283 | `visual-layout:mae-2021:image_region:1:40b27e8b4073d98f` |
| 13 | natural_lookup | mae-2021 | image_region | 275 | 2 | 273 | `visual-layout:mae-2021:image_region:1:258a60cc216d8130` |
| 14 | keyword_lookup | mae-2021 | image_region | 265 | 2 | 263 | `visual-layout:mae-2021:image_region:1:40b27e8b4073d98f` |
| 15 | natural_lookup | mae-2021 | table_region | 258 | 6 | 252 | `visual-layout:mae-2021:table_region:8:71f56f58347cc664` |
| 16 | keyword_lookup | mae-2021 | image_region | 245 | 2 | 243 | `visual-layout:mae-2021:image_region:1:258a60cc216d8130` |
| 17 | natural_lookup | mae-2021 | figure_caption_region | 102 | 2 | 100 | `visual-layout:mae-2021:figure_caption_region:2:b35cefa766216e3f` |
| 18 | natural_lookup | clip-2021 | layout_region | 101 | 2 | 99 | `visual-layout:clip-2021:layout_region:16:0850e569d769717b` |
| 19 | keyword_lookup | clip-2021 | layout_region | 96 | 2 | 94 | `visual-layout:clip-2021:layout_region:16:0850e569d769717b` |
| 20 | natural_lookup | clip-2021 | equation_region | 30 | 1 | 29 | `visual-layout:clip-2021:equation_region:1:dd31b4b813b73af2` |
| 21 | natural_lookup | clip-2021 | image_region | 29 | 1 | 28 | `visual-layout:clip-2021:image_region:15:1da1f1cf8ce81bd5` |
| 22 | natural_lookup | clip-2021 | equation_region | 26 | 4 | 22 | `visual-layout:clip-2021:equation_region:10:b13e1298e1e46bb8` |
| 23 | natural_lookup | resnet-2015 | equation_region | 20 | 1 | 19 | `visual-layout:resnet-2015:equation_region:3:0e80570b407b6793` |
| 24 | natural_lookup | clip-2021 | layout_region | 19 | 1 | 18 | `visual-layout:clip-2021:layout_region:1:7614f0e5b3451acd` |

## Warnings

- `This report measures retrieval-hint search utility only; it does not prove scientific answerability.`
- `Visual derived text remains non-evidence and must not become citation-grade evidence.`
- `The next step is limited apply design, not vector indexing or answer runtime exposure.`
