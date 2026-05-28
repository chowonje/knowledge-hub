# Visual Retrieval Hint Usefulness Eval

- schema: `knowledge-hub.paper.visual-retrieval-hint-usefulness-eval.v1`
- status: `ready`
- decision: `ready_for_targeted_visual_retrieval_hint_search_eval`
- generatedAt: `2026-05-27T03:38:59Z`
- inputHintRows: `66`
- evaluatedRows: `66`
- highUsefulnessRows: `63`
- mediumUsefulnessRows: `3`
- lowUsefulnessRows: `0`
- textOnlyTop5Rows: `0`
- augmentedTop5Rows: `66`
- rankImprovedRows: `66`
- blockedRows: `0`

## Mutation Guarantees

- writes: `report_only`
- candidateStoreWriteRows: `0`
- vectorIndexing: `False`
- searchIndexQueryRows: `0`
- answerGenerationRows: `0`
- databaseMutationRows: `0`
- indexMutationRows: `0`
- strictEvidencePromotionRows: `0`
- runtimeAnswerVisibleExposureRows: `0`

## Type Summary

| type | rows | high | medium | low | textTop5 | augmentedTop5 |
|---|---:|---:|---:|---:|---:|---:|
| equation_region | 13 | 13 | 0 | 0 | 0 | 13 |
| figure_caption_region | 24 | 23 | 1 | 0 | 0 | 24 |
| image_region | 8 | 8 | 0 | 0 | 0 | 8 |
| layout_region | 4 | 4 | 0 | 0 | 0 | 4 |
| table_region | 17 | 15 | 2 | 0 | 0 | 17 |

## Top Rows

| # | tier | paperId | type | page | textRank | augmentedRank | delta | sourceCandidateId |
|---:|---|---|---|---:|---:|---:|---:|---|
| 1 | high | alexnet-2012 | equation_region | 4 | None | 1 | 467 | `visual-layout:alexnet-2012:equation_region:4:79cf21420d190e08` |
| 2 | high | alexnet-2012 | equation_region | 4 | None | 1 | 467 | `visual-layout:alexnet-2012:equation_region:4:f9d505b3e6aabed6` |
| 3 | high | alexnet-2012 | equation_region | 6 | None | 1 | 467 | `visual-layout:alexnet-2012:equation_region:6:81ead85a04bce51a` |
| 4 | high | alexnet-2012 | figure_caption_region | 3 | None | 1 | 467 | `visual-layout:alexnet-2012:figure_caption_region:3:5e4d346d4c7b2c53` |
| 5 | high | alexnet-2012 | figure_caption_region | 5 | None | 1 | 467 | `visual-layout:alexnet-2012:figure_caption_region:5:03c450b40ade0263` |
| 6 | high | alexnet-2012 | figure_caption_region | 6 | None | 1 | 467 | `visual-layout:alexnet-2012:figure_caption_region:6:f8996246c800f4f1` |
| 7 | high | alexnet-2012 | figure_caption_region | 8 | None | 1 | 467 | `visual-layout:alexnet-2012:figure_caption_region:8:889b5135e8e9ab9e` |
| 8 | high | alexnet-2012 | image_region | 8 | None | 1 | 467 | `visual-layout:alexnet-2012:image_region:8:219e7594a10f3a3f` |
| 9 | high | alexnet-2012 | layout_region | 1 | None | 1 | 467 | `visual-layout:alexnet-2012:layout_region:1:0a3974d993387e3b` |
| 10 | high | alexnet-2012 | table_region | 7 | None | 1 | 467 | `visual-layout:alexnet-2012:table_region:7:0777b7d11b932456` |
| 11 | high | alexnet-2012 | table_region | 7 | None | 1 | 467 | `visual-layout:alexnet-2012:table_region:7:b9f742e1da360378` |
| 12 | high | clip-2021 | equation_region | 10 | None | 1 | 467 | `visual-layout:clip-2021:equation_region:10:b13e1298e1e46bb8` |
| 13 | high | clip-2021 | equation_region | 10 | None | 1 | 467 | `visual-layout:clip-2021:equation_region:10:db8ab5cab587035e` |
| 14 | high | clip-2021 | equation_region | 1 | None | 1 | 467 | `visual-layout:clip-2021:equation_region:1:dd31b4b813b73af2` |
| 15 | high | clip-2021 | equation_region | 5 | None | 1 | 467 | `visual-layout:clip-2021:equation_region:5:1c4738912cb4bdf9` |
| 16 | high | clip-2021 | equation_region | 5 | None | 1 | 467 | `visual-layout:clip-2021:equation_region:5:1fa7893d5e5d7b10` |
| 17 | high | clip-2021 | equation_region | 5 | None | 1 | 467 | `visual-layout:clip-2021:equation_region:5:6936e394c7364af3` |
| 18 | high | clip-2021 | equation_region | 5 | None | 1 | 467 | `visual-layout:clip-2021:equation_region:5:7b890343228eec5b` |
| 19 | high | clip-2021 | figure_caption_region | 10 | None | 1 | 467 | `visual-layout:clip-2021:figure_caption_region:10:967a1812d18f9d62` |
| 20 | high | clip-2021 | figure_caption_region | 2 | None | 1 | 467 | `visual-layout:clip-2021:figure_caption_region:2:b354160446b3a89c` |
| 21 | high | clip-2021 | figure_caption_region | 3 | None | 1 | 467 | `visual-layout:clip-2021:figure_caption_region:3:fd2db71adcd98ec4` |
| 22 | high | clip-2021 | figure_caption_region | 5 | None | 1 | 467 | `visual-layout:clip-2021:figure_caption_region:5:87777b9ff7911bbe` |
| 23 | high | clip-2021 | figure_caption_region | 7 | None | 1 | 467 | `visual-layout:clip-2021:figure_caption_region:7:d3e128cd03f8c6bd` |
| 24 | high | clip-2021 | figure_caption_region | 8 | None | 1 | 467 | `visual-layout:clip-2021:figure_caption_region:8:75613ea3ae6b4e06` |

## Warnings

- `This report estimates utility only; it does not prove answerability.`
- `Visual derived text remains retrieval-hint-only and non-evidence.`
- `Do not scale to all PDFs until targeted search eval confirms useful signal.`
