# Visual Retrieval Hint Usefulness Eval

- schema: `knowledge-hub.paper.visual-retrieval-hint-usefulness-eval.v1`
- status: `ready`
- decision: `ready_for_targeted_visual_retrieval_hint_search_eval`
- generatedAt: `2026-05-27T07:48:32Z`
- inputHintRows: `90`
- evaluatedRows: `90`
- highUsefulnessRows: `86`
- mediumUsefulnessRows: `4`
- lowUsefulnessRows: `0`
- textOnlyTop5Rows: `0`
- augmentedTop5Rows: `90`
- rankImprovedRows: `90`
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
| equation_region | 17 | 17 | 0 | 0 | 0 | 17 |
| figure_caption_region | 32 | 31 | 1 | 0 | 0 | 32 |
| image_region | 8 | 8 | 0 | 0 | 0 | 8 |
| layout_region | 8 | 8 | 0 | 0 | 0 | 8 |
| table_region | 25 | 22 | 3 | 0 | 0 | 25 |

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
| 10 | high | alexnet-2012 | layout_region | 1 | None | 1 | 467 | `visual-layout:alexnet-2012:layout_region:1:e3b7474586959627` |
| 11 | high | alexnet-2012 | table_region | 7 | None | 1 | 467 | `visual-layout:alexnet-2012:table_region:7:0777b7d11b932456` |
| 12 | high | alexnet-2012 | table_region | 7 | None | 1 | 467 | `visual-layout:alexnet-2012:table_region:7:b9f742e1da360378` |
| 13 | high | clip-2021 | equation_region | 10 | None | 1 | 467 | `visual-layout:clip-2021:equation_region:10:db8ab5cab587035e` |
| 14 | high | clip-2021 | equation_region | 15 | None | 1 | 467 | `visual-layout:clip-2021:equation_region:15:36e881ed149ab580` |
| 15 | high | clip-2021 | equation_region | 16 | None | 1 | 467 | `visual-layout:clip-2021:equation_region:16:ae86cff7e225293a` |
| 16 | high | clip-2021 | equation_region | 17 | None | 1 | 467 | `visual-layout:clip-2021:equation_region:17:65b08b3d7099f2e0` |
| 17 | high | clip-2021 | equation_region | 19 | None | 1 | 467 | `visual-layout:clip-2021:equation_region:19:95c566a959a29f80` |
| 18 | high | clip-2021 | equation_region | 1 | None | 1 | 467 | `visual-layout:clip-2021:equation_region:1:dd31b4b813b73af2` |
| 19 | high | clip-2021 | equation_region | 5 | None | 1 | 467 | `visual-layout:clip-2021:equation_region:5:1c4738912cb4bdf9` |
| 20 | high | clip-2021 | equation_region | 5 | None | 1 | 467 | `visual-layout:clip-2021:equation_region:5:1fa7893d5e5d7b10` |
| 21 | high | clip-2021 | equation_region | 5 | None | 1 | 467 | `visual-layout:clip-2021:equation_region:5:6936e394c7364af3` |
| 22 | high | clip-2021 | equation_region | 5 | None | 1 | 467 | `visual-layout:clip-2021:equation_region:5:7b890343228eec5b` |
| 23 | high | clip-2021 | figure_caption_region | 10 | None | 1 | 467 | `visual-layout:clip-2021:figure_caption_region:10:967a1812d18f9d62` |
| 24 | high | clip-2021 | figure_caption_region | 10 | None | 1 | 467 | `visual-layout:clip-2021:figure_caption_region:10:f7695e0650ad9755` |

## Warnings

- `This report estimates utility only; it does not prove answerability.`
- `Visual derived text remains retrieval-hint-only and non-evidence.`
- `Do not scale to all PDFs until targeted search eval confirms useful signal.`
