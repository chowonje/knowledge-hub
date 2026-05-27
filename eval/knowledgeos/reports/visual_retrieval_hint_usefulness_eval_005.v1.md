# Visual Retrieval Hint Usefulness Eval

- schema: `knowledge-hub.paper.visual-retrieval-hint-usefulness-eval.v1`
- status: `ready`
- decision: `ready_for_targeted_visual_retrieval_hint_search_eval`
- generatedAt: `2026-05-27T10:32:29Z`
- inputHintRows: `130`
- evaluatedRows: `130`
- highUsefulnessRows: `126`
- mediumUsefulnessRows: `4`
- lowUsefulnessRows: `0`
- textOnlyTop5Rows: `0`
- augmentedTop5Rows: `130`
- rankImprovedRows: `130`
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
| equation_region | 25 | 25 | 0 | 0 | 0 | 25 |
| figure_caption_region | 40 | 39 | 1 | 0 | 0 | 40 |
| image_region | 16 | 16 | 0 | 0 | 0 | 16 |
| layout_region | 16 | 16 | 0 | 0 | 0 | 16 |
| table_region | 33 | 30 | 3 | 0 | 0 | 33 |

## Top Rows

| # | tier | paperId | type | page | textRank | augmentedRank | delta | sourceCandidateId |
|---:|---|---|---|---:|---:|---:|---:|---|
| 1 | high | alexnet-2012 | equation_region | 4 | None | 1 | 36664 | `visual-layout:alexnet-2012:equation_region:4:79cf21420d190e08` |
| 2 | high | alexnet-2012 | equation_region | 4 | None | 1 | 36664 | `visual-layout:alexnet-2012:equation_region:4:f9d505b3e6aabed6` |
| 3 | high | alexnet-2012 | equation_region | 6 | None | 1 | 36664 | `visual-layout:alexnet-2012:equation_region:6:81ead85a04bce51a` |
| 4 | high | alexnet-2012 | figure_caption_region | 3 | None | 1 | 36664 | `visual-layout:alexnet-2012:figure_caption_region:3:5e4d346d4c7b2c53` |
| 5 | high | alexnet-2012 | figure_caption_region | 5 | None | 1 | 36664 | `visual-layout:alexnet-2012:figure_caption_region:5:03c450b40ade0263` |
| 6 | high | alexnet-2012 | figure_caption_region | 6 | None | 1 | 36664 | `visual-layout:alexnet-2012:figure_caption_region:6:f8996246c800f4f1` |
| 7 | high | alexnet-2012 | figure_caption_region | 8 | None | 1 | 36664 | `visual-layout:alexnet-2012:figure_caption_region:8:889b5135e8e9ab9e` |
| 8 | high | alexnet-2012 | image_region | 8 | None | 1 | 36664 | `visual-layout:alexnet-2012:image_region:8:219e7594a10f3a3f` |
| 9 | high | alexnet-2012 | layout_region | 1 | None | 1 | 36664 | `visual-layout:alexnet-2012:layout_region:1:0a3974d993387e3b` |
| 10 | high | alexnet-2012 | layout_region | 1 | None | 1 | 36664 | `visual-layout:alexnet-2012:layout_region:1:e3b7474586959627` |
| 11 | high | alexnet-2012 | table_region | 7 | None | 1 | 36664 | `visual-layout:alexnet-2012:table_region:7:0777b7d11b932456` |
| 12 | high | alexnet-2012 | table_region | 7 | None | 1 | 36664 | `visual-layout:alexnet-2012:table_region:7:b9f742e1da360378` |
| 13 | high | arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts | equation_region | 3 | None | 1 | 36664 | `visual-layout:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:equation_region:3:f2208953da94826f` |
| 14 | high | arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts | figure_caption_region | 2 | None | 1 | 36664 | `visual-layout:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:figure_caption_region:2:9768341fe65e7ef2` |
| 15 | high | arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts | image_region | 1 | None | 1 | 36664 | `visual-layout:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:image_region:1:0792bc4980e9c1ad` |
| 16 | high | arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts | layout_region | 1 | None | 1 | 36664 | `visual-layout:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:layout_region:1:9f959e82ebee205d` |
| 17 | high | arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts | table_region | 8 | None | 1 | 36664 | `visual-layout:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:table_region:8:96e7bcf8b6c5c548` |
| 18 | high | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | equation_region | 4 | None | 1 | 36664 | `visual-layout:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:equation_region:4:45200729f88c94d7` |
| 19 | high | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | figure_caption_region | 1 | None | 1 | 36664 | `visual-layout:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:figure_caption_region:1:3db7901a05fe4358` |
| 20 | high | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | image_region | 1 | None | 1 | 36664 | `visual-layout:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:image_region:1:0a83cc7b8ce39324` |
| 21 | high | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | layout_region | 1 | None | 1 | 36664 | `visual-layout:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:layout_region:1:c96ad0bb30a29669` |
| 22 | high | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | table_region | 15 | None | 1 | 36664 | `visual-layout:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:table_region:15:bf53567890d6c9c9` |
| 23 | high | clip-2021 | equation_region | 10 | None | 1 | 36664 | `visual-layout:clip-2021:equation_region:10:b13e1298e1e46bb8` |
| 24 | high | clip-2021 | equation_region | 10 | None | 1 | 36664 | `visual-layout:clip-2021:equation_region:10:db8ab5cab587035e` |

## Warnings

- `This report estimates utility only; it does not prove answerability.`
- `Visual derived text remains retrieval-hint-only and non-evidence.`
- `Do not scale to all PDFs until targeted search eval confirms useful signal.`
