# Visual Retrieval Hint Search Eval

- schema: `knowledge-hub.paper.visual-retrieval-hint-search-eval.v1`
- status: `ready`
- decision: `ready_for_limited_visual_retrieval_hint_candidate_store_apply_design`
- generatedAt: `2026-05-27T10:32:56Z`
- inputHintRows: `130`
- queryRows: `260`
- textOnlyHitAt5Rows: `168`
- augmentedHitAt5Rows: `258`
- rankImprovedRows: `151`
- rankRegressedRows: `3`
- textOnlyMrr: `0.506347`
- augmentedMrr: `0.908049`

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
| equation_region | 50 | 17 | 50 | 46 | 0 |
| figure_caption_region | 80 | 68 | 79 | 29 | 1 |
| image_region | 32 | 10 | 32 | 28 | 0 |
| layout_region | 32 | 20 | 32 | 23 | 0 |
| table_region | 66 | 53 | 65 | 25 | 2 |

## Sample Query Rows

| # | kind | paperId | type | textRank | augmentedRank | delta | sourceCandidateId |
|---:|---|---|---|---:|---:|---:|---|
| 1 | natural_lookup | alexnet-2012 | equation_region | None | 1 | 36664 | `visual-layout:alexnet-2012:equation_region:4:79cf21420d190e08` |
| 2 | natural_lookup | clip-2021 | equation_region | None | 1 | 36664 | `visual-layout:clip-2021:equation_region:19:95c566a959a29f80` |
| 3 | keyword_lookup | clip-2021 | image_region | None | 1 | 36664 | `visual-layout:clip-2021:image_region:2:2616032b2a9d352d` |
| 4 | keyword_lookup | clip-2021 | image_region | None | 1 | 36664 | `visual-layout:clip-2021:image_region:2:2642c8db8bd29d0a` |
| 5 | natural_lookup | clip-2021 | image_region | None | 1 | 36664 | `visual-layout:clip-2021:image_region:2:2642c8db8bd29d0a` |
| 6 | keyword_lookup | emu3.5-native-multimodal-models-are-world-learners | equation_region | None | 1 | 36664 | `visual-layout:emu3.5-native-multimodal-models-are-world-learners:equation_region:2:688127b9a1184640` |
| 7 | keyword_lookup | faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks | equation_region | None | 1 | 36664 | `visual-layout:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:equation_region:5:128a0a411f67a324` |
| 8 | natural_lookup | faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks | equation_region | None | 1 | 36664 | `visual-layout:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:equation_region:5:128a0a411f67a324` |
| 9 | keyword_lookup | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | equation_region | None | 1 | 36664 | `visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:equation_region:4:803dd3f7dbd45c6f` |
| 10 | keyword_lookup | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | image_region | None | 1 | 36664 | `visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:image_region:4:04ae14ad971d2f06` |
| 11 | natural_lookup | qwen-image-technical-report | equation_region | None | 1 | 36664 | `visual-layout:qwen-image-technical-report:equation_region:14:1e8a7149fc3a29d8` |
| 12 | keyword_lookup | alexnet-2012 | image_region | None | 2 | 36663 | `visual-layout:alexnet-2012:image_region:6:508eb98021de0fd7` |
| 13 | natural_lookup | alexnet-2012 | image_region | None | 2 | 36663 | `visual-layout:alexnet-2012:image_region:6:508eb98021de0fd7` |
| 14 | natural_lookup | clip-2021 | equation_region | None | 2 | 36663 | `visual-layout:clip-2021:equation_region:5:1c4738912cb4bdf9` |
| 15 | natural_lookup | clip-2021 | image_region | None | 2 | 36663 | `visual-layout:clip-2021:image_region:2:2616032b2a9d352d` |
| 16 | natural_lookup | emu3.5-native-multimodal-models-are-world-learners | equation_region | None | 2 | 36663 | `visual-layout:emu3.5-native-multimodal-models-are-world-learners:equation_region:2:688127b9a1184640` |
| 17 | natural_lookup | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | image_region | None | 2 | 36663 | `visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:image_region:4:04ae14ad971d2f06` |
| 18 | natural_lookup | clip-2021 | equation_region | None | 3 | 36662 | `visual-layout:clip-2021:equation_region:5:7b890343228eec5b` |
| 19 | natural_lookup | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | equation_region | None | 4 | 36661 | `visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:equation_region:4:803dd3f7dbd45c6f` |
| 20 | natural_lookup | resnet-2015 | figure_caption_region | None | 7 | 36658 | `visual-layout:resnet-2015:figure_caption_region:8:5cf998f6a7a3739f` |
| 21 | natural_lookup | mae-2021 | image_region | 3823 | 3 | 3820 | `visual-layout:mae-2021:image_region:1:258a60cc216d8130` |
| 22 | keyword_lookup | mae-2021 | image_region | 3117 | 2 | 3115 | `visual-layout:mae-2021:image_region:1:40b27e8b4073d98f` |
| 23 | natural_lookup | mae-2021 | image_region | 2814 | 4 | 2810 | `visual-layout:mae-2021:image_region:1:40b27e8b4073d98f` |
| 24 | keyword_lookup | mae-2021 | image_region | 1489 | 4 | 1485 | `visual-layout:mae-2021:image_region:1:258a60cc216d8130` |

## Warnings

- `This report measures retrieval-hint search utility only; it does not prove scientific answerability.`
- `Visual derived text remains non-evidence and must not become citation-grade evidence.`
- `The next step is limited apply design, not vector indexing or answer runtime exposure.`
