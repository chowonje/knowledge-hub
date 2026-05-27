# Limited Visual Retrieval Hint Labs Vector Index Search Quality Eval

- schema: `knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-labs-vector-index-search-quality-eval.v1`
- status: `ready`
- decision: `ready_for_limited_visual_retrieval_hint_production_vector_db_integration_design`
- nextRecommendedTranche: `limited_visual_retrieval_hint_production_vector_db_integration_design`
- sourcePlannedVectorUpsertRows: `125`
- actualLabsVectorIndexRows: `125`
- matchedVectorRecordRows: `125`
- queryRows: `250`
- textOnlyHitAt5Rows: `133`
- labsVectorHitAt5Rows: `234`
- hybridHitAt5Rows: `241`
- hybridHitAt5LiftRows: `108`
- textOnlyMrr: `0.445518`
- labsVectorMrr: `0.788217`
- hybridMrr: `0.843291`
- qualityGatePassed: `True`

## Mutation Guarantees

- candidateStoreWriteRows: `0`
- embeddingCallRows: `0`
- vectorIndexWriteRows: `0`
- productionVectorIndexWriteRows: `0`
- databaseMutationRows: `0`
- indexMutationRows: `0`
- runtimeVisibleRows: `0`
- strictEvidenceRows: `0`
- citationGradeRows: `0`

## Type Summary

| type | queryRows | textHit@5 | labsHit@5 | hybridHit@5 | improved | regressed |
|---|---:|---:|---:|---:|---:|---:|
| equation_region | 50 | 13 | 46 | 46 | 46 | 0 |
| figure_caption_region | 76 | 56 | 69 | 74 | 26 | 0 |
| image_region | 32 | 7 | 29 | 29 | 28 | 0 |
| layout_region | 32 | 12 | 32 | 32 | 27 | 0 |
| table_region | 60 | 45 | 58 | 60 | 22 | 0 |

## Sample Query Rows

| # | kind | paperId | type | textRank | labsRank | hybridRank | delta | sourceCandidateId |
|---:|---|---|---|---:|---:|---:|---:|---|
| 1 | keyword_lookup | alexnet-2012 | image_region | None | 1 | 1 | 36664 | `visual-layout:alexnet-2012:image_region:6:508eb98021de0fd7` |
| 2 | natural_lookup | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | equation_region | None | 1 | 1 | 36664 | `visual-layout:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:equation_region:4:45200729f88c94d7` |
| 3 | keyword_lookup | clip-2021 | equation_region | None | 1 | 1 | 36664 | `visual-layout:clip-2021:equation_region:5:7b890343228eec5b` |
| 4 | keyword_lookup | clip-2021 | image_region | None | 1 | 1 | 36664 | `visual-layout:clip-2021:image_region:2:2616032b2a9d352d` |
| 5 | keyword_lookup | clip-2021 | image_region | None | 1 | 1 | 36664 | `visual-layout:clip-2021:image_region:2:2642c8db8bd29d0a` |
| 6 | natural_lookup | clip-2021 | image_region | None | 1 | 1 | 36664 | `visual-layout:clip-2021:image_region:2:2642c8db8bd29d0a` |
| 7 | keyword_lookup | emu3.5-native-multimodal-models-are-world-learners | equation_region | None | 1 | 1 | 36664 | `visual-layout:emu3.5-native-multimodal-models-are-world-learners:equation_region:2:688127b9a1184640` |
| 8 | natural_lookup | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | equation_region | None | 1 | 1 | 36664 | `visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:equation_region:4:803dd3f7dbd45c6f` |
| 9 | keyword_lookup | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | image_region | None | 1 | 1 | 36664 | `visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:image_region:4:04ae14ad971d2f06` |
| 10 | keyword_lookup | qwen-image-technical-report | equation_region | None | 1 | 1 | 36664 | `visual-layout:qwen-image-technical-report:equation_region:14:1e8a7149fc3a29d8` |
| 11 | natural_lookup | resnet-2015 | equation_region | None | 1 | 1 | 36664 | `visual-layout:resnet-2015:equation_region:3:0e80570b407b6793` |
| 12 | natural_lookup | resnet-2015 | table_region | None | 1 | 1 | 36664 | `visual-layout:resnet-2015:table_region:5:749ede3c6c93e4fa` |
| 13 | natural_lookup | alexnet-2012 | equation_region | None | 2 | 2 | 36663 | `visual-layout:alexnet-2012:equation_region:4:79cf21420d190e08` |
| 14 | natural_lookup | alexnet-2012 | image_region | None | 2 | 2 | 36663 | `visual-layout:alexnet-2012:image_region:6:508eb98021de0fd7` |
| 15 | natural_lookup | emu3.5-native-multimodal-models-are-world-learners | table_region | None | 2 | 2 | 36663 | `visual-layout:emu3.5-native-multimodal-models-are-world-learners:table_region:6:7e5ddeb596848dc5` |
| 16 | natural_lookup | qwen-image-technical-report | equation_region | None | 2 | 2 | 36663 | `visual-layout:qwen-image-technical-report:equation_region:14:1e8a7149fc3a29d8` |
| 17 | natural_lookup | resnet-2015 | layout_region | None | 2 | 2 | 36663 | `visual-layout:resnet-2015:layout_region:1:1442b7b9fe2f781a` |
| 18 | natural_lookup | clip-2021 | equation_region | None | 3 | 3 | 36662 | `visual-layout:clip-2021:equation_region:1:dd31b4b813b73af2` |
| 19 | natural_lookup | clip-2021 | equation_region | None | 3 | 3 | 36662 | `visual-layout:clip-2021:equation_region:5:1c4738912cb4bdf9` |
| 20 | natural_lookup | clip-2021 | equation_region | None | 3 | 3 | 36662 | `visual-layout:clip-2021:equation_region:5:7b890343228eec5b` |
| 21 | natural_lookup | emu3.5-native-multimodal-models-are-world-learners | equation_region | None | 3 | 3 | 36662 | `visual-layout:emu3.5-native-multimodal-models-are-world-learners:equation_region:2:688127b9a1184640` |
| 22 | natural_lookup | clip-2021 | image_region | None | 4 | 4 | 36661 | `visual-layout:clip-2021:image_region:2:2616032b2a9d352d` |
| 23 | natural_lookup | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | image_region | None | 4 | 4 | 36661 | `visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:image_region:4:04ae14ad971d2f06` |
| 24 | natural_lookup | resnet-2015 | figure_caption_region | None | 5 | 5 | 36660 | `visual-layout:resnet-2015:figure_caption_region:2:9687efdba9452012` |

## Warnings

- `This report evaluates labs vector search quality only; visual hints remain non-evidence.`
- `Production vector DB integration still requires a separate design/apply gate.`
