# Visual Retrieval Hint Candidate Store Expansion Design

- schema: `knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-design.v1`
- status: `ready`
- decision: `ready_for_visual_retrieval_hint_candidate_store_expansion_dry_run`
- generatedAt: `2026-05-27T10:24:01Z`
- sourceValidationReport: `eval/knowledgeos/reports/visual_annotation_expansion_web_output_005.validation.v1.json`
- sourceValidationRows: `40`
- candidateRows: `40`
- indexEligibleRows: `0`
- runtimeVisibleRows: `0`
- privatePathLeakRows: `0`

## Mutation Guarantees

- writes: `report_only`
- apiCalls: `False`
- modelCalls: `False`
- webModelCalls: `False`
- candidateStoreMutationRows: `0`
- vectorIndexing: `False`
- indexEligibleRows: `0`
- strictEvidencePromotionRows: `0`
- runtimeAnswerVisibleExposureRows: `0`
- databaseMutationRows: `0`
- indexMutationRows: `0`
- reindexOrReembedRows: `0`
- vaultScanRows: `0`
- externalDownloadRows: `0`
- answerabilityGateBypassRows: `0`

## Whole-Image Timing

- currentTrancheSendsWholeImagesToGpt: `False`
- earliestRecommendedTranche: `visual_full_image_annotation_pack_design`
- recommendedMaxRowsPerBatch: `24`
- rule: `Only after a separate pack-design gate, and only for candidates where context crops are insufficient, uncertainty is high, or image-region candidates need object/layout inspection.`

## Candidate Rows

| # | paperId | type | page | sourceCandidateId | indexEligible | runtimeVisible |
|---:|---|---|---:|---|---|---|
| 1 | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | table_region | 2 | `visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:table_region:2:43ba20bbd82c40cc` | `False` | `False` |
| 2 | emu3.5-native-multimodal-models-are-world-learners | table_region | 6 | `visual-layout:emu3.5-native-multimodal-models-are-world-learners:table_region:6:7e5ddeb596848dc5` | `False` | `False` |
| 3 | faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks | table_region | 6 | `visual-layout:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:table_region:6:bf4da7f980d217f7` | `False` | `False` |
| 4 | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | table_region | 15 | `visual-layout:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:table_region:15:bf53567890d6c9c9` | `False` | `False` |
| 5 | arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts | table_region | 8 | `visual-layout:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:table_region:8:96e7bcf8b6c5c548` | `False` | `False` |
| 6 | high-resolution-image-synthesis-with-latent-diffusion-models | table_region | 4 | `visual-layout:high-resolution-image-synthesis-with-latent-diffusion-models:table_region:4:cdab6c275bd25d05` | `False` | `False` |
| 7 | qwen-image-technical-report | table_region | 9 | `visual-layout:qwen-image-technical-report:table_region:9:47a89c68476bd4bb` | `False` | `False` |
| 8 | photorealistic-text-to-image-diffusion-models-with-deep-language-understanding | table_region | 7 | `visual-layout:photorealistic-text-to-image-diffusion-models-with-deep-language-understanding:table_region:7:c8163ca9120a3550` | `False` | `False` |
| 9 | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | figure_caption_region | 4 | `visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:figure_caption_region:4:92b8890073250699` | `False` | `False` |
| 10 | emu3.5-native-multimodal-models-are-world-learners | figure_caption_region | 1 | `visual-layout:emu3.5-native-multimodal-models-are-world-learners:figure_caption_region:1:4281f17d9edca1b7` | `False` | `False` |
| 11 | faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks | figure_caption_region | 2 | `visual-layout:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:figure_caption_region:2:053df2c84c04975d` | `False` | `False` |
| 12 | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | figure_caption_region | 1 | `visual-layout:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:figure_caption_region:1:3db7901a05fe4358` | `False` | `False` |
| 13 | arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts | figure_caption_region | 2 | `visual-layout:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:figure_caption_region:2:9768341fe65e7ef2` | `False` | `False` |
| 14 | high-resolution-image-synthesis-with-latent-diffusion-models | figure_caption_region | 1 | `visual-layout:high-resolution-image-synthesis-with-latent-diffusion-models:figure_caption_region:1:bd010932c7e494a5` | `False` | `False` |
| 15 | qwen-image-technical-report | figure_caption_region | 2 | `visual-layout:qwen-image-technical-report:figure_caption_region:2:a2a6bdca9d8f65d3` | `False` | `False` |
| 16 | photorealistic-text-to-image-diffusion-models-with-deep-language-understanding | figure_caption_region | 2 | `visual-layout:photorealistic-text-to-image-diffusion-models-with-deep-language-understanding:figure_caption_region:2:16dc56c857fd0641` | `False` | `False` |
| 17 | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | equation_region | 4 | `visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:equation_region:4:803dd3f7dbd45c6f` | `False` | `False` |
| 18 | emu3.5-native-multimodal-models-are-world-learners | equation_region | 2 | `visual-layout:emu3.5-native-multimodal-models-are-world-learners:equation_region:2:688127b9a1184640` | `False` | `False` |
| 19 | faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks | equation_region | 5 | `visual-layout:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:equation_region:5:128a0a411f67a324` | `False` | `False` |
| 20 | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | equation_region | 4 | `visual-layout:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:equation_region:4:45200729f88c94d7` | `False` | `False` |
| 21 | arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts | equation_region | 3 | `visual-layout:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:equation_region:3:f2208953da94826f` | `False` | `False` |
| 22 | high-resolution-image-synthesis-with-latent-diffusion-models | equation_region | 1 | `visual-layout:high-resolution-image-synthesis-with-latent-diffusion-models:equation_region:1:0dc11401a800bfef` | `False` | `False` |
| 23 | qwen-image-technical-report | equation_region | 14 | `visual-layout:qwen-image-technical-report:equation_region:14:1e8a7149fc3a29d8` | `False` | `False` |
| 24 | photorealistic-text-to-image-diffusion-models-with-deep-language-understanding | equation_region | 20 | `visual-layout:photorealistic-text-to-image-diffusion-models-with-deep-language-understanding:equation_region:20:2eb72e2516a0abf7` | `False` | `False` |
| 25 | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | layout_region | 1 | `visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:layout_region:1:45ff0082a3c9dd85` | `False` | `False` |
| 26 | emu3.5-native-multimodal-models-are-world-learners | layout_region | 1 | `visual-layout:emu3.5-native-multimodal-models-are-world-learners:layout_region:1:03b6139e0d9b28fb` | `False` | `False` |
| 27 | faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks | layout_region | 1 | `visual-layout:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:layout_region:1:f15a56f92bf8e4d7` | `False` | `False` |
| 28 | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | layout_region | 1 | `visual-layout:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:layout_region:1:c96ad0bb30a29669` | `False` | `False` |
| 29 | arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts | layout_region | 1 | `visual-layout:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:layout_region:1:9f959e82ebee205d` | `False` | `False` |
| 30 | high-resolution-image-synthesis-with-latent-diffusion-models | layout_region | 1 | `visual-layout:high-resolution-image-synthesis-with-latent-diffusion-models:layout_region:1:47134ad1cd52c96a` | `False` | `False` |
| 31 | qwen-image-technical-report | layout_region | 6 | `visual-layout:qwen-image-technical-report:layout_region:6:ed37c9f9cfe1ded1` | `False` | `False` |
| 32 | photorealistic-text-to-image-diffusion-models-with-deep-language-understanding | layout_region | 1 | `visual-layout:photorealistic-text-to-image-diffusion-models-with-deep-language-understanding:layout_region:1:3d2326e60384515a` | `False` | `False` |
| 33 | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | image_region | 4 | `visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:image_region:4:04ae14ad971d2f06` | `False` | `False` |
| 34 | emu3.5-native-multimodal-models-are-world-learners | image_region | 1 | `visual-layout:emu3.5-native-multimodal-models-are-world-learners:image_region:1:03dc21098ebf92d6` | `False` | `False` |
| 35 | faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks | image_region | 2 | `visual-layout:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:image_region:2:2ea5949dc69e0e4c` | `False` | `False` |
| 36 | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | image_region | 1 | `visual-layout:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:image_region:1:0a83cc7b8ce39324` | `False` | `False` |
| 37 | arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts | image_region | 1 | `visual-layout:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:image_region:1:0792bc4980e9c1ad` | `False` | `False` |
| 38 | high-resolution-image-synthesis-with-latent-diffusion-models | image_region | 1 | `visual-layout:high-resolution-image-synthesis-with-latent-diffusion-models:image_region:1:2360b13807a5f368` | `False` | `False` |
| 39 | qwen-image-technical-report | image_region | 1 | `visual-layout:qwen-image-technical-report:image_region:1:50985e4339ab85a3` | `False` | `False` |
| 40 | photorealistic-text-to-image-diffusion-models-with-deep-language-understanding | image_region | 2 | `visual-layout:photorealistic-text-to-image-diffusion-models-with-deep-language-understanding:image_region:2:204aa4cc291f768f` | `False` | `False` |

## Warnings

- `This is a design report only; the planned candidate store is not written.`
- `All projected rows remain retrieval_hint_only, not evidence.`
- `Whole-image or whole-page GPT/VLM batches require a later pack-design gate.`
