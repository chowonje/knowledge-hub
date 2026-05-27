# Visual Annotation Expansion Pack Design

- schema: `knowledge-hub.paper.visual-annotation-expansion-pack-design.v1`
- status: `ready`
- decision: `ready_for_visual_annotation_expansion_attachment_pack`
- generatedAt: `2026-05-27T08:29:23Z`
- packId: `visual_annotation_corpus_expansion_pack_005`
- selectedExpansionRows: `40`
- imageCandidateRows: `8`
- figureCandidateRows: `8`
- tableCandidateRows: `8`
- equationCandidateRows: `8`
- wholeImageRows: `0`
- privatePathLeakRows: `0`

## Mutation Guarantees

- writes: `report_only`
- modelCalls: `False`
- webModelCalls: `False`
- wholeImageGptRows: `0`
- cropWriteRows: `0`
- vectorIndexing: `False`
- candidateStoreMutationRows: `0`
- runtimeAnswerVisibleExposureRows: `0`
- strictEvidencePromotionRows: `0`
- vaultScanRows: `0`
- externalDownloadRows: `0`

## Expansion Rows

| # | paperId | type | page | sourceCandidateId | reason | attachment |
|---:|---|---|---:|---|---|---|
| 1 | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | table_region | 2 | `visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:table_region:2:43ba20bbd82c40cc` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 2 | emu3.5-native-multimodal-models-are-world-learners | table_region | 6 | `visual-layout:emu3.5-native-multimodal-models-are-world-learners:table_region:6:7e5ddeb596848dc5` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 3 | faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks | table_region | 6 | `visual-layout:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:table_region:6:bf4da7f980d217f7` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 4 | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | table_region | 15 | `visual-layout:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:table_region:15:bf53567890d6c9c9` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 5 | arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts | table_region | 8 | `visual-layout:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:table_region:8:96e7bcf8b6c5c548` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 6 | high-resolution-image-synthesis-with-latent-diffusion-models | table_region | 4 | `visual-layout:high-resolution-image-synthesis-with-latent-diffusion-models:table_region:4:cdab6c275bd25d05` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 7 | qwen-image-technical-report | table_region | 9 | `visual-layout:qwen-image-technical-report:table_region:9:47a89c68476bd4bb` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 8 | photorealistic-text-to-image-diffusion-models-with-deep-language-understanding | table_region | 7 | `visual-layout:photorealistic-text-to-image-diffusion-models-with-deep-language-understanding:table_region:7:c8163ca9120a3550` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 9 | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | figure_caption_region | 4 | `visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:figure_caption_region:4:92b8890073250699` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 10 | emu3.5-native-multimodal-models-are-world-learners | figure_caption_region | 1 | `visual-layout:emu3.5-native-multimodal-models-are-world-learners:figure_caption_region:1:4281f17d9edca1b7` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 11 | faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks | figure_caption_region | 2 | `visual-layout:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:figure_caption_region:2:053df2c84c04975d` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 12 | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | figure_caption_region | 1 | `visual-layout:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:figure_caption_region:1:3db7901a05fe4358` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 13 | arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts | figure_caption_region | 2 | `visual-layout:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:figure_caption_region:2:9768341fe65e7ef2` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 14 | high-resolution-image-synthesis-with-latent-diffusion-models | figure_caption_region | 1 | `visual-layout:high-resolution-image-synthesis-with-latent-diffusion-models:figure_caption_region:1:bd010932c7e494a5` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 15 | qwen-image-technical-report | figure_caption_region | 2 | `visual-layout:qwen-image-technical-report:figure_caption_region:2:a2a6bdca9d8f65d3` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 16 | photorealistic-text-to-image-diffusion-models-with-deep-language-understanding | figure_caption_region | 2 | `visual-layout:photorealistic-text-to-image-diffusion-models-with-deep-language-understanding:figure_caption_region:2:16dc56c857fd0641` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 17 | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | equation_region | 4 | `visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:equation_region:4:803dd3f7dbd45c6f` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 18 | emu3.5-native-multimodal-models-are-world-learners | equation_region | 2 | `visual-layout:emu3.5-native-multimodal-models-are-world-learners:equation_region:2:688127b9a1184640` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 19 | faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks | equation_region | 5 | `visual-layout:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:equation_region:5:128a0a411f67a324` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 20 | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | equation_region | 4 | `visual-layout:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:equation_region:4:45200729f88c94d7` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 21 | arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts | equation_region | 3 | `visual-layout:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:equation_region:3:f2208953da94826f` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 22 | high-resolution-image-synthesis-with-latent-diffusion-models | equation_region | 1 | `visual-layout:high-resolution-image-synthesis-with-latent-diffusion-models:equation_region:1:0dc11401a800bfef` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 23 | qwen-image-technical-report | equation_region | 14 | `visual-layout:qwen-image-technical-report:equation_region:14:1e8a7149fc3a29d8` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 24 | photorealistic-text-to-image-diffusion-models-with-deep-language-understanding | equation_region | 20 | `visual-layout:photorealistic-text-to-image-diffusion-models-with-deep-language-understanding:equation_region:20:2eb72e2516a0abf7` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 25 | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | layout_region | 1 | `visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:layout_region:1:45ff0082a3c9dd85` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 26 | emu3.5-native-multimodal-models-are-world-learners | layout_region | 1 | `visual-layout:emu3.5-native-multimodal-models-are-world-learners:layout_region:1:03b6139e0d9b28fb` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 27 | faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks | layout_region | 1 | `visual-layout:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:layout_region:1:f15a56f92bf8e4d7` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 28 | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | layout_region | 1 | `visual-layout:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:layout_region:1:c96ad0bb30a29669` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 29 | arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts | layout_region | 1 | `visual-layout:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:layout_region:1:9f959e82ebee205d` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 30 | high-resolution-image-synthesis-with-latent-diffusion-models | layout_region | 1 | `visual-layout:high-resolution-image-synthesis-with-latent-diffusion-models:layout_region:1:47134ad1cd52c96a` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 31 | qwen-image-technical-report | layout_region | 6 | `visual-layout:qwen-image-technical-report:layout_region:6:ed37c9f9cfe1ded1` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 32 | photorealistic-text-to-image-diffusion-models-with-deep-language-understanding | layout_region | 1 | `visual-layout:photorealistic-text-to-image-diffusion-models-with-deep-language-understanding:layout_region:1:3d2326e60384515a` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 33 | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | image_region | 4 | `visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:image_region:4:04ae14ad971d2f06` | `first_image_region_probe_after_text_caption_batch` | `context_crop_png` |
| 34 | emu3.5-native-multimodal-models-are-world-learners | image_region | 1 | `visual-layout:emu3.5-native-multimodal-models-are-world-learners:image_region:1:03dc21098ebf92d6` | `first_image_region_probe_after_text_caption_batch` | `context_crop_png` |
| 35 | faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks | image_region | 2 | `visual-layout:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:image_region:2:2ea5949dc69e0e4c` | `first_image_region_probe_after_text_caption_batch` | `context_crop_png` |
| 36 | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | image_region | 1 | `visual-layout:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:image_region:1:0a83cc7b8ce39324` | `first_image_region_probe_after_text_caption_batch` | `context_crop_png` |
| 37 | arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts | image_region | 1 | `visual-layout:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:image_region:1:0792bc4980e9c1ad` | `first_image_region_probe_after_text_caption_batch` | `context_crop_png` |
| 38 | high-resolution-image-synthesis-with-latent-diffusion-models | image_region | 1 | `visual-layout:high-resolution-image-synthesis-with-latent-diffusion-models:image_region:1:2360b13807a5f368` | `first_image_region_probe_after_text_caption_batch` | `context_crop_png` |
| 39 | qwen-image-technical-report | image_region | 1 | `visual-layout:qwen-image-technical-report:image_region:1:50985e4339ab85a3` | `first_image_region_probe_after_text_caption_batch` | `context_crop_png` |
| 40 | photorealistic-text-to-image-diffusion-models-with-deep-language-understanding | image_region | 2 | `visual-layout:photorealistic-text-to-image-diffusion-models-with-deep-language-understanding:image_region:2:204aa4cc291f768f` | `first_image_region_probe_after_text_caption_batch` | `context_crop_png` |

## Warnings

- `This expansion design writes no crop files and sends no images to GPT/VLM.`
- `Image-region candidates are included only as bounded context-crop annotation candidates.`
- `Whole-image or whole-page annotation remains deferred to visual_full_image_annotation_pack_design.`
- `Any future derivedTextForRetrieval remains retrieval_hint_only and non-evidence.`
