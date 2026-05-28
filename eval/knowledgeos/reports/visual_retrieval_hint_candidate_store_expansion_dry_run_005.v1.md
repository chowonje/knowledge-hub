# Visual Retrieval Hint Candidate Store Expansion Dry Run

- schema: `knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-dry-run.v1`
- status: `ready`
- decision: `ready_for_visual_retrieval_hint_candidate_store_expansion_review`
- generatedAt: `2026-05-27T10:24:09Z`
- sourceDesignReport: `eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_design_005.v1.json`
- dryRunRows: `40`
- plannedWriteRows: `40`
- candidateStoreWriteRows: `0`
- indexEligibleRows: `0`
- runtimeVisibleRows: `0`
- privatePathLeakRows: `0`

## Mutation Guarantees

- writes: `report_only`
- candidateStoreWriteRows: `0`
- vectorIndexing: `False`
- indexMutationRows: `0`
- runtimeAnswerVisibleExposureRows: `0`
- databaseMutationRows: `0`
- reindexOrReembedRows: `0`
- vaultScanRows: `0`
- externalDownloadRows: `0`
- strictEvidencePromotionRows: `0`
- answerabilityGateBypassRows: `0`

## Dry Run Rows

| # | paperId | type | page | hintCandidateId | wouldWriteOnApply | recordSha256 |
|---:|---|---|---:|---|---|---|
| 1 | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | table_region | 2 | `visual-retrieval-hint:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:table_region:2:c9f10ad539df2768` | `True` | `sha256:c01bbed8d7b87e2ec5fe4ae21f985022bc84ae6237bd27042b4c5978da96e19f` |
| 2 | emu3.5-native-multimodal-models-are-world-learners | table_region | 6 | `visual-retrieval-hint:emu3.5-native-multimodal-models-are-world-learners:table_region:6:12cfdc0604700bae` | `True` | `sha256:b543f40c3059a8d4baaffc6a6bea6d860ecc241552bf5d181a79bcd9faa24669` |
| 3 | faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks | table_region | 6 | `visual-retrieval-hint:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:table_region:6:db1aff0a9626c727` | `True` | `sha256:c2d2e87b751104e40e850165f48ee7e2e470801a12d9ca7dd38636f95c4c3c81` |
| 4 | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | table_region | 15 | `visual-retrieval-hint:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:table_region:15:500bcf7d725d5276` | `True` | `sha256:5e13c956af468ea00359cc07cad78c673a2f9e059dcb9cfe5b5b92bf99ee4120` |
| 5 | arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts | table_region | 8 | `visual-retrieval-hint:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:table_region:8:7761cf6e5433c260` | `True` | `sha256:4a6536cec246270ce5e4b453d7802941ef658ab38e995883b7ace4f49704db27` |
| 6 | high-resolution-image-synthesis-with-latent-diffusion-models | table_region | 4 | `visual-retrieval-hint:high-resolution-image-synthesis-with-latent-diffusion-models:table_region:4:f3d383bf253df868` | `True` | `sha256:c2baa214bb112413ce45387c3ca52fba453fc9b0ace3e7264e6e1fa1c87c5f1f` |
| 7 | qwen-image-technical-report | table_region | 9 | `visual-retrieval-hint:qwen-image-technical-report:table_region:9:611ead6dee4da370` | `True` | `sha256:d1cba77d22af04be2621c904bc1ec06c77ab0fe83dd89713cf9f1842752a710e` |
| 8 | photorealistic-text-to-image-diffusion-models-with-deep-language-understanding | table_region | 7 | `visual-retrieval-hint:photorealistic-text-to-image-diffusion-models-with-deep-language-understanding:table_region:7:12fb28b217d61a66` | `True` | `sha256:e04a8d259fbf58fdc85feeca6570f3436cc6f12578b985e8d5762efc9db2f49f` |
| 9 | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | figure_caption_region | 4 | `visual-retrieval-hint:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:figure_caption_region:4:11c75af942e648e4` | `True` | `sha256:fb6026a738aa60c454f1c122ba5a537b7b3a8f1b8eab5e8db78f9d699995403f` |
| 10 | emu3.5-native-multimodal-models-are-world-learners | figure_caption_region | 1 | `visual-retrieval-hint:emu3.5-native-multimodal-models-are-world-learners:figure_caption_region:1:8b375157964405cd` | `True` | `sha256:23a4d5384bd63edf6928048fb1837ce4fc20d29e458f0f34399dd55dad32f8a0` |
| 11 | faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks | figure_caption_region | 2 | `visual-retrieval-hint:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:figure_caption_region:2:398b3e56c69cb996` | `True` | `sha256:2f4b27bb4dfaf8ad4aadafcf000d6f876cc1b94dd76ae8e83220e909bbbecb18` |
| 12 | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | figure_caption_region | 1 | `visual-retrieval-hint:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:figure_caption_region:1:85347f338931992f` | `True` | `sha256:d240a1b35bee626902afe996c9cb2e63d0c8a1610a22bf0f85c62c622499c0cd` |
| 13 | arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts | figure_caption_region | 2 | `visual-retrieval-hint:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:figure_caption_region:2:0440a64e09a213c1` | `True` | `sha256:1471da3b22662a52e6403743858d1ac215e0cd5945f79dad3f9a0f926d123393` |
| 14 | high-resolution-image-synthesis-with-latent-diffusion-models | figure_caption_region | 1 | `visual-retrieval-hint:high-resolution-image-synthesis-with-latent-diffusion-models:figure_caption_region:1:eb65a40d5f902ae8` | `True` | `sha256:5f694850c993d175bdfba0d562317e71bfc2773ff8da24668abde334524f6b1a` |
| 15 | qwen-image-technical-report | figure_caption_region | 2 | `visual-retrieval-hint:qwen-image-technical-report:figure_caption_region:2:cd585568dc55abd4` | `True` | `sha256:c6aeec2447d3b85c7291412f243529b36f268176eb5c285606c5c582dc53a792` |
| 16 | photorealistic-text-to-image-diffusion-models-with-deep-language-understanding | figure_caption_region | 2 | `visual-retrieval-hint:photorealistic-text-to-image-diffusion-models-with-deep-language-understanding:figure_caption_region:2:66d17cb3d9982211` | `True` | `sha256:1560921263aee4152bbfb6765a60948edbf689fa9ce3babbe4708ad0f2754e4f` |
| 17 | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | equation_region | 4 | `visual-retrieval-hint:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:equation_region:4:a89de815eb106b60` | `True` | `sha256:d1a82d86d2ca3c9bcd3d0ded2e9153140b0f866a68ef44d267b0652db2d15749` |
| 18 | emu3.5-native-multimodal-models-are-world-learners | equation_region | 2 | `visual-retrieval-hint:emu3.5-native-multimodal-models-are-world-learners:equation_region:2:8273d162567b7ddd` | `True` | `sha256:ead68f8276484f934d2e8e5845c4669f66d782e94052fdefbe2aaf62436c4cf0` |
| 19 | faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks | equation_region | 5 | `visual-retrieval-hint:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:equation_region:5:e0288e83b106cf2b` | `True` | `sha256:338785e743d832d2fbef4bf688345be749d3959cee857007f8a8fd5c9aa90f50` |
| 20 | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | equation_region | 4 | `visual-retrieval-hint:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:equation_region:4:4e3d90d2dc3fe4b4` | `True` | `sha256:41961ebbf3581866b3603c39cf17c33a3e91684f720cbe2dffd0d64d820f573a` |
| 21 | arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts | equation_region | 3 | `visual-retrieval-hint:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:equation_region:3:3518ac3c8b80a65e` | `True` | `sha256:c3b2efdd62322e616a24ac3f2ecaa2d9d1cd6201be8fcb3221090d0cdc840812` |
| 22 | high-resolution-image-synthesis-with-latent-diffusion-models | equation_region | 1 | `visual-retrieval-hint:high-resolution-image-synthesis-with-latent-diffusion-models:equation_region:1:5cb887e41e8ed993` | `True` | `sha256:ae23582607c7be36cfa0e3eda3369104f64f0947b01e6f2e1893dc309fafec42` |
| 23 | qwen-image-technical-report | equation_region | 14 | `visual-retrieval-hint:qwen-image-technical-report:equation_region:14:0a407be89fee1864` | `True` | `sha256:35e0e5a468efaf32fa9fd9e39f51ce359b35b8cb783fd3979c3a3015299de0a0` |
| 24 | photorealistic-text-to-image-diffusion-models-with-deep-language-understanding | equation_region | 20 | `visual-retrieval-hint:photorealistic-text-to-image-diffusion-models-with-deep-language-understanding:equation_region:20:f3937ddab7fb8a11` | `True` | `sha256:2d4b1efccfeb7820f0ce8616850868f0e28570be30b1fe9d7c07dbf6b4cc4ee4` |
| 25 | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | layout_region | 1 | `visual-retrieval-hint:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:layout_region:1:6cfe39087f3c23fc` | `True` | `sha256:02fbd5d1bae35d1a0cf5e07d4fe89a09b3b72001baee44f5ce0dc6147f7a2453` |
| 26 | emu3.5-native-multimodal-models-are-world-learners | layout_region | 1 | `visual-retrieval-hint:emu3.5-native-multimodal-models-are-world-learners:layout_region:1:b489e9d7d1366176` | `True` | `sha256:8832c3e6615a6e71891311f222f0958b37274a4bebe5104837779c98ae7f30b5` |
| 27 | faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks | layout_region | 1 | `visual-retrieval-hint:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:layout_region:1:43fb5985c62ef435` | `True` | `sha256:5e9f3cb3b1668449c21035e7417f62fdf2f09db91c384bef01a46653695992c2` |
| 28 | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | layout_region | 1 | `visual-retrieval-hint:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:layout_region:1:73d7fdff4b9ac430` | `True` | `sha256:0f39dc199d66afc2e4d1f7953b1eaf679de6fc05bc079102890faa862f19cf7f` |
| 29 | arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts | layout_region | 1 | `visual-retrieval-hint:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:layout_region:1:d01f180b0461dd27` | `True` | `sha256:66b5851c4a9aef7a8f24614cfa24df20e7ff3192db84e165f944d1648221e068` |
| 30 | high-resolution-image-synthesis-with-latent-diffusion-models | layout_region | 1 | `visual-retrieval-hint:high-resolution-image-synthesis-with-latent-diffusion-models:layout_region:1:a83d60fd0c5186b9` | `True` | `sha256:0275862d4e078d993a8be8c9472a8954518f1a5c3b54f6806b2198f1f199b994` |
| 31 | qwen-image-technical-report | layout_region | 6 | `visual-retrieval-hint:qwen-image-technical-report:layout_region:6:7bdf1c8c5e3b3e5d` | `True` | `sha256:5713e526f7d6c7c82b551feb916c355b369c57f0e34491ee3786f3157dcb1b82` |
| 32 | photorealistic-text-to-image-diffusion-models-with-deep-language-understanding | layout_region | 1 | `visual-retrieval-hint:photorealistic-text-to-image-diffusion-models-with-deep-language-understanding:layout_region:1:b247ad4c2d2932bd` | `True` | `sha256:dfbe43a074adbdc7becba7e871bba82236237461dbbea9727e057c79b1121698` |
| 33 | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | image_region | 4 | `visual-retrieval-hint:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:image_region:4:6be0d4c55233b64f` | `True` | `sha256:3b175f1c1a798c32640e39a8a57a02c69b2709f578a8d4f1419818e6bb19535e` |
| 34 | emu3.5-native-multimodal-models-are-world-learners | image_region | 1 | `visual-retrieval-hint:emu3.5-native-multimodal-models-are-world-learners:image_region:1:05c8847d022a65cf` | `True` | `sha256:e6959e39a647786a84b8c39f55098738dc3868831247d726a743a3ff3a1b396b` |
| 35 | faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks | image_region | 2 | `visual-retrieval-hint:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:image_region:2:d18da95efae5b2ad` | `True` | `sha256:e8ffb46c00996c573ff727a8bc8acb91d0df8b178ebf892a738650535a1b8348` |
| 36 | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | image_region | 1 | `visual-retrieval-hint:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:image_region:1:48ca8c4f3dc9c536` | `True` | `sha256:5342263cc3e4856e32e7ccb1845da10cc18b4340d14a429e60f3adeba11a78d5` |
| 37 | arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts | image_region | 1 | `visual-retrieval-hint:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:image_region:1:180d76ce9c7e2a8a` | `True` | `sha256:b6abf5d0abb4b8dfd6e0f67978cee1b085dd4ce83ca009d2059905193c205692` |
| 38 | high-resolution-image-synthesis-with-latent-diffusion-models | image_region | 1 | `visual-retrieval-hint:high-resolution-image-synthesis-with-latent-diffusion-models:image_region:1:f5c7531523c94f16` | `True` | `sha256:0fdeb5695281a6433dc05045776f0cf76ad18c487d18ff0249d7e9d28615146f` |
| 39 | qwen-image-technical-report | image_region | 1 | `visual-retrieval-hint:qwen-image-technical-report:image_region:1:c0a6a36c3296a60c` | `True` | `sha256:9575301c962ab4a8581d99d99619e8ba8018c46e2fde17e53265f4a451394133` |
| 40 | photorealistic-text-to-image-diffusion-models-with-deep-language-understanding | image_region | 2 | `visual-retrieval-hint:photorealistic-text-to-image-diffusion-models-with-deep-language-understanding:image_region:2:2cc16bd6666b95b0` | `True` | `sha256:4c221d855c1fd864ec17816e37c9756170ca17c10618bc2ff865f45b219f67b1` |

## Warnings

- `This dry-run previews future JSONL records but writes no candidate store.`
- `All rows remain unindexed and not runtime-visible after dry-run.`
- `A separate apply tranche is required before any store mutation, and a later gate is required before indexing.`
