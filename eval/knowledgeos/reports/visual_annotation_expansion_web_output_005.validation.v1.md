# Visual Annotation Expansion Web Output 002 Validation

- schema: `knowledge-hub.paper.visual-annotation-expansion-web-output-validation.v1`
- status: `ready`
- decision: `ready_for_visual_retrieval_hint_candidate_store_expansion_design`
- generatedAt: `2026-05-27T10:23:40Z`
- sourceOutput: `eval/knowledgeos/reports/visual_annotation_expansion_web_output_005.manual.json`
- sourcePackRows: `40`
- outputRows: `40`
- matchedRows: `40`
- blockedRows: `0`
- privatePathLeakRows: `0`

## Mutation Guarantees

- writes: `report_only`
- apiCalls: `False`
- modelCalls: `False`
- webModelCalls: `False`
- manualWebModelOutputRows: `40`
- vectorIndexing: `False`
- strictEvidencePromotionRows: `0`
- runtimeAnswerVisibleExposureRows: `0`
- databaseMutationRows: `0`
- indexMutationRows: `0`
- reindexOrReembedRows: `0`
- vaultScanRows: `0`
- externalDownloadRows: `0`
- answerabilityGateBypassRows: `0`

## Captured Rows

| # | paperId | type | page | sourceCandidateId | status | keywords |
|---:|---|---|---:|---|---|---|
| 1 | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | table_region | 2 | `visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:table_region:2:43ba20bbd82c40cc` | `image_attached` | NewtonBench, scientific law discovery benchmark, memorization-free, scientific relevance, active exploration, model system |
| 2 | emu3.5-native-multimodal-models-are-world-learners | table_region | 6 | `visual-layout:emu3.5-native-multimodal-models-are-world-learners:table_region:6:7e5ddeb596848dc5` | `image_attached` | Emu3.5, model configurations, hidden size 5120, intermediate size 25600, SFT Data 150B, RL Prompts 100k |
| 3 | faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks | table_region | 6 | `visual-layout:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:table_region:6:bf4da7f980d217f7` | `image_attached` | Faster R-CNN, region proposal networks, anchor, proposal size, ZF net, s = 600 |
| 4 | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | table_region | 15 | `visual-layout:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:table_region:15:bf53567890d6c9c9` | `image_attached` | AutoGen, multi-agent systems, conversation pattern, execution-capable, human involvement, CAMEL |
| 5 | arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts | table_region | 8 | `visual-layout:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:table_region:8:96e7bcf8b6c5c548` | `image_attached` | arithmetic in the wild, omega task, addition neurons, layer 18 MLP neurons, neuron ablations, Fourier features |
| 6 | high-resolution-image-synthesis-with-latent-diffusion-models | table_region | 4 | `visual-layout:high-resolution-image-synthesis-with-latent-diffusion-models:table_region:4:cdab6c275bd25d05` | `image_attached` | latent diffusion models, LDM objective, conditioning mechanisms, conditional denoising autoencoder, cross-attention, UNet backbone |
| 7 | qwen-image-technical-report | table_region | 9 | `visual-layout:qwen-image-technical-report:table_region:9:47a89c68476bd4bb` | `image_attached` | Qwen-Image, architecture configuration, VLM, VAE, MMDiT, ViT |
| 8 | photorealistic-text-to-image-diffusion-models-with-deep-language-understanding | table_region | 7 | `visual-layout:photorealistic-text-to-image-diffusion-models-with-deep-language-understanding:table_region:7:c8163ca9120a3550` | `image_attached` | photorealistic text-to-image diffusion, COCO 256 x 256, human evaluation, Imagen, photorealism, alignment |
| 9 | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | figure_caption_region | 4 | `visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:figure_caption_region:4:92b8890073250699` | `image_attached` | NewtonBench, experimentation complexity, physical law curation, counterfactual shifts, agentic exploration, physics domains |
| 10 | emu3.5-native-multimodal-models-are-world-learners | figure_caption_region | 1 | `visual-layout:emu3.5-native-multimodal-models-are-world-learners:figure_caption_region:1:4281f17d9edca1b7` | `image_attached` | Emu3.5, image generation benchmarks, LongText-Bench, LeX-Bench, CVTG-2K, ImgEdit |
| 11 | faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks | figure_caption_region | 2 | `visual-layout:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:figure_caption_region:2:053df2c84c04975d` | `image_attached` | Faster R-CNN, region proposal networks, anchor boxes, reference boxes, image pyramids, feature maps |
| 12 | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | figure_caption_region | 1 | `visual-layout:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:figure_caption_region:1:3db7901a05fe4358` | `image_attached` | AutoGen, multi-agent conversations, agent customization, conversable agents, flexible conversation patterns, joint chat |
| 13 | arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts | figure_caption_region | 2 | `visual-layout:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:figure_caption_region:2:9768341fe65e7ef2` | `image_attached` | Llama-3.1-8B, cyclic concepts, base-10 addition, Fourier number space, period neurons, month arithmetic |
| 14 | high-resolution-image-synthesis-with-latent-diffusion-models | figure_caption_region | 1 | `visual-layout:high-resolution-image-synthesis-with-latent-diffusion-models:figure_caption_region:1:bd010932c7e494a5` | `image_attached` | Latent Diffusion Models, less aggressive downsampling, autoencoding models, DALL-E, VQGAN, PSNR |
| 15 | qwen-image-technical-report | figure_caption_region | 2 | `visual-layout:qwen-image-technical-report:figure_caption_region:2:a2a6bdca9d8f65d3` | `image_attached` | Qwen-Image, complex text rendering, multi-line layouts, paragraph-level semantics, fine-grained details, alphabetic languages |
| 16 | photorealistic-text-to-image-diffusion-models-with-deep-language-understanding | figure_caption_region | 2 | `visual-layout:photorealistic-text-to-image-diffusion-models-with-deep-language-understanding:figure_caption_region:2:16dc56c857fd0641` | `image_attached` | Imagen, photorealistic text-to-image, 1024 x 1024 samples, deep language understanding, rocketship moon, dragon fruit karate belt |
| 17 | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | equation_region | 4 | `visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:equation_region:4:803dd3f7dbd45c6f` | `image_attached` | NewtonBench, agentic exploration, scientific law discovery, code interpreter tool-use, final law submission, ENV |
| 18 | emu3.5-native-multimodal-models-are-world-learners | equation_region | 2 | `visual-layout:emu3.5-native-multimodal-models-are-world-learners:equation_region:2:688127b9a1184640` | `image_attached` | Emu3.5, native multimodal models, world learners, sticky note, Meeting at 2pm, third-person camera |
| 19 | faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks | equation_region | 5 | `visual-layout:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:equation_region:5:128a0a411f67a324` | `image_attached` | Faster R-CNN, bounding box regression, parameterizations, anchor box, ground-truth box, predicted box |
| 20 | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | equation_region | 4 | `visual-layout:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:equation_region:4:45200729f88c94d7` | `image_attached` | AutoGen, ConversableAgent, AssistantAgent, UserProxyAgent, GroupChatManager, Agent Customization |
| 21 | arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts | equation_region | 3 | `visual-layout:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:equation_region:3:f2208953da94826f` | `image_attached` | Arithmetic in the Wild, DAS, base-10 addition, cyclic concepts, months task, interchange intervention accuracy |
| 22 | high-resolution-image-synthesis-with-latent-diffusion-models | equation_region | 1 | `visual-layout:high-resolution-image-synthesis-with-latent-diffusion-models:equation_region:1:0dc11401a800bfef` | `image_attached` | latent diffusion, CompVis, Runway ML, Dominik Lorenz, Patrick Esser, Bjorn Ommer |
| 23 | qwen-image-technical-report | equation_region | 14 | `visual-layout:qwen-image-technical-report:equation_region:14:1e8a7149fc3a29d8` | `image_attached` | Qwen-Image, text rendering, simple backgrounds, pre-defined templates, Template1, Template2 |
| 24 | photorealistic-text-to-image-diffusion-models-with-deep-language-understanding | equation_region | 20 | `visual-layout:photorealistic-text-to-image-diffusion-models-with-deep-language-understanding:equation_region:20:2eb72e2516a0abf7` | `image_attached` | photorealistic text-to-image diffusion, deep language understanding, DDIM, denoising score matching, x parameterization, z_s |
| 25 | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | layout_region | 1 | `visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:layout_region:1:45ff0082a3c9dd85` | `image_attached` | NEWTONBENCH, scientific law discovery, code interpreter, exploration to exploitation, interactive environments, large language models |
| 26 | emu3.5-native-multimodal-models-are-world-learners | layout_region | 1 | `visual-layout:emu3.5-native-multimodal-models-are-world-learners:layout_region:1:03b6139e0d9b28fb` | `image_attached` | Emu3.5, BAAI, emu.world, multimodal world model, vision-language, sequential frames |
| 27 | faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks | layout_region | 1 | `visual-layout:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:layout_region:1:f15a56f92bf8e4d7` | `image_attached` | Faster R-CNN, object detection, region proposal networks, RPN, PASCAL VOC, Fast R-CNN |
| 28 | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | layout_region | 1 | `visual-layout:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:layout_region:1:c96ad0bb30a29669` | `image_attached` | AutoGen, conversation patterns, hierarchical chat, LLM-based applications, multi-agent conversation, conversable agents |
| 29 | arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts | layout_region | 1 | `visual-layout:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:layout_region:1:9f959e82ebee205d` | `image_attached` | Llama-3.1, base-10 addition, cyclic concepts, weekdays, months after August, feature clusters |
| 30 | high-resolution-image-synthesis-with-latent-diffusion-models | layout_region | 1 | `visual-layout:high-resolution-image-synthesis-with-latent-diffusion-models:layout_region:1:47134ad1cd52c96a` | `image_attached` | High-Resolution Image Synthesis, latent diffusion models, denoising autoencoders, diffusion models, pixel space, guiding mechanism |
| 31 | qwen-image-technical-report | layout_region | 6 | `visual-layout:qwen-image-technical-report:layout_region:6:ed37c9f9cfe1ded1` | `image_attached` | Qwen-Image, technical report, Introduction, image generation models, text-to-image, modern artificial intelligence |
| 32 | photorealistic-text-to-image-diffusion-models-with-deep-language-understanding | layout_region | 1 | `visual-layout:photorealistic-text-to-image-diffusion-models-with-deep-language-understanding:layout_region:1:3d2326e60384515a` | `image_attached` | Imagen, text-to-image diffusion, COCO, DALL-E 2, imagen.research.google, multimodal learning |
| 33 | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | image_region | 4 | `visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:image_region:4:04ae14ad971d2f06` | `image_attached` | NewtonBench, Counterfactual Shifts, Equation Difficulty, Agentic Exploration, Interpreter Tool-use, Law Submission |
| 34 | emu3.5-native-multimodal-models-are-world-learners | image_region | 1 | `visual-layout:emu3.5-native-multimodal-models-are-world-learners:image_region:1:03dc21098ebf92d6` | `image_attached` | Emu3.5, BAAI, multimodal world model, Discrete Diffusion, DiDA, Nano Banana |
| 35 | faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks | image_region | 2 | `visual-layout:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:image_region:2:2ea5949dc69e0e4c` | `image_attached` | Faster R-CNN, multiple scaled images, image pyramid, object detection, classifier at all scales, reference boxes |
| 36 | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | image_region | 1 | `visual-layout:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:image_region:1:0a83cc7b8ce39324` | `image_attached` | AutoGen, Conversable agent, Agent Customization, multi-agent conversation, LLM tools, human-in-the-loop |
| 37 | arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts | image_region | 1 | `visual-layout:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:image_region:1:0792bc4980e9c1ad` | `image_attached` | Arithmetic in the Wild, Reason About, Sheridan Feucht, Usha Bhalla, Jack Merullo, Singh Lubana |
| 38 | high-resolution-image-synthesis-with-latent-diffusion-models | image_region | 1 | `visual-layout:high-resolution-image-synthesis-with-latent-diffusion-models:image_region:1:2360b13807a5f368` | `image_attached` | Latent Diffusion Models, Patrick Esser, Runway ML, DALL-E, VQGAN, PSNR |
| 39 | qwen-image-technical-report | image_region | 1 | `visual-layout:qwen-image-technical-report:image_region:1:50985e4339ab85a3` | `image_attached` | Qwen, Qwen-Image, technical report, GitHub, text rendering, image generation |
| 40 | photorealistic-text-to-image-diffusion-models-with-deep-language-understanding | image_region | 2 | `visual-layout:photorealistic-text-to-image-diffusion-models-with-deep-language-understanding:image_region:2:204aa4cc291f768f` | `image_attached` | Imagen, photorealistic text-to-image diffusion, text-to-image samples, Shiba Inu riding a bike, corgi sushi house, prompt captions |

## Warnings

- `derivedTextForRetrieval is accepted only as a retrieval hint candidate.`
- `No visual annotation row is promoted to strict evidence or citation-grade evidence.`
- `The next tranche must design an expansion candidate store before any vectorization decision.`
