# Visual Annotation Expansion Web Output 002 Validation

- schema: `knowledge-hub.paper.visual-annotation-expansion-web-output-validation.v1`
- status: `ready`
- decision: `ready_for_visual_retrieval_hint_candidate_store_expansion_design`
- generatedAt: `2026-05-27T03:16:39Z`
- sourceOutput: `eval/knowledgeos/reports/visual_annotation_expansion_web_output_003.manual.json`
- sourcePackRows: `24`
- outputRows: `24`
- matchedRows: `24`
- blockedRows: `0`
- privatePathLeakRows: `0`

## Mutation Guarantees

- writes: `report_only`
- apiCalls: `False`
- modelCalls: `False`
- webModelCalls: `False`
- manualWebModelOutputRows: `24`
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
| 1 | clip-2021 | table_region | 21 | `visual-layout:clip-2021:table_region:21:4bad77d0856486fe` | `image_attached` | CLIP, FairFace, Non-White, Race Gender Age classification, Linear Probe CLIP, Zero-Shot CLIP |
| 2 | mae-2021 | table_region | 7 | `visual-layout:mae-2021:table_region:7:8f36af5d66d8b78e` | `image_attached` | MAE, ImageNet-1K, ViT-B, ViT-L, ViT-H, DINO |
| 3 | resnet-2015 | table_region | 6 | `visual-layout:resnet-2015:table_region:6:b1729ae3f7fd50f8` | `image_attached` | ResNet, ImageNet validation, 10-crop testing, single-model results, top-1 error, top-5 error |
| 4 | clip-2021 | table_region | 21 | `visual-layout:clip-2021:table_region:21:ba742e93b9b8eb98` | `image_attached` | CLIP, FairFace, White, Race Gender Age classification, Linear Probe CLIP, Zero-Shot CLIP |
| 5 | mae-2021 | table_region | 8 | `visual-layout:mae-2021:table_region:8:33d1507d0cd15829` | `image_attached` | MAE, COCO, object detection, segmentation, Mask R-CNN, APbox |
| 6 | resnet-2015 | table_region | 6 | `visual-layout:resnet-2015:table_region:6:e1c63e0b5921b1e5` | `image_attached` | ResNet, ILSVRC, ImageNet, ensemble, top-5 error, test set |
| 7 | clip-2021 | table_region | 22 | `visual-layout:clip-2021:table_region:22:71b8cf43d8079375` | `image_attached` | CLIP, FairFace, crime-related categories, non-human categories, race category, age category |
| 8 | mae-2021 | table_region | 8 | `visual-layout:mae-2021:table_region:8:5f83ffc6688fce4b` | `image_attached` | MAE, pixels vs tokens, dVAE token, reconstruction target, IN1K, COCO |
| 9 | clip-2021 | figure_caption_region | 7 | `visual-layout:clip-2021:figure_caption_region:7:d3e128cd03f8c6bd` | `image_attached` | CLIP, zero-shot performance, prompt engineering, ensembling, contextless class names, Model GFLOPs |
| 10 | mae-2021 | figure_caption_region | 3 | `visual-layout:mae-2021:figure_caption_region:3:534077a253e5a0fa` | `image_attached` | MAE, masked autoencoder, ImageNet validation, reconstructions, mask 75%, mask 85% |
| 11 | resnet-2015 | figure_caption_region | 8 | `visual-layout:resnet-2015:figure_caption_region:8:ddd82bd40f652201` | `image_attached` | ResNet, CIFAR-10, training error, testing error, plain networks, residual networks |
| 12 | clip-2021 | figure_caption_region | 8 | `visual-layout:clip-2021:figure_caption_region:8:75613ea3ae6b4e06` | `image_attached` | CLIP, zero-shot CLIP, linear probe, ResNet50, Delta Score, supervised baseline |
| 13 | mae-2021 | figure_caption_region | 4 | `visual-layout:mae-2021:figure_caption_region:4:5260bb56bfc85410` | `image_attached` | MAE, masking ratio, fine-tuning, linear probing, ImageNet-1K validation accuracy, 75% masking |
| 14 | clip-2021 | figure_caption_region | 9 | `visual-layout:clip-2021:figure_caption_region:9:37cc4e2380f5208f` | `image_attached` | CLIP, zero-shot CLIP, few-shot linear probes, Linear Probe CLIP, BiT-M, ImageNet-21K |
| 15 | mae-2021 | figure_caption_region | 6 | `visual-layout:mae-2021:figure_caption_region:6:dc979079d8bc6a42` | `image_attached` | MAE, training schedules, epochs log-scale, fine-tuning, linear probing, ViT-L |
| 16 | clip-2021 | figure_caption_region | 10 | `visual-layout:clip-2021:figure_caption_region:10:967a1812d18f9d62` | `image_attached` | CLIP, zero-shot performance, linear probe performance, correlation, r = 0.82, sub-optimal |
| 17 | clip-2021 | equation_region | 5 | `visual-layout:clip-2021:equation_region:5:6936e394c7364af3` | `image_attached` | CLIP Figure 3, NumPy-like pseudocode, image_encoder, text_encoder, joint multimodal embedding, scaled pairwise cosine similarities |
| 18 | clip-2021 | equation_region | 5 | `visual-layout:clip-2021:equation_region:5:7b890343228eec5b` | `image_attached` | CLIP pseudocode, aligned images, aligned texts, learned temperature parameter, l2_normalize, np.dot |
| 19 | clip-2021 | equation_region | 10 | `visual-layout:clip-2021:equation_region:10:b13e1298e1e46bb8` | `image_attached` | CLIP Performance, r = 0.82, zero-shot, linear, fully supervised, HatefulMemes |
| 20 | clip-2021 | equation_region | 10 | `visual-layout:clip-2021:equation_region:10:db8ab5cab587035e` | `image_attached` | dashed y = x line, optimal zero-shot classifier, fully supervised equivalent, zero-shot classifiers, 10% to 25%, task-learning |
| 21 | clip-2021 | layout_region | 1 | `visual-layout:clip-2021:layout_region:1:7614f0e5b3451acd` | `image_attached` | CLIP first page, Learning Transferable Visual, Radford, Jong Wook Kim, Amanda Askell, Abstract |
| 22 | mae-2021 | layout_region | 1 | `visual-layout:mae-2021:layout_region:1:aadf3033f0ed984e` | `image_attached` | MAE first page, Masked Autoenco, Kaiming He, Xinlei Chen, Abstract, masked autoencoders |
| 23 | alexnet-2012 | layout_region | 1 | `visual-layout:alexnet-2012:layout_region:1:0a3974d993387e3b` | `image_attached` | AlexNet first page, 1 Introduction, neural network, five convolutional layers, three fully connected, ILSVRC-2012 |
| 24 | resnet-2015 | layout_region | 1 | `visual-layout:resnet-2015:layout_region:1:1442b7b9fe2f781a` | `image_attached` | ResNet first page, Deep Residual, Kaiming He, Abstract, residual learning framework, substantially deeper |

## Warnings

- `derivedTextForRetrieval is accepted only as a retrieval hint candidate.`
- `No visual annotation row is promoted to strict evidence or citation-grade evidence.`
- `The next tranche must design an expansion candidate store before any vectorization decision.`
