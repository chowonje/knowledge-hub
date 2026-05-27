# Visual Annotation Expansion Web Output 002 Validation

- schema: `knowledge-hub.paper.visual-annotation-expansion-web-output-validation.v1`
- status: `ready`
- decision: `ready_for_visual_retrieval_hint_candidate_store_expansion_design`
- generatedAt: `2026-05-27T07:44:05Z`
- sourceOutput: `eval/knowledgeos/reports/visual_annotation_expansion_web_output_004.manual.json`
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
| 1 | clip-2021 | table_region | 21 | `visual-layout:clip-2021:table_region:21:badaf33f92dc4c28` | `image_attached` | CLIP, FairFace, Table 5, gender classification, race category, Zero-Shot CLIP |
| 2 | mae-2021 | table_region | 8 | `visual-layout:mae-2021:table_region:8:71f56f58347cc664` | `image_attached` | MAE, Table 5, Table 6, Table 7, ADE20K semantic segmentation, transfer learning accuracy |
| 3 | resnet-2015 | table_region | 7 | `visual-layout:resnet-2015:table_region:7:7e079100aed8aaf6` | `image_attached` | ResNet, CIFAR-10, Table 6, classification error, data augmentation, ResNet-110 |
| 4 | clip-2021 | table_region | 22 | `visual-layout:clip-2021:table_region:22:cd7ebed3203dab3d` | `image_attached` | CLIP, FairFace, Table 7, crime-related categories, non-human categories, child category |
| 5 | mae-2021 | table_region | 8 | `visual-layout:mae-2021:table_region:8:a31e2f20793cd157` | `image_attached` | MAE, Table 6, Table 7, transfer learning accuracy, iNat, Places365 |
| 6 | resnet-2015 | table_region | 8 | `visual-layout:resnet-2015:table_region:8:499a16649cac8d1b` | `image_attached` | ResNet, object detection, PASCAL VOC, MS COCO, Table 7, Table 8 |
| 7 | clip-2021 | table_region | 25 | `visual-layout:clip-2021:table_region:25:795f0bd1367145d9` | `image_attached` | CLIP, CelebA, Table 8, zero-shot, top-1 identity recognition, celebrity recognition |
| 8 | mae-2021 | table_region | 11 | `visual-layout:mae-2021:table_region:11:229ba015a36e1bc6` | `image_attached` | MAE, Table 8, Table 9, pre-training setting, end-to-end fine-tuning, AdamW |
| 9 | clip-2021 | figure_caption_region | 10 | `visual-layout:clip-2021:figure_caption_region:10:f7695e0650ad9755` | `image_attached` | Figure 7, data efficiency, zero-shot transfer, labeled examples per class, CLIP, FER2013 |
| 10 | mae-2021 | figure_caption_region | 6 | `visual-layout:mae-2021:figure_caption_region:6:fb84bd6d006dd177` | `image_attached` | Figure 6, mask sampling strategies, random 75%, block 50%, grid 75%, MAE |
| 11 | clip-2021 | figure_caption_region | 11 | `visual-layout:clip-2021:figure_caption_region:11:d9042e6c82546c0e` | `image_attached` | Figure 9, zero-shot CLIP performance, model compute, Model GFLOPs, Error (%), RN50 |
| 12 | mae-2021 | figure_caption_region | 7 | `visual-layout:mae-2021:figure_caption_region:7:5e6020a5e57aa862` | `image_attached` | Figure 9, partial fine-tuning, ViT-L, fine-tuned Transformer blocks, MAE baseline, MoCo v3 |
| 13 | clip-2021 | figure_caption_region | 12 | `visual-layout:clip-2021:figure_caption_region:12:36479d77a4a94f87` | `image_attached` | Figure 10, linear probe performance, CLIP models, state-of-the-art computer vision models, Average Score (%), Forward-pass GFLOPs/image |
| 14 | mae-2021 | figure_caption_region | 13 | `visual-layout:mae-2021:figure_caption_region:13:d74e50f7fc0acd19` | `image_attached` | Figure 10, uncurated random samples, ImageNet validation images, masked image, MAE reconstruction, ground-truth |
| 15 | clip-2021 | figure_caption_region | 13 | `visual-layout:clip-2021:figure_caption_region:13:c57a4c0a359c0b1e` | `image_attached` | Figure 11, CLIP features, EfficientNet L2 NS, Logistic Regression, Delta Score, Noisy Student EfficientNet-L2 |
| 16 | mae-2021 | figure_caption_region | 14 | `visual-layout:mae-2021:figure_caption_region:14:78c3aa6879bfbb7e` | `image_attached` | Figure 11, uncurated random samples, COCO validation images, MAE trained on ImageNet, masked image, MAE reconstruction |
| 17 | clip-2021 | equation_region | 15 | `visual-layout:clip-2021:equation_region:15:36e881ed149ab580` | `image_attached` | CLIP, natural distribution shift, Zero-Shot CLIP, standard ImageNet training, existing robustness techniques, ImageNetV2 |
| 18 | clip-2021 | equation_region | 16 | `visual-layout:clip-2021:equation_region:16:ae86cff7e225293a` | `image_attached` | CLIP, ImageNet adaptation, effective robustness, zero-shot ImageNet, logistic regression CLIP, ObjectNet |
| 19 | clip-2021 | equation_region | 17 | `visual-layout:clip-2021:equation_region:17:65b08b3d7099f2e0` | `image_attached` | CLIP, effective robustness, Few-Shot CLIP, Zero-Shot CLIP, ImageNet models, robustness intervention |
| 20 | clip-2021 | equation_region | 19 | `visual-layout:clip-2021:equation_region:19:95c566a959a29f80` | `image_attached` | CLIP, Birdsnap, Country211, p-value, significance, p < 0.05 |
| 21 | clip-2021 | layout_region | 16 | `visual-layout:clip-2021:layout_region:16:0850e569d769717b` | `image_attached` | CLIP, ImageNet adaptation, zero-shot ImageNet, ObjectNet, ImageNet-R, Youtube-BB |
| 22 | mae-2021 | layout_region | 12 | `visual-layout:mae-2021:layout_region:12:c7e7678d2ec73caa` | `image_attached` | MAE, masked encoding methods, linear probing, iGPT, BEiT, ViT-B |
| 23 | alexnet-2012 | layout_region | 1 | `visual-layout:alexnet-2012:layout_region:1:e3b7474586959627` | `image_attached` | Neural Networks, Ilya Sutskever, University of Toronto, ImageNet LSVRC-2010, top-1, top-5 |
| 24 | resnet-2015 | layout_region | 6 | `visual-layout:resnet-2015:layout_region:6:99aa935003a72db9` | `image_attached` | ResNet, ImageNet validation, error rates, 10-crop testing, top-1 err, top-5 err |

## Warnings

- `derivedTextForRetrieval is accepted only as a retrieval hint candidate.`
- `No visual annotation row is promoted to strict evidence or citation-grade evidence.`
- `The next tranche must design an expansion candidate store before any vectorization decision.`
