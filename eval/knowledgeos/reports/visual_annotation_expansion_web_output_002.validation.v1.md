# Visual Annotation Expansion Web Output 002 Validation

- schema: `knowledge-hub.paper.visual-annotation-expansion-web-output-validation.v1`
- status: `ready`
- decision: `ready_for_visual_retrieval_hint_candidate_store_expansion_design`
- generatedAt: `2026-05-26T15:32:21Z`
- sourceOutput: `eval/knowledgeos/reports/visual_annotation_expansion_web_output_002.manual.json`
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
| 1 | clip-2021 | image_region | 2 | `visual-layout:clip-2021:image_region:2:2616032b2a9d352d` | `image_attached` | CLIP Figure 1, Create dataset classifier from labels, A photo of a {object}, zero-shot prediction, Image Encoder, plane car dog bird |
| 2 | mae-2021 | image_region | 1 | `visual-layout:mae-2021:image_region:1:258a60cc216d8130` | `image_attached` | MAE architecture, masked input patches, encoder, input, Figure 1, visible patches |
| 3 | alexnet-2012 | image_region | 6 | `visual-layout:alexnet-2012:image_region:6:508eb98021de0fd7` | `image_attached` | AlexNet Figure 3, 96 convolutional kernels, 11×11×3, first convolutional layer, 224×224×3 input images, GPU 1 GPU 2 |
| 4 | clip-2021 | image_region | 2 | `visual-layout:clip-2021:image_region:2:2642c8db8bd29d0a` | `image_attached` | CLIP Figure 1, contrastive pre-training, Pepper the aussie pup, Text Encoder, Image Encoder, image text pairs |
| 5 | mae-2021 | image_region | 1 | `visual-layout:mae-2021:image_region:1:40b27e8b4073d98f` | `image_attached` | MAE Figure 1, Our MAE architecture, masked image patches, input, encoder, visible patches |
| 6 | alexnet-2012 | image_region | 8 | `visual-layout:alexnet-2012:image_region:8:219e7594a10f3a3f` | `image_attached` | AlexNet Figure 4, ILSVRC-2010 test images, top-5 predictions, red bar, nearest training images, Euclidean distance |
| 7 | clip-2021 | image_region | 15 | `visual-layout:clip-2021:image_region:15:1da1f1cf8ce81bd5` | `image_attached` | Language Supervision, Dataset Example, distribution shift, ImageNet, ImageNetV2, ImageNet-R |
| 8 | mae-2021 | image_region | 2 | `visual-layout:mae-2021:image_region:2:0b66280633a636cc` | `image_attached` | MAE reconstruction examples, masked image, MAE reconstruction, 96 patches, appendix examples, visible patches |
| 9 | clip-2021 | figure_caption_region | 2 | `visual-layout:clip-2021:figure_caption_region:2:b354160446b3a89c` | `image_attached` | CLIP, contrastive pre-training, zero-shot prediction, text encoder, image encoder, dataset classifier |
| 10 | mae-2021 | figure_caption_region | 1 | `visual-layout:mae-2021:figure_caption_region:1:f3477cc8ce7ec8b1` | `image_attached` | MAE, masked autoencoder, architecture, masked patches, visible patches, encoder |
| 11 | resnet-2015 | figure_caption_region | 6 | `visual-layout:resnet-2015:figure_caption_region:6:6b3dd4d05dc69888` | `image_attached` | ResNet, residual function, ImageNet, building block, bottleneck, ResNet-34 |
| 12 | clip-2021 | figure_caption_region | 3 | `visual-layout:clip-2021:figure_caption_region:3:fd2db71adcd98ec4` | `image_attached` | CLIP, zero-shot transfer, ImageNet accuracy, images processed, Bag of Words Contrastive, Bag of Words Prediction |
| 13 | mae-2021 | figure_caption_region | 2 | `visual-layout:mae-2021:figure_caption_region:2:460a5652014c6369` | `image_attached` | MAE, ImageNet validation, masked image, reconstruction, ground-truth, 80% masking ratio |
| 14 | resnet-2015 | figure_caption_region | 8 | `visual-layout:resnet-2015:figure_caption_region:8:5cf998f6a7a3739f` | `image_attached` | ResNet, CIFAR-10, training error, testing error, plain networks, layer responses |
| 15 | clip-2021 | figure_caption_region | 5 | `visual-layout:clip-2021:figure_caption_region:5:87777b9ff7911bbe` | `image_attached` | CLIP, Numpy-like pseudocode, image_encoder, text_encoder, aligned images, aligned texts |
| 16 | mae-2021 | figure_caption_region | 2 | `visual-layout:mae-2021:figure_caption_region:2:b35cefa766216e3f` | `image_attached` | MAE, COCO validation, ImageNet validation, masked image, reconstruction, ground truth |
| 17 | clip-2021 | table_region | 7 | `visual-layout:clip-2021:table_region:7:74d3127c812d1e9b` | `image_attached` | CLIP, Visual N-Grams, zero-shot transfer, aYahoo, ImageNet, SUN |
| 18 | mae-2021 | table_region | 5 | `visual-layout:mae-2021:table_region:5:09cee49d2841b6a3` | `image_attached` | MAE, masked autoencoder, Table 1, ablation experiments, ViT-L/16, ImageNet-1K |
| 19 | resnet-2015 | table_region | 6 | `visual-layout:resnet-2015:table_region:6:8b15bd4cbaa96ede` | `image_attached` | ResNet, Table 5, ImageNet, top-5 error, ensembles, ILSVRC |
| 20 | clip-2021 | table_region | 17 | `visual-layout:clip-2021:table_region:17:21f8887ab432c389` | `image_attached` | CLIP, Table 2, Oxford IIT Pets, human performance, zero-shot CLIP, zero-shot human |
| 21 | mae-2021 | table_region | 5 | `visual-layout:mae-2021:table_region:5:1179c62488dbb733` | `image_attached` | MAE, Table 2, wall-clock time, mask token, ViT-L, ViT-H |
| 22 | clip-2021 | equation_region | 1 | `visual-layout:clip-2021:equation_region:1:dd31b4b813b73af2` | `image_attached` | CLIP, Introduction and Motivating Work, pre-training, raw text, NLP, OpenAI |
| 23 | clip-2021 | equation_region | 5 | `visual-layout:clip-2021:equation_region:5:1c4738912cb4bdf9` | `image_attached` | CLIP, Figure 3, Numpy-like pseudocode, learned temperature parameter, image_encoder, text_encoder |
| 24 | clip-2021 | equation_region | 5 | `visual-layout:clip-2021:equation_region:5:1fa7893d5e5d7b10` | `image_attached` | CLIP, Figure 3, contrastive loss, image_encoder, text_encoder, ResNet |

## Warnings

- `derivedTextForRetrieval is accepted only as a retrieval hint candidate.`
- `No visual annotation row is promoted to strict evidence or citation-grade evidence.`
- `The next tranche must design an expansion candidate store before any vectorization decision.`
