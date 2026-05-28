# Visual Annotation Web Output 001 Validation

- schema: `knowledge-hub.paper.visual-annotation-web-output-validation.v1`
- status: `ready`
- decision: `ready_for_retrieval_hint_candidate_store_design`
- generatedAt: `2026-05-26T13:10:34Z`
- sourceOutput: `eval/knowledgeos/reports/visual_annotation_web_output_001.manual.json`
- sourcePackRows: `18`
- outputRows: `18`
- matchedRows: `18`
- blockedRows: `0`
- privatePathLeakRows: `0`

## Mutation Guarantees

- writes: `report_only`
- apiCalls: `False`
- modelCalls: `False`
- webModelCalls: `False`
- manualWebModelOutputRows: `18`
- vectorIndexing: `False`
- strictEvidencePromotionRows: `0`
- runtimeAnswerVisibleExposureRows: `0`
- databaseMutationRows: `0`
- indexMutationRows: `0`
- reindexOrReembedRows: `0`
- vaultScanRows: `0`
- externalDownloadRows: `0`
- answerabilityGateBypassRows: `0`
- cropWriteRows: `0`

## Captured Rows

| # | paperId | type | page | sourceCandidateId | status | keywords |
|---:|---|---|---:|---|---|---|
| 1 | alexnet-2012 | figure_caption_region | 3 | `visual-layout:alexnet-2012:figure_caption_region:3:5e4d346d4c7b2c53` | `image_attached` | AlexNet, Figure 1, ReLU nonlinearity, CIFAR-10, training error, epochs |
| 2 | alexnet-2012 | equation_region | 4 | `visual-layout:alexnet-2012:equation_region:4:f9d505b3e6aabed6` | `image_attached` | AlexNet, page 4, equation, local response normalization, LRN, response-normalized activity |
| 3 | alexnet-2012 | equation_region | 4 | `visual-layout:alexnet-2012:equation_region:4:79cf21420d190e08` | `image_attached` | AlexNet, page 4, local normalization, summation bounds, min N, max j |
| 4 | alexnet-2012 | figure_caption_region | 5 | `visual-layout:alexnet-2012:figure_caption_region:5:03c450b40ade0263` | `image_attached` | AlexNet, Figure 2, CNN architecture, two GPUs, GPU split, convolutional layers |
| 5 | alexnet-2012 | figure_caption_region | 6 | `visual-layout:alexnet-2012:figure_caption_region:6:f8996246c800f4f1` | `image_attached` | AlexNet, Figure 3, convolutional kernels, first layer filters, 11x11x3, 224x224x3 |
| 6 | alexnet-2012 | equation_region | 6 | `visual-layout:alexnet-2012:equation_region:6:81ead85a04bce51a` | `image_attached` | AlexNet, page 6, PCA color augmentation, RGB pixel, eigenvectors, eigenvalues |
| 7 | alexnet-2012 | table_region | 7 | `visual-layout:alexnet-2012:table_region:7:0777b7d11b932456` | `image_attached` | AlexNet, Table 2, ILSVRC-2012, validation error, test error, Top-1 val |
| 8 | alexnet-2012 | table_region | 7 | `visual-layout:alexnet-2012:table_region:7:b9f742e1da360378` | `image_attached` | AlexNet, Table 1, ILSVRC-2010, test set, Top-1, Top-5 |
| 9 | alexnet-2012 | figure_caption_region | 8 | `visual-layout:alexnet-2012:figure_caption_region:8:889b5135e8e9ab9e` | `image_attached` | AlexNet, Figure 4, ILSVRC-2010, test images, top five labels, red bar |
| 10 | resnet-2015 | figure_caption_region | 1 | `visual-layout:resnet-2015:figure_caption_region:1:aba2bf94ef1625a2` | `image_attached` | ResNet, Figure 1, CIFAR-10, training error, test error, 20-layer |
| 11 | resnet-2015 | figure_caption_region | 2 | `visual-layout:resnet-2015:figure_caption_region:2:9687efdba9452012` | `image_attached` | ResNet, Figure 2, residual learning, building block, identity shortcut, weight layer |
| 12 | resnet-2015 | equation_region | 3 | `visual-layout:resnet-2015:equation_region:3:0e80570b407b6793` | `image_attached` | ResNet, Identity Mapping by Shortcuts, equation 1, y equals F plus x, residual mapping, x and y |
| 13 | resnet-2015 | equation_region | 3 | `visual-layout:resnet-2015:equation_region:3:69248b1db8c80503` | `image_attached` | ResNet, equation 2, projection shortcut, W_s x, dimension matching, shortcut connections |
| 14 | resnet-2015 | equation_region | 3 | `visual-layout:resnet-2015:equation_region:3:7068ad040e2d0243` | `image_attached` | ResNet, equation 1, residual mapping, Identity Mapping by Shortcuts, x and y, W2 sigma W1x |
| 15 | resnet-2015 | figure_caption_region | 4 | `visual-layout:resnet-2015:figure_caption_region:4:0b3754b79959d64e` | `image_attached` | ResNet, Figure 3, ImageNet architectures, VGG-19, 34-layer plain, 34-layer residual |
| 16 | resnet-2015 | figure_caption_region | 5 | `visual-layout:resnet-2015:figure_caption_region:5:666c24606607fbbf` | `image_attached` | ResNet, Figure 4, ImageNet training, plain-18, plain-34, ResNet-18 |
| 17 | resnet-2015 | table_region | 5 | `visual-layout:resnet-2015:table_region:5:749ede3c6c93e4fa` | `image_attached` | ResNet, Table 1, ImageNet architectures, conv3_x, conv4_x, conv5_x |
| 18 | resnet-2015 | table_region | 5 | `visual-layout:resnet-2015:table_region:5:6f67c711d387611a` | `image_attached` | ResNet, Table 2, Top-1 error, ImageNet validation, 10-crop testing, plain |

## Warnings

- `derivedTextForRetrieval is accepted only as a retrieval hint candidate.`
- `No visual annotation row is promoted to strict evidence or citation-grade evidence.`
- `The next tranche must design a candidate store before any vectorization decision.`
