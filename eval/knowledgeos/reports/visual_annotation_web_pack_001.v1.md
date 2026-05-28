# Visual Annotation Web Pack 001

- schema: `knowledge-hub.paper.visual-annotation-web-pack.v1`
- status: `ready`
- decision: `ready_for_manual_web_vlm_calibration`
- generatedAt: `2026-05-26T12:11:01Z`
- packId: `visual_annotation_web_pack_001`
- sourceReport: `eval/knowledgeos/reports/visual_layout_candidate_list_report.v1.json`
- selectedCandidateRows: `18`
- selectedPaperRows: `2`
- privatePathLeakRows: `0`

## Mutation Guarantees

- writes: `report_only`
- apiCalls: `False`
- modelCalls: `False`
- webModelCalls: `False`
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

## Copy-Paste Prompt

### System

```text
You are generating visual/layout retrieval hints only.
Do not create citation-grade evidence.
Do not answer scientific questions.
Do not infer facts not visible in the attached image or supplied nearby text.
Do not mark anything as strict evidence.
Return only schema-valid JSON.
If an image/page crop is not attached, set visualObservationStatus to image_not_attached.
The output field derivedTextForRetrieval is retrieval_hint_only and must not be used as answer evidence.
```

### User

```text
For each candidate row in this pack, produce one output row.
Describe only visible layout/object/table/figure/equation structure, readable visible text, retrieval keywords, uncertainty, and limitations.
Keep strictEvidence=false, citationGrade=false, and answerableWithoutTextEvidence=false for every row.
Return JSON with top-level schema knowledge-hub.paper.visual-annotation-web-output.v1 and a rows array.
```

### Output Shape

```json
{
  "schema": "knowledge-hub.paper.visual-annotation-web-output.v1",
  "rows": [
    {
      "sourceCandidateId": "string",
      "visualObservationStatus": "image_attached | image_not_attached | unclear",
      "derivedTextForRetrieval": "string",
      "visibleText": "string",
      "retrievalKeywords": [
        "string"
      ],
      "uncertainty": "string",
      "limitations": "string",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    }
  ]
}
```

## Candidate Rows

| # | paperId | type | page | bbox | plannedAttachmentRef | caption/context |
|---:|---|---|---:|---|---|---|
| 1 | alexnet-2012 | figure_caption_region | 3 | `[333.92, 272.94, 504.0, 403.21]` | `papers_dir/visual_layout_planned_crops/alexnet-2012/page-3/visual-layout-alexnet-2012-figure_caption_region-3-5e4d346d4c7b2c53.png` | A four-layer convolutional neural network with ReLUs (solid line) reaches a 25% training error rate on CIFAR-10 six times faster than an equivalent network w... |
| 2 | alexnet-2012 | equation_region | 4 | `[211.09, 185.62, 262.24, 199.82]` | `papers_dir/visual_layout_planned_crops/alexnet-2012/page-4/visual-layout-alexnet-2012-equation_region-4-f9d505b3e6aabed6.png` | ReLUs have the desirable property that they do not require input normalization to prevent them from saturating. If at least some training examples produce a... |
| 3 | alexnet-2012 | equation_region | 4 | `[297.63, 185.62, 386.33, 209.08]` | `papers_dir/visual_layout_planned_crops/alexnet-2012/page-4/visual-layout-alexnet-2012-equation_region-4-79cf21420d190e08.png` | min(N−1,i+n/2) X j=max(0,i−n/2) (aj x,y)2  |
| 4 | alexnet-2012 | figure_caption_region | 5 | `[108.0, 216.17, 504.0, 269.73]` | `papers_dir/visual_layout_planned_crops/alexnet-2012/page-5/visual-layout-alexnet-2012-figure_caption_region-5-03c450b40ade0263.png` | An illustration of the architecture of our CNN, explicitly showing the delineation of responsibilities between the two GPUs. One GPU runs the layer-parts at... |
| 5 | alexnet-2012 | figure_caption_region | 6 | `[348.09, 501.02, 504.0, 565.54]` | `papers_dir/visual_layout_planned_crops/alexnet-2012/page-6/visual-layout-alexnet-2012-figure_caption_region-6-f8996246c800f4f1.png` | 96 convolutional kernels of size 11×11×3 learned by the ﬁrst convolutional layer on the 224×224×3 input images. The top 48 kernels were learned on GPU 1 whil... |
| 6 | alexnet-2012 | equation_region | 6 | `[108.0, 84.27, 504.0, 119.0]` | `papers_dir/visual_layout_planned_crops/alexnet-2012/page-6/visual-layout-alexnet-2012-equation_region-6-81ead85a04bce51a.png` | with magnitudes proportional to the corresponding eigenvalues times a random variable drawn from a Gaussian with mean zero and standard deviation 0.1. Theref... |
| 7 | alexnet-2012 | table_region | 7 | `[250.56, 511.36, 504.0, 553.96]` | `papers_dir/visual_layout_planned_crops/alexnet-2012/page-7/visual-layout-alexnet-2012-table_region-7-0777b7d11b932456.png` | Comparison of error rates on ILSVRC-2012 validation and test sets. In italics are best results achieved by others. Models with an asterisk* were “pre-trained... |
| 8 | alexnet-2012 | table_region | 7 | `[341.64, 290.39, 504.0, 322.03]` | `papers_dir/visual_layout_planned_crops/alexnet-2012/page-7/visual-layout-alexnet-2012-table_region-7-b9f742e1da360378.png` | Comparison of results on ILSVRC- 2010 test set. In italics are best results achieved by others. |
| 9 | alexnet-2012 | figure_caption_region | 8 | `[108.0, 244.05, 504.0, 297.6]` | `papers_dir/visual_layout_planned_crops/alexnet-2012/page-8/visual-layout-alexnet-2012-figure_caption_region-8-889b5135e8e9ab9e.png` | (Left) Eight ILSVRC-2010 test images and the ﬁve labels considered most probable by our model. The correct label is written under each image, and the probabi... |
| 10 | resnet-2015 | figure_caption_region | 1 | `[308.86, 304.89, 545.11, 346.73]` | `papers_dir/visual_layout_planned_crops/resnet-2015/page-1/visual-layout-resnet-2015-figure_caption_region-1-aba2bf94ef1625a2.png` | Training error (left) and test error (right) on CIFAR-10 with 20-layer and 56-layer “plain” networks. The deeper network has higher training error, and thus... |
| 11 | resnet-2015 | figure_caption_region | 2 | `[86.62, 158.15, 249.86, 167.12]` | `papers_dir/visual_layout_planned_crops/resnet-2015/page-2/visual-layout-resnet-2015-figure_caption_region-2-9687efdba9452012.png` | Residual learning: a building block. |
| 12 | resnet-2015 | equation_region | 3 | `[123.47, 626.46, 286.36, 643.81]` | `papers_dir/visual_layout_planned_crops/resnet-2015/page-3/visual-layout-resnet-2015-equation_region-3-0e80570b407b6793.png` | We adopt residual learning to every few stacked layers. A building block is shown in Fig. 2. Formally, in this paper we consider a building block deﬁned as:... |
| 13 | resnet-2015 | equation_region | 3 | `[375.38, 264.43, 545.11, 281.78]` | `papers_dir/visual_layout_planned_crops/resnet-2015/page-3/visual-layout-resnet-2015-equation_region-3-69248b1db8c80503.png` | ReLU [29] and the biases are omitted for simplifying no- tations. The operation F + x is performed by a shortcut connection and element-wise addition. We ado... |
| 14 | resnet-2015 | equation_region | 3 | `[50.11, 648.82, 286.37, 701.8]` | `papers_dir/visual_layout_planned_crops/resnet-2015/page-3/visual-layout-resnet-2015-equation_region-3-7068ad040e2d0243.png` | y = F(x, {Wi}) + x. (1) Here x and y are the input and output vectors of the lay- ers considered. The function F(x, {Wi}) represents the residual mapping to... |
| 15 | resnet-2015 | figure_caption_region | 4 | `[50.11, 632.83, 286.37, 696.68]` | `papers_dir/visual_layout_planned_crops/resnet-2015/page-4/visual-layout-resnet-2015-figure_caption_region-4-0b3754b79959d64e.png` | Example network architectures for ImageNet. Left: the VGG-19 model [41] (19.6 billion FLOPs) as a reference. Mid- dle: a plain network with 34 parameter laye... |
| 16 | resnet-2015 | figure_caption_region | 5 | `[50.11, 392.29, 545.12, 423.26]` | `papers_dir/visual_layout_planned_crops/resnet-2015/page-5/visual-layout-resnet-2015-figure_caption_region-5-666c24606607fbbf.png` | Training on ImageNet. Thin curves denote training error, and bold curves denote validation error of the center crops. Left: plain networks of 18 and 34 layer... |
| 17 | resnet-2015 | table_region | 5 | `[50.11, 224.67, 545.11, 244.6]` | `papers_dir/visual_layout_planned_crops/resnet-2015/page-5/visual-layout-resnet-2015-table_region-5-749ede3c6c93e4fa.png` | Architectures for ImageNet. Building blocks are shown in brackets (see also Fig. 5), with the numbers of blocks stacked. Down- sampling is performed by conv3... |
| 18 | resnet-2015 | table_region | 5 | `[50.11, 485.61, 286.36, 516.49]` | `papers_dir/visual_layout_planned_crops/resnet-2015/page-5/visual-layout-resnet-2015-table_region-5-6f67c711d387611a.png` | Top-1 error (%, 10-crop testing) on ImageNet validation. Here the ResNets have no extra parameter compared to their plain counterparts. Fig. 4 shows the trai... |

## Row JSON

```json
[
  {
    "schema": "knowledge-hub.paper.visual-annotation-web-pack-row.v1",
    "packCandidateId": "visual-annotation-web-pack:visual_annotation_web_pack_001:ecc446ac2dee9045",
    "sourceCandidateId": "visual-layout:alexnet-2012:figure_caption_region:3:5e4d346d4c7b2c53",
    "paperId": "alexnet-2012",
    "paperRef": "papers_dir/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf",
    "sourceContentHash": "sha256:90137160c57217953d5f61857e64ca58e85f06e1b13b4f475c918b1b582b9771",
    "page": 3,
    "bbox": [
      333.92,
      272.94,
      504.0,
      403.21
    ],
    "candidateType": "figure_caption_region",
    "priority": 1,
    "annotationTask": "describe_figure_caption_region_for_retrieval",
    "webInput": {
      "copyPasteContext": "paperId=alexnet-2012\npage=3\ncandidateType=figure_caption_region\nbbox=[333.92, 272.94, 504.0, 403.21]\ncaptionText=A four-layer convolutional neural network with ReLUs (solid line) reaches a 25% training error rate on CIFAR-10 six times faster than an equivalent network with tanh neurons (dashed line). The learning rates for each net- work were chosen independently to make train- ing as fast as possible. No regularization of any...\nheadingPath=Page 3\nnearbyText=3.1 ReLU Nonlinearity Figure 1: A four-layer convolutional neural network with ReLUs (solid line) reaches a 25% training error rate on CIFAR-10 six times faster than an equivalent network with tanh neurons (dashed line). The learning rates for each net- work were chosen independently to make train- ing as fast as possible. No regularization of any kind was employed. The magnitude of the effect demonstrated here va...",
      "attachmentInstruction": "Attach the matching page or crop image manually if available. If no image is attached, do not invent visual details; use only supplied text and bbox metadata.",
      "plannedAttachmentRef": "papers_dir/visual_layout_planned_crops/alexnet-2012/page-3/visual-layout-alexnet-2012-figure_caption_region-3-5e4d346d4c7b2c53.png",
      "pageImageRequired": true
    },
    "textContext": {
      "nearbyText": "3.1 ReLU Nonlinearity Figure 1: A four-layer convolutional neural network with ReLUs (solid line) reaches a 25% training error rate on CIFAR-10 six times faster than an equivalent network with tanh neurons (dashed line). The learning rates for each net- work were chosen independently to make train- ing as fast as possible. No regularization of any kind was employed. The magnitude of the effect demonstrated here va...",
      "captionText": "A four-layer convolutional neural network with ReLUs (solid line) reaches a 25% training error rate on CIFAR-10 six times faster than an equivalent network with tanh neurons (dashed line). The learning rates for each net- work were chosen independently to make train- ing as fast as possible. No regularization of any...",
      "headingPath": [
        "Page 3"
      ]
    },
    "visualContext": {
      "cropRef": "papers_dir/visual_layout_planned_crops/alexnet-2012/page-3/visual-layout-alexnet-2012-figure_caption_region-3-5e4d346d4c7b2c53.png",
      "imageHash": "",
      "pageImageRequired": true
    },
    "retrievalHintPlan": {
      "targetDerivedTextField": "derivedTextForRetrieval",
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "expectedOutputContract": {
      "schema": "knowledge-hub.paper.visual-annotation-web-output-row.v1",
      "requiredFields": [
        "sourceCandidateId",
        "derivedTextForRetrieval",
        "visibleText",
        "retrievalKeywords",
        "uncertainty",
        "limitations",
        "strictEvidence",
        "citationGrade",
        "answerableWithoutTextEvidence"
      ],
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "provenance": {
      "sourceReportSchema": "knowledge-hub.paper.visual-layout-candidate-list-report.v1",
      "sourceCandidateId": "visual-layout:alexnet-2012:figure_caption_region:3:5e4d346d4c7b2c53",
      "sourceContentHash": "sha256:90137160c57217953d5f61857e64ca58e85f06e1b13b4f475c918b1b582b9771",
      "page": 3,
      "bbox": [
        333.92,
        272.94,
        504.0,
        403.21
      ],
      "extractionMethod": "pymupdf_text_block_caption_regex_v1"
    },
    "blockerReason": ""
  },
  {
    "schema": "knowledge-hub.paper.visual-annotation-web-pack-row.v1",
    "packCandidateId": "visual-annotation-web-pack:visual_annotation_web_pack_001:9bf65abe7dadd15e",
    "sourceCandidateId": "visual-layout:alexnet-2012:equation_region:4:f9d505b3e6aabed6",
    "paperId": "alexnet-2012",
    "paperRef": "papers_dir/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf",
    "sourceContentHash": "sha256:90137160c57217953d5f61857e64ca58e85f06e1b13b4f475c918b1b582b9771",
    "page": 4,
    "bbox": [
      211.09,
      185.62,
      262.24,
      199.82
    ],
    "candidateType": "equation_region",
    "priority": 2,
    "annotationTask": "describe_equation_region_for_retrieval",
    "webInput": {
      "copyPasteContext": "paperId=alexnet-2012\npage=4\ncandidateType=equation_region\nbbox=[211.09, 185.62, 262.24, 199.82]\nheadingPath=Page 4\nnearbyText=ReLUs have the desirable property that they do not require input normalization to prevent them from saturating. If at least some training examples produce a positive input to a ReLU, learning will happen in that neuron. However, we still ﬁnd that the following local normalization scheme aids generalization. Denoting by ai x,y the activity of a neuron computed by applying kernel i at position (x, y) and then applyi...",
      "attachmentInstruction": "Attach the matching page or crop image manually if available. If no image is attached, do not invent visual details; use only supplied text and bbox metadata.",
      "plannedAttachmentRef": "papers_dir/visual_layout_planned_crops/alexnet-2012/page-4/visual-layout-alexnet-2012-equation_region-4-f9d505b3e6aabed6.png",
      "pageImageRequired": true
    },
    "textContext": {
      "nearbyText": "ReLUs have the desirable property that they do not require input normalization to prevent them from saturating. If at least some training examples produce a positive input to a ReLU, learning will happen in that neuron. However, we still ﬁnd that the following local normalization scheme aids generalization. Denoting by ai x,y the activity of a neuron computed by applying kernel i at position (x, y) and then applyi...",
      "captionText": "",
      "headingPath": [
        "Page 4"
      ]
    },
    "visualContext": {
      "cropRef": "papers_dir/visual_layout_planned_crops/alexnet-2012/page-4/visual-layout-alexnet-2012-equation_region-4-f9d505b3e6aabed6.png",
      "imageHash": "",
      "pageImageRequired": true
    },
    "retrievalHintPlan": {
      "targetDerivedTextField": "derivedTextForRetrieval",
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "expectedOutputContract": {
      "schema": "knowledge-hub.paper.visual-annotation-web-output-row.v1",
      "requiredFields": [
        "sourceCandidateId",
        "derivedTextForRetrieval",
        "visibleText",
        "retrievalKeywords",
        "uncertainty",
        "limitations",
        "strictEvidence",
        "citationGrade",
        "answerableWithoutTextEvidence"
      ],
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "provenance": {
      "sourceReportSchema": "knowledge-hub.paper.visual-layout-candidate-list-report.v1",
      "sourceCandidateId": "visual-layout:alexnet-2012:equation_region:4:f9d505b3e6aabed6",
      "sourceContentHash": "sha256:90137160c57217953d5f61857e64ca58e85f06e1b13b4f475c918b1b582b9771",
      "page": 4,
      "bbox": [
        211.09,
        185.62,
        262.24,
        199.82
      ],
      "extractionMethod": "pymupdf_text_block_equation_heuristic_v1"
    },
    "blockerReason": ""
  },
  {
    "schema": "knowledge-hub.paper.visual-annotation-web-pack-row.v1",
    "packCandidateId": "visual-annotation-web-pack:visual_annotation_web_pack_001:ec8063e0b7464d36",
    "sourceCandidateId": "visual-layout:alexnet-2012:equation_region:4:79cf21420d190e08",
    "paperId": "alexnet-2012",
    "paperRef": "papers_dir/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf",
    "sourceContentHash": "sha256:90137160c57217953d5f61857e64ca58e85f06e1b13b4f475c918b1b582b9771",
    "page": 4,
    "bbox": [
      297.63,
      185.62,
      386.33,
      209.08
    ],
    "candidateType": "equation_region",
    "priority": 3,
    "annotationTask": "describe_equation_region_for_retrieval",
    "webInput": {
      "copyPasteContext": "paperId=alexnet-2012\npage=4\ncandidateType=equation_region\nbbox=[297.63, 185.62, 386.33, 209.08]\nheadingPath=Page 4\nnearbyText=min(N−1,i+n/2) X j=max(0,i−n/2) (aj x,y)2 ",
      "attachmentInstruction": "Attach the matching page or crop image manually if available. If no image is attached, do not invent visual details; use only supplied text and bbox metadata.",
      "plannedAttachmentRef": "papers_dir/visual_layout_planned_crops/alexnet-2012/page-4/visual-layout-alexnet-2012-equation_region-4-79cf21420d190e08.png",
      "pageImageRequired": true
    },
    "textContext": {
      "nearbyText": "min(N−1,i+n/2) X j=max(0,i−n/2) (aj x,y)2 ",
      "captionText": "",
      "headingPath": [
        "Page 4"
      ]
    },
    "visualContext": {
      "cropRef": "papers_dir/visual_layout_planned_crops/alexnet-2012/page-4/visual-layout-alexnet-2012-equation_region-4-79cf21420d190e08.png",
      "imageHash": "",
      "pageImageRequired": true
    },
    "retrievalHintPlan": {
      "targetDerivedTextField": "derivedTextForRetrieval",
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "expectedOutputContract": {
      "schema": "knowledge-hub.paper.visual-annotation-web-output-row.v1",
      "requiredFields": [
        "sourceCandidateId",
        "derivedTextForRetrieval",
        "visibleText",
        "retrievalKeywords",
        "uncertainty",
        "limitations",
        "strictEvidence",
        "citationGrade",
        "answerableWithoutTextEvidence"
      ],
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "provenance": {
      "sourceReportSchema": "knowledge-hub.paper.visual-layout-candidate-list-report.v1",
      "sourceCandidateId": "visual-layout:alexnet-2012:equation_region:4:79cf21420d190e08",
      "sourceContentHash": "sha256:90137160c57217953d5f61857e64ca58e85f06e1b13b4f475c918b1b582b9771",
      "page": 4,
      "bbox": [
        297.63,
        185.62,
        386.33,
        209.08
      ],
      "extractionMethod": "pymupdf_text_block_equation_heuristic_v1"
    },
    "blockerReason": ""
  },
  {
    "schema": "knowledge-hub.paper.visual-annotation-web-pack-row.v1",
    "packCandidateId": "visual-annotation-web-pack:visual_annotation_web_pack_001:9b24b8f106cc4516",
    "sourceCandidateId": "visual-layout:alexnet-2012:figure_caption_region:5:03c450b40ade0263",
    "paperId": "alexnet-2012",
    "paperRef": "papers_dir/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf",
    "sourceContentHash": "sha256:90137160c57217953d5f61857e64ca58e85f06e1b13b4f475c918b1b582b9771",
    "page": 5,
    "bbox": [
      108.0,
      216.17,
      504.0,
      269.73
    ],
    "candidateType": "figure_caption_region",
    "priority": 4,
    "annotationTask": "describe_figure_caption_region_for_retrieval",
    "webInput": {
      "copyPasteContext": "paperId=alexnet-2012\npage=5\ncandidateType=figure_caption_region\nbbox=[108.0, 216.17, 504.0, 269.73]\ncaptionText=An illustration of the architecture of our CNN, explicitly showing the delineation of responsibilities between the two GPUs. One GPU runs the layer-parts at the top of the ﬁgure while the other runs the layer-parts at the bottom. The GPUs communicate only at certain layers. The network’s input is 150,528-dimensional...\nheadingPath=Page 5\nnearbyText=Figure 2: An illustration of the architecture of our CNN, explicitly showing the delineation of responsibilities between the two GPUs. One GPU runs the layer-parts at the top of the ﬁgure while the other runs the layer-parts at the bottom. The GPUs communicate only at certain layers. The network’s input is 150,528-dimensional, and the number of neurons in the network’s remaining layers is given by 253,440–186,624–...",
      "attachmentInstruction": "Attach the matching page or crop image manually if available. If no image is attached, do not invent visual details; use only supplied text and bbox metadata.",
      "plannedAttachmentRef": "papers_dir/visual_layout_planned_crops/alexnet-2012/page-5/visual-layout-alexnet-2012-figure_caption_region-5-03c450b40ade0263.png",
      "pageImageRequired": true
    },
    "textContext": {
      "nearbyText": "Figure 2: An illustration of the architecture of our CNN, explicitly showing the delineation of responsibilities between the two GPUs. One GPU runs the layer-parts at the top of the ﬁgure while the other runs the layer-parts at the bottom. The GPUs communicate only at certain layers. The network’s input is 150,528-dimensional, and the number of neurons in the network’s remaining layers is given by 253,440–186,624–...",
      "captionText": "An illustration of the architecture of our CNN, explicitly showing the delineation of responsibilities between the two GPUs. One GPU runs the layer-parts at the top of the ﬁgure while the other runs the layer-parts at the bottom. The GPUs communicate only at certain layers. The network’s input is 150,528-dimensional...",
      "headingPath": [
        "Page 5"
      ]
    },
    "visualContext": {
      "cropRef": "papers_dir/visual_layout_planned_crops/alexnet-2012/page-5/visual-layout-alexnet-2012-figure_caption_region-5-03c450b40ade0263.png",
      "imageHash": "",
      "pageImageRequired": true
    },
    "retrievalHintPlan": {
      "targetDerivedTextField": "derivedTextForRetrieval",
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "expectedOutputContract": {
      "schema": "knowledge-hub.paper.visual-annotation-web-output-row.v1",
      "requiredFields": [
        "sourceCandidateId",
        "derivedTextForRetrieval",
        "visibleText",
        "retrievalKeywords",
        "uncertainty",
        "limitations",
        "strictEvidence",
        "citationGrade",
        "answerableWithoutTextEvidence"
      ],
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "provenance": {
      "sourceReportSchema": "knowledge-hub.paper.visual-layout-candidate-list-report.v1",
      "sourceCandidateId": "visual-layout:alexnet-2012:figure_caption_region:5:03c450b40ade0263",
      "sourceContentHash": "sha256:90137160c57217953d5f61857e64ca58e85f06e1b13b4f475c918b1b582b9771",
      "page": 5,
      "bbox": [
        108.0,
        216.17,
        504.0,
        269.73
      ],
      "extractionMethod": "pymupdf_text_block_caption_regex_v1"
    },
    "blockerReason": ""
  },
  {
    "schema": "knowledge-hub.paper.visual-annotation-web-pack-row.v1",
    "packCandidateId": "visual-annotation-web-pack:visual_annotation_web_pack_001:61e72c38fc4ec26b",
    "sourceCandidateId": "visual-layout:alexnet-2012:figure_caption_region:6:f8996246c800f4f1",
    "paperId": "alexnet-2012",
    "paperRef": "papers_dir/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf",
    "sourceContentHash": "sha256:90137160c57217953d5f61857e64ca58e85f06e1b13b4f475c918b1b582b9771",
    "page": 6,
    "bbox": [
      348.09,
      501.02,
      504.0,
      565.54
    ],
    "candidateType": "figure_caption_region",
    "priority": 5,
    "annotationTask": "describe_figure_caption_region_for_retrieval",
    "webInput": {
      "copyPasteContext": "paperId=alexnet-2012\npage=6\ncandidateType=figure_caption_region\nbbox=[348.09, 501.02, 504.0, 565.54]\ncaptionText=96 convolutional kernels of size 11×11×3 learned by the ﬁrst convolutional layer on the 224×224×3 input images. The top 48 kernels were learned on GPU 1 while the bottom 48 kernels were learned on GPU 2. See Section 6.1 for details.\nheadingPath=Page 6\nnearbyText=We use dropout in the ﬁrst two fully-connected layers of Figure 2. Without dropout, our network ex- hibits substantial overﬁtting. Dropout roughly doubles the number of iterations required to converge. Figure 3: 96 convolutional kernels of size 11×11×3 learned by the ﬁrst convolutional layer on the 224×224×3 input images. The top 48 kernels were learned on GPU 1 while the bottom 48 kernels were learned on GPU 2. S...",
      "attachmentInstruction": "Attach the matching page or crop image manually if available. If no image is attached, do not invent visual details; use only supplied text and bbox metadata.",
      "plannedAttachmentRef": "papers_dir/visual_layout_planned_crops/alexnet-2012/page-6/visual-layout-alexnet-2012-figure_caption_region-6-f8996246c800f4f1.png",
      "pageImageRequired": true
    },
    "textContext": {
      "nearbyText": "We use dropout in the ﬁrst two fully-connected layers of Figure 2. Without dropout, our network ex- hibits substantial overﬁtting. Dropout roughly doubles the number of iterations required to converge. Figure 3: 96 convolutional kernels of size 11×11×3 learned by the ﬁrst convolutional layer on the 224×224×3 input images. The top 48 kernels were learned on GPU 1 while the bottom 48 kernels were learned on GPU 2. S...",
      "captionText": "96 convolutional kernels of size 11×11×3 learned by the ﬁrst convolutional layer on the 224×224×3 input images. The top 48 kernels were learned on GPU 1 while the bottom 48 kernels were learned on GPU 2. See Section 6.1 for details.",
      "headingPath": [
        "Page 6"
      ]
    },
    "visualContext": {
      "cropRef": "papers_dir/visual_layout_planned_crops/alexnet-2012/page-6/visual-layout-alexnet-2012-figure_caption_region-6-f8996246c800f4f1.png",
      "imageHash": "",
      "pageImageRequired": true
    },
    "retrievalHintPlan": {
      "targetDerivedTextField": "derivedTextForRetrieval",
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "expectedOutputContract": {
      "schema": "knowledge-hub.paper.visual-annotation-web-output-row.v1",
      "requiredFields": [
        "sourceCandidateId",
        "derivedTextForRetrieval",
        "visibleText",
        "retrievalKeywords",
        "uncertainty",
        "limitations",
        "strictEvidence",
        "citationGrade",
        "answerableWithoutTextEvidence"
      ],
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "provenance": {
      "sourceReportSchema": "knowledge-hub.paper.visual-layout-candidate-list-report.v1",
      "sourceCandidateId": "visual-layout:alexnet-2012:figure_caption_region:6:f8996246c800f4f1",
      "sourceContentHash": "sha256:90137160c57217953d5f61857e64ca58e85f06e1b13b4f475c918b1b582b9771",
      "page": 6,
      "bbox": [
        348.09,
        501.02,
        504.0,
        565.54
      ],
      "extractionMethod": "pymupdf_text_block_caption_regex_v1"
    },
    "blockerReason": ""
  },
  {
    "schema": "knowledge-hub.paper.visual-annotation-web-pack-row.v1",
    "packCandidateId": "visual-annotation-web-pack:visual_annotation_web_pack_001:38da736f7cd50f40",
    "sourceCandidateId": "visual-layout:alexnet-2012:equation_region:6:81ead85a04bce51a",
    "paperId": "alexnet-2012",
    "paperRef": "papers_dir/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf",
    "sourceContentHash": "sha256:90137160c57217953d5f61857e64ca58e85f06e1b13b4f475c918b1b582b9771",
    "page": 6,
    "bbox": [
      108.0,
      84.27,
      504.0,
      119.0
    ],
    "candidateType": "equation_region",
    "priority": 6,
    "annotationTask": "describe_equation_region_for_retrieval",
    "webInput": {
      "copyPasteContext": "paperId=alexnet-2012\npage=6\ncandidateType=equation_region\nbbox=[108.0, 84.27, 504.0, 119.0]\nheadingPath=Page 6\nnearbyText=with magnitudes proportional to the corresponding eigenvalues times a random variable drawn from a Gaussian with mean zero and standard deviation 0.1. Therefore to each RGB image pixel Ixy = [IR xy, IG xy, IB xy]T we add the following quantity: [p1, p2, p3][α1λ1, α2λ2, α3λ3]T",
      "attachmentInstruction": "Attach the matching page or crop image manually if available. If no image is attached, do not invent visual details; use only supplied text and bbox metadata.",
      "plannedAttachmentRef": "papers_dir/visual_layout_planned_crops/alexnet-2012/page-6/visual-layout-alexnet-2012-equation_region-6-81ead85a04bce51a.png",
      "pageImageRequired": true
    },
    "textContext": {
      "nearbyText": "with magnitudes proportional to the corresponding eigenvalues times a random variable drawn from a Gaussian with mean zero and standard deviation 0.1. Therefore to each RGB image pixel Ixy = [IR xy, IG xy, IB xy]T we add the following quantity: [p1, p2, p3][α1λ1, α2λ2, α3λ3]T",
      "captionText": "",
      "headingPath": [
        "Page 6"
      ]
    },
    "visualContext": {
      "cropRef": "papers_dir/visual_layout_planned_crops/alexnet-2012/page-6/visual-layout-alexnet-2012-equation_region-6-81ead85a04bce51a.png",
      "imageHash": "",
      "pageImageRequired": true
    },
    "retrievalHintPlan": {
      "targetDerivedTextField": "derivedTextForRetrieval",
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "expectedOutputContract": {
      "schema": "knowledge-hub.paper.visual-annotation-web-output-row.v1",
      "requiredFields": [
        "sourceCandidateId",
        "derivedTextForRetrieval",
        "visibleText",
        "retrievalKeywords",
        "uncertainty",
        "limitations",
        "strictEvidence",
        "citationGrade",
        "answerableWithoutTextEvidence"
      ],
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "provenance": {
      "sourceReportSchema": "knowledge-hub.paper.visual-layout-candidate-list-report.v1",
      "sourceCandidateId": "visual-layout:alexnet-2012:equation_region:6:81ead85a04bce51a",
      "sourceContentHash": "sha256:90137160c57217953d5f61857e64ca58e85f06e1b13b4f475c918b1b582b9771",
      "page": 6,
      "bbox": [
        108.0,
        84.27,
        504.0,
        119.0
      ],
      "extractionMethod": "pymupdf_text_block_equation_heuristic_v1"
    },
    "blockerReason": ""
  },
  {
    "schema": "knowledge-hub.paper.visual-annotation-web-pack-row.v1",
    "packCandidateId": "visual-annotation-web-pack:visual_annotation_web_pack_001:ce0d1c3d6093078e",
    "sourceCandidateId": "visual-layout:alexnet-2012:table_region:7:0777b7d11b932456",
    "paperId": "alexnet-2012",
    "paperRef": "papers_dir/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf",
    "sourceContentHash": "sha256:90137160c57217953d5f61857e64ca58e85f06e1b13b4f475c918b1b582b9771",
    "page": 7,
    "bbox": [
      250.56,
      511.36,
      504.0,
      553.96
    ],
    "candidateType": "table_region",
    "priority": 7,
    "annotationTask": "describe_table_region_for_retrieval",
    "webInput": {
      "copyPasteContext": "paperId=alexnet-2012\npage=7\ncandidateType=table_region\nbbox=[250.56, 511.36, 504.0, 553.96]\ncaptionText=Comparison of error rates on ILSVRC-2012 validation and test sets. In italics are best results achieved by others. Models with an asterisk* were “pre-trained” to classify the entire ImageNet 2011 Fall release. See Section 6 for details.\nheadingPath=Page 7\nnearbyText=Model Top-1 (val) Top-5 (val) Top-5 (test) SIFT + FVs [7] — — 26.2% 1 CNN 40.7% 18.2% — 5 CNNs 38.1% 16.4% 16.4% 1 CNN* 39.0% 16.6% — 7 CNNs* 36.7% 15.4% 15.3% Table 2: Comparison of error rates on ILSVRC-2012 validation and test sets. In italics are best results achieved by others. Models with an asterisk* were “pre-trained” to classify the entire ImageNet 2011 Fall release. See Section 6 for details. Finally, we...",
      "attachmentInstruction": "Attach the matching page or crop image manually if available. If no image is attached, do not invent visual details; use only supplied text and bbox metadata.",
      "plannedAttachmentRef": "papers_dir/visual_layout_planned_crops/alexnet-2012/page-7/visual-layout-alexnet-2012-table_region-7-0777b7d11b932456.png",
      "pageImageRequired": true
    },
    "textContext": {
      "nearbyText": "Model Top-1 (val) Top-5 (val) Top-5 (test) SIFT + FVs [7] — — 26.2% 1 CNN 40.7% 18.2% — 5 CNNs 38.1% 16.4% 16.4% 1 CNN* 39.0% 16.6% — 7 CNNs* 36.7% 15.4% 15.3% Table 2: Comparison of error rates on ILSVRC-2012 validation and test sets. In italics are best results achieved by others. Models with an asterisk* were “pre-trained” to classify the entire ImageNet 2011 Fall release. See Section 6 for details. Finally, we...",
      "captionText": "Comparison of error rates on ILSVRC-2012 validation and test sets. In italics are best results achieved by others. Models with an asterisk* were “pre-trained” to classify the entire ImageNet 2011 Fall release. See Section 6 for details.",
      "headingPath": [
        "Page 7"
      ]
    },
    "visualContext": {
      "cropRef": "papers_dir/visual_layout_planned_crops/alexnet-2012/page-7/visual-layout-alexnet-2012-table_region-7-0777b7d11b932456.png",
      "imageHash": "",
      "pageImageRequired": true
    },
    "retrievalHintPlan": {
      "targetDerivedTextField": "derivedTextForRetrieval",
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "expectedOutputContract": {
      "schema": "knowledge-hub.paper.visual-annotation-web-output-row.v1",
      "requiredFields": [
        "sourceCandidateId",
        "derivedTextForRetrieval",
        "visibleText",
        "retrievalKeywords",
        "uncertainty",
        "limitations",
        "strictEvidence",
        "citationGrade",
        "answerableWithoutTextEvidence"
      ],
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "provenance": {
      "sourceReportSchema": "knowledge-hub.paper.visual-layout-candidate-list-report.v1",
      "sourceCandidateId": "visual-layout:alexnet-2012:table_region:7:0777b7d11b932456",
      "sourceContentHash": "sha256:90137160c57217953d5f61857e64ca58e85f06e1b13b4f475c918b1b582b9771",
      "page": 7,
      "bbox": [
        250.56,
        511.36,
        504.0,
        553.96
      ],
      "extractionMethod": "pymupdf_text_block_caption_regex_v1"
    },
    "blockerReason": ""
  },
  {
    "schema": "knowledge-hub.paper.visual-annotation-web-pack-row.v1",
    "packCandidateId": "visual-annotation-web-pack:visual_annotation_web_pack_001:6193c06eb07d2cc6",
    "sourceCandidateId": "visual-layout:alexnet-2012:table_region:7:b9f742e1da360378",
    "paperId": "alexnet-2012",
    "paperRef": "papers_dir/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf",
    "sourceContentHash": "sha256:90137160c57217953d5f61857e64ca58e85f06e1b13b4f475c918b1b582b9771",
    "page": 7,
    "bbox": [
      341.64,
      290.39,
      504.0,
      322.03
    ],
    "candidateType": "table_region",
    "priority": 8,
    "annotationTask": "describe_table_region_for_retrieval",
    "webInput": {
      "copyPasteContext": "paperId=alexnet-2012\npage=7\ncandidateType=table_region\nbbox=[341.64, 290.39, 504.0, 322.03]\ncaptionText=Comparison of results on ILSVRC- 2010 test set. In italics are best results achieved by others.\nheadingPath=Page 7\nnearbyText=Model Top-1 Top-5 Sparse coding [2] 47.1% 28.2% SIFT + FVs [24] 45.7% 25.7% CNN 37.5% 17.0% Table 1: Comparison of results on ILSVRC- 2010 test set. In italics are best results achieved by others. We also entered our model in the ILSVRC-2012 com- petition and report our results in Table 2. Since the ILSVRC-2012 test set labels are not publicly available, we cannot report test error rates for all the models that we...",
      "attachmentInstruction": "Attach the matching page or crop image manually if available. If no image is attached, do not invent visual details; use only supplied text and bbox metadata.",
      "plannedAttachmentRef": "papers_dir/visual_layout_planned_crops/alexnet-2012/page-7/visual-layout-alexnet-2012-table_region-7-b9f742e1da360378.png",
      "pageImageRequired": true
    },
    "textContext": {
      "nearbyText": "Model Top-1 Top-5 Sparse coding [2] 47.1% 28.2% SIFT + FVs [24] 45.7% 25.7% CNN 37.5% 17.0% Table 1: Comparison of results on ILSVRC- 2010 test set. In italics are best results achieved by others. We also entered our model in the ILSVRC-2012 com- petition and report our results in Table 2. Since the ILSVRC-2012 test set labels are not publicly available, we cannot report test error rates for all the models that we...",
      "captionText": "Comparison of results on ILSVRC- 2010 test set. In italics are best results achieved by others.",
      "headingPath": [
        "Page 7"
      ]
    },
    "visualContext": {
      "cropRef": "papers_dir/visual_layout_planned_crops/alexnet-2012/page-7/visual-layout-alexnet-2012-table_region-7-b9f742e1da360378.png",
      "imageHash": "",
      "pageImageRequired": true
    },
    "retrievalHintPlan": {
      "targetDerivedTextField": "derivedTextForRetrieval",
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "expectedOutputContract": {
      "schema": "knowledge-hub.paper.visual-annotation-web-output-row.v1",
      "requiredFields": [
        "sourceCandidateId",
        "derivedTextForRetrieval",
        "visibleText",
        "retrievalKeywords",
        "uncertainty",
        "limitations",
        "strictEvidence",
        "citationGrade",
        "answerableWithoutTextEvidence"
      ],
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "provenance": {
      "sourceReportSchema": "knowledge-hub.paper.visual-layout-candidate-list-report.v1",
      "sourceCandidateId": "visual-layout:alexnet-2012:table_region:7:b9f742e1da360378",
      "sourceContentHash": "sha256:90137160c57217953d5f61857e64ca58e85f06e1b13b4f475c918b1b582b9771",
      "page": 7,
      "bbox": [
        341.64,
        290.39,
        504.0,
        322.03
      ],
      "extractionMethod": "pymupdf_text_block_caption_regex_v1"
    },
    "blockerReason": ""
  },
  {
    "schema": "knowledge-hub.paper.visual-annotation-web-pack-row.v1",
    "packCandidateId": "visual-annotation-web-pack:visual_annotation_web_pack_001:f9f67bf644d50d61",
    "sourceCandidateId": "visual-layout:alexnet-2012:figure_caption_region:8:889b5135e8e9ab9e",
    "paperId": "alexnet-2012",
    "paperRef": "papers_dir/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf",
    "sourceContentHash": "sha256:90137160c57217953d5f61857e64ca58e85f06e1b13b4f475c918b1b582b9771",
    "page": 8,
    "bbox": [
      108.0,
      244.05,
      504.0,
      297.6
    ],
    "candidateType": "figure_caption_region",
    "priority": 9,
    "annotationTask": "describe_figure_caption_region_for_retrieval",
    "webInput": {
      "copyPasteContext": "paperId=alexnet-2012\npage=8\ncandidateType=figure_caption_region\nbbox=[108.0, 244.05, 504.0, 297.6]\ncaptionText=(Left) Eight ILSVRC-2010 test images and the ﬁve labels considered most probable by our model. The correct label is written under each image, and the probability assigned to the correct label is also shown with a red bar (if it happens to be in the top 5). (Right) Five ILSVRC-2010 test images in the ﬁrst column. The...\nheadingPath=Page 8\nnearbyText=Figure 4: (Left) Eight ILSVRC-2010 test images and the ﬁve labels considered most probable by our model. The correct label is written under each image, and the probability assigned to the correct label is also shown with a red bar (if it happens to be in the top 5). (Right) Five ILSVRC-2010 test images in the ﬁrst column. The remaining columns show the six training images that produce feature vectors in the last h...",
      "attachmentInstruction": "Attach the matching page or crop image manually if available. If no image is attached, do not invent visual details; use only supplied text and bbox metadata.",
      "plannedAttachmentRef": "papers_dir/visual_layout_planned_crops/alexnet-2012/page-8/visual-layout-alexnet-2012-figure_caption_region-8-889b5135e8e9ab9e.png",
      "pageImageRequired": true
    },
    "textContext": {
      "nearbyText": "Figure 4: (Left) Eight ILSVRC-2010 test images and the ﬁve labels considered most probable by our model. The correct label is written under each image, and the probability assigned to the correct label is also shown with a red bar (if it happens to be in the top 5). (Right) Five ILSVRC-2010 test images in the ﬁrst column. The remaining columns show the six training images that produce feature vectors in the last h...",
      "captionText": "(Left) Eight ILSVRC-2010 test images and the ﬁve labels considered most probable by our model. The correct label is written under each image, and the probability assigned to the correct label is also shown with a red bar (if it happens to be in the top 5). (Right) Five ILSVRC-2010 test images in the ﬁrst column. The...",
      "headingPath": [
        "Page 8"
      ]
    },
    "visualContext": {
      "cropRef": "papers_dir/visual_layout_planned_crops/alexnet-2012/page-8/visual-layout-alexnet-2012-figure_caption_region-8-889b5135e8e9ab9e.png",
      "imageHash": "",
      "pageImageRequired": true
    },
    "retrievalHintPlan": {
      "targetDerivedTextField": "derivedTextForRetrieval",
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "expectedOutputContract": {
      "schema": "knowledge-hub.paper.visual-annotation-web-output-row.v1",
      "requiredFields": [
        "sourceCandidateId",
        "derivedTextForRetrieval",
        "visibleText",
        "retrievalKeywords",
        "uncertainty",
        "limitations",
        "strictEvidence",
        "citationGrade",
        "answerableWithoutTextEvidence"
      ],
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "provenance": {
      "sourceReportSchema": "knowledge-hub.paper.visual-layout-candidate-list-report.v1",
      "sourceCandidateId": "visual-layout:alexnet-2012:figure_caption_region:8:889b5135e8e9ab9e",
      "sourceContentHash": "sha256:90137160c57217953d5f61857e64ca58e85f06e1b13b4f475c918b1b582b9771",
      "page": 8,
      "bbox": [
        108.0,
        244.05,
        504.0,
        297.6
      ],
      "extractionMethod": "pymupdf_text_block_caption_regex_v1"
    },
    "blockerReason": ""
  },
  {
    "schema": "knowledge-hub.paper.visual-annotation-web-pack-row.v1",
    "packCandidateId": "visual-annotation-web-pack:visual_annotation_web_pack_001:004ad1cc6c36d072",
    "sourceCandidateId": "visual-layout:resnet-2015:figure_caption_region:1:aba2bf94ef1625a2",
    "paperId": "resnet-2015",
    "paperRef": "papers_dir/Deep Residual Learning for Image Recognition.pdf",
    "sourceContentHash": "sha256:1e0651b6810ecba34a3dbc5b5b0209226f889004607c1f203540a48d64e5a93a",
    "page": 1,
    "bbox": [
      308.86,
      304.89,
      545.11,
      346.73
    ],
    "candidateType": "figure_caption_region",
    "priority": 10,
    "annotationTask": "describe_figure_caption_region_for_retrieval",
    "webInput": {
      "copyPasteContext": "paperId=resnet-2015\npage=1\ncandidateType=figure_caption_region\nbbox=[308.86, 304.89, 545.11, 346.73]\ncaptionText=Training error (left) and test error (right) on CIFAR-10 with 20-layer and 56-layer “plain” networks. The deeper network has higher training error, and thus test error. Similar phenomena on ImageNet is presented in Fig. 4.\nheadingPath=Page 1\nnearbyText=20-layer Figure 1. Training error (left) and test error (right) on CIFAR-10 with 20-layer and 56-layer “plain” networks. The deeper network has higher training error, and thus test error. Similar phenomena on ImageNet is presented in Fig. 4. greatly beneﬁted from very deep models. Driven by the signiﬁcance of depth, a question arises: Is learning better networks as easy as stacking more layers? An obstacle to answ...",
      "attachmentInstruction": "Attach the matching page or crop image manually if available. If no image is attached, do not invent visual details; use only supplied text and bbox metadata.",
      "plannedAttachmentRef": "papers_dir/visual_layout_planned_crops/resnet-2015/page-1/visual-layout-resnet-2015-figure_caption_region-1-aba2bf94ef1625a2.png",
      "pageImageRequired": true
    },
    "textContext": {
      "nearbyText": "20-layer Figure 1. Training error (left) and test error (right) on CIFAR-10 with 20-layer and 56-layer “plain” networks. The deeper network has higher training error, and thus test error. Similar phenomena on ImageNet is presented in Fig. 4. greatly beneﬁted from very deep models. Driven by the signiﬁcance of depth, a question arises: Is learning better networks as easy as stacking more layers? An obstacle to answ...",
      "captionText": "Training error (left) and test error (right) on CIFAR-10 with 20-layer and 56-layer “plain” networks. The deeper network has higher training error, and thus test error. Similar phenomena on ImageNet is presented in Fig. 4.",
      "headingPath": [
        "Page 1"
      ]
    },
    "visualContext": {
      "cropRef": "papers_dir/visual_layout_planned_crops/resnet-2015/page-1/visual-layout-resnet-2015-figure_caption_region-1-aba2bf94ef1625a2.png",
      "imageHash": "",
      "pageImageRequired": true
    },
    "retrievalHintPlan": {
      "targetDerivedTextField": "derivedTextForRetrieval",
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "expectedOutputContract": {
      "schema": "knowledge-hub.paper.visual-annotation-web-output-row.v1",
      "requiredFields": [
        "sourceCandidateId",
        "derivedTextForRetrieval",
        "visibleText",
        "retrievalKeywords",
        "uncertainty",
        "limitations",
        "strictEvidence",
        "citationGrade",
        "answerableWithoutTextEvidence"
      ],
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "provenance": {
      "sourceReportSchema": "knowledge-hub.paper.visual-layout-candidate-list-report.v1",
      "sourceCandidateId": "visual-layout:resnet-2015:figure_caption_region:1:aba2bf94ef1625a2",
      "sourceContentHash": "sha256:1e0651b6810ecba34a3dbc5b5b0209226f889004607c1f203540a48d64e5a93a",
      "page": 1,
      "bbox": [
        308.86,
        304.89,
        545.11,
        346.73
      ],
      "extractionMethod": "pymupdf_text_block_caption_regex_v1"
    },
    "blockerReason": ""
  },
  {
    "schema": "knowledge-hub.paper.visual-annotation-web-pack-row.v1",
    "packCandidateId": "visual-annotation-web-pack:visual_annotation_web_pack_001:25f9c51cc846c7e7",
    "sourceCandidateId": "visual-layout:resnet-2015:figure_caption_region:2:9687efdba9452012",
    "paperId": "resnet-2015",
    "paperRef": "papers_dir/Deep Residual Learning for Image Recognition.pdf",
    "sourceContentHash": "sha256:1e0651b6810ecba34a3dbc5b5b0209226f889004607c1f203540a48d64e5a93a",
    "page": 2,
    "bbox": [
      86.62,
      158.15,
      249.86,
      167.12
    ],
    "candidateType": "figure_caption_region",
    "priority": 11,
    "annotationTask": "describe_figure_caption_region_for_retrieval",
    "webInput": {
      "copyPasteContext": "paperId=resnet-2015\npage=2\ncandidateType=figure_caption_region\nbbox=[86.62, 158.15, 249.86, 167.12]\ncaptionText=Residual learning: a building block.\nheadingPath=Page 2\nnearbyText=F(x) x Figure 2. Residual learning: a building block. are comparably good or better than the constructed solution (or unable to do so in feasible time). In this paper, we address the degradation problem by introducing a deep residual learning framework. In- stead of hoping each few stacked layers directly ﬁt a desired underlying mapping, we explicitly let these lay- ers ﬁt a residual mapping. Formally, denoting th...",
      "attachmentInstruction": "Attach the matching page or crop image manually if available. If no image is attached, do not invent visual details; use only supplied text and bbox metadata.",
      "plannedAttachmentRef": "papers_dir/visual_layout_planned_crops/resnet-2015/page-2/visual-layout-resnet-2015-figure_caption_region-2-9687efdba9452012.png",
      "pageImageRequired": true
    },
    "textContext": {
      "nearbyText": "F(x) x Figure 2. Residual learning: a building block. are comparably good or better than the constructed solution (or unable to do so in feasible time). In this paper, we address the degradation problem by introducing a deep residual learning framework. In- stead of hoping each few stacked layers directly ﬁt a desired underlying mapping, we explicitly let these lay- ers ﬁt a residual mapping. Formally, denoting th...",
      "captionText": "Residual learning: a building block.",
      "headingPath": [
        "Page 2"
      ]
    },
    "visualContext": {
      "cropRef": "papers_dir/visual_layout_planned_crops/resnet-2015/page-2/visual-layout-resnet-2015-figure_caption_region-2-9687efdba9452012.png",
      "imageHash": "",
      "pageImageRequired": true
    },
    "retrievalHintPlan": {
      "targetDerivedTextField": "derivedTextForRetrieval",
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "expectedOutputContract": {
      "schema": "knowledge-hub.paper.visual-annotation-web-output-row.v1",
      "requiredFields": [
        "sourceCandidateId",
        "derivedTextForRetrieval",
        "visibleText",
        "retrievalKeywords",
        "uncertainty",
        "limitations",
        "strictEvidence",
        "citationGrade",
        "answerableWithoutTextEvidence"
      ],
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "provenance": {
      "sourceReportSchema": "knowledge-hub.paper.visual-layout-candidate-list-report.v1",
      "sourceCandidateId": "visual-layout:resnet-2015:figure_caption_region:2:9687efdba9452012",
      "sourceContentHash": "sha256:1e0651b6810ecba34a3dbc5b5b0209226f889004607c1f203540a48d64e5a93a",
      "page": 2,
      "bbox": [
        86.62,
        158.15,
        249.86,
        167.12
      ],
      "extractionMethod": "pymupdf_text_block_caption_regex_v1"
    },
    "blockerReason": ""
  },
  {
    "schema": "knowledge-hub.paper.visual-annotation-web-pack-row.v1",
    "packCandidateId": "visual-annotation-web-pack:visual_annotation_web_pack_001:2b004128c5a9ea24",
    "sourceCandidateId": "visual-layout:resnet-2015:equation_region:3:0e80570b407b6793",
    "paperId": "resnet-2015",
    "paperRef": "papers_dir/Deep Residual Learning for Image Recognition.pdf",
    "sourceContentHash": "sha256:1e0651b6810ecba34a3dbc5b5b0209226f889004607c1f203540a48d64e5a93a",
    "page": 3,
    "bbox": [
      123.47,
      626.46,
      286.36,
      643.81
    ],
    "candidateType": "equation_region",
    "priority": 12,
    "annotationTask": "describe_equation_region_for_retrieval",
    "webInput": {
      "copyPasteContext": "paperId=resnet-2015\npage=3\ncandidateType=equation_region\nbbox=[123.47, 626.46, 286.36, 643.81]\nheadingPath=Page 3\nnearbyText=We adopt residual learning to every few stacked layers. A building block is shown in Fig. 2. Formally, in this paper we consider a building block deﬁned as: y = F(x, {Wi}) + x. (1) Here x and y are the input and output vectors of the lay- ers considered. The function F(x, {Wi}) represents the residual mapping to be learned. For the example in Fig. 2 that has two layers, F = W2σ(W1x) in which σ denotes",
      "attachmentInstruction": "Attach the matching page or crop image manually if available. If no image is attached, do not invent visual details; use only supplied text and bbox metadata.",
      "plannedAttachmentRef": "papers_dir/visual_layout_planned_crops/resnet-2015/page-3/visual-layout-resnet-2015-equation_region-3-0e80570b407b6793.png",
      "pageImageRequired": true
    },
    "textContext": {
      "nearbyText": "We adopt residual learning to every few stacked layers. A building block is shown in Fig. 2. Formally, in this paper we consider a building block deﬁned as: y = F(x, {Wi}) + x. (1) Here x and y are the input and output vectors of the lay- ers considered. The function F(x, {Wi}) represents the residual mapping to be learned. For the example in Fig. 2 that has two layers, F = W2σ(W1x) in which σ denotes",
      "captionText": "",
      "headingPath": [
        "Page 3"
      ]
    },
    "visualContext": {
      "cropRef": "papers_dir/visual_layout_planned_crops/resnet-2015/page-3/visual-layout-resnet-2015-equation_region-3-0e80570b407b6793.png",
      "imageHash": "",
      "pageImageRequired": true
    },
    "retrievalHintPlan": {
      "targetDerivedTextField": "derivedTextForRetrieval",
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "expectedOutputContract": {
      "schema": "knowledge-hub.paper.visual-annotation-web-output-row.v1",
      "requiredFields": [
        "sourceCandidateId",
        "derivedTextForRetrieval",
        "visibleText",
        "retrievalKeywords",
        "uncertainty",
        "limitations",
        "strictEvidence",
        "citationGrade",
        "answerableWithoutTextEvidence"
      ],
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "provenance": {
      "sourceReportSchema": "knowledge-hub.paper.visual-layout-candidate-list-report.v1",
      "sourceCandidateId": "visual-layout:resnet-2015:equation_region:3:0e80570b407b6793",
      "sourceContentHash": "sha256:1e0651b6810ecba34a3dbc5b5b0209226f889004607c1f203540a48d64e5a93a",
      "page": 3,
      "bbox": [
        123.47,
        626.46,
        286.36,
        643.81
      ],
      "extractionMethod": "pymupdf_text_block_equation_heuristic_v1"
    },
    "blockerReason": ""
  },
  {
    "schema": "knowledge-hub.paper.visual-annotation-web-pack-row.v1",
    "packCandidateId": "visual-annotation-web-pack:visual_annotation_web_pack_001:3e357cb116aba58e",
    "sourceCandidateId": "visual-layout:resnet-2015:equation_region:3:69248b1db8c80503",
    "paperId": "resnet-2015",
    "paperRef": "papers_dir/Deep Residual Learning for Image Recognition.pdf",
    "sourceContentHash": "sha256:1e0651b6810ecba34a3dbc5b5b0209226f889004607c1f203540a48d64e5a93a",
    "page": 3,
    "bbox": [
      375.38,
      264.43,
      545.11,
      281.78
    ],
    "candidateType": "equation_region",
    "priority": 13,
    "annotationTask": "describe_equation_region_for_retrieval",
    "webInput": {
      "copyPasteContext": "paperId=resnet-2015\npage=3\ncandidateType=equation_region\nbbox=[375.38, 264.43, 545.11, 281.78]\nheadingPath=Page 3\nnearbyText=ReLU [29] and the biases are omitted for simplifying no- tations. The operation F + x is performed by a shortcut connection and element-wise addition. We adopt the sec- ond nonlinearity after the addition (i.e., σ(y), see Fig. 2). The shortcut connections in Eqn.(1) introduce neither ex- tra parameter nor computation complexity. This is not only attractive in practice but also important in our comparisons between...",
      "attachmentInstruction": "Attach the matching page or crop image manually if available. If no image is attached, do not invent visual details; use only supplied text and bbox metadata.",
      "plannedAttachmentRef": "papers_dir/visual_layout_planned_crops/resnet-2015/page-3/visual-layout-resnet-2015-equation_region-3-69248b1db8c80503.png",
      "pageImageRequired": true
    },
    "textContext": {
      "nearbyText": "ReLU [29] and the biases are omitted for simplifying no- tations. The operation F + x is performed by a shortcut connection and element-wise addition. We adopt the sec- ond nonlinearity after the addition (i.e., σ(y), see Fig. 2). The shortcut connections in Eqn.(1) introduce neither ex- tra parameter nor computation complexity. This is not only attractive in practice but also important in our comparisons between...",
      "captionText": "",
      "headingPath": [
        "Page 3"
      ]
    },
    "visualContext": {
      "cropRef": "papers_dir/visual_layout_planned_crops/resnet-2015/page-3/visual-layout-resnet-2015-equation_region-3-69248b1db8c80503.png",
      "imageHash": "",
      "pageImageRequired": true
    },
    "retrievalHintPlan": {
      "targetDerivedTextField": "derivedTextForRetrieval",
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "expectedOutputContract": {
      "schema": "knowledge-hub.paper.visual-annotation-web-output-row.v1",
      "requiredFields": [
        "sourceCandidateId",
        "derivedTextForRetrieval",
        "visibleText",
        "retrievalKeywords",
        "uncertainty",
        "limitations",
        "strictEvidence",
        "citationGrade",
        "answerableWithoutTextEvidence"
      ],
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "provenance": {
      "sourceReportSchema": "knowledge-hub.paper.visual-layout-candidate-list-report.v1",
      "sourceCandidateId": "visual-layout:resnet-2015:equation_region:3:69248b1db8c80503",
      "sourceContentHash": "sha256:1e0651b6810ecba34a3dbc5b5b0209226f889004607c1f203540a48d64e5a93a",
      "page": 3,
      "bbox": [
        375.38,
        264.43,
        545.11,
        281.78
      ],
      "extractionMethod": "pymupdf_text_block_equation_heuristic_v1"
    },
    "blockerReason": ""
  },
  {
    "schema": "knowledge-hub.paper.visual-annotation-web-pack-row.v1",
    "packCandidateId": "visual-annotation-web-pack:visual_annotation_web_pack_001:73e80d3f26c63da5",
    "sourceCandidateId": "visual-layout:resnet-2015:equation_region:3:7068ad040e2d0243",
    "paperId": "resnet-2015",
    "paperRef": "papers_dir/Deep Residual Learning for Image Recognition.pdf",
    "sourceContentHash": "sha256:1e0651b6810ecba34a3dbc5b5b0209226f889004607c1f203540a48d64e5a93a",
    "page": 3,
    "bbox": [
      50.11,
      648.82,
      286.37,
      701.8
    ],
    "candidateType": "equation_region",
    "priority": 14,
    "annotationTask": "describe_equation_region_for_retrieval",
    "webInput": {
      "copyPasteContext": "paperId=resnet-2015\npage=3\ncandidateType=equation_region\nbbox=[50.11, 648.82, 286.37, 701.8]\nheadingPath=Page 3\nnearbyText=y = F(x, {Wi}) + x. (1) Here x and y are the input and output vectors of the lay- ers considered. The function F(x, {Wi}) represents the residual mapping to be learned. For the example in Fig. 2 that has two layers, F = W2σ(W1x) in which σ denotes 2This hypothesis, however, is still an open question. See [28].",
      "attachmentInstruction": "Attach the matching page or crop image manually if available. If no image is attached, do not invent visual details; use only supplied text and bbox metadata.",
      "plannedAttachmentRef": "papers_dir/visual_layout_planned_crops/resnet-2015/page-3/visual-layout-resnet-2015-equation_region-3-7068ad040e2d0243.png",
      "pageImageRequired": true
    },
    "textContext": {
      "nearbyText": "y = F(x, {Wi}) + x. (1) Here x and y are the input and output vectors of the lay- ers considered. The function F(x, {Wi}) represents the residual mapping to be learned. For the example in Fig. 2 that has two layers, F = W2σ(W1x) in which σ denotes 2This hypothesis, however, is still an open question. See [28].",
      "captionText": "",
      "headingPath": [
        "Page 3"
      ]
    },
    "visualContext": {
      "cropRef": "papers_dir/visual_layout_planned_crops/resnet-2015/page-3/visual-layout-resnet-2015-equation_region-3-7068ad040e2d0243.png",
      "imageHash": "",
      "pageImageRequired": true
    },
    "retrievalHintPlan": {
      "targetDerivedTextField": "derivedTextForRetrieval",
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "expectedOutputContract": {
      "schema": "knowledge-hub.paper.visual-annotation-web-output-row.v1",
      "requiredFields": [
        "sourceCandidateId",
        "derivedTextForRetrieval",
        "visibleText",
        "retrievalKeywords",
        "uncertainty",
        "limitations",
        "strictEvidence",
        "citationGrade",
        "answerableWithoutTextEvidence"
      ],
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "provenance": {
      "sourceReportSchema": "knowledge-hub.paper.visual-layout-candidate-list-report.v1",
      "sourceCandidateId": "visual-layout:resnet-2015:equation_region:3:7068ad040e2d0243",
      "sourceContentHash": "sha256:1e0651b6810ecba34a3dbc5b5b0209226f889004607c1f203540a48d64e5a93a",
      "page": 3,
      "bbox": [
        50.11,
        648.82,
        286.37,
        701.8
      ],
      "extractionMethod": "pymupdf_text_block_equation_heuristic_v1"
    },
    "blockerReason": ""
  },
  {
    "schema": "knowledge-hub.paper.visual-annotation-web-pack-row.v1",
    "packCandidateId": "visual-annotation-web-pack:visual_annotation_web_pack_001:d7a6ec0319b177d6",
    "sourceCandidateId": "visual-layout:resnet-2015:figure_caption_region:4:0b3754b79959d64e",
    "paperId": "resnet-2015",
    "paperRef": "papers_dir/Deep Residual Learning for Image Recognition.pdf",
    "sourceContentHash": "sha256:1e0651b6810ecba34a3dbc5b5b0209226f889004607c1f203540a48d64e5a93a",
    "page": 4,
    "bbox": [
      50.11,
      632.83,
      286.37,
      696.68
    ],
    "candidateType": "figure_caption_region",
    "priority": 15,
    "annotationTask": "describe_figure_caption_region_for_retrieval",
    "webInput": {
      "copyPasteContext": "paperId=resnet-2015\npage=4\ncandidateType=figure_caption_region\nbbox=[50.11, 632.83, 286.37, 696.68]\ncaptionText=Example network architectures for ImageNet. Left: the VGG-19 model [41] (19.6 billion FLOPs) as a reference. Mid- dle: a plain network with 34 parameter layers (3.6 billion FLOPs). Right: a residual network with 34 parameter layers (3.6 billion FLOPs). The dotted shortcuts increase dimensions. Table 1 shows more det...\nheadingPath=Page 4\nnearbyText=image 34-layer residual Figure 3. Example network architectures for ImageNet. Left: the VGG-19 model [41] (19.6 billion FLOPs) as a reference. Mid- dle: a plain network with 34 parameter layers (3.6 billion FLOPs). Right: a residual network with 34 parameter layers (3.6 billion FLOPs). The dotted shortcuts increase dimensions. Table 1 shows more details and other variants. Residual Network. Based on the above plai...",
      "attachmentInstruction": "Attach the matching page or crop image manually if available. If no image is attached, do not invent visual details; use only supplied text and bbox metadata.",
      "plannedAttachmentRef": "papers_dir/visual_layout_planned_crops/resnet-2015/page-4/visual-layout-resnet-2015-figure_caption_region-4-0b3754b79959d64e.png",
      "pageImageRequired": true
    },
    "textContext": {
      "nearbyText": "image 34-layer residual Figure 3. Example network architectures for ImageNet. Left: the VGG-19 model [41] (19.6 billion FLOPs) as a reference. Mid- dle: a plain network with 34 parameter layers (3.6 billion FLOPs). Right: a residual network with 34 parameter layers (3.6 billion FLOPs). The dotted shortcuts increase dimensions. Table 1 shows more details and other variants. Residual Network. Based on the above plai...",
      "captionText": "Example network architectures for ImageNet. Left: the VGG-19 model [41] (19.6 billion FLOPs) as a reference. Mid- dle: a plain network with 34 parameter layers (3.6 billion FLOPs). Right: a residual network with 34 parameter layers (3.6 billion FLOPs). The dotted shortcuts increase dimensions. Table 1 shows more det...",
      "headingPath": [
        "Page 4"
      ]
    },
    "visualContext": {
      "cropRef": "papers_dir/visual_layout_planned_crops/resnet-2015/page-4/visual-layout-resnet-2015-figure_caption_region-4-0b3754b79959d64e.png",
      "imageHash": "",
      "pageImageRequired": true
    },
    "retrievalHintPlan": {
      "targetDerivedTextField": "derivedTextForRetrieval",
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "expectedOutputContract": {
      "schema": "knowledge-hub.paper.visual-annotation-web-output-row.v1",
      "requiredFields": [
        "sourceCandidateId",
        "derivedTextForRetrieval",
        "visibleText",
        "retrievalKeywords",
        "uncertainty",
        "limitations",
        "strictEvidence",
        "citationGrade",
        "answerableWithoutTextEvidence"
      ],
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "provenance": {
      "sourceReportSchema": "knowledge-hub.paper.visual-layout-candidate-list-report.v1",
      "sourceCandidateId": "visual-layout:resnet-2015:figure_caption_region:4:0b3754b79959d64e",
      "sourceContentHash": "sha256:1e0651b6810ecba34a3dbc5b5b0209226f889004607c1f203540a48d64e5a93a",
      "page": 4,
      "bbox": [
        50.11,
        632.83,
        286.37,
        696.68
      ],
      "extractionMethod": "pymupdf_text_block_caption_regex_v1"
    },
    "blockerReason": ""
  },
  {
    "schema": "knowledge-hub.paper.visual-annotation-web-pack-row.v1",
    "packCandidateId": "visual-annotation-web-pack:visual_annotation_web_pack_001:02625e2f31bc8265",
    "sourceCandidateId": "visual-layout:resnet-2015:figure_caption_region:5:666c24606607fbbf",
    "paperId": "resnet-2015",
    "paperRef": "papers_dir/Deep Residual Learning for Image Recognition.pdf",
    "sourceContentHash": "sha256:1e0651b6810ecba34a3dbc5b5b0209226f889004607c1f203540a48d64e5a93a",
    "page": 5,
    "bbox": [
      50.11,
      392.29,
      545.12,
      423.26
    ],
    "candidateType": "figure_caption_region",
    "priority": 16,
    "annotationTask": "describe_figure_caption_region_for_retrieval",
    "webInput": {
      "copyPasteContext": "paperId=resnet-2015\npage=5\ncandidateType=figure_caption_region\nbbox=[50.11, 392.29, 545.12, 423.26]\ncaptionText=Training on ImageNet. Thin curves denote training error, and bold curves denote validation error of the center crops. Left: plain networks of 18 and 34 layers. Right: ResNets of 18 and 34 layers. In this plot, the residual networks have no extra parameter compared to their plain counterparts.\nheadingPath=Page 5\nnearbyText=34-layer Figure 4. Training on ImageNet. Thin curves denote training error, and bold curves denote validation error of the center crops. Left: plain networks of 18 and 34 layers. Right: ResNets of 18 and 34 layers. In this plot, the residual networks have no extra parameter compared to their plain counterparts. plain ResNet 18 layers 27.94 27.88 34 layers 28.54 25.03",
      "attachmentInstruction": "Attach the matching page or crop image manually if available. If no image is attached, do not invent visual details; use only supplied text and bbox metadata.",
      "plannedAttachmentRef": "papers_dir/visual_layout_planned_crops/resnet-2015/page-5/visual-layout-resnet-2015-figure_caption_region-5-666c24606607fbbf.png",
      "pageImageRequired": true
    },
    "textContext": {
      "nearbyText": "34-layer Figure 4. Training on ImageNet. Thin curves denote training error, and bold curves denote validation error of the center crops. Left: plain networks of 18 and 34 layers. Right: ResNets of 18 and 34 layers. In this plot, the residual networks have no extra parameter compared to their plain counterparts. plain ResNet 18 layers 27.94 27.88 34 layers 28.54 25.03",
      "captionText": "Training on ImageNet. Thin curves denote training error, and bold curves denote validation error of the center crops. Left: plain networks of 18 and 34 layers. Right: ResNets of 18 and 34 layers. In this plot, the residual networks have no extra parameter compared to their plain counterparts.",
      "headingPath": [
        "Page 5"
      ]
    },
    "visualContext": {
      "cropRef": "papers_dir/visual_layout_planned_crops/resnet-2015/page-5/visual-layout-resnet-2015-figure_caption_region-5-666c24606607fbbf.png",
      "imageHash": "",
      "pageImageRequired": true
    },
    "retrievalHintPlan": {
      "targetDerivedTextField": "derivedTextForRetrieval",
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "expectedOutputContract": {
      "schema": "knowledge-hub.paper.visual-annotation-web-output-row.v1",
      "requiredFields": [
        "sourceCandidateId",
        "derivedTextForRetrieval",
        "visibleText",
        "retrievalKeywords",
        "uncertainty",
        "limitations",
        "strictEvidence",
        "citationGrade",
        "answerableWithoutTextEvidence"
      ],
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "provenance": {
      "sourceReportSchema": "knowledge-hub.paper.visual-layout-candidate-list-report.v1",
      "sourceCandidateId": "visual-layout:resnet-2015:figure_caption_region:5:666c24606607fbbf",
      "sourceContentHash": "sha256:1e0651b6810ecba34a3dbc5b5b0209226f889004607c1f203540a48d64e5a93a",
      "page": 5,
      "bbox": [
        50.11,
        392.29,
        545.12,
        423.26
      ],
      "extractionMethod": "pymupdf_text_block_caption_regex_v1"
    },
    "blockerReason": ""
  },
  {
    "schema": "knowledge-hub.paper.visual-annotation-web-pack-row.v1",
    "packCandidateId": "visual-annotation-web-pack:visual_annotation_web_pack_001:e206894f3bae718a",
    "sourceCandidateId": "visual-layout:resnet-2015:table_region:5:749ede3c6c93e4fa",
    "paperId": "resnet-2015",
    "paperRef": "papers_dir/Deep Residual Learning for Image Recognition.pdf",
    "sourceContentHash": "sha256:1e0651b6810ecba34a3dbc5b5b0209226f889004607c1f203540a48d64e5a93a",
    "page": 5,
    "bbox": [
      50.11,
      224.67,
      545.11,
      244.6
    ],
    "candidateType": "table_region",
    "priority": 17,
    "annotationTask": "describe_table_region_for_retrieval",
    "webInput": {
      "copyPasteContext": "paperId=resnet-2015\npage=5\ncandidateType=table_region\nbbox=[50.11, 224.67, 545.11, 244.6]\ncaptionText=Architectures for ImageNet. Building blocks are shown in brackets (see also Fig. 5), with the numbers of blocks stacked. Down- sampling is performed by conv3 1, conv4 1, and conv5 1 with a stride of 2.\nheadingPath=Page 5\nnearbyText=1×1 average pool, 1000-d fc, softmax FLOPs 1.8×109 3.6×109 3.8×109 7.6×109 11.3×109 Table 1. Architectures for ImageNet. Building blocks are shown in brackets (see also Fig. 5), with the numbers of blocks stacked. Down- sampling is performed by conv3 1, conv4 1, and conv5 1 with a stride of 2. 0 10 20 30 40 50 20",
      "attachmentInstruction": "Attach the matching page or crop image manually if available. If no image is attached, do not invent visual details; use only supplied text and bbox metadata.",
      "plannedAttachmentRef": "papers_dir/visual_layout_planned_crops/resnet-2015/page-5/visual-layout-resnet-2015-table_region-5-749ede3c6c93e4fa.png",
      "pageImageRequired": true
    },
    "textContext": {
      "nearbyText": "1×1 average pool, 1000-d fc, softmax FLOPs 1.8×109 3.6×109 3.8×109 7.6×109 11.3×109 Table 1. Architectures for ImageNet. Building blocks are shown in brackets (see also Fig. 5), with the numbers of blocks stacked. Down- sampling is performed by conv3 1, conv4 1, and conv5 1 with a stride of 2. 0 10 20 30 40 50 20",
      "captionText": "Architectures for ImageNet. Building blocks are shown in brackets (see also Fig. 5), with the numbers of blocks stacked. Down- sampling is performed by conv3 1, conv4 1, and conv5 1 with a stride of 2.",
      "headingPath": [
        "Page 5"
      ]
    },
    "visualContext": {
      "cropRef": "papers_dir/visual_layout_planned_crops/resnet-2015/page-5/visual-layout-resnet-2015-table_region-5-749ede3c6c93e4fa.png",
      "imageHash": "",
      "pageImageRequired": true
    },
    "retrievalHintPlan": {
      "targetDerivedTextField": "derivedTextForRetrieval",
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "expectedOutputContract": {
      "schema": "knowledge-hub.paper.visual-annotation-web-output-row.v1",
      "requiredFields": [
        "sourceCandidateId",
        "derivedTextForRetrieval",
        "visibleText",
        "retrievalKeywords",
        "uncertainty",
        "limitations",
        "strictEvidence",
        "citationGrade",
        "answerableWithoutTextEvidence"
      ],
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "provenance": {
      "sourceReportSchema": "knowledge-hub.paper.visual-layout-candidate-list-report.v1",
      "sourceCandidateId": "visual-layout:resnet-2015:table_region:5:749ede3c6c93e4fa",
      "sourceContentHash": "sha256:1e0651b6810ecba34a3dbc5b5b0209226f889004607c1f203540a48d64e5a93a",
      "page": 5,
      "bbox": [
        50.11,
        224.67,
        545.11,
        244.6
      ],
      "extractionMethod": "pymupdf_text_block_caption_regex_v1"
    },
    "blockerReason": ""
  },
  {
    "schema": "knowledge-hub.paper.visual-annotation-web-pack-row.v1",
    "packCandidateId": "visual-annotation-web-pack:visual_annotation_web_pack_001:d26faca025185255",
    "sourceCandidateId": "visual-layout:resnet-2015:table_region:5:6f67c711d387611a",
    "paperId": "resnet-2015",
    "paperRef": "papers_dir/Deep Residual Learning for Image Recognition.pdf",
    "sourceContentHash": "sha256:1e0651b6810ecba34a3dbc5b5b0209226f889004607c1f203540a48d64e5a93a",
    "page": 5,
    "bbox": [
      50.11,
      485.61,
      286.36,
      516.49
    ],
    "candidateType": "table_region",
    "priority": 18,
    "annotationTask": "describe_table_region_for_retrieval",
    "webInput": {
      "copyPasteContext": "paperId=resnet-2015\npage=5\ncandidateType=table_region\nbbox=[50.11, 485.61, 286.36, 516.49]\ncaptionText=Top-1 error (%, 10-crop testing) on ImageNet validation. Here the ResNets have no extra parameter compared to their plain counterparts. Fig. 4 shows the training procedures.\nheadingPath=Page 5\nnearbyText=plain ResNet 18 layers 27.94 27.88 34 layers 28.54 25.03 Table 2. Top-1 error (%, 10-crop testing) on ImageNet validation. Here the ResNets have no extra parameter compared to their plain counterparts. Fig. 4 shows the training procedures. 34-layer plain net has higher training error throughout the whole training procedure, even though the solution space of the 18-layer plain network is a subspace of that of the 3...",
      "attachmentInstruction": "Attach the matching page or crop image manually if available. If no image is attached, do not invent visual details; use only supplied text and bbox metadata.",
      "plannedAttachmentRef": "papers_dir/visual_layout_planned_crops/resnet-2015/page-5/visual-layout-resnet-2015-table_region-5-6f67c711d387611a.png",
      "pageImageRequired": true
    },
    "textContext": {
      "nearbyText": "plain ResNet 18 layers 27.94 27.88 34 layers 28.54 25.03 Table 2. Top-1 error (%, 10-crop testing) on ImageNet validation. Here the ResNets have no extra parameter compared to their plain counterparts. Fig. 4 shows the training procedures. 34-layer plain net has higher training error throughout the whole training procedure, even though the solution space of the 18-layer plain network is a subspace of that of the 3...",
      "captionText": "Top-1 error (%, 10-crop testing) on ImageNet validation. Here the ResNets have no extra parameter compared to their plain counterparts. Fig. 4 shows the training procedures.",
      "headingPath": [
        "Page 5"
      ]
    },
    "visualContext": {
      "cropRef": "papers_dir/visual_layout_planned_crops/resnet-2015/page-5/visual-layout-resnet-2015-table_region-5-6f67c711d387611a.png",
      "imageHash": "",
      "pageImageRequired": true
    },
    "retrievalHintPlan": {
      "targetDerivedTextField": "derivedTextForRetrieval",
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "expectedOutputContract": {
      "schema": "knowledge-hub.paper.visual-annotation-web-output-row.v1",
      "requiredFields": [
        "sourceCandidateId",
        "derivedTextForRetrieval",
        "visibleText",
        "retrievalKeywords",
        "uncertainty",
        "limitations",
        "strictEvidence",
        "citationGrade",
        "answerableWithoutTextEvidence"
      ],
      "allowedUse": "retrieval_hint_only",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    "provenance": {
      "sourceReportSchema": "knowledge-hub.paper.visual-layout-candidate-list-report.v1",
      "sourceCandidateId": "visual-layout:resnet-2015:table_region:5:6f67c711d387611a",
      "sourceContentHash": "sha256:1e0651b6810ecba34a3dbc5b5b0209226f889004607c1f203540a48d64e5a93a",
      "page": 5,
      "bbox": [
        50.11,
        485.61,
        286.36,
        516.49
      ],
      "extractionMethod": "pymupdf_text_block_caption_regex_v1"
    },
    "blockerReason": ""
  }
]
```

## Warnings

- `plannedAttachmentRef values are sanitized planned refs only; this tranche writes no crop files.`
- `image_region candidates are intentionally excluded from pack 001 to avoid noisy web/VLM calibration.`
- `Any returned derivedTextForRetrieval must remain retrieval_hint_only and non-evidence.`
