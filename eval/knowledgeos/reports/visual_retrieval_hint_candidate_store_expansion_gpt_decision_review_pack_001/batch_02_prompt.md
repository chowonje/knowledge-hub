# Visual Retrieval Hint Decision Recommendation Batch 02

Use only the row text provided in this prompt. Do not use outside sources, web search, PDFs, screenshots, full pages, or images.
Your job is to recommend a decision for a human/product reviewer. You are not making the final decision.
Return JSON only with top-level schema `knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-gpt-decision-recommendation-output.v1` and a `rows` array.
For every row, set `finalHumanDecision=false`, `applyAllowed=false`, `candidateStoreWrite=false`, `strictEvidence=false`, `citationGrade=false`, `answerableWithoutTextEvidence=false`, `runtimeVisible=false`, and `indexEligible=false`.

Allowed `suggestedDecision` values:
- `approve_store_candidate_only`
- `hold_pending_more_context`
- `reject_visual_hint_candidate`
- `request_recrop_or_reannotation`

Decision guidance:
- Suggest `approve_store_candidate_only` only when the retrieval hint is coherent, useful for future retrieval, and clearly remains non-evidence.
- Suggest `hold_pending_more_context` when the row may be useful but the text is incomplete, uncertain, or needs product review.
- Suggest `reject_visual_hint_candidate` when the row is off-topic, too noisy, or not useful as a retrieval hint.
- Suggest `request_recrop_or_reannotation` when the row appears affected by crop/context problems or a likely annotation mismatch.

## Rows To Review

| # | paperId | type | page | currentDecision | hintCandidateId | sourceCandidateId |
|---:|---|---|---:|---|---|---|
| 1 | clip-2021 | figure_caption_region | 2 | `hold_pending_human_product_review` | `visual-retrieval-hint:clip-2021:figure_caption_region:2:791bd02c68004e94` | `visual-layout:clip-2021:figure_caption_region:2:b354160446b3a89c` |
| 2 | mae-2021 | figure_caption_region | 1 | `hold_pending_human_product_review` | `visual-retrieval-hint:mae-2021:figure_caption_region:1:7f1d38a62ebc3ebe` | `visual-layout:mae-2021:figure_caption_region:1:f3477cc8ce7ec8b1` |
| 3 | resnet-2015 | figure_caption_region | 6 | `hold_pending_human_product_review` | `visual-retrieval-hint:resnet-2015:figure_caption_region:6:beeb9fb9ad38518d` | `visual-layout:resnet-2015:figure_caption_region:6:6b3dd4d05dc69888` |
| 4 | clip-2021 | figure_caption_region | 3 | `hold_pending_human_product_review` | `visual-retrieval-hint:clip-2021:figure_caption_region:3:7de4aa3a082925a4` | `visual-layout:clip-2021:figure_caption_region:3:fd2db71adcd98ec4` |
| 5 | mae-2021 | figure_caption_region | 2 | `hold_pending_human_product_review` | `visual-retrieval-hint:mae-2021:figure_caption_region:2:b1c1635f3ea9867d` | `visual-layout:mae-2021:figure_caption_region:2:460a5652014c6369` |
| 6 | resnet-2015 | figure_caption_region | 8 | `hold_pending_human_product_review` | `visual-retrieval-hint:resnet-2015:figure_caption_region:8:5983f29673800b6b` | `visual-layout:resnet-2015:figure_caption_region:8:5cf998f6a7a3739f` |
| 7 | clip-2021 | figure_caption_region | 5 | `hold_pending_human_product_review` | `visual-retrieval-hint:clip-2021:figure_caption_region:5:aa1447f2ba9526ae` | `visual-layout:clip-2021:figure_caption_region:5:87777b9ff7911bbe` |
| 8 | mae-2021 | figure_caption_region | 2 | `hold_pending_human_product_review` | `visual-retrieval-hint:mae-2021:figure_caption_region:2:eea2a43fb3574de9` | `visual-layout:mae-2021:figure_caption_region:2:b35cefa766216e3f` |

## Row Context

### Row 1
- sourceDecisionRowId: `visual-retrieval-hint-expansion-decision:c762f661957ea19c5690`
- hintCandidateId: `visual-retrieval-hint:clip-2021:figure_caption_region:2:791bd02c68004e94`
- paperId: `clip-2021`
- candidateType: `figure_caption_region`
- page: `2`
- currentDecision: `hold_pending_human_product_review`
- derivedTextForRetrievalSnippet: Retrieval hint only: CLIP Figure 1 summary diagram showing contrastive pre-training, creating a dataset classifier from label text, and zero-shot prediction with image and text encoders; text-image similarity matrix uses I1...IN and T1.....
- visibleTextSnippet: Visible fragments include: "Learning Transferable Visual Models From Natural Language Supervision"; "(1) Contrastive pre-training"; "Pepper the aussie pup"; "Text Encoder"; "Image Encoder"; "I1"; "I2"; "I3"; "IN"; "T1"; "T2"; "T3"; "TN";...
- retrievalKeywordCount: `10`

### Row 2
- sourceDecisionRowId: `visual-retrieval-hint-expansion-decision:0d7de4c618a6da6a92f4`
- hintCandidateId: `visual-retrieval-hint:mae-2021:figure_caption_region:1:7f1d38a62ebc3ebe`
- paperId: `mae-2021`
- candidateType: `figure_caption_region`
- page: `1`
- currentDecision: `hold_pending_human_product_review`
- derivedTextForRetrievalSnippet: Retrieval hint only: MAE Figure 1 architecture diagram showing masked image patches, visible-patch encoder, mask tokens, lightweight decoder, and reconstructed target image.
- visibleTextSnippet: Visible fragments include: "Are Scalable Vision Learners"; "Yanghao Li"; "Piotr Dollar"; "Ross Girshick"; "AI Research (FAIR)"; "input"; "encoder"; "decoder"; "target"; "Figure 1. Our MAE architecture. During pre-training, a large random...
- retrievalKeywordCount: `10`

### Row 3
- sourceDecisionRowId: `visual-retrieval-hint-expansion-decision:13761eef7ec3fe530d98`
- hintCandidateId: `visual-retrieval-hint:resnet-2015:figure_caption_region:6:beeb9fb9ad38518d`
- paperId: `resnet-2015`
- candidateType: `figure_caption_region`
- page: `6`
- currentDecision: `hold_pending_human_product_review`
- derivedTextForRetrievalSnippet: Retrieval hint only: ResNet Figure 5 compares a basic residual block for ResNet-34 with a bottleneck residual block for ResNet-50/101/152 on ImageNet, including identity shortcut connections and convolution stack dimensions.
- visibleTextSnippet: Visible fragments include: "64-d"; "3x3, 64"; "relu"; "3x3, 64"; "+"; "relu"; "256-d"; "1x1, 64"; "relu"; "3x3, 64"; "relu"; "1x1, 256"; "+"; "relu"; "Figure 5. A deeper residual function F for ImageNet. Left: a building block (on 56x56...
- retrievalKeywordCount: `10`

### Row 4
- sourceDecisionRowId: `visual-retrieval-hint-expansion-decision:5bd268b815e24a1a028c`
- hintCandidateId: `visual-retrieval-hint:clip-2021:figure_caption_region:3:7de4aa3a082925a4`
- paperId: `clip-2021`
- candidateType: `figure_caption_region`
- page: `3`
- currentDecision: `hold_pending_human_product_review`
- derivedTextForRetrievalSnippet: Retrieval hint only: CLIP Figure 2 line chart showing zero-shot ImageNet accuracy versus number of images processed; Bag of Words Contrastive (CLIP) is more efficient than Bag of Words Prediction and Transformer Language Model baselines,...
- visibleTextSnippet: Visible fragments include: "Learning Transferable Visual Models From Natural Language Supervision"; "Zero-Shot ImageNet Accuracy"; "# of images processed"; "2M"; "33M"; "67M"; "134M"; "268M"; "400M"; "4X efficiency"; "3X efficiency"; "Ba...
- retrievalKeywordCount: `10`

### Row 5
- sourceDecisionRowId: `visual-retrieval-hint-expansion-decision:13cbdac7b3dead9257ee`
- hintCandidateId: `visual-retrieval-hint:mae-2021:figure_caption_region:2:b1c1635f3ea9867d`
- paperId: `mae-2021`
- candidateType: `figure_caption_region`
- page: `2`
- currentDecision: `hold_pending_human_product_review`
- derivedTextForRetrievalSnippet: Retrieval hint only: MAE Figure 2 grid of ImageNet validation examples showing triplets of masked image, MAE reconstruction, and ground truth; caption states 80% masking ratio and 39 of 196 patches visible. A partial row of another examp...
- visibleTextSnippet: Visible fragments include: "Figure 2. Example results on ImageNet validation images. For each triplet, we show the masked image (left), our MAE reconstruction (middle), and the ground-truth (right). The masking ratio is 80%, leaving only...
- retrievalKeywordCount: `10`

### Row 6
- sourceDecisionRowId: `visual-retrieval-hint-expansion-decision:beac111660267e9fccb6`
- hintCandidateId: `visual-retrieval-hint:resnet-2015:figure_caption_region:8:5983f29673800b6b`
- paperId: `resnet-2015`
- candidateType: `figure_caption_region`
- page: `8`
- currentDecision: `hold_pending_human_product_review`
- derivedTextForRetrievalSnippet: Retrieval hint only: ResNet page crop containing part of Figure 6 CIFAR-10 training curves and Figure 7 layer-response standard deviation plots comparing plain networks and ResNets by original layer order and sorted magnitude.
- visibleTextSnippet: Visible fragments include: "error (%)"; "1e4"; "plain-20"; "plain-32"; "plain-44"; "plain-56"; "56-layer"; "20-layer"; "Figure 6. Training on CIFAR-10. Dashed lines denote training error, and bol..."; "of plain-110 is higher than 60% and...
- retrievalKeywordCount: `10`

### Row 7
- sourceDecisionRowId: `visual-retrieval-hint-expansion-decision:34698183e6ed4f83f286`
- hintCandidateId: `visual-retrieval-hint:clip-2021:figure_caption_region:5:aa1447f2ba9526ae`
- paperId: `clip-2021`
- candidateType: `figure_caption_region`
- page: `5`
- currentDecision: `hold_pending_human_product_review`
- derivedTextForRetrievalSnippet: Retrieval hint only: CLIP Figure 3 NumPy-like pseudocode for the core implementation: encode aligned images and texts, project and L2-normalize embeddings, compute scaled pairwise cosine similarities, and optimize symmetric cross-entropy...
- visibleTextSnippet: Visible fragments include: "# image_encoder - ResNet or Vision Transformer"; "# text_encoder - CBOW or Text Transformer"; "# I[n, h, w, c] - minibatch of aligned images"; "# T[n, l] - minibatch of aligned texts"; "# W_i[d_i, d_e] - learn...
- retrievalKeywordCount: `10`

### Row 8
- sourceDecisionRowId: `visual-retrieval-hint-expansion-decision:70478a24097b4a06b2dd`
- hintCandidateId: `visual-retrieval-hint:mae-2021:figure_caption_region:2:eea2a43fb3574de9`
- paperId: `mae-2021`
- candidateType: `figure_caption_region`
- page: `2`
- currentDecision: `hold_pending_human_product_review`
- derivedTextForRetrievalSnippet: Retrieval hint only: MAE page 2 crop showing continuation of ImageNet reconstruction examples plus Figure 3 COCO validation examples from an ImageNet-trained MAE, with captions emphasizing masked image, reconstruction, ground truth, and...
- visibleTextSnippet: Visible fragments include: "Figure 2. Example results on ImageNet validation images. For each triplet, we show the masked image (left), our MAE reconstruction (middle), and the ground-truth (right). The masking ratio is 80%, leaving only...
- retrievalKeywordCount: `10`

## JSON Shape To Return

```json
{
  "schema": "knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-gpt-decision-recommendation-output.v1",
  "rows": [
    {
      "sourceDecisionRowId": "visual-retrieval-hint-expansion-decision:c762f661957ea19c5690",
      "hintCandidateId": "visual-retrieval-hint:clip-2021:figure_caption_region:2:791bd02c68004e94",
      "sourceCandidateId": "visual-layout:clip-2021:figure_caption_region:2:b354160446b3a89c",
      "suggestedDecision": "FILL_IN_ONE_ALLOWED_DECISION",
      "recommendationRationale": "FILL_IN_SHORT_REASON",
      "risk": "FILL_IN_LOW_MEDIUM_HIGH_WITH_REASON",
      "needsHumanCheck": true,
      "finalHumanDecision": false,
      "applyAllowed": false,
      "candidateStoreWrite": false,
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false,
      "runtimeVisible": false,
      "indexEligible": false
    },
    {
      "sourceDecisionRowId": "visual-retrieval-hint-expansion-decision:0d7de4c618a6da6a92f4",
      "hintCandidateId": "visual-retrieval-hint:mae-2021:figure_caption_region:1:7f1d38a62ebc3ebe",
      "sourceCandidateId": "visual-layout:mae-2021:figure_caption_region:1:f3477cc8ce7ec8b1",
      "suggestedDecision": "FILL_IN_ONE_ALLOWED_DECISION",
      "recommendationRationale": "FILL_IN_SHORT_REASON",
      "risk": "FILL_IN_LOW_MEDIUM_HIGH_WITH_REASON",
      "needsHumanCheck": true,
      "finalHumanDecision": false,
      "applyAllowed": false,
      "candidateStoreWrite": false,
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false,
      "runtimeVisible": false,
      "indexEligible": false
    },
    {
      "sourceDecisionRowId": "visual-retrieval-hint-expansion-decision:13761eef7ec3fe530d98",
      "hintCandidateId": "visual-retrieval-hint:resnet-2015:figure_caption_region:6:beeb9fb9ad38518d",
      "sourceCandidateId": "visual-layout:resnet-2015:figure_caption_region:6:6b3dd4d05dc69888",
      "suggestedDecision": "FILL_IN_ONE_ALLOWED_DECISION",
      "recommendationRationale": "FILL_IN_SHORT_REASON",
      "risk": "FILL_IN_LOW_MEDIUM_HIGH_WITH_REASON",
      "needsHumanCheck": true,
      "finalHumanDecision": false,
      "applyAllowed": false,
      "candidateStoreWrite": false,
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false,
      "runtimeVisible": false,
      "indexEligible": false
    },
    {
      "sourceDecisionRowId": "visual-retrieval-hint-expansion-decision:5bd268b815e24a1a028c",
      "hintCandidateId": "visual-retrieval-hint:clip-2021:figure_caption_region:3:7de4aa3a082925a4",
      "sourceCandidateId": "visual-layout:clip-2021:figure_caption_region:3:fd2db71adcd98ec4",
      "suggestedDecision": "FILL_IN_ONE_ALLOWED_DECISION",
      "recommendationRationale": "FILL_IN_SHORT_REASON",
      "risk": "FILL_IN_LOW_MEDIUM_HIGH_WITH_REASON",
      "needsHumanCheck": true,
      "finalHumanDecision": false,
      "applyAllowed": false,
      "candidateStoreWrite": false,
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false,
      "runtimeVisible": false,
      "indexEligible": false
    },
    {
      "sourceDecisionRowId": "visual-retrieval-hint-expansion-decision:13cbdac7b3dead9257ee",
      "hintCandidateId": "visual-retrieval-hint:mae-2021:figure_caption_region:2:b1c1635f3ea9867d",
      "sourceCandidateId": "visual-layout:mae-2021:figure_caption_region:2:460a5652014c6369",
      "suggestedDecision": "FILL_IN_ONE_ALLOWED_DECISION",
      "recommendationRationale": "FILL_IN_SHORT_REASON",
      "risk": "FILL_IN_LOW_MEDIUM_HIGH_WITH_REASON",
      "needsHumanCheck": true,
      "finalHumanDecision": false,
      "applyAllowed": false,
      "candidateStoreWrite": false,
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false,
      "runtimeVisible": false,
      "indexEligible": false
    },
    {
      "sourceDecisionRowId": "visual-retrieval-hint-expansion-decision:beac111660267e9fccb6",
      "hintCandidateId": "visual-retrieval-hint:resnet-2015:figure_caption_region:8:5983f29673800b6b",
      "sourceCandidateId": "visual-layout:resnet-2015:figure_caption_region:8:5cf998f6a7a3739f",
      "suggestedDecision": "FILL_IN_ONE_ALLOWED_DECISION",
      "recommendationRationale": "FILL_IN_SHORT_REASON",
      "risk": "FILL_IN_LOW_MEDIUM_HIGH_WITH_REASON",
      "needsHumanCheck": true,
      "finalHumanDecision": false,
      "applyAllowed": false,
      "candidateStoreWrite": false,
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false,
      "runtimeVisible": false,
      "indexEligible": false
    },
    {
      "sourceDecisionRowId": "visual-retrieval-hint-expansion-decision:34698183e6ed4f83f286",
      "hintCandidateId": "visual-retrieval-hint:clip-2021:figure_caption_region:5:aa1447f2ba9526ae",
      "sourceCandidateId": "visual-layout:clip-2021:figure_caption_region:5:87777b9ff7911bbe",
      "suggestedDecision": "FILL_IN_ONE_ALLOWED_DECISION",
      "recommendationRationale": "FILL_IN_SHORT_REASON",
      "risk": "FILL_IN_LOW_MEDIUM_HIGH_WITH_REASON",
      "needsHumanCheck": true,
      "finalHumanDecision": false,
      "applyAllowed": false,
      "candidateStoreWrite": false,
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false,
      "runtimeVisible": false,
      "indexEligible": false
    },
    {
      "sourceDecisionRowId": "visual-retrieval-hint-expansion-decision:70478a24097b4a06b2dd",
      "hintCandidateId": "visual-retrieval-hint:mae-2021:figure_caption_region:2:eea2a43fb3574de9",
      "sourceCandidateId": "visual-layout:mae-2021:figure_caption_region:2:b35cefa766216e3f",
      "suggestedDecision": "FILL_IN_ONE_ALLOWED_DECISION",
      "recommendationRationale": "FILL_IN_SHORT_REASON",
      "risk": "FILL_IN_LOW_MEDIUM_HIGH_WITH_REASON",
      "needsHumanCheck": true,
      "finalHumanDecision": false,
      "applyAllowed": false,
      "candidateStoreWrite": false,
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false,
      "runtimeVisible": false,
      "indexEligible": false
    }
  ]
}
```
