# Visual Retrieval Hint Decision Recommendation Batch 01

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
| 1 | clip-2021 | image_region | 2 | `hold_pending_human_product_review` | `visual-retrieval-hint:clip-2021:image_region:2:e25902a7066b85fb` | `visual-layout:clip-2021:image_region:2:2616032b2a9d352d` |
| 2 | mae-2021 | image_region | 1 | `hold_pending_human_product_review` | `visual-retrieval-hint:mae-2021:image_region:1:d54434235dc1f73e` | `visual-layout:mae-2021:image_region:1:258a60cc216d8130` |
| 3 | alexnet-2012 | image_region | 6 | `hold_pending_human_product_review` | `visual-retrieval-hint:alexnet-2012:image_region:6:6bc72fab3308cc56` | `visual-layout:alexnet-2012:image_region:6:508eb98021de0fd7` |
| 4 | clip-2021 | image_region | 2 | `hold_pending_human_product_review` | `visual-retrieval-hint:clip-2021:image_region:2:0d674a922cfdab51` | `visual-layout:clip-2021:image_region:2:2642c8db8bd29d0a` |
| 5 | mae-2021 | image_region | 1 | `hold_pending_human_product_review` | `visual-retrieval-hint:mae-2021:image_region:1:e3eb86ab7dd4815d` | `visual-layout:mae-2021:image_region:1:40b27e8b4073d98f` |
| 6 | alexnet-2012 | image_region | 8 | `hold_pending_human_product_review` | `visual-retrieval-hint:alexnet-2012:image_region:8:aebea8a49393b7e9` | `visual-layout:alexnet-2012:image_region:8:219e7594a10f3a3f` |
| 7 | clip-2021 | image_region | 15 | `hold_pending_human_product_review` | `visual-retrieval-hint:clip-2021:image_region:15:b2eef0626d7910cf` | `visual-layout:clip-2021:image_region:15:1da1f1cf8ce81bd5` |
| 8 | mae-2021 | image_region | 2 | `hold_pending_human_product_review` | `visual-retrieval-hint:mae-2021:image_region:2:47e41fe986d634ac` | `visual-layout:mae-2021:image_region:2:0b66280633a636cc` |

## Row Context

### Row 1
- sourceDecisionRowId: `visual-retrieval-hint-expansion-decision:6eb061dfeece81c4626f`
- hintCandidateId: `visual-retrieval-hint:clip-2021:image_region:2:e25902a7066b85fb`
- paperId: `clip-2021`
- candidateType: `image_region`
- page: `2`
- currentDecision: `hold_pending_human_product_review`
- derivedTextForRetrievalSnippet: Retrieval hint only: Cropped CLIP Figure 1 panels for creating a dataset classifier from text labels and using it for zero-shot prediction; label list flows into the prompt template “A photo of a {object}.” and an image of a dog flows in...
- visibleTextSnippet: Visible fragments include: “(2) Create dataset classifier from labe”; “plane”; “car”; “dog”; “bird”; “A photo of a {object}.”; “(3) Use for zero-shot prediction”; “Image Encoder”; fragments of caption/body text including “jointly train a...
- retrievalKeywordCount: `6`

### Row 2
- sourceDecisionRowId: `visual-retrieval-hint-expansion-decision:df43a4bfbe69d6bcb4a1`
- hintCandidateId: `visual-retrieval-hint:mae-2021:image_region:1:d54434235dc1f73e`
- paperId: `mae-2021`
- candidateType: `image_region`
- page: `1`
- currentDecision: `hold_pending_human_product_review`
- derivedTextForRetrievalSnippet: Retrieval hint only: Cropped MAE architecture figure showing a masked input image grid, a column of visible patches, an encoder block, and blue latent/output tokens; title/author area and Figure 1 caption are partially visible.
- visibleTextSnippet: Visible fragments include: “...nghao Li”; “Piotr Dollá...”; “†project lead”; “...ch (FAIR)”; “input”; “encoder”; “...gure 1.”; “Our MAE architectu...”; “...dom subset of image patches”.
- retrievalKeywordCount: `6`

### Row 3
- sourceDecisionRowId: `visual-retrieval-hint-expansion-decision:b559f2041324d26c0de5`
- hintCandidateId: `visual-retrieval-hint:alexnet-2012:image_region:6:6bc72fab3308cc56`
- paperId: `alexnet-2012`
- candidateType: `image_region`
- page: `6`
- currentDecision: `hold_pending_human_product_review`
- derivedTextForRetrievalSnippet: Retrieval hint only: Cropped AlexNet Figure 3 showing a grid of 96 first-layer convolutional kernels, mostly edge/color filters, with caption describing 11×11×3 kernels learned from 224×224×3 inputs and split across GPU 1 and GPU 2.
- visibleTextSnippet: Visible fragments include: “Without dropout, our network ex-”; “...oubles the number of iterations required to converge.”; “Figure 3: 96 convolutional kernels of size 11×11×3 learned by the first convolutional layer on the 224×224×3 inpu...
- retrievalKeywordCount: `7`

### Row 4
- sourceDecisionRowId: `visual-retrieval-hint-expansion-decision:5b4b81e34fa9b65bc885`
- hintCandidateId: `visual-retrieval-hint:clip-2021:image_region:2:0d674a922cfdab51`
- paperId: `clip-2021`
- candidateType: `image_region`
- page: `2`
- currentDecision: `hold_pending_human_product_review`
- derivedTextForRetrievalSnippet: Retrieval hint only: Cropped CLIP Figure 1 contrastive pre-training panel showing stacked text prompts, a stacked dog image, and arrows into partially visible Text Encoder and Image Encoder blocks.
- visibleTextSnippet: Visible fragments include: “(1) Contrastive pre-training”; “Pepper the aussie pup”; “Text Enco...” or “Text Encoder” partially cropped; “Image Enco...” or “Image Encoder” partially cropped; “Figure 1. Summary of ou...”; “some label, CLIP...
- retrievalKeywordCount: `6`

### Row 5
- sourceDecisionRowId: `visual-retrieval-hint-expansion-decision:690182582317bc8522c1`
- hintCandidateId: `visual-retrieval-hint:mae-2021:image_region:1:e3eb86ab7dd4815d`
- paperId: `mae-2021`
- candidateType: `image_region`
- page: `1`
- currentDecision: `hold_pending_human_product_review`
- derivedTextForRetrievalSnippet: Retrieval hint only: Cropped MAE Figure 1 architecture close-up showing a masked image patch grid labeled input, selected visible image patches, a gray encoder block, and a vertical stack of blue tokens.
- visibleTextSnippet: Visible fragments include: “...nghao Li”; “Piotr Dollá...”; “†project lead”; “...ch (FAIR)”; “input”; “encoder”; “...ure 1.”; “Our MAE architectu...”; “...dom subset of image patches”; “...coder is applied to the small”.
- retrievalKeywordCount: `7`

### Row 6
- sourceDecisionRowId: `visual-retrieval-hint-expansion-decision:b81e3df56b0d667ab017`
- hintCandidateId: `visual-retrieval-hint:alexnet-2012:image_region:8:aebea8a49393b7e9`
- paperId: `alexnet-2012`
- candidateType: `image_region`
- page: `8`
- currentDecision: `hold_pending_human_product_review`
- derivedTextForRetrievalSnippet: Retrieval hint only: Cropped AlexNet Figure 4 showing ILSVRC-2010 test images with top-5 prediction bar charts on the left and a partial nearest-neighbor/training-image panel on the right; visible classes include mite, container ship, mo...
- visibleTextSnippet: Visible fragments include: “mite”; “container ship”; “motor scooter”; “leopard”; “grille”; “mushroom”; “cherry”; “Madagascar cat”; prediction labels such as “black widow”, “cockroach”, “tick”, “starfish”, “lifeboat”, “amphibian”, “firebo...
- retrievalKeywordCount: `8`

### Row 7
- sourceDecisionRowId: `visual-retrieval-hint-expansion-decision:15f9df5f512fed72e7ca`
- hintCandidateId: `visual-retrieval-hint:clip-2021:image_region:15:b2eef0626d7910cf`
- paperId: `clip-2021`
- candidateType: `image_region`
- page: `15`
- currentDecision: `hold_pending_human_product_review`
- derivedTextForRetrievalSnippet: Retrieval hint only: Cropped CLIP appendix/section figure under “Language Supervision” showing dataset-example grids for banana-like images across rows with partially visible dataset names and a heading about distribution shift.
- visibleTextSnippet: Visible fragments include: “Language Supervision”; “Dataset Example”; partially cropped row-label fragments including “...geNet”, “...NetV2”, “...Net-R”, “...ctNet”, “...geNet Sketch”, and “...Net-A”; bottom fragments include “distributi...
- retrievalKeywordCount: `10`

### Row 8
- sourceDecisionRowId: `visual-retrieval-hint-expansion-decision:c8d7d60056f58534cc8b`
- hintCandidateId: `visual-retrieval-hint:mae-2021:image_region:2:47e41fe986d634ac`
- paperId: `mae-2021`
- candidateType: `image_region`
- page: `2`
- currentDecision: `hold_pending_human_product_review`
- derivedTextForRetrievalSnippet: Retrieval hint only: Cropped MAE reconstruction examples showing rows of original/masked/reconstructed images including a bird, butterfly, horn player, coffee cups, bears, harbor scene, escalator, church, horse rider, people, street scen...
- visibleTextSnippet: Visible fragments include: “masked image (left), our MAE reconstruction†”; “96 patches. More examples are in the appendix.”; “One can simply overlay the output with the visible”; “demonstrate the method’s behavior.”
- retrievalKeywordCount: `7`

## JSON Shape To Return

```json
{
  "schema": "knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-gpt-decision-recommendation-output.v1",
  "rows": [
    {
      "sourceDecisionRowId": "visual-retrieval-hint-expansion-decision:6eb061dfeece81c4626f",
      "hintCandidateId": "visual-retrieval-hint:clip-2021:image_region:2:e25902a7066b85fb",
      "sourceCandidateId": "visual-layout:clip-2021:image_region:2:2616032b2a9d352d",
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
      "sourceDecisionRowId": "visual-retrieval-hint-expansion-decision:df43a4bfbe69d6bcb4a1",
      "hintCandidateId": "visual-retrieval-hint:mae-2021:image_region:1:d54434235dc1f73e",
      "sourceCandidateId": "visual-layout:mae-2021:image_region:1:258a60cc216d8130",
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
      "sourceDecisionRowId": "visual-retrieval-hint-expansion-decision:b559f2041324d26c0de5",
      "hintCandidateId": "visual-retrieval-hint:alexnet-2012:image_region:6:6bc72fab3308cc56",
      "sourceCandidateId": "visual-layout:alexnet-2012:image_region:6:508eb98021de0fd7",
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
      "sourceDecisionRowId": "visual-retrieval-hint-expansion-decision:5b4b81e34fa9b65bc885",
      "hintCandidateId": "visual-retrieval-hint:clip-2021:image_region:2:0d674a922cfdab51",
      "sourceCandidateId": "visual-layout:clip-2021:image_region:2:2642c8db8bd29d0a",
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
      "sourceDecisionRowId": "visual-retrieval-hint-expansion-decision:690182582317bc8522c1",
      "hintCandidateId": "visual-retrieval-hint:mae-2021:image_region:1:e3eb86ab7dd4815d",
      "sourceCandidateId": "visual-layout:mae-2021:image_region:1:40b27e8b4073d98f",
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
      "sourceDecisionRowId": "visual-retrieval-hint-expansion-decision:b81e3df56b0d667ab017",
      "hintCandidateId": "visual-retrieval-hint:alexnet-2012:image_region:8:aebea8a49393b7e9",
      "sourceCandidateId": "visual-layout:alexnet-2012:image_region:8:219e7594a10f3a3f",
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
      "sourceDecisionRowId": "visual-retrieval-hint-expansion-decision:15f9df5f512fed72e7ca",
      "hintCandidateId": "visual-retrieval-hint:clip-2021:image_region:15:b2eef0626d7910cf",
      "sourceCandidateId": "visual-layout:clip-2021:image_region:15:1da1f1cf8ce81bd5",
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
      "sourceDecisionRowId": "visual-retrieval-hint-expansion-decision:c8d7d60056f58534cc8b",
      "hintCandidateId": "visual-retrieval-hint:mae-2021:image_region:2:47e41fe986d634ac",
      "sourceCandidateId": "visual-layout:mae-2021:image_region:2:0b66280633a636cc",
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
