# Visual Retrieval Hint Decision Recommendation Batch 03

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
| 1 | clip-2021 | table_region | 7 | `hold_pending_human_product_review` | `visual-retrieval-hint:clip-2021:table_region:7:202c4c0be3960b79` | `visual-layout:clip-2021:table_region:7:74d3127c812d1e9b` |
| 2 | mae-2021 | table_region | 5 | `hold_pending_human_product_review` | `visual-retrieval-hint:mae-2021:table_region:5:626e13443e1ee40c` | `visual-layout:mae-2021:table_region:5:09cee49d2841b6a3` |
| 3 | resnet-2015 | table_region | 6 | `hold_pending_human_product_review` | `visual-retrieval-hint:resnet-2015:table_region:6:7adc39075ec4efe2` | `visual-layout:resnet-2015:table_region:6:8b15bd4cbaa96ede` |
| 4 | clip-2021 | table_region | 17 | `hold_pending_human_product_review` | `visual-retrieval-hint:clip-2021:table_region:17:0c93a8dbedbc0a03` | `visual-layout:clip-2021:table_region:17:21f8887ab432c389` |
| 5 | mae-2021 | table_region | 5 | `hold_pending_human_product_review` | `visual-retrieval-hint:mae-2021:table_region:5:eeae9d1ecf9c1122` | `visual-layout:mae-2021:table_region:5:1179c62488dbb733` |
| 6 | clip-2021 | equation_region | 1 | `hold_pending_human_product_review` | `visual-retrieval-hint:clip-2021:equation_region:1:1578f0d850ce7830` | `visual-layout:clip-2021:equation_region:1:dd31b4b813b73af2` |
| 7 | clip-2021 | equation_region | 5 | `hold_pending_human_product_review` | `visual-retrieval-hint:clip-2021:equation_region:5:a85a9e64dfa04951` | `visual-layout:clip-2021:equation_region:5:1c4738912cb4bdf9` |
| 8 | clip-2021 | equation_region | 5 | `hold_pending_human_product_review` | `visual-retrieval-hint:clip-2021:equation_region:5:2d6d39c3bbc50090` | `visual-layout:clip-2021:equation_region:5:1fa7893d5e5d7b10` |

## Row Context

### Row 1
- sourceDecisionRowId: `visual-retrieval-hint-expansion-decision:db120d08b4cac0dbd6db`
- hintCandidateId: `visual-retrieval-hint:clip-2021:table_region:7:202c4c0be3960b79`
- paperId: `clip-2021`
- candidateType: `table_region`
- page: `7`
- currentDecision: `hold_pending_human_product_review`
- derivedTextForRetrievalSnippet: Retrieval hint only: CLIP Table 1 compares prior zero-shot transfer image classification results with Visual N-Grams on aYahoo, ImageNet, and SUN. Visible scores are Visual N-Grams 72.4, 11.5, 23.0 and CLIP 98.4, 76.2, 58.5. The caption...
- visibleTextSnippet: Visible fragments include: paper header “Learning Transferable Visual Models From Natural Language Supervision”; table columns “aYahoo”, “ImageNet”, “SUN”; rows “Visual N-Grams 72.4 11.5 23.0” and “CLIP 98.4 76.2 58.5”; caption beginning...
- retrievalKeywordCount: `10`

### Row 2
- sourceDecisionRowId: `visual-retrieval-hint-expansion-decision:abfd5b806f9806ea46f1`
- hintCandidateId: `visual-retrieval-hint:mae-2021:table_region:5:626e13443e1ee40c`
- paperId: `mae-2021`
- candidateType: `table_region`
- page: `5`
- currentDecision: `hold_pending_human_product_review`
- derivedTextForRetrievalSnippet: Retrieval hint only: MAE Table 1 shows ablation experiments with ViT-L/16 on ImageNet-1K, reporting fine-tuning and linear probing accuracy. Visible subpanels cover reconstruction target, data augmentation, and mask sampling. Reconstruct...
- visibleTextSnippet: Visible fragments include: subtable “(d) Reconstruction target” with columns “case”, “ft”, “lin”; rows “pixel (w/o norm) 84.9 73.5”, “pixel (w/ norm) 85.4 73.9”, “PCA 84.6 72.3”, “dVAE token 85.3 71.6”; subtable “(e) Data augmentation” w...
- retrievalKeywordCount: `11`

### Row 3
- sourceDecisionRowId: `visual-retrieval-hint-expansion-decision:eb537adfee4af2e90637`
- hintCandidateId: `visual-retrieval-hint:resnet-2015:table_region:6:7adc39075ec4efe2`
- paperId: `resnet-2015`
- candidateType: `table_region`
- page: `6`
- currentDecision: `hold_pending_human_product_review`
- derivedTextForRetrievalSnippet: Retrieval hint only: ResNet Table 5 reports ImageNet test-set top-5 error rates for ensembles. The methods and visible top-5 error values are VGG ILSVRC’14 7.32, GoogLeNet ILSVRC’14 6.66, VGG v5 6.8, PReLU-net 4.94, BN-inception 4.82, an...
- visibleTextSnippet: Visible fragments include: table columns “method” and “top-5 err. (test)”; rows “VGG [41] (ILSVRC’14) 7.32”, “GoogLeNet [44] (ILSVRC’14) 6.66”, “VGG [41] (v5) 6.8”, “PReLU-net [13] 4.94”, “BN-inception [16] 4.82”, “ResNet (ILSVRC’15) 3.5...
- retrievalKeywordCount: `11`

### Row 4
- sourceDecisionRowId: `visual-retrieval-hint-expansion-decision:cf9e699ce8bb3a923a6c`
- hintCandidateId: `visual-retrieval-hint:clip-2021:table_region:17:0c93a8dbedbc0a03`
- paperId: `clip-2021`
- candidateType: `table_region`
- page: `17`
- currentDecision: `hold_pending_human_product_review`
- derivedTextForRetrievalSnippet: Retrieval hint only: CLIP Table 2 compares human performance and zero-shot CLIP on Oxford IIT Pets. Columns are Accuracy, Majority Vote on Full Dataset, Accuracy on Guesses, and Majority Vote Accuracy on Guesses. Rows show zero-shot huma...
- visibleTextSnippet: Visible fragments include: table columns “Accuracy”, “Majority Vote on Full Dataset”, “Accuracy on Guesses”, “Majority Vote Accuracy on Guesses”; rows “Zero-shot human 53.7 57.0 69.7 63.9”, “Zero-shot CLIP 93.5 93.5 93.5 93.5”, “One-shot...
- retrievalKeywordCount: `11`

### Row 5
- sourceDecisionRowId: `visual-retrieval-hint-expansion-decision:1823d9c2e5740eb6bafd`
- hintCandidateId: `visual-retrieval-hint:mae-2021:table_region:5:eeae9d1ecf9c1122`
- paperId: `mae-2021`
- candidateType: `table_region`
- page: `5`
- currentDecision: `hold_pending_human_product_review`
- derivedTextForRetrievalSnippet: Retrieval hint only: MAE Table 2 reports wall-clock training time for MAE training over 800 epochs on 128 TPU-v3 cores with TensorFlow. Columns are encoder, decoder depth, fine-tuning accuracy, hours, and speedup. Visible rows include Vi...
- visibleTextSnippet: Visible fragments include: table columns “encoder”, “dec. depth”, “ft acc”, “hours”, “speedup”; rows “ViT-L, w/ [M] 8 84.2 42.4 -”, “ViT-L 8 84.9 15.4 2.8×”, “ViT-L 1 84.8 11.6 3.7×”, “ViT-H, w/ [M] 8 - 119.6† -”, “ViT-H 8 85.8 34.5 3.5×...
- retrievalKeywordCount: `11`

### Row 6
- sourceDecisionRowId: `visual-retrieval-hint-expansion-decision:807cba0239437c743ee9`
- hintCandidateId: `visual-retrieval-hint:clip-2021:equation_region:1:1578f0d850ce7830`
- paperId: `clip-2021`
- candidateType: `equation_region`
- page: `1`
- currentDecision: `hold_pending_human_product_review`
- derivedTextForRetrievalSnippet: Retrieval hint only: CLIP page 1 shows the section heading Introduction and Motivating Work, an opening paragraph about pre-training methods that learn directly from raw text revolutionizing NLP, and an author footnote for equal contribu...
- visibleTextSnippet: Visible fragments include: section title “1. Introduction and Motivating Work”; paragraph “Pre-training methods which learn directly from raw text have revolutionized NLP over the last few years” followed by citations including “Dai & Le...
- retrievalKeywordCount: `11`

### Row 7
- sourceDecisionRowId: `visual-retrieval-hint-expansion-decision:01845f08da4cf9d4e8a2`
- hintCandidateId: `visual-retrieval-hint:clip-2021:equation_region:5:a85a9e64dfa04951`
- paperId: `clip-2021`
- candidateType: `equation_region`
- page: `5`
- currentDecision: `hold_pending_human_product_review`
- derivedTextForRetrievalSnippet: Retrieval hint only: CLIP Figure 3 lower pseudocode shows the learned temperature parameter, extraction of image and text features, projection into a joint multimodal embedding, L2 normalization, scaled pairwise cosine similarities via l...
- visibleTextSnippet: Visible fragments include: “# t - learned temperature parameter”; “# extract feature representations of each modality”; “I_f = image_encoder(I) #[n, d_i]”; “T_f = text_encoder(T) #[n, d_t]”; “# joint multimodal embedding [n, d_e]”; “I_e...
- retrievalKeywordCount: `11`

### Row 8
- sourceDecisionRowId: `visual-retrieval-hint-expansion-decision:b24230b3a36d5cd1fbf3`
- hintCandidateId: `visual-retrieval-hint:clip-2021:equation_region:5:2d6d39c3bbc50090`
- paperId: `clip-2021`
- candidateType: `equation_region`
- page: `5`
- currentDecision: `hold_pending_human_product_review`
- derivedTextForRetrievalSnippet: Retrieval hint only: CLIP Figure 3 pseudocode defines image_encoder as ResNet or Vision Transformer, text_encoder as CBOW or Text Transformer, image minibatch I[n,h,w,c], text minibatch T[n,l], projection matrices W_i and W_t, and learne...
- visibleTextSnippet: Visible fragments include: paper header “Learning Transferable Visual Models From Natural Language Supervision”; code comments “# image_encoder - ResNet or Vision Transformer”, “# text_encoder - CBOW or Text Transformer”, “# I[n, h, w, c...
- retrievalKeywordCount: `12`

## JSON Shape To Return

```json
{
  "schema": "knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-gpt-decision-recommendation-output.v1",
  "rows": [
    {
      "sourceDecisionRowId": "visual-retrieval-hint-expansion-decision:db120d08b4cac0dbd6db",
      "hintCandidateId": "visual-retrieval-hint:clip-2021:table_region:7:202c4c0be3960b79",
      "sourceCandidateId": "visual-layout:clip-2021:table_region:7:74d3127c812d1e9b",
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
      "sourceDecisionRowId": "visual-retrieval-hint-expansion-decision:abfd5b806f9806ea46f1",
      "hintCandidateId": "visual-retrieval-hint:mae-2021:table_region:5:626e13443e1ee40c",
      "sourceCandidateId": "visual-layout:mae-2021:table_region:5:09cee49d2841b6a3",
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
      "sourceDecisionRowId": "visual-retrieval-hint-expansion-decision:eb537adfee4af2e90637",
      "hintCandidateId": "visual-retrieval-hint:resnet-2015:table_region:6:7adc39075ec4efe2",
      "sourceCandidateId": "visual-layout:resnet-2015:table_region:6:8b15bd4cbaa96ede",
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
      "sourceDecisionRowId": "visual-retrieval-hint-expansion-decision:cf9e699ce8bb3a923a6c",
      "hintCandidateId": "visual-retrieval-hint:clip-2021:table_region:17:0c93a8dbedbc0a03",
      "sourceCandidateId": "visual-layout:clip-2021:table_region:17:21f8887ab432c389",
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
      "sourceDecisionRowId": "visual-retrieval-hint-expansion-decision:1823d9c2e5740eb6bafd",
      "hintCandidateId": "visual-retrieval-hint:mae-2021:table_region:5:eeae9d1ecf9c1122",
      "sourceCandidateId": "visual-layout:mae-2021:table_region:5:1179c62488dbb733",
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
      "sourceDecisionRowId": "visual-retrieval-hint-expansion-decision:807cba0239437c743ee9",
      "hintCandidateId": "visual-retrieval-hint:clip-2021:equation_region:1:1578f0d850ce7830",
      "sourceCandidateId": "visual-layout:clip-2021:equation_region:1:dd31b4b813b73af2",
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
      "sourceDecisionRowId": "visual-retrieval-hint-expansion-decision:01845f08da4cf9d4e8a2",
      "hintCandidateId": "visual-retrieval-hint:clip-2021:equation_region:5:a85a9e64dfa04951",
      "sourceCandidateId": "visual-layout:clip-2021:equation_region:5:1c4738912cb4bdf9",
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
      "sourceDecisionRowId": "visual-retrieval-hint-expansion-decision:b24230b3a36d5cd1fbf3",
      "hintCandidateId": "visual-retrieval-hint:clip-2021:equation_region:5:2d6d39c3bbc50090",
      "sourceCandidateId": "visual-layout:clip-2021:equation_region:5:1fa7893d5e5d7b10",
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
