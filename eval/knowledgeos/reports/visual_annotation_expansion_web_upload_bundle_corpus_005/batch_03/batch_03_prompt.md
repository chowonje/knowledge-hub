# Visual Annotation Expansion Batch 03

Use only the attached context-crop PNG files listed below. Do not use whole pages, whole images, or outside sources.
Return JSON only with top-level schema `knowledge-hub.paper.visual-annotation-web-output.v1` and a `rows` array.
Every row must keep `strictEvidence=false`, `citationGrade=false`, and `answerableWithoutTextEvidence=false`.
`derivedTextForRetrieval` is a retrieval hint only; it is not evidence and not answer-visible text.

## Attachments To Upload

- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/17-newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents-p4-equation_region-66dc218ed3.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/18-emu3.5-native-multimodal-models-are-world-learners-p2-equation_region-a0e33d60d3.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/19-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks-p5-equation_region-dbe0abfb6f.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/20-autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation-p4-equation_region-f3a9aecab1.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/21-arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts-p3-equation_region-dc3fd3a2d6.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/22-high-resolution-image-synthesis-with-latent-diffusion-models-p1-equation_region-aca1ff9b30.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/23-qwen-image-technical-report-p14-equation_region-a3d9713657.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/24-photorealistic-text-to-image-diffusion-models-with-deep-language-understanding-p20-equation_region-3538b1d2cc.png`

## Rows To Fill

| # | paperId | type | page | sourceCandidateId | attachmentRef |
|---:|---|---|---:|---|---|
| 1 | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | equation_region | 4 | `visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:equation_region:4:803dd3f7dbd45c6f` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/17-newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents-p4-equation_region-66dc218ed3.png` |
| 2 | emu3.5-native-multimodal-models-are-world-learners | equation_region | 2 | `visual-layout:emu3.5-native-multimodal-models-are-world-learners:equation_region:2:688127b9a1184640` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/18-emu3.5-native-multimodal-models-are-world-learners-p2-equation_region-a0e33d60d3.png` |
| 3 | faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks | equation_region | 5 | `visual-layout:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:equation_region:5:128a0a411f67a324` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/19-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks-p5-equation_region-dbe0abfb6f.png` |
| 4 | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | equation_region | 4 | `visual-layout:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:equation_region:4:45200729f88c94d7` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/20-autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation-p4-equation_region-f3a9aecab1.png` |
| 5 | arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts | equation_region | 3 | `visual-layout:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:equation_region:3:f2208953da94826f` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/21-arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts-p3-equation_region-dc3fd3a2d6.png` |
| 6 | high-resolution-image-synthesis-with-latent-diffusion-models | equation_region | 1 | `visual-layout:high-resolution-image-synthesis-with-latent-diffusion-models:equation_region:1:0dc11401a800bfef` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/22-high-resolution-image-synthesis-with-latent-diffusion-models-p1-equation_region-aca1ff9b30.png` |
| 7 | qwen-image-technical-report | equation_region | 14 | `visual-layout:qwen-image-technical-report:equation_region:14:1e8a7149fc3a29d8` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/23-qwen-image-technical-report-p14-equation_region-a3d9713657.png` |
| 8 | photorealistic-text-to-image-diffusion-models-with-deep-language-understanding | equation_region | 20 | `visual-layout:photorealistic-text-to-image-diffusion-models-with-deep-language-understanding:equation_region:20:2eb72e2516a0abf7` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/24-photorealistic-text-to-image-diffusion-models-with-deep-language-understanding-p20-equation_region-3538b1d2cc.png` |

## JSON Shape To Return

```json
{
  "schema": "knowledge-hub.paper.visual-annotation-web-output.v1",
  "rows": [
    {
      "sourceCandidateId": "visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:equation_region:4:803dd3f7dbd45c6f",
      "visualObservationStatus": "image_attached",
      "derivedTextForRetrieval": "Retrieval hint only: FILL_IN_FROM_ATTACHED_IMAGE.",
      "visibleText": "Visible fragments include: FILL_IN_VISIBLE_TEXT_ONLY.",
      "retrievalKeywords": [
        "FILL_IN_KEYWORD"
      ],
      "uncertainty": "FILL_IN_UNCERTAINTY.",
      "limitations": "Retrieval hint only, not evidence.",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    {
      "sourceCandidateId": "visual-layout:emu3.5-native-multimodal-models-are-world-learners:equation_region:2:688127b9a1184640",
      "visualObservationStatus": "image_attached",
      "derivedTextForRetrieval": "Retrieval hint only: FILL_IN_FROM_ATTACHED_IMAGE.",
      "visibleText": "Visible fragments include: FILL_IN_VISIBLE_TEXT_ONLY.",
      "retrievalKeywords": [
        "FILL_IN_KEYWORD"
      ],
      "uncertainty": "FILL_IN_UNCERTAINTY.",
      "limitations": "Retrieval hint only, not evidence.",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    {
      "sourceCandidateId": "visual-layout:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:equation_region:5:128a0a411f67a324",
      "visualObservationStatus": "image_attached",
      "derivedTextForRetrieval": "Retrieval hint only: FILL_IN_FROM_ATTACHED_IMAGE.",
      "visibleText": "Visible fragments include: FILL_IN_VISIBLE_TEXT_ONLY.",
      "retrievalKeywords": [
        "FILL_IN_KEYWORD"
      ],
      "uncertainty": "FILL_IN_UNCERTAINTY.",
      "limitations": "Retrieval hint only, not evidence.",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    {
      "sourceCandidateId": "visual-layout:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:equation_region:4:45200729f88c94d7",
      "visualObservationStatus": "image_attached",
      "derivedTextForRetrieval": "Retrieval hint only: FILL_IN_FROM_ATTACHED_IMAGE.",
      "visibleText": "Visible fragments include: FILL_IN_VISIBLE_TEXT_ONLY.",
      "retrievalKeywords": [
        "FILL_IN_KEYWORD"
      ],
      "uncertainty": "FILL_IN_UNCERTAINTY.",
      "limitations": "Retrieval hint only, not evidence.",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    {
      "sourceCandidateId": "visual-layout:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:equation_region:3:f2208953da94826f",
      "visualObservationStatus": "image_attached",
      "derivedTextForRetrieval": "Retrieval hint only: FILL_IN_FROM_ATTACHED_IMAGE.",
      "visibleText": "Visible fragments include: FILL_IN_VISIBLE_TEXT_ONLY.",
      "retrievalKeywords": [
        "FILL_IN_KEYWORD"
      ],
      "uncertainty": "FILL_IN_UNCERTAINTY.",
      "limitations": "Retrieval hint only, not evidence.",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    {
      "sourceCandidateId": "visual-layout:high-resolution-image-synthesis-with-latent-diffusion-models:equation_region:1:0dc11401a800bfef",
      "visualObservationStatus": "image_attached",
      "derivedTextForRetrieval": "Retrieval hint only: FILL_IN_FROM_ATTACHED_IMAGE.",
      "visibleText": "Visible fragments include: FILL_IN_VISIBLE_TEXT_ONLY.",
      "retrievalKeywords": [
        "FILL_IN_KEYWORD"
      ],
      "uncertainty": "FILL_IN_UNCERTAINTY.",
      "limitations": "Retrieval hint only, not evidence.",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    {
      "sourceCandidateId": "visual-layout:qwen-image-technical-report:equation_region:14:1e8a7149fc3a29d8",
      "visualObservationStatus": "image_attached",
      "derivedTextForRetrieval": "Retrieval hint only: FILL_IN_FROM_ATTACHED_IMAGE.",
      "visibleText": "Visible fragments include: FILL_IN_VISIBLE_TEXT_ONLY.",
      "retrievalKeywords": [
        "FILL_IN_KEYWORD"
      ],
      "uncertainty": "FILL_IN_UNCERTAINTY.",
      "limitations": "Retrieval hint only, not evidence.",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    {
      "sourceCandidateId": "visual-layout:photorealistic-text-to-image-diffusion-models-with-deep-language-understanding:equation_region:20:2eb72e2516a0abf7",
      "visualObservationStatus": "image_attached",
      "derivedTextForRetrieval": "Retrieval hint only: FILL_IN_FROM_ATTACHED_IMAGE.",
      "visibleText": "Visible fragments include: FILL_IN_VISIBLE_TEXT_ONLY.",
      "retrievalKeywords": [
        "FILL_IN_KEYWORD"
      ],
      "uncertainty": "FILL_IN_UNCERTAINTY.",
      "limitations": "Retrieval hint only, not evidence.",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    }
  ]
}
```
