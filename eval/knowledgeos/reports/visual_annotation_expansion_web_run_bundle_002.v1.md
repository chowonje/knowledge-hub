# Visual Annotation Expansion Web Run Bundle 002

- schema: `knowledge-hub.paper.visual-annotation-expansion-web-run-bundle.v1`
- status: `ready`
- decision: `ready_for_operator_web_batch_run`
- generatedAt: `2026-05-26T14:30:40Z`
- bundleId: `visual_annotation_expansion_web_run_bundle_002`
- sourceWebOutputTemplate: `eval/knowledgeos/reports/visual_annotation_expansion_web_output_template_002.v1.json`
- targetOutputRef: `eval/knowledgeos/reports/visual_annotation_expansion_web_output_002.manual.json`
- validationCommand: `PYTHONPATH=. python eval/knowledgeos/scripts/validate_visual_annotation_expansion_web_output.py`
- sourceTemplateRows: `24`
- batchRows: `3`
- bundleArtifactRows: `6`
- completedWebOutputRows: `0`
- privatePathLeakRows: `0`

## Mutation Guarantees

- writes: `report_only`
- apiCalls: `False`
- modelCalls: `False`
- webModelCalls: `False`
- operatorBatchPromptRows: `3`
- completedWebOutputRows: `0`
- vectorIndexing: `False`
- strictEvidencePromotionRows: `0`
- runtimeAnswerVisibleExposureRows: `0`
- candidateStoreMutationRows: `0`
- wholeImageGptRows: `0`

## Batch Bundles

| batch | rows | promptRef | fillTemplateRef |
|---:|---:|---|---|
| 1 | 8 | `eval/knowledgeos/reports/visual_annotation_expansion_web_run_bundle_002/batch_01_prompt.md` | `eval/knowledgeos/reports/visual_annotation_expansion_web_run_bundle_002/batch_01_fill_template.v1.json` |
| 2 | 8 | `eval/knowledgeos/reports/visual_annotation_expansion_web_run_bundle_002/batch_02_prompt.md` | `eval/knowledgeos/reports/visual_annotation_expansion_web_run_bundle_002/batch_02_fill_template.v1.json` |
| 3 | 8 | `eval/knowledgeos/reports/visual_annotation_expansion_web_run_bundle_002/batch_03_prompt.md` | `eval/knowledgeos/reports/visual_annotation_expansion_web_run_bundle_002/batch_03_fill_template.v1.json` |

## Operator Instructions

1. Run one batch at a time in web GPT/Pro.
2. For each batch, upload only the attachment refs listed in that batch prompt.
3. Paste the batch prompt and use the batch fill template only as a shape guide.
4. After all batches return JSON rows, combine the rows into the target output file.
5. Validate the combined file with: PYTHONPATH=. python eval/knowledgeos/scripts/validate_visual_annotation_expansion_web_output.py

## Warnings

- `This bundle is not web/VLM output and contains placeholder text.`
- `Batch fill templates use a batch-template schema, not the completed web-output schema.`
- `Do not upload whole pages or whole images for this gate.`
- `Do not treat derivedTextForRetrieval as strict, citation-grade, or answer-visible evidence.`
