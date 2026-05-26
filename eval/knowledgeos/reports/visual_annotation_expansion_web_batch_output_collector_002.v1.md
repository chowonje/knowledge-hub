# Visual Annotation Expansion Web Batch Output Collector 002

- schema: `knowledge-hub.paper.visual-annotation-expansion-web-batch-output-collector.v1`
- status: `ready`
- decision: `ready_for_combined_manual_output_validation`
- generatedAt: `2026-05-26T15:32:06Z`
- collectorId: `visual_annotation_expansion_web_batch_output_collector_002`
- sourceWebRunBundle: `eval/knowledgeos/reports/visual_annotation_expansion_web_run_bundle_002.v1.json`
- targetOutputRef: `eval/knowledgeos/reports/visual_annotation_expansion_web_output_002.manual.json`
- validationCommand: `PYTHONPATH=. python eval/knowledgeos/scripts/validate_visual_annotation_expansion_web_output.py`
- expectedBatchRows: `3`
- presentBatchRows: `3`
- missingBatchRows: `0`
- validBatchRows: `3`
- collectedOutputRows: `24`
- blockedRows: `0`

## Mutation Guarantees

- writes: `report_only`
- apiCalls: `False`
- modelCalls: `False`
- webModelCalls: `False`
- manualWebModelOutputRows: `24`
- combinedOutputWriteRows: `0`
- vectorIndexing: `False`
- strictEvidencePromotionRows: `0`
- runtimeAnswerVisibleExposureRows: `0`
- candidateStoreMutationRows: `0`

## Expected Batch Outputs

| batch | present | status | expected | output | outputRef | blockers |
|---:|---|---|---:|---:|---|---|
| 1 | `True` | `ready` | 8 | 8 | `eval/knowledgeos/reports/visual_annotation_expansion_web_batch_outputs_002/batch_01_web_output.manual.json` | - |
| 2 | `True` | `ready` | 8 | 8 | `eval/knowledgeos/reports/visual_annotation_expansion_web_batch_outputs_002/batch_02_web_output.manual.json` | - |
| 3 | `True` | `ready` | 8 | 8 | `eval/knowledgeos/reports/visual_annotation_expansion_web_batch_outputs_002/batch_03_web_output.manual.json` | - |

## Operator Instructions

1. Paste each web GPT/Pro batch result into the matching batch output file.
2. Expected files live under eval/knowledgeos/reports/visual_annotation_expansion_web_batch_outputs_002.
3. After all batch outputs are valid, combine their rows into eval/knowledgeos/reports/visual_annotation_expansion_web_output_002.manual.json.
4. Then run validation with: PYTHONPATH=. python eval/knowledgeos/scripts/validate_visual_annotation_expansion_web_output.py

## Warnings

- `This collector report is not a completed web/VLM output.`
- `Batch fill-template files must not be used as batch outputs.`
- `Rows containing FILL_IN placeholders remain blocked.`
- `Do not treat derivedTextForRetrieval as strict, citation-grade, or answer-visible evidence.`
