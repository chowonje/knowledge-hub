# Text Complex QA Eval Alignment

- status: `ready`
- caseRows: `10`
- textAnswerableRows: `3`
- candidateOnlyRows: `3`
- visualUnsupportedRows: `2`
- noAnswerRows: `2`
- privatePathLeakRows: `0`

## Cases

| caseId | category | disposition | blocker |
|---|---|---|---|
| `figure-caption-alexnet-figure-1-caption` | `figure_caption_qa` | `text_answerable` | `` |
| `figure-caption-alexnet-figure-1-visual-detail` | `figure_caption_qa` | `visual_unsupported` | `visual_reasoning_not_supported_in_text_evidence_v01` |
| `table-numeric-alexnet-table-1-result` | `table_numeric_qa` | `candidate_only` | `table_cell_identity_not_available` |
| `table-numeric-resnet-table-3-error-rate` | `table_numeric_qa` | `candidate_only` | `table_cell_identity_not_available` |
| `equation-resnet-equation-1-residual` | `equation_citation_qa` | `candidate_only` | `equation_latex_reconstruction_not_available` |
| `equation-mae-equation-1-missing` | `equation_citation_qa` | `no_answer` | `equation_locator_not_found` |
| `method-resnet-residual-learning-text` | `method_comparison_qa` | `text_answerable` | `` |
| `method-mae-masked-autoencoder-text` | `method_comparison_qa` | `text_answerable` | `` |
| `limitation-visual-chart-inspection-v0-1` | `limitation_qa` | `visual_unsupported` | `visual_reasoning_not_supported_in_text_evidence_v01` |
| `limitation-missing-table-99-no-answer` | `limitation_qa` | `no_answer` | `table_candidate_not_found` |
