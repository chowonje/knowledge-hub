# Text Equation Locator Context Artifacts

- status: `ready`
- candidateRows: `9`
- labeledEquationRows: `2`
- equationLikeRows: `7`
- contextRows: `8`
- blockerRows: `1`
- privatePathLeakRows: `0`

## Paper Diagnostics

| paperId | candidates | labeled | context |
|---|---:|---:|---:|
| alexnet-2012 | 3 | 0 | 2 |
| resnet-2015 | 2 | 2 | 2 |
| clip-2021 | 4 | 0 | 4 |
| mae-2021 | 0 | 0 | 0 |

## Sample Candidates

- `alexnet-2012` `unlabeled` page `4` grade `equation_like_context` context `False`
- `alexnet-2012` `unlabeled` page `6` grade `equation_like_context` context `True`
- `alexnet-2012` `unlabeled` page `6` grade `equation_like_context` context `True`
- `resnet-2015` `Equation 2` page `3` grade `labeled_equation_context` context `True`
- `resnet-2015` `Equation 1` page `3` grade `labeled_equation_context` context `True`
- `clip-2021` `unlabeled` page `5` grade `equation_like_context` context `True`
- `clip-2021` `unlabeled` page `5` grade `equation_like_context` context `True`
- `clip-2021` `unlabeled` page `5` grade `equation_like_context` context `True`
- `clip-2021` `unlabeled` page `5` grade `equation_like_context` context `True`

## Blockers

- `mae-2021`: `equation_locator_not_found`
