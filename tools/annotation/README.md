# Human audit tools

1. `annotation_template.xlsx`: practice workbook with English and Chinese instructions and four worked examples.
2. `build_annotation_sheet.py`: builds the real sample (stratified by source and language) as `annotation_A.xlsx`, `annotation_B.xlsx` (overlap subset in a different order) and the hidden `annotation_key.csv`. Do not share the key with annotators and do not commit it.
3. `agreement.py`: reads the completed workbooks and the key; writes agreement, kappa, acceptance rates with exact intervals, and the list of accepted pair identifiers.
4. `rename_screening_fields.py`: makes the release copy of a pair file (screening_status, screening_method, human_adjudication). Never run it on files the evaluation script reads.

Answer codes: 1 yes, 0 no, ? cannot judge. See each script's `--help`.
