# Licence for the released pairs and derived text

The pair files in `data/` contain passages and queries taken or adapted from
CUREv1 (https://huggingface.co/datasets/clinia/CUREv1). The CUREv1 dataset card
states the licence as Creative Commons Attribution Non Commercial 4.0
(CC BY-NC 4.0). The pair files are therefore released under
**CC BY-NC 4.0** (https://creativecommons.org/licenses/by-nc/4.0/).

Conditions that follow from this:

1. Attribution. Cite CUREv1 and this collection (see `CITATION.cff`).
2. Non-commercial use only.
3. Changes are indicated. Each `negative_passage` is a modified copy of the
   CUREv1 passage in `positive_passage`, edited at the span recorded in
   `source_span` and `replacement_span`. Each `counterfactual_query` is a
   rewritten version of the CUREv1 query in `original_query`.

The edited passages are synthetic and are not medical information.

Maintainer's note before release: the CUREv1 card does not spell out terms for
redistribution of derived text. Confirm with the CUREv1 authors if there is any
doubt, and record their answer here.
