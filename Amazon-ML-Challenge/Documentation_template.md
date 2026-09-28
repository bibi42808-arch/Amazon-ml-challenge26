# Entity Resolution Challenge - Methodology Documentation

## Methodology used

The implemented pipeline has two stages: (1) deterministic blocking creates a bounded set of plausible Source 2/Source 3 records per Source 1 row; (2) separate lightweight logistic classifiers score those candidate pairs. The training labels come only from the supplied `train_ground_truth.tsv`. The model uses no external data and no pretrained entity-resolution model.

## Candidate generation/blocking strategy

The cleaner preserves original fields and appends case-folded text fields after removing control characters and normalizing whitespace. Blocking applies Unicode NFKC normalization, case folding, punctuation/spacing normalization, and legal-suffix variants to names. Country is incorporated as a string in country-aware blocks and is never restricted to the training values.

Default candidate generation uses exact normalized name, country plus exact name, exact normalized address, country plus exact address, selected informative name/address tokens, token pairs, and short name-prefix blocks. SQLite indexes postings on disk; high-fanout blocks and token counts are bounded. The optional fast mode retains exact name and address blocks only. `candidate_pairs.tsv` is exported from the same candidate rows provided to the inference scorer.

## Model architecture and feature engineering

Source 2 and Source 3 have separate NumPy mini-batch logistic models. The 34 pair features cover exact name/address/country flags, character-bigram Dice, token Jaccard and overlap/containment, informative-name overlap, prefix/suffix and length ratios, address digit and long-number agreement, missing-field indicators, blocking-method flags, and per-Source-1 candidate count. Missing strings are handled as empty values.

Training retains every positive in the training partition and samples deterministic negatives. A stable hash of `source1_entity_id` creates a grouped holdout, preventing one Source 1 entity from appearing in both training and validation. Per-source thresholds are selected by sweeping the validation predictions for macro F0.5, with singleton rows included (empty truth and empty prediction score 1.0).

## Validation

Candidate recall is measured against the supplied training truth, separately for Source 2 and Source 3. Model validation reports grouped held-out macro F0.5, micro precision/recall, and selected thresholds. Test labels are unavailable and are not used for tuning.

## Any other relevant information about the approach

Only the provided challenge dataset is used. There are no external lookups, APIs, geocoding services, business registries, manual test answers, or pretrained entity databases. Output formatting is checked with the validator distributed in the supplied challenge ZIP.
