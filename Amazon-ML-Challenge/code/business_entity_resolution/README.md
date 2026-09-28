# Business Entity Resolution Pipeline

This is a reproducible, local-only implementation of the AWS ML Challenge 2026 business entity-resolution pipeline. It uses the official TSV data and training ground truth only. The project root is the working directory for the commands below.

## Requirements

Python 3.14.3, NumPy 2.5.3, and pandas 3.0.6. Install with:

```powershell
python -m pip install -r code/business_entity_resolution/requirements.txt
```

## Run end to end

The official files must be placed in `dataset/train/` and `dataset/test/`. First create cleaned copies (only if the cleaned files do not already exist):

```powershell
python code/business_entity_resolution/src/clean_data.py --input dataset --output output_clean
```

Then generate training candidates, validate their recall, train the grouped model, and generate both test TSVs:

```powershell
python code/business_entity_resolution/src/generate_candidates.py --data-dir output_clean/train --output-dir output_candidates --chunk-size 50000
python code/business_entity_resolution/src/validate_candidate_recall.py --data-dir output_clean/train --candidate-dir output_candidates/train --report output_candidates/candidate_recall_report.txt
python code/business_entity_resolution/src/train_model.py --data-dir output_clean/train --candidate-dir output_candidates/train --artifact-dir models/v1
python code/business_entity_resolution/src/predict.py --data-dir output_clean/test --model models/v1/logistic_models.npz --candidate-dir output_candidates_test/test --output-dir output --chunk-size 50000
```

The prediction command writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`. The candidate file is the exact set scored by the model. The optional `--fast-candidates` flag uses exact name/address blocks and has lower recall; default mode uses the broader blocking strategy.

Validate with the official challenge checker from the repository root:

```powershell
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
