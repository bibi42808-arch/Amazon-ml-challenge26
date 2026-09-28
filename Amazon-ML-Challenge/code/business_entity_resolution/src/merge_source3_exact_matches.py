"""Score exact-blocked Source 3 pairs and merge them into a matching TSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from .features import FEATURE_NAMES, feature_vector
except ImportError:
    from features import FEATURE_NAMES, feature_vector


MATCH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_COLUMNS = ["source1_entity_id", "candidate_entity_id", "blocking_method"]
MAX_SUBMISSION_BYTES = 512 * 1024 * 1024
BATCH_SIZE = 20_000


def read_records(path: Path, wanted: set[str]) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file, delimiter="\t")
        for row in reader:
            entity_id = row["entity_id_normalized"]
            if entity_id in wanted:
                records[entity_id] = row
                if len(records) == len(wanted):
                    break
    missing = wanted - records.keys()
    if missing:
        raise ValueError(f"Could not find {len(missing):,} records in {path}.")
    return records


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_root / "output_clean" / "test")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=project_root / "output_candidates_test_fast" / "test" / "candidate_pairs_source3.tsv",
    )
    parser.add_argument("--model", type=Path, default=project_root / "models" / "v1" / "logistic_models.npz")
    parser.add_argument(
        "--matching",
        type=Path,
        default=project_root / "submissions" / "urgent" / "output" / "matching_results.tsv",
    )
    args = parser.parse_args()

    model = np.load(args.model, allow_pickle=False)
    if model["feature_names"].tolist() != list(FEATURE_NAMES):
        raise ValueError("Saved model feature order does not match the current feature implementation.")
    weights = model["weights_source3"]
    bias = float(model["bias_source3"])
    threshold = float(model["threshold_source3"])

    candidate_pairs: list[tuple[str, str, str]] = []
    source_ids: set[str] = set()
    target_ids: set[str] = set()
    candidate_counts: dict[str, int] = defaultdict(int)
    with args.candidates.open("r", encoding="utf-8", newline="") as candidate_file:
        reader = csv.DictReader(candidate_file, delimiter="\t")
        if reader.fieldnames != CANDIDATE_COLUMNS:
            raise ValueError(f"Unexpected Source 3 candidate header: {reader.fieldnames}")
        for row in reader:
            source_id = row["source1_entity_id"]
            target_id = row["candidate_entity_id"]
            candidate_pairs.append((source_id, target_id, row["blocking_method"]))
            source_ids.add(source_id)
            target_ids.add(target_id)
            candidate_counts[source_id] += 1

    source_records = read_records(args.data_dir / "test_source1_cleaned.tsv", source_ids)
    target_records = read_records(args.data_dir / "test_source3_cleaned.tsv", target_ids)
    matched: dict[str, list[str]] = defaultdict(list)
    scored_pairs = 0
    for start in range(0, len(candidate_pairs), BATCH_SIZE):
        batch = candidate_pairs[start : start + BATCH_SIZE]
        matrix = np.asarray(
            [
                feature_vector(
                    source_records[source_id],
                    target_records[target_id],
                    method,
                    candidate_counts[source_id],
                )
                for source_id, target_id, method in batch
            ],
            dtype=np.float64,
        )
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(matrix @ weights + bias, -40.0, 40.0)))
        for (source_id, target_id, _), probability in zip(batch, probabilities):
            if float(probability) >= threshold:
                matched[source_id].append(target_id)
        scored_pairs += len(batch)

    temporary_output = args.matching.with_name(args.matching.name + ".tmp")
    rows = 0
    with args.matching.open("r", encoding="utf-8", newline="") as current_file, temporary_output.open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        reader = csv.DictReader(current_file, delimiter="\t")
        if reader.fieldnames != MATCH_COLUMNS:
            raise ValueError(f"Unexpected matching header: {reader.fieldnames}")
        writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
        writer.writerow(MATCH_COLUMNS)
        for row in reader:
            source_id = row["source1_entity_id"]
            target_set = set(filter(None, row["matched_entity_ids"].split(",")))
            target_set.update(matched.get(source_id, ()))
            writer.writerow([source_id, ",".join(sorted(target_set))])
            rows += 1

    if rows != 1_732_544 or temporary_output.stat().st_size >= MAX_SUBMISSION_BYTES:
        temporary_output.unlink(missing_ok=True)
        raise RuntimeError(f"Unexpected output row count ({rows:,}) or file size.")
    temporary_output.replace(args.matching)
    print(
        f"Rows: {rows:,}; Source 3 candidate pairs scored: {scored_pairs:,}; "
        f"Source 3 matches added: {sum(map(len, matched.values())):,}; "
        f"threshold: {threshold:.3f}; total size: {args.matching.stat().st_size / (1024 * 1024):.2f} MiB; "
        f"output: {args.matching}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())