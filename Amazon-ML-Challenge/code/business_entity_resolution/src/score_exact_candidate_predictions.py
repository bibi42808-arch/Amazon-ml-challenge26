"""Score high-confidence exact-name/address candidates without loading all targets."""

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


SOURCE_COLUMNS = [
    "entity_id_normalized",
    "business_name_normalized",
    "business_address_normalized",
    "country_normalized",
]
OUTPUT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
MAX_SUBMISSION_BYTES = 512 * 1024 * 1024


def read_candidates(path: Path) -> tuple[dict[str, list[tuple[str, str]]], dict[str, int]]:
    selected: dict[str, list[tuple[str, str]]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file, delimiter="\t")
        expected = {"source1_entity_id", "candidate_entity_id", "blocking_method"}
        if set(reader.fieldnames or ()) != expected:
            raise ValueError(f"Unexpected candidate columns: {reader.fieldnames}")
        for row in reader:
            source_id = row["source1_entity_id"]
            counts[source_id] += 1
            methods = set(row["blocking_method"].split(","))
            if {"exact_name", "exact_address"} <= methods:
                selected[source_id].append((row["candidate_entity_id"], row["blocking_method"]))
    return selected, counts


def read_records(path: Path, wanted_ids: set[str]) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file, delimiter="\t")
        missing = set(SOURCE_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing normalized source columns in {path}: {sorted(missing)}")
        for row in reader:
            entity_id = row["entity_id_normalized"]
            if entity_id in wanted_ids:
                records[entity_id] = row
                if len(records) == len(wanted_ids):
                    break
    missing_ids = wanted_ids - records.keys()
    if missing_ids:
        raise ValueError(f"Could not find {len(missing_ids):,} requested IDs in {path}.")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("output_clean/test"))
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("output_candidates_test_fast/test/candidate_pairs_source2.tsv"),
    )
    parser.add_argument("--model", type=Path, default=Path("models/v1/logistic_models.npz"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submissions/urgent/output/matching_results.tsv"),
    )
    args = parser.parse_args()

    model = np.load(args.model, allow_pickle=False)
    if model["feature_names"].tolist() != list(FEATURE_NAMES):
        raise ValueError("Saved model feature order does not match the current feature implementation.")
    weights = model["weights_source2"]
    bias = float(model["bias_source2"])
    threshold = float(model["threshold_source2"])

    candidates, candidate_counts = read_candidates(args.candidates)
    source_ids = set(candidates)
    target_ids = {target_id for pairs in candidates.values() for target_id, _ in pairs}
    source_records = read_records(args.data_dir / "test_source1_cleaned.tsv", source_ids)
    target_records = read_records(args.data_dir / "test_source2_cleaned.tsv", target_ids)

    matches: dict[str, list[str]] = defaultdict(list)
    matrix_rows: list[list[float]] = []
    pair_rows: list[tuple[str, str]] = []
    for source_id, pairs in candidates.items():
        for target_id, method in pairs:
            matrix_rows.append(
                feature_vector(source_records[source_id], target_records[target_id], method, candidate_counts[source_id])
            )
            pair_rows.append((source_id, target_id))

    if matrix_rows:
        matrix = np.asarray(matrix_rows, dtype=np.float64)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(matrix @ weights + bias, -40.0, 40.0)))
        for (source_id, target_id), probability in zip(pair_rows, probabilities):
            if float(probability) >= threshold:
                matches[source_id].append(target_id)

    source1_path = args.data_dir / "test_source1_cleaned.tsv"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(args.output.name + ".tmp")
    row_count = 0
    with source1_path.open("r", encoding="utf-8", newline="") as source_file, temporary_output.open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        source_reader = csv.DictReader(source_file, delimiter="\t")
        writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
        writer.writerow(OUTPUT_COLUMNS)
        for row in source_reader:
            source_id = row["entity_id_normalized"]
            writer.writerow([source_id, ",".join(sorted(set(matches.get(source_id, ()))) )])
            row_count += 1

    if row_count == 0 or temporary_output.stat().st_size >= MAX_SUBMISSION_BYTES:
        temporary_output.unlink(missing_ok=True)
        raise RuntimeError("Generated output is empty or exceeds the 512 MiB upload limit.")
    temporary_output.replace(args.output)
    print(
        f"Rows: {row_count:,}; scored pairs: {len(pair_rows):,}; "
        f"matched pairs: {sum(map(len, matches.values())):,}; "
        f"threshold: {threshold:.3f}; size: {args.output.stat().st_size / (1024 * 1024):.2f} MiB; "
        f"output: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())