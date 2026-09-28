"""Inspect cleaned training tables without modifying any dataset files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXPECTED_ROWS = {
    "train_source1_cleaned.tsv": 2_206_821,
    "train_source2_cleaned.tsv": 5_034_616,
    "train_source3_cleaned.tsv": 5_285_603,
    "train_ground_truth_cleaned.tsv": 2_206_821,
}
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
DISPLAY_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def read_header_and_samples(path: Path) -> list[str]:
    """Print the exact TSV header and five example records."""
    header = pd.read_csv(path, sep="\t", nrows=0, dtype="string").columns.tolist()
    samples = pd.read_csv(
        path,
        sep="\t",
        nrows=5,
        dtype="string",
        keep_default_na=False,
        na_filter=False,
    )

    print(f"\n{'=' * 88}\nFILE: {path}")
    print(f"Exact columns ({len(header)}): {header}")
    print("First 5 data rows:")
    print(samples.to_string(index=False, max_colwidth=120))
    return header


def count_ground_truth(
    path: Path,
    chunk_size: int,
    example_limit: int,
) -> tuple[int, dict[str, int], list[dict[str, str]]]:
    """Count match-list categories and retain only a few example rows."""
    totals = {
        "blank_matched_entity_ids": 0,
        "nonblank_matched_entity_ids": 0,
        "rows_with_one_matched_id": 0,
        "rows_with_multiple_matched_ids": 0,
        "nonblank_rows_with_no_parseable_ids": 0,
    }
    row_count = 0
    examples: list[dict[str, str]] = []

    reader = pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        keep_default_na=False,
        na_filter=False,
        chunksize=chunk_size,
    )
    for chunk in reader:
        row_count += len(chunk)
        matched_values = chunk["matched_entity_ids"].astype("string")
        blank_mask = matched_values.str.strip().eq("")
        totals["blank_matched_entity_ids"] += int(blank_mask.sum())
        totals["nonblank_matched_entity_ids"] += int((~blank_mask).sum())

        for source_id, matched_value in zip(
            chunk["source1_entity_id"].astype(str), matched_values.astype(str)
        ):
            matched_ids = [value.strip() for value in matched_value.split(",") if value.strip()]
            if not matched_ids:
                if matched_value.strip():
                    totals["nonblank_rows_with_no_parseable_ids"] += 1
                continue

            if len(matched_ids) == 1:
                totals["rows_with_one_matched_id"] += 1
            else:
                totals["rows_with_multiple_matched_ids"] += 1
            if len(examples) < example_limit:
                examples.append(
                    {
                        "source1_entity_id": source_id,
                        "matched_entity_ids": matched_value,
                    }
                )

    return row_count, totals, examples


def scan_source_for_rows(
    path: Path,
    chunk_size: int,
    target_ids: set[str],
) -> tuple[int, dict[str, list[dict[str, str]]]]:
    """Count all rows and retain only records whose IDs are requested."""
    row_count = 0
    found: dict[str, list[dict[str, str]]] = {}
    reader = pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        keep_default_na=False,
        na_filter=False,
        chunksize=chunk_size,
    )

    for chunk in reader:
        row_count += len(chunk)
        if target_ids:
            matches = chunk.loc[chunk["entity_id"].isin(target_ids), DISPLAY_COLUMNS]
            for record in matches.to_dict(orient="records"):
                entity_id = str(record["entity_id"])
                found.setdefault(entity_id, []).append(
                    {column: str(record[column]) for column in DISPLAY_COLUMNS}
                )

    return row_count, found


def print_source_records(
    label: str,
    matched_ids: list[str],
    records_by_id: dict[str, list[dict[str, str]]],
) -> None:
    print(f"  {label} records:")
    if not matched_ids:
        print("    (no IDs of this source were listed)")
        return

    for entity_id in matched_ids:
        records = records_by_id.get(entity_id, [])
        if not records:
            print(f"    {entity_id}: no matching record found")
            continue
        for record in records:
            print(
                "    ID: {entity_id}\n"
                "      Business name: {business_name}\n"
                "      Business address: {business_address}\n"
                "      Country: {country}".format(**record)
            )


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Read-only, memory-safe inspection of the cleaned training TSV files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "output_clean" / "train",
        help="Folder containing the four cleaned training TSVs.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Rows processed at a time (default: 100000).",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=3,
        help="Number of nonblank ground-truth examples to link (default: 3).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunk_size <= 0 or args.examples < 0:
        raise ValueError("--chunk-size must be positive and --examples cannot be negative.")

    data_dir = args.data_dir.resolve()
    paths = {
        filename: data_dir / filename
        for filename in EXPECTED_ROWS
    }
    missing_files = [str(path) for path in paths.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError("Required training files were not found:\n" + "\n".join(missing_files))

    print("TRAINING FILE HEADERS AND SAMPLE ROWS")
    headers = {filename: read_header_and_samples(path) for filename, path in paths.items()}

    required = {
        "train_source1_cleaned.tsv": SOURCE_COLUMNS,
        "train_source2_cleaned.tsv": SOURCE_COLUMNS,
        "train_source3_cleaned.tsv": SOURCE_COLUMNS,
        "train_ground_truth_cleaned.tsv": GROUND_TRUTH_COLUMNS,
    }
    for filename, required_columns in required.items():
        missing_columns = [column for column in required_columns if column not in headers[filename]]
        if missing_columns:
            raise ValueError(f"{filename} is missing required columns: {missing_columns}")

    print(f"\n{'=' * 88}\nGROUND-TRUTH MATCH-LIST ANALYSIS")
    ground_truth_path = paths["train_ground_truth_cleaned.tsv"]
    ground_truth_rows, match_totals, examples = count_ground_truth(
        ground_truth_path,
        args.chunk_size,
        args.examples,
    )
    for label, count in match_totals.items():
        print(f"{label.replace('_', ' ').capitalize()}: {count:,}")

    example_match_ids = {
        example["source1_entity_id"] for example in examples
    }
    source_match_ids = {
        matched_id.strip()
        for example in examples
        for matched_id in example["matched_entity_ids"].split(",")
        if matched_id.strip()
    }

    row_counts = {"train_ground_truth_cleaned.tsv": ground_truth_rows}
    found_records: dict[str, dict[str, list[dict[str, str]]]] = {}
    for filename in (
        "train_source1_cleaned.tsv",
        "train_source2_cleaned.tsv",
        "train_source3_cleaned.tsv",
    ):
        print(f"\nCounting rows and checking selected IDs in {filename}...")
        target_ids = example_match_ids if filename == "train_source1_cleaned.tsv" else source_match_ids
        row_counts[filename], found_records[filename] = scan_source_for_rows(
            paths[filename], args.chunk_size, target_ids
        )

    print(f"\n{'=' * 88}\nROW-COUNT VERIFICATION")
    for filename, expected in EXPECTED_ROWS.items():
        actual = row_counts[filename]
        status = "PASS" if actual == expected else "MISMATCH"
        print(f"{filename}: {actual:,} rows (expected {expected:,}) [{status}]")

    print(f"\n{'=' * 88}\nLINKED GROUND-TRUTH EXAMPLES")
    if not examples:
        print("No nonblank ground-truth match lists were selected.")
    for example_number, example in enumerate(examples, start=1):
        source1_id = example["source1_entity_id"]
        matched_ids = [
            value.strip()
            for value in example["matched_entity_ids"].split(",")
            if value.strip()
        ]
        source2_ids = [entity_id for entity_id in matched_ids if entity_id.startswith("S2-")]
        source3_ids = [entity_id for entity_id in matched_ids if entity_id.startswith("S3-")]
        other_ids = [
            entity_id
            for entity_id in matched_ids
            if entity_id not in source2_ids and entity_id not in source3_ids
        ]
        print(f"\nExample {example_number}")
        print(f"  Source 1 entity ID: {source1_id}")
        source1_records = found_records["train_source1_cleaned.tsv"].get(source1_id, [])
        if not source1_records:
            print("  Source 1 record: no matching record found")
        else:
            for record in source1_records:
                print(f"  Source 1 business name: {record['business_name']}")
                print(f"  Source 1 business address: {record['business_address']}")
                print(f"  Source 1 country: {record['country']}")
        print(f"  Matched entity IDs: {example['matched_entity_ids']}")
        print_source_records(
            "Source 2",
            source2_ids,
            found_records["train_source2_cleaned.tsv"],
        )
        print_source_records(
            "Source 3",
            source3_ids,
            found_records["train_source3_cleaned.tsv"],
        )
        if other_ids:
            print(f"  IDs with an unrecognized source prefix: {other_ids}")

    mismatches = [
        filename
        for filename, expected in EXPECTED_ROWS.items()
        if row_counts[filename] != expected
    ]
    print(f"\n{'=' * 88}\nSUMMARY")
    print(
        "Each ground-truth row identifies a Source 1 record through source1_entity_id. "
        "Its matched_entity_ids field lists related entity IDs to look up in Source 2 "
        "and Source 3. This script checked a small number of those links and counted "
        "all rows in chunks; it did not modify any files."
    )
    if mismatches:
        print(f"Warning: row counts differ from the stated expectations: {mismatches}")
        return 1
    print("All four training row counts match the expected values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())