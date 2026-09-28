"""Generate local-only candidates sharing an exact normalized name and address."""

from __future__ import annotations

import argparse
import csv
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

try:
    from .generate_candidates import SOURCE_COLUMNS, normalize_block_text, open_tsv_chunks, row_values, strip_legal_suffixes
except ImportError:
    from generate_candidates import SOURCE_COLUMNS, normalize_block_text, open_tsv_chunks, row_values, strip_legal_suffixes


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_root / "output_clean" / "test")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--source", choices=("source2", "source3"), default="source3")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()

    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    source1_path = args.data_dir / f"{args.split}_source1_cleaned.tsv"
    target_path = args.data_dir / f"{args.split}_{args.source}_cleaned.tsv"
    default_output_root = "output_candidates_test_fast" if args.split == "test" else "output_candidates_exact_train"
    output_path = args.output or project_root / default_output_root / args.split / f"candidate_pairs_{args.source}.tsv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(output_path.name + ".tmp")

    for path in (source1_path, target_path):
        if not path.is_file():
            parser.error(f"Required cleaned input does not exist: {path}")

    with tempfile.TemporaryDirectory(prefix="exact_address_candidates_") as temp_name:
        connection = sqlite3.connect(Path(temp_name) / "targets.sqlite")
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA cache_size=-131072")
            connection.execute(
                "CREATE TABLE targets (country TEXT, name_key TEXT, address TEXT, entity_id TEXT)"
            )
            target_count = 0
            insert_sql = "INSERT INTO targets VALUES (?, ?, ?, ?)"
            for chunk in open_tsv_chunks(target_path, args.chunk_size):
                postings = []
                for entity_id, name, address, country in row_values(chunk):
                    country_key = normalize_block_text(country)
                    name_key = normalize_block_text(name)
                    address_key = normalize_block_text(address)
                    if not entity_id or not country_key or not name_key or len(address_key) < 8:
                        continue
                    postings.extend(
                        (country_key, variant, address_key, entity_id)
                        for variant in strip_legal_suffixes(name_key)
                    )
                connection.executemany(insert_sql, postings)
                target_count += len(chunk)
                if target_count and target_count % 500_000 < len(chunk):
                    print(f"Indexed {target_count:,} {args.source} records...", flush=True)
            connection.commit()
            connection.execute(
                "CREATE INDEX exact_lookup ON targets(country, name_key, address, entity_id)"
            )
            connection.commit()

            output_pairs = 0
            covered_source1 = 0
            method = "country_exact_name,exact_name,country_exact_address,exact_address"
            query = "SELECT entity_id FROM targets WHERE country=? AND name_key=? AND address=?"
            with source1_path.open("r", encoding="utf-8", newline="") as source_file, temporary_output.open(
                "w", encoding="utf-8", newline=""
            ) as output_file:
                source_reader = pd.read_csv(
                    source_file,
                    sep="\t",
                    usecols=SOURCE_COLUMNS,
                    dtype="string",
                    keep_default_na=False,
                    na_filter=False,
                    chunksize=args.chunk_size,
                )
                writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
                writer.writerow(["source1_entity_id", "candidate_entity_id", "blocking_method"])
                for chunk in source_reader:
                    for entity_id, name, address, country in row_values(chunk):
                        country_key = normalize_block_text(country)
                        address_key = normalize_block_text(address)
                        if not entity_id or not country_key or len(address_key) < 8:
                            continue
                        found: set[str] = set()
                        for variant in strip_legal_suffixes(normalize_block_text(name)):
                            found.update(
                                str(row[0])
                                for row in connection.execute(query, (country_key, variant, address_key))
                            )
                        if found:
                            covered_source1 += 1
                            rows = sorted((entity_id, target_id, method) for target_id in found)
                            writer.writerows(rows)
                            output_pairs += len(rows)
            if output_pairs == 0:
                temporary_output.unlink(missing_ok=True)
                raise RuntimeError("No exact name-and-address candidate pairs found; existing output was preserved.")
            temporary_output.replace(output_path)
            print(
                f"Indexed target records: {target_count:,}; source1 IDs covered: {covered_source1:,}; "
                f"candidate pairs: {output_pairs:,}; output: {output_path}",
                flush=True,
            )
        finally:
            connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())