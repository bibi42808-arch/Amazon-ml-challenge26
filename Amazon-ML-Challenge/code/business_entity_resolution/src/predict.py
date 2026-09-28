"""Generate test candidates, score them, and write portal-format TSV outputs."""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .features import feature_vector
    from .generate_candidates import generate_for_target
    from .train_model import sigmoid
except ImportError:
    from features import feature_vector
    from generate_candidates import generate_for_target
    from train_model import sigmoid

TEST_SOURCE1 = "test_source1_cleaned.tsv"
TEST_TARGETS = {"source2": "test_source2_cleaned.tsv", "source3": "test_source3_cleaned.tsv"}
RECORD_COLUMNS = ["entity_id_normalized", "business_name_normalized", "business_address_normalized", "country_normalized"]
PAIR_COLUMNS = ["source1_entity_id", "candidate_entity_id", "blocking_method"]
MAX_SUBMISSION_BYTES = 512 * 1024 * 1024
BATCH_SIZE = 2048


def chunks(path: Path, columns: list[str], chunk_size: int):
    return pd.read_csv(path, sep="\t", usecols=columns, dtype="string", keep_default_na=False, na_filter=False, chunksize=chunk_size)


def create_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-262144")
    connection.executescript(
        """
        CREATE TABLE source1_staging (
            entity_id TEXT NOT NULL,
            name TEXT NOT NULL,
            address TEXT NOT NULL,
            country TEXT NOT NULL
        );
        CREATE TABLE target_staging (
            source TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            name TEXT NOT NULL,
            address TEXT NOT NULL,
            country TEXT NOT NULL
        );
        CREATE TABLE candidate_staging (
            source1_entity_id TEXT NOT NULL,
            candidate_entity_id TEXT NOT NULL,
            blocking_method TEXT NOT NULL,
            source TEXT NOT NULL
        );
        CREATE TABLE selected_matches (
            source1_entity_id TEXT NOT NULL,
            candidate_entity_id TEXT NOT NULL,
            source TEXT NOT NULL
        );
        """
    )
    return connection


def load_test_records(connection: sqlite3.Connection, data_dir: Path, chunk_size: int) -> tuple[int, dict[str, int]]:
    source1_count = 0
    source1_insert = "INSERT INTO source1_staging VALUES (?, ?, ?, ?)"
    for frame in chunks(data_dir / TEST_SOURCE1, RECORD_COLUMNS, chunk_size):
        values = [tuple(str(value) for value in row) for row in frame.itertuples(index=False, name=None)]
        connection.executemany(source1_insert, values)
        source1_count += len(values)
    connection.commit()

    target_counts: dict[str, int] = {}
    target_insert = "INSERT INTO target_staging VALUES (?, ?, ?, ?, ?)"
    for source, filename in TEST_TARGETS.items():
        count = 0
        for frame in chunks(data_dir / filename, RECORD_COLUMNS, chunk_size):
            values = [
                (source, *(str(value) for value in row))
                for row in frame.itertuples(index=False, name=None)
            ]
            connection.executemany(target_insert, values)
            count += len(values)
        connection.commit()
        target_counts[source] = count
        print(f"Loaded {count:,} {source} test records into disk-backed lookup.", flush=True)

    connection.execute("CREATE UNIQUE INDEX test_source1_lookup ON source1_staging(entity_id)")
    connection.execute("CREATE UNIQUE INDEX test_target_lookup ON target_staging(source, entity_id)")
    connection.commit()
    return source1_count, target_counts


def load_candidates(connection: sqlite3.Connection, candidate_dir: Path, chunk_size: int) -> dict[str, int]:
    pair_counts: dict[str, int] = {}
    insert_sql = "INSERT INTO candidate_staging VALUES (?, ?, ?, ?)"
    for source in TEST_TARGETS:
        path = candidate_dir / f"candidate_pairs_{source}.tsv"
        pair_count = 0
        for frame in chunks(path, PAIR_COLUMNS, chunk_size):
            values = [
                (str(source1_id), str(candidate_id), str(method), source)
                for source1_id, candidate_id, method in frame.itertuples(index=False, name=None)
            ]
            connection.executemany(insert_sql, values)
            pair_count += len(values)
        connection.commit()
        pair_counts[source] = pair_count

    connection.execute("CREATE UNIQUE INDEX test_candidate_lookup ON candidate_staging(source, source1_entity_id, candidate_entity_id)")
    connection.execute(
        "CREATE TABLE candidate_counts AS SELECT source, source1_entity_id, COUNT(*) AS candidate_count "
        "FROM candidate_staging GROUP BY source, source1_entity_id"
    )
    connection.execute("CREATE UNIQUE INDEX test_candidate_counts_lookup ON candidate_counts(source, source1_entity_id)")
    connection.execute("CREATE UNIQUE INDEX selected_matches_lookup ON selected_matches(source, source1_entity_id, candidate_entity_id)")
    connection.commit()
    return pair_counts


def score_source(
    connection: sqlite3.Connection,
    source: str,
    weights: np.ndarray,
    bias: float,
    threshold: float,
) -> tuple[int, int]:
    query = """
        SELECT c.source1_entity_id, c.candidate_entity_id, c.blocking_method,
               s.name, s.address, s.country,
               t.name, t.address, t.country,
               COALESCE(cc.candidate_count, 1)
        FROM candidate_staging c
        JOIN source1_staging s ON s.entity_id=c.source1_entity_id
        JOIN target_staging t ON t.source=c.source AND t.entity_id=c.candidate_entity_id
        LEFT JOIN candidate_counts cc
          ON cc.source=c.source AND cc.source1_entity_id=c.source1_entity_id
        WHERE c.source=?
        ORDER BY c.source1_entity_id, c.candidate_entity_id
    """
    cursor = connection.execute(query, (source,))
    insert_sql = "INSERT OR IGNORE INTO selected_matches VALUES (?, ?, ?)"
    scored_count = 0
    match_count = 0
    while rows := cursor.fetchmany(BATCH_SIZE):
        matrix = np.empty((len(rows), 34), dtype=np.float64)
        for index, row in enumerate(rows):
            source_record = {
                "business_name_normalized": row[3],
                "business_address_normalized": row[4],
                "country_normalized": row[5],
            }
            candidate_record = {
                "business_name_normalized": row[6],
                "business_address_normalized": row[7],
                "country_normalized": row[8],
            }
            matrix[index] = feature_vector(source_record, candidate_record, str(row[2]), int(row[9]))
        probabilities = sigmoid(matrix @ weights + bias)
        selected = [
            (str(row[0]), str(row[1]), source)
            for row, probability in zip(rows, probabilities)
            if float(probability) >= threshold
        ]
        connection.executemany(insert_sql, selected)
        connection.commit()
        scored_count += len(rows)
        match_count += len(selected)
    return scored_count, match_count


def write_grouped_ids(
    connection: sqlite3.Connection,
    output_path: Path,
    mode: str,
) -> tuple[int, int]:
    if mode == "matching":
        header = ["source1_entity_id", "matched_entity_ids"]
        table = "selected_matches"
        id_column = "candidate_entity_id"
        join = "LEFT JOIN selected_matches m ON m.source1_entity_id=s.entity_id"
    else:
        header = ["source1_entity_id", "candidate_entity_ids"]
        table = "candidate_staging"
        id_column = "candidate_entity_id"
        join = "LEFT JOIN candidate_staging m ON m.source1_entity_id=s.entity_id"
    query = (
        f"SELECT s.entity_id, m.{id_column} FROM source1_staging s {join} "
        f"ORDER BY s.entity_id, m.{id_column}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    id_count = 0
    with output_path.open("x", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        cursor = connection.execute(query)
        current_id: str | None = None
        id_values: list[str] = []
        for entity_id, matched_id in cursor:
            entity_id = str(entity_id)
            if current_id is not None and entity_id != current_id:
                writer.writerow([current_id, ",".join(id_values)])
                row_count += 1
                id_count += len(id_values)
                id_values.clear()
            current_id = entity_id
            if matched_id is not None:
                id_values.append(str(matched_id))
        if current_id is not None:
            writer.writerow([current_id, ",".join(id_values)])
            row_count += 1
            id_count += len(id_values)
    return row_count, id_count


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate full test matches in official portal TSV format.")
    parser.add_argument("--data-dir", type=Path, default=project_root / "output_clean" / "test")
    parser.add_argument("--model", type=Path, default=project_root / "models" / "v1" / "logistic_models.npz")
    parser.add_argument("--candidate-dir", type=Path, default=project_root / "output_candidates_test_v1" / "test")
    parser.add_argument("--output-dir", type=Path, default=project_root / "submissions" / "v1" / "output")
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--skip-candidate-generation", action="store_true")
    parser.add_argument(
        "--fast-candidates",
        action="store_true",
        help="Use exact name/address blocking only to produce a quicker, lower-recall submission.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunk_size <= 0:
        print("--chunk-size must be positive", file=sys.stderr)
        return 2
    if not args.model.is_file():
        print(f"Model file not found: {args.model}", file=sys.stderr)
        return 2
    if args.output_dir.exists():
        print(f"Refusing to overwrite output directory: {args.output_dir}", file=sys.stderr)
        return 2

    for filename in (TEST_SOURCE1, *TEST_TARGETS.values()):
        if not (args.data_dir / filename).is_file():
            print(f"Missing cleaned test input: {args.data_dir / filename}", file=sys.stderr)
            return 2

    candidate_paths = [args.candidate_dir / f"candidate_pairs_{source}.tsv" for source in TEST_TARGETS]
    if not args.skip_candidate_generation:
        existing = [path for path in candidate_paths if path.exists()]
        if existing:
            print("Refusing to overwrite existing test candidate files:", file=sys.stderr)
            for path in existing:
                print(f"  {path}", file=sys.stderr)
            return 2
        args.candidate_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="test_candidate_index_") as temp_name:
            temp_dir = Path(temp_name)
            for source, filename in TEST_TARGETS.items():
                generate_for_target(
                    args.data_dir / TEST_SOURCE1,
                    args.data_dir / filename,
                    args.candidate_dir / f"candidate_pairs_{source}.tsv",
                    args.chunk_size,
                    temp_dir,
                    source,
                    fast_exact_only=args.fast_candidates,
                )
    elif not all(path.is_file() for path in candidate_paths):
        print("--skip-candidate-generation requires both existing candidate files.", file=sys.stderr)
        return 2

    model = np.load(args.model, allow_pickle=False)
    model_params = {
        "source2": (model["weights_source2"], float(model["bias_source2"]), float(model["threshold_source2"])),
        "source3": (model["weights_source3"], float(model["bias_source3"]), float(model["threshold_source3"])),
    }
    args.output_dir.mkdir(parents=True)
    matching_path = args.output_dir / "matching_results.tsv"
    candidate_submission_path = args.output_dir / "candidate_pairs.tsv"
    with tempfile.TemporaryDirectory(prefix="test_prediction_") as temp_name:
        connection = create_database(Path(temp_name) / "prediction.sqlite")
        try:
            source1_count, _ = load_test_records(connection, args.data_dir, args.chunk_size)
            pair_counts = load_candidates(connection, args.candidate_dir, args.chunk_size)
            scored_counts = {}
            predicted_counts = {}
            for source, (weights, bias, threshold) in model_params.items():
                scored_counts[source], predicted_counts[source] = score_source(
                    connection, source, weights, bias, threshold
                )
                print(
                    f"{source}: scored {scored_counts[source]:,} candidates; "
                    f"selected {predicted_counts[source]:,} matches at threshold {threshold:.3f}.",
                    flush=True,
                )
            matching_rows, matching_ids = write_grouped_ids(connection, matching_path, "matching")
            candidate_rows, candidate_ids = write_grouped_ids(connection, candidate_submission_path, "candidate")
        finally:
            connection.close()

    size_bytes = matching_path.stat().st_size
    if matching_rows != source1_count or candidate_rows != source1_count:
        raise RuntimeError(
            f"Output row count mismatch: matching={matching_rows:,}, candidates={candidate_rows:,}, Source1={source1_count:,}."
        )
    if size_bytes > MAX_SUBMISSION_BYTES:
        raise RuntimeError(
            f"matching_results.tsv is {size_bytes:,} bytes, above the 512 MiB limit."
        )
    print(f"matching_results.tsv rows: {matching_rows:,}; matched IDs: {matching_ids:,}")
    print(f"candidate_pairs.tsv rows: {candidate_rows:,}; candidate IDs: {candidate_ids:,}")
    print(f"Candidate pair inputs: {pair_counts}")
    print(f"matching_results.tsv size: {size_bytes / (1024 * 1024):.2f} MiB")
    print(f"Ready for official-format validation: {matching_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
