"""Measure candidate recall against ground truth using disk-backed SQLite."""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


SOURCE1_COLUMNS = ["entity_id", "entity_id_normalized", "business_name_normalized", "business_address_normalized", "country_normalized"]
TARGET_COLUMNS = ["entity_id", "business_name_normalized", "business_address_normalized", "country_normalized"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_COLUMNS = ["source1_entity_id", "candidate_entity_id", "blocking_method"]
EXAMPLE_LIMIT = 10


def chunks(path: Path, columns: list[str], chunk_size: int) -> Any:
    return pd.read_csv(
        path,
        sep="\t",
        usecols=columns,
        dtype="string",
        keep_default_na=False,
        na_filter=False,
        chunksize=chunk_size,
    )


def create_schema(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.executescript(
        """
        CREATE TABLE source1 (
            entity_id TEXT PRIMARY KEY,
            entity_id_normalized TEXT,
            business_name_normalized TEXT,
            business_address_normalized TEXT,
            country_normalized TEXT
        ) WITHOUT ROWID;
        CREATE TABLE truth (
            source1_entity_id TEXT NOT NULL,
            target_entity_id TEXT NOT NULL,
            PRIMARY KEY (source1_entity_id, target_entity_id)
        ) WITHOUT ROWID;
        CREATE TABLE candidates (
            source1_entity_id TEXT NOT NULL,
            candidate_entity_id TEXT NOT NULL,
            blocking_method TEXT NOT NULL,
            PRIMARY KEY (source1_entity_id, candidate_entity_id)
        ) WITHOUT ROWID;
        CREATE TABLE candidate_source1 (
            source1_entity_id TEXT PRIMARY KEY
        ) WITHOUT ROWID;
        """
    )


def insert_source1(connection: sqlite3.Connection, path: Path, chunk_size: int) -> int:
    row_count = 0
    sql = "INSERT OR IGNORE INTO source1 VALUES (?, ?, ?, ?, ?)"
    for frame in chunks(path, SOURCE1_COLUMNS, chunk_size):
        row_count += len(frame)
        connection.executemany(sql, frame.itertuples(index=False, name=None))
        connection.commit()
    return row_count


def insert_truth(
    connection: sqlite3.Connection,
    path: Path,
    prefix: str,
    chunk_size: int,
) -> None:
    sql = "INSERT OR IGNORE INTO truth(source1_entity_id, target_entity_id) VALUES (?, ?)"
    for frame in chunks(path, GROUND_TRUTH_COLUMNS, chunk_size):
        rows = []
        for source1_id, matched_ids in frame.itertuples(index=False, name=None):
            source1_id = str(source1_id).strip()
            for target_id in str(matched_ids).split(","):
                target_id = target_id.strip()
                if target_id.startswith(prefix):
                    rows.append((source1_id, target_id))
        connection.executemany(sql, rows)
        connection.commit()


def insert_candidates(
    connection: sqlite3.Connection,
    path: Path,
    prefix: str,
    chunk_size: int,
) -> tuple[int, int]:
    raw_rows = 0
    duplicate_rows = 0
    sql = "INSERT OR IGNORE INTO candidates VALUES (?, ?, ?)"
    for frame in chunks(path, CANDIDATE_COLUMNS, chunk_size):
        raw_rows += len(frame)
        filtered = []
        for source1_id, candidate_id, method in frame.itertuples(index=False, name=None):
            source1_id = str(source1_id).strip()
            candidate_id = str(candidate_id).strip()
            if source1_id and candidate_id.startswith(prefix):
                filtered.append((source1_id, candidate_id, str(method)))
        before = connection.total_changes
        connection.executemany(sql, filtered)
        inserted = connection.total_changes - before
        duplicate_rows += len(filtered) - inserted
        connection.executemany(
            "INSERT OR IGNORE INTO candidate_source1 VALUES (?)",
            ((row[0],) for row in filtered),
        )
        connection.commit()
    return raw_rows, duplicate_rows


def source_metrics(connection: sqlite3.Connection) -> dict[str, Any]:
    true_count = int(connection.execute("SELECT COUNT(*) FROM truth").fetchone()[0])
    retained_count = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM truth AS t
            JOIN candidates AS c
              ON c.source1_entity_id = t.source1_entity_id
             AND c.candidate_entity_id = t.target_entity_id
            """
        ).fetchone()[0]
    )
    true_rows = int(
        connection.execute(
            "SELECT COUNT(DISTINCT source1_entity_id) FROM truth"
        ).fetchone()[0]
    )
    hit_rows = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT DISTINCT t.source1_entity_id
                FROM truth AS t
                JOIN candidates AS c
                  ON c.source1_entity_id = t.source1_entity_id
                 AND c.candidate_entity_id = t.target_entity_id
            )
            """
        ).fetchone()[0]
    )
    all_retained_rows = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT t.source1_entity_id
                FROM truth AS t
                LEFT JOIN candidates AS c
                  ON c.source1_entity_id = t.source1_entity_id
                 AND c.candidate_entity_id = t.target_entity_id
                GROUP BY t.source1_entity_id
                HAVING COUNT(*) = COUNT(c.candidate_entity_id)
            )
            """
        ).fetchone()[0]
    )
    unique_source1 = int(connection.execute("SELECT COUNT(*) FROM source1").fetchone()[0])
    candidate_source1 = int(
        connection.execute("SELECT COUNT(*) FROM candidate_source1").fetchone()[0]
    )
    zero_candidate_rows = unique_source1 - int(
        connection.execute(
            "SELECT COUNT(*) FROM source1 AS s JOIN candidate_source1 AS c ON c.source1_entity_id=s.entity_id"
        ).fetchone()[0]
    )
    distribution = [
        (int(candidate_count), int(source1_count))
        for candidate_count, source1_count in connection.execute(
            """
            SELECT candidate_count, COUNT(*)
            FROM (
                SELECT s.entity_id, COUNT(c.candidate_entity_id) AS candidate_count
                FROM source1 AS s
                LEFT JOIN candidates AS c ON c.source1_entity_id = s.entity_id
                GROUP BY s.entity_id
            )
            GROUP BY candidate_count
            ORDER BY candidate_count
            """
        )
    ]
    missing = connection.execute(
        """
        SELECT t.source1_entity_id, t.target_entity_id
        FROM truth AS t
        LEFT JOIN candidates AS c
          ON c.source1_entity_id = t.source1_entity_id
         AND c.candidate_entity_id = t.target_entity_id
        WHERE c.candidate_entity_id IS NULL
        ORDER BY t.source1_entity_id, t.target_entity_id
        LIMIT ?
        """,
        (EXAMPLE_LIMIT,),
    ).fetchall()
    methods = connection.execute(
        "SELECT blocking_method, COUNT(*) FROM candidates GROUP BY blocking_method ORDER BY blocking_method"
    ).fetchall()
    max_candidates = distribution[-1][0] if distribution else 0

    return {
        "true_match_ids": true_count,
        "retained_true_match_ids": retained_count,
        "id_recall": retained_count / true_count if true_count else None,
        "source1_rows_with_true_match": true_rows,
        "source1_rows_with_match_retained": hit_rows,
        "row_recall": hit_rows / true_rows if true_rows else None,
        "source1_rows_with_all_true_matches_retained": all_retained_rows,
        "all_true_matches_recall": all_retained_rows / true_rows if true_rows else None,
        "zero_candidate_source1_rows": zero_candidate_rows,
        "candidate_count_distribution": distribution,
        "unusually_large_candidate_rows_gt_100": sum(
            count for candidate_count, count in distribution if candidate_count > 100
        ),
        "max_candidates": max_candidates,
        "missing_examples": missing,
        "blocking_method_counts": methods,
        "candidate_pairs": int(connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]),
        "represented_source1_ids": candidate_source1,
        "unique_source1_ids": unique_source1,
    }


def fetch_examples(
    connection: sqlite3.Connection,
    source1_path: Path,
    target_path: Path,
    chunk_size: int,
) -> list[tuple[str, str, dict[str, str] | None, dict[str, str] | None]]:
    misses = connection.execute(
        """
        SELECT t.source1_entity_id, t.target_entity_id
        FROM truth AS t
        LEFT JOIN candidates AS c
          ON c.source1_entity_id = t.source1_entity_id
         AND c.candidate_entity_id = t.target_entity_id
        WHERE c.candidate_entity_id IS NULL
        ORDER BY t.source1_entity_id, t.target_entity_id
        LIMIT ?
        """,
        (EXAMPLE_LIMIT,),
    ).fetchall()
    source_ids = {row[0] for row in misses}
    target_ids = {row[1] for row in misses}
    found_source: dict[str, dict[str, str]] = {}
    found_target: dict[str, dict[str, str]] = {}

    for frame in chunks(source1_path, SOURCE1_COLUMNS, chunk_size):
        selected = frame.loc[frame["entity_id"].isin(source_ids)]
        for row in selected.to_dict(orient="records"):
            found_source[str(row["entity_id"])] = {
                "name": str(row["business_name_normalized"]),
                "address": str(row["business_address_normalized"]),
                "country": str(row["country_normalized"]),
            }
        if source_ids.issubset(found_source):
            break

    for frame in chunks(target_path, TARGET_COLUMNS, chunk_size):
        selected = frame.loc[frame["entity_id"].isin(target_ids)]
        for row in selected.to_dict(orient="records"):
            found_target[str(row["entity_id"])] = {
                "name": str(row["business_name_normalized"]),
                "address": str(row["business_address_normalized"]),
                "country": str(row["country_normalized"]),
            }
        if target_ids.issubset(found_target):
            break

    return [
        (source_id, target_id, found_source.get(source_id), found_target.get(target_id))
        for source_id, target_id in misses
    ]


def format_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f} ({value:.2%})"


def format_report(
    results: dict[str, dict[str, Any]],
    raw_candidate_rows: dict[str, int],
    duplicate_candidate_rows: dict[str, int],
    examples: dict[str, list[tuple[str, str, dict[str, str] | None, dict[str, str] | None]]],
) -> str:
    lines = [
        "CANDIDATE RECALL VALIDATION",
        "Ground truth and candidate files were read in chunks; pair sets were compared in temporary SQLite indexes.",
        "No input files were modified.",
    ]
    for label in ("source2", "source3"):
        result = results[label]
        lines.extend(
            [
                "",
                f"{'=' * 76}",
                f"SOURCE 1 -> {label.upper()}",
                f"Raw candidate rows read: {raw_candidate_rows[label]:,}",
                f"Duplicate candidate rows ignored: {duplicate_candidate_rows[label]:,}",
                f"Unique candidate pairs: {result['candidate_pairs']:,}",
                f"Unique Source 1 IDs represented: {result['represented_source1_ids']:,} / {result['unique_source1_ids']:,}",
                f"True-match ID count: {result['true_match_ids']:,}",
                f"True-match IDs retained: {result['retained_true_match_ids']:,}",
                f"ID-level recall: {format_ratio(result['id_recall'])}",
                f"Source 1 rows with at least one true match: {result['source1_rows_with_true_match']:,}",
                f"Rows with at least one true match retained: {result['source1_rows_with_match_retained']:,}",
                f"Source 1 row-level recall: {format_ratio(result['row_recall'])}",
                f"Rows with all known true matches retained: {result['source1_rows_with_all_true_matches_retained']:,}",
                f"All-known-matches row fraction: {format_ratio(result['all_true_matches_recall'])}",
                f"Zero-candidate Source 1 rows: {result['zero_candidate_source1_rows']:,}",
                f"Source 1 rows with >100 candidates: {result['unusually_large_candidate_rows_gt_100']:,}",
                f"Maximum candidates for one Source 1 row: {result['max_candidates']:,}",
                "Candidate-count distribution (candidate count: Source 1 rows):",
                ", ".join(f"{count}: {frequency:,}" for count, frequency in result["candidate_count_distribution"]),
                "Candidate method counts (joined method labels):",
            ]
        )
        lines.extend(
            f"  {method}: {count:,}" for method, count in result["blocking_method_counts"]
        )
        lines.append("Missing-match examples (normalized source values):")
        if not examples[label]:
            lines.append("  none")
        for source_id, target_id, source_record, target_record in examples[label]:
            lines.extend(
                [
                    f"  Source 1 {source_id} missing target {target_id}",
                    f"    Source 1: {source_record}",
                    f"    Target:   {target_record}",
                ]
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Measure S2/S3 candidate recall against the supplied training ground truth."
    )
    parser.add_argument("--data-dir", type=Path, default=project_root / "output_clean" / "train")
    parser.add_argument("--candidate-dir", type=Path, default=project_root / "output_candidates" / "train")
    parser.add_argument("--report", type=Path, default=project_root / "output_candidates" / "candidate_recall_report.txt")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    return parser.parse_args()


def validate_source(
    label: str,
    prefix: str,
    data_dir: Path,
    candidate_dir: Path,
    chunk_size: int,
    temp_dir: Path,
) -> tuple[dict[str, Any], int, int, list[tuple[str, str, dict[str, str] | None, dict[str, str] | None]]]:
    source1_path = data_dir / "train_source1_cleaned.tsv"
    target_path = data_dir / f"train_{label}_cleaned.tsv"
    truth_path = data_dir / "train_ground_truth_cleaned.tsv"
    candidate_path = candidate_dir / f"candidate_pairs_{label}.tsv"
    db_path = temp_dir / f"recall_{label}.sqlite"
    connection = sqlite3.connect(db_path)
    try:
        create_schema(connection)
        source_rows = insert_source1(connection, source1_path, chunk_size)
        if label == "source2":
            print(f"Source 1 rows indexed for recall: {source_rows:,}")
        insert_truth(connection, truth_path, f"{prefix}-", chunk_size)
        raw_rows, duplicates = insert_candidates(connection, candidate_path, f"{prefix}-", chunk_size)
        results = source_metrics(connection)
        examples = fetch_examples(connection, source1_path, target_path, chunk_size)
        return results, raw_rows, duplicates, examples
    finally:
        connection.close()
        db_path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    if args.chunk_size <= 0:
        print("--chunk-size must be positive.", file=sys.stderr)
        return 2

    data_dir = args.data_dir.resolve()
    candidate_dir = args.candidate_dir.resolve()
    report_path = args.report.resolve()
    required = [
        data_dir / "train_source1_cleaned.tsv",
        data_dir / "train_source2_cleaned.tsv",
        data_dir / "train_source3_cleaned.tsv",
        data_dir / "train_ground_truth_cleaned.tsv",
        candidate_dir / "candidate_pairs_source2.tsv",
        candidate_dir / "candidate_pairs_source3.tsv",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        print("Required input files are missing:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2
    if report_path.exists():
        print(f"Refusing to overwrite existing report: {report_path}", file=sys.stderr)
        return 2

    candidate_dir_headers = {
        "candidate_pairs_source2.tsv": CANDIDATE_COLUMNS,
        "candidate_pairs_source3.tsv": CANDIDATE_COLUMNS,
    }
    for filename, expected_columns in candidate_dir_headers.items():
        header = pd.read_csv(candidate_dir / filename, sep="\t", nrows=0).columns.tolist()
        if header != expected_columns:
            print(f"Unexpected candidate header in {filename}: {header}", file=sys.stderr)
            return 2

    results: dict[str, dict[str, Any]] = {}
    raw_rows: dict[str, int] = {}
    duplicates: dict[str, int] = {}
    examples: dict[str, list[tuple[str, str, dict[str, str] | None, dict[str, str] | None]]] = {}
    with tempfile.TemporaryDirectory(prefix="candidate_recall_") as temp_name:
        temp_dir = Path(temp_name)
        for label, prefix in (("source2", "S2"), ("source3", "S3")):
            print(f"Validating Source 1 -> {prefix} candidates in chunks...")
            results[label], raw_rows[label], duplicates[label], examples[label] = validate_source(
                label,
                prefix,
                data_dir,
                candidate_dir,
                args.chunk_size,
                temp_dir,
            )

    report_text = format_report(results, raw_rows, duplicates, examples)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8", newline="") as report_file:
        report_file.write(report_text)
    print(report_text)
    print(f"Saved report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())