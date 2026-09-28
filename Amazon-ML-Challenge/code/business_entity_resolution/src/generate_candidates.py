"""Generate bounded entity-resolution candidates using deterministic blocking."""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import sys
import tempfile
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd


SOURCE1_FILE = "train_source1_cleaned.tsv"
TARGET_FILES = {
    "source2": "train_source2_cleaned.tsv",
    "source3": "train_source3_cleaned.tsv",
}
SOURCE_COLUMNS = [
    "entity_id_normalized",
    "business_name_normalized",
    "business_address_normalized",
    "country_normalized",
]
GENERIC_NAME_TOKENS = {
    "and",
    "associates",
    "business",
    "company",
    "co",
    "corp",
    "corporation",
    "group",
    "inc",
    "incorporated",
    "international",
    "limited",
    "llc",
    "llp",
    "ltd",
    "of",
    "private",
    "public",
    "services",
    "solutions",
    "the",
    "trading",
}
GENERIC_ADDRESS_TOKENS = {
    "avenue",
    "boulevard",
    "building",
    "district",
    "floor",
    "near",
    "opposite",
    "road",
    "sector",
    "street",
    "suite",
    "tower",
    "unit",
}
LEGAL_SUFFIXES = (
    "private limited",
    "pvt ltd",
    "pvt. ltd.",
    "pvt limited",
    "limited",
    "llp",
    "ltd",
    "ltd.",
    "inc",
    "incorporated",
    "corporation",
    "company",
    "co",
    "associates",
    "group",
    "solutions",
    "services",
)
PLACEHOLDER_VALUES = {"na", "n/a", "null", "none", "nil", "unknown", "undefined", "-"}
TOKEN_FANOUT_LIMIT = 50
OTHER_BLOCK_FANOUT_LIMIT = 50
TOKENS_INDEXED_PER_FIELD = 8
TOKENS_RETAINED_PER_SOURCE1_FIELD = 5
UNUSUALLY_LARGE_CANDIDATE_COUNT = 100
OUTPUT_COLUMNS = ["source1_entity_id", "candidate_entity_id", "blocking_method"]


def clean_key_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def is_useful(value: str, minimum_length: int = 2) -> bool:
    return len(value) >= minimum_length and value.casefold() not in PLACEHOLDER_VALUES


def country_key(country: str, value: str) -> str:
    return f"{country}\x1f{value}"


def name_tokens(name: str) -> list[str]:
    """Return informative Unicode-aware name tokens for target-side indexing."""
    return field_tokens(name, GENERIC_NAME_TOKENS, minimum_length=6)


def address_tokens(address: str) -> list[str]:
    """Return informative Unicode-aware address tokens, including long numbers."""
    return field_tokens(address, GENERIC_ADDRESS_TOKENS, minimum_length=3, address=True)


def field_tokens(
    value: str,
    generic_tokens: set[str],
    minimum_length: int,
    address: bool = False,
) -> list[str]:
    token_parts: list[str] = []
    tokens: set[str] = set()
    for character in value:
        category = unicodedata.category(character)
        if category[0] in {"L", "N"} or (category[0] == "M" and token_parts):
            token_parts.append(character)
        elif token_parts:
            tokens.add("".join(token_parts))
            token_parts.clear()
    if token_parts:
        tokens.add("".join(token_parts))

    tokens = {
        token
        for token in tokens
        if (len(token) >= minimum_length and (not address or not token.isdigit()))
        or (address and token.isdigit() and len(token) >= 3)
        if token.casefold() not in generic_tokens
    }
    sort_key = (lambda token: (not token.isdigit(), -len(token), token)) if address else (lambda token: (-len(token), token))
    return sorted(tokens, key=sort_key)[:TOKENS_INDEXED_PER_FIELD]


def normalize_block_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if not text:
        return ""
    normalized = []
    for character in text.casefold():
        category = unicodedata.category(character)
        if category[0] in {"L", "N"} or character.isspace() or character in "-_/":
            normalized.append(character)
        elif character in ".,()&":
            normalized.append(" ")
        elif character in "/":
            normalized.append(" ")
        else:
            normalized.append(" ")
    return re.sub(r"\s+", " ", "".join(normalized)).strip()


def strip_legal_suffixes(value: str) -> set[str]:
    variants: set[str] = set()
    compact = normalize_block_text(value)
    if not compact:
        return variants
    variants.add(compact)
    for suffix in LEGAL_SUFFIXES:
        suffix_key = normalize_block_text(suffix)
        if not suffix_key:
            continue
        if compact.endswith(suffix_key):
            prefix = compact[: -len(suffix_key)].strip()
            if prefix:
                variants.add(prefix)
        if compact.endswith(f" {suffix_key}"):
            prefix = compact[: -(len(suffix_key) + 1)].strip()
            if prefix:
                variants.add(prefix)
    return {variant for variant in variants if is_useful(variant, minimum_length=2)}


def informative_tokens(value: str, generic_tokens: set[str], minimum_length: int, address: bool = False) -> list[str]:
    normalized = normalize_block_text(value)
    tokens = field_tokens(normalized, generic_tokens, minimum_length=minimum_length, address=address)
    if not tokens:
        tokens = [token for token in normalized.split() if len(token) >= minimum_length and token not in generic_tokens]
    return tokens[:TOKENS_INDEXED_PER_FIELD]


def blocking_keys(
    entity_id: str,
    name: str,
    address: str,
    country: str,
    fast_exact_only: bool = False,
) -> Iterator[tuple[str, str, str]]:
    has_country = is_useful(country)
    normalized_name = normalize_block_text(name)
    normalized_address = normalize_block_text(address)

    if is_useful(normalized_name):
        for variant in strip_legal_suffixes(normalized_name):
            yield "exact_name", variant, entity_id
            if has_country:
                yield "country_exact_name", country_key(country, variant), entity_id
            if not fast_exact_only and len(variant) <= 24:
                yield "name_prefix", variant[:10], entity_id

        name_tokens = (
            []
            if fast_exact_only
            else informative_tokens(normalized_name, GENERIC_NAME_TOKENS, 3, address=False)
        )
        if name_tokens:
            for token in name_tokens:
                yield "name_token", token, entity_id
                if has_country:
                    yield "country_name_token", country_key(country, token), entity_id
            pair = " ".join(name_tokens[:2])
            if pair:
                yield "name_pair", pair, entity_id
                if has_country:
                    yield "country_name_pair", country_key(country, pair), entity_id

    if is_useful(normalized_address, minimum_length=8):
        for variant in {normalized_address}:
            if is_useful(variant, minimum_length=8):
                yield "exact_address", variant, entity_id
                if has_country:
                    yield "country_exact_address", country_key(country, variant), entity_id

        address_tokens = (
            []
            if fast_exact_only
            else informative_tokens(normalized_address, GENERIC_ADDRESS_TOKENS, 3, address=True)
        )
        if address_tokens:
            for token in address_tokens:
                yield "address_token", token, entity_id
                if has_country:
                    yield "country_address_token", country_key(country, token), entity_id
            pair = " ".join(address_tokens[:2])
            if pair:
                yield "address_pair", pair, entity_id
                if has_country:
                    yield "country_address_pair", country_key(country, pair), entity_id

        if has_country and not fast_exact_only:
            for token in informative_tokens(normalized_address, GENERIC_ADDRESS_TOKENS, 2, address=True):
                yield "country_address_token", country_key(country, token), entity_id

    if has_country and is_useful(normalized_name) and not fast_exact_only:
        if has_country:
            for token in informative_tokens(normalized_name, GENERIC_NAME_TOKENS, 2, address=False):
                yield "country_name_token", country_key(country, token), entity_id


def row_values(frame: pd.DataFrame) -> Iterator[tuple[str, str, str, str]]:
    for values in frame.itertuples(index=False, name=None):
        yield tuple(clean_key_value(value) for value in values)  # type: ignore[misc]


def open_tsv_chunks(path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    return pd.read_csv(
        path,
        sep="\t",
        usecols=SOURCE_COLUMNS,
        dtype="string",
        keep_default_na=False,
        na_filter=False,
        chunksize=chunk_size,
    )


def create_database(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-262144")
    connection.execute(
        """
        CREATE TABLE postings (
            method TEXT NOT NULL,
            block_key TEXT NOT NULL,
            entity_id TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE candidates (
            source1_entity_id TEXT NOT NULL,
            candidate_entity_id TEXT NOT NULL,
            blocking_method TEXT NOT NULL,
            PRIMARY KEY (source1_entity_id, candidate_entity_id)
        ) WITHOUT ROWID
        """
    )
    connection.execute(
        "CREATE TABLE source1_ids (source1_entity_id TEXT PRIMARY KEY) WITHOUT ROWID"
    )
    connection.execute(
        """
        CREATE TEMP TABLE source1_blocks (
            source1_entity_id TEXT NOT NULL,
            method TEXT NOT NULL,
            block_key TEXT NOT NULL,
            PRIMARY KEY (source1_entity_id, method, block_key)
        ) WITHOUT ROWID
        """
    )
    return connection


def build_target_index(
    connection: sqlite3.Connection,
    target_path: Path,
    chunk_size: int,
    fast_exact_only: bool = False,
) -> int:
    target_rows = 0
    insert_sql = "INSERT INTO postings(method, block_key, entity_id) VALUES (?, ?, ?)"

    for chunk in open_tsv_chunks(target_path, chunk_size):
        target_rows += len(chunk)
        posting_rows = sorted(
            posting
            for entity_id, name, address, country in row_values(chunk)
            if entity_id
            for posting in blocking_keys(entity_id, name, address, country, fast_exact_only)
        )
        connection.executemany(insert_sql, posting_rows)

    connection.commit()
    connection.execute(
        "CREATE TABLE unique_postings AS "
        "SELECT DISTINCT method, block_key, entity_id FROM postings"
    )
    connection.execute("DROP TABLE postings")
    connection.execute("ALTER TABLE unique_postings RENAME TO postings")
    connection.execute(
        "CREATE UNIQUE INDEX postings_lookup ON postings(method, block_key, entity_id)"
    )
    connection.execute(
        "CREATE TABLE block_counts AS "
        "SELECT method, block_key, COUNT(*) AS target_count "
        "FROM postings GROUP BY method, block_key"
    )
    connection.execute(
        "CREATE UNIQUE INDEX block_counts_lookup ON block_counts(method, block_key)"
    )
    connection.commit()
    return target_rows


def insert_source1_chunk(
    connection: sqlite3.Connection,
    chunk: pd.DataFrame,
    fast_exact_only: bool = False,
) -> int:
    records_processed = len(chunk)
    block_rows: list[tuple[str, str, str]] = []
    source_ids: set[str] = set()

    for entity_id, name, address, country in row_values(chunk):
        if not entity_id:
            continue
        source_ids.add(entity_id)
        for method, key, _ in blocking_keys(entity_id, name, address, country, fast_exact_only):
            block_rows.append((entity_id, method, key))

    connection.executemany(
        "INSERT OR IGNORE INTO source1_ids(source1_entity_id) VALUES (?)",
        ((entity_id,) for entity_id in source_ids),
    )
    connection.execute("DELETE FROM source1_blocks")
    connection.executemany(
        "INSERT OR IGNORE INTO source1_blocks(source1_entity_id, method, block_key) "
        "VALUES (?, ?, ?)",
        sorted(block_rows),
    )

    candidate_cursor = connection.execute(
        """
        WITH ranked_source_blocks AS (
            SELECT
                source_blocks.source1_entity_id,
                source_blocks.method,
                source_blocks.block_key,
                counts.target_count,
                ROW_NUMBER() OVER (
                    PARTITION BY source_blocks.source1_entity_id, source_blocks.method
                    ORDER BY counts.target_count, source_blocks.block_key
                ) AS token_rank
            FROM source1_blocks AS source_blocks
            JOIN block_counts AS counts
              ON counts.method = source_blocks.method
             AND counts.block_key = source_blocks.block_key
        ),
        eligible_source_blocks AS (
            SELECT source1_entity_id, method, block_key
            FROM ranked_source_blocks
            WHERE (
                method = 'country_name_token'
                AND target_count <= ?
                AND token_rank <= ?
            ) OR (
                method = 'country_address_token'
                AND target_count <= ?
                AND token_rank <= ?
            ) OR (
                method NOT IN ('country_name_token', 'country_address_token')
                AND target_count <= ?
            )
        )
        SELECT DISTINCT
            eligible_source_blocks.source1_entity_id,
            postings.entity_id,
            eligible_source_blocks.method
        FROM eligible_source_blocks
        JOIN postings
          ON postings.method = eligible_source_blocks.method
         AND postings.block_key = eligible_source_blocks.block_key
        ORDER BY eligible_source_blocks.source1_entity_id, postings.entity_id, eligible_source_blocks.method
        """,
        (
            TOKEN_FANOUT_LIMIT,
            TOKENS_RETAINED_PER_SOURCE1_FIELD,
            TOKEN_FANOUT_LIMIT,
            TOKENS_RETAINED_PER_SOURCE1_FIELD,
            OTHER_BLOCK_FANOUT_LIMIT,
        ),
    )
    upsert_sql = """
        INSERT INTO candidates(source1_entity_id, candidate_entity_id, blocking_method)
        VALUES (?, ?, ?)
        ON CONFLICT(source1_entity_id, candidate_entity_id) DO UPDATE SET
            blocking_method = CASE
                WHEN instr(',' || candidates.blocking_method || ',', ',' || excluded.blocking_method || ',') > 0
                THEN candidates.blocking_method
                ELSE candidates.blocking_method || ',' || excluded.blocking_method
            END
    """
    while rows := candidate_cursor.fetchmany(50_000):
        connection.executemany(upsert_sql, rows)
    connection.commit()
    return records_processed


def export_candidates(connection: sqlite3.Connection, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("x", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
        writer.writerow(OUTPUT_COLUMNS)
        cursor = connection.execute(
            "SELECT source1_entity_id, candidate_entity_id, blocking_method "
            "FROM candidates ORDER BY source1_entity_id, candidate_entity_id"
        )
        while rows := cursor.fetchmany(100_000):
            writer.writerows(rows)
            count += len(rows)
    return count


def percentile_from_histogram(histogram: list[tuple[int, int]], percentile: float) -> int:
    total = sum(frequency for _, frequency in histogram)
    if total == 0:
        return 0
    target_rank = max(1, int((total - 1) * percentile) + 1)
    cumulative = 0
    for candidate_count, frequency in histogram:
        cumulative += frequency
        if cumulative >= target_rank:
            return candidate_count
    return histogram[-1][0]


def print_candidate_report(
    connection: sqlite3.Connection,
    records_processed: int,
    output_path: Path,
    source_label: str,
) -> int:
    pair_count = int(connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
    represented = int(
        connection.execute(
            "SELECT COUNT(DISTINCT source1_entity_id) FROM candidates"
        ).fetchone()[0]
    )
    unique_source1_ids = int(
        connection.execute("SELECT COUNT(*) FROM source1_ids").fetchone()[0]
    )
    histogram = [
        (int(candidate_count), int(frequency))
        for candidate_count, frequency in connection.execute(
            """
            SELECT candidate_count, COUNT(*)
            FROM (
                SELECT source1_entity_id, COUNT(*) AS candidate_count
                FROM candidates
                GROUP BY source1_entity_id
            )
            GROUP BY candidate_count
            ORDER BY candidate_count
            """
        )
    ]
    zero_candidates = unique_source1_ids - represented
    unusually_large = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT source1_entity_id
                FROM candidates
                GROUP BY source1_entity_id
                HAVING COUNT(*) > ?
            )
            """,
            (UNUSUALLY_LARGE_CANDIDATE_COUNT,),
        ).fetchone()[0]
    )

    print(f"\n{source_label.upper()} CANDIDATE REPORT")
    print(f"Source 1 records processed: {records_processed:,}")
    print(f"Unique Source 1 IDs: {unique_source1_ids:,}")
    print(f"Candidate pairs generated: {pair_count:,}")
    print(f"Unique Source 1 IDs represented: {represented:,}")
    print(f"Source 1 IDs with zero candidates: {zero_candidates:,}")
    print(
        "Source 1 IDs with unusually large candidate sets (> {}): {:,}".format(
            UNUSUALLY_LARGE_CANDIDATE_COUNT, unusually_large
        )
    )
    if histogram:
        print(
            "Candidate-count distribution per represented Source 1 ID: "
            "min={}, median={}, p90={}, p95={}, p99={}, max={}".format(
                histogram[0][0],
                percentile_from_histogram(histogram, 0.50),
                percentile_from_histogram(histogram, 0.90),
                percentile_from_histogram(histogram, 0.95),
                percentile_from_histogram(histogram, 0.99),
                histogram[-1][0],
            )
        )
        print("Candidate-count histogram (candidates per Source 1 ID: number of IDs):")
        print(", ".join(f"{count}: {frequency:,}" for count, frequency in histogram))
    else:
        print("Candidate-count distribution: no candidates generated")
    print(f"Output: {output_path}")
    return pair_count


def generate_for_target(
    source1_path: Path,
    target_path: Path,
    output_path: Path,
    chunk_size: int,
    temp_directory: Path,
    source_label: str,
    fast_exact_only: bool = False,
) -> None:
    database_path = temp_directory / f"candidate_{source_label}.sqlite"
    connection = create_database(database_path)
    try:
        print(f"\nBuilding disk-backed index for {target_path.name}...")
        target_rows = build_target_index(connection, target_path, chunk_size, fast_exact_only)
        print(f"Indexed target rows: {target_rows:,}")

        records_processed = 0
        for chunk_number, chunk in enumerate(open_tsv_chunks(source1_path, chunk_size), start=1):
            records_processed += insert_source1_chunk(connection, chunk, fast_exact_only)
            if chunk_number % 5 == 0:
                print(f"  Processed {records_processed:,} Source 1 records...")

        pair_count = int(connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
        export_count = export_candidates(connection, output_path)
        if export_count != pair_count:
            raise RuntimeError(
                f"Exported {export_count:,} pairs but database contained {pair_count:,}."
            )
        print_candidate_report(connection, records_processed, output_path, source_label)
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate deterministic, fanout-limited candidates from cleaned TSVs."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "output_clean" / "train",
        help="Directory containing the cleaned training TSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "output_candidates",
        help="Output root; candidate files are written into its train/ subfolder.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="Rows loaded per pandas chunk (default: 50000).",
    )
    parser.add_argument(
        "--fast-exact-only",
        action="store_true",
        help="Use exact normalized name/address blocks only (faster, lower recall).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunk_size <= 0:
        print("--chunk-size must be positive.", file=sys.stderr)
        return 2

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve() / "train"
    source1_path = data_dir / SOURCE1_FILE
    target_paths = {
        label: data_dir / filename
        for label, filename in TARGET_FILES.items()
    }
    missing_files = [
        path for path in [source1_path, *target_paths.values()] if not path.is_file()
    ]
    if missing_files:
        print("Required cleaned training files are missing:", file=sys.stderr)
        for path in missing_files:
            print(f"  {path}", file=sys.stderr)
        return 2

    output_paths = {
        label: output_dir / f"candidate_pairs_{label}.tsv"
        for label in TARGET_FILES
    }
    existing_outputs = [path for path in output_paths.values() if path.exists()]
    if existing_outputs:
        print("Refusing to overwrite existing candidate files:", file=sys.stderr)
        for path in existing_outputs:
            print(f"  {path}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="entity_candidates_") as temporary_name:
        temp_directory = Path(temporary_name)
        for label, target_path in target_paths.items():
            generate_for_target(
                source1_path,
                target_path,
                output_paths[label],
                args.chunk_size,
                temp_directory,
                label,
                fast_exact_only=args.fast_exact_only,
            )

    print("\nCandidate generation complete. Existing input and cleaned files were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())