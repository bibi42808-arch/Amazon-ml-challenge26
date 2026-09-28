"""Create non-destructive, normalized TSV copies for the entity data."""

from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import TextIO

import pandas as pd


TEXT_COLUMNS = {"business_name", "business_address", "country"}
ID_COLUMNS = {"entity_id", "source1_entity_id"}
MATCH_LIST_COLUMNS = {"matched_entity_ids"}
CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE = re.compile(r"\s+")


def normalized_text(values: pd.Series) -> pd.Series:
    """Normalize text for comparison while leaving punctuation and Unicode intact."""
    cleaned = values.str.replace(CONTROL_CHARACTERS, "", regex=True)
    cleaned = cleaned.str.replace(WHITESPACE, " ", regex=True).str.strip()
    return cleaned.str.casefold()


def normalize_match_list(value: str) -> str:
    """Trim whitespace around IDs without changing their case or order."""
    if value == "":
        return ""
    return ",".join(item.strip() for item in value.split(","))


def add_normalized_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    """Append derived columns and return change and missing-value counts."""
    changed: dict[str, int] = {}
    blank_counts: dict[str, int] = {}

    for column in frame.columns:
        values = frame[column].astype("string")
        blank_counts[column] = int(values.str.strip().eq("").sum())

        if column in TEXT_COLUMNS:
            normalized = normalized_text(values)
        elif column in ID_COLUMNS:
            normalized = values.str.strip()
        elif column in MATCH_LIST_COLUMNS:
            normalized = values.map(normalize_match_list)
        else:
            continue

        output_column = f"{column}_normalized"
        frame[output_column] = normalized
        changed[output_column] = int(values.ne(normalized).sum())

    return frame, changed, blank_counts


def output_relative_path(relative_path: Path | PurePosixPath) -> Path:
    """Keep train/test folders and mark the copy so it cannot look original."""
    path = Path(*relative_path.parts)
    return path.with_name(f"{path.stem}_cleaned.tsv")


def find_directory_inputs(input_path: Path) -> list[tuple[Path, Path]]:
    dataset_root = input_path / "dataset" if (input_path / "dataset").is_dir() else input_path
    files = []
    for source_path in sorted(dataset_root.rglob("*.tsv")):
        relative_path = source_path.relative_to(dataset_root)
        if source_path.name.startswith("._"):
            continue
        if relative_path.parts and relative_path.parts[0] in {"train", "test"}:
            files.append((source_path, relative_path))
    return files


def find_archive_inputs(archive: zipfile.ZipFile) -> list[tuple[str, Path]]:
    expected_files = {
        "train": {
            "train_source1.tsv",
            "train_source2.tsv",
            "train_source3.tsv",
            "train_ground_truth.tsv",
        },
        "test": {"test_source1.tsv", "test_source2.tsv", "test_source3.tsv"},
    }
    files = []
    for member in archive.namelist():
        member_path = PurePosixPath(member)
        if (
            member_path.suffix.lower() != ".tsv"
            or member_path.name.startswith("._")
            or "__MACOSX" in member_path.parts
            or "dataset" not in member_path.parts
        ):
            continue
        dataset_index = member_path.parts.index("dataset")
        dataset_parts = member_path.parts[dataset_index + 1 :]
        if len(dataset_parts) != 2:
            continue
        split, filename = dataset_parts
        if filename in expected_files.get(split, set()):
            files.append((member, Path(split, filename)))
    return sorted(files, key=lambda item: item[1].as_posix())


def clean_stream(
    source: TextIO,
    output_path: Path,
    source_name: str,
    chunk_size: int,
) -> None:
    row_count = 0
    changed_counts: dict[str, int] = {}
    blank_counts: dict[str, int] = {}
    first_chunk = True

    reader = pd.read_csv(
        source,
        sep="\t",
        dtype="string",
        keep_default_na=False,
        na_filter=False,
        chunksize=chunk_size,
    )

    for chunk in reader:
        row_count += len(chunk)
        chunk, changes, blanks = add_normalized_columns(chunk)

        for column, count in changes.items():
            changed_counts[column] = changed_counts.get(column, 0) + count
        for column, count in blanks.items():
            blank_counts[column] = blank_counts.get(column, 0) + count

        chunk.to_csv(
            output_path,
            sep="\t",
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            na_rep="",
            lineterminator="\n",
        )
        first_chunk = False

    if first_chunk:
        raise ValueError(f"No data or header was found in {source_name}")

    missing_summary = {column: count for column, count in blank_counts.items() if count}
    print(f"\n{source_name}")
    print(f"  Rows copied: {row_count:,}")
    print(f"  Blank/whitespace-only source values kept: {missing_summary or 'none'}")
    print(f"  Values changed in added normalized columns: {changed_counts or 'none'}")
    print(f"  Output: {output_path}")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Inspect and normalize TSV fields without deleting rows or source columns."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "dataset",
        help="Dataset directory or ZIP archive (default: project dataset/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "output",
        help="Folder for cleaned TSV copies (default: project output/).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Rows processed per chunk (default: 100000).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_root = args.output.resolve()

    if args.chunk_size <= 0:
        print("--chunk-size must be a positive integer.", file=sys.stderr)
        return 2

    if not input_path.exists():
        print(f"Input path does not exist: {input_path}", file=sys.stderr)
        return 2

    if input_path.is_file():
        try:
            archive = zipfile.ZipFile(input_path)
        except zipfile.BadZipFile:
            print(f"Input file is not a readable ZIP archive: {input_path}", file=sys.stderr)
            return 2
        with archive:
            inputs = find_archive_inputs(archive)
            if not inputs:
                print("No TSV files under dataset/train or dataset/test were found.", file=sys.stderr)
                return 2
            planned_outputs = [output_root / output_relative_path(relative) for _, relative in inputs]
            existing = [path for path in planned_outputs if path.exists()]
            if existing:
                print("Refusing to overwrite existing output files:", file=sys.stderr)
                for path in existing:
                    print(f"  {path}", file=sys.stderr)
                return 2

            for member, relative in inputs:
                destination = output_root / output_relative_path(relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as binary_source:
                    with io.TextIOWrapper(binary_source, encoding="utf-8-sig", errors="replace", newline="") as source:
                        clean_stream(source, destination, member, args.chunk_size)
    else:
        inputs = find_directory_inputs(input_path)
        if not inputs:
            print("No TSV files under train/ or test/ were found.", file=sys.stderr)
            return 2
        planned_outputs = [output_root / output_relative_path(relative) for _, relative in inputs]
        existing = [path for path in planned_outputs if path.exists()]
        if existing:
            print("Refusing to overwrite existing output files:", file=sys.stderr)
            for path in existing:
                print(f"  {path}", file=sys.stderr)
            return 2

        for source_path, relative in inputs:
            destination = output_root / output_relative_path(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as source:
                clean_stream(source, destination, str(source_path), args.chunk_size)

    print("\nCleaning complete. Original inputs were not modified; no rows were removed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())