"""Train and validate grouped candidate-pair logistic models using local data only."""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .features import FEATURE_NAMES, feature_vector
except ImportError:
    from features import FEATURE_NAMES, feature_vector

SOURCE1_COLUMNS = ["entity_id", "business_name_normalized", "business_address_normalized", "country_normalized"]
TARGET_COLUMNS = ["entity_id", "business_name_normalized", "business_address_normalized", "country_normalized"]
CANDIDATE_COLUMNS = ["source1_entity_id", "candidate_entity_id", "blocking_method"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
BATCH_SIZE = 2048
EPOCHS = 3
THRESHOLDS = np.arange(0.10, 0.951, 0.05, dtype=np.float64)


def validation_group(entity_id: str) -> int:
    digest = hashlib.blake2b(entity_id.encode("utf-8"), digest_size=8, person=b"aml-split").digest()
    return int.from_bytes(digest, "big") % 5


def read_chunks(path: Path, columns: list[str], chunk_size: int):
    return pd.read_csv(
        path,
        sep="\t",
        usecols=columns,
        dtype="string",
        keep_default_na=False,
        na_filter=False,
        chunksize=chunk_size,
    )


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
            country TEXT NOT NULL,
            is_validation INTEGER NOT NULL
        );
        CREATE TABLE target_staging (
            entity_id TEXT NOT NULL,
            name TEXT NOT NULL,
            address TEXT NOT NULL,
            country TEXT NOT NULL,
            source TEXT NOT NULL
        );
        CREATE TABLE truth (
            source1_entity_id TEXT NOT NULL,
            candidate_entity_id TEXT NOT NULL,
            source TEXT NOT NULL
        );
        CREATE TABLE candidate_staging (
            source1_entity_id TEXT NOT NULL,
            candidate_entity_id TEXT NOT NULL,
            blocking_method TEXT NOT NULL,
            source TEXT NOT NULL
        );
        CREATE TABLE validation_ids (
            entity_id TEXT PRIMARY KEY
        ) WITHOUT ROWID;
        """
    )
    return connection


def load_records(connection: sqlite3.Connection, data_dir: Path, chunk_size: int) -> tuple[int, dict[str, int]]:
    source1_path = data_dir / "train_source1_cleaned.tsv"
    source1_count = 0
    source_sql = "INSERT INTO source1_staging VALUES (?, ?, ?, ?, ?)"
    for frame in read_chunks(source1_path, SOURCE1_COLUMNS, chunk_size):
        values = []
        for entity_id, name, address, country in frame.itertuples(index=False, name=None):
            entity_id = str(entity_id)
            values.append((entity_id, str(name), str(address), str(country), int(validation_group(entity_id) == 0)))
            if validation_group(entity_id) == 0:
                connection.execute("INSERT OR IGNORE INTO validation_ids VALUES (?)", (entity_id,))
        source1_count += len(frame)
        connection.executemany(source_sql, values)
    connection.commit()

    target_counts: dict[str, int] = {}
    target_sql = "INSERT INTO target_staging VALUES (?, ?, ?, ?, ?)"
    for source in ("source2", "source3"):
        target_path = data_dir / f"train_{source}_cleaned.tsv"
        count = 0
        for frame in read_chunks(target_path, TARGET_COLUMNS, chunk_size):
            values = [
                (str(entity_id), str(name), str(address), str(country), source)
                for entity_id, name, address, country in frame.itertuples(index=False, name=None)
            ]
            connection.executemany(target_sql, values)
            count += len(frame)
        connection.commit()
        target_counts[source] = count
        print(f"Loaded {count:,} {source} records to disk-backed lookup staging.", flush=True)

    return source1_count, target_counts


def load_labels_and_candidates(
    connection: sqlite3.Connection,
    data_dir: Path,
    candidate_dir: Path,
    chunk_size: int,
) -> dict[str, int]:
    truth_path = data_dir / "train_ground_truth_cleaned.tsv"
    truth_insert = "INSERT OR IGNORE INTO truth VALUES (?, ?, ?)"
    for frame in read_chunks(truth_path, GROUND_TRUTH_COLUMNS, chunk_size):
        truth_rows = []
        for source1_id, match_list in frame.itertuples(index=False, name=None):
            for candidate_id in str(match_list).split(","):
                candidate_id = candidate_id.strip()
                if candidate_id.startswith("S2-"):
                    truth_rows.append((str(source1_id), candidate_id, "source2"))
                elif candidate_id.startswith("S3-"):
                    truth_rows.append((str(source1_id), candidate_id, "source3"))
        connection.executemany(truth_insert, truth_rows)
    connection.commit()

    pair_counts: dict[str, int] = {}
    candidate_insert = "INSERT OR IGNORE INTO candidate_staging VALUES (?, ?, ?, ?)"
    for source in ("source2", "source3"):
        candidate_path = candidate_dir / f"candidate_pairs_{source}.tsv"
        pair_count = 0
        for frame in read_chunks(candidate_path, CANDIDATE_COLUMNS, chunk_size):
            values = [
                (str(source1_id), str(candidate_id), str(method), source)
                for source1_id, candidate_id, method in frame.itertuples(index=False, name=None)
            ]
            connection.executemany(candidate_insert, values)
            pair_count += len(frame)
        connection.commit()
        pair_counts[source] = pair_count
        print(f"Loaded {pair_count:,} {source} candidate rows.", flush=True)

    connection.execute("CREATE UNIQUE INDEX source1_lookup ON source1_staging(entity_id)")
    connection.execute("CREATE UNIQUE INDEX target_lookup ON target_staging(source, entity_id)")
    connection.execute("CREATE UNIQUE INDEX truth_lookup ON truth(source, source1_entity_id, candidate_entity_id)")
    connection.execute("CREATE UNIQUE INDEX candidate_lookup ON candidate_staging(source, source1_entity_id, candidate_entity_id)")
    connection.execute(
        "CREATE TABLE candidate_counts AS SELECT source, source1_entity_id, COUNT(*) AS candidate_count "
        "FROM candidate_staging GROUP BY source, source1_entity_id"
    )
    connection.execute("CREATE UNIQUE INDEX candidate_counts_lookup ON candidate_counts(source, source1_entity_id)")
    connection.commit()
    return pair_counts


def candidate_query(source: str, training_sample: bool = False) -> str:
    sample_filter = "AND s.is_validation = 0 AND (truth.candidate_entity_id IS NOT NULL OR CAST(substr(c.candidate_entity_id, 4) AS INTEGER) % 5 = 0)" if training_sample else ""
    return f"""
        SELECT
            c.source1_entity_id, c.candidate_entity_id, c.blocking_method,
            s.name, s.address, s.country,
            t.name, t.address, t.country,
            COALESCE(cc.candidate_count, 1),
            CASE WHEN truth.candidate_entity_id IS NULL THEN 0 ELSE 1 END,
            s.is_validation
        FROM candidate_staging AS c
        JOIN source1_staging AS s ON s.entity_id = c.source1_entity_id
        JOIN target_staging AS t
          ON t.source = c.source AND t.entity_id = c.candidate_entity_id
        LEFT JOIN truth
          ON truth.source = c.source
         AND truth.source1_entity_id = c.source1_entity_id
         AND truth.candidate_entity_id = c.candidate_entity_id
        LEFT JOIN candidate_counts AS cc
          ON cc.source = c.source AND cc.source1_entity_id = c.source1_entity_id
        WHERE c.source = ?
                {sample_filter}
        ORDER BY c.source1_entity_id, c.candidate_entity_id
    """


def build_features(row: tuple[Any, ...]) -> np.ndarray:
    (
        _source1_id, _candidate_id, method,
        source_name, source_address, source_country,
        target_name, target_address, target_country,
        candidate_count, _label, _is_validation,
    ) = row
    source = {
        "business_name_normalized": source_name,
        "business_address_normalized": source_address,
        "country_normalized": source_country,
    }
    target = {
        "business_name_normalized": target_name,
        "business_address_normalized": target_address,
        "country_normalized": target_country,
    }
    return np.asarray(feature_vector(source, target, str(method), int(candidate_count)), dtype=np.float64)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def train_one_source(
    connection: sqlite3.Connection,
    source: str,
    learning_rate: float,
) -> tuple[np.ndarray, float, int, int]:
    positive_count = int(connection.execute(
        "SELECT COUNT(*) FROM candidate_staging c JOIN truth t "
        "ON t.source=c.source AND t.source1_entity_id=c.source1_entity_id "
        "AND t.candidate_entity_id=c.candidate_entity_id WHERE c.source=? "
        "AND c.source1_entity_id NOT IN (SELECT entity_id FROM validation_ids)",
        (source,),
    ).fetchone()[0])
    train_count = int(connection.execute(
        "SELECT COUNT(*) FROM candidate_staging c "
        "JOIN source1_staging s ON s.entity_id=c.source1_entity_id "
        "LEFT JOIN truth t ON t.source=c.source AND t.source1_entity_id=c.source1_entity_id "
        "AND t.candidate_entity_id=c.candidate_entity_id "
        "WHERE c.source=? AND s.is_validation=0 "
        "AND (t.candidate_entity_id IS NOT NULL OR CAST(substr(c.candidate_entity_id, 4) AS INTEGER) % 5 = 0)",
        (source,),
    ).fetchone()[0])
    negative_count = max(0, train_count - positive_count)
    positive_weight = min(20.0, negative_count / max(1, positive_count))
    weights = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    bias = 0.0
    cursor = connection.execute(candidate_query(source, training_sample=True), (source,))
    batch_features: list[np.ndarray] = []
    batch_labels: list[float] = []
    train_seen = 0

    def update_batch() -> None:
        nonlocal weights, bias
        if not batch_features:
            return
        matrix = np.vstack(batch_features)
        labels = np.asarray(batch_labels, dtype=np.float64)
        sample_weights = np.where(labels > 0, positive_weight, 1.0)
        probabilities = sigmoid(matrix @ weights + bias)
        gradient = (matrix.T @ ((probabilities - labels) * sample_weights)) / len(labels)
        bias_gradient = float(np.sum((probabilities - labels) * sample_weights) / len(labels))
        weights -= learning_rate * (gradient + 1e-4 * weights)
        bias -= learning_rate * bias_gradient
        batch_features.clear()
        batch_labels.clear()

    for epoch in range(EPOCHS):
        if epoch:
            cursor = connection.execute(candidate_query(source, training_sample=True), (source,))
        while rows := cursor.fetchmany(BATCH_SIZE):
            for row in rows:
                if row[11]:
                    continue
                label = float(row[10])
                batch_features.append(build_features(row))
                batch_labels.append(label)
                train_seen += int(epoch == 0)
            update_batch()
        update_batch()
        print(f"  {source}: completed training pass {epoch + 1}/{EPOCHS}.", flush=True)

    return weights, bias, train_seen, positive_count


def score_validation(
    connection: sqlite3.Connection,
    source: str,
    weights: np.ndarray,
    bias: float,
) -> int:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS validation_scores ("
        "source1_entity_id TEXT NOT NULL, candidate_entity_id TEXT NOT NULL, "
        "source TEXT NOT NULL, score REAL NOT NULL, label INTEGER NOT NULL, "
        "PRIMARY KEY(source, source1_entity_id, candidate_entity_id)) WITHOUT ROWID"
    )
    insert_sql = "INSERT OR REPLACE INTO validation_scores VALUES (?, ?, ?, ?, ?)"
    cursor = connection.execute(candidate_query(source), (source,))
    written = 0
    while rows := cursor.fetchmany(BATCH_SIZE):
        score_rows = []
        feature_matrix = []
        valid_rows = []
        for row in rows:
            if not row[11]:
                continue
            feature_matrix.append(build_features(row))
            valid_rows.append(row)
        if feature_matrix:
            probabilities = sigmoid(np.vstack(feature_matrix) @ weights + bias)
            score_rows.extend(
                (str(row[0]), str(row[1]), source, float(score), int(row[10]))
                for row, score in zip(valid_rows, probabilities)
            )
            connection.executemany(insert_sql, score_rows)
            written += len(score_rows)
        if written and written % 100_000 < BATCH_SIZE:
            connection.commit()
    connection.commit()
    return written


def optimize_thresholds(connection: sqlite3.Connection) -> tuple[float, float, float, float, float, int, int]:
    validation_entities = [
        str(row[0]) for row in connection.execute(
            "SELECT entity_id FROM source1_staging WHERE is_validation=1 ORDER BY entity_id"
        )
    ]
    entity_index = {entity_id: index for index, entity_id in enumerate(validation_entities)}
    validation_size = len(validation_entities)
    prediction_counts = np.zeros((validation_size, 2, len(THRESHOLDS)), dtype=np.int32)
    true_positive_counts = np.zeros_like(prediction_counts)
    truth_counts = np.zeros((validation_size, 2), dtype=np.int32)
    for entity_id, source, count in connection.execute(
        "SELECT source1_entity_id, source, COUNT(*) FROM truth "
        "WHERE source1_entity_id IN (SELECT entity_id FROM validation_ids) "
        "GROUP BY source1_entity_id, source"
    ):
        index = entity_index.get(str(entity_id))
        if index is not None:
            truth_counts[index, 0 if source == "source2" else 1] = int(count)

    for entity_id, source, score, label in connection.execute(
        "SELECT source1_entity_id, source, score, label FROM validation_scores"
    ):
        index = entity_index.get(str(entity_id))
        if index is None:
            continue
        target_index = 0 if source == "source2" else 1
        threshold_count = int(np.searchsorted(THRESHOLDS, float(score), side="right"))
        if threshold_count:
            prediction_counts[index, target_index, :threshold_count] += 1
            if label:
                true_positive_counts[index, target_index, :threshold_count] += 1

    best_score = -1.0
    best_source2 = float(THRESHOLDS[0])
    best_source3 = float(THRESHOLDS[0])
    best_precision = 0.0
    best_recall = 0.0
    best_predictions = 0
    best_true_positives = 0
    for index2, threshold2 in enumerate(THRESHOLDS):
        for index3, threshold3 in enumerate(THRESHOLDS):
            predicted = prediction_counts[:, 0, index2] + prediction_counts[:, 1, index3]
            true_positive = true_positive_counts[:, 0, index2] + true_positive_counts[:, 1, index3]
            truth = truth_counts.sum(axis=1)
            score = np.ones(validation_size, dtype=np.float64)
            nonempty = (truth > 0) | (predicted > 0)
            precision = np.divide(true_positive, predicted, out=np.zeros_like(score), where=predicted > 0)
            recall = np.divide(true_positive, truth, out=np.zeros_like(score), where=truth > 0)
            denominator = 0.25 * precision + recall
            score[nonempty] = np.divide(
                1.25 * precision * recall,
                denominator,
                out=np.zeros_like(score),
                where=denominator > 0,
            )[nonempty]
            macro = float(score.mean()) if validation_size else 0.0
            if macro > best_score:
                best_score = macro
                best_source2 = float(threshold2)
                best_source3 = float(threshold3)
                best_predictions = int(predicted.sum())
                best_true_positives = int(true_positive.sum())
                total_truth = int(truth.sum())
                best_precision = best_true_positives / best_predictions if best_predictions else 0.0
                best_recall = best_true_positives / total_truth if total_truth else 0.0
    return (
        best_source2,
        best_source3,
        best_score,
        best_precision,
        best_recall,
        best_predictions,
        best_true_positives,
    )


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train grouped validation logistic models on candidate pairs.")
    parser.add_argument("--data-dir", type=Path, default=project_root / "output_clean" / "train")
    parser.add_argument("--candidate-dir", type=Path, default=project_root / "output_candidates_v2" / "train")
    parser.add_argument("--artifact-dir", type=Path, default=project_root / "models" / "v1")
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunk_size <= 0 or args.learning_rate <= 0:
        print("chunk size and learning rate must be positive", file=sys.stderr)
        return 2
    if args.artifact_dir.exists():
        print(f"Refusing to overwrite existing model artifacts: {args.artifact_dir}", file=sys.stderr)
        return 2

    required = [
        args.data_dir / "train_source1_cleaned.tsv",
        args.data_dir / "train_source2_cleaned.tsv",
        args.data_dir / "train_source3_cleaned.tsv",
        args.data_dir / "train_ground_truth_cleaned.tsv",
        args.candidate_dir / "candidate_pairs_source2.tsv",
        args.candidate_dir / "candidate_pairs_source3.tsv",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        print("Missing required inputs:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2

    args.artifact_dir.mkdir(parents=True)
    report_lines = ["Grouped candidate-pair logistic validation", "Validation split: deterministic 20% of Source 1 IDs."]
    with tempfile.TemporaryDirectory(prefix="entity_model_") as temp_name:
        connection = create_database(Path(temp_name) / "training.sqlite")
        try:
            source1_count, target_counts = load_records(connection, args.data_dir, args.chunk_size)
            pair_counts = load_labels_and_candidates(connection, args.data_dir, args.candidate_dir, args.chunk_size)
            report_lines.append(f"Source 1 entities indexed: {source1_count:,}")
            models = {}
            for source in ("source2", "source3"):
                print(f"\nTraining {source} logistic model...")
                weights, bias, training_pairs, positives = train_one_source(
                    connection, source, args.learning_rate
                )
                validation_pairs = score_validation(connection, source, weights, bias)
                models[source] = (weights, bias)
                report_lines.extend([
                    f"{source}: training candidate pairs={training_pairs:,}",
                    f"{source}: training positives={positives:,}",
                    f"{source}: validation candidate pairs scored={validation_pairs:,}",
                    f"{source}: target records={target_counts[source]:,}",
                    f"{source}: candidate rows loaded={pair_counts[source]:,}",
                ])

            (
                threshold2,
                threshold3,
                macro_f05,
                micro_precision,
                micro_recall,
                predicted_matches,
                correct_matches,
            ) = optimize_thresholds(connection)
            np.savez(
                args.artifact_dir / "logistic_models.npz",
                feature_names=np.asarray(FEATURE_NAMES),
                weights_source2=models["source2"][0],
                bias_source2=np.asarray(models["source2"][1]),
                weights_source3=models["source3"][0],
                bias_source3=np.asarray(models["source3"][1]),
                threshold_source2=np.asarray(threshold2),
                threshold_source3=np.asarray(threshold3),
            )
            report_lines.extend([
                f"Selected Source 2 threshold: {threshold2:.3f}",
                f"Selected Source 3 threshold: {threshold3:.3f}",
                f"Validation macro F0.5: {macro_f05:.6f}",
                f"Validation micro precision: {micro_precision:.6f}",
                f"Validation micro recall: {micro_recall:.6f}",
                f"Validation predicted pairs: {predicted_matches:,}",
                f"Validation correct pairs: {correct_matches:,}",
                "Model: NumPy mini-batch logistic regression, three deterministic passes.",
                "Training negatives: deterministic 20% sample; every positive retained.",
            ])
        finally:
            connection.close()

    report_path = args.artifact_dir / "validation_report.txt"
    with report_path.open("x", encoding="utf-8", newline="") as report_file:
        report_file.write("\n".join(report_lines) + "\n")
    print("\n".join(report_lines))
    print(f"Model artifacts: {args.artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
