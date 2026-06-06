import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from data_preprocessing.path_utils import resolve_manifest_artifact_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an entity-level train/val/test benchmark from the synthetic dataset manifest."
    )
    parser.add_argument("--manifest", required=True, help="Path to the synthetic manifest.json file.")
    parser.add_argument(
        "--output-dir",
        default="data_preprocessing/output/benchmarks",
        help="Directory where the benchmark bundle will be written.",
    )
    parser.add_argument(
        "--benchmark-name",
        default="shanse001_10branch_entity_split",
        help="Name of the benchmark bundle.",
    )
    parser.add_argument(
        "--num-branches",
        type=int,
        default=10,
        help="Keep only datasets whose branch count matches this value.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Entity-level train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Entity-level validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Entity-level test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for entity-level shuffling.")
    parser.add_argument(
        "--train-source",
        choices=["combined", "base"],
        default="combined",
        help="Use the combined CSV or only the base CSV for the training entities.",
    )
    parser.add_argument(
        "--val-source",
        choices=["combined", "base"],
        default="base",
        help="Use the combined CSV or only the base CSV for the validation entities.",
    )
    parser.add_argument(
        "--test-source",
        choices=["combined", "base"],
        default="base",
        help="Use the combined CSV or only the base CSV for the test entities.",
    )
    parser.add_argument(
        "--allow-zero-signal-datasets",
        action="store_true",
        help="Keep datasets whose base CSV never contains a nonzero total current or total power value.",
    )
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> List[Dict[str, object]]:
    payload = json.loads(manifest_path.read_text())
    return list(payload.get("datasets", []))


def select_datasets(items: Sequence[Dict[str, object]], num_branches: int) -> List[Dict[str, object]]:
    selected = [item for item in items if int(item.get("num_branches", -1)) == num_branches]
    if len(selected) < 3:
        raise ValueError(f"Need at least 3 datasets with {num_branches} branches; found {len(selected)}.")
    return sorted(selected, key=lambda item: str(item["dataset_name"]))


def has_nonzero_signal(base_csv_path: Path) -> bool:
    with base_csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if float(row["total_residual_current"]) != 0.0 or float(row["total_power"]) != 0.0:
                return True
    return False


def filter_zero_signal_datasets(
    items: Sequence[Dict[str, object]],
    manifest_path: Path,
) -> Tuple[List[Dict[str, object]], List[str]]:
    kept: List[Dict[str, object]] = []
    dropped: List[str] = []
    for item in items:
        base_csv_path = resolve_manifest_artifact_path(manifest_path, str(item["base_csv_path"]))
        if has_nonzero_signal(base_csv_path):
            kept.append(item)
        else:
            dropped.append(str(item["dataset_name"]))
    return kept, dropped


def split_entities(
    items: Sequence[Dict[str, object]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0.")

    ordered = list(items)
    rng = pd.Series(range(len(ordered))).sample(frac=1.0, random_state=seed).tolist()
    shuffled = [ordered[index] for index in rng]
    total = len(shuffled)
    exact_counts = [total * train_ratio, total * val_ratio, total * test_ratio]
    counts = [int(value) for value in exact_counts]
    remainder = total - sum(counts)
    fractional_order = sorted(
        range(3),
        key=lambda idx: (exact_counts[idx] - counts[idx], idx),
        reverse=True,
    )
    for idx in fractional_order[:remainder]:
        counts[idx] += 1
    train_count, val_count, test_count = counts
    if min(train_count, val_count, test_count) <= 0:
        raise ValueError("Entity split produced an empty subset.")
    train_items = shuffled[:train_count]
    val_items = shuffled[train_count : train_count + val_count]
    test_items = shuffled[train_count + val_count :]
    return train_items, val_items, test_items


def read_split_frames(
    items: Iterable[Dict[str, object]],
    *,
    source: str,
    manifest_path: Path,
) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for item in items:
        csv_key = "csv_path" if source == "combined" else "base_csv_path"
        csv_path = resolve_manifest_artifact_path(manifest_path, str(item[csv_key]))
        frame = pd.read_csv(csv_path)
        frame["entity_id"] = str(item["dataset_name"])
        frames.append(frame)
    return frames


def concat_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("No frames available for concatenation.")
    return pd.concat(frames, ignore_index=True)


def rows_to_hours(rows: int, interval_minutes: int, synthetic_variants: int = 0) -> float:
    effective_rows = rows / float(max(synthetic_variants + 1, 1))
    return effective_rows * interval_minutes / 60.0


def summarize_entities(items: Sequence[Dict[str, object]], *, train_source: str) -> Dict[str, object]:
    combined_rows = sum(int(item["dataset_rows_kept"]) for item in items)
    base_rows = 0
    for item in items:
        variants = int(item.get("synthetic_variants", 0))
        base_rows += int(round(int(item["dataset_rows_kept"]) / float(max(variants + 1, 1))))
    interval_minutes = int(items[0].get("target_interval_minutes", 1))
    return {
        "entity_count": len(items),
        "entity_names": [str(item["dataset_name"]) for item in items],
        "combined_rows": combined_rows,
        "base_rows": base_rows,
        "combined_hours": rows_to_hours(combined_rows, interval_minutes),
        "base_hours": rows_to_hours(base_rows, interval_minutes, synthetic_variants=0),
        "train_source": train_source,
    }


def write_bundle(
    output_dir: Path,
    benchmark_name: str,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    selected_items: Sequence[Dict[str, object]],
    train_items: Sequence[Dict[str, object]],
    val_items: Sequence[Dict[str, object]],
    test_items: Sequence[Dict[str, object]],
    train_source: str,
    val_source: str,
    test_source: str,
    seed: int,
    dropped_zero_signal: Sequence[str],
) -> None:
    bundle_dir = output_dir / benchmark_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    train_csv_path = bundle_dir / f"{benchmark_name}.train.csv"
    val_csv_path = bundle_dir / f"{benchmark_name}.val.csv"
    test_csv_path = bundle_dir / f"{benchmark_name}.test.csv"
    train_frame.to_csv(train_csv_path, index=False)
    val_frame.to_csv(val_csv_path, index=False)
    test_frame.to_csv(test_csv_path, index=False)

    summary = {
        "benchmark_name": benchmark_name,
        "num_selected_entities": len(selected_items),
        "num_branches": int(selected_items[0]["num_branches"]),
        "target_interval_minutes": int(selected_items[0].get("target_interval_minutes", 1)),
        "base_interval_minutes": int(selected_items[0].get("base_interval_minutes", 1)),
        "train_source": train_source,
        "val_source": val_source,
        "test_source": test_source,
        "seed": seed,
        "dropped_zero_signal_datasets": list(dropped_zero_signal),
        "train": summarize_entities(train_items, train_source=train_source),
        "val": summarize_entities(val_items, train_source=val_source),
        "test": summarize_entities(test_items, train_source=test_source),
        "train_csv_path": train_csv_path.name,
        "val_csv_path": val_csv_path.name,
        "test_csv_path": test_csv_path.name,
        "notes": [
            "Entity-level split. The same entity never appears in more than one split.",
            "Each split can independently use combined synthetic CSVs or base CSVs.",
            "All downstream methods should reuse this benchmark bundle unchanged.",
        ],
    }
    summary_path = bundle_dir / "split_config.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote benchmark bundle: {bundle_dir}")
    print(f"Train entities: {', '.join(summary['train']['entity_names'])}")
    print(f"Val entities: {', '.join(summary['val']['entity_names'])}")
    print(f"Test entities: {', '.join(summary['test']['entity_names'])}")
    print(f"Split config: {summary_path}")


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_manifest(manifest_path)
    selected_items = select_datasets(items, args.num_branches)
    dropped_zero_signal: List[str] = []
    if not args.allow_zero_signal_datasets:
        selected_items, dropped_zero_signal = filter_zero_signal_datasets(selected_items, manifest_path)
        if len(selected_items) < 3:
            raise ValueError(
                "Fewer than 3 nonzero datasets remain after zero-signal filtering. "
                f"Dropped: {', '.join(dropped_zero_signal) or 'none'}"
            )
    train_items, val_items, test_items = split_entities(
        selected_items,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    train_frame = concat_frames(read_split_frames(train_items, source=args.train_source, manifest_path=manifest_path))
    val_frame = concat_frames(read_split_frames(val_items, source=args.val_source, manifest_path=manifest_path))
    test_frame = concat_frames(read_split_frames(test_items, source=args.test_source, manifest_path=manifest_path))

    write_bundle(
        output_dir=output_dir,
        benchmark_name=args.benchmark_name,
        train_frame=train_frame,
        val_frame=val_frame,
        test_frame=test_frame,
        selected_items=selected_items,
        train_items=train_items,
        val_items=val_items,
        test_items=test_items,
        train_source=args.train_source,
        val_source=args.val_source,
        test_source=args.test_source,
        seed=args.seed,
        dropped_zero_signal=dropped_zero_signal,
    )


if __name__ == "__main__":
    main()
