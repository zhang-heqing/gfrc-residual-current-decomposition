import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from data_preprocessing.path_utils import resolve_manifest_artifact_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a chronological train/val/test split from one nonzero synthetic dataset bundle."
    )
    parser.add_argument("--manifest", required=True, help="Path to the synthetic manifest.json file.")
    parser.add_argument("--dataset-name", required=True, help="Dataset name from the synthetic manifest.")
    parser.add_argument(
        "--output-dir",
        default="data_preprocessing/output/benchmarks",
        help="Directory where the benchmark bundle will be written.",
    )
    parser.add_argument(
        "--benchmark-name",
        help="Optional benchmark bundle name. Defaults to <dataset-name>_temporal_split.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Chronological train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Chronological validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Chronological test split ratio.")
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> Dict[str, Dict[str, object]]:
    payload = json.loads(manifest_path.read_text())
    result: Dict[str, Dict[str, object]] = {}
    for item in payload.get("datasets", []):
        name = item.get("dataset_name")
        if isinstance(name, str):
            result[name] = item
    return result


def ensure_nonzero_base(base_csv_path: Path) -> None:
    frame = pd.read_csv(base_csv_path, usecols=["total_residual_current"])
    if not (frame["total_residual_current"].astype(float) != 0.0).any():
        raise ValueError(f"Base dataset has no nonzero residual current: {base_csv_path}")


def split_base_frame(
    base_frame: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0.")
    total = len(base_frame)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train_frame = base_frame.iloc[:train_end].copy()
    val_frame = base_frame.iloc[train_end:val_end].copy()
    test_frame = base_frame.iloc[val_end:].copy()
    if min(len(train_frame), len(val_frame), len(test_frame)) <= 0:
        raise ValueError("Chronological split produced an empty subset.")
    return train_frame, val_frame, test_frame


def apply_segment_id(frame: pd.DataFrame, segment_id: str) -> pd.DataFrame:
    result = frame.copy()
    result["segment_id"] = segment_id
    return result


def timestamps_set(frame: pd.DataFrame) -> set:
    return set(frame["timestamp"].tolist())


def subset_variant_frame(variant_path: Path, timestamps: set, segment_id: str) -> pd.DataFrame:
    frame = pd.read_csv(variant_path)
    frame = frame[frame["timestamp"].isin(timestamps)].copy()
    if frame.empty:
        raise ValueError(f"Variant frame became empty after timestamp filtering: {variant_path}")
    return apply_segment_id(frame, segment_id)


def summarize_frame(frame: pd.DataFrame, interval_minutes: int) -> Dict[str, object]:
    return {
        "rows": int(len(frame)),
        "hours": float(len(frame) * interval_minutes / 60.0),
        "start": str(frame["timestamp"].iloc[0]),
        "end": str(frame["timestamp"].iloc[-1]),
    }


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    dataset_map = load_manifest(manifest_path)
    if args.dataset_name not in dataset_map:
        raise ValueError(f"Dataset not found in manifest: {args.dataset_name}")

    item = dataset_map[args.dataset_name]
    benchmark_name = args.benchmark_name or f"{args.dataset_name}_temporal_split"
    output_dir = Path(args.output_dir).resolve()
    bundle_dir = output_dir / benchmark_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    base_csv_path = resolve_manifest_artifact_path(manifest_path, str(item["base_csv_path"]))
    ensure_nonzero_base(base_csv_path)
    base_frame = pd.read_csv(base_csv_path)
    base_frame = base_frame.sort_values("timestamp").reset_index(drop=True)

    train_base, val_base, test_base = split_base_frame(
        base_frame,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    dataset_name = str(item["dataset_name"])
    train_base = apply_segment_id(train_base, f"{dataset_name}_train_base")
    val_base = apply_segment_id(val_base, f"{dataset_name}_val_base")
    test_base = apply_segment_id(test_base, f"{dataset_name}_test_base")

    train_timestamps = timestamps_set(train_base)
    train_frames: List[pd.DataFrame] = [train_base]
    for variant_index, variant_path_value in enumerate(item.get("variant_csv_paths", []), start=1):
        variant_path = resolve_manifest_artifact_path(manifest_path, str(variant_path_value))
        variant_frame = subset_variant_frame(
            variant_path,
            timestamps=train_timestamps,
            segment_id=f"{dataset_name}_train_variant_{variant_index}",
        )
        train_frames.append(variant_frame)

    train_frame = pd.concat(train_frames, ignore_index=True)
    val_frame = val_base.reset_index(drop=True)
    test_frame = test_base.reset_index(drop=True)

    train_csv_path = bundle_dir / f"{benchmark_name}.train.csv"
    val_csv_path = bundle_dir / f"{benchmark_name}.val.csv"
    test_csv_path = bundle_dir / f"{benchmark_name}.test.csv"
    train_frame.to_csv(train_csv_path, index=False)
    val_frame.to_csv(val_csv_path, index=False)
    test_frame.to_csv(test_csv_path, index=False)

    interval_minutes = int(item.get("target_interval_minutes", 1))
    summary = {
        "benchmark_name": benchmark_name,
        "dataset_name": dataset_name,
        "num_branches": int(item["num_branches"]),
        "target_interval_minutes": interval_minutes,
        "base_interval_minutes": int(item.get("base_interval_minutes", 1)),
        "train": {
            **summarize_frame(train_frame, interval_minutes),
            "source": "base + aligned synthetic variants",
        },
        "val": {
            **summarize_frame(val_frame, interval_minutes),
            "source": "base only",
        },
        "test": {
            **summarize_frame(test_frame, interval_minutes),
            "source": "base only",
        },
        "train_csv_path": train_csv_path.name,
        "val_csv_path": val_csv_path.name,
        "test_csv_path": test_csv_path.name,
        "notes": [
            "Chronological split from the nonzero base sequence.",
            "Validation and test contain only real base observations.",
            "Training includes synthetic variants restricted to the same train timestamps.",
        ],
    }
    summary_path = bundle_dir / "split_config.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote benchmark bundle: {bundle_dir}")
    print(f"Split config: {summary_path}")


if __name__ == "__main__":
    main()
