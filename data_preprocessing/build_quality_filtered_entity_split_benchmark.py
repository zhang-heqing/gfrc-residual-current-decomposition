import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from data_preprocessing.path_utils import resolve_manifest_artifact_path
from build_entity_split_benchmark import (
    concat_frames,
    load_manifest,
    read_split_frames,
    select_datasets,
    split_entities,
    write_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a quality-filtered entity-level benchmark from the rebuilt dataset manifest."
    )
    parser.add_argument("--manifest", required=True, help="Path to the manifest.json file.")
    parser.add_argument(
        "--output-dir",
        default="data_preprocessing/output/benchmarks",
        help="Directory where the benchmark bundle will be written.",
    )
    parser.add_argument(
        "--benchmark-name",
        default="shanse001_rebuilt_entity_split_qc_v1",
        help="Name of the benchmark bundle.",
    )
    parser.add_argument("--num-branches", type=int, default=12, help="Keep only datasets with this branch count.")
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
        "--min-mean-total-current",
        type=float,
        default=1.0,
        help="Drop entities whose base CSV mean total residual current is below this threshold.",
    )
    parser.add_argument(
        "--min-mean-branch-corr",
        type=float,
        default=0.2,
        help="Drop entities whose mean branch power/current correlation is below this threshold.",
    )
    parser.add_argument(
        "--min-row-total-current",
        type=float,
        default=0.1,
        help="Drop rows whose total residual current is below this threshold.",
    )
    parser.add_argument(
        "--min-row-active-branches",
        type=int,
        default=2,
        help="Drop rows that have fewer than this number of active branches.",
    )
    parser.add_argument(
        "--active-branch-current-threshold",
        type=float,
        default=0.05,
        help="A branch is considered active when its current is at least this threshold.",
    )
    parser.add_argument(
        "--context-rows",
        type=int,
        default=60,
        help="Keep this many rows of context before and after each active row.",
    )
    parser.add_argument(
        "--min-segment-rows",
        type=int,
        default=120,
        help="Drop retained contiguous segments shorter than this length after filtering.",
    )
    return parser.parse_args()


def compute_entity_quality(item: Dict[str, object], manifest_path: Path) -> Dict[str, float]:
    base_csv_path = resolve_manifest_artifact_path(manifest_path, str(item["base_csv_path"]))
    frame = pd.read_csv(base_csv_path)
    total_current = frame["total_residual_current"].to_numpy(dtype=np.float64)
    total_power = frame["total_power"].to_numpy(dtype=np.float64)

    mean_total_current = float(total_current.mean())
    p90_total_current = float(np.quantile(total_current, 0.9))
    mean_total_delta = float(np.mean(np.abs(np.diff(total_current)))) if len(total_current) > 1 else 0.0
    total_corr = (
        float(np.corrcoef(total_current, total_power)[0, 1])
        if np.std(total_current) > 1e-6 and np.std(total_power) > 1e-6
        else 0.0
    )

    branch_corrs: List[float] = []
    branch_means: List[float] = []
    active_ratios: List[float] = []
    for index in range(1, int(item["num_branches"]) + 1):
        power = frame[f"branch_{index}_power"].to_numpy(dtype=np.float64)
        current = frame[f"branch_{index}_current"].to_numpy(dtype=np.float64)
        branch_means.append(float(np.mean(np.abs(current))))
        active_ratios.append(float((np.abs(current) >= 0.05).mean()))
        if np.std(power) > 1e-6 and np.std(current) > 1e-6:
            branch_corrs.append(float(np.corrcoef(power, current)[0, 1]))

    return {
        "mean_total_current": mean_total_current,
        "p90_total_current": p90_total_current,
        "mean_total_delta": mean_total_delta,
        "total_power_current_corr": total_corr,
        "mean_branch_power_current_corr": float(np.mean(branch_corrs)) if branch_corrs else 0.0,
        "median_branch_abs_current": float(np.median(branch_means)) if branch_means else 0.0,
        "median_branch_active_ratio": float(np.median(active_ratios)) if active_ratios else 0.0,
        "base_rows": int(len(frame)),
    }


def filter_entities(
    items: Sequence[Dict[str, object]],
    *,
    manifest_path: Path,
    min_mean_total_current: float,
    min_mean_branch_corr: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    kept: List[Dict[str, object]] = []
    dropped: List[Dict[str, object]] = []
    for item in items:
        quality = compute_entity_quality(item, manifest_path)
        enriched = {**item, "quality": quality}
        if (
            quality["mean_total_current"] >= min_mean_total_current
            and quality["mean_branch_power_current_corr"] >= min_mean_branch_corr
        ):
            kept.append(enriched)
        else:
            dropped.append(enriched)
    return kept, dropped


def filter_frame_rows(
    frame: pd.DataFrame,
    *,
    min_row_total_current: float,
    min_row_active_branches: int,
    active_branch_current_threshold: float,
    context_rows: int,
    min_segment_rows: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    current_cols = [name for name in frame.columns if name.startswith("branch_") and name.endswith("_current")]
    branch_currents = frame[current_cols].abs()
    active_branch_count = (branch_currents >= active_branch_current_threshold).sum(axis=1)
    active_mask = (
        frame["total_residual_current"].abs() >= min_row_total_current
    ) & (active_branch_count >= min_row_active_branches)

    keep_mask = active_mask.to_numpy(copy=True)
    if context_rows > 0 and keep_mask.any():
        active_positions = np.flatnonzero(keep_mask)
        expanded = np.zeros_like(keep_mask, dtype=bool)
        for position in active_positions.tolist():
            start = max(0, position - context_rows)
            stop = min(len(expanded), position + context_rows + 1)
            expanded[start:stop] = True
        keep_mask = expanded

    filtered = frame.loc[keep_mask].copy()
    if filtered.empty:
        report = {
            "rows_before": int(len(frame)),
            "rows_after": 0,
            "rows_dropped": int(len(frame)),
            "row_keep_ratio": 0.0,
            "active_rows_before_context": int(active_mask.sum()),
            "segments_after_filter": 0,
        }
        return filtered, report

    filtered["_orig_index"] = filtered.index.to_numpy()
    if "segment_id" in filtered.columns:
        segment_values = filtered["segment_id"].astype(str)
    else:
        segment_values = pd.Series(["0"] * len(filtered), index=filtered.index)
    index_gap = filtered["_orig_index"].diff().fillna(1).ne(1)
    segment_gap = segment_values.ne(segment_values.shift(1)).fillna(True)
    filtered["segment_id"] = (index_gap | segment_gap).cumsum().astype(int)

    if min_segment_rows > 1:
        segment_lengths = filtered.groupby("segment_id").size()
        keep_segments = segment_lengths[segment_lengths >= min_segment_rows].index
        filtered = filtered[filtered["segment_id"].isin(keep_segments)].copy()
        if not filtered.empty:
            filtered["segment_id"] = filtered.groupby("segment_id").ngroup().astype(int)

    filtered = filtered.drop(columns=["_orig_index"])

    report = {
        "rows_before": int(len(frame)),
        "rows_after": int(len(filtered)),
        "rows_dropped": int(len(frame) - len(filtered)),
        "row_keep_ratio": float(len(filtered) / max(len(frame), 1)),
        "active_rows_before_context": int(active_mask.sum()),
        "segments_after_filter": int(filtered["segment_id"].nunique()) if not filtered.empty else 0,
    }
    return filtered.reset_index(drop=True), report


def build_quality_report(
    kept_items: Sequence[Dict[str, object]],
    dropped_items: Sequence[Dict[str, object]],
    *,
    row_reports: Dict[str, Dict[str, float]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    return {
        "benchmark_name": args.benchmark_name,
        "quality_filter": {
            "min_mean_total_current": args.min_mean_total_current,
            "min_mean_branch_corr": args.min_mean_branch_corr,
            "min_row_total_current": args.min_row_total_current,
            "min_row_active_branches": args.min_row_active_branches,
            "active_branch_current_threshold": args.active_branch_current_threshold,
            "context_rows": args.context_rows,
            "min_segment_rows": args.min_segment_rows,
        },
        "kept_entities": [
            {
                "dataset_name": item["dataset_name"],
                "quality": item["quality"],
                "row_filter_report": row_reports.get(str(item["dataset_name"]), {}),
            }
            for item in kept_items
        ],
        "dropped_entities": [
            {
                "dataset_name": item["dataset_name"],
                "quality": item["quality"],
            }
            for item in dropped_items
        ],
    }


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_manifest(manifest_path)
    selected_items = select_datasets(items, args.num_branches)
    kept_items, dropped_items = filter_entities(
        selected_items,
        manifest_path=manifest_path,
        min_mean_total_current=args.min_mean_total_current,
        min_mean_branch_corr=args.min_mean_branch_corr,
    )
    if len(kept_items) < 3:
        raise ValueError(
            f"Only {len(kept_items)} entities remain after quality filtering; need at least 3."
        )

    train_items, val_items, test_items = split_entities(
        kept_items,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    row_reports: Dict[str, Dict[str, float]] = {}

    def prepare_frames(split_items: Sequence[Dict[str, object]], *, source: str) -> List[pd.DataFrame]:
        frames = read_split_frames(split_items, source=source, manifest_path=manifest_path)
        cleaned_frames: List[pd.DataFrame] = []
        for item, frame in zip(split_items, frames):
            cleaned, report = filter_frame_rows(
                frame,
                min_row_total_current=args.min_row_total_current,
                min_row_active_branches=args.min_row_active_branches,
                active_branch_current_threshold=args.active_branch_current_threshold,
                context_rows=args.context_rows,
                min_segment_rows=args.min_segment_rows,
            )
            if len(cleaned) == 0:
                raise ValueError(f"All rows were filtered out for {item['dataset_name']}.")
            row_reports[str(item["dataset_name"])] = report
            cleaned_frames.append(cleaned)
        return cleaned_frames

    train_frame = concat_frames(prepare_frames(train_items, source=args.train_source))
    val_frame = concat_frames(prepare_frames(val_items, source=args.val_source))
    test_frame = concat_frames(prepare_frames(test_items, source=args.test_source))

    write_bundle(
        output_dir=output_dir,
        benchmark_name=args.benchmark_name,
        train_frame=train_frame,
        val_frame=val_frame,
        test_frame=test_frame,
        selected_items=kept_items,
        train_items=train_items,
        val_items=val_items,
        test_items=test_items,
        train_source=args.train_source,
        val_source=args.val_source,
        test_source=args.test_source,
        seed=args.seed,
        dropped_zero_signal=[str(item["dataset_name"]) for item in dropped_items],
    )

    bundle_dir = output_dir / args.benchmark_name
    quality_report = build_quality_report(kept_items, dropped_items, row_reports=row_reports, args=args)
    (bundle_dir / "quality_report.json").write_text(json.dumps(quality_report, indent=2))
    print(f"Quality report: {bundle_dir / 'quality_report.json'}")


if __name__ == "__main__":
    main()
