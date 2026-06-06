import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from build_quality_filtered_entity_split_benchmark import filter_frame_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an activity-filtered benchmark from an existing split_config.json.")
    parser.add_argument("--split-config", required=True, help="Path to an existing split_config.json.")
    parser.add_argument("--output-dir", default="data_preprocessing/output/benchmarks", help="Output benchmark root.")
    parser.add_argument("--benchmark-name", required=True, help="New benchmark name.")
    parser.add_argument("--min-row-total-current", type=float, default=0.12)
    parser.add_argument("--min-row-active-branches", type=int, default=2)
    parser.add_argument("--active-branch-current-threshold", type=float, default=0.05)
    parser.add_argument("--context-rows", type=int, default=60)
    parser.add_argument("--min-segment-rows", type=int, default=120)
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def resolve_csv_path(split_config_path: Path, csv_path_value: str) -> Path:
    csv_path = Path(csv_path_value)
    candidates: List[Path] = []
    if csv_path.is_absolute():
        candidates.append(csv_path)
        candidates.append(split_config_path.parent / csv_path.name)
    else:
        candidates.append((Path.cwd() / csv_path).resolve())
        candidates.append((split_config_path.parent / csv_path).resolve())
        candidates.append((split_config_path.parent / csv_path.name).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[-1].resolve()


def summarize_frame(frame: pd.DataFrame, interval_minutes: int) -> Dict[str, object]:
    entity_names = sorted(frame["entity_id"].astype(str).unique().tolist()) if "entity_id" in frame.columns else []
    return {
        "entity_count": len(entity_names),
        "entity_names": entity_names,
        "combined_rows": int(len(frame)),
        "base_rows": int(len(frame)),
        "combined_hours": float(len(frame) * interval_minutes / 60.0),
        "base_hours": float(len(frame) * interval_minutes / 60.0),
    }


def main() -> None:
    args = parse_args()
    split_config_path = Path(args.split_config).resolve()
    split_config = read_json(split_config_path)
    interval_minutes = int(split_config.get("target_interval_minutes", 1))

    bundle_dir = Path(args.output_dir).resolve() / args.benchmark_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    reports: Dict[str, Dict[str, float]] = {}
    output_paths: Dict[str, Path] = {}
    filtered_frames: Dict[str, pd.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        source_csv = resolve_csv_path(split_config_path, str(split_config[f"{split_name}_csv_path"]))
        frame = pd.read_csv(source_csv)
        filtered, report = filter_frame_rows(
            frame,
            min_row_total_current=args.min_row_total_current,
            min_row_active_branches=args.min_row_active_branches,
            active_branch_current_threshold=args.active_branch_current_threshold,
            context_rows=args.context_rows,
            min_segment_rows=args.min_segment_rows,
        )
        if filtered.empty:
            raise ValueError(f"{split_name} split became empty after filtering.")
        filtered_frames[split_name] = filtered
        reports[split_name] = report
        out_path = bundle_dir / f"{args.benchmark_name}.{split_name}.csv"
        filtered.to_csv(out_path, index=False)
        output_paths[split_name] = out_path

    new_config = {
        **split_config,
        "benchmark_name": args.benchmark_name,
        "source_split_config": str(split_config_path),
        "train_csv_path": str(output_paths["train"].resolve()),
        "val_csv_path": str(output_paths["val"].resolve()),
        "test_csv_path": str(output_paths["test"].resolve()),
        "train": {
            **summarize_frame(filtered_frames["train"], interval_minutes),
            "train_source": split_config.get("train_source", split_config.get("train", {}).get("train_source", "combined")),
        },
        "val": {
            **summarize_frame(filtered_frames["val"], interval_minutes),
            "train_source": split_config.get("val_source", split_config.get("val", {}).get("train_source", "base")),
        },
        "test": {
            **summarize_frame(filtered_frames["test"], interval_minutes),
            "train_source": split_config.get("test_source", split_config.get("test", {}).get("train_source", "base")),
        },
        "filtering": {
            "min_row_total_current": args.min_row_total_current,
            "min_row_active_branches": args.min_row_active_branches,
            "active_branch_current_threshold": args.active_branch_current_threshold,
            "context_rows": args.context_rows,
            "min_segment_rows": args.min_segment_rows,
            "reports": reports,
        },
    }
    notes = list(split_config.get("notes", []))
    notes.append("Activity-focused filtering keeps only sustained residual-current event segments.")
    new_config["notes"] = notes
    (bundle_dir / "split_config.json").write_text(json.dumps(new_config, indent=2))
    (bundle_dir / "filter_report.json").write_text(json.dumps(reports, indent=2))
    print(json.dumps(new_config["filtering"], indent=2))


if __name__ == "__main__":
    main()
