#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

from data_preprocessing.path_utils import resolve_split_artifact_path


DEFAULT_SPLIT_CONFIG = Path("data_preprocessing/output/benchmarks/shanse001_rebuilt_entity_split/split_config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize benchmark split statistics.")
    parser.add_argument(
        "--split-config",
        action="append",
        default=None,
        help="Path to a split_config.json file. Can be provided multiple times.",
    )
    parser.add_argument("--output-csv", default="experiments/output/dataset_stats.csv")
    parser.add_argument("--output-json", default="experiments/output/dataset_stats.json")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, object]:
    return json.loads(Path(path).read_text())


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_frame(rows: List[Dict[str, str]]) -> Dict[str, object]:
    row_count = len(rows)
    entities = len({row.get("entity_id", "") for row in rows if row.get("entity_id")})
    branch_current_cols = [key for key in rows[0].keys() if key.startswith("branch_") and key.endswith("_current")] if rows else []

    summary: Dict[str, object] = {
        "rows": int(row_count),
        "entities": int(entities),
        "branches": len(branch_current_cols),
        "has_segment_id": int("segment_id" in rows[0]) if rows else 0,
        "has_synthetic_variant": int("synthetic_variant" in rows[0]) if rows else 0,
    }

    if rows and "timestamp" in rows[0] and len(rows) > 1:
        timestamps = [row["timestamp"] for row in rows]
        parsed = [parse_timestamp(value) for value in timestamps if value]
        parsed = [value for value in parsed if value is not None]
        parsed.sort()
        if len(parsed) > 1:
            diffs = [
                (parsed[idx] - parsed[idx - 1]).total_seconds() / 60.0
                for idx in range(1, len(parsed))
            ]
            diffs.sort()
            mid = len(diffs) // 2
            if len(diffs) % 2 == 1:
                interval_minutes = float(diffs[mid])
            else:
                interval_minutes = float((diffs[mid - 1] + diffs[mid]) / 2.0)
            duration_hours = float((parsed[-1] - parsed[0]).total_seconds() / 3600.0) + (
                interval_minutes / 60.0
            )
        else:
            interval_minutes = 0.0
            duration_hours = 0.0
        summary.update(
            {
                "interval_minutes_inferred": interval_minutes,
                "duration_hours_inferred": duration_hours,
            }
        )
    else:
        summary.update(
            {
                "interval_minutes_inferred": None,
                "duration_hours_inferred": None,
            }
        )

    if rows and "synthetic_variant" in rows[0]:
        synthetic_rows = sum(1 for row in rows if row.get("synthetic_variant", "0") not in {"0", "", "false", "False"})
        summary["synthetic_rows"] = synthetic_rows
        summary["synthetic_ratio"] = float(synthetic_rows / max(len(rows), 1))
    else:
        summary["synthetic_rows"] = 0
        summary["synthetic_ratio"] = 0.0

    if rows and "segment_id" in rows[0]:
        summary["segments"] = int(len({row.get("segment_id", "") for row in rows if row.get("segment_id")}))
    else:
        summary["segments"] = 0

    return summary


def build_rows(split_configs: Iterable[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for config_path in split_configs:
        split_config = load_config(config_path)
        split_config_path = Path(config_path).resolve()
        benchmark_name = str(split_config.get("benchmark_name", Path(config_path).stem))
        for split_name in ("train", "val", "test"):
            split_info = split_config[split_name]
            csv_path = resolve_split_artifact_path(split_config_path, str(split_config[f"{split_name}_csv_path"]))
            frame_rows = read_csv_rows(csv_path)
            stats = summarize_frame(frame_rows)
            rows.append(
                {
                    "benchmark_name": benchmark_name,
                    "split": split_name,
                    "csv_path": str(csv_path),
                    "train_source": split_info.get("train_source"),
                    "entity_count": int(split_info.get("entity_count", 0)),
                    "entity_names": ";".join(split_info.get("entity_names", [])),
                    "combined_rows": int(split_info.get("combined_rows", 0)),
                    "base_rows": int(split_info.get("base_rows", 0)),
                    "combined_hours": float(split_info.get("combined_hours", 0.0)),
                    "base_hours": float(split_info.get("base_hours", 0.0)),
                    **stats,
                }
            )
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    split_configs = args.split_config or [str(DEFAULT_SPLIT_CONFIG)]
    rows = build_rows(split_configs)
    write_csv(Path(args.output_csv), rows)
    write_json(Path(args.output_json), rows)
    for row in rows:
        print(
            f"{row['benchmark_name']}[{row['split']}]: "
            f"rows={row['rows']}, entities={row['entities']}, branches={row['branches']}, "
            f"synthetic_ratio={row['synthetic_ratio']:.3f}"
        )


def parse_timestamp(value: str):
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            from datetime import datetime

            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    try:
        from datetime import datetime

        return datetime.fromisoformat(value)
    except ValueError:
        return None


if __name__ == "__main__":
    main()
