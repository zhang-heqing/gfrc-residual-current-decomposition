import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
PHASE_NAMES = ("A", "B", "C")
POWER_KEYSETS = {
    "active": ("Pa", "Pb", "Pc"),
    "fundamental": ("Pba", "Pbb", "Pbc"),
}


@dataclass
class BranchRecord:
    timestamp: datetime
    residual_current: float
    phase_powers: Tuple[float, float, float]


@dataclass
class BranchSeries:
    source_id: str
    records: List[BranchRecord]
    dominant_phase: str
    dominant_share: float
    phase_shares: Tuple[float, float, float]
    projection_mode: str
    projected_power: List[Optional[float]]
    residual_current: List[Optional[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw WeChat-exported three-phase branch folders into model-ready single-phase CSV datasets."
    )
    parser.add_argument(
        "--input-root",
        action="append",
        required=True,
        help="Root directory containing one folder per branch/device. Can be provided multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default="data_preprocessing/output",
        help="Directory where converted CSV datasets and metadata will be written.",
    )
    parser.add_argument(
        "--grouping",
        choices=["all", "prefix", "chunks"],
        default="all",
        help="`all` treats each input root as one branch panel; `prefix` groups subfolders by the numeric prefix before '-'; `chunks` builds fixed-size branch panels from valid directories.",
    )
    parser.add_argument(
        "--dataset-label",
        action="append",
        help="Optional label for each input root when grouping=all. Must match the number of --input-root arguments if provided.",
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=10,
        help="Fixed resampling interval in minutes.",
    )
    parser.add_argument(
        "--tolerance-seconds",
        type=int,
        default=180,
        help="Maximum distance from a bucket center when snapping a raw record to the fixed grid.",
    )
    parser.add_argument(
        "--max-gap-steps",
        type=int,
        default=2,
        help="Linearly interpolate missing runs of at most this many grid steps.",
    )
    parser.add_argument(
        "--phase-threshold",
        type=float,
        default=0.55,
        help="If one phase exceeds this long-run share, keep that phase directly; otherwise build a weighted single-phase equivalent.",
    )
    parser.add_argument(
        "--power-mode",
        choices=sorted(POWER_KEYSETS),
        default="active",
        help="Which power fields to use for the single-phase projection.",
    )
    parser.add_argument(
        "--residual-key",
        choices=["Inb", "In"],
        default="Inb",
        help="Residual-current field to use. Inb is the default because it is closer to the paper's fundamental residual current target.",
    )
    parser.add_argument(
        "--min-branches",
        type=int,
        default=2,
        help="Skip groups with fewer branches than this.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=12,
        help="When grouping=chunks, build fixed-size branch panels with this many valid branches.",
    )
    parser.add_argument(
        "--schema-filter",
        choices=["any", "full_residual_power"],
        default="any",
        help="Filter raw branch directories by schema before grouping. `full_residual_power` keeps only files that contain the requested residual key and all requested power keys.",
    )
    parser.add_argument(
        "--require-nonzero-residual",
        action="store_true",
        help="Keep only branch directories that contain at least one nonzero residual-current value for the selected residual key.",
    )
    parser.add_argument(
        "--min-duration-hours",
        type=float,
        default=0.0,
        help="Keep only branch directories whose record span is at least this many hours.",
    )
    return parser.parse_args()


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def sanitize_name(name: str) -> str:
    allowed = []
    for char in name:
        if char.isalnum() or char in {"-", "_"}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "dataset"


def branch_sort_key(path: Path) -> Tuple[int, int, str]:
    left, _, right = path.name.partition("-")
    try:
        left_value = int(left)
    except ValueError:
        left_value = 10**9
    try:
        right_value = int(right)
    except ValueError:
        right_value = 10**9
    return left_value, right_value, path.name


def load_json_records(branch_dir: Path, residual_key: str, power_keys: Sequence[str]) -> List[BranchRecord]:
    payload = json.loads((branch_dir / "instance_datas_dict.json").read_text())
    top_key = next(iter(payload))
    raw_records = payload[top_key]
    records: List[BranchRecord] = []
    for item in raw_records:
        data = item.get("data", {})
        timestamp = datetime.strptime(item["createTime"], DATETIME_FORMAT)
        residual_current = safe_float(data.get(residual_key, {}).get("value"))
        phase_powers = tuple(safe_float(data.get(key, {}).get("value")) for key in power_keys)
        records.append(
            BranchRecord(
                timestamp=timestamp,
                residual_current=residual_current,
                phase_powers=phase_powers,
            )
        )
    records.sort(key=lambda record: record.timestamp)
    return records


def branch_dir_matches_schema(
    branch_dir: Path,
    residual_key: str,
    power_keys: Sequence[str],
    schema_filter: str,
    require_nonzero_residual: bool,
    min_duration_hours: float,
) -> bool:
    payload = json.loads((branch_dir / "instance_datas_dict.json").read_text())
    top_key = next(iter(payload))
    raw_records = payload[top_key]
    if not raw_records:
        return False

    if min_duration_hours > 0.0:
        start = datetime.strptime(raw_records[0]["createTime"], DATETIME_FORMAT)
        end = datetime.strptime(raw_records[-1]["createTime"], DATETIME_FORMAT)
        duration_hours = (end - start).total_seconds() / 3600.0
        if duration_hours < min_duration_hours:
            return False

    first_data = raw_records[0].get("data", {})
    keys = set(first_data.keys())
    if schema_filter == "full_residual_power":
        if residual_key not in keys or not set(power_keys).issubset(keys):
            return False

    if require_nonzero_residual:
        for item in raw_records:
            data = item.get("data", {})
            value = safe_float(data.get(residual_key, {}).get("value"))
            if value != 0.0:
                return True
        return False

    return True


def infer_phase_projection(
    records: Sequence[BranchRecord],
    phase_threshold: float,
) -> Tuple[str, float, Tuple[float, float, float], str]:
    totals = [0.0, 0.0, 0.0]
    for record in records:
        for index, value in enumerate(record.phase_powers):
            totals[index] += abs(value)
    denominator = sum(totals)
    if denominator <= 0.0:
        shares = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    else:
        shares = tuple(value / denominator for value in totals)
    dominant_index = max(range(3), key=lambda idx: shares[idx])
    dominant_share = shares[dominant_index]
    projection_mode = "dominant_phase" if dominant_share >= phase_threshold else "weighted_equivalent"
    return PHASE_NAMES[dominant_index], dominant_share, shares, projection_mode


def ceil_to_grid(timestamp: datetime, interval_minutes: int) -> datetime:
    bucket = timestamp.replace(second=0, microsecond=0)
    minute = (bucket.minute // interval_minutes) * interval_minutes
    bucket = bucket.replace(minute=minute)
    if bucket < timestamp:
        bucket += timedelta(minutes=interval_minutes)
    return bucket


def floor_to_grid(timestamp: datetime, interval_minutes: int) -> datetime:
    bucket = timestamp.replace(second=0, microsecond=0)
    minute = (bucket.minute // interval_minutes) * interval_minutes
    return bucket.replace(minute=minute)


def build_grid(start: datetime, end: datetime, interval_minutes: int) -> List[datetime]:
    current = ceil_to_grid(start, interval_minutes)
    stop = floor_to_grid(end, interval_minutes)
    grid: List[datetime] = []
    while current <= stop:
        grid.append(current)
        current += timedelta(minutes=interval_minutes)
    return grid


def median_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(median(values))


def snap_records_to_grid(
    records: Sequence[BranchRecord],
    grid: Sequence[datetime],
    interval_minutes: int,
    tolerance_seconds: int,
) -> Tuple[List[Optional[float]], List[List[Optional[float]]]]:
    step_seconds = interval_minutes * 60
    if not grid:
        return [], [[], [], []]

    buckets: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: {"residual": [], "phase_0": [], "phase_1": [], "phase_2": []})
    origin = grid[0]
    for record in records:
        offset = (record.timestamp - origin).total_seconds()
        slot_index = int(round(offset / step_seconds))
        if slot_index < 0 or slot_index >= len(grid):
            continue
        slot_time = grid[slot_index]
        if abs((record.timestamp - slot_time).total_seconds()) > tolerance_seconds:
            continue
        bucket = buckets[slot_index]
        bucket["residual"].append(record.residual_current)
        for phase_index, phase_value in enumerate(record.phase_powers):
            bucket[f"phase_{phase_index}"].append(phase_value)

    residual_series: List[Optional[float]] = []
    phase_series: List[List[Optional[float]]] = [[], [], []]
    for index in range(len(grid)):
        bucket = buckets.get(index)
        if not bucket:
            residual_series.append(None)
            for phase_index in range(3):
                phase_series[phase_index].append(None)
            continue
        residual_series.append(median_or_none(bucket["residual"]))
        for phase_index in range(3):
            phase_series[phase_index].append(median_or_none(bucket[f"phase_{phase_index}"]))
    return residual_series, phase_series


def interpolate_short_gaps(values: Sequence[Optional[float]], max_gap_steps: int) -> List[Optional[float]]:
    filled = list(values)
    index = 0
    while index < len(filled):
        if filled[index] is not None:
            index += 1
            continue
        start = index
        while index < len(filled) and filled[index] is None:
            index += 1
        end = index
        gap_size = end - start
        left = filled[start - 1] if start > 0 else None
        right = filled[end] if end < len(filled) else None
        if gap_size <= max_gap_steps and left is not None and right is not None:
            step = (right - left) / (gap_size + 1)
            for offset in range(gap_size):
                filled[start + offset] = left + step * (offset + 1)
    return filled


def build_branch_series(
    branch_dir: Path,
    grid: Sequence[datetime],
    interval_minutes: int,
    tolerance_seconds: int,
    max_gap_steps: int,
    phase_threshold: float,
    residual_key: str,
    power_keys: Sequence[str],
) -> BranchSeries:
    records = load_json_records(branch_dir, residual_key=residual_key, power_keys=power_keys)
    dominant_phase, dominant_share, phase_shares, projection_mode = infer_phase_projection(
        records=records,
        phase_threshold=phase_threshold,
    )
    residual_series, phase_series = snap_records_to_grid(
        records=records,
        grid=grid,
        interval_minutes=interval_minutes,
        tolerance_seconds=tolerance_seconds,
    )
    residual_series = interpolate_short_gaps(residual_series, max_gap_steps=max_gap_steps)
    phase_series = [interpolate_short_gaps(series, max_gap_steps=max_gap_steps) for series in phase_series]

    dominant_index = PHASE_NAMES.index(dominant_phase)
    projected_power: List[Optional[float]] = []
    for slot_index in range(len(grid)):
        phase_values = [phase_series[0][slot_index], phase_series[1][slot_index], phase_series[2][slot_index]]
        if any(value is None for value in phase_values):
            projected_power.append(None)
            continue
        if projection_mode == "dominant_phase":
            projected_power.append(phase_values[dominant_index])
        else:
            projected_power.append(
                phase_shares[0] * phase_values[0]
                + phase_shares[1] * phase_values[1]
                + phase_shares[2] * phase_values[2]
            )

    return BranchSeries(
        source_id=branch_dir.name,
        records=records,
        dominant_phase=dominant_phase,
        dominant_share=dominant_share,
        phase_shares=phase_shares,
        projection_mode=projection_mode,
        projected_power=projected_power,
        residual_current=residual_series,
    )


def count_missing(values: Sequence[Optional[float]]) -> int:
    return sum(value is None for value in values)


def write_dataset(output_csv: Path, grid: Sequence[datetime], branches: Sequence[BranchSeries]) -> Tuple[int, int]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = ["timestamp", "total_residual_current", "total_power"]
    for index in range(len(branches)):
        headers.append(f"branch_{index + 1}_power")
        headers.append(f"branch_{index + 1}_current")

    total_rows = len(grid)
    kept_rows = 0
    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for slot_index, timestamp in enumerate(grid):
            branch_values: List[float] = []
            total_residual_current = 0.0
            total_power = 0.0
            missing = False
            for branch in branches:
                branch_power = branch.projected_power[slot_index]
                branch_current = branch.residual_current[slot_index]
                if branch_power is None or branch_current is None:
                    missing = True
                    break
                total_power += branch_power
                total_residual_current += branch_current
                branch_values.extend([branch_power, branch_current])
            if missing:
                continue
            row = [timestamp.strftime(DATETIME_FORMAT), f"{total_residual_current:.6f}", f"{total_power:.6f}"]
            row.extend(f"{value:.6f}" for value in branch_values)
            writer.writerow(row)
            kept_rows += 1
    return total_rows, kept_rows


def build_group_definitions(
    roots: Sequence[Path],
    grouping: str,
    labels: Optional[Sequence[str]],
    min_branches: int,
    chunk_size: int,
    schema_filter: str,
    residual_key: str,
    power_mode: str,
    require_nonzero_residual: bool,
    min_duration_hours: float,
) -> List[Tuple[str, Path, List[Path]]]:
    definitions: List[Tuple[str, Path, List[Path]]] = []
    power_keys = POWER_KEYSETS[power_mode]

    def valid_branch_dirs(root: Path) -> List[Path]:
        candidates = sorted([path for path in root.iterdir() if path.is_dir()], key=branch_sort_key)
        return [
            path
            for path in candidates
            if branch_dir_matches_schema(
                path,
                residual_key=residual_key,
                power_keys=power_keys,
                schema_filter=schema_filter,
                require_nonzero_residual=require_nonzero_residual,
                min_duration_hours=min_duration_hours,
            )
        ]

    if grouping == "all":
        if labels and len(labels) != len(roots):
            raise ValueError("When --dataset-label is provided, it must match the number of --input-root arguments.")
        for index, root in enumerate(roots):
            branch_dirs = valid_branch_dirs(root)
            if len(branch_dirs) < min_branches:
                continue
            label = labels[index] if labels else root.name
            definitions.append((sanitize_name(label), root, branch_dirs))
        return definitions

    if grouping == "chunks":
        if chunk_size < min_branches:
            raise ValueError("--chunk-size must be greater than or equal to --min-branches.")
        if labels and len(labels) != len(roots):
            raise ValueError("When --dataset-label is provided, it must match the number of --input-root arguments.")
        for index, root in enumerate(roots):
            branch_dirs = valid_branch_dirs(root)
            label = labels[index] if labels else root.name
            chunk_index = 1
            for start in range(0, len(branch_dirs), chunk_size):
                chunk = branch_dirs[start : start + chunk_size]
                if len(chunk) < chunk_size:
                    continue
                dataset_name = sanitize_name(f"{label}_chunk_{chunk_index:02d}")
                definitions.append((dataset_name, root, chunk))
                chunk_index += 1
        return definitions

    for root in roots:
        grouped: Dict[str, List[Path]] = defaultdict(list)
        for branch_dir in valid_branch_dirs(root):
            prefix = branch_dir.name.split("-", 1)[0]
            grouped[prefix].append(branch_dir)
        for prefix, branch_dirs in sorted(grouped.items(), key=lambda item: int(item[0]) if item[0].isdigit() else 10**9):
            if len(branch_dirs) < min_branches:
                continue
            definitions.append((sanitize_name(f"{root.name}_{prefix}"), root, branch_dirs))
    return definitions


def write_metadata(
    metadata_path: Path,
    input_root: Path,
    dataset_name: str,
    branches: Sequence[BranchSeries],
    grid: Sequence[datetime],
    total_rows: int,
    kept_rows: int,
    interval_minutes: int,
    tolerance_seconds: int,
    max_gap_steps: int,
    phase_threshold: float,
    power_mode: str,
    residual_key: str,
) -> None:
    metadata = {
        "dataset_name": dataset_name,
        "input_root": str(input_root),
        "interval_minutes": interval_minutes,
        "tolerance_seconds": tolerance_seconds,
        "max_gap_steps": max_gap_steps,
        "phase_threshold": phase_threshold,
        "power_mode": power_mode,
        "residual_key": residual_key,
        "grid_rows_total": total_rows,
        "dataset_rows_kept": kept_rows,
        "grid_start": grid[0].strftime(DATETIME_FORMAT) if grid else None,
        "grid_end": grid[-1].strftime(DATETIME_FORMAT) if grid else None,
        "num_branches": len(branches),
        "branches": [],
    }
    for branch_index, branch in enumerate(branches, start=1):
        metadata["branches"].append(
            {
                "branch_index": branch_index,
                "source_id": branch.source_id,
                "records_loaded": len(branch.records),
                "dominant_phase": branch.dominant_phase,
                "dominant_share": round(branch.dominant_share, 6),
                "phase_shares": {
                    phase_name: round(share, 6)
                    for phase_name, share in zip(PHASE_NAMES, branch.phase_shares)
                },
                "projection_mode": branch.projection_mode,
                "missing_projected_power_points": count_missing(branch.projected_power),
                "missing_residual_points": count_missing(branch.residual_current),
            }
        )
    metadata_path.write_text(json.dumps(metadata, indent=2))


def process_group(
    dataset_name: str,
    root: Path,
    branch_dirs: Sequence[Path],
    output_dir: Path,
    interval_minutes: int,
    tolerance_seconds: int,
    max_gap_steps: int,
    phase_threshold: float,
    residual_key: str,
    power_mode: str,
) -> Dict[str, object]:
    power_keys = POWER_KEYSETS[power_mode]
    branch_records: Dict[str, List[BranchRecord]] = {}
    boundaries: List[Tuple[datetime, datetime]] = []
    for branch_dir in branch_dirs:
        records = load_json_records(branch_dir, residual_key=residual_key, power_keys=power_keys)
        if not records:
            continue
        branch_records[branch_dir.name] = records
        boundaries.append((records[0].timestamp, records[-1].timestamp))
    if len(branch_records) < 2:
        raise ValueError(f"{dataset_name}: fewer than two branches contained records.")

    common_start = max(start for start, _ in boundaries)
    common_end = min(end for _, end in boundaries)
    grid = build_grid(common_start, common_end, interval_minutes=interval_minutes)
    if not grid:
        raise ValueError(f"{dataset_name}: overlapping time range is empty after snapping to the grid.")

    branches = []
    for branch_dir in branch_dirs:
        if branch_dir.name not in branch_records:
            continue
        branches.append(
            build_branch_series(
                branch_dir=branch_dir,
                grid=grid,
                interval_minutes=interval_minutes,
                tolerance_seconds=tolerance_seconds,
                max_gap_steps=max_gap_steps,
                phase_threshold=phase_threshold,
                residual_key=residual_key,
                power_keys=power_keys,
            )
        )

    dataset_dir = output_dir / dataset_name
    csv_path = dataset_dir / f"{dataset_name}.csv"
    metadata_path = dataset_dir / f"{dataset_name}.metadata.json"
    total_rows, kept_rows = write_dataset(csv_path, grid=grid, branches=branches)
    write_metadata(
        metadata_path=metadata_path,
        input_root=root,
        dataset_name=dataset_name,
        branches=branches,
        grid=grid,
        total_rows=total_rows,
        kept_rows=kept_rows,
        interval_minutes=interval_minutes,
        tolerance_seconds=tolerance_seconds,
        max_gap_steps=max_gap_steps,
        phase_threshold=phase_threshold,
        power_mode=power_mode,
        residual_key=residual_key,
    )

    return {
        "dataset_name": dataset_name,
        "input_root": str(root),
        "csv_path": str(csv_path),
        "metadata_path": str(metadata_path),
        "num_branches": len(branches),
        "grid_rows_total": total_rows,
        "dataset_rows_kept": kept_rows,
        "grid_start": grid[0].strftime(DATETIME_FORMAT),
        "grid_end": grid[-1].strftime(DATETIME_FORMAT),
    }


def write_manifest(output_dir: Path, results: Sequence[Dict[str, object]]) -> Path:
    manifest_path = output_dir / "manifest.json"
    existing: Dict[str, Dict[str, object]] = {}
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text())
        for item in payload.get("datasets", []):
            name = item.get("dataset_name")
            if isinstance(name, str):
                existing[name] = item
    for item in results:
        existing[str(item["dataset_name"])] = item
    payload = {
        "generated_at": datetime.now().strftime(DATETIME_FORMAT),
        "datasets": [existing[key] for key in sorted(existing)],
    }
    manifest_path.write_text(json.dumps(payload, indent=2))
    return manifest_path


def main() -> None:
    args = parse_args()
    roots = [Path(path) for path in args.input_root]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    group_definitions = build_group_definitions(
        roots=roots,
        grouping=args.grouping,
        labels=args.dataset_label,
        min_branches=args.min_branches,
        chunk_size=args.chunk_size,
        schema_filter=args.schema_filter,
        residual_key=args.residual_key,
        power_mode=args.power_mode,
        require_nonzero_residual=args.require_nonzero_residual,
        min_duration_hours=args.min_duration_hours,
    )
    if not group_definitions:
        raise ValueError("No branch groups matched the requested settings.")

    results = []
    for dataset_name, root, branch_dirs in group_definitions:
        result = process_group(
            dataset_name=dataset_name,
            root=root,
            branch_dirs=branch_dirs,
            output_dir=output_dir,
            interval_minutes=args.interval_minutes,
            tolerance_seconds=args.tolerance_seconds,
            max_gap_steps=args.max_gap_steps,
            phase_threshold=args.phase_threshold,
            residual_key=args.residual_key,
            power_mode=args.power_mode,
        )
        results.append(result)
        print(
            f"{dataset_name}: branches={result['num_branches']}, "
            f"rows={result['dataset_rows_kept']}/{result['grid_rows_total']}, "
            f"range={result['grid_start']} -> {result['grid_end']}"
        )

    manifest_path = write_manifest(output_dir=output_dir, results=results)
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
