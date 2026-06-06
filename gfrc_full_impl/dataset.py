import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def _sorted_branch_columns(columns: Sequence[str], suffix: str) -> List[str]:
    pattern = re.compile(rf"^branch_(\d+)_{suffix}$")
    matched: List[Tuple[int, str]] = []
    for name in columns:
        result = pattern.match(name)
        if result:
            matched.append((int(result.group(1)), name))
    matched.sort(key=lambda item: item[0])
    return [name for _, name in matched]


@dataclass
class NormalizationStats:
    total_current_mean: float
    total_current_std: float
    total_power_mean: float
    total_power_std: float
    branch_power_mean: np.ndarray
    branch_power_std: np.ndarray
    branch_current_mean: np.ndarray
    branch_current_std: np.ndarray

    def normalize_total_current(self, value: np.ndarray) -> np.ndarray:
        return ((value - self.total_current_mean) / self.total_current_std).astype(np.float32)

    def normalize_total_power(self, value: np.ndarray) -> np.ndarray:
        return ((value - self.total_power_mean) / self.total_power_std).astype(np.float32)

    def normalize_branch_powers(self, value: np.ndarray) -> np.ndarray:
        return ((value - self.branch_power_mean[:, None]) / self.branch_power_std[:, None]).astype(np.float32)

    def normalize_branch_currents(self, value: np.ndarray) -> np.ndarray:
        return ((value - self.branch_current_mean[:, None]) / self.branch_current_std[:, None]).astype(np.float32)

    def denormalize_branch_currents(self, value: np.ndarray) -> np.ndarray:
        return (value * self.branch_current_std[:, None] + self.branch_current_mean[:, None]).astype(np.float32)

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_current_mean": self.total_current_mean,
            "total_current_std": self.total_current_std,
            "total_power_mean": self.total_power_mean,
            "total_power_std": self.total_power_std,
            "branch_power_mean": self.branch_power_mean.tolist(),
            "branch_power_std": self.branch_power_std.tolist(),
            "branch_current_mean": self.branch_current_mean.tolist(),
            "branch_current_std": self.branch_current_std.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "NormalizationStats":
        return cls(
            total_current_mean=float(payload["total_current_mean"]),
            total_current_std=float(payload["total_current_std"]),
            total_power_mean=float(payload["total_power_mean"]),
            total_power_std=float(payload["total_power_std"]),
            branch_power_mean=np.asarray(payload["branch_power_mean"], dtype=np.float32),
            branch_power_std=np.asarray(payload["branch_power_std"], dtype=np.float32),
            branch_current_mean=np.asarray(payload["branch_current_mean"], dtype=np.float32),
            branch_current_std=np.asarray(payload["branch_current_std"], dtype=np.float32),
        )


def _safe_std(array: np.ndarray) -> np.ndarray:
    std = array.std(axis=0)
    return np.where(std < 1e-6, 1.0, std)


class ResidualCurrentWindowDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        seq_len: int,
        start_indices: Sequence[int],
        stats: NormalizationStats,
        augment: bool = False,
        noise_std: float = 0.01,
        scale_range: Tuple[float, float] = (0.95, 1.05),
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.seq_len = seq_len
        self.start_indices = list(start_indices)
        self.stats = stats
        self.augment = augment
        self.noise_std = noise_std
        self.scale_range = scale_range

        self.total_current_col = "total_residual_current"
        self.total_power_col = "total_power"
        self.branch_power_cols = _sorted_branch_columns(self.dataframe.columns, "power")
        self.branch_current_cols = _sorted_branch_columns(self.dataframe.columns, "current")

        if not self.branch_power_cols or len(self.branch_power_cols) != len(self.branch_current_cols):
            raise ValueError("Could not infer matched branch power/current columns from the CSV file.")

    def __len__(self) -> int:
        return len(self.start_indices)

    def _apply_augmentation(
        self,
        x_r: np.ndarray,
        x_p: np.ndarray,
        branch_powers: np.ndarray,
        branch_currents: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.augment:
            return x_r, x_p, branch_powers, branch_currents

        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        x_r = x_r * scale + np.random.normal(0.0, self.noise_std, size=x_r.shape)
        x_p = x_p * scale
        branch_powers = branch_powers * scale
        branch_currents = branch_currents * scale
        return x_r, x_p, branch_powers, branch_currents

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        start = self.start_indices[index]
        stop = start + self.seq_len
        window = self.dataframe.iloc[start:stop]

        x_r = window[self.total_current_col].to_numpy(dtype=np.float32)
        x_p = window[self.total_power_col].to_numpy(dtype=np.float32)
        branch_powers = window[self.branch_power_cols].to_numpy(dtype=np.float32).T
        branch_currents = window[self.branch_current_cols].to_numpy(dtype=np.float32).T

        x_r, x_p, branch_powers, branch_currents = self._apply_augmentation(
            x_r, x_p, branch_powers, branch_currents
        )

        x_r = self.stats.normalize_total_current(x_r)
        x_p = self.stats.normalize_total_power(x_p)
        branch_powers = self.stats.normalize_branch_powers(branch_powers)
        branch_currents = self.stats.normalize_branch_currents(branch_currents)

        return {
            "x_r": torch.from_numpy(x_r),
            "x_p": torch.from_numpy(x_p),
            "branch_powers": torch.from_numpy(branch_powers),
            "branch_currents": torch.from_numpy(branch_currents),
        }


def infer_branch_columns(dataframe: pd.DataFrame) -> Tuple[List[str], List[str]]:
    branch_power_cols = _sorted_branch_columns(dataframe.columns, "power")
    branch_current_cols = _sorted_branch_columns(dataframe.columns, "current")
    if not branch_power_cols or len(branch_power_cols) != len(branch_current_cols):
        raise ValueError("The input CSV must contain matched branch power/current columns.")
    return branch_power_cols, branch_current_cols


def infer_branch_power_columns(dataframe: pd.DataFrame) -> List[str]:
    branch_power_cols = _sorted_branch_columns(dataframe.columns, "power")
    if not branch_power_cols:
        raise ValueError("The input CSV must contain branch_*_power columns.")
    return branch_power_cols


def _ensure_matching_branch_layout(
    reference_power_cols: Sequence[str],
    reference_current_cols: Sequence[str],
    *,
    compare_power_cols: Sequence[str],
    compare_current_cols: Sequence[str],
    label: str,
) -> None:
    if list(reference_power_cols) != list(compare_power_cols) or list(reference_current_cols) != list(compare_current_cols):
        raise ValueError(f"{label} does not match the branch layout of the training data.")


def build_normalization_stats(
    dataframe: pd.DataFrame,
    train_row_indices: Sequence[int],
    branch_power_cols: Sequence[str],
    branch_current_cols: Sequence[str],
) -> NormalizationStats:
    if not train_row_indices:
        raise ValueError("No training rows available to build normalization statistics.")
    unique_indices = sorted(set(int(index) for index in train_row_indices))
    train_frame = dataframe.iloc[unique_indices]

    total_current = train_frame["total_residual_current"].to_numpy(dtype=np.float32)
    total_power = train_frame["total_power"].to_numpy(dtype=np.float32)
    branch_powers = train_frame[list(branch_power_cols)].to_numpy(dtype=np.float32)
    branch_currents = train_frame[list(branch_current_cols)].to_numpy(dtype=np.float32)

    return NormalizationStats(
        total_current_mean=float(total_current.mean()),
        total_current_std=float(max(total_current.std(), 1e-6)),
        total_power_mean=float(total_power.mean()),
        total_power_std=float(max(total_power.std(), 1e-6)),
        branch_power_mean=branch_powers.mean(axis=0),
        branch_power_std=_safe_std(branch_powers),
        branch_current_mean=branch_currents.mean(axis=0),
        branch_current_std=_safe_std(branch_currents),
    )


def _segment_start_indices(segment_positions: Sequence[int], seq_len: int) -> List[int]:
    if len(segment_positions) <= seq_len:
        return []
    first = int(segment_positions[0])
    last = int(segment_positions[-1])
    return list(range(first, last - seq_len + 1))


def build_window_indices(dataframe: pd.DataFrame, seq_len: int) -> List[int]:
    if len(dataframe) <= seq_len:
        raise ValueError("The dataset is shorter than the requested sequence length.")

    if "segment_id" not in dataframe.columns:
        return list(range(0, len(dataframe) - seq_len))

    start_indices: List[int] = []
    current_segment = object()
    current_positions: List[int] = []
    for position, segment_id in enumerate(dataframe["segment_id"].tolist()):
        if current_positions and segment_id != current_segment:
            start_indices.extend(_segment_start_indices(current_positions, seq_len))
            current_positions = []
        current_segment = segment_id
        current_positions.append(position)

    if current_positions:
        start_indices.extend(_segment_start_indices(current_positions, seq_len))

    if not start_indices:
        raise ValueError("No valid windows could be formed from the dataset segments.")
    return start_indices


def split_indices(
    indices: Sequence[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[int], List[int], List[int]]:
    total = len(indices)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train_indices = list(indices[:train_end])
    val_indices = list(indices[train_end:val_end])
    test_indices = list(indices[val_end:])
    if not train_indices or not val_indices or not test_indices:
        raise ValueError("Train/validation/test split produced an empty subset.")
    return train_indices, val_indices, test_indices


def _train_row_indices_from_windows(start_indices: Sequence[int], seq_len: int) -> List[int]:
    train_row_indices: Set[int] = set()
    for idx in start_indices:
        train_row_indices.update(range(idx, idx + seq_len))
    return sorted(train_row_indices)


def build_signal_sampling_weights(
    dataframe: pd.DataFrame,
    start_indices: Sequence[int],
    seq_len: int,
) -> torch.DoubleTensor:
    total_current = dataframe["total_residual_current"].to_numpy(dtype=np.float32)
    total_power = dataframe["total_power"].to_numpy(dtype=np.float32)
    abs_current = np.abs(total_current)
    current_delta = np.abs(np.diff(total_current, prepend=total_current[:1]))
    abs_power = np.abs(total_power)

    def window_mean(values: np.ndarray) -> np.ndarray:
        prefix = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
        starts = np.asarray(start_indices, dtype=np.int64)
        stops = starts + seq_len
        sums = prefix[stops] - prefix[starts]
        return (sums / float(seq_len)).astype(np.float32)

    current_score = window_mean(abs_current)
    delta_score = window_mean(current_delta)
    power_score = window_mean(abs_power)

    current_scale = max(float(np.quantile(current_score, 0.9)), 1e-6)
    delta_scale = max(float(np.quantile(delta_score, 0.9)), 1e-6)
    power_scale = max(float(np.quantile(power_score, 0.9)), 1e-6)

    weights = (
        0.35
        + 0.45 * np.clip(current_score / current_scale, 0.0, 3.0)
        + 0.35 * np.clip(delta_score / delta_scale, 0.0, 3.0)
        + 0.15 * np.clip(power_score / power_scale, 0.0, 3.0)
    )

    if "entity_id" in dataframe.columns:
        entity_names = dataframe["entity_id"].astype(str).to_numpy()
        start_entities = entity_names[np.asarray(start_indices, dtype=np.int64)]
        unique_entities, counts = np.unique(start_entities, return_counts=True)
        entity_balance = {
            name: float(np.sqrt(len(start_entities) / max(count, 1)))
            for name, count in zip(unique_entities.tolist(), counts.tolist())
        }
        weights = weights * np.asarray([entity_balance[name] for name in start_entities], dtype=np.float32)

    weights = np.clip(weights, 0.2, 6.0)
    return torch.as_tensor(weights, dtype=torch.double)


def filter_by_synthetic_mode(dataframe: pd.DataFrame, synthetic_mode: str, label: str) -> pd.DataFrame:
    if synthetic_mode == "all" or "synthetic_variant" not in dataframe.columns:
        return dataframe.reset_index(drop=True)
    if synthetic_mode == "base_only":
        filtered = dataframe[dataframe["synthetic_variant"] == 0]
    elif synthetic_mode == "synthetic_only":
        filtered = dataframe[dataframe["synthetic_variant"] != 0]
    else:
        raise ValueError(f"Unsupported synthetic_mode for {label}: {synthetic_mode}")

    if filtered.empty:
        raise ValueError(f"{label} became empty after applying synthetic_mode={synthetic_mode}.")
    return filtered.reset_index(drop=True)


def create_explicit_split_data_loaders(
    train_data_path: str,
    val_data_path: str,
    test_data_path: str,
    seq_len: int,
    batch_size: int,
    num_workers: int = 0,
    train_sampling: str = "signal_weighted",
    augment_noise_std: float = 0.01,
    augment_scale_range: Tuple[float, float] = (0.97, 1.03),
    train_synthetic_mode: str = "all",
    val_synthetic_mode: str = "all",
    test_synthetic_mode: str = "all",
) -> Tuple[DataLoader, DataLoader, DataLoader, NormalizationStats, int]:
    train_frame = pd.read_csv(train_data_path)
    val_frame = pd.read_csv(val_data_path)
    test_frame = pd.read_csv(test_data_path)

    train_frame = filter_by_synthetic_mode(train_frame, train_synthetic_mode, "Training split")
    val_frame = filter_by_synthetic_mode(val_frame, val_synthetic_mode, "Validation split")
    test_frame = filter_by_synthetic_mode(test_frame, test_synthetic_mode, "Test split")

    train_branch_power_cols, train_branch_current_cols = infer_branch_columns(train_frame)
    val_branch_power_cols, val_branch_current_cols = infer_branch_columns(val_frame)
    test_branch_power_cols, test_branch_current_cols = infer_branch_columns(test_frame)

    _ensure_matching_branch_layout(
        train_branch_power_cols,
        train_branch_current_cols,
        compare_power_cols=val_branch_power_cols,
        compare_current_cols=val_branch_current_cols,
        label="Validation split",
    )
    _ensure_matching_branch_layout(
        train_branch_power_cols,
        train_branch_current_cols,
        compare_power_cols=test_branch_power_cols,
        compare_current_cols=test_branch_current_cols,
        label="Test split",
    )

    train_indices = build_window_indices(train_frame, seq_len)
    val_indices = build_window_indices(val_frame, seq_len)
    test_indices = build_window_indices(test_frame, seq_len)

    stats = build_normalization_stats(
        train_frame,
        _train_row_indices_from_windows(train_indices, seq_len),
        train_branch_power_cols,
        train_branch_current_cols,
    )

    train_dataset = ResidualCurrentWindowDataset(
        dataframe=train_frame,
        seq_len=seq_len,
        start_indices=train_indices,
        stats=stats,
        augment=True,
        noise_std=augment_noise_std,
        scale_range=augment_scale_range,
    )
    val_dataset = ResidualCurrentWindowDataset(
        dataframe=val_frame,
        seq_len=seq_len,
        start_indices=val_indices,
        stats=stats,
        augment=False,
    )
    test_dataset = ResidualCurrentWindowDataset(
        dataframe=test_frame,
        seq_len=seq_len,
        start_indices=test_indices,
        stats=stats,
        augment=False,
    )

    if train_sampling == "signal_weighted":
        train_weights = build_signal_sampling_weights(train_frame, train_indices, seq_len)
        train_sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    num_branches = len(train_branch_power_cols)
    return train_loader, val_loader, test_loader, stats, num_branches


def create_data_loaders(
    data_path: str,
    seq_len: int,
    batch_size: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    num_workers: int = 0,
    train_sampling: str = "signal_weighted",
    augment_noise_std: float = 0.01,
    augment_scale_range: Tuple[float, float] = (0.97, 1.03),
) -> Tuple[DataLoader, DataLoader, DataLoader, NormalizationStats, int]:
    dataframe = pd.read_csv(data_path)
    branch_power_cols, branch_current_cols = infer_branch_columns(dataframe)
    all_indices = build_window_indices(dataframe, seq_len)
    train_indices, val_indices, test_indices = split_indices(all_indices, train_ratio, val_ratio, test_ratio)
    stats = build_normalization_stats(
        dataframe,
        _train_row_indices_from_windows(train_indices, seq_len),
        branch_power_cols,
        branch_current_cols,
    )

    train_dataset = ResidualCurrentWindowDataset(
        dataframe=dataframe,
        seq_len=seq_len,
        start_indices=train_indices,
        stats=stats,
        augment=True,
        noise_std=augment_noise_std,
        scale_range=augment_scale_range,
    )
    val_dataset = ResidualCurrentWindowDataset(
        dataframe=dataframe,
        seq_len=seq_len,
        start_indices=val_indices,
        stats=stats,
        augment=False,
    )
    test_dataset = ResidualCurrentWindowDataset(
        dataframe=dataframe,
        seq_len=seq_len,
        start_indices=test_indices,
        stats=stats,
        augment=False,
    )

    if train_sampling == "signal_weighted":
        train_weights = build_signal_sampling_weights(dataframe, train_indices, seq_len)
        train_sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    num_branches = len(branch_power_cols)
    return train_loader, val_loader, test_loader, stats, num_branches
