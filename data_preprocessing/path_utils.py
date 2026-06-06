from pathlib import Path
from typing import Iterable, List


def _dedupe(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    ordered: List[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


def _resolve_first_existing(candidates: Iterable[Path]) -> Path:
    ordered = _dedupe(path.expanduser() for path in candidates)
    for candidate in ordered:
        if candidate.exists():
            return candidate.resolve()
    return ordered[-1].resolve()


def resolve_manifest_artifact_path(manifest_path: Path, path_value: str) -> Path:
    raw_path = Path(path_value).expanduser()
    base_dir = manifest_path.parent
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
        candidates.append(base_dir / raw_path.name)
        if raw_path.parent.name:
            candidates.append(base_dir / raw_path.parent.name / raw_path.name)
    else:
        candidates.append((Path.cwd() / raw_path).resolve())
        candidates.append((base_dir / raw_path).resolve())
        candidates.append((base_dir / raw_path.name).resolve())
        if len(raw_path.parts) >= 2:
            candidates.append((base_dir / raw_path.parts[-2] / raw_path.name).resolve())
    return _resolve_first_existing(candidates)


def resolve_split_artifact_path(split_config_path: Path, path_value: str) -> Path:
    raw_path = Path(path_value).expanduser()
    base_dir = split_config_path.parent
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
        candidates.append(base_dir / raw_path.name)
    else:
        candidates.append((Path.cwd() / raw_path).resolve())
        candidates.append((base_dir / raw_path).resolve())
        candidates.append((base_dir / raw_path.name).resolve())
    return _resolve_first_existing(candidates)
