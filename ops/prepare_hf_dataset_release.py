import json
import shutil
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = REPO_ROOT / "data_preprocessing" / "output" / "benchmarks" / "shanse001_rebuilt_entity_split"
PROCESSED_DIR = REPO_ROOT / "data_preprocessing" / "output" / "shanse001_rebuilt_v2"
TEMPLATE_README = REPO_ROOT / "release_templates" / "hf_dataset_README.md"
OUTPUT_DIR = REPO_ROOT / "release" / "hf_dataset_repo"
PUBLIC_PROCESSED_ROOT = Path("processed_entities")


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def dataset_public_paths(dataset_name: str, variant_count: int) -> dict:
    dataset_root = PUBLIC_PROCESSED_ROOT / dataset_name
    stem = dataset_root / dataset_name
    return {
        "combined": f"{stem}.csv",
        "base": f"{stem}.base.csv",
        "metadata": f"{stem}.metadata.json",
        "variants": [f"{stem}.variant_{idx}.csv" for idx in range(1, variant_count + 1)],
    }


def append_unique_note(notes: list, note: str) -> list:
    if note not in notes:
        notes.append(note)
    return notes


def build_row_example_block(train_csv_path: Path) -> str:
    with train_csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)

    keys = [
        "timestamp",
        "total_residual_current",
        "total_power",
        "branch_1_power",
        "branch_1_current",
        "branch_2_power",
        "branch_2_current",
        "branch_3_power",
        "branch_3_current",
        "branch_12_power",
        "branch_12_current",
        "synthetic_variant",
        "segment_id",
        "entity_id",
    ]

    lines = ["```json", "{"]
    for index, key in enumerate(keys):
        value = row[key]
        if key in {"synthetic_variant"}:
            rendered = value
        else:
            rendered = json.dumps(value, ensure_ascii=False)
        suffix = "," if index < len(keys) - 1 else ""
        lines.append(f'  "{key}": {rendered}{suffix}')
    lines.extend(["}", "```"])
    return "\n".join(lines)


def build_target_example_block(train_csv_path: Path) -> str:
    with train_csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)

    lines = [
        "```text",
        "target branch: 12",
        f'inputs  = (total_residual_current={row["total_residual_current"]}, total_power={row["total_power"]}, branch_12_power={row["branch_12_power"]})',
        f'target  = branch_12_current={row["branch_12_current"]}',
        "```",
    ]
    return "\n".join(lines)


def render_dataset_readme(template_path: Path, train_csv_path: Path) -> str:
    text = template_path.read_text()
    text = text.replace("{{ROW_EXAMPLE_BLOCK}}", build_row_example_block(train_csv_path))
    text = text.replace("{{TARGET_EXAMPLE_BLOCK}}", build_target_example_block(train_csv_path))
    return text


def verify_inputs() -> None:
    required = [
        BENCHMARK_DIR / "shanse001_rebuilt_entity_split.train.csv",
        BENCHMARK_DIR / "shanse001_rebuilt_entity_split.val.csv",
        BENCHMARK_DIR / "shanse001_rebuilt_entity_split.test.csv",
        BENCHMARK_DIR / "split_config.json",
        PROCESSED_DIR / "manifest.json",
        TEMPLATE_README,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))


def rewrite_benchmark_split_config(src: Path, dst: Path) -> None:
    payload = json.loads(src.read_text())
    payload["train_csv_path"] = "train.csv"
    payload["val_csv_path"] = "validation.csv"
    payload["test_csv_path"] = "test.csv"
    write_json(dst, payload)


def rewrite_public_manifest(manifest_path: Path) -> None:
    payload = json.loads(manifest_path.read_text())
    payload["generated_from_manifest"] = "source manifest path omitted in the public release bundle"
    for entry in payload.get("datasets", []):
        variant_count = int(entry.get("synthetic_variants", len(entry.get("variant_csv_paths", []))))
        public_paths = dataset_public_paths(entry["dataset_name"], variant_count)
        entry["csv_path"] = public_paths["combined"]
        entry["metadata_path"] = public_paths["metadata"]
        entry["variant_csv_paths"] = public_paths["variants"]
        entry["base_csv_path"] = public_paths["base"]
    write_json(manifest_path, payload)


def rewrite_public_metadata(metadata_path: Path) -> None:
    payload = json.loads(metadata_path.read_text())
    variant_count = int(payload.get("augmented_variant_count", len(payload.get("variant_csv_paths", []))))
    public_paths = dataset_public_paths(payload["dataset_name"], variant_count)
    payload["source_csv_path"] = public_paths["base"]
    payload["base_csv_path"] = public_paths["base"]
    payload["combined_csv_path"] = public_paths["combined"]
    payload["variant_csv_paths"] = public_paths["variants"]
    payload["notes"] = append_unique_note(
        payload.get("notes", []),
        "All file paths in this public metadata file are relative to the dataset repository root.",
    )
    write_json(metadata_path, payload)


def rewrite_public_processed_bundle(processed_dir: Path) -> None:
    rewrite_public_manifest(processed_dir / "manifest.json")
    for metadata_path in sorted(processed_dir.glob("*/*.metadata.json")):
        rewrite_public_metadata(metadata_path)


def main() -> None:
    verify_inputs()

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "README.md").write_text(
        render_dataset_readme(
            TEMPLATE_README,
            BENCHMARK_DIR / "shanse001_rebuilt_entity_split.train.csv",
        )
    )

    benchmark_out = OUTPUT_DIR / "benchmark"
    benchmark_out.mkdir(parents=True, exist_ok=True)
    copy_file(
        BENCHMARK_DIR / "shanse001_rebuilt_entity_split.train.csv",
        benchmark_out / "train.csv",
    )
    copy_file(
        BENCHMARK_DIR / "shanse001_rebuilt_entity_split.val.csv",
        benchmark_out / "validation.csv",
    )
    copy_file(
        BENCHMARK_DIR / "shanse001_rebuilt_entity_split.test.csv",
        benchmark_out / "test.csv",
    )
    rewrite_benchmark_split_config(
        BENCHMARK_DIR / "split_config.json",
        benchmark_out / "split_config.json",
    )

    processed_out = OUTPUT_DIR / "processed_entities"
    copy_tree(PROCESSED_DIR, processed_out)
    rewrite_public_processed_bundle(processed_out)

    print(f"Prepared Hugging Face dataset release folder: {OUTPUT_DIR}")
    print("Upload this directory to a Hugging Face dataset repository.")


if __name__ == "__main__":
    main()
