import argparse
import time
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOLDER = REPO_ROOT / "release" / "hf_dataset_repo"
PLACEHOLDER_MARKERS = [
    "Update this section",
    "once you decide",
]
FORBIDDEN_INTERNAL_PATH_MARKERS = [
    "/Users/",
    "data_preprocessing/output/",
]


def verify_release_folder(folder: Path) -> None:
    required = [
        folder / "README.md",
        folder / "benchmark" / "train.csv",
        folder / "benchmark" / "validation.csv",
        folder / "benchmark" / "test.csv",
        folder / "benchmark" / "split_config.json",
        folder / "processed_entities" / "manifest.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required release files:\n" + "\n".join(missing))

    readme_text = (folder / "README.md").read_text()
    for marker in PLACEHOLDER_MARKERS:
        if marker in readme_text:
            raise RuntimeError(f"Dataset README still contains placeholder text: {marker}")

    json_files = [
        folder / "processed_entities" / "manifest.json",
        *sorted(folder.glob("processed_entities/*/*.metadata.json")),
    ]
    for json_path in json_files:
        text = json_path.read_text()
        for marker in FORBIDDEN_INTERNAL_PATH_MARKERS:
            if marker in text:
                raise RuntimeError(f"Internal path marker '{marker}' still present in {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload the prepared HF dataset release folder.")
    parser.add_argument("--repo-id", required=True, help="Target HF dataset repo, e.g. haayan/safeleak-rcd")
    parser.add_argument(
        "--folder",
        default=str(DEFAULT_FOLDER),
        help="Local release folder to upload",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create the repo as public. Default behavior is private staging.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries for each upload step.",
    )
    return parser.parse_args()


def run_with_retries(action_name: str, max_retries: int, fn):
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            result = fn()
            print(f"{action_name}: ok (attempt {attempt}/{max_retries})")
            return result
        except HfHubHTTPError as exc:
            last_error = exc
            print(f"{action_name}: failed with HF error on attempt {attempt}/{max_retries}: {exc}")
            if attempt < max_retries:
                time.sleep(5 * attempt)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"{action_name}: failed on attempt {attempt}/{max_retries}: {exc}")
            if attempt < max_retries:
                time.sleep(5 * attempt)
    raise last_error


def upload_release_in_chunks(api: HfApi, repo_id: str, folder: Path, max_retries: int) -> None:
    readme_path = folder / "README.md"
    benchmark_dir = folder / "benchmark"
    processed_dir = folder / "processed_entities"

    run_with_retries(
        "upload README",
        max_retries,
        lambda: api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            commit_message="Add dataset card",
        ),
    )

    run_with_retries(
        "upload benchmark split",
        max_retries,
        lambda: api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(benchmark_dir),
            path_in_repo="benchmark",
            commit_message="Add benchmark split files",
            ignore_patterns=[".DS_Store", "**/.DS_Store"],
        ),
    )

    manifest_path = processed_dir / "manifest.json"
    run_with_retries(
        "upload processed manifest",
        max_retries,
        lambda: api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=str(manifest_path),
            path_in_repo="processed_entities/manifest.json",
            commit_message="Add processed entity manifest",
        ),
    )

    entity_dirs = sorted(path for path in processed_dir.iterdir() if path.is_dir())
    for entity_dir in entity_dirs:
        remote_dir = f"processed_entities/{entity_dir.name}"
        run_with_retries(
            f"upload {remote_dir}",
            max_retries,
            lambda entity_dir=entity_dir, remote_dir=remote_dir: api.upload_folder(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=str(entity_dir),
                path_in_repo=remote_dir,
                commit_message=f"Add {entity_dir.name}",
                ignore_patterns=[".DS_Store", "**/.DS_Store"],
            ),
        )


def main() -> None:
    args = parse_args()
    folder = Path(args.folder).resolve()
    verify_release_folder(folder)

    api = HfApi()
    user = api.whoami()["name"]
    print(f"Authenticated as: {user}")

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
    )
    print(f"Repository ready: https://huggingface.co/datasets/{args.repo_id}")

    upload_release_in_chunks(api, args.repo_id, folder, args.max_retries)
    print("Upload completed.")

    files = api.list_repo_files(repo_id=args.repo_id, repo_type="dataset")
    print("Key files present:")
    for key_file in [
        "README.md",
        "benchmark/train.csv",
        "benchmark/validation.csv",
        "benchmark/test.csv",
        "benchmark/split_config.json",
        "processed_entities/manifest.json",
    ]:
        print(f"  {key_file}: {'yes' if key_file in files else 'no'}")

    print(f"Dataset URL: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
