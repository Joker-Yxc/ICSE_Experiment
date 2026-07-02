"""
Prepare the two replacement datasets for EsCapturer experiments.

Zenodo:
  https://zenodo.org/records/11079764

Kaggle:
  https://www.kaggle.com/datasets/dmitrijstrizna/quo-vadis-malware-emulation
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path


ZENODO_RECORD_API = "https://zenodo.org/api/records/11079764"
KAGGLE_DATASET = "dmitrijstrizna/quo-vadis-malware-emulation"


def md5sum(path):
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_url(url, target, expected_size=None, expected_md5=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    existing_size = target.stat().st_size if target.exists() else 0
    if expected_size and existing_size == expected_size:
        if expected_md5 is None or md5sum(target) == expected_md5:
            return

    headers = {}
    mode = "wb"
    if existing_size and expected_size and existing_size < expected_size:
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response, open(target, mode) as out:
        shutil.copyfileobj(response, out)

    if expected_size and target.stat().st_size != expected_size:
        raise RuntimeError(f"incomplete download for {target}: {target.stat().st_size}/{expected_size}")
    if expected_md5 and md5sum(target) != expected_md5:
        raise RuntimeError(f"md5 mismatch for {target}")


def prepare_zenodo(target, full=False):
    target.mkdir(parents=True, exist_ok=True)
    metadata_path = target / "record_metadata.json"
    download_url(ZENODO_RECORD_API, metadata_path)

    metadata = json.loads(metadata_path.read_text())
    files = metadata.get("files", [])
    for file_info in files:
        key = file_info.get("key")
        links = file_info.get("links", {})
        url = links.get("self")
        if not key or not url:
            continue
        checksum = file_info.get("checksum", "")
        expected_md5 = checksum.split(":", 1)[1] if checksum.startswith("md5:") else None
        expected_size = file_info.get("size")
        if key == "shas_by_families.json" or full:
            download_url(url, target / key, expected_size=expected_size, expected_md5=expected_md5)


def prepare_kaggle(target):
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", KAGGLE_DATASET,
            "-p", str(target),
            "--unzip",
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zenodo-dir",
        default="datasets_50k/zenodo_11079764/data/raw",
        help="Zenodo source files directory",
    )
    parser.add_argument(
        "--quo-vadis-dir",
        default="datasets_50k/quo_vadis/data/raw",
        help="Quo Vadis source files directory",
    )
    parser.add_argument("--zenodo-full", action="store_true", help="同时下载 Zenodo 9.3GB 主压缩包")
    parser.add_argument("--skip-kaggle", action="store_true", help="跳过 Kaggle 下载")
    args = parser.parse_args()

    prepare_zenodo(Path(args.zenodo_dir), full=args.zenodo_full)
    if not args.skip_kaggle:
        prepare_kaggle(Path(args.quo_vadis_dir))


if __name__ == "__main__":
    main()
