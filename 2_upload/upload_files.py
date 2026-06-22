#!/usr/bin/env python3
"""
Upload the TEP subset CSVs to the Archetype AI platform.

Usage:
    python 2_upload/upload_files.py                 # uploads all of data/subset/*.csv
    python 2_upload/upload_files.py path/to/a.csv   # uploads specific files

Flow (per file):
    1. POST /v0.5/files/uploads/initiate          -> presigned part URLs
    2. PUT each part to S3                        -> collect part tokens
    3. POST /v0.5/files/uploads/{id}/complete     -> finalize

The uploaded filename becomes the file_id referenced by the job scripts.
"""

import glob
import os
import sys

import requests

ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
with open(ENV_PATH) as f:
    for line in f:
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)

API_KEY = os.environ["ATAI_API_KEY"]
API_ENDPOINT = os.environ["ATAI_API_ENDPOINT"]
BASE_URL = f"{API_ENDPOINT}/v0.5"
AUTH = {"Authorization": f"Bearer {API_KEY}"}
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "subset")


def initiate_upload(filename: str, file_size: int) -> dict | None:
    resp = requests.post(
        f"{BASE_URL}/files/uploads/initiate",
        headers={**AUTH, "Content-Type": "application/json"},
        json={"filename": filename, "file_type": "text/csv", "num_bytes": file_size},
    )
    # 409 = a file with this id already exists (platform file_ids are immutable).
    # Treat as "already uploaded" so re-runs are idempotent.
    if resp.status_code == 409:
        return None
    resp.raise_for_status()
    return resp.json()


def upload_part(url: str, data: bytes) -> str:
    resp = requests.put(url, data=data, headers={"Content-Length": str(len(data))})
    resp.raise_for_status()
    return resp.headers.get("ETag", "").strip('"')


def complete_upload(upload_id: str, parts: list) -> dict:
    resp = requests.post(
        f"{BASE_URL}/files/uploads/{upload_id}/complete",
        headers={**AUTH, "Content-Type": "application/json"},
        json={"parts": parts},
    )
    resp.raise_for_status()
    return resp.json()


def abort_upload(upload_id: str) -> None:
    requests.post(f"{BASE_URL}/files/uploads/{upload_id}/abort", headers=AUTH)


# Optional suffix appended to each file_id (before .csv). Lets you re-upload
# changed data under fresh ids, since platform file_ids are immutable.
FILE_SUFFIX = os.environ.get("FILE_SUFFIX", "")


def to_file_id(filename: str) -> str:
    if FILE_SUFFIX and filename.endswith(".csv"):
        return f"{filename[:-4]}{FILE_SUFFIX}.csv"
    return filename


def upload_file(file_path: str) -> None:
    filename = to_file_id(os.path.basename(file_path))
    file_size = os.path.getsize(file_path)
    print(f"  {filename} ({file_size / 1024:.0f} KB) ... ", end="", flush=True)

    init = initiate_upload(filename, file_size)
    if init is None:
        print("already uploaded, skipping")
        return
    upload_id = init["upload_id"]

    completed_parts = []
    try:
        with open(file_path, "rb") as f:
            for part in init["parts"]:
                f.seek(part["offset"])
                etag = upload_part(part["url"], f.read(part["length"]))
                completed_parts.append({"part_number": part["part_number"], "part_token": etag})
    except Exception as e:
        print(f"FAILED: {e}")
        abort_upload(upload_id)
        raise

    complete_upload(upload_id, completed_parts)
    print(f"done (file_id={filename})")


def main() -> None:
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not paths:
        print("No files found. Run 1_prepare_data/make_subset.py first.")
        sys.exit(1)

    print(f"Uploading {len(paths)} files to {BASE_URL} ...")
    for path in paths:
        upload_file(path)
    print("All uploads complete.")


if __name__ == "__main__":
    main()
