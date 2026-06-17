#!/usr/bin/env python3
"""
Create and monitor an Omega fine-tuning job on the Archetype AI platform.

Fine-tunes a classification head on top of the frozen Omega 1.5 encoder using
labeled SWaT CSV files (one class per file, declared via input metadata).

Usage:
    python 3_fine_tune/create_fine_tune_job.py

On success, the resulting checkpoint id is written to data/checkpoint_id.txt
for the inference step to pick up.
"""

import csv
import json
import os
import sys
import time

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
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# The Omega 1.5 encoder checkpoint resolved by the platform worker.
OMEGA_15_BASE_MODEL_PATH = os.environ.get(
    "OMEGA_15_BASE_MODEL_PATH",
    "s3://atai-platform-dev-platform-data-us-west-2/model_checkpoints/omega_1_5/omega1.5_target_encoder.pt",
)

POLL_INTERVAL_SEC = 5
TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED"}


import glob
import json

SUBSET_DIR = os.path.join(DATA_DIR, "subset")
CLASSES = ("normal", "attack")

# Optional suffix on file_ids — must match FILE_SUFFIX used at upload time.
FILE_SUFFIX = os.environ.get("FILE_SUFFIX", "")


def finetune_window() -> int:
    """Train the head at the KNN grid's winning window, so the baseline and the
    head see identical embeddings (a controlled comparison). Falls back to 32 if
    the grid hasn't run yet."""
    path = os.path.join(DATA_DIR, "best_knn_config.json")
    if os.path.exists(path):
        with open(path) as f:
            return int(json.load(f)["window_size"])
    print("  [note] no best_knn_config.json — run 4_inference/optimize_knn.py first;"
          " defaulting head window to 32.")
    return 32


FINETUNE_WINDOW = finetune_window()


def get_data_columns() -> list:
    sample = sorted(glob.glob(os.path.join(SUBSET_DIR, "swat_*_nshot_0.csv")))[0]
    with open(sample) as f:
        header = next(csv.reader(f))
    return [c for c in header if c != "timestamp"]


def split_files(split: str, with_metadata: bool) -> list:
    """All file_ids for a split (e.g. 'train'), across both classes."""
    files = []
    for cls in CLASSES:
        for path in sorted(glob.glob(os.path.join(SUBSET_DIR, f"swat_{cls}_{split}_*.csv"))):
            stem = os.path.basename(path)[:-len(".csv")]
            file_id = f"{stem}{FILE_SUFFIX}.csv"
            files.append({"file_id": file_id, "metadata": {"class": cls}} if with_metadata
                         else {"file_id": file_id})
    if not files:
        raise SystemExit(f"No swat_*_{split}_*.csv in {SUBSET_DIR}. Run 1_prepare_data/make_subset.py first.")
    return files


# train split trains the head; tune split is the validation set.
TRAIN_FILES = split_files("train", with_metadata=True)
TEST_FILES = split_files("tune", with_metadata=True)

JOB_PAYLOAD = {
    "name": "swat-omega-fine-tune",
    "pipeline_type": "training",
    "pipeline_key": "fine-tuning-omega",
    "inputs": {
        "worker.train_data": TRAIN_FILES,
        "worker.eval_data": TEST_FILES,
    },
    "parameters": {
        "worker": {
            "parallelism": 1,
            "config": {
                "model_type": "omega_1_5_base",
                "omega_1_5": {"base_model_path": OMEGA_15_BASE_MODEL_PATH},
                "reader_config": {
                    # Train at the KNN grid's winning window so both classifiers
                    # see identical embeddings. Fine-tuned inference must reuse
                    # this same window (recorded to data/finetune_window.txt).
                    "window_size": FINETUNE_WINDOW,
                    # step 4 over the 30k-row/class train split => ~15k windows.
                    "step_size": 4,
                    "data_columns": get_data_columns(),
                },
                "batch_size": 32,
                "data_source": {"source_type": "s3"},
                "train_config": {"epochs": 30},
            },
        }
    },
}


def create_job(payload: dict) -> dict:
    resp = requests.post(
        f"{BASE_URL}/batch/jobs",
        headers={**AUTH, "Content-Type": "application/json"},
        json=payload,
    )
    if not resp.ok:
        print(f"Job creation failed ({resp.status_code}): {resp.text}")
        resp.raise_for_status()
    return resp.json()


def get_job(job_id: str) -> dict:
    resp = requests.get(f"{BASE_URL}/batch/jobs/{job_id}", headers=AUTH)
    resp.raise_for_status()
    return resp.json()


def list_paginated(job_id: str, kind: str, offset: int) -> dict:
    resp = requests.get(
        f"{BASE_URL}/batch/jobs/{job_id}/{kind}",
        headers=AUTH,
        params={"offset": offset, "limit": 100},
    )
    resp.raise_for_status()
    return resp.json()


def watch_job(job_id: str) -> str:
    last_status = None
    event_offset = 0
    progress_offset = 0

    print("\nWatching job progress ...")
    while True:
        job = get_job(job_id)
        status = job["status"]
        if status != last_status:
            print(f"[status] {status}")
            last_status = status

        events = list_paginated(job_id, "events", event_offset).get("events", [])
        for event in events:
            print(f"[event][{event.get('level')}] {event.get('event_type')}: {event.get('message') or '-'}")
        event_offset += len(events)

        progress = list_paginated(job_id, "progress", progress_offset).get("entries", [])
        for entry in progress:
            print(f"[progress][{entry.get('kind')} step={entry.get('step')}] {entry.get('message') or '-'}")
        progress_offset += len(progress)

        if status in TERMINAL_STATUSES:
            return status

        time.sleep(POLL_INTERVAL_SEC)


def list_checkpoints(job_id: str) -> list:
    resp = requests.get(
        f"{BASE_URL}/batch/jobs/{job_id}/checkpoints",
        headers=AUTH,
        params={"offset": 0, "limit": 100},
    )
    resp.raise_for_status()
    return resp.json().get("checkpoints", [])


def main() -> None:
    print("=" * 60)
    print(f" Omega Fine-Tune Job (SWaT, {len(CLASSES)} classes: {', '.join(CLASSES)})")
    print("=" * 60)
    print(f"  Endpoint: {BASE_URL}")
    print(f"  Window:   {FINETUNE_WINDOW} (matched to KNN grid winner)")
    print(f"  Train:    {[f['file_id'] for f in TRAIN_FILES]}")
    print(f"  Test:     {[f['file_id'] for f in TEST_FILES]}")

    job = create_job(JOB_PAYLOAD)
    job_id = job["id"]
    print(f"  Job ID:   {job_id}")
    print(f"  Status:   {job['status']}")

    status = watch_job(job_id)
    print(f"\nJob finished with status: {status}")
    if status != "COMPLETED":
        sys.exit(1)

    checkpoints = list_checkpoints(job_id)
    if not checkpoints:
        print("No checkpoints were produced.")
        sys.exit(1)

    print("Checkpoints:")
    for ckpt in checkpoints:
        print(f"  - id={ckpt['id']} name={ckpt.get('name')} step={ckpt.get('step')}")
        print(f"    metrics={json.dumps(ckpt.get('metrics'))}")

    # The job saves a checkpoint each time validation improves, so the highest
    # step is the best-performing one. (The list is not guaranteed ordered.)
    best = max(checkpoints, key=lambda c: c.get("step", 0))
    checkpoint_id = best["id"]
    ckpt_path = os.path.join(DATA_DIR, "checkpoint_id.txt")
    with open(ckpt_path, "w") as f:
        f.write(checkpoint_id)
    # Record the training window so fine-tuned inference uses the same one.
    with open(os.path.join(DATA_DIR, "finetune_window.txt"), "w") as f:
        f.write(str(FINETUNE_WINDOW))
    print(f"\nSaved checkpoint id to {ckpt_path}: {checkpoint_id}")
    print(f"Trained at window {FINETUNE_WINDOW} (saved to data/finetune_window.txt)")
    print("Next: python 4_inference/create_inference_job.py")


if __name__ == "__main__":
    main()
