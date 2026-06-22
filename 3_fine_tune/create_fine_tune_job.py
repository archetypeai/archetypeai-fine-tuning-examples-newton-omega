#!/usr/bin/env python3
"""
Create and monitor an Omega head fine-tuning job via the canonical fine-tuning API.

Fine-tunes a classification head on top of the frozen Omega 1.5 encoder using
labeled SWaT CSV files. Uses the OpenAI-style fine-tuning service (the same one
the Console uses, and the Newton/"fusion" example uses):

    POST /v0.6/fine_tuning/jobs            create the job  (model="omega")
    GET  /v0.6/fine_tuning/jobs/{id}       status
    GET  /v0.6/fine_tuning/jobs/{id}/events|checkpoints

This produces an `ftj_...` job and `ckp_...` checkpoints; the platform backs the
"omega"/"head" method onto the internal `fine-tuning-omega` pipeline. The resulting
checkpoint id is written to data/checkpoint_id.txt for the inference step.

Usage:
    python 3_fine_tune/create_fine_tune_job.py
"""

import csv
import glob
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
API_ENDPOINT = os.environ["ATAI_API_ENDPOINT"].rstrip("/")
FT = f"{API_ENDPOINT}/v0.6/fine_tuning"
AUTH = {"Authorization": f"Bearer {API_KEY}"}
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SUBSET_DIR = os.path.join(DATA_DIR, "subset")
CLASSES = ("normal", "attack")

# Must match FILE_SUFFIX used at upload time (platform file_ids are immutable).
FILE_SUFFIX = os.environ.get("FILE_SUFFIX", "")
# Optional reproducibility seed (Denis: omega results vary by seed; pin for repeatability).
SEED = int(os.environ["OMEGA_FT_SEED"]) if os.environ.get("OMEGA_FT_SEED") else 42

POLL_INTERVAL_SEC = 15
# v0.6 uses SUCCEEDED; accept the JOS-style terminals too.
TERMINAL = {"SUCCEEDED", "COMPLETED", "FAILED", "CANCELLED", "CANCELED", "STOPPED"}
SUCCESS = {"SUCCEEDED", "COMPLETED"}


def finetune_window() -> int:
    """Train at the KNN grid's winning window so the baseline and the head see
    identical embeddings (controlled comparison). Falls back to 32."""
    path = os.path.join(DATA_DIR, "best_knn_config.json")
    if os.path.exists(path):
        with open(path) as f:
            return int(json.load(f)["window_size"])
    print("  [note] no best_knn_config.json — run 4_inference/optimize_knn.py first; defaulting window to 32.")
    return 32


def get_data_columns() -> list:
    sample = sorted(glob.glob(os.path.join(SUBSET_DIR, "swat_*_nshot_0.csv")))[0]
    with open(sample) as f:
        return [c for c in next(csv.reader(f)) if c != "timestamp"]


def omega_files(split: str) -> list:
    """v0.6 Omega n-shot files for a split: one entry per file, class in `label`."""
    out = []
    for cls in CLASSES:
        for path in sorted(glob.glob(os.path.join(SUBSET_DIR, f"swat_{cls}_{split}_*.csv"))):
            stem = os.path.basename(path)[:-len(".csv")]
            out.append({"type": "n_shot", "file_id": f"{stem}{FILE_SUFFIX}.csv",
                        "label": cls, "format": "csv"})
    if not out:
        raise SystemExit(f"No swat_*_{split}_*.csv in {SUBSET_DIR}. Run 1_prepare_data/make_subset.py first.")
    return out


FINETUNE_WINDOW = finetune_window()
TRAIN_FILES = omega_files("train")   # labeled training set
EVAL_FILES = omega_files("tune")     # labeled validation/test set (required for omega)


def build_request() -> dict:
    return {
        "name": "swat-omega-fine-tune",
        "seed": SEED,
        "model": "omega",
        "method": {
            "type": "head",
            "head": {
                "hyperparameters": {"batch_size": 32, "n_epochs": 30},
                "reader": {
                    "window_size": FINETUNE_WINDOW,
                    "step_size": 4,
                    "data_columns": get_data_columns(),
                },
            },
        },
        "training_files": TRAIN_FILES,
        "validation_files": EVAL_FILES,
    }


def _get(path: str, params: dict | None = None, attempts: int = 6) -> dict:
    last = None
    for i in range(attempts):
        try:
            r = requests.get(f"{FT}{path}", headers=AUTH, params=params, timeout=30)
            if r.status_code >= 500:
                last = f"{r.status_code} {r.reason}"
                time.sleep(min(5 * (i + 1), 30)); continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last = str(e); time.sleep(min(5 * (i + 1), 30))
    raise RuntimeError(f"GET {path} failed after {attempts} attempts: {last}")


def create_job(body: dict) -> dict:
    r = requests.post(f"{FT}/jobs", headers={**AUTH, "Content-Type": "application/json"},
                      json=body, timeout=60)
    if not r.ok:
        print(f"Job creation failed ({r.status_code}): {r.text}")
        r.raise_for_status()
    return r.json()


def watch(job_id: str) -> str:
    last_status = None
    seen: set[str] = set()
    print("\nWatching job progress ...")
    while True:
        job = _get(f"/jobs/{job_id}")
        status = job.get("status")
        if status != last_status:
            print(f"[status] {status}")
            last_status = status
        # events are newest-first; print unseen oldest-first
        for e in reversed(_get(f"/jobs/{job_id}/events", {"limit": 100}).get("data", [])):
            eid = e.get("id") or f"{e.get('created_at')}-{e.get('message')}"
            if eid in seen:
                continue
            seen.add(eid)
            print(f"  [event][{e.get('level','')}] {e.get('type') or e.get('event_type','')}: {e.get('message','')}")
        if status in TERMINAL:
            return status
        time.sleep(POLL_INTERVAL_SEC)


def main() -> None:
    print("=" * 60)
    print(f" Omega Head Fine-Tune via /v0.6/fine_tuning/jobs (SWaT: {', '.join(CLASSES)})")
    print("=" * 60)
    print(f"  Endpoint: {FT}/jobs")
    print(f"  Window:   {FINETUNE_WINDOW} (matched to KNN grid winner) | seed: {SEED}")
    print(f"  Train:    {[f['file_id'] for f in TRAIN_FILES]}")
    print(f"  Val:      {[f['file_id'] for f in EVAL_FILES]}")

    job = create_job(build_request())
    job_id = job["id"]
    print(f"  Job ID:   {job_id}   (fine_tuned_model resolves on success)")
    print(f"  Status:   {job.get('status')}")

    status = watch(job_id)
    print(f"\nJob finished with status: {status}")
    if status not in SUCCESS:
        sys.exit(1)

    checkpoints = _get(f"/jobs/{job_id}/checkpoints").get("data", [])
    if not checkpoints:
        print("No checkpoints were produced.")
        sys.exit(1)

    print("Checkpoints:")
    for c in sorted(checkpoints, key=lambda x: x.get("step", 0)):
        print(f"  - id={c.get('id')} step={c.get('step')} metrics={json.dumps(c.get('metrics'))}")

    # Best = highest step (the worker saves on each validation improvement).
    best = max(checkpoints, key=lambda c: c.get("step", 0))
    checkpoint_id = best["id"]
    with open(os.path.join(DATA_DIR, "checkpoint_id.txt"), "w") as f:
        f.write(checkpoint_id)
    with open(os.path.join(DATA_DIR, "finetune_window.txt"), "w") as f:
        f.write(str(FINETUNE_WINDOW))
    print(f"\nSaved checkpoint id: {checkpoint_id} (step {best.get('step')})")
    print(f"Trained at window {FINETUNE_WINDOW} (saved to data/finetune_window.txt)")
    print("Next: python 4_inference/create_inference_job.py")


if __name__ == "__main__":
    main()
