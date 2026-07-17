#!/usr/bin/env python3
"""
Create and monitor a Machine State inference job.

Classifies the held-out SWaT eval files using a fine-tuning checkpoint produced by
3_fine_tune/create_fine_tune_job.py — either kind:

    --method head (default)  the trained Omega head (data/checkpoint_id.txt)
    --method knn             the KNN vectors stored from the train split
                             (data/knn_checkpoint_id.txt)
    --baseline               no checkpoint: few-shot KNN computed inline from the
                             small n-shot reference on worker.n_shots (legacy path,
                             kept by the platform for backward compatibility)

Usage:
    python 4_inference/create_inference_job.py [--method head|knn] [checkpoint_id]
    python 4_inference/create_inference_job.py --baseline

If checkpoint_id is omitted, it is read from the method's data/ file above.
"""

import csv
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

OMEGA_15_BASE_MODEL_PATH = os.environ.get(
    "OMEGA_15_BASE_MODEL_PATH",
    # Shared marketplace depot — resolves across dev/stage/prod (matches the pipeline default).
    "s3://atai-marketplace-model-depot/omega/1.5/model.pt",
)

POLL_INTERVAL_SEC = 5
TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED"}


import glob
import json

SUBSET_DIR = os.path.join(DATA_DIR, "subset")
CLASSES = ("normal", "attack")

# Optional suffix on file_ids — must match FILE_SUFFIX used at upload time.
FILE_SUFFIX = os.environ.get("FILE_SUFFIX", "")

# Eval inference step (1 = densest, ~150k windows ≈ 25 min; raise to speed up).
EVAL_STEP = int(os.environ.get("EVAL_STEP", "1"))


def finetune_window() -> int:
    """Window the fine-tuned head was trained at (recorded by 3_fine_tune).
    Inference must reuse it, or the head gets embeddings it was not trained on."""
    path = os.path.join(DATA_DIR, "finetune_window.txt")
    if os.path.exists(path):
        with open(path) as f:
            return int(f.read().strip())
    return 32


def get_data_columns() -> list:
    sample = sorted(glob.glob(os.path.join(SUBSET_DIR, "swat_*_nshot_0.csv")))[0]
    with open(sample) as f:
        header = next(csv.reader(f))
    return [c for c in header if c != "timestamp"]


def split_files(split: str, with_metadata: bool) -> list:
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


# Few-shot KNN reference; held-out eval set both models are scored on.
N_SHOT_FILES = split_files("nshot", with_metadata=True)
INFERENCE_FILES = split_files("eval", with_metadata=False)


def load_knn_config() -> dict:
    """Best KNN config from the grid search, or a sensible default."""
    path = os.path.join(DATA_DIR, "best_knn_config.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    print("  [note] no data/best_knn_config.json — run 4_inference/optimize_knn.py;"
          " falling back to w32/k11/cosine.")
    return {"window_size": 32, "n_neighbors": 11, "metric": "cosine", "weights": "uniform"}


def make_payload(checkpoint_id: str | None, method: str) -> dict:
    # No checkpoint: few-shot KNN over base Omega 1.5 embeddings, computed inline
    # from worker.n_shots with the grid-search-optimized config. With a checkpoint
    # the worker rebuilds the classifier from it — trained head (--method head) or
    # stored KNN vectors (--method knn) — and takes window_size/data_columns/KNN
    # params from the checkpoint's metadata, overriding this config. Same encoder,
    # same eval set throughout — only the classifier differs.
    # (n_shots stays attached in all modes: the pipeline requires the port; it is
    # ignored when a checkpoint is present.)
    inputs = {
        "worker.inference": INFERENCE_FILES,
        "worker.n_shots": N_SHOT_FILES,
    }
    if checkpoint_id:
        inputs["worker.fine_tune_checkpoint"] = [
            {"kind": "checkpoint", "checkpoint_id": checkpoint_id, "metadata": {}}
        ]

    if checkpoint_id and method == "head":
        # The head fixes the window; KNN params are unused on this path.
        window, knn = finetune_window(), {"n_neighbors": 11, "metric": "cosine", "weights": "uniform"}
    else:
        cfg = load_knn_config()
        window = cfg["window_size"]
        knn = {k: cfg[k] for k in ("n_neighbors", "metric", "weights")}

    if checkpoint_id:
        name = "swat-machine-state-fine-tuned" if method == "head" else "swat-machine-state-knn"
    else:
        name = "swat-machine-state-baseline"
    payload = {
        "name": name,
        "pipeline_type": "batch",
        "pipeline_key": "machine-state-classification",
        "inputs": inputs,
        "parameters": {
            "worker": {
                "parallelism": 1,
                "config": {
                    "model_type": "omega_1_5_base",
                    "omega_1_5": {"base_model_path": OMEGA_15_BASE_MODEL_PATH},
                    "reader_config": {
                        "window_size": window,
                        "step_size": EVAL_STEP,
                        "timestamp_column": "timestamp",
                        "data_columns": get_data_columns(),
                    },
                    "classifier_config": knn,
                    "batch_size": 32,
                    "data_source": {"source_type": "s3"},
                    "flush_every_n_iteration": 100,
                },
            }
        },
    }
    # Pin a specific pipeline version (e.g. a draft not yet published org-wide).
    pipeline_version = os.environ.get("MSJ_PIPELINE_VERSION")
    if pipeline_version:
        payload["pipeline_version"] = pipeline_version
    return payload


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


def _get_with_retry(url: str, params: dict | None = None, attempts: int = 6) -> dict:
    # Dev is occasionally flaky (transient 5xx); retry rather than crash the poll loop.
    last = None
    for i in range(attempts):
        try:
            resp = requests.get(url, headers=AUTH, params=params, timeout=30)
            if resp.status_code >= 500:
                last = f"{resp.status_code} {resp.reason}"
                time.sleep(min(5 * (i + 1), 30))
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last = str(e)
            time.sleep(min(5 * (i + 1), 30))
    raise RuntimeError(f"GET {url} failed after {attempts} attempts: {last}")


def get_job(job_id: str) -> dict:
    return _get_with_retry(f"{BASE_URL}/batch/jobs/{job_id}")


def list_paginated(job_id: str, kind: str, offset: int) -> dict:
    return _get_with_retry(f"{BASE_URL}/batch/jobs/{job_id}/{kind}", {"offset": offset, "limit": 100})


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


def main() -> None:
    usage = f"Usage: {sys.argv[0]} [--baseline | --method head|knn] [checkpoint_id]"
    args = sys.argv[1:]
    baseline = "--baseline" in args
    args = [a for a in args if a != "--baseline"]
    method = "head"
    if "--method" in args:
        i = args.index("--method")
        method = args[i + 1] if i + 1 < len(args) else ""
        del args[i:i + 2]
    if method not in ("head", "knn") or (baseline and args):
        raise SystemExit(usage)

    checkpoint_id = None
    if not baseline:
        ckpt_file = "checkpoint_id.txt" if method == "head" else "knn_checkpoint_id.txt"
        if args:
            checkpoint_id = args[0]
        else:
            ckpt_path = os.path.join(DATA_DIR, ckpt_file)
            if not os.path.exists(ckpt_path):
                ft_cmd = "3_fine_tune/create_fine_tune_job.py" + ("" if method == "head" else " --method knn")
                print(f"No checkpoint id. Run {ft_cmd} first,")
                print(f"or pass one explicitly: {sys.argv[0]} --method {method} <checkpoint_id>")
                print(f"Or run the legacy inline baseline: {sys.argv[0]} --baseline")
                sys.exit(1)
            with open(ckpt_path) as f:
                checkpoint_id = f.read().strip()

    if baseline:
        mode = "baseline (few-shot KNN, inline)"
    elif method == "head":
        mode = "fine-tuned Omega head"
    else:
        mode = "fine-tuned KNN (stored train vectors)"
    print("=" * 60)
    print(f" Machine State Inference ({mode})")
    print("=" * 60)
    print(f"  Endpoint:   {BASE_URL}")
    print(f"  Checkpoint: {checkpoint_id or '(none — base model + KNN)'}")
    print(f"  Inference:  {[f['file_id'] for f in INFERENCE_FILES]}")

    job = create_job(make_payload(checkpoint_id, method))
    job_id = job["id"]
    print(f"  Job ID:     {job_id}")
    print(f"  Status:     {job['status']}")

    status = watch_job(job_id)
    print(f"\nJob finished with status: {status}")
    if status != "COMPLETED":
        sys.exit(1)

    if baseline:
        filename = "baseline_inference_job_id.txt"
    elif method == "head":
        filename = "last_inference_job_id.txt"
    else:
        filename = "knn_inference_job_id.txt"
    job_id_path = os.path.join(DATA_DIR, filename)
    with open(job_id_path, "w") as f:
        f.write(job_id)
    print(f"Saved job id to {job_id_path}")
    print(f"Next: python 5_evaluate/evaluate_results.py {job_id}")
    print("  or: python 5_evaluate/evaluate_results.py --compare   (baseline vs fine-tuned)")


if __name__ == "__main__":
    main()
