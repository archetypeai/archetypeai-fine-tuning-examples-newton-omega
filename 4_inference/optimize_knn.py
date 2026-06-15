#!/usr/bin/env python3
"""
Grid-search the few-shot KNN hyperparameters on Omega 1.5, to give the baseline
its best shot before comparing against the fine-tuned head.

For each config it runs a machine-state-classification job with the `nshot` split
as the KNN reference and the `tune` split as the scored set, then picks the config
with the best macro-F1 and writes it to data/best_knn_config.json (consumed by
4_inference/create_inference_job.py --baseline).

Tuning on `tune` (held out from the final `eval` set, same stitched distribution)
avoids the trap where the published grid winner overfit a balanced slice and
collapsed at full scale.

Usage:
    python 4_inference/optimize_knn.py
"""

import csv
import glob
import io
import json
import os
import re
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
SUBSET_DIR = os.path.join(DATA_DIR, "subset")
FILE_SUFFIX = os.environ.get("FILE_SUFFIX", "")
MSJ_VERSION = os.environ.get("MSJ_PIPELINE_VERSION")
OMEGA_15_BASE_MODEL_PATH = os.environ.get(
    "OMEGA_15_BASE_MODEL_PATH",
    "s3://atai-platform-dev-platform-data-us-west-2/model_checkpoints/omega_1_5/omega1.5_target_encoder.pt",
)

CLASSES = ("normal", "attack")
CLASS_RE = re.compile(r"swat_(normal|attack)_(?:nshot|train|tune|eval)_\d+")
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED"}

# Grid over window x n_neighbors x metric. `weights` is fixed to uniform: in the
# batch-examples-swat sweep uniform and distance scored identically (a no-op here),
# so re-running it would only double the job count for no information.
GRID = [
    {"window_size": w, "n_neighbors": k, "metric": m, "weights": "uniform"}
    for w in (16, 32, 64, 128)
    for k in (3, 5, 7, 11, 15)
    for m in ("cosine", "euclidean")
]


def split_files(split: str, with_metadata: bool) -> list:
    files = []
    for cls in CLASSES:
        for path in sorted(glob.glob(os.path.join(SUBSET_DIR, f"swat_{cls}_{split}_*.csv"))):
            stem = os.path.basename(path)[:-len(".csv")]
            fid = f"{stem}{FILE_SUFFIX}.csv"
            files.append({"file_id": fid, "metadata": {"class": cls}} if with_metadata
                         else {"file_id": fid})
    return files


def data_columns() -> list:
    sample = sorted(glob.glob(os.path.join(SUBSET_DIR, "swat_*_nshot_0.csv")))[0]
    with open(sample) as f:
        return [c for c in next(csv.reader(f)) if c != "timestamp"]


N_SHOT = split_files("nshot", True)
SCORE = split_files("tune", False)
COLUMNS = data_columns()


def submit(cfg: dict) -> str:
    payload = {
        "name": f"knn-grid-w{cfg['window_size']}-k{cfg['n_neighbors']}-{cfg['metric']}",
        "pipeline_type": "batch",
        "pipeline_key": "machine-state-classification",
        "inputs": {"worker.inference": SCORE, "worker.n_shots": N_SHOT},
        "parameters": {"worker": {"parallelism": 1, "config": {
            "model_type": "omega_1_5_base",
            "omega_1_5": {"base_model_path": OMEGA_15_BASE_MODEL_PATH},
            "reader_config": {"window_size": cfg["window_size"], "step_size": 1,
                              "timestamp_column": "timestamp", "data_columns": COLUMNS},
            "classifier_config": {k: cfg[k] for k in ("n_neighbors", "metric", "weights")},
            "batch_size": 32, "data_source": {"source_type": "s3"}, "flush_every_n_iteration": 100,
        }}},
    }
    if MSJ_VERSION:
        payload["pipeline_version"] = MSJ_VERSION
    resp = requests.post(f"{BASE_URL}/batch/jobs", headers={**AUTH, "Content-Type": "application/json"}, json=payload)
    if not resp.ok:
        print(f"  submit failed ({resp.status_code}): {resp.text}")
        resp.raise_for_status()
    return resp.json()["id"]


def job_status(job_id: str) -> str:
    return requests.get(f"{BASE_URL}/batch/jobs/{job_id}", headers=AUTH).json()["status"]


def macro_f1(job_id: str) -> float:
    outputs = []
    offset = 0
    while True:
        data = requests.get(f"{BASE_URL}/batch/jobs/{job_id}/outputs",
                            headers=AUTH, params={"limit": 50, "offset": offset}).json()
        outputs.extend(data["outputs"])
        if offset + 50 >= data["total"]:
            break
        offset += 50
    confusion = {a: {p: 0 for p in CLASSES} for a in CLASSES}
    for out in outputs:
        m = CLASS_RE.search(out["data"]["filename"])
        if not m:
            continue
        actual = m.group(1)
        url = out["data"]["ref"]
        text = (requests.get(f"{BASE_URL}{url}", headers=AUTH) if url.startswith("/")
                else requests.get(url)).text
        for row in csv.DictReader(io.StringIO(text)):
            pred = row["Prediction"]
            if pred in CLASSES:
                confusion[actual][pred] += 1
    f1s = []
    for c in CLASSES:
        tp = confusion[c][c]
        fp = sum(confusion[o][c] for o in CLASSES if o != c)
        fn = sum(confusion[c][o] for o in CLASSES if o != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def main() -> None:
    print("=" * 60)
    print(f" KNN grid search on Omega 1.5 ({len(GRID)} configs)")
    print("=" * 60)
    print(f"  reference (n_shots): {len(N_SHOT)} files | scored (tune): {len(SCORE)} files")

    print("\nSubmitting jobs ...")
    jobs = []
    for cfg in GRID:
        job_id = submit(cfg)
        jobs.append((cfg, job_id))
        print(f"  {job_id}  w{cfg['window_size']} k{cfg['n_neighbors']} {cfg['metric']}")

    print("\nWaiting for completion ...")
    results = []
    pending = list(jobs)
    while pending:
        time.sleep(10)
        still = []
        for cfg, job_id in pending:
            status = job_status(job_id)
            if status in TERMINAL:
                f1 = macro_f1(job_id) if status == "COMPLETED" else 0.0
                results.append((f1, cfg, status))
                print(f"  [{status}] macro-F1={f1:.4f}  w{cfg['window_size']} k{cfg['n_neighbors']} {cfg['metric']}")
            else:
                still.append((cfg, job_id))
        pending = still

    results.sort(reverse=True, key=lambda r: r[0])
    print("\nRanking:")
    for f1, cfg, status in results:
        print(f"  macro-F1={f1:.4f}  w{cfg['window_size']} k{cfg['n_neighbors']} {cfg['metric']} {cfg['weights']}")

    best_f1, best_cfg, _ = results[0]
    out_path = os.path.join(DATA_DIR, "best_knn_config.json")
    with open(out_path, "w") as f:
        json.dump(best_cfg, f, indent=2)
    print(f"\nBest config (macro-F1={best_f1:.4f}) written to {out_path}:")
    print(f"  {best_cfg}")
    print("Next: python 4_inference/create_inference_job.py --baseline")


if __name__ == "__main__":
    main()
