#!/usr/bin/env python3
"""
Download inference job outputs and score predictions against ground truth.

Ground truth is file-level: each held-out file swat_<class>_4.csv contains exactly
one class, so every prediction in an output file inherits that file's class. The
true class is recovered from the output filename (..._output_swat_<class>_4*.csv),
which is robust for any number of classes.

Usage:
    python 5_evaluate/evaluate_results.py [job_id]
    python 5_evaluate/evaluate_results.py --compare [job_id ...]

With no args, scores the job in data/last_inference_job_id.txt. With --compare and
no ids, reports every saved run it finds (needs at least two), deltas vs the first:
    data/baseline_inference_job_id.txt   baseline (few-shot KNN, inline)
    data/knn_inference_job_id.txt        fine-tuned KNN (stored train vectors)
    data/last_inference_job_id.txt       fine-tuned head
"""

import csv
import io
import os
import re
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
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# True class from an output filename like "..._output_swat_attack_eval_3_bin.csv".
CLASS_RE = re.compile(r"swat_(normal|attack)_(?:nshot|train|tune|eval)_\d+")


def get_outputs(job_id: str) -> list:
    outputs = []
    offset = 0
    while True:
        resp = requests.get(
            f"{BASE_URL}/batch/jobs/{job_id}/outputs",
            headers=AUTH,
            params={"limit": 50, "offset": offset},
        )
        resp.raise_for_status()
        data = resp.json()
        outputs.extend(data["outputs"])
        if offset + 50 >= data["total"]:
            break
        offset += 50
    return outputs


def fetch_csv(url: str) -> str:
    if url.startswith("/"):
        # Relative API path rather than a presigned URL — needs auth.
        resp = requests.get(f"{BASE_URL}{url}", headers=AUTH)
    else:
        resp = requests.get(url)
    resp.raise_for_status()
    return resp.text


def collect_pairs(job_id: str) -> list:
    """Return (actual_class, predicted_class) for every prediction in the job."""
    pairs = []
    for out in get_outputs(job_id):
        filename = out["data"]["filename"]
        m = CLASS_RE.search(filename)
        if not m:
            print(f"  [warn] cannot parse class from {filename}, skipping")
            continue
        actual = m.group(1)
        text = fetch_csv(out["data"]["ref"])
        for row in csv.DictReader(io.StringIO(text)):
            pairs.append((actual, row["Prediction"]))
    return pairs


def score(pairs: list) -> dict:
    classes = sorted({a for a, _ in pairs} | {p for _, p in pairs})
    confusion = {a: {p: 0 for p in classes} for a in classes}
    for actual, predicted in pairs:
        confusion[actual][predicted] += 1
    total = len(pairs)
    correct = sum(confusion[c][c] for c in classes)
    return {"classes": classes, "confusion": confusion, "total": total, "correct": correct}


def print_report(job_id: str, result: dict) -> None:
    classes, confusion = result["classes"], result["confusion"]
    total, correct = result["total"], result["correct"]
    print(f"\n  Job: {job_id}")
    print(f"  Accuracy: {correct}/{total} = {correct / total:.1%}" if total else "  No predictions.")

    print("\n  Confusion matrix (rows=actual, cols=predicted):")
    print("  " + " " * 10 + "".join(f"{c[:9]:>10}" for c in classes))
    for a in classes:
        print(f"  {a[:9]:>10}" + "".join(f"{confusion[a][p]:>10}" for p in classes))

    print()
    for c in classes:
        tp = confusion[c][c]
        fp = sum(confusion[o][c] for o in classes if o != c)
        fn = sum(confusion[c][o] for o in classes if o != c)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(f"  {c:>10}: precision={precision:5.1%} recall={recall:5.1%} f1={f1:5.1%}")


def read_id(filename: str) -> str | None:
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return None


def main() -> None:
    args = sys.argv[1:]
    compare = "--compare" in args
    args = [a for a in args if a != "--compare"]

    print("=" * 60)
    print(" Evaluate Omega Predictions" + (" — comparison" if compare else ""))
    print("=" * 60)

    if compare:
        if args:
            runs = [(f"Job {i + 1} ({jid})", jid) for i, jid in enumerate(args)]
        else:
            runs = [(label, read_id(fname)) for label, fname in (
                ("Baseline (few-shot KNN, inline)", "baseline_inference_job_id.txt"),
                ("Fine-tuned KNN (train vectors)", "knn_inference_job_id.txt"),
                ("Fine-tuned head", "last_inference_job_id.txt"),
            )]
            runs = [(label, job_id) for label, job_id in runs if job_id]
        if len(runs) < 2:
            print("Need at least two runs to compare (pass job ids, or save runs via 4_inference).")
            sys.exit(1)

        scored = []
        for label, job_id in runs:
            print(f"\n--- {label} ---")
            result = score(collect_pairs(job_id))
            print_report(job_id, result)
            scored.append((label, result))

        ref_label, ref = scored[0]
        ref_acc = ref["correct"] / ref["total"] if ref["total"] else 0
        width = max(len(label) for label, _ in scored)
        print("\n" + "=" * 60)
        for i, (label, res) in enumerate(scored):
            acc = res["correct"] / res["total"] if res["total"] else 0
            delta = "" if i == 0 else f"   ({acc - ref_acc:+.1%} vs {ref_label})"
            print(f"  {label:<{width}} : {acc:.1%}{delta}")
        print("=" * 60)
        return

    if args:
        job_id = args[0]
    else:
        job_id = read_id("last_inference_job_id.txt")
        if not job_id:
            print(f"Usage: {sys.argv[0]} <job_id>  |  --compare <baseline> <finetuned>")
            sys.exit(1)

    print_report(job_id, score(collect_pairs(job_id)))


if __name__ == "__main__":
    main()
