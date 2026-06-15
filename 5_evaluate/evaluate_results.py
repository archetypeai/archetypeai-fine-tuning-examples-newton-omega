#!/usr/bin/env python3
"""
Download inference job outputs and score predictions against ground truth.

Ground truth is file-level: each held-out file swat_<class>_4.csv contains exactly
one class, so every prediction in an output file inherits that file's class. The
true class is recovered from the output filename (..._output_swat_<class>_4*.csv),
which is robust for any number of classes.

Pass two job ids to print a baseline-vs-fine-tuned comparison:
    python 5_evaluate/evaluate_results.py <job_id>
    python 5_evaluate/evaluate_results.py --compare <baseline_job_id> <finetuned_job_id>

With no args, scores the job in data/last_inference_job_id.txt. With --compare and
no ids, reads data/baseline_inference_job_id.txt and data/last_inference_job_id.txt.
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
    print(" Evaluate Omega Predictions" + (" — baseline vs fine-tuned" if compare else ""))
    print("=" * 60)

    if compare:
        if len(args) >= 2:
            baseline_id, finetuned_id = args[0], args[1]
        else:
            baseline_id = read_id("baseline_inference_job_id.txt")
            finetuned_id = read_id("last_inference_job_id.txt")
        if not baseline_id or not finetuned_id:
            print("Need both baseline and fine-tuned job ids (args or saved files).")
            sys.exit(1)

        print("\n--- BASELINE (few-shot KNN, no fine-tuning) ---")
        base = score(collect_pairs(baseline_id))
        print_report(baseline_id, base)

        print("\n--- FINE-TUNED (Omega head) ---")
        fine = score(collect_pairs(finetuned_id))
        print_report(finetuned_id, fine)

        b = base["correct"] / base["total"] if base["total"] else 0
        f = fine["correct"] / fine["total"] if fine["total"] else 0
        print("\n" + "=" * 60)
        print(f"  Baseline accuracy:   {b:.1%}")
        print(f"  Fine-tuned accuracy: {f:.1%}")
        print(f"  Delta:               {f - b:+.1%}")
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
