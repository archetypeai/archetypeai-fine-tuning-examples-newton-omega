#!/usr/bin/env python3
"""Build a binary SWaT subset for the Omega fine-tune example.

Task: binary classification, normal vs attack, on the Secure Water Treatment
(SWaT) testbed — 51 sensors/actuators sampled at 1 Hz, with a clean Normal/Attack
label. We classify a window of sensor readings as normal operation or a cyber-
physical attack.

Reuses the channel selection from archetypeai-batch-examples-swat: 11 constant
actuators (they never change) are dropped, leaving 40 channels. Unlike that repo,
we keep the REAL datetime timestamp (e.g. "28/12/2015 10:29:14 AM") rather than a
synthetic integer index — the inference reader treats the timestamp column as an
opaque string, so a genuine date carries through cleanly.

Four disjoint splits are carved from the same distribution, all z-scored using
statistics computed across normal + attack records (matching batch-examples-swat):
  - nshot : few-shot KNN reference (n_shots) for baseline + grid search
  - train : fine-tune training set
  - tune  : fine-tune validation set    + KNN grid-search scoring (model selection)
  - eval  : held-out test set, scored once for both baseline and fine-tuned

Attack data is the limiting resource (~54.6k Attack rows total in SWaT), so the
split sizes are bounded accordingly.

Source: raw SWaT CSVs from Kaggle (vishala28/swat-dataset-...): `normal.csv`
(7-day normal run) and `attack.csv` (attack scenarios). Columns: " Timestamp",
51 channels, "Normal/Attack". Only regeneration needs these raw files.

Usage:
    python 1_prepare_data/make_subset.py [--source-dir PATH] [--rows-per-file 5000]
        [--nshot-per-class N] [--train-per-class N] [--tune-per-class N] [--eval-per-class N]
"""

import argparse
import csv
import statistics
from pathlib import Path

DEFAULT_SOURCE = Path.home() / "Documents/archetypeai-batch-examples-swat/data"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "subset"

NORMAL_FILE = "normal.csv"
ATTACK_FILE = "attack.csv"

# 11 actuators that never change in SWaT — dropped (matches batch-examples-swat).
CONSTANT_ACTUATORS = {
    "P102", "P201", "P202", "P204", "P206",
    "P401", "P403", "P404", "P502", "P601", "P603",
}


def _norm(s: str) -> str:
    return s.strip()


def _resolve(fieldnames: list[str], name: str) -> str:
    for f in fieldnames:
        if _norm(f).lower() == name.lower():
            return f
    raise SystemExit(f"column '{name}' not found; header starts with {fieldnames[:4]}")


def channels(fieldnames: list[str]) -> list[str]:
    """The 40 kept channels, in file order (drop Timestamp, label, constants)."""
    out = []
    for f in fieldnames:
        n = _norm(f)
        if n.lower() in ("timestamp", "normal/attack", "label"):
            continue
        if n in CONSTANT_ACTUATORS:
            continue
        out.append(f)
    return out


# Values treated as missing and forward-filled.
MISSING = {"", "nan", "NaN", "None", "null", "NULL", "inf", "-inf", "Inf"}


def read_class(path: Path, want_label: str, limit: int) -> tuple[list[str], list[list[float]]]:
    """Return (timestamps, channel-rows) for contiguous rows matching want_label.

    Missing/invalid sensor values are forward-filled from the last valid reading
    for that channel (the standard imputation for sensor dropouts). Rows with a
    missing value and no prior reading to fill from are skipped.
    """
    timestamps: list[str] = []
    rows: list[list[float]] = []
    skipped = 0
    with path.open() as f:
        reader = csv.DictReader(f)
        ts_col = _resolve(reader.fieldnames, "timestamp")
        label_col = _resolve(reader.fieldnames, "normal/attack")
        chans = channels(reader.fieldnames)
        last: list[float | None] = [None] * len(chans)
        for row in reader:
            if _norm(row[label_col]).lower() != want_label:
                continue
            values: list[float] = []
            fillable = True
            for j, c in enumerate(chans):
                raw = _norm(row[c])
                v: float | None
                if raw in MISSING:
                    v = last[j]  # forward-fill
                else:
                    try:
                        v = float(raw)
                    except ValueError:
                        v = last[j]
                if v is None:  # missing with no prior reading
                    fillable = False
                    break
                last[j] = v
                values.append(v)
            if not fillable:
                skipped += 1
                continue
            timestamps.append(_norm(row[ts_col]))
            rows.append(values)
            if len(rows) >= limit:
                break
    if skipped:
        print(f"  forward-fill: skipped {skipped} rows with leading missing values (no prior to fill)")
    if len(rows) < limit:
        raise SystemExit(f"{path.name}: {len(rows)} '{want_label}' rows, need {limit}")
    return timestamps, rows


def channel_names(path: Path) -> list[str]:
    with path.open() as f:
        return [_norm(c) for c in channels(next(csv.reader(f)))]


def zscore_stats(rows: list[list[float]]):
    cols = list(zip(*rows))
    return [statistics.mean(c) for c in cols], [statistics.pstdev(c) or 1.0 for c in cols]


def write_file(path: Path, names: list[str], ts: list[str], rows, mu, sd) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp"] + names)
        for t, r in zip(ts, rows):
            w.writerow([t] + [f"{(r[j] - mu[j]) / sd[j]:.6f}" for j in range(len(names))])


def write_split(cls: str, split: str, ts, rows, names, mu, sd, rows_per_file: int) -> None:
    n_files = (len(rows) + rows_per_file - 1) // rows_per_file
    for i in range(n_files):
        lo, hi = i * rows_per_file, (i + 1) * rows_per_file
        write_file(OUT_DIR / f"swat_{cls}_{split}_{i}.csv", names, ts[lo:hi], rows[lo:hi], mu, sd)
    print(f"  {cls}/{split}: {len(rows)} rows -> {n_files} files")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--rows-per-file", type=int, default=5000)
    p.add_argument("--nshot-per-class", type=int, default=2000)
    p.add_argument("--train-per-class", type=int, default=20000)
    p.add_argument("--tune-per-class", type=int, default=2000)
    p.add_argument("--eval-per-class", type=int, default=28000)
    args = p.parse_args()

    splits = [("nshot", args.nshot_per_class), ("train", args.train_per_class),
              ("tune", args.tune_per_class), ("eval", args.eval_per_class)]
    total = sum(n for _, n in splits)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.csv"):
        old.unlink()

    names = channel_names(args.source_dir / NORMAL_FILE)
    print(f"{len(names)} channels kept (dropped {len(CONSTANT_ACTUATORS)} constant actuators)")
    print(f"Reading {total} Normal rows from {NORMAL_FILE} ...")
    n_ts, n_rows = read_class(args.source_dir / NORMAL_FILE, "normal", total)
    print(f"Reading {total} Attack rows from {ATTACK_FILE} ...")
    a_ts, a_rows = read_class(args.source_dir / ATTACK_FILE, "attack", total)

    mu, sd = zscore_stats(n_rows + a_rows)  # statistics across normal + attack records

    print("Writing splits (real SWaT timestamps preserved):")
    offset = 0
    for split, n in splits:
        lo, hi = offset, offset + n
        write_split("normal", split, n_ts[lo:hi], n_rows[lo:hi], names, mu, sd, args.rows_per_file)
        write_split("attack", split, a_ts[lo:hi], a_rows[lo:hi], names, mu, sd, args.rows_per_file)
        offset += n

    print(f"\nDone. Binary SWaT (normal vs attack), z-scored, in {OUT_DIR}")


if __name__ == "__main__":
    main()
