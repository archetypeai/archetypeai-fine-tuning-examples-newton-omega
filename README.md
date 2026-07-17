# Fine-Tuning Omega on Archetype AI — an end-to-end example

This repository is a hands-on, end-to-end example of **fine-tuning an Omega 1.5
classification head** on the Archetype AI platform, and measuring whether it
actually beats the no-training alternative (few-shot K-nearest-neighbours).

The task is **cyber-physical attack detection**: given a window of sensor readings
from a water-treatment plant, decide whether the plant is operating **normally** or
is **under attack**. We use the
[Secure Water Treatment (SWaT)](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)
testbed — 51 sensors/actuators sampled at 1 Hz with a clean binary `Normal/Attack`
label.

Everything runs in the cloud through the platform API. You don't need a GPU, the
model weights, or anything beyond an API key.

## The idea

Omega is a foundation model for time-series: it turns a window of sensor data into
an embedding vector. There are two ways to build a classifier on top of it:

1. **Few-shot KNN (no training).** Embed a handful of labeled examples, then label
   each new window by its nearest neighbours. Fast, zero training — but it leans
   entirely on the *frozen* embedding space.
2. **Fine-tuning (this example).** Train a small classification *head* on top of the
   frozen Omega encoder using your labeled data. More setup, but the head can learn
   decision boundaries that raw nearest-neighbour retrieval misses.

Both methods share the same first stage and differ only in the final step:

```
Few-shot KNN (baseline):
  sensor window → [Omega encoder, frozen] → embedding → nearest-neighbour lookup → label

Fine-tuned head (this example):
  sensor window → [Omega encoder, frozen] → embedding → [small trained network] → label
                                                          ↑ the "head" — only it learns
```

A few clarifications, since the terminology trips people up:

- **KNN is not a model you train.** It has no weights — it just stores labeled example
  embeddings and classifies by nearest-neighbour vote. (The FT API does offer a
  `knn` *method*, but it trains nothing: it embeds your labeled files once and stores
  the vectors as a checkpoint — "fine-tuning" KNN means giving it more reference
  data, nothing else. The head, by contrast, actually learns.)
- **The "head" is a small neural network** that maps an Omega embedding to a class
  score. It has its own learnable weights, trained on your labeled data.
- **The Omega encoder stays frozen in both cases** — its weights never change. Only the
  head learns. (Fine-tuning runs through `model: "omega"` on the canonical FT API; the
  encoder itself is not retrained, which is also why it's fast and cheap.)
- **Why the head can beat KNN:** nearest-neighbour can only draw boundaries from raw
  distance in the frozen embedding space; a trained head can learn which embedding
  dimensions actually distinguish an attack, expressing boundaries KNN cannot.

This example gives the KNN baseline its best shot — a hyperparameter grid search,
plus (where the `knn` fine-tuning method is deployed) a run with 10× the reference
vectors from the same files the head trains on — and compares it head-to-head against
the fine-tuned model: **does fine-tuning a head do better, on the same data?**

## Pipeline at a glance

| Step | Command | What it does |
|------|---------|--------------|
| 1. Prepare | `python3 1_prepare_data/make_subset.py` | Build z-scored, imputed splits from raw SWaT |
| 2. Upload | `python3 2_upload/upload_files.py` | Upload the CSVs to the platform |
| 3. Optimize KNN | `python3 4_inference/optimize_knn.py` | Grid-search the baseline's best KNN config |
| 4. Baseline | `python3 4_inference/create_inference_job.py --baseline` | Few-shot KNN on the eval set — the "before" |
| 5. Fine-tune | `python3 3_fine_tune/create_fine_tune_job.py` | Train the Omega 1.5 head |
| 6. Fine-tuned | `python3 4_inference/create_inference_job.py` | Classify the eval set with the head — the "after" |
| 7. Compare | `python3 5_evaluate/evaluate_results.py --compare` | Baseline vs fine-tuned, macro-F1 |

State flows between steps through small files in `data/` (`best_knn_config.json`,
`checkpoint_id.txt`, `*_inference_job_id.txt`).

Steps 5–6 use the `head` fine-tuning method. The API also offers a second method,
`knn`: no gradient training — the job embeds the *same* labeled train files once and
stores the vectors as the checkpoint; inference re-fits a KNN from them. Where it is
deployed (Dev today) you can fine-tune both ways on identical data and compare all
three runs:

```bash
python3 3_fine_tune/create_fine_tune_job.py --method knn     # embeds train files → ckp_…
python3 4_inference/create_inference_job.py --method knn     # classifies eval with that checkpoint
python3 5_evaluate/evaluate_results.py --compare             # baseline vs KNN vs head
```

`--compare` reports every run it finds saved in `data/` (inline few-shot baseline,
fine-tuned KNN, fine-tuned head) with deltas against the baseline.

## Setup

```bash
cp .env.example .env          # then set ATAI_API_KEY and ATAI_API_ENDPOINT
uv sync                        # or: python3 -m venv .venv && .venv/bin/pip install requests
```

## Getting the data

**Path A (default — no download).** The prepared splits are committed under
`data/subset/`, so you can skip straight to uploading.

**Path B (regenerate from Kaggle).** To rebuild the files from source:

1. Download the SWaT CSVs from Kaggle
   [`vishala28/swat-dataset-secure-water-treatment-system`](https://www.kaggle.com/datasets/vishala28/swat-dataset-secure-water-treatment-system)
   — you need `normal.csv` (the 7-day normal run) and `attack.csv` (attack scenarios).
   Put them in one folder, e.g. `~/Downloads/swat`.
2. Regenerate:
   ```bash
   python3 1_prepare_data/make_subset.py --source-dir ~/Downloads/swat
   ```

## The data

`make_subset.py` builds four disjoint splits from the same distribution. The three
prep steps:

- **Contiguous shots** — each split is a contiguous slice of the timeline (not shuffled).
- **Z-scoring** — channels are standardized using statistics computed across all records
  (normal + attack), matching `batch-examples-swat`. This keeps every channel on one
  consistent scale and avoids extreme values (normal-only stats would blow attack
  readings up to ~500σ).
- **Imputation** — missing/invalid sensor readings are forward-filled from the last
  valid value (the standard fix for sensor dropouts).

It keeps the **real SWaT timestamp** (e.g. `28/12/2015 10:29:14 AM`) — the inference
reader treats the timestamp column as an opaque string, so a genuine datetime carries
through. It keeps **40 channels**, dropping 11 constant actuators that never change.

| Split | Rows/class | Used for |
|---|---|---|
| `nshot` | 2,000 | few-shot KNN reference (baseline + grid search) |
| `train` | 20,000 | fine-tune training |
| `tune` | 2,000 | fine-tune validation + KNN grid-search selection |
| `eval` | 28,000 | held-out test set, scored once for every run |

Sizes are bounded by SWaT's attack data (~54.6k Attack rows total — the limiting
resource). **Tune and eval are drawn the same way**, so a KNN config selected on `tune`
generalizes to `eval`. The held-out `eval` (~56k rows, step 1 → ~56k windows) runs
well under an hour at ~100 windows/sec; raise `EVAL_STEP` or worker `parallelism` to
go faster.

## Tuning the KNN baseline (grid search)

For the comparison to be meaningful, the baseline must be KNN at its *best*. So
`4_inference/optimize_knn.py` grid-searches **40 configs** (window {16,32,64,128} × k
{3,5,7,11,15} × metric {cosine,euclidean}; `weights` fixed to `uniform`, an empirical
no-op) on the `tune` split and keeps the best macro-F1. The winner is written to
`data/best_knn_config.json`; the baseline run uses it on `eval`. The winning
`window_size` is also reused for the fine-tuned head, so both classifiers see identical
embeddings and the comparison isolates the classifier, not the window.

## Results

On the held-out `eval` set (balanced, ~54.5k windows), Omega 1.5, all `seed=0`, all
three runs on the same day and worker build (KNN config w128/k15/cosine throughout):

| Model | Reference/training data | Accuracy | Macro-F1 |
|---|---|---|---|
| Baseline (few-shot KNN, inline) | `nshot` (2k rows/class) | 79.7% | 0.796 |
| Fine-tuned KNN (stored train vectors) | `train` (20k rows/class) | 82.0% | 0.819 |
| **Fine-tuned head** | `train` (20k rows/class) | **90.3%** | **0.903** |

Two comparisons in one table:

- **Head vs the few-shot baseline** (+10.5% accuracy / +0.107 macro-F1): fine-tuning
  wins decisively.
- **Head vs KNN on identical data** (+8.3% / +0.084): giving KNN 10× the reference
  vectors buys only ~2 points — the trained head extracts far more from the same
  files. (KNN also flatters itself on validation: 0.964 on `tune` vs 0.819 on `eval`,
  while the head generalizes, 0.984 → 0.903.)

*Where* the head wins is telling. Attack recall is essentially identical across all
three runs (~88% — every classifier misses the same subtle attacks). The head's
entire edge is on the normal class (recall 71.9% → 92.5%): **the same detection rate
with ~3.7× fewer false alarms** (7,659 → 2,055 misclassified normal windows).

### It's robust across seeds (but not every seed)

Head fine-tuning has run-to-run variance (random weight init, data-shuffle order,
dropout — all seed-driven). Across a 5-seed sweep, **every seed beat the few-shot KNN
baseline** (0.796):

| Seed | Macro-F1 | vs baseline |
|---|---|---|
| 0 *(default)* | 0.918 | +0.122 |
| 13 | 0.917 | +0.121 |
| 1 | 0.913 | +0.117 |
| 7 | 0.900 | +0.104 |
| 123 | 0.834 | +0.038 |

(The sweep predates the run in the Results table by a few worker releases — same seed,
same data, 0.918 then vs 0.903 now — so treat ±0.02 as normal variance across builds
as well as seeds.)

Most seeds land ~0.90–0.92; a few generalize less well (and one outlier, `seed=42`,
*underfit* this split at 0.746 — below baseline). So: fine-tuning reliably wins here, but
if a run underperforms, **rerun with a different `OMEGA_FT_SEED`**. Internal experiments
(Hasan Doğan) found a **smaller or MLP head** is more seed-robust; exposing head
architecture as a tunable is planned (today only `batch_size`, `n_epochs`, and the reader
window are configurable).

### What was wrong before (now fixed)

Earlier every run collapsed to a single `step_1` checkpoint at validation macro-F1
**0.333** (predict-one-class on balanced binary). Root cause: the fine-tune worker's
training **data loader had shuffling off**, so each epoch fed all of one class then all
the other, driving the head to collapse (the optimizer was also reset each epoch). Fixed
in `atai_core` #5512 (shuffle on) and deployed to dev. Throughout, KNN scored 0.67–0.80
on the same embeddings — confirming the signal was always there and this was a training
bug, not a data or task problem.

## How it works

- **Fine-tune** — via the canonical fine-tuning API `POST /v0.6/fine_tuning/jobs`
  (`model: "omega"`), the same OpenAI-style service the Console and the Newton example
  use. It produces an `ftj_…` job (visible on the Fine-Tuning page) and a `ckp_…`
  checkpoint. Two methods:
  - `method.type: "head"` (default) — the worker windows each labeled CSV, embeds the
    windows with the frozen Omega 1.5 encoder, trains a classification head, and saves
    it as the checkpoint. (Internal pipeline: `fine-tuning-omega`.)
  - `method.type: "knn"` — no gradient training: the worker embeds the same labeled
    training files once and saves the vectors as the checkpoint; inference re-fits a
    KNN from them. "Fine-tuning" here means growing KNN's reference set from the
    2k-row n-shot files to the full 20k-row train split. (Internal pipeline:
    `knn-fine-tuning-omega`; on Dev since 2026-07-15, rolling out to other envs.)

  Both evaluate on the `validation_files` and report `macro_f1`.
- **Inference** (`pipeline_type: batch`, `pipeline_key: machine-state-classification`):
  classifies each eval window. With no checkpoint it uses few-shot KNN over the
  `n_shots` reference computed inline (legacy path, kept for backward compatibility);
  attaching a `ckp_…` on the `fine_tune_checkpoint` port swaps in the checkpointed
  classifier — the trained head or the stored KNN — with window size, data columns,
  and KNN params taken from the checkpoint's metadata.

### How the splits map to job inputs

For the committed SWaT splits, the jobs resolve their inputs like this:

- **Fine-tune** (either method — `head` and `knn` take identical inputs) — window
  **128** (the KNN grid winner, reused so all classifiers see identical embeddings),
  **10 inputs**: 8 `train` files (4 normal + 4 attack) as `training_files`, and 2
  `tune` files (1 each) as `validation_files`.
- **Baseline** — KNN at the grid-winning config **w128 / k15 / cosine**, **14 inputs**:
  2 `nshot` files (1 each) on `worker.n_shots` as the few-shot reference, and 12 `eval`
  files (6 each) on `worker.inference` as the held-out set being classified.
- **Fine-tuned inference** (head or knn checkpoint) — the same 12 `eval` files, plus
  the checkpoint on `worker.fine_tune_checkpoint` (and the `n_shots` reference, which
  is ignored once a checkpoint is attached).

(Input counts follow directly from the split sizes and `--rows-per-file`; window 128 is
whatever the grid picks for the committed data.)

## Serving the fine-tuned model (how the "after" works)

The fine-tune produces a **checkpoint** (`ckp_…`). To serve it, run the *same*
`machine-state-classification` batch job as the baseline and attach the checkpoint on the
**`worker.fine_tune_checkpoint` input port**. The worker rebuilds the checkpointed
classifier — the trained head's weights, or a KNN re-fit from the stored vectors — and
classifies with it **instead of** building the few-shot KNN inline; same frozen Omega
1.5 encoder underneath, different classifier on top. Full request:

```json
POST /v0.5/batch/jobs
{
  "name": "swat-machine-state-fine-tuned",
  "pipeline_type": "batch",
  "pipeline_key": "machine-state-classification",
  "inputs": {
    "worker.inference":            [ { "file_id": "swat_normal_eval_0_g.csv" }, ... ],
    "worker.n_shots":              [ { "file_id": "swat_normal_nshot_0_g.csv", "metadata": {"class": "normal"} }, ... ],
    "worker.fine_tune_checkpoint": [ { "kind": "checkpoint", "checkpoint_id": "ckp_5kn596dy9v8wztr2fcqyhvrsa0", "metadata": {} } ]
  },
  "parameters": {
    "worker": {
      "parallelism": 1,
      "config": {
        "model_type": "omega_1_5_base",
        "omega_1_5": { "base_model_path": "s3://.../omega1.5_target_encoder.pt" },
        "reader_config": { "window_size": 128, "step_size": 1, "timestamp_column": "timestamp", "data_columns": [ ... ] }
      }
    }
  }
}
```

The **only** difference from a baseline job is the added `worker.fine_tune_checkpoint`
input — everything else (encoder, eval files, reader) is identical:

```diff
  "inputs": {
    "worker.inference":            [ ... ],
    "worker.n_shots":              [ ... ],
+   "worker.fine_tune_checkpoint": [ { "kind": "checkpoint", "checkpoint_id": "ckp_..." } ]
  }
```

`4_inference/create_inference_job.py` builds this automatically: with no `--baseline` it
reads `data/checkpoint_id.txt` (written by step 5) and attaches it on
`worker.fine_tune_checkpoint`; override with a positional arg
(`python 4_inference/create_inference_job.py <ckp_id>`).

**Picking the checkpoint.** A fine-tune saves several checkpoints — one each time
validation macro-F1 improves — so the **highest step is the best-validation one**.
`create_fine_tune_job.py` keeps that and writes it to `data/checkpoint_id.txt`.

> **⚠️ Two things that must match, or the "after" silently misleads you.**
> - **Use the `worker.fine_tune_checkpoint` port**, not a `config` field. When a
>   checkpoint is attached the worker ignores `classifier_config` and the `n_shots`
>   reference (they're the KNN path); the head replaces them. Pass them anyway — the port
>   is required — they're just inert.
> - **`window_size` must equal the fine-tune's training window** (here 128, recorded in
>   `data/finetune_window.txt`). The head only understands embeddings from the window it
>   was trained on; a mismatch feeds it the wrong-shaped input and quietly tanks accuracy.
>
> Unlike the Newton/fusion path, there's no `fine_tuned_model` handle to reference — you
> attach the `ckp_…` checkpoint id directly.

## Notes

- **Omega 1.5 only.** Fine-tuning supports `omega_1_5_base`; both the baseline KNN and
  the fine-tuned run use 1.5 so the comparison is apples-to-apples.
- The fine-tuned model is reachable via batch jobs (JOS), **not** the direct Query API.
- `FILE_SUFFIX` (in `.env`) appends a tag to file_ids so changed data can be
  re-uploaded — platform file_ids are immutable.

## One-time pipeline registration (internal — remove before making this repo public)

The `fine-tuning-omega` (fine-tune) and `machine-state-classification` (inference)
pipelines must be deployed and resolvable in your Dev org, or job creation fails with
`404 NOT_FOUND: Pipeline ... not found`. Both are now platform-deployed with published
versions that have the needed ports, so the scripts **auto-resolve** them — no manual
seeding or version pinning required.

If you ever hit a 404 (e.g. a version was replaced mid-deploy), you can pin an explicit
version: `MSJ_PIPELINE_VERSION` for inference, `OMEGA_FT_PIPELINE_VERSION` for fine-tune
(both commented out in `.env.example`). List what your org exposes with
`GET /v0.5/batch/registry/pipelines?status=published`. If a pipeline is missing
entirely, it needs (re-)seeding into your org by the platform team.
