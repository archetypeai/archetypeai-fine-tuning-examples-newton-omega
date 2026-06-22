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
  embeddings and classifies by nearest-neighbour vote. Fine-tuning does *not* fine-tune
  KNN; the trained head *replaces* it.
- **The "head" is a small neural network** that maps an Omega embedding to a class
  score. It has its own learnable weights, trained on your labeled data.
- **The Omega encoder stays frozen in both cases** — its weights never change. Only the
  head learns. (Fine-tuning runs through `model: "omega"` on the canonical FT API; the
  encoder itself is not retrained, which is also why it's fast and cheap.)
- **Why the head can beat KNN:** nearest-neighbour can only draw boundaries from raw
  distance in the frozen embedding space; a trained head can learn which embedding
  dimensions actually distinguish an attack, expressing boundaries KNN cannot.

This example gives the KNN baseline its best shot (a hyperparameter grid search) and
compares it head-to-head against the fine-tuned model: **does fine-tuning a head do
better, on the same data?**

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
| `eval` | 28,000 | held-out test set, scored once for both models |

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

On the held-out `eval` set (balanced, ~54.5k windows), Omega 1.5:

| Model | Accuracy | Macro-F1 | Behavior |
|---|---|---|---|
| **Baseline** (tuned few-shot KNN, w128/k15/cosine) | **79.7%** | **0.796** | catches 87.6% of attacks; some false alarms on normal |
| **Fine-tuned** (Omega head) | 50.0% | 0.333 | **collapsed — predicts `attack` for everything** |

So on this run the fine-tuned head **loses to the baseline by ~30 points**. This is
*not* a result about Omega embeddings or the task — tuned KNN reaches 0.796 on the very
same embeddings, so the signal is clearly there. It's a training failure in the
fine-tune worker; see below.

### Known limitation: the fine-tune head collapses

Every fine-tune run produces only a single `step_1` checkpoint with validation
macro-F1 ≈ 0.333 — exactly the score of predicting one class for everything on a
balanced binary set. What's happening:

- The worker saves a checkpoint **only when validation macro-F1 improves**. `step_1`
  alone means validation peaked at epoch 1 and never improved across all 30 epochs —
  the head collapsed to a single-class prediction immediately and never recovered.
- The training engine uses **AdamW at a fixed `learning_rate = 1e-3`, no schedule, no
  warmup, `balance_classes = False`**, on a transformer head. That LR is too aggressive
  for this head: the logits saturate on the first pass, gradients vanish, and the model
  is stuck. None of these knobs are exposed in the job config (only `batch_size`,
  `epochs`, and the window are).
- The collapse reproduced **identically across two unrelated datasets** (TEP and SWaT)
  **and both normalization schemes** (normal-only z-scoring with 500σ extremes, and
  global z-scoring with values capped near 16σ) — while KNN on the same embeddings
  scores 0.67–0.80. That rules out the data, the task, and the input scale, leaving the
  optimizer as the sole cause.

**The fix is worker-side** (lower learning rate / add warmup + LR schedule / enable
class balancing in `omega-fine-tune-job`) and is not addressable from this example
alone. Until then, this repo demonstrates the full pipeline end-to-end but does not yet
show fine-tuning beating the baseline.

## How it works

- **Fine-tune** — via the canonical fine-tuning API `POST /v0.6/fine_tuning/jobs`
  (`model: "omega"`, `method.type: "head"`), the same OpenAI-style service the Console
  and the Newton example use. It produces an `ftj_…` job (visible on the Fine-Tuning
  page); the worker windows each labeled CSV, embeds the windows with the frozen Omega
  1.5 encoder, trains a classification head, and saves it as a `ckp_…` checkpoint.
  (Under the hood the platform backs `model: "omega"` onto the internal
  `fine-tuning-omega` pipeline.)
- **Inference** (`pipeline_type: batch`, `pipeline_key: machine-state-classification`):
  classifies each eval window. With no checkpoint it uses few-shot KNN over the
  `n_shots` reference; attaching the fine-tuned `ckp_…` on the `fine_tune_checkpoint`
  port swaps in the trained head instead.

### How the splits map to job inputs

For the committed SWaT splits, the jobs resolve their inputs like this:

- **Fine-tune** — window **128** (the KNN grid winner, reused so both classifiers see
  identical embeddings), **10 inputs**: 8 `train` files (4 normal + 4 attack) on
  `worker.train`, and 2 `tune` files (1 each) on `worker.test` (validation).
- **Baseline** — KNN at the grid-winning config **w128 / k15 / cosine**, **14 inputs**:
  2 `nshot` files (1 each) on `worker.n_shots` as the few-shot reference, and 12 `eval`
  files (6 each) on `worker.inference` as the held-out set being classified.
- **Fine-tuned inference** — the same 12 `eval` files, plus the trained checkpoint on
  `worker.fine_tune_checkpoint` (and the `n_shots` reference, which is ignored once a
  checkpoint is attached).

(Input counts follow directly from the split sizes and `--rows-per-file`; window 128 is
whatever the grid picks for the committed data.)

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
