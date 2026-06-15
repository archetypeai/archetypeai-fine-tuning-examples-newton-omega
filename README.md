# Fine-Tuning Omega on Archetype AI — a worked example

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
  head learns. (The platform labels the job `omega-fine-tune-job`; the encoder itself
  is not retrained, which is also why it's fast and cheap.)
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
- **Z-scoring** — channels are standardized using normal-operation statistics, so an
  attack shows up as a deviation from normal.
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

Run step 7 to produce the comparison (macro-F1, preferred over accuracy on imbalanced
data, plus a confusion matrix per model):

```
  Baseline accuracy:   <fill in>
  Fine-tuned accuracy: <fill in>
  Delta:               <fill in>
```

## How it works

- **Fine-tune** (`pipeline_type: training`, `pipeline_key: omega-fine-tune-job`):
  windows each labeled CSV, embeds the windows with the frozen Omega 1.5 encoder, and
  trains a classification head, saving it as a platform checkpoint.
- **Inference** (`pipeline_type: batch`, `pipeline_key: machine-state-classification`):
  classifies each eval window. With no checkpoint it uses few-shot KNN over the
  `n_shots` reference; attaching the fine-tuned checkpoint on the
  `fine_tune_checkpoint` port swaps in the trained head instead.

## Notes

- **Omega 1.5 only.** Fine-tuning supports `omega_1_5_base`; both the baseline KNN and
  the fine-tuned run use 1.5 so the comparison is apples-to-apples.
- The fine-tuned model is reachable via batch jobs (JOS), **not** the direct Query API.
- `FILE_SUFFIX` (in `.env`) appends a tag to file_ids so changed data can be
  re-uploaded — platform file_ids are immutable.

## One-time pipeline registration (internal — remove before making this repo public)

The `omega-fine-tune-job` and `machine-state-classification` pipelines must have a
registered version in the Dev org, or job creation fails with
`404 NOT_FOUND: Pipeline ... has no active versions`. Registration uses internal seed
modules from `atai_core`:

```bash
cd <atai_core>/jobs/machine_state_job && uv sync
cp env_settings/register_omega_fine_tune_job_and_launch_test_run_env_vars_template.ini \
   env_settings/register_omega_fine_tune_job_and_launch_test_run_env_vars.ini   # set PLATFORM_API_KEY
make register_omega_fine_tune_job_and_launch_test_run
# repeat the template-copy + make for register_msj_and_launch_test_run
```

Seeded versions register as `status=draft`; job creation auto-resolves only *published*
versions. `machine-state-classification` in Dev has an old published version without
the `fine_tune_checkpoint` port, so the inference/grid scripts pin the seeded draft via
`MSJ_PIPELINE_VERSION` in `.env` (list drafts with
`GET /v0.5/batch/registry/pipelines?status=draft`). Drop it once a version with the
port is published.
