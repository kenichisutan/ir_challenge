# IR Challenge Retrieval Repository

This repository contains our working pipelines, experiments, and submissions for the Scientific Information Retrieval challenge.

Goal: given a query paper, rank candidate papers so true cited references are as high as possible (primary metric: `NDCG@10`).

## Repository Structure

- `code/` — iteration scripts and retrieval pipelines.
- `data/` — challenge data files (queries/corpus/qrels) and embedding/model caches.
- `submissions/` — generated `submission_data.json` and zipped submission artifacts.
- `scripts/` — shared utilities (data loading, validation, packaging helpers).
- `notebooks/` — exploratory notebooks and offline analysis.
- `reports/` — benchmark reports and experiment notes.
- `checkpoints/` — optional model checkpoints from local training/tuning runs.

## How To Use

1. Pick a pipeline script from `code/`.
2. Run it with Python.
3. Upload the generated zip from `submissions/<iteration>/` to Codabench.

## Best Confirmed Result

- **Primary metric:** `NDCG@10`
- **Best Codabench submission:** `iteration_best.zip`
- **Scores:**
  - `NDCG@10: 0.7258`
  - `MAP: 0.6481`
  - `Recall@100: 0.9283`
- **Script:** `code/iteration_best.py`

## Active Pipelines

### 1) Foundation Hybrid

- **Script:** `code/iteration_foundation_hybrid.py`
- **Output folder:** `submissions/iteration_foundation_hybrid/`
- **Method:**
  1. TF-IDF sparse retrieval on `title + abstract + full_text`.
  2. Dense retrieval with MiniLM and SPECTER2 embeddings.
  3. Hybrid score interpolation with metadata priors (`domain`, `venue`, `year`).
  4. Public-set tuning on `qrels.json` for `NDCG@10`.
  5. Top-100 predictions for held-out queries.
- **Purpose:** baseline for stable comparisons and ablations.

### 2) Hybrid + SPECTER

- **Script:** `code/iteration_hybrid_specter.py`
- **Output folder:** `submissions/iteration_hybrid_specter/`
- **Method:**
  1. Hybrid sparse+dense retrieval.
  2. Explicit MiniLM + SPECTER2 dense fusion.
  3. Score interpolation with optional RRF contribution.
  4. Public tuning followed by held-out inference.
- **Purpose:** stronger semantic variant than the foundation baseline.

### 3) Best Pipeline (Current Champion)

- **Script:** `code/iteration_best.py`
- **Output folder:** `submissions/iteration_best/`
- **Method:**
  1. Two sparse channels:
     - TF-IDF on richer full text
     - TF-IDF on `title + abstract`
  2. Sparse-channel blending via `sparse_beta`.
  3. Dense fusion of MiniLM + SPECTER2 via `eta`.
  4. Calibrated interpolation (`alpha`) with lightweight metadata boosts.
  5. Tiny optional rank-fusion correction (`gamma_rrf`) when beneficial.
  6. Public-set tuning on `NDCG@10` and held-out top-100 generation.
- **Current best Codabench scores:**
  - `NDCG@10: 0.7258`
  - `MAP: 0.6481`
  - `Recall@100: 0.9283`

## Submission Workflow

1. Run one of the three scripts above.
2. Confirm `submission_data.json` and `<iteration>.zip` are generated under `submissions/<iteration>/`.
3. Upload the zip to Codabench.
4. Record official metrics here.
