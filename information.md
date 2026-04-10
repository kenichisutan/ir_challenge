# IR Challenge Iteration Notes

Quick reference of what each iteration does, why it was added, and what to try next.

### Leaderboard summary (Codabench)

| Iteration | NDCG@10 | MAP | Recall@100 | Notes |
|-----------|---------|-----|------------|--------|
| 1 — TF-IDF | 0.4536 | 0.3616 | 0.7325 | Strong lexical baseline |
| 2 — BM25 | 0.3726 | 0.2805 | 0.6015 | Regressed |
| 3 — TF-IDF + MiniLM + RRF | **0.5336** | **0.4367** | **0.8378** | **Current best** |

Primary metric for ranking: **NDCG@10**.

## Iteration 1

- **Method:** TF-IDF baseline (`title + abstract`) with cosine similarity ranking.
- **Prediction size:** Top-100 corpus documents per query.
- **Output files:**
  - `submissions/iteration_1/submission_data.json`
  - `submissions/iteration_1/iteration_1.zip`
- **Script:** `code/iteration_1.py`
- **Shared utilities:** `scripts/submission_utils.py`

### Implementation Details

- Uses deterministic ranking: `np.argsort(..., kind="stable")`.
- Submission validation checks:
  - every query has an entry,
  - each query has exactly 100 doc IDs,
  - all doc IDs are strings,
  - no duplicate doc IDs inside a query ranking,
  - all predicted IDs exist in `data/corpus.parquet`.
- Query source selection:
  - prefers held-out queries from one of:
    - `data/held_out_queries.parquet`
    - `held_out_queries.parquet`
    - `starter_kit/held_out_queries.parquet`
  - fails fast if held-out queries are not found.
  - submission keys must match held-out `doc_id` values exactly.

### Why This Iteration

- Establishes a clean and reproducible baseline.
- Creates a stable foundation for comparing later improvements (BM25, dense retrieval, hybrid fusion).

### Codabench Scores

- `NDCG@10: 0.4536`
- `MAP: 0.3616`
- `Recall@100: 0.7325`

### Status vs later work

- Iteration 2 (BM25 alone) regressed; hybrid in Iteration 3 is the direction to extend.

## Iteration 2

- **Method:** BM25 (`title + abstract`) with `k1=1.2`, `b=0.75`.
- **Prediction size:** Top-100 corpus documents per query.
- **Output files:**
  - `submissions/iteration_2/submission_data.json`
  - `submissions/iteration_2/iteration_2.zip`
- **Script:** `code/iteration_2.py`
- **Why this change:** stronger lexical ranking than plain TF-IDF baseline.
- **Codabench scores:**
  - `NDCG@10: 0.3726`
  - `MAP: 0.2805`
  - `Recall@100: 0.6015`
- **Outcome:** significantly worse than Iteration 1; do not use as current best submission.

## Iteration 3

- **Method:** Hybrid — Iteration 1–style TF-IDF top-200 + MiniLM dense top-200, fused with **RRF** (`k=60`), final top-100.
- **Dense queries:** Held-out IDs are not in precomputed `query_embeddings.npy`, so queries are encoded at runtime with `sentence-transformers/all-MiniLM-L6-v2` (same as `scripts/embed.py`). Corpus uses `data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_embeddings.npy`.
- **Prediction size:** Top-100 corpus documents per query.
- **Output files:**
  - `submissions/iteration_3/submission_data.json`
  - `submissions/iteration_3/iteration_3.zip`
- **Script:** `code/iteration_3.py`
- **Codabench scores:**
  - `NDCG@10: 0.5336`
  - `MAP: 0.4367`
  - `Recall@100: 0.8378`
- **Outcome:** best so far on all three metrics; use `iteration_3.zip` as the reference submission until a later iteration beats it.

---

## Ideas for Iteration 4+ (brainstorm)

Ordered roughly by effort vs likely impact on **NDCG@10**:

1. **Tune the hybrid you already have**
   - RRF constant `k` (e.g. 20, 40, 80, 100).
   - Fusion depth (e.g. top-300 or top-500 from each retriever before RRF) to surface more overlap candidates.
   - Weighted RRF or score interpolation (normalize TF-IDF cosine and dense dot product, then blend) if RRF plateaus.

2. **Richer text for sparse and/or dense**
   - Add a **truncated `full_text`** prefix to query and corpus strings (same cap for both sides) so citations and methods appear without blowing up length.
   - **Title upweighting** for sparse (repeat title tokens or separate title field BM25 — careful with BM25 alone; hybrid may tolerate it better).

3. **Stronger dense encoder**
   - **SPECTER / SPECTER2** (citation-trained) for scientific papers — replace or add as a third list in RRF (TF-IDF + MiniLM + SPECTER) if runtime and deps are acceptable.
   - Precompute corpus embeddings for the new model once; keep runtime encoding only for held-out queries.

4. **Second-stage reranking (expensive)**
   - Retrieve top-100 with current hybrid, rerank top-20 with a **cross-encoder** (query paper vs candidate title+abstract). Risky if the CE is not domain-tuned; try on a dev split if you later get labels.

5. **Query expansion (sparse side)**
   - PRF / pseudo-relevance from top-k TF-IDF docs to reformulate the query text before a second retrieval pass, then fuse with dense.

Pick 1–2 items per submission day to stay within submission limits and keep causes of gains attributable.

---

## Template for New Iterations

Copy this section for each new iteration:

### Iteration N

- **Method:**
- **Script:**
- **Output files:**
- **Key parameters:**
- **Why this change:**
- **Local metrics:**
- **Codabench feedback:**
- **Next hypothesis:**

## Submission Workflow (Challenge-Aligned)

1. Ensure `held_out_queries.parquet` exists in one accepted location.
2. Run iteration script (e.g. `python code/iteration_1.py`).
3. Confirm output includes:
   - `Query source: ...held_out_queries.parquet`
   - `Saved to: submissions/<iteration>/submission_data.json`
   - `Zipped to: submissions/<iteration>/<iteration>.zip`
4. Upload the zip file to Codabench.
