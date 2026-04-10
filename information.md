# IR Challenge Iteration Notes

Quick reference of what each iteration does, why it was added, and what to try next.

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
  - validates query keys against `data/sample_submission.json`.

### Why This Iteration

- Establishes a clean and reproducible baseline.
- Creates a stable foundation for comparing later improvements (BM25, dense retrieval, hybrid fusion).

### Codabench Scores

- `NDCG@10: 0.4536`
- `MAP: 0.3616`
- `Recall@100: 0.7325`

### Next Iterations (Planned)

- **Iteration 2:** BM25 sparse retrieval baseline.
- **Iteration 3:** Dense retrieval with MiniLM embeddings.
- **Iteration 4:** Hybrid retrieval (e.g., RRF of sparse + dense).

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
