# IR Challenge Iteration Notes

Quick reference of what each iteration does, why it was added, and what to try next.

### Leaderboard summary (Codabench)

| Iteration | NDCG@10 | MAP | Recall@100 | Notes |
|-----------|---------|-----|------------|--------|
| 1 — TF-IDF | 0.4536 | 0.3616 | 0.7325 | Strong lexical baseline |
| 2 — BM25 | 0.3726 | 0.2805 | 0.6015 | Regressed |
| 3 — TF-IDF + MiniLM + RRF | 0.5336 | 0.4367 | 0.8378 | Strong hybrid; superseded by Iteration 4 |
| 4 — rich text + tuned RRF + MiniLM | 0.5508 | 0.4558 | 0.8398 | Previous best |
| 5 — triple RRF + title-upweighted TF-IDF | *pending* | *pending* | *pending* | Upload `iteration_5.zip` for scores |

Primary metric for ranking: **NDCG@10**. Use `iteration_4.zip` as fallback until Iteration 5 is scored; then promote the better of the two.

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
- **Outcome:** strong on all metrics; superseded by Iteration 4. Keep `iteration_3.zip` as a lighter-weight fallback (no full-corpus re-encode).

## Iteration 4

- **Method:** Same hybrid family as Iteration 3, plus **both** of the following at once:
  - **RRF tuning:** `RRF_K=40`, `FUSION_DEPTH=300` (vs Iteration 3’s 60 / 200).
  - **Rich text:** `title + abstract +` first **2000** characters of `full_text` (whitespace-normalized) for **queries and corpus** on both TF-IDF and dense paths.
- **Dense corpus:** Precomputed MiniLM vectors are title+abstract-only, so this iteration **re-encodes all 20k corpus documents** at runtime with MiniLM for a consistent dense space (expect ~1–2 minutes on CPU for corpus batches).
- **Output files:**
  - `submissions/iteration_4/submission_data.json`
  - `submissions/iteration_4/iteration_4.zip`
- **Script:** `code/iteration_4.py`
- **Codabench scores:**
  - `NDCG@10: 0.5508`
  - `MAP: 0.4558`
  - `Recall@100: 0.8398`
- **Outcome:** best confirmed through Iteration 4 (+0.0172 NDCG@10 vs Iteration 3). May be superseded by Iteration 5 after Codabench.

## Iteration 5

- **Method:** **Triple RRF** (three retrieval streams, depth 350, `RRF_K=50`):
  1. **TF-IDF** on rich text with **title doubled** (`title title abstract … +` truncated `full_text`, max **2500** chars).
  2. **MiniLM dense (rich):** same rich strings for queries + full corpus re-encode (aligned dense space).
  3. **MiniLM dense (TA):** title+abstract query encodes + **precomputed** TA corpus embeddings (Iteration 3–style leg; no second full-corpus encode).
- **Output files:**
  - `submissions/iteration_5/submission_data.json`
  - `submissions/iteration_5/iteration_5.zip`
- **Script:** `code/iteration_5.py`
- **Codabench scores:** *(fill in after upload)*

---

## Ideas for Iteration 6+ (brainstorm)

Iteration 4–5 push hybrid + text + multi-list RRF. Next knobs:

Ordered roughly by effort vs likely impact on **NDCG@10**:

1. **More RRF / fusion tuning**
   - Try other `k` (e.g. 20, 40, 80) and depths (400–500) if Iteration 5 underperforms.
   - Weighted RRF or score interpolation (normalize TF-IDF cosine and dense dot product, then blend) if RRF plateaus.

2. **Text / field tweaks**
   - Iteration 5 already doubles the title in the sparse string; try 3× title or a separate title-only TF-IDF channel.
   - Adjust `MAX_FULLTEXT_CHARS` (e.g. 1000 vs 3500) or use intro paragraphs only.

3. **Stronger dense encoder**
   - **SPECTER / SPECTER2** (citation-trained) — add as a **fourth** RRF list or replace the TA MiniLM leg if runtime and deps are acceptable.
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
