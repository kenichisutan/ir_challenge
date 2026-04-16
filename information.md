# IR Challenge Iteration Notes

Quick reference of what each iteration does, why it was added, and what to try next.

### Leaderboard summary (Codabench)

| Iteration | NDCG@10 | MAP | Recall@100 | Notes |
|-----------|---------|-----|------------|--------|
| 1 — TF-IDF | 0.4536 | 0.3616 | 0.7325 | Strong lexical baseline |
| 2 — BM25 | 0.3726 | 0.2805 | 0.6015 | Regressed |
| 3 — TF-IDF + MiniLM + RRF | 0.5336 | 0.4367 | 0.8378 | Strong hybrid; superseded by Iteration 4 |
| 4 — rich text + tuned RRF + MiniLM | **0.5508** | **0.4558** | **0.8398** | **Current best on NDCG@10** |
| 5 — triple RRF + title-upweighted TF-IDF | 0.5305 | 0.4379 | 0.8460 | MAP/Recall up; **NDCG@10 regressed −0.0203** vs Iteration 4 |
| 6a — Iteration 4 + cross-encoder rerank | 0.3450 | 0.4218 | 0.8398 | Regressed; CE on top-100 hurt top-10 ordering |
| 6b — Iteration 4 + SPECTER2 leg (RRF×3) | 0.5213 | 0.4316 | 0.8441 | Better than 6a, still below Iteration 4 NDCG@10 |
| 7 — sparse PRF TF-IDF (new) | 0.4682 | 0.3813 | 0.7574 | Regressed strongly vs Iteration 4 |
| 8 — dense-only SPECTER2 (new) | 0.3341 | 0.4122 | 0.7221 | Worst NDCG so far; pure dense underfits lexical matching |
| 9 — hybrid + CE top-50 blend (new) | 0.4378 | 0.3643 | 0.8398 | CE blend still hurts top-10 |
| 10 — weighted hybrid fusion (new) | 0.5435 | 0.4511 | 0.8378 | Best since Iteration 4; weighted fusion helps |
| 4.1 — team feedback stack (BM25 chunks + SPECTER2 + CE) | ~0.5400 | — | — | User-reported NDCG@10 only; add MAP/R@100 from Codabench log |
| 4.2 — BM25 rich + MiniLM rich (weighted RRF) | *pending* | *pending* | *pending* | Aligned sparse/dense text with Iteration 4; no CE |

Primary metric for ranking: **NDCG@10**. Ship **`iteration_4.zip`** until another zip beats **0.5508** on that metric.

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
- **Outcome:** best confirmed through Iteration 4 (+0.0172 NDCG@10 vs Iteration 3). Remains the **NDCG@10** leader after Iteration 5 regressed.

## Iteration 5

- **Method:** **Triple RRF** (three retrieval streams, depth 350, `RRF_K=50`):
  1. **TF-IDF** on rich text with **title doubled** (`title title abstract … +` truncated `full_text`, max **2500** chars).
  2. **MiniLM dense (rich):** same rich strings for queries + full corpus re-encode (aligned dense space).
  3. **MiniLM dense (TA):** title+abstract query encodes + **precomputed** TA corpus embeddings (Iteration 3–style leg; no second full-corpus encode).
- **Output files:**
  - `submissions/iteration_5/submission_data.json`
  - `submissions/iteration_5/iteration_5.zip`
- **Script:** `code/iteration_5.py`
- **Codabench scores:**
  - `NDCG@10: 0.5305` (worse than Iteration 4’s **0.5508**)
  - `MAP: 0.4379`
  - `Recall@100: 0.8460`
- **Outcome:** adding a third MiniLM (TA) list + title-doubling hurt **top-10** ordering despite slightly better MAP/Recall; prefer Iteration 4 for the leaderboard until 6a/6b.

## Iteration 6a (prototype)

- **Method:** Same retrieval as **Iteration 4** (rich TF-IDF + rich MiniLM + RRF → top-100), then **cross-encoder** reranking on the full top-100 using `cross-encoder/ms-marco-MiniLM-L-6-v2` (Class 4 notebook pattern). Query–doc pairs use the same **rich** strings as Iteration 4. Candidate indices come from `iteration_4.compute_iter4_fused_indices` so fusion matches Iteration 4 exactly.
- **Output:** `submissions/iteration_6a/submission_data.json`, `iteration_6a.zip`
- **Script:** `code/iteration_6a.py`
- **Local status:** submission zip generated successfully (held-out queries).
- **Codabench scores:**
  - `NDCG@10: 0.3450`
  - `MAP: 0.4218`
  - `Recall@100: 0.8398`
- **Outcome:** this configuration underperformed strongly on top-10 relevance.

## Iteration 6b (prototype)

- **Method:** **Triple RRF**: Iteration 4’s TF-IDF (rich) + MiniLM dense (rich) + **SPECTER2** (`allenai/specter2_base`) dense on **title `[SEP]` abstract** (tokenizer `sep_token`; matches common SPECTER/SPECTER2 usage). Uses same `FUSION_DEPTH=300` and `RRF_K=40` as Iteration 4. SPECTER2 **corpus embeddings are cached** under `data/embeddings/allenai_specter2_base/` after first run (delete cache if `corpus.parquet` order changes).
- **Output:** `submissions/iteration_6b/submission_data.json`, `iteration_6b.zip`
- **Script:** `code/iteration_6b.py`
- **Local status:** submission zip generated successfully; first run builds ~11 min SPECTER2 corpus encode on CPU (then loads from cache).
- **Codabench scores:**
  - `NDCG@10: 0.5213`
  - `MAP: 0.4316`
  - `Recall@100: 0.8441`
- **Outcome:** improved over 6a but still behind Iteration 4 on NDCG@10.

## Iteration 7 (prototype, new)

- **Method:** sparse-only retrieval with **Rocchio-style pseudo-relevance feedback** on TF-IDF rich text (first-pass top-30, then expanded query vector).
- **Output:** `submissions/iteration_7/submission_data.json`, `iteration_7.zip`
- **Script:** `code/iteration_7.py`
- **Local status:** submission zip generated successfully.
- **Codabench scores:**
  - `NDCG@10: 0.4682`
  - `MAP: 0.3813`
  - `Recall@100: 0.7574`
- **Outcome:** PRF expansion over-amplified lexical drift on this benchmark.

## Iteration 8 (prototype, new)

- **Method:** **dense-only SPECTER2** retrieval on `title [SEP] abstract` (no sparse leg, no fusion), cosine via normalized dot product.
- **Output:** `submissions/iteration_8/submission_data.json`, `iteration_8.zip`
- **Script:** `code/iteration_8.py`
- **Local status:** submission zip generated successfully; reuses cached SPECTER2 corpus embeddings.
- **Codabench scores:**
  - `NDCG@10: 0.3341`
  - `MAP: 0.4122`
  - `Recall@100: 0.7221`
- **Outcome:** dense-only SPECTER2 is not competitive here without sparse support.

## Iteration 9 (prototype, new)

- **Method:** Iteration 4 candidate generation, then **rerank only top-50** with cross-encoder and **blend CE score with base rank prior** (`lambda=0.35`) to reduce destructive reordering.
- **Output:** `submissions/iteration_9/submission_data.json`, `iteration_9.zip`
- **Script:** `code/iteration_9.py`
- **Local status:** submission zip generated successfully.
- **Codabench scores:**
  - `NDCG@10: 0.4378`
  - `MAP: 0.3643`
  - `Recall@100: 0.8398`
- **Outcome:** limiting CE rerank depth and blending did not recover top-10 quality.

## Iteration 10 (prototype, new)

- **Method:** weighted hybrid fusion with **three legs**:
  1. TF-IDF rich
  2. MiniLM dense rich
  3. SPECTER2 dense TA
  using **weighted RRF** (`k=40`, depth `300`) with weights `(0.45, 0.40, 0.15)` to keep SPECTER2 as a supporting signal.
- **Output:** `submissions/iteration_10/submission_data.json`, `iteration_10.zip`
- **Script:** `code/iteration_10.py`
- **Local status:** submission zip generated successfully.
- **Codabench scores:**
  - `NDCG@10: 0.5435`
  - `MAP: 0.4511`
  - `Recall@100: 0.8378`
- **Outcome:** clear gain over Iterations 6–9 and very close to Iteration 4; weighted fusion appears to be the right direction.

## Iteration 4.1 (team feedback prototype)

- **Method:** combines several notebook-style upgrades at once: **BM25** over **full-text chunks** (max-pooled to documents), **SPECTER2** dense (title `[SEP]` abstract), **hybrid** via min–max **score interpolation** plus **RRF**, then **cross-encoder** reranking with **paragraph-level max** scores (top-30), blended with the hybrid prior.
- **Caveat:** sparse query text for BM25 is title+abstract only while corpus uses chunks + TA head (see script). Dense is document-level SPECTER2, not chunk-level embeddings.
- **Output:** `submissions/iteration_4_1/submission_data.json`, `iteration_4_1.zip`
- **Script:** `code/iteration_4_1.py`
- **Codabench (user-reported):** NDCG@10 ≈ **0.54** (did not beat Iteration 4); fill full triple from leaderboard when available.

## Iteration 4.2 (Iteration 4 family, sparse upgrade)

- **Method:** same **rich** strings as Iteration 4 for queries and corpus (`format_rich_text`, 2000-char `full_text`). Replace TF-IDF with **BM25Okapi** (`k1=1.2`, `b=0.75`). Keep **MiniLM** rich dense identical to Iteration 4. Fuse with **weighted RRF** only (`weights=0.52/0.48`, `k=40`, depth `300`) — **no** cross-encoder and **no** third dense list (avoids known CE / extra-list regressions).
- **Output:** `submissions/iteration_4_2/submission_data.json`, `iteration_4_2.zip`
- **Script:** `code/iteration_4_2.py`
- **Codabench scores:** *(after upload)*

## Benchmark + RAG utility (new)

- **Script:** `scripts/benchmark_rag.py`
- **Purpose:** computes MAP/Recall@100/NDCG@10 against `data/qrels.json` and writes a markdown report with a lightweight extractive RAG context view.
- **Report example:** `reports/benchmark_rag_iteration4.md`

## Future: bi-encoder fine-tuning (optional)

Not implemented as a script (time / GPU / risk of overfitting the public 100 queries). Sketch:

1. Use [`data/qrels.json`](data/qrels.json) on **public** `queries.parquet` only — positive = cited `doc_id`; negatives = in-batch random or BM25 top-k non-relevant.
2. **Train/val split** the 80/20 query IDs; never tune on held-out leaderboard queries.
3. `sentence_transformers` training with **MultipleNegativesRankingLoss** (or similar) on `(query_text, positive_doc_text)` pairs.
4. Export new embeddings or save a fine-tuned model folder; then plug into the same RRF pipeline as another dense leg.

---

## Ideas for Iteration 7+ (brainstorm)

Iteration **6a/6b** implement cross-encoder rerank and SPECTER2 RRF (see above). Further knobs:

Ordered roughly by effort vs likely impact on **NDCG@10**:

1. **More RRF / fusion tuning**
   - Tune `k` / depth after 6b scores; weighted RRF or score interpolation if needed.

2. **Text / field tweaks**
   - Adjust `MAX_FULLTEXT_CHARS`; title-only sparse channel; avoid piling on redundant dense lists (lesson from Iteration 5).

3. **Stronger or extra dense encoders**
   - If 6b helps, try SPECTER2 **rich** text (re-encode corpus) or a fourth list — watch runtime.

4. **Cross-encoder variants**
   - If 6a hurts, try rerank **top 50** only, blend CE score with RRF rank, or a more domain-matched CE.

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
