from __future__ import annotations

"""
Iteration 6a — Iteration 4 retrieve, then cross-encoder rerank (Class 4 notebook pattern).

Pipeline:
1. Same TF-IDF + MiniLM rich + RRF as Iteration 4 → top-100 candidate indices per query.
2. For each query, score 100 (query, document) pairs with
   `cross-encoder/ms-marco-MiniLM-L-6-v2`, reorder by CE score (stable sort).

Uses the same rich query/corpus strings as Iteration 4 for both stages.
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

# Same-directory import (run as `python code/iteration_6a.py` from `ir_challenge/`).
_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

import iteration_4 as i4

from submission_utils import (
    DEFAULT_TOP_K,
    create_submission_zip,
    data_paths,
    iteration_submission_paths,
    load_queries_corpus,
    save_submission,
    validate_doc_ids_in_corpus,
    validate_submission,
)

ITERATION_NAME = "iteration_6a"
TOP_K = DEFAULT_TOP_K
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_BATCH_SIZE = 32


def rerank_with_cross_encoder(
    query_texts: List[str],
    corpus_texts: List[str],
    candidate_indices: np.ndarray,
    ce: CrossEncoder,
) -> np.ndarray:
    """
    candidate_indices: (n_queries, TOP_K) corpus row indices.
    Returns same shape, reordered by CE score within each row.
    """
    nq, k = candidate_indices.shape
    out = np.zeros_like(candidate_indices)
    for i in range(nq):
        q = query_texts[i]
        idx_row = candidate_indices[i]
        pairs: List[List[str]] = []
        for j in range(k):
            doc_i = int(idx_row[j])
            pairs.append([q, corpus_texts[doc_i]])
        scores = ce.predict(pairs, batch_size=CE_BATCH_SIZE, show_progress_bar=False)
        order = np.argsort(-np.asarray(scores), kind="stable")
        out[i] = idx_row[order]
    return out


def main() -> None:
    data = data_paths()
    if not data["using_held_out_queries"]:
        raise RuntimeError(
            "Submission mode requires held-out queries. "
            "Add `held_out_queries.parquet` to `data/`, project root, or `starter_kit/`."
        )

    paths = {**data, **iteration_submission_paths(ITERATION_NAME)}
    queries, corpus = load_queries_corpus(paths["queries_path"], paths["corpus_path"])
    parquet_ids = corpus["doc_id"].astype(str).tolist()

    query_texts = queries.apply(lambda r: i4.format_rich_text(r, i4.MAX_FULLTEXT_CHARS), axis=1).tolist()
    corpus_texts = corpus.apply(lambda r: i4.format_rich_text(r, i4.MAX_FULLTEXT_CHARS), axis=1).tolist()

    print(f"Loading {i4.MINILM_MODEL} …")
    bi = SentenceTransformer(i4.MINILM_MODEL)

    print("Iteration 4 hybrid — candidate indices …")
    cand = i4.compute_iter4_fused_indices(
        queries, corpus, bi, top_k=TOP_K, fusion_depth=i4.FUSION_DEPTH
    )

    print(f"Loading cross-encoder {CE_MODEL} …")
    ce = CrossEncoder(CE_MODEL)

    print("Cross-encoder reranking (100 pairs × 100 queries) …")
    reranked = rerank_with_cross_encoder(query_texts, corpus_texts, cand, ce)

    submission: Dict[str, List[str]] = {}
    qids = queries["doc_id"].astype(str).tolist()
    for i, qid in enumerate(qids):
        submission[qid] = [parquet_ids[int(j)] for j in reranked[i]]

    validate_submission(submission=submission, expected_query_ids=qids, top_k=TOP_K)
    validate_doc_ids_in_corpus(submission=submission, corpus_doc_ids=parquet_ids)
    save_submission(submission=submission, output_file=paths["output_file"])
    create_submission_zip(submission_file=paths["output_file"], zip_file=paths["zip_file"])

    print("Iteration 6a submission generated.")
    print(f"Queries: {len(queries)} | Corpus: {len(corpus)} | Top-K: {TOP_K}")
    print(f"Method: Iteration 4 + CE rerank ({CE_MODEL})")
    print(f"Query source: {paths['queries_source']}")
    print(f"Saved to: {paths['output_file']}")
    print(f"Zipped to: {paths['zip_file']}")


if __name__ == "__main__":
    main()
