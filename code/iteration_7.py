from __future__ import annotations

"""
Iteration 7 — sparse-only with pseudo-relevance feedback (Rocchio-style TF-IDF).

Untried direction: keep retrieval fully sparse, but expand each query vector using
its top lexical hits before final ranking.
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

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

ITERATION_NAME = "iteration_7"
TOP_K = DEFAULT_TOP_K
FIRST_PASS_DEPTH = 30
ALPHA = 1.0
BETA = 0.35


def rank_sparse_with_prf(
    query_texts: List[str],
    corpus_texts: List[str],
    top_k: int,
    first_pass_depth: int,
    alpha: float,
    beta: float,
) -> np.ndarray:
    vec = TfidfVectorizer(
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X = vec.fit_transform(corpus_texts).tocsr()
    Q = vec.transform(query_texts).tocsr()

    first_scores = Q @ X.T
    first_ranks = np.argsort(-first_scores.toarray(), axis=1, kind="stable")[:, :first_pass_depth]

    n_queries = Q.shape[0]
    all_ranks = np.zeros((n_queries, top_k), dtype=np.int64)
    for i in range(n_queries):
        q0 = Q.getrow(i)
        fb_idx = first_ranks[i]
        fb_centroid = X[fb_idx].mean(axis=0)
        fb_centroid = sparse.csr_matrix(fb_centroid)
        q_expanded = (q0.multiply(alpha) + fb_centroid.multiply(beta)).tocsr()
        scores = (q_expanded @ X.T).toarray().ravel()
        all_ranks[i] = np.argsort(-scores, kind="stable")[:top_k]

    return all_ranks


def main() -> None:
    data = data_paths()
    if not data["using_held_out_queries"]:
        raise RuntimeError(
            "Submission mode requires held-out queries. "
            "Add `held_out_queries.parquet` to `data/`, project root, or `starter_kit/`."
        )

    paths = {**data, **iteration_submission_paths(ITERATION_NAME)}
    queries, corpus = load_queries_corpus(paths["queries_path"], paths["corpus_path"])

    query_texts = queries.apply(lambda r: i4.format_rich_text(r, i4.MAX_FULLTEXT_CHARS), axis=1).tolist()
    corpus_texts = corpus.apply(lambda r: i4.format_rich_text(r, i4.MAX_FULLTEXT_CHARS), axis=1).tolist()

    ranks = rank_sparse_with_prf(
        query_texts,
        corpus_texts,
        top_k=TOP_K,
        first_pass_depth=FIRST_PASS_DEPTH,
        alpha=ALPHA,
        beta=BETA,
    )

    corpus_ids = corpus["doc_id"].astype(str).tolist()
    query_ids = queries["doc_id"].astype(str).tolist()
    submission: Dict[str, List[str]] = {}
    for i, qid in enumerate(query_ids):
        submission[qid] = [corpus_ids[int(j)] for j in ranks[i]]

    validate_submission(submission=submission, expected_query_ids=query_ids, top_k=TOP_K)
    validate_doc_ids_in_corpus(submission=submission, corpus_doc_ids=corpus_ids)
    save_submission(submission=submission, output_file=paths["output_file"])
    create_submission_zip(submission_file=paths["output_file"], zip_file=paths["zip_file"])

    print("Iteration 7 submission generated.")
    print(f"Method: sparse TF-IDF + PRF (depth={FIRST_PASS_DEPTH}, alpha={ALPHA}, beta={BETA})")
    print(f"Saved to: {paths['zip_file']}")


if __name__ == "__main__":
    main()
