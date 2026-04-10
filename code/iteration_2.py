from __future__ import annotations

"""
Iteration 2 Notes (presentation-ready context)
=============================================
What we changed:
- Replaced TF-IDF ranking with BM25 ranking on `title + abstract`.
- Kept the same strict submission validation/export pipeline from Iteration 1.

Why this may improve:
- BM25 generally handles lexical matching better than plain TF-IDF through TF saturation
  and document length normalization.

Key settings:
- BM25Okapi with k1=1.2, b=0.75
- Top-K = 100
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

# Allow `import submission_utils` when running this file from `code/`.
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from submission_utils import (
    DEFAULT_TOP_K,
    create_submission_zip,
    data_paths,
    format_title_abstract,
    iteration_submission_paths,
    load_queries_corpus,
    save_submission,
    validate_doc_ids_in_corpus,
    validate_submission,
)

ITERATION_NAME = "iteration_2"
TOP_K = DEFAULT_TOP_K
BM25_K1 = 1.2
BM25_B = 0.75


def tokenize(text: str) -> List[str]:
    return text.lower().split()


def rank_with_bm25(
    query_texts: List[str],
    corpus_texts: List[str],
    top_k: int = TOP_K,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> np.ndarray:
    tokenized_corpus = [tokenize(text) for text in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
    doc_count = len(corpus_texts)
    all_top_idx = np.zeros((len(query_texts), top_k), dtype=np.int64)

    for i, query in enumerate(query_texts):
        query_tokens = tokenize(query)
        scores = bm25.get_scores(query_tokens)
        # Stable sort for deterministic tie behavior.
        all_top_idx[i] = np.argsort(-scores, kind="stable")[:top_k]
        if doc_count < top_k:
            raise ValueError(f"Corpus has {doc_count} docs; cannot return top-{top_k}.")

    return all_top_idx


def build_bm25_submission(queries: pd.DataFrame, corpus: pd.DataFrame, top_k: int = TOP_K) -> Dict[str, List[str]]:
    query_texts = queries.apply(format_title_abstract, axis=1).tolist()
    corpus_texts = corpus.apply(format_title_abstract, axis=1).tolist()

    top_idx = rank_with_bm25(query_texts=query_texts, corpus_texts=corpus_texts, top_k=top_k)
    corpus_ids = corpus["doc_id"].astype(str).tolist()
    query_ids = queries["doc_id"].astype(str).tolist()

    submission: Dict[str, List[str]] = {}
    for i, qid in enumerate(query_ids):
        submission[qid] = [corpus_ids[j] for j in top_idx[i]]
    return submission


def main() -> None:
    data = data_paths()
    if not data["using_held_out_queries"]:
        raise RuntimeError(
            "Submission mode requires held-out queries. "
            "Add `held_out_queries.parquet` to `data/`, project root, or `starter_kit/`."
        )

    paths = {**data, **iteration_submission_paths(ITERATION_NAME)}
    queries, corpus = load_queries_corpus(paths["queries_path"], paths["corpus_path"])
    submission = build_bm25_submission(queries=queries, corpus=corpus, top_k=TOP_K)

    expected_query_ids = queries["doc_id"].astype(str).tolist()
    validate_submission(submission=submission, expected_query_ids=expected_query_ids, top_k=TOP_K)
    validate_doc_ids_in_corpus(submission=submission, corpus_doc_ids=corpus["doc_id"].astype(str).tolist())
    save_submission(submission=submission, output_file=paths["output_file"])
    create_submission_zip(submission_file=paths["output_file"], zip_file=paths["zip_file"])

    print("Iteration 2 submission generated.")
    print(f"Queries: {len(queries)} | Corpus: {len(corpus)} | Top-K: {TOP_K}")
    print(f"Method: BM25 (k1={BM25_K1}, b={BM25_B})")
    print(f"Query source: {paths['queries_source']}")
    print(f"Saved to: {paths['output_file']}")
    print(f"Zipped to: {paths['zip_file']}")


if __name__ == "__main__":
    main()
