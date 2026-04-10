from __future__ import annotations

"""
Iteration 1 Notes (presentation-ready context)
=============================================
What we did:
- Built a TF-IDF retrieval baseline over `title + abstract`.
- Ranked all corpus documents for each query and kept top-100.
- Exported `submission_data.json` and `iteration_1.zip` automatically.

How we did it:
- Used `TfidfVectorizer` + cosine similarity for scoring.
- Used deterministic sorting for stable rankings.
- Shared validation/export/zip logic lives in `scripts/submission_utils.py`.

Why we did it:
- Establish a simple, reproducible baseline for Codabench submissions.
- Create a reliable reference point for later iterations (BM25, dense, hybrid).

Template for future iterations:
- What changed this iteration?
- Why this change was expected to improve NDCG@10?
- What exact settings/model/parameters were used?
- What result/feedback did Codabench return?
- What is the next hypothesis?
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Allow `import submission_utils` when running this file from `code/`.
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from submission_utils import (
    create_submission_zip,
    data_paths,
    format_title_abstract,
    iteration_submission_paths,
    load_queries_corpus,
    save_submission,
    validate_submission,
    validate_doc_ids_in_corpus,
    DEFAULT_TOP_K,
)

ITERATION_NAME = "iteration_1"
TOP_K = DEFAULT_TOP_K


def rank_with_tfidf(
    query_texts: List[str],
    corpus_texts: List[str],
    top_k: int = TOP_K,
) -> np.ndarray:
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 1),
        stop_words="english",
    )
    corpus_matrix = vectorizer.fit_transform(corpus_texts)
    query_matrix = vectorizer.transform(query_texts)
    sim_matrix = cosine_similarity(query_matrix, corpus_matrix)
    return np.argsort(-sim_matrix, axis=1, kind="stable")[:, :top_k]


def build_tfidf_submission(queries: pd.DataFrame, corpus: pd.DataFrame, top_k: int = TOP_K) -> Dict[str, List[str]]:
    query_texts = queries.apply(format_title_abstract, axis=1).tolist()
    corpus_texts = corpus.apply(format_title_abstract, axis=1).tolist()

    top_idx = rank_with_tfidf(query_texts=query_texts, corpus_texts=corpus_texts, top_k=top_k)
    corpus_ids = corpus["doc_id"].astype(str).tolist()
    query_ids = queries["doc_id"].astype(str).tolist()

    submission: Dict[str, List[str]] = {}
    for i, qid in enumerate(query_ids):
        ranked_doc_ids = [corpus_ids[j] for j in top_idx[i]]
        submission[qid] = ranked_doc_ids
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
    submission = build_tfidf_submission(queries=queries, corpus=corpus, top_k=TOP_K)
    # Per challenge instructions, submission keys must match held_out_queries.parquet.
    expected_query_ids = queries["doc_id"].astype(str).tolist()
    validate_submission(submission=submission, expected_query_ids=expected_query_ids, top_k=TOP_K)
    validate_doc_ids_in_corpus(submission=submission, corpus_doc_ids=corpus["doc_id"].astype(str).tolist())
    save_submission(submission=submission, output_file=paths["output_file"])
    create_submission_zip(submission_file=paths["output_file"], zip_file=paths["zip_file"])

    print("Iteration 1 submission generated.")
    print(f"Queries: {len(queries)} | Corpus: {len(corpus)} | Top-K: {TOP_K}")
    print(f"Query source: {paths['queries_source']}")
    print(f"Saved to: {paths['output_file']}")
    print(f"Zipped to: {paths['zip_file']}")


if __name__ == "__main__":
    main()
