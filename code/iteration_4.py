from __future__ import annotations

"""
Iteration 4 Notes (presentation-ready context)
=============================================
What we changed (two directions at once):
1. **RRF tuning:** deeper pool (`FUSION_DEPTH=300`) and smaller RRF constant (`RRF_K=40`)
   vs Iteration 3 (200 / 60).
2. **Richer text:** `title + abstract + truncated full_text` for both sparse and dense,
   so both sides stay aligned. Dense re-encodes the full corpus at runtime (precomputed
   MiniLM vectors are title+abstract-only and would mismatch rich queries).

Model: `sentence-transformers/all-MiniLM-L6-v2`, L2-normalized.
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

ITERATION_NAME = "iteration_4"
TOP_K = DEFAULT_TOP_K
FUSION_DEPTH = 300
RRF_K = 40.0
MAX_FULLTEXT_CHARS = 2000
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ENCODE_BATCH_SIZE = 128


def format_rich_text(row: pd.Series, max_fulltext_chars: int = MAX_FULLTEXT_CHARS) -> str:
    base = format_title_abstract(row)
    ft = row.get("full_text")
    if ft is None or pd.isna(ft):
        return base
    s = str(ft).replace("\n", " ").strip()
    if not s:
        return base
    if len(s) > max_fulltext_chars:
        s = s[:max_fulltext_chars]
    return f"{base} {s}".strip() if base else s


def rank_with_tfidf_topk(
    query_texts: List[str],
    corpus_texts: List[str],
    top_k: int,
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


def encode_minilm(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=ENCODE_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return emb.astype(np.float32)


def rank_dense_topk(query_emb: np.ndarray, corpus_emb: np.ndarray, top_k: int) -> np.ndarray:
    sim = query_emb @ corpus_emb.T
    return np.argsort(-sim, axis=1, kind="stable")[:, :top_k]


def rrf_fuse_two_lists(rank_a: np.ndarray, rank_b: np.ndarray, k: float, top_n: int) -> List[np.ndarray]:
    nq = rank_a.shape[0]
    out: List[np.ndarray] = []
    for i in range(nq):
        scores: Dict[int, float] = {}
        for rank, doc_idx in enumerate(rank_a[i], start=1):
            scores[int(doc_idx)] = scores.get(int(doc_idx), 0.0) + 1.0 / (k + rank)
        for rank, doc_idx in enumerate(rank_b[i], start=1):
            scores[int(doc_idx)] = scores.get(int(doc_idx), 0.0) + 1.0 / (k + rank)
        ordered = sorted(scores.keys(), key=lambda d: (-scores[d], d))
        if len(ordered) < top_n:
            raise ValueError(f"RRF produced only {len(ordered)} docs; need {top_n}")
        out.append(np.array(ordered[:top_n], dtype=np.int64))
    return out


def build_hybrid_submission(
    queries: pd.DataFrame,
    corpus: pd.DataFrame,
    model: SentenceTransformer,
    top_k: int = TOP_K,
    fusion_depth: int = FUSION_DEPTH,
) -> Dict[str, List[str]]:
    query_texts = queries.apply(lambda r: format_rich_text(r, MAX_FULLTEXT_CHARS), axis=1).tolist()
    corpus_texts = corpus.apply(lambda r: format_rich_text(r, MAX_FULLTEXT_CHARS), axis=1).tolist()

    tfidf_ranks = rank_with_tfidf_topk(query_texts, corpus_texts, top_k=fusion_depth)

    print("Encoding corpus with MiniLM (rich text) …")
    corpus_emb = encode_minilm(model, corpus_texts)
    print("Encoding queries with MiniLM (rich text) …")
    query_emb = encode_minilm(model, query_texts)

    dense_ranks = rank_dense_topk(query_emb, corpus_emb, top_k=fusion_depth)
    fused = rrf_fuse_two_lists(tfidf_ranks, dense_ranks, k=RRF_K, top_n=top_k)

    corpus_ids = corpus["doc_id"].astype(str).tolist()
    query_ids = queries["doc_id"].astype(str).tolist()
    submission: Dict[str, List[str]] = {}
    for i, qid in enumerate(query_ids):
        submission[qid] = [corpus_ids[j] for j in fused[i]]
    return submission


def compute_iter4_fused_indices(
    queries: pd.DataFrame,
    corpus: pd.DataFrame,
    model: SentenceTransformer,
    top_k: int = TOP_K,
    fusion_depth: int = FUSION_DEPTH,
) -> np.ndarray:
    """Corpus row indices after Iteration 4 hybrid, shape (n_queries, top_k)."""
    query_texts = queries.apply(lambda r: format_rich_text(r, MAX_FULLTEXT_CHARS), axis=1).tolist()
    corpus_texts = corpus.apply(lambda r: format_rich_text(r, MAX_FULLTEXT_CHARS), axis=1).tolist()

    tfidf_ranks = rank_with_tfidf_topk(query_texts, corpus_texts, top_k=fusion_depth)

    print("Encoding corpus with MiniLM (rich text) …")
    corpus_emb = encode_minilm(model, corpus_texts)
    print("Encoding queries with MiniLM (rich text) …")
    query_emb = encode_minilm(model, query_texts)

    dense_ranks = rank_dense_topk(query_emb, corpus_emb, top_k=fusion_depth)
    fused = rrf_fuse_two_lists(tfidf_ranks, dense_ranks, k=RRF_K, top_n=top_k)

    return np.stack(fused, axis=0)


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

    print(f"Loading {MINILM_MODEL} …")
    model = SentenceTransformer(MINILM_MODEL)

    submission = build_hybrid_submission(
        queries, corpus, model, top_k=TOP_K, fusion_depth=FUSION_DEPTH
    )

    expected_query_ids = queries["doc_id"].astype(str).tolist()
    validate_submission(submission=submission, expected_query_ids=expected_query_ids, top_k=TOP_K)
    validate_doc_ids_in_corpus(submission=submission, corpus_doc_ids=parquet_ids)
    save_submission(submission=submission, output_file=paths["output_file"])
    create_submission_zip(submission_file=paths["output_file"], zip_file=paths["zip_file"])

    print("Iteration 4 submission generated.")
    print(f"Queries: {len(queries)} | Corpus: {len(corpus)} | Top-K: {TOP_K}")
    print(
        f"Method: rich text (≤{MAX_FULLTEXT_CHARS} chars full_text) + "
        f"TF-IDF + MiniLM + RRF (k={RRF_K}, fusion_depth={FUSION_DEPTH})"
    )
    print(f"Query source: {paths['queries_source']}")
    print(f"Saved to: {paths['output_file']}")
    print(f"Zipped to: {paths['zip_file']}")


if __name__ == "__main__":
    main()
