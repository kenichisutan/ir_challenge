from __future__ import annotations

"""
Iteration 3 Notes (presentation-ready context)
=============================================
What we changed:
- Hybrid retrieval: TF-IDF (same as Iteration 1) + dense MiniLM similarity.
- Fuse rankings with Reciprocal Rank Fusion (RRF).

Why:
- Held-out query IDs do not match precomputed `query_embeddings.npy` (public queries only),
  so we encode held-out queries at runtime with the same model as `embed.py`.
- Corpus vectors stay precomputed (`corpus_embeddings.npy`).

Settings:
- MiniLM: `sentence-transformers/all-MiniLM-L6-v2`, L2-normalized embeddings.
- RRF k = 60; each retriever contributes top-200 ranks before fusion; final top-100.
"""

import json
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

ITERATION_NAME = "iteration_3"
TOP_K = DEFAULT_TOP_K
FUSION_DEPTH = 200
RRF_K = 60.0
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ENCODE_BATCH_SIZE = 64


def embedding_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "embeddings" / "sentence-transformers_all-MiniLM-L6-v2"


def format_text_for_minilm(row: pd.Series) -> str:
    title = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    if title and abstract:
        return title + " " + abstract
    return title or abstract


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


def encode_queries_minilm(texts: List[str]) -> np.ndarray:
    model = SentenceTransformer(MINILM_MODEL)
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
    """rank_* shape (n_queries, depth) corpus row indices, best-first."""
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
    corpus_emb: np.ndarray,
    top_k: int = TOP_K,
    fusion_depth: int = FUSION_DEPTH,
) -> Dict[str, List[str]]:
    query_texts_tfidf = queries.apply(format_title_abstract, axis=1).tolist()
    corpus_texts_tfidf = corpus.apply(format_title_abstract, axis=1).tolist()
    query_texts_dense = queries.apply(format_text_for_minilm, axis=1).tolist()

    tfidf_ranks = rank_with_tfidf_topk(query_texts_tfidf, corpus_texts_tfidf, top_k=fusion_depth)
    query_emb = encode_queries_minilm(query_texts_dense)
    dense_ranks = rank_dense_topk(query_emb, corpus_emb, top_k=fusion_depth)

    fused = rrf_fuse_two_lists(tfidf_ranks, dense_ranks, k=RRF_K, top_n=top_k)
    corpus_ids = corpus["doc_id"].astype(str).tolist()
    query_ids = queries["doc_id"].astype(str).tolist()

    submission: Dict[str, List[str]] = {}
    for i, qid in enumerate(query_ids):
        submission[qid] = [corpus_ids[j] for j in fused[i]]
    return submission


def main() -> None:
    data = data_paths()
    if not data["using_held_out_queries"]:
        raise RuntimeError(
            "Submission mode requires held-out queries. "
            "Add `held_out_queries.parquet` to `data/`, project root, or `starter_kit/`."
        )

    emb_path = embedding_dir()
    corpus_npy = emb_path / "corpus_embeddings.npy"
    corpus_ids_path = emb_path / "corpus_ids.json"
    if not corpus_npy.exists() or not corpus_ids_path.exists():
        raise FileNotFoundError(f"Missing corpus embeddings under {emb_path}")

    corpus_emb = np.load(corpus_npy)
    with corpus_ids_path.open("r", encoding="utf-8") as f:
        emb_corpus_ids = [str(x) for x in json.load(f)]

    paths = {**data, **iteration_submission_paths(ITERATION_NAME)}
    queries, corpus = load_queries_corpus(paths["queries_path"], paths["corpus_path"])

    parquet_ids = corpus["doc_id"].astype(str).tolist()
    if parquet_ids != emb_corpus_ids:
        raise ValueError("corpus.parquet row order does not match corpus_ids.json — cannot use precomputed embeddings.")

    submission = build_hybrid_submission(queries, corpus, corpus_emb, top_k=TOP_K, fusion_depth=FUSION_DEPTH)

    expected_query_ids = queries["doc_id"].astype(str).tolist()
    validate_submission(submission=submission, expected_query_ids=expected_query_ids, top_k=TOP_K)
    validate_doc_ids_in_corpus(submission=submission, corpus_doc_ids=parquet_ids)
    save_submission(submission=submission, output_file=paths["output_file"])
    create_submission_zip(submission_file=paths["output_file"], zip_file=paths["zip_file"])

    print("Iteration 3 submission generated.")
    print(f"Queries: {len(queries)} | Corpus: {len(corpus)} | Top-K: {TOP_K}")
    print(f"Method: TF-IDF + MiniLM dense + RRF (k={RRF_K}, fusion_depth={FUSION_DEPTH})")
    print(f"Query source: {paths['queries_source']}")
    print(f"Saved to: {paths['output_file']}")
    print(f"Zipped to: {paths['zip_file']}")


if __name__ == "__main__":
    main()
