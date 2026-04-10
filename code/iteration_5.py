from __future__ import annotations

"""
Iteration 5 Notes (presentation-ready context)
=============================================
What we changed vs Iteration 4:
1. **Triple RRF:** fuse three ranked lists instead of two:
   - TF-IDF on **title-doubled** rich text (stronger lexical weight on titles).
   - MiniLM dense on **rich** text (full corpus re-encode, same idea as Iteration 4).
   - MiniLM dense on **title+abstract only** using **precomputed** corpus embeddings
     + TA query encodes (complementary to rich dense; no second corpus encode).
2. **Mild tuning:** `FUSION_DEPTH=350`, `RRF_K=50`, `MAX_FULLTEXT_CHARS=2500`.

Model: `sentence-transformers/all-MiniLM-L6-v2`, L2-normalized.
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

ITERATION_NAME = "iteration_5"
TOP_K = DEFAULT_TOP_K
FUSION_DEPTH = 350
RRF_K = 50.0
MAX_FULLTEXT_CHARS = 2500
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ENCODE_BATCH_SIZE = 128


def embedding_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "embeddings" / "sentence-transformers_all-MiniLM-L6-v2"


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


def format_sparse_title_doubled_rich(row: pd.Series, max_fulltext_chars: int = MAX_FULLTEXT_CHARS) -> str:
    """Rich text for TF-IDF with title tokens repeated once (2× title weight in the bag)."""
    title = "" if pd.isna(row.get("title")) else str(row.get("title")).strip()
    abstract = "" if pd.isna(row.get("abstract")) else str(row.get("abstract")).strip()
    if title and abstract:
        head = f"{title} {title} {abstract}"
    elif title:
        head = f"{title} {title}"
    elif abstract:
        head = abstract
    else:
        head = ""
    ft = row.get("full_text")
    if ft is None or pd.isna(ft):
        return head.strip()
    s = str(ft).replace("\n", " ").strip()
    if not s:
        return head.strip()
    if len(s) > max_fulltext_chars:
        s = s[:max_fulltext_chars]
    return f"{head} {s}".strip() if head else s


def format_title_abstract_minilm(row: pd.Series) -> str:
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


def rrf_fuse_many(rankings: List[np.ndarray], k: float, top_n: int) -> List[np.ndarray]:
    nq = rankings[0].shape[0]
    out: List[np.ndarray] = []
    for i in range(nq):
        scores: Dict[int, float] = {}
        for ranks in rankings:
            for rank, doc_idx in enumerate(ranks[i], start=1):
                idx = int(doc_idx)
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
        ordered = sorted(scores.keys(), key=lambda d: (-scores[d], d))
        if len(ordered) < top_n:
            raise ValueError(f"RRF produced only {len(ordered)} docs; need {top_n}")
        out.append(np.array(ordered[:top_n], dtype=np.int64))
    return out


def build_hybrid_submission(
    queries: pd.DataFrame,
    corpus: pd.DataFrame,
    model: SentenceTransformer,
    corpus_emb_ta: np.ndarray,
    top_k: int = TOP_K,
    fusion_depth: int = FUSION_DEPTH,
) -> Dict[str, List[str]]:
    query_sparse = queries.apply(lambda r: format_sparse_title_doubled_rich(r, MAX_FULLTEXT_CHARS), axis=1).tolist()
    corpus_sparse = corpus.apply(lambda r: format_sparse_title_doubled_rich(r, MAX_FULLTEXT_CHARS), axis=1).tolist()
    tfidf_ranks = rank_with_tfidf_topk(query_sparse, corpus_sparse, top_k=fusion_depth)

    query_rich = queries.apply(lambda r: format_rich_text(r, MAX_FULLTEXT_CHARS), axis=1).tolist()
    corpus_rich = corpus.apply(lambda r: format_rich_text(r, MAX_FULLTEXT_CHARS), axis=1).tolist()

    print("Encoding corpus with MiniLM (rich text) …")
    corpus_emb_rich = encode_minilm(model, corpus_rich)
    print("Encoding queries with MiniLM (rich text) …")
    query_emb_rich = encode_minilm(model, query_rich)
    dense_rich_ranks = rank_dense_topk(query_emb_rich, corpus_emb_rich, top_k=fusion_depth)

    query_ta = queries.apply(format_title_abstract_minilm, axis=1).tolist()
    print("Encoding queries with MiniLM (title+abstract only) …")
    query_emb_ta = encode_minilm(model, query_ta)
    dense_ta_ranks = rank_dense_topk(query_emb_ta, corpus_emb_ta, top_k=fusion_depth)

    fused = rrf_fuse_many([tfidf_ranks, dense_rich_ranks, dense_ta_ranks], k=RRF_K, top_n=top_k)

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
        raise FileNotFoundError(f"Missing TA corpus embeddings under {emb_path}")

    corpus_emb_ta = np.load(corpus_npy)
    with corpus_ids_path.open("r", encoding="utf-8") as f:
        emb_corpus_ids = [str(x) for x in json.load(f)]

    paths = {**data, **iteration_submission_paths(ITERATION_NAME)}
    queries, corpus = load_queries_corpus(paths["queries_path"], paths["corpus_path"])
    parquet_ids = corpus["doc_id"].astype(str).tolist()
    if parquet_ids != emb_corpus_ids:
        raise ValueError("corpus.parquet row order does not match corpus_ids.json — cannot use precomputed TA embeddings.")

    print(f"Loading {MINILM_MODEL} …")
    model = SentenceTransformer(MINILM_MODEL)

    submission = build_hybrid_submission(
        queries, corpus, model, corpus_emb_ta, top_k=TOP_K, fusion_depth=FUSION_DEPTH
    )

    expected_query_ids = queries["doc_id"].astype(str).tolist()
    validate_submission(submission=submission, expected_query_ids=expected_query_ids, top_k=TOP_K)
    validate_doc_ids_in_corpus(submission=submission, corpus_doc_ids=parquet_ids)
    save_submission(submission=submission, output_file=paths["output_file"])
    create_submission_zip(submission_file=paths["output_file"], zip_file=paths["zip_file"])

    print("Iteration 5 submission generated.")
    print(f"Queries: {len(queries)} | Corpus: {len(corpus)} | Top-K: {TOP_K}")
    print(
        f"Method: triple RRF — TF-IDF (title×2 + rich), dense rich, dense TA | "
        f"k={RRF_K}, depth={FUSION_DEPTH}, full_text≤{MAX_FULLTEXT_CHARS}"
    )
    print(f"Query source: {paths['queries_source']}")
    print(f"Saved to: {paths['output_file']}")
    print(f"Zipped to: {paths['zip_file']}")


if __name__ == "__main__":
    main()
