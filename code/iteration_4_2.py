from __future__ import annotations

"""
Iteration 4.2 — BM25 (rich) + MiniLM (rich), weighted RRF.

Motivation (team feedback + empirical lessons):
- Upgrade sparse leg from TF-IDF to **BM25** while keeping **exact Iteration-4 text
  alignment** (title + abstract + truncated `full_text` for queries and corpus).
- Keep the proven **MiniLM rich** dense leg identical to Iteration 4.
- Fuse with **weighted RRF** (no cross-encoder; CE runs regressed NDCG@10 before).
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

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

ITERATION_NAME = "iteration_4_2"
TOP_K = DEFAULT_TOP_K
FUSION_DEPTH = i4.FUSION_DEPTH
RRF_K = i4.RRF_K
# Slightly favor lexical leg after TF-IDF→BM25 swap (tunable after Codabench).
W_BM25 = 0.52
W_MINILM = 0.48
BM25_K1 = 1.2
BM25_B = 0.75


def tokenize(text: str) -> List[str]:
    return text.lower().split()


def rank_bm25_topk(query_texts: List[str], corpus_texts: List[str], top_k: int) -> np.ndarray:
    tokenized_corpus = [tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus, k1=BM25_K1, b=BM25_B)
    n_docs = len(corpus_texts)
    out = np.zeros((len(query_texts), top_k), dtype=np.int64)
    for i, q in enumerate(query_texts):
        scores = bm25.get_scores(tokenize(q))
        out[i] = np.argsort(-scores, kind="stable")[:top_k]
        if n_docs < top_k:
            raise ValueError(f"Corpus has {n_docs} docs; cannot return top-{top_k}.")
    return out


def weighted_rrf(rankings: List[np.ndarray], weights: List[float], k: float, top_n: int) -> List[np.ndarray]:
    nq = rankings[0].shape[0]
    fused: List[np.ndarray] = []
    for i in range(nq):
        scores: Dict[int, float] = {}
        for w, ranks in zip(weights, rankings):
            for rank, doc_idx in enumerate(ranks[i], start=1):
                j = int(doc_idx)
                scores[j] = scores.get(j, 0.0) + w / (k + rank)
        ordered = sorted(scores.keys(), key=lambda d: (-scores[d], d))
        if len(ordered) < top_n:
            raise ValueError(f"RRF produced only {len(ordered)} docs; need {top_n}")
        fused.append(np.array(ordered[:top_n], dtype=np.int64))
    return fused


def main() -> None:
    data = data_paths()
    if not data["using_held_out_queries"]:
        raise RuntimeError(
            "Submission mode requires held-out queries. "
            "Add `held_out_queries.parquet` to `data/`, project root, or `starter_kit/`."
        )

    paths = {**data, **iteration_submission_paths(ITERATION_NAME)}
    queries, corpus = load_queries_corpus(paths["queries_path"], paths["corpus_path"])
    corpus_ids = corpus["doc_id"].astype(str).tolist()
    query_ids = queries["doc_id"].astype(str).tolist()

    query_texts = queries.apply(lambda r: i4.format_rich_text(r, i4.MAX_FULLTEXT_CHARS), axis=1).tolist()
    corpus_texts = corpus.apply(lambda r: i4.format_rich_text(r, i4.MAX_FULLTEXT_CHARS), axis=1).tolist()

    print(f"BM25 sparse ranks (rich text, k1={BM25_K1}, b={BM25_B}) …")
    bm25_ranks = rank_bm25_topk(query_texts, corpus_texts, top_k=FUSION_DEPTH)

    print(f"Loading {i4.MINILM_MODEL} …")
    model = SentenceTransformer(i4.MINILM_MODEL)
    print("Encoding corpus with MiniLM (rich text) …")
    corpus_emb = i4.encode_minilm(model, corpus_texts)
    print("Encoding queries with MiniLM (rich text) …")
    query_emb = i4.encode_minilm(model, query_texts)
    dense_ranks = i4.rank_dense_topk(query_emb, corpus_emb, top_k=FUSION_DEPTH)

    fused = weighted_rrf(
        [bm25_ranks, dense_ranks],
        [W_BM25, W_MINILM],
        k=RRF_K,
        top_n=TOP_K,
    )

    submission: Dict[str, List[str]] = {}
    for i, qid in enumerate(query_ids):
        submission[qid] = [corpus_ids[int(j)] for j in fused[i]]

    validate_submission(submission=submission, expected_query_ids=query_ids, top_k=TOP_K)
    validate_doc_ids_in_corpus(submission=submission, corpus_doc_ids=corpus_ids)
    save_submission(submission=submission, output_file=paths["output_file"])
    create_submission_zip(submission_file=paths["output_file"], zip_file=paths["zip_file"])

    print("Iteration 4.2 submission generated.")
    print(
        f"Method: BM25 rich + MiniLM rich | weighted RRF weights=({W_BM25}, {W_MINILM}), "
        f"k={RRF_K}, depth={FUSION_DEPTH}, full_text≤{i4.MAX_FULLTEXT_CHARS}"
    )
    print(f"Query source: {paths['queries_source']}")
    print(f"Saved to: {paths['output_file']}")
    print(f"Zipped to: {paths['zip_file']}")


if __name__ == "__main__":
    main()
