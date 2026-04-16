from __future__ import annotations

"""
Iteration 12 — interpolation + RRF blend (TF-IDF rich + MiniLM rich + SPECTER2 TA).

We compute for each query:
- Per-retriever normalized score vectors on a common candidate pool.
- Score interpolation (weighted sum of normalized scores).
- Rank-based RRF scores from each retriever's rank list.
- Final score = (1 - GAMMA) * interpolation + GAMMA * normalized_RRF

This is the "clean" version of: interpolation + RRF (without BM25 chunks or CE),
using the Iteration 10 retrievers.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
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

ITERATION_NAME = "iteration_12"
TOP_K = DEFAULT_TOP_K
DEPTH = 400  # deeper candidate set for interpolation stability
RRF_K = 40.0

SPECTER2_MODEL = "allenai/specter2_base"
SPECTER_BATCH = 32

# interpolation weights (must sum to 1 for interpretability)
W_TFIDF = 0.48
W_MINILM = 0.42
W_SPECTER2 = 0.10

# blend interpolation vs RRF
GAMMA_RRF = 0.35


def specter_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "embeddings" / "allenai_specter2_base"


def format_specter(row: pd.Series, sep: str) -> str:
    title = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    if title and abstract:
        return f"{title} {sep} {abstract}".strip()
    return title or abstract


def encode_specter(model: SentenceTransformer, texts: Sequence[str]) -> np.ndarray:
    emb = model.encode(
        list(texts),
        batch_size=SPECTER_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return emb.astype(np.float32)


def load_or_cache_specter_corpus(model: SentenceTransformer, texts: Sequence[str], ids: List[str]) -> np.ndarray:
    d = specter_dir()
    d.mkdir(parents=True, exist_ok=True)
    npy = d / "corpus_embeddings.npy"
    idf = d / "corpus_ids.json"
    if npy.exists() and idf.exists():
        with idf.open("r", encoding="utf-8") as f:
            saved = [str(x) for x in json.load(f)]
        if saved == ids:
            print(f"Loading cached SPECTER2 corpus embeddings from {npy} …")
            return np.load(npy)
    print("Encoding corpus with SPECTER2 and caching …")
    emb = encode_specter(model, texts)
    np.save(npy, emb)
    with idf.open("w", encoding="utf-8") as f:
        json.dump(ids, f)
    return emb


def minmax(x: np.ndarray) -> np.ndarray:
    lo = float(x.min())
    hi = float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def rrf_from_rank(rank_idx: np.ndarray, n_docs: int, k: float) -> np.ndarray:
    """rank_idx: (depth,) doc indices. Returns full-length score vector."""
    s = np.zeros(n_docs, dtype=np.float32)
    for r, j in enumerate(rank_idx, start=1):
        s[int(j)] += 1.0 / (k + r)
    return s


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
    n_docs = len(corpus_ids)

    query_rich = queries.apply(lambda r: i4.format_rich_text(r, i4.MAX_FULLTEXT_CHARS), axis=1).tolist()
    corpus_rich = corpus.apply(lambda r: i4.format_rich_text(r, i4.MAX_FULLTEXT_CHARS), axis=1).tolist()

    # TF-IDF dense score matrix is too big; we use ranks and compute scores only on candidate pool.
    tfidf_ranks = i4.rank_with_tfidf_topk(query_rich, corpus_rich, top_k=DEPTH)

    print(f"Loading {i4.MINILM_MODEL} …")
    minilm = SentenceTransformer(i4.MINILM_MODEL)
    print("Encoding corpus with MiniLM (rich text) …")
    corpus_emb_m = i4.encode_minilm(minilm, corpus_rich)
    print("Encoding queries with MiniLM (rich text) …")
    query_emb_m = i4.encode_minilm(minilm, query_rich)
    minilm_sim = (query_emb_m @ corpus_emb_m.T).astype(np.float32)
    minilm_ranks = np.argsort(-minilm_sim, axis=1, kind="stable")[:, :DEPTH]

    print(f"Loading {SPECTER2_MODEL} …")
    specter = SentenceTransformer(SPECTER2_MODEL)
    sep = specter.tokenizer.sep_token or "[SEP]"
    query_s = queries.apply(lambda r: format_specter(r, sep), axis=1).tolist()
    corpus_s = corpus.apply(lambda r: format_specter(r, sep), axis=1).tolist()
    corpus_emb_s = load_or_cache_specter_corpus(specter, corpus_s, corpus_ids)
    print("Encoding queries with SPECTER2 …")
    query_emb_s = encode_specter(specter, query_s)
    specter_sim = (query_emb_s @ corpus_emb_s.T).astype(np.float32)
    specter_ranks = np.argsort(-specter_sim, axis=1, kind="stable")[:, :DEPTH]

    # Build final per-query predictions
    submission: Dict[str, List[str]] = {}
    for i, qid in enumerate(query_ids):
        cand = sorted(set(tfidf_ranks[i].tolist()) | set(minilm_ranks[i].tolist()) | set(specter_ranks[i].tolist()))
        cand = np.asarray(cand, dtype=np.int64)

        # score interpolation on candidate pool
        # TF-IDF: use rank-based proxy score (1/rank) within depth, normalized on cand
        tf_pos = {int(d): r + 1 for r, d in enumerate(tfidf_ranks[i].tolist())}
        tf_score = np.asarray([1.0 / tf_pos.get(int(d), DEPTH + 1) for d in cand], dtype=np.float32)
        tf_score = minmax(tf_score)

        m_score = minmax(minilm_sim[i, cand])
        s_score = minmax(specter_sim[i, cand])
        interp = (W_TFIDF * tf_score) + (W_MINILM * m_score) + (W_SPECTER2 * s_score)

        # RRF score vector built from three rank lists, then evaluated on cand
        rrf_full = (
            rrf_from_rank(tfidf_ranks[i], n_docs, RRF_K)
            + rrf_from_rank(minilm_ranks[i], n_docs, RRF_K)
            + rrf_from_rank(specter_ranks[i], n_docs, RRF_K)
        )
        rrf_c = minmax(rrf_full[cand])

        final = (1.0 - GAMMA_RRF) * interp + GAMMA_RRF * rrf_c
        order = np.argsort(-final, kind="stable")
        ranked_idx = cand[order][:TOP_K]

        # pad if candidate pool smaller than TOP_K
        if len(ranked_idx) < TOP_K:
            seen = set(ranked_idx.tolist())
            filler = [int(j) for j in tfidf_ranks[i].tolist() if int(j) not in seen]
            ranked_idx = np.asarray((ranked_idx.tolist() + filler)[:TOP_K], dtype=np.int64)

        submission[qid] = [corpus_ids[int(j)] for j in ranked_idx]

    validate_submission(submission=submission, expected_query_ids=query_ids, top_k=TOP_K)
    validate_doc_ids_in_corpus(submission=submission, corpus_doc_ids=corpus_ids)
    save_submission(submission=submission, output_file=paths["output_file"])
    create_submission_zip(submission_file=paths["output_file"], zip_file=paths["zip_file"])

    print("Iteration 12 submission generated.")
    print(
        "Method: interpolation(norm scores) + RRF blend | "
        f"interp_w=({W_TFIDF},{W_MINILM},{W_SPECTER2}), gamma_rrf={GAMMA_RRF}, depth={DEPTH}"
    )
    print(f"Saved to: {paths['zip_file']}")


if __name__ == "__main__":
    main()

