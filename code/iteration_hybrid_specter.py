#!/usr/bin/env python3
"""
Iteration C1 + SPECTER2 — Charlotte-style hybrid with a **fused dense** leg.

Uses cached `allenai/specter2_base` **corpus** embeddings; **queries** are encoded
at runtime with the same title/abstract format as elsewhere in this repo
(`title [SEP] abstract`).

Dense similarity (before metadata boosts) is a convex blend of MiniLM and SPECTER2
dot-products (both L2-normalized → cosine):

    dense_raw = (1 - eta) * minilm_dot + eta * specter_dot

Then the same pipeline as `iteration_c1.py`: add domain / venue / year boosts,
per-query min-max, blend with TF-IDF via `alpha`, optional RRF stage `gamma_rrf`.

Hyperparameters (`eta`, `alpha`, boosts, `gamma_rrf`) are tuned on **public**
`qrels.json` for mean NDCG@10, then applied to held-out queries.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from submission_utils import (
    DEFAULT_TOP_K,
    challenge_dir,
    create_submission_zip,
    data_paths,
    iteration_submission_paths,
    load_queries_corpus,
    save_submission,
    validate_doc_ids_in_corpus,
    validate_submission,
)

ITERATION_NAME = "iteration_hybrid_specter"
TOP_K = DEFAULT_TOP_K

MINILM_EMB = "sentence-transformers_all-MiniLM-L6-v2"
SPECTER_EMB = "allenai_specter2_base"
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPECTER_MODEL = "allenai/specter2_base"
MINILM_ENCODE_BATCH = 128
SPECTER_ENCODE_BATCH = 32
RRF_K = 60.0


def format_ta(df: pd.DataFrame) -> List[str]:
    return (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist()


def format_specter(row: pd.Series, sep: str) -> str:
    title = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    if title and abstract:
        return f"{title} {sep} {abstract}".strip()
    return title or abstract


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ndcg_at_k(ranked: List[str], relevant: Iterable[str], k: int = 10) -> float:
    rel = set(relevant)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked[:k], start=1)
        if doc_id in rel
    )
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_ndcg10(submission: Dict[str, List[str]], qrels: Dict[str, List[str]]) -> float:
    vals = [ndcg_at_k(submission[qid], rels, 10) for qid, rels in qrels.items()]
    return float(np.mean(vals))


def minmax_per_query(scores: np.ndarray) -> np.ndarray:
    mins = scores.min(axis=1, keepdims=True)
    maxs = scores.max(axis=1, keepdims=True)
    return (scores - mins) / (maxs - mins + 1e-12)


def rrf_from_full_scores(dense: np.ndarray, sparse: np.ndarray, k: float) -> np.ndarray:
    nq, nd = dense.shape
    out = np.zeros((nq, nd), dtype=np.float32)
    for i in range(nq):
        rd = np.argsort(-dense[i], kind="stable")
        rs = np.argsort(-sparse[i], kind="stable")
        s = np.zeros(nd, dtype=np.float32)
        for rank, j in enumerate(rd, start=1):
            s[int(j)] += 1.0 / (k + rank)
        for rank, j in enumerate(rs, start=1):
            s[int(j)] += 1.0 / (k + rank)
        out[i] = s
    return minmax_per_query(out.astype(np.float32))


def build_submission(
    query_ids: List[str],
    corpus_ids: np.ndarray,
    minilm_scores: np.ndarray,
    specter_scores: np.ndarray,
    sparse_scores: np.ndarray,
    query_domains: np.ndarray,
    corpus_domains: np.ndarray,
    query_venues: np.ndarray,
    corpus_venues: np.ndarray,
    query_years: np.ndarray,
    corpus_years: np.ndarray,
    eta: float,
    alpha: float,
    domain_boost: float,
    venue_boost: float,
    year_boost: float,
    gamma_rrf: float,
    rrf_norm: np.ndarray,
) -> Dict[str, List[str]]:
    dense_scores = (1.0 - eta) * minilm_scores.astype(np.float32) + eta * specter_scores.astype(
        np.float32
    )

    domain_match = (query_domains[:, None] == corpus_domains[None, :]).astype(np.float32)
    venue_match = (
        (query_venues[:, None] == corpus_venues[None, :])
        & (query_venues[:, None] != "")
    ).astype(np.float32)
    qy = query_years[:, None]
    cy = corpus_years[None, :]
    year_ok = (qy >= 0) & (cy >= 0) & (qy == cy)
    year_match = year_ok.astype(np.float32)

    dense_aug = (
        dense_scores
        + domain_boost * domain_match
        + venue_boost * venue_match
        + year_boost * year_match
    )
    sparse_aug = (
        sparse_scores.astype(np.float32)
        + domain_boost * domain_match
        + venue_boost * venue_match
        + year_boost * year_match
    )

    charlotte = alpha * minmax_per_query(dense_aug) + (1.0 - alpha) * minmax_per_query(sparse_aug)
    final_scores = (1.0 - gamma_rrf) * charlotte + gamma_rrf * rrf_norm

    top_idx = np.argsort(-final_scores, axis=1)[:, :TOP_K]
    submission: Dict[str, List[str]] = {}
    for i, qid in enumerate(query_ids):
        ranked = corpus_ids[top_idx[i]].tolist()
        ranked = [doc_id for doc_id in ranked if doc_id != qid]
        if len(ranked) < TOP_K:
            seen = set(ranked)
            extras = [
                doc_id
                for doc_id in corpus_ids[np.argsort(-final_scores[i])]
                if doc_id not in seen and doc_id != qid
            ]
            ranked.extend(extras[: TOP_K - len(ranked)])
        submission[qid] = ranked[:TOP_K]
    return submission


def years_as_int64(series: pd.Series) -> np.ndarray:
    out = np.full(len(series), -1, dtype=np.int64)
    raw = series.to_numpy()
    for i, v in enumerate(raw):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        try:
            out[i] = int(v)
        except (TypeError, ValueError):
            continue
    return out


def encode_specter_with_model(model, texts: List[str]) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=SPECTER_ENCODE_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return emb.astype(np.float32)


def encode_minilm(texts: List[str], device: str | None) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    kwargs: dict = {}
    if device:
        kwargs["device"] = device
    model = SentenceTransformer(MINILM_MODEL, **kwargs)
    emb = model.encode(
        texts,
        batch_size=MINILM_ENCODE_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return emb.astype(np.float32)


def tune_joint(
    qrels: Dict[str, List[str]],
    public_query_ids: List[str],
    corpus_ids: np.ndarray,
    minilm_scores: np.ndarray,
    specter_scores: np.ndarray,
    sparse_scores: np.ndarray,
    query_domains: np.ndarray,
    corpus_domains: np.ndarray,
    query_venues: np.ndarray,
    corpus_venues: np.ndarray,
    query_years: np.ndarray,
    corpus_years: np.ndarray,
    rrf_norm: np.ndarray,
) -> Tuple[dict, float]:
    best_cfg = None
    best_ndcg = -1.0
    for eta in [0.0, 0.15, 0.30, 0.45, 0.60, 0.75]:
        for alpha in [0.50, 0.60, 0.65, 0.70, 0.80]:
            for domain_boost in [0.05, 0.10, 0.15]:
                for venue_boost in [0.00, 0.01, 0.02]:
                    for year_boost in [0.00, 0.05, 0.10]:
                        sub = build_submission(
                            public_query_ids,
                            corpus_ids,
                            minilm_scores,
                            specter_scores,
                            sparse_scores,
                            query_domains,
                            corpus_domains,
                            query_venues,
                            corpus_venues,
                            query_years,
                            corpus_years,
                            eta=eta,
                            alpha=alpha,
                            domain_boost=domain_boost,
                            venue_boost=venue_boost,
                            year_boost=year_boost,
                            gamma_rrf=0.0,
                            rrf_norm=rrf_norm,
                        )
                        score = evaluate_ndcg10(sub, qrels)
                        if score > best_ndcg:
                            best_ndcg = score
                            best_cfg = {
                                "eta": eta,
                                "alpha": alpha,
                                "domain_boost": domain_boost,
                                "venue_boost": venue_boost,
                                "year_boost": year_boost,
                            }
    assert best_cfg is not None
    return best_cfg, best_ndcg


def tune_stage2_rrf(
    qrels: Dict[str, List[str]],
    public_query_ids: List[str],
    corpus_ids: np.ndarray,
    minilm_scores: np.ndarray,
    specter_scores: np.ndarray,
    sparse_scores: np.ndarray,
    query_domains: np.ndarray,
    corpus_domains: np.ndarray,
    query_venues: np.ndarray,
    corpus_venues: np.ndarray,
    query_years: np.ndarray,
    corpus_years: np.ndarray,
    rrf_norm: np.ndarray,
    base: dict,
) -> Tuple[dict, float]:
    best = {**base, "gamma_rrf": 0.0}
    best_ndcg = -1.0
    for gamma_rrf in [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]:
        sub = build_submission(
            public_query_ids,
            corpus_ids,
            minilm_scores,
            specter_scores,
            sparse_scores,
            query_domains,
            corpus_domains,
            query_venues,
            corpus_venues,
            query_years,
            corpus_years,
            eta=base["eta"],
            alpha=base["alpha"],
            domain_boost=base["domain_boost"],
            venue_boost=base["venue_boost"],
            year_boost=base["year_boost"],
            gamma_rrf=gamma_rrf,
            rrf_norm=rrf_norm,
        )
        score = evaluate_ndcg10(sub, qrels)
        if score > best_ndcg:
            best_ndcg = score
            best = {**base, "gamma_rrf": gamma_rrf}
    return best, best_ndcg


def main() -> None:
    root = challenge_dir()
    data_dir = root / "data"
    minilm_dir = data_dir / "embeddings" / MINILM_EMB
    specter_dir = data_dir / "embeddings" / SPECTER_EMB

    dp = data_paths()
    if not dp["using_held_out_queries"]:
        raise RuntimeError("Expected held_out_queries.parquet.")
    paths = {**dp, **iteration_submission_paths(ITERATION_NAME)}
    out_dir = paths["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    queries, corpus = load_queries_corpus(data_dir / "queries.parquet", data_dir / "corpus.parquet")
    heldout = pd.read_parquet(dp["queries_path"])
    qrels = load_json(data_dir / "qrels.json")

    public_query_ids = [str(x) for x in queries["doc_id"].tolist()]
    heldout_query_ids = [str(x) for x in heldout["doc_id"].tolist()]

    corpus_ids_m = np.array(load_json(minilm_dir / "corpus_ids.json"), dtype=str)
    corpus_ids_s = np.array(load_json(specter_dir / "corpus_ids.json"), dtype=str)
    if not np.array_equal(corpus_ids_m, corpus_ids_s):
        raise ValueError("MiniLM and SPECTER2 corpus_ids.json differ; cannot fuse.")
    corpus_ids = corpus_ids_m

    public_dense_ids = [str(x) for x in load_json(minilm_dir / "query_ids.json")]
    if public_dense_ids != public_query_ids:
        raise ValueError("MiniLM public query IDs do not match queries.parquet.")

    print("Building sparse index (TF-IDF TA, uni+bi-grams) …")
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        stop_words="english",
    )
    corpus_ta = format_ta(corpus)
    public_ta = format_ta(queries)
    heldout_ta = format_ta(heldout)
    corpus_sparse = vectorizer.fit_transform(corpus_ta)
    public_sparse = vectorizer.transform(public_ta)
    heldout_sparse = vectorizer.transform(heldout_ta)
    public_sparse_scores = cosine_similarity(public_sparse, corpus_sparse).astype(np.float32)
    heldout_sparse_scores = cosine_similarity(heldout_sparse, corpus_sparse).astype(np.float32)

    print("Loading MiniLM embeddings …")
    corpus_minilm = np.load(minilm_dir / "corpus_embeddings.npy").astype(np.float32)
    public_minilm = np.load(minilm_dir / "query_embeddings.npy").astype(np.float32)
    held_min_path = out_dir / "heldout_minilm_query_embeddings.npy"
    if held_min_path.exists():
        heldout_minilm = np.load(held_min_path).astype(np.float32)
    else:
        print("Encoding held-out queries (MiniLM, TA) …")
        heldout_minilm = encode_minilm(heldout_ta, device=None)
        np.save(held_min_path, heldout_minilm)

    print("Loading SPECTER2 corpus embeddings …")
    corpus_specter = np.load(specter_dir / "corpus_embeddings.npy").astype(np.float32)

    from sentence_transformers import SentenceTransformer

    specter_model = SentenceTransformer(SPECTER_MODEL)
    sep = specter_model.tokenizer.sep_token or "[SEP]"

    public_specter_texts = queries.apply(lambda r: format_specter(r, sep), axis=1).tolist()
    heldout_specter_texts = heldout.apply(lambda r: format_specter(r, sep), axis=1).tolist()

    pub_spec_path = out_dir / "public_specter_query_embeddings.npy"
    if pub_spec_path.exists():
        print(f"Loading cached public SPECTER2 query embeddings from {pub_spec_path} …")
        public_specter = np.load(pub_spec_path).astype(np.float32)
    else:
        print("Encoding public queries (SPECTER2) …")
        public_specter = encode_specter_with_model(specter_model, public_specter_texts)
        np.save(pub_spec_path, public_specter)

    held_spec_path = out_dir / "heldout_specter_query_embeddings.npy"
    if held_spec_path.exists():
        print(f"Loading cached held-out SPECTER2 query embeddings from {held_spec_path} …")
        heldout_specter = np.load(held_spec_path).astype(np.float32)
    else:
        print("Encoding held-out queries (SPECTER2) …")
        heldout_specter = encode_specter_with_model(specter_model, heldout_specter_texts)
        np.save(held_spec_path, heldout_specter)

    if len(corpus_minilm) != len(corpus_ids) or len(corpus_specter) != len(corpus_ids):
        raise ValueError("Corpus embedding rows do not match corpus_ids.")

    public_minilm_scores = (public_minilm @ corpus_minilm.T).astype(np.float32)
    heldout_minilm_scores = (heldout_minilm @ corpus_minilm.T).astype(np.float32)
    public_specter_scores = (public_specter @ corpus_specter.T).astype(np.float32)
    heldout_specter_scores = (heldout_specter @ corpus_specter.T).astype(np.float32)

    query_meta_public = queries.set_index("doc_id").loc[public_query_ids]
    query_meta_heldout = heldout.set_index("doc_id").loc[heldout_query_ids]
    corpus_domains = corpus["domain"].fillna("").to_numpy()
    corpus_venues = corpus["venue"].fillna("").to_numpy()
    corpus_years = years_as_int64(corpus["year"])

    qd_pub = query_meta_public["domain"].fillna("").to_numpy()
    qv_pub = query_meta_public["venue"].fillna("").to_numpy()
    qy_pub = years_as_int64(query_meta_public["year"])
    qd_hold = query_meta_heldout["domain"].fillna("").to_numpy()
    qv_hold = query_meta_heldout["venue"].fillna("").to_numpy()
    qy_hold = years_as_int64(query_meta_heldout["year"])

    print("Precomputing RRF(MiniLM dense, sparse) …")
    rrf_pub = rrf_from_full_scores(public_minilm_scores, public_sparse_scores, RRF_K)

    print("Tuning stage 1 (eta + Charlotte params, gamma=0) …")
    cfg1, ndcg1 = tune_joint(
        qrels,
        public_query_ids,
        corpus_ids,
        public_minilm_scores,
        public_specter_scores,
        public_sparse_scores,
        qd_pub,
        corpus_domains,
        qv_pub,
        corpus_venues,
        qy_pub,
        corpus_years,
        rrf_pub,
    )
    print(f"Stage-1 best public NDCG@10: {ndcg1:.5f} | {cfg1}")

    print("Tuning stage 2 (gamma RRF) …")
    cfg2, ndcg2 = tune_stage2_rrf(
        qrels,
        public_query_ids,
        corpus_ids,
        public_minilm_scores,
        public_specter_scores,
        public_sparse_scores,
        qd_pub,
        corpus_domains,
        qv_pub,
        corpus_venues,
        qy_pub,
        corpus_years,
        rrf_pub,
        cfg1,
    )
    print(f"Stage-2 best public NDCG@10: {ndcg2:.5f} | {cfg2}")

    rrf_hold = rrf_from_full_scores(heldout_minilm_scores, heldout_sparse_scores, RRF_K)

    heldout_submission = build_submission(
        heldout_query_ids,
        corpus_ids,
        heldout_minilm_scores,
        heldout_specter_scores,
        heldout_sparse_scores,
        qd_hold,
        corpus_domains,
        qv_hold,
        corpus_venues,
        qy_hold,
        corpus_years,
        eta=cfg2["eta"],
        alpha=cfg2["alpha"],
        domain_boost=cfg2["domain_boost"],
        venue_boost=cfg2["venue_boost"],
        year_boost=cfg2["year_boost"],
        gamma_rrf=cfg2["gamma_rrf"],
        rrf_norm=rrf_hold,
    )

    validate_submission(submission=heldout_submission, expected_query_ids=heldout_query_ids, top_k=TOP_K)
    validate_doc_ids_in_corpus(submission=heldout_submission, corpus_doc_ids=corpus_ids.tolist())
    save_submission(submission=heldout_submission, output_file=paths["output_file"])
    create_submission_zip(submission_file=paths["output_file"], zip_file=paths["zip_file"])
    print(f"Saved to: {paths['zip_file']}")


if __name__ == "__main__":
    main()
