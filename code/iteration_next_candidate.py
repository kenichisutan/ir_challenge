#!/usr/bin/env python3
"""
Iteration C1 + SPECTER2 **v2 (full text)** — same hybrid as `iteration_c1_specter.py`,
but retrieval text uses **title + abstract + full_text** instead of title+abstract only.

**Sparse (TF-IDF):** concatenated fields; each document string is truncated to
`SPARSE_MAX_CHARS` characters (head preserved) so the vocabulary fit stays practical.

**Dense (MiniLM + SPECTER2):** starter-kit **TA-only** `.npy` caches are not used for
scores. Corpus and all queries are **re-encoded** with the same full-text string
(model truncation keeps the **beginning**, so title/abstract are placed first).
Embeddings are cached under this iteration's `submissions/<name>/` folder
(`*_fulltext.npy`); first run is slow, later runs load from disk.

Hyperparameters are tuned on public `qrels.json` (NDCG@10) like v1.
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

ITERATION_NAME = "iteration_next_candidate"
TOP_K = DEFAULT_TOP_K

MINILM_EMB = "sentence-transformers_all-MiniLM-L6-v2"
SPECTER_EMB = "allenai_specter2_base"
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPECTER_MODEL = "allenai/specter2_base"
MINILM_ENCODE_BATCH = 128
SPECTER_ENCODE_BATCH = 32
RRF_K = 60.0

# TF-IDF: cap document length so the sparse matrix stays tractable on 20k papers.
SPARSE_MAX_CHARS = 80_000


def format_dense_document(row: pd.Series) -> str:
    """Single string for bi-encoders; model truncates by token count from the start."""
    title = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    body = str(row.get("full_text", "") or "").strip()
    parts = [p for p in (title, abstract, body) if p]
    return " ".join(parts)


def format_sparse_document(row: pd.Series, max_chars: int = SPARSE_MAX_CHARS) -> str:
    s = format_dense_document(row)
    if len(s) > max_chars:
        return s[:max_chars]
    return s


def format_sparse_ta(row: pd.Series) -> str:
    title = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    return f"{title} {abstract}".strip()


def format_specter_fulltext(row: pd.Series, sep: str) -> str:
    """SPECTER-style segments: title SEP abstract SEP body (empty parts skipped)."""
    title = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    body = str(row.get("full_text", "") or "").strip()
    parts = [p for p in (title, abstract, body) if p]
    if not parts:
        return ""
    return f" {sep} ".join(parts)


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
    alpha_domain_boost: float,
    alpha_gap_delta: float,
    conf_quantile: float,
    domain_boost: float,
    venue_boost: float,
    year_boost: float,
    year_close_boost: float,
    gamma_rrf: float,
    rrf_norm: np.ndarray,
    hard_domain_first: bool,
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
    year_close_ok = (qy >= 0) & (cy >= 0) & (np.abs(qy - cy) <= 1)
    year_close = (year_close_ok & ~year_ok).astype(np.float32)

    dense_aug = (
        dense_scores
        + domain_boost * domain_match
        + venue_boost * venue_match
        + year_boost * year_match
        + year_close_boost * year_close
    )
    sparse_aug = (
        sparse_scores.astype(np.float32)
        + domain_boost * domain_match
        + venue_boost * venue_match
        + year_boost * year_match
        + year_close_boost * year_close
    )

    dense_norm = minmax_per_query(dense_aug)
    sparse_norm = minmax_per_query(sparse_aug)
    q_counts: Dict[str, int] = {}
    for d in query_domains.tolist():
        s = str(d)
        q_counts[s] = q_counts.get(s, 0) + 1
    rare = np.array([1.0 if q_counts.get(str(d), 0) <= 4 else 0.0 for d in query_domains], dtype=np.float32)[:, None]
    alpha_q = np.clip((alpha - alpha_domain_boost * rare).astype(np.float32), 0.20, 0.90)
    if alpha_gap_delta > 0.0:
        ord2 = np.argsort(-sparse_aug, axis=1)[:, :2]
        top1 = sparse_aug[np.arange(sparse_aug.shape[0]), ord2[:, 0]]
        top2 = sparse_aug[np.arange(sparse_aug.shape[0]), ord2[:, 1]]
        gaps = top1 - top2
        thresh = float(np.quantile(gaps, conf_quantile))
        sharp = (gaps >= thresh).astype(np.float32)[:, None]
        # High sparse-confidence queries lean a bit more sparse.
        alpha_q = np.clip(alpha_q - alpha_gap_delta * sharp + alpha_gap_delta * (1.0 - sharp), 0.20, 0.90)
    charlotte = alpha_q * dense_norm + (1.0 - alpha_q) * sparse_norm
    final_scores = (1.0 - gamma_rrf) * charlotte + gamma_rrf * rrf_norm

    submission: Dict[str, List[str]] = {}
    for i, qid in enumerate(query_ids):
        base_order = np.argsort(-final_scores[i], kind="stable")
        if hard_domain_first and str(query_domains[i]) != "":
            dm = domain_match[i].astype(bool)
            base_order = np.concatenate([base_order[dm[base_order]], base_order[~dm[base_order]]])
        ranked = [str(corpus_ids[j]) for j in base_order if str(corpus_ids[j]) != qid][:TOP_K]
        if len(ranked) < TOP_K:
            seen = set(ranked)
            extras = [
                doc_id
                for doc_id in corpus_ids[np.argsort(-final_scores[i], kind="stable")]
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


def corpus_df_ordered(corpus: pd.DataFrame, corpus_ids: np.ndarray) -> pd.DataFrame:
    ids = [str(x) for x in corpus_ids.tolist()]
    by = corpus.set_index(corpus["doc_id"].astype(str), drop=False)
    missing = set(ids) - set(by.index)
    if missing:
        raise ValueError(f"{len(missing)} corpus_ids missing from corpus.parquet")
    return by.loc[ids]


def load_or_encode_minilm_matrix(
    path: Path,
    texts: List[str],
    expected_rows: int,
    device: str | None,
) -> np.ndarray:
    if path.exists():
        arr = np.load(path).astype(np.float32)
        if arr.shape[0] == expected_rows:
            print(f"Loading cached MiniLM matrix {path} ({expected_rows} rows) …")
            return arr
        print(f"Cache {path} has wrong shape {arr.shape}; re-encoding …")
    print(f"Encoding MiniLM → {path} ({expected_rows} texts) …")
    emb = encode_minilm(texts, device)
    np.save(path, emb)
    return emb


def load_or_encode_specter_matrix(
    path: Path,
    model,
    texts: List[str],
    expected_rows: int,
) -> np.ndarray:
    if path.exists():
        arr = np.load(path).astype(np.float32)
        if arr.shape[0] == expected_rows:
            print(f"Loading cached SPECTER2 matrix {path} ({expected_rows} rows) …")
            return arr
        print(f"Cache {path} has wrong shape {arr.shape}; re-encoding …")
    print(f"Encoding SPECTER2 → {path} ({expected_rows} texts) …")
    emb = encode_specter_with_model(model, texts)
    np.save(path, emb)
    return emb


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
    sparse_ta_scores: np.ndarray,
) -> Tuple[dict, float]:
    best_cfg = None
    best_ndcg = -1.0
    for eta in [0.32]:
        for alpha in [0.29, 0.30, 0.31]:
            for alpha_domain_boost in [0.01]:
                for alpha_gap_delta in [0.005]:
                    for conf_quantile in [0.62, 0.65, 0.68, 0.70]:
                        for domain_boost in [0.08]:
                            for venue_boost in [0.005]:
                                for year_boost in [0.00]:
                                    for year_close_boost in [0.0, 0.002, 0.004, 0.006]:
                                        for sparse_beta in [0.99]:
                                            for hard_domain_first in [False, True]:
                                                sparse_mix = sparse_beta * sparse_scores + (1.0 - sparse_beta) * sparse_ta_scores
                                                sub = build_submission(
                                                    public_query_ids,
                                                    corpus_ids,
                                                    minilm_scores,
                                                    specter_scores,
                                                    sparse_mix,
                                                    query_domains,
                                                    corpus_domains,
                                                    query_venues,
                                                    corpus_venues,
                                                    query_years,
                                                    corpus_years,
                                                    eta=eta,
                                                    alpha=alpha,
                                                    alpha_domain_boost=alpha_domain_boost,
                                                    alpha_gap_delta=alpha_gap_delta,
                                                    conf_quantile=conf_quantile,
                                                    domain_boost=domain_boost,
                                                    venue_boost=venue_boost,
                                                    year_boost=year_boost,
                                                    year_close_boost=year_close_boost,
                                                    gamma_rrf=0.0,
                                                    rrf_norm=rrf_norm,
                                                    hard_domain_first=hard_domain_first,
                                                )
                                                score = evaluate_ndcg10(sub, qrels)
                                                if score > best_ndcg:
                                                    best_ndcg = score
                                                    best_cfg = {
                                                        "eta": eta,
                                                        "alpha": alpha,
                                                        "alpha_domain_boost": alpha_domain_boost,
                                                        "alpha_gap_delta": alpha_gap_delta,
                                                        "conf_quantile": conf_quantile,
                                                        "domain_boost": domain_boost,
                                                        "venue_boost": venue_boost,
                                                        "year_boost": year_boost,
                                                        "year_close_boost": year_close_boost,
                                                        "sparse_beta": sparse_beta,
                                                        "hard_domain_first": hard_domain_first,
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
    base: dict,
    sparse_ta_scores: np.ndarray,
) -> Tuple[dict, float]:
    best = {**base, "gamma_rrf": 0.0, "eta_rrf": base["eta"]}
    best_ndcg = -1.0
    sparse_mix = base["sparse_beta"] * sparse_scores + (1.0 - base["sparse_beta"]) * sparse_ta_scores
    for eta_rrf in [max(0.0, base["eta"] - 0.04), base["eta"], min(1.0, base["eta"] + 0.04)]:
        dense_for_rrf = (1.0 - eta_rrf) * minilm_scores + eta_rrf * specter_scores
        rrf_norm = rrf_from_full_scores(dense_for_rrf, sparse_mix, RRF_K)
        for gamma_rrf in [0.0, 0.001, 0.002, 0.004]:
            sub = build_submission(
                public_query_ids,
                corpus_ids,
                minilm_scores,
                specter_scores,
                sparse_mix,
                query_domains,
                corpus_domains,
                query_venues,
                corpus_venues,
                query_years,
                corpus_years,
                eta=base["eta"],
                alpha=base["alpha"],
                alpha_domain_boost=base["alpha_domain_boost"],
                alpha_gap_delta=base["alpha_gap_delta"],
                conf_quantile=base["conf_quantile"],
                domain_boost=base["domain_boost"],
                venue_boost=base["venue_boost"],
                year_boost=base["year_boost"],
                year_close_boost=base["year_close_boost"],
                gamma_rrf=gamma_rrf,
                rrf_norm=rrf_norm,
                hard_domain_first=base["hard_domain_first"],
            )
            score = evaluate_ndcg10(sub, qrels)
            if score > best_ndcg:
                best_ndcg = score
                best = {**base, "gamma_rrf": gamma_rrf, "eta_rrf": eta_rrf}
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
    embed_cache = root / "submissions" / "iteration_c1_specter_v2_fulltext"
    if (embed_cache / "corpus_minilm_fulltext.npy").exists():
        print(f"Reusing dense embedding cache from {embed_cache} …")
    else:
        embed_cache = out_dir
    embed_cache.mkdir(parents=True, exist_ok=True)

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

    corpus_ordered = corpus_df_ordered(corpus, corpus_ids)
    corpus_domains = corpus_ordered["domain"].fillna("").to_numpy()
    corpus_venues = corpus_ordered["venue"].fillna("").to_numpy()
    corpus_years = years_as_int64(corpus_ordered["year"])

    print(
        f"Building sparse index (TF-IDF title+abstract+full_text, ≤{SPARSE_MAX_CHARS} chars/doc, uni+bi-grams) …"
    )
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        stop_words="english",
    )
    corpus_sparse_texts = [format_sparse_document(r) for _, r in corpus_ordered.iterrows()]
    public_sparse_texts = [format_sparse_document(r) for _, r in queries.iterrows()]
    heldout_sparse_texts = [format_sparse_document(r) for _, r in heldout.iterrows()]
    corpus_sparse = vectorizer.fit_transform(corpus_sparse_texts)
    public_sparse = vectorizer.transform(public_sparse_texts)
    heldout_sparse = vectorizer.transform(heldout_sparse_texts)
    public_sparse_scores = cosine_similarity(public_sparse, corpus_sparse).astype(np.float32)
    heldout_sparse_scores = cosine_similarity(heldout_sparse, corpus_sparse).astype(np.float32)

    print("Building TA-only sparse channel (TF-IDF title+abstract) …")
    vectorizer_ta = TfidfVectorizer(
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        stop_words="english",
    )
    corpus_sparse_ta_texts = [format_sparse_ta(r) for _, r in corpus_ordered.iterrows()]
    public_sparse_ta_texts = [format_sparse_ta(r) for _, r in queries.iterrows()]
    heldout_sparse_ta_texts = [format_sparse_ta(r) for _, r in heldout.iterrows()]
    corpus_sparse_ta = vectorizer_ta.fit_transform(corpus_sparse_ta_texts)
    public_sparse_ta = vectorizer_ta.transform(public_sparse_ta_texts)
    heldout_sparse_ta = vectorizer_ta.transform(heldout_sparse_ta_texts)
    public_sparse_ta_scores = cosine_similarity(public_sparse_ta, corpus_sparse_ta).astype(np.float32)
    heldout_sparse_ta_scores = cosine_similarity(heldout_sparse_ta, corpus_sparse_ta).astype(np.float32)

    from sentence_transformers import SentenceTransformer

    specter_model = SentenceTransformer(SPECTER_MODEL)
    sep = specter_model.tokenizer.sep_token or "[SEP]"

    corpus_dense_texts = [format_dense_document(r) for _, r in corpus_ordered.iterrows()]
    public_dense_texts = [format_dense_document(r) for _, r in queries.iterrows()]
    heldout_dense_texts = [format_dense_document(r) for _, r in heldout.iterrows()]

    corpus_specter_texts = [format_specter_fulltext(r, sep) for _, r in corpus_ordered.iterrows()]
    public_specter_texts = [format_specter_fulltext(r, sep) for _, r in queries.iterrows()]
    heldout_specter_texts = [format_specter_fulltext(r, sep) for _, r in heldout.iterrows()]

    n_corpus = len(corpus_ids)
    corpus_minilm = load_or_encode_minilm_matrix(
        embed_cache / "corpus_minilm_fulltext.npy",
        corpus_dense_texts,
        n_corpus,
        device=None,
    )
    public_minilm = load_or_encode_minilm_matrix(
        embed_cache / "public_minilm_query_fulltext.npy",
        public_dense_texts,
        len(public_query_ids),
        device=None,
    )
    heldout_minilm = load_or_encode_minilm_matrix(
        embed_cache / "heldout_minilm_query_fulltext.npy",
        heldout_dense_texts,
        len(heldout_query_ids),
        device=None,
    )

    corpus_specter = load_or_encode_specter_matrix(
        embed_cache / "corpus_specter2_fulltext.npy",
        specter_model,
        corpus_specter_texts,
        n_corpus,
    )
    public_specter = load_or_encode_specter_matrix(
        embed_cache / "public_specter2_query_fulltext.npy",
        specter_model,
        public_specter_texts,
        len(public_query_ids),
    )
    heldout_specter = load_or_encode_specter_matrix(
        embed_cache / "heldout_specter2_query_fulltext.npy",
        specter_model,
        heldout_specter_texts,
        len(heldout_query_ids),
    )

    if len(corpus_minilm) != len(corpus_ids) or len(corpus_specter) != len(corpus_ids):
        raise ValueError("Corpus embedding rows do not match corpus_ids.")

    public_minilm_scores = (public_minilm @ corpus_minilm.T).astype(np.float32)
    heldout_minilm_scores = (heldout_minilm @ corpus_minilm.T).astype(np.float32)
    public_specter_scores = (public_specter @ corpus_specter.T).astype(np.float32)
    heldout_specter_scores = (heldout_specter @ corpus_specter.T).astype(np.float32)

    query_meta_public = queries.set_index("doc_id").loc[public_query_ids]
    query_meta_heldout = heldout.set_index("doc_id").loc[heldout_query_ids]

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
        public_sparse_ta_scores,
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
        cfg1,
        public_sparse_ta_scores,
    )
    print(f"Stage-2 best public NDCG@10: {ndcg2:.5f} | {cfg2}")

    sparse_mix_hold = cfg2["sparse_beta"] * heldout_sparse_scores + (1.0 - cfg2["sparse_beta"]) * heldout_sparse_ta_scores
    dense_hold_rrf = (1.0 - cfg2["eta_rrf"]) * heldout_minilm_scores + cfg2["eta_rrf"] * heldout_specter_scores
    rrf_hold = rrf_from_full_scores(dense_hold_rrf, sparse_mix_hold, RRF_K)

    heldout_submission = build_submission(
        heldout_query_ids,
        corpus_ids,
        heldout_minilm_scores,
        heldout_specter_scores,
        sparse_mix_hold,
        qd_hold,
        corpus_domains,
        qv_hold,
        corpus_venues,
        qy_hold,
        corpus_years,
        eta=cfg2["eta"],
        alpha=cfg2["alpha"],
        alpha_domain_boost=cfg2["alpha_domain_boost"],
        alpha_gap_delta=cfg2["alpha_gap_delta"],
        conf_quantile=cfg2["conf_quantile"],
        domain_boost=cfg2["domain_boost"],
        venue_boost=cfg2["venue_boost"],
        year_boost=cfg2["year_boost"],
        year_close_boost=cfg2["year_close_boost"],
        gamma_rrf=cfg2["gamma_rrf"],
        rrf_norm=rrf_hold,
        hard_domain_first=cfg2["hard_domain_first"],
    )

    validate_submission(submission=heldout_submission, expected_query_ids=heldout_query_ids, top_k=TOP_K)
    validate_doc_ids_in_corpus(submission=heldout_submission, corpus_doc_ids=corpus_ids.tolist())
    save_submission(submission=heldout_submission, output_file=paths["output_file"])
    create_submission_zip(submission_file=paths["output_file"], zip_file=paths["zip_file"])
    print(f"Saved to: {paths['zip_file']}")


if __name__ == "__main__":
    main()
