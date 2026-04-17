#!/usr/bin/env python3
"""
Strong submission pipeline for the Scientific Article IR challenge.

What it does
------------
1. Loads the public queries / corpus / qrels.
2. Builds a sparse retriever (TF-IDF on title + abstract).
3. Loads MiniLM dense corpus embeddings and public query embeddings.
4. Tunes a small hybrid on the public set using the PRIMARY metric: NDCG@10.
5. Encodes held-out queries with the same MiniLM model (unless cached embeddings exist).
6. Produces:
   - submission_data.json
   - submission.zip

Why this works
--------------
The challenge is citation retrieval, and the public data strongly suggests:
- dense semantic retrieval helps;
- sparse lexical retrieval adds complementary signal;
- domain metadata is very predictive and safe to use because it is part of the provided data.

Hybrid score per query-document pair:
    final = alpha * dense_norm + (1 - alpha) * sparse_norm
          + domain_boost
          + venue_boost

The script tunes alpha / boosts on the 100 public queries, then applies the best
setting to held-out queries. It never uses any held-out qrels.

Example
-------
python ir_hybrid_submit.py \
  --challenge-dir /path/to/unzipped/starter_kit \
  --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--challenge-dir",
        type=Path,
        default=Path("."),
        help="Path to the unzipped starter kit root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./final_submission"),
        help="Where submission_data.json and submission.zip will be written.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Dense model used for held-out query encoding.",
    )
    parser.add_argument(
        "--heldout-query-embeddings",
        type=Path,
        default=None,
        help="Optional .npy file for precomputed held-out query embeddings.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional sentence-transformers device override, e.g. cpu / cuda.",
    )
    return parser.parse_args()


def format_ta(df: pd.DataFrame) -> List[str]:
    return (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist()


def load_json(path: Path):
    with open(path) as f:
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


def build_submission(
    query_ids: List[str],
    corpus_ids: np.ndarray,
    dense_scores: np.ndarray,
    sparse_scores: np.ndarray,
    query_domains: np.ndarray,
    corpus_domains: np.ndarray,
    query_venues: np.ndarray,
    corpus_venues: np.ndarray,
    alpha: float,
    domain_boost: float,
    venue_boost: float,
) -> Dict[str, List[str]]:
    domain_match = (query_domains[:, None] == corpus_domains[None, :]).astype(np.float32)
    venue_match = (
        (query_venues[:, None] == corpus_venues[None, :])
        & (query_venues[:, None] != "")
    ).astype(np.float32)

    dense_aug = dense_scores + domain_boost * domain_match + venue_boost * venue_match
    sparse_aug = sparse_scores + domain_boost * domain_match + venue_boost * venue_match

    final_scores = alpha * minmax_per_query(dense_aug) + (1.0 - alpha) * minmax_per_query(sparse_aug)

    top_idx = np.argsort(-final_scores, axis=1)[:, :100]
    submission: Dict[str, List[str]] = {}
    for i, qid in enumerate(query_ids):
        ranked = corpus_ids[top_idx[i]].tolist()
        # Defensive: never retrieve the query paper itself if it is present in corpus.
        ranked = [doc_id for doc_id in ranked if doc_id != qid]
        if len(ranked) < 100:
            extras = [doc_id for doc_id in corpus_ids[np.argsort(-final_scores[i])] if doc_id not in set(ranked) and doc_id != qid]
            ranked.extend(extras[: 100 - len(ranked)])
        submission[qid] = ranked[:100]
    return submission


def tune_hybrid(
    qrels: Dict[str, List[str]],
    public_query_ids: List[str],
    corpus_ids: np.ndarray,
    dense_scores: np.ndarray,
    sparse_scores: np.ndarray,
    query_domains: np.ndarray,
    corpus_domains: np.ndarray,
    query_venues: np.ndarray,
    corpus_venues: np.ndarray,
) -> Tuple[dict, float]:
    best_cfg = None
    best_ndcg = -1.0

    for alpha in [0.50, 0.60, 0.65, 0.70, 0.80]:
        for domain_boost in [0.05, 0.10, 0.15]:
            for venue_boost in [0.00, 0.01, 0.02]:
                sub = build_submission(
                    public_query_ids,
                    corpus_ids,
                    dense_scores,
                    sparse_scores,
                    query_domains,
                    corpus_domains,
                    query_venues,
                    corpus_venues,
                    alpha=alpha,
                    domain_boost=domain_boost,
                    venue_boost=venue_boost,
                )
                score = evaluate_ndcg10(sub, qrels)
                if score > best_ndcg:
                    best_ndcg = score
                    best_cfg = {
                        "alpha": alpha,
                        "domain_boost": domain_boost,
                        "venue_boost": venue_boost,
                    }
    assert best_cfg is not None
    return best_cfg, best_ndcg


def encode_heldout_queries(texts: List[str], model_name: str, device: str | None) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    kwargs = {}
    if device:
        kwargs["device"] = device
    model = SentenceTransformer(model_name, **kwargs)
    embs = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embs.astype(np.float32)


def main() -> None:
    args = parse_args()
    root = args.challenge_dir
    data_dir = root / "data"
    emb_dir = data_dir / "embeddings" / "sentence-transformers_all-MiniLM-L6-v2"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    queries = pd.read_parquet(data_dir / "queries.parquet")
    heldout = pd.read_parquet(root / "held_out_queries.parquet")
    corpus = pd.read_parquet(data_dir / "corpus.parquet")
    qrels = load_json(data_dir / "qrels.json")

    public_query_ids = queries["doc_id"].tolist()
    heldout_query_ids = heldout["doc_id"].tolist()
    corpus_ids = np.array(load_json(emb_dir / "corpus_ids.json"))
    public_dense_ids = load_json(emb_dir / "query_ids.json")

    print("Building sparse index (TF-IDF title+abstract, uni+bi-grams)...")
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

    public_sparse_scores = cosine_similarity(public_sparse, corpus_sparse)
    heldout_sparse_scores = cosine_similarity(heldout_sparse, corpus_sparse)

    print("Loading dense embeddings...")
    corpus_dense = np.load(emb_dir / "corpus_embeddings.npy").astype(np.float32)
    public_dense = np.load(emb_dir / "query_embeddings.npy").astype(np.float32)

    if public_dense_ids != public_query_ids:
        raise ValueError("Public query embedding IDs do not match queries.parquet order.")

    if args.heldout_query_embeddings and args.heldout_query_embeddings.exists():
        print(f"Loading held-out query embeddings from {args.heldout_query_embeddings}")
        heldout_dense = np.load(args.heldout_query_embeddings).astype(np.float32)
    else:
        cache_path = args.output_dir / "heldout_query_embeddings.npy"
        if cache_path.exists():
            print(f"Loading cached held-out embeddings from {cache_path}")
            heldout_dense = np.load(cache_path).astype(np.float32)
        else:
            print("Encoding held-out queries with sentence-transformers...")
            heldout_dense = encode_heldout_queries(heldout_ta, args.model_name, args.device)
            np.save(cache_path, heldout_dense)
            print(f"Saved held-out embeddings to {cache_path}")

    if len(corpus_dense) != len(corpus_ids):
        raise ValueError("Corpus embedding count does not match corpus_ids.json")
    if len(heldout_dense) != len(heldout_query_ids):
        raise ValueError("Held-out embedding count does not match held_out_queries.parquet")

    public_dense_scores = public_dense @ corpus_dense.T
    heldout_dense_scores = heldout_dense @ corpus_dense.T

    query_meta_public = queries.set_index("doc_id").loc[public_query_ids]
    query_meta_heldout = heldout.set_index("doc_id").loc[heldout_query_ids]

    corpus_domains = corpus["domain"].fillna("").to_numpy()
    corpus_venues = corpus["venue"].fillna("").to_numpy()

    print("Tuning hybrid on public queries (objective: NDCG@10)...")
    best_cfg, best_ndcg = tune_hybrid(
        qrels=qrels,
        public_query_ids=public_query_ids,
        corpus_ids=corpus_ids,
        dense_scores=public_dense_scores,
        sparse_scores=public_sparse_scores,
        query_domains=query_meta_public["domain"].fillna("").to_numpy(),
        corpus_domains=corpus_domains,
        query_venues=query_meta_public["venue"].fillna("").to_numpy(),
        corpus_venues=corpus_venues,
    )
    print(f"Best public NDCG@10: {best_ndcg:.5f}")
    print(f"Best config: {best_cfg}")

    heldout_submission = build_submission(
        query_ids=heldout_query_ids,
        corpus_ids=corpus_ids,
        dense_scores=heldout_dense_scores,
        sparse_scores=heldout_sparse_scores,
        query_domains=query_meta_heldout["domain"].fillna("").to_numpy(),
        corpus_domains=corpus_domains,
        query_venues=query_meta_heldout["venue"].fillna("").to_numpy(),
        corpus_venues=corpus_venues,
        alpha=best_cfg["alpha"],
        domain_boost=best_cfg["domain_boost"],
        venue_boost=best_cfg["venue_boost"],
    )

    # Validate format strictly.
    missing = set(heldout_query_ids) - set(heldout_submission)
    if missing:
        raise ValueError(f"Missing predictions for {len(missing)} held-out queries")
    for qid in heldout_query_ids:
        docs = heldout_submission[qid]
        if len(docs) != 100:
            raise ValueError(f"{qid} has {len(docs)} docs instead of 100")
        if len(set(docs)) != 100:
            raise ValueError(f"{qid} has duplicate doc IDs")

    submission_json = args.output_dir / "submission_data.json"
    with open(submission_json, "w") as f:
        json.dump(heldout_submission, f)

    submission_zip = args.output_dir / "submission.zip"
    with zipfile.ZipFile(submission_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(submission_json, arcname="submission_data.json")

    print(f"Wrote: {submission_json}")
    print(f"Wrote: {submission_zip}")
    print("Done.")


if __name__ == "__main__":
    main()
