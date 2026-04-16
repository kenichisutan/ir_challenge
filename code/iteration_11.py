from __future__ import annotations

"""
Iteration 11 — learned fusion ranker (sklearn) on public qrels.

Train on public queries using candidate pools from BM25 + MiniLM + SPECTER2,
learn a non-linear relevance model, then infer rankings for held-out queries.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestRegressor

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

ITERATION_NAME = "iteration_11"
TOP_K = DEFAULT_TOP_K
POOL_K = 350
SPECTER2_MODEL = "allenai/specter2_base"
SPECTER_BATCH = 32
MIN_POS_PER_QUERY = 1


def challenge_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_public_train_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, set[str]]]:
    root = challenge_root()
    queries = pd.read_parquet(root / "data" / "queries.parquet")
    corpus = pd.read_parquet(root / "data" / "corpus.parquet")
    with (root / "data" / "qrels.json").open("r", encoding="utf-8") as f:
        raw = json.load(f)
    qrels = {str(qid): {str(doc) for doc in docs} for qid, docs in raw.items()}
    return queries, corpus, qrels


def specter_dir() -> Path:
    return challenge_root() / "data" / "embeddings" / "allenai_specter2_base"


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


def bm25_rank_and_score(query_texts: List[str], corpus_texts: List[str], top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    tokenized_corpus = [t.lower().split() for t in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus, k1=1.2, b=0.75)
    n_queries = len(query_texts)
    ranks = np.zeros((n_queries, top_k), dtype=np.int64)
    scores = np.zeros((n_queries, top_k), dtype=np.float32)
    for i, q in enumerate(query_texts):
        s = bm25.get_scores(q.lower().split())
        order = np.argsort(-s, kind="stable")[:top_k]
        ranks[i] = order
        scores[i] = s[order].astype(np.float32)
    return ranks, scores


def dense_rank_and_score(query_emb: np.ndarray, corpus_emb: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    sim = query_emb @ corpus_emb.T
    ranks = np.argsort(-sim, axis=1, kind="stable")[:, :top_k]
    rows = np.arange(sim.shape[0])[:, None]
    scores = sim[rows, ranks].astype(np.float32)
    return ranks, scores


def build_rank_pos(rank_idx: np.ndarray, n_docs: int) -> np.ndarray:
    n_q, depth = rank_idx.shape
    out = np.full((n_q, n_docs), fill_value=depth + 1, dtype=np.int16)
    for i in range(n_q):
        out[i, rank_idx[i]] = np.arange(1, depth + 1, dtype=np.int16)
    return out


def build_training_matrix(
    qids: List[str],
    corpus_ids: List[str],
    qrels: Dict[str, set[str]],
    bm25_ranks: np.ndarray,
    bm25_scores: np.ndarray,
    minilm_ranks: np.ndarray,
    minilm_scores: np.ndarray,
    specter_ranks: np.ndarray,
    specter_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n_docs = len(corpus_ids)
    bm25_pos = build_rank_pos(bm25_ranks, n_docs)
    minilm_pos = build_rank_pos(minilm_ranks, n_docs)
    specter_pos = build_rank_pos(specter_ranks, n_docs)

    X_rows: List[List[float]] = []
    y: List[float] = []
    doc_id_by_idx = corpus_ids

    for i, qid in enumerate(qids):
        cand = set(bm25_ranks[i].tolist()) | set(minilm_ranks[i].tolist()) | set(specter_ranks[i].tolist())
        rel = qrels.get(qid, set())
        if len(rel) < MIN_POS_PER_QUERY:
            continue

        # Fast lookup maps for top-k scores
        b_s = {int(idx): float(score) for idx, score in zip(bm25_ranks[i], bm25_scores[i])}
        m_s = {int(idx): float(score) for idx, score in zip(minilm_ranks[i], minilm_scores[i])}
        s_s = {int(idx): float(score) for idx, score in zip(specter_ranks[i], specter_scores[i])}

        for j in cand:
            pb = int(bm25_pos[i, j])
            pm = int(minilm_pos[i, j])
            ps = int(specter_pos[i, j])
            features = [
                b_s.get(j, -50.0),
                m_s.get(j, -1.0),
                s_s.get(j, -1.0),
                1.0 / pb,
                1.0 / pm,
                1.0 / ps,
                (1.0 / pb + 1.0 / pm + 1.0 / ps),
            ]
            X_rows.append(features)
            y.append(1.0 if doc_id_by_idx[j] in rel else 0.0)

    return np.asarray(X_rows, dtype=np.float32), np.asarray(y, dtype=np.float32)


def predict_query_scores(
    model: RandomForestRegressor,
    i: int,
    n_docs: int,
    bm25_ranks: np.ndarray,
    bm25_scores: np.ndarray,
    minilm_ranks: np.ndarray,
    minilm_scores: np.ndarray,
    specter_ranks: np.ndarray,
    specter_scores: np.ndarray,
) -> np.ndarray:
    # build per-query candidate pool from all three retrievers
    cand = np.array(sorted(set(bm25_ranks[i].tolist()) | set(minilm_ranks[i].tolist()) | set(specter_ranks[i].tolist())), dtype=np.int64)
    depth = bm25_ranks.shape[1]
    pos_b = np.full(n_docs, depth + 1, dtype=np.int16); pos_b[bm25_ranks[i]] = np.arange(1, depth + 1)
    pos_m = np.full(n_docs, depth + 1, dtype=np.int16); pos_m[minilm_ranks[i]] = np.arange(1, depth + 1)
    pos_s = np.full(n_docs, depth + 1, dtype=np.int16); pos_s[specter_ranks[i]] = np.arange(1, depth + 1)

    b_s = {int(idx): float(score) for idx, score in zip(bm25_ranks[i], bm25_scores[i])}
    m_s = {int(idx): float(score) for idx, score in zip(minilm_ranks[i], minilm_scores[i])}
    s_s = {int(idx): float(score) for idx, score in zip(specter_ranks[i], specter_scores[i])}

    X = []
    for j in cand:
        pb = int(pos_b[j]); pm = int(pos_m[j]); ps = int(pos_s[j])
        X.append([
            b_s.get(int(j), -50.0),
            m_s.get(int(j), -1.0),
            s_s.get(int(j), -1.0),
            1.0 / pb, 1.0 / pm, 1.0 / ps,
            (1.0 / pb + 1.0 / pm + 1.0 / ps),
        ])
    X = np.asarray(X, dtype=np.float32)
    pred = model.predict(X)

    # fallback prior from reciprocal ranks (keeps stable ordering if predictions tie)
    prior = (1.0 / pos_b[cand]) + (1.0 / pos_m[cand]) + (1.0 / pos_s[cand])
    final = pred + 0.02 * prior
    order = np.argsort(-final, kind="stable")
    ranked = cand[order]
    return ranked[:TOP_K]


def main() -> None:
    # held-out target
    data = data_paths()
    if not data["using_held_out_queries"]:
        raise RuntimeError(
            "Submission mode requires held-out queries. "
            "Add `held_out_queries.parquet` to `data/`, project root, or `starter_kit/`."
        )
    held_queries, corpus = load_queries_corpus(data["queries_path"], data["corpus_path"])
    corpus_ids = corpus["doc_id"].astype(str).tolist()
    corpus_id_to_idx = {d: i for i, d in enumerate(corpus_ids)}

    # public train data
    train_queries, train_corpus, qrels = load_public_train_data()
    train_corpus_ids = train_corpus["doc_id"].astype(str).tolist()
    if train_corpus_ids != corpus_ids:
        raise ValueError("Expected same corpus order for public and held-out pipelines.")

    # common text views
    held_query_rich = held_queries.apply(lambda r: i4.format_rich_text(r, i4.MAX_FULLTEXT_CHARS), axis=1).tolist()
    train_query_rich = train_queries.apply(lambda r: i4.format_rich_text(r, i4.MAX_FULLTEXT_CHARS), axis=1).tolist()
    corpus_rich = corpus.apply(lambda r: i4.format_rich_text(r, i4.MAX_FULLTEXT_CHARS), axis=1).tolist()

    # sparse BM25
    print("BM25 retrieval for train/held …")
    train_bm25_r, train_bm25_s = bm25_rank_and_score(train_query_rich, corpus_rich, top_k=POOL_K)
    held_bm25_r, held_bm25_s = bm25_rank_and_score(held_query_rich, corpus_rich, top_k=POOL_K)

    # dense MiniLM rich
    print(f"Loading {i4.MINILM_MODEL} …")
    minilm = SentenceTransformer(i4.MINILM_MODEL)
    print("Encoding corpus with MiniLM …")
    corpus_emb_minilm = i4.encode_minilm(minilm, corpus_rich)
    print("Encoding train queries with MiniLM …")
    train_emb_minilm = i4.encode_minilm(minilm, train_query_rich)
    print("Encoding held-out queries with MiniLM …")
    held_emb_minilm = i4.encode_minilm(minilm, held_query_rich)
    train_minilm_r, train_minilm_s = dense_rank_and_score(train_emb_minilm, corpus_emb_minilm, top_k=POOL_K)
    held_minilm_r, held_minilm_s = dense_rank_and_score(held_emb_minilm, corpus_emb_minilm, top_k=POOL_K)

    # dense SPECTER2 TA
    print(f"Loading {SPECTER2_MODEL} …")
    specter = SentenceTransformer(SPECTER2_MODEL)
    sep = specter.tokenizer.sep_token or "[SEP]"
    train_query_specter = train_queries.apply(lambda r: format_specter(r, sep), axis=1).tolist()
    held_query_specter = held_queries.apply(lambda r: format_specter(r, sep), axis=1).tolist()
    corpus_specter = corpus.apply(lambda r: format_specter(r, sep), axis=1).tolist()
    corpus_emb_s = load_or_cache_specter_corpus(specter, corpus_specter, corpus_ids)
    print("Encoding train queries with SPECTER2 …")
    train_emb_s = encode_specter(specter, train_query_specter)
    print("Encoding held-out queries with SPECTER2 …")
    held_emb_s = encode_specter(specter, held_query_specter)
    train_s_r, train_s_s = dense_rank_and_score(train_emb_s, corpus_emb_s, top_k=POOL_K)
    held_s_r, held_s_s = dense_rank_and_score(held_emb_s, corpus_emb_s, top_k=POOL_K)

    # build train matrix
    train_qids = train_queries["doc_id"].astype(str).tolist()
    X, y = build_training_matrix(
        qids=train_qids,
        corpus_ids=corpus_ids,
        qrels=qrels,
        bm25_ranks=train_bm25_r,
        bm25_scores=train_bm25_s,
        minilm_ranks=train_minilm_r,
        minilm_scores=train_minilm_s,
        specter_ranks=train_s_r,
        specter_scores=train_s_s,
    )
    print(f"Training rows: {len(X)} | positives: {int(y.sum())}")
    if len(X) == 0 or y.sum() == 0:
        raise RuntimeError("Training matrix is empty or has no positives.")

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y)

    # infer held-out
    held_qids = held_queries["doc_id"].astype(str).tolist()
    submission: Dict[str, List[str]] = {}
    n_docs = len(corpus_ids)
    for i, qid in enumerate(held_qids):
        ranked_idx = predict_query_scores(
            model=model,
            i=i,
            n_docs=n_docs,
            bm25_ranks=held_bm25_r,
            bm25_scores=held_bm25_s,
            minilm_ranks=held_minilm_r,
            minilm_scores=held_minilm_s,
            specter_ranks=held_s_r,
            specter_scores=held_s_s,
        )
        # pad if candidate pool unexpectedly smaller than TOP_K
        if len(ranked_idx) < TOP_K:
            seen = set(ranked_idx.tolist())
            filler = [j for j in held_bm25_r[i].tolist() if j not in seen]
            ranked_idx = np.array((ranked_idx.tolist() + filler)[:TOP_K], dtype=np.int64)
        submission[qid] = [corpus_ids[int(j)] for j in ranked_idx]

    paths = {**data, **iteration_submission_paths(ITERATION_NAME)}
    validate_submission(submission=submission, expected_query_ids=held_qids, top_k=TOP_K)
    validate_doc_ids_in_corpus(submission=submission, corpus_doc_ids=corpus_ids)
    save_submission(submission=submission, output_file=paths["output_file"])
    create_submission_zip(submission_file=paths["output_file"], zip_file=paths["zip_file"])

    print("Iteration 11 submission generated.")
    print("Method: sklearn learned fusion over BM25 + MiniLM + SPECTER2 candidate features.")
    print(f"Saved to: {paths['zip_file']}")


if __name__ == "__main__":
    main()
