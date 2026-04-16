from __future__ import annotations

"""
Iteration 11b — learned fusion (query-normalized + balanced sampling).

Why this variant:
- Iteration 11 likely regressed due to heavy class imbalance + unnormalized score
  scales across retrievers and queries.
- This version trains a simpler model (logistic regression) on a balanced set:
  all positives + sampled negatives *per query*, and uses per-query min-max
  normalized scores as features.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

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

ITERATION_NAME = "iteration_11b"
TOP_K = DEFAULT_TOP_K
POOL_K = 600
SPECTER2_MODEL = "allenai/specter2_base"
SPECTER_BATCH = 32
NEG_PER_POS = 6
SEED = 42


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


def minmax_1d(x: np.ndarray) -> np.ndarray:
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def build_pos_map(rank_idx: np.ndarray, n_docs: int) -> np.ndarray:
    n_q, depth = rank_idx.shape
    out = np.full((n_q, n_docs), fill_value=depth + 1, dtype=np.int16)
    for i in range(n_q):
        out[i, rank_idx[i]] = np.arange(1, depth + 1, dtype=np.int16)
    return out


def make_feature_row(
    doc_idx: int,
    b_score: float,
    m_score: float,
    s_score: float,
    b_pos: int,
    m_pos: int,
    s_pos: int,
) -> List[float]:
    rb = 1.0 / b_pos
    rm = 1.0 / m_pos
    rs = 1.0 / s_pos
    return [
        b_score,
        m_score,
        s_score,
        rb,
        rm,
        rs,
        rb + rm + rs,
        float(min(b_pos, m_pos, s_pos)),
    ]


def build_balanced_train(
    qids: List[str],
    corpus_ids: List[str],
    qrels: Dict[str, set[str]],
    bm25_r: np.ndarray,
    bm25_s: np.ndarray,
    minilm_r: np.ndarray,
    minilm_s: np.ndarray,
    specter_r: np.ndarray,
    specter_s: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    n_docs = len(corpus_ids)
    depth = bm25_r.shape[1]
    b_pos = build_pos_map(bm25_r, n_docs)
    m_pos = build_pos_map(minilm_r, n_docs)
    s_pos = build_pos_map(specter_r, n_docs)

    X_rows: List[List[float]] = []
    y: List[int] = []

    for i, qid in enumerate(qids):
        rel_ids = qrels.get(qid, set())
        if not rel_ids:
            continue

        # Candidate pool union
        cand = sorted(set(bm25_r[i].tolist()) | set(minilm_r[i].tolist()) | set(specter_r[i].tolist()))
        if not cand:
            continue

        # Build per-query score maps + normalized score maps (within top-k)
        b_map = {int(idx): float(score) for idx, score in zip(bm25_r[i], bm25_s[i])}
        m_map = {int(idx): float(score) for idx, score in zip(minilm_r[i], minilm_s[i])}
        s_map = {int(idx): float(score) for idx, score in zip(specter_r[i], specter_s[i])}

        b_norm = minmax_1d(bm25_s[i])
        m_norm = minmax_1d(minilm_s[i])
        s_norm = minmax_1d(specter_s[i])
        b_norm_map = {int(idx): float(v) for idx, v in zip(bm25_r[i], b_norm)}
        m_norm_map = {int(idx): float(v) for idx, v in zip(minilm_r[i], m_norm)}
        s_norm_map = {int(idx): float(v) for idx, v in zip(specter_r[i], s_norm)}

        is_rel = np.array([corpus_ids[j] in rel_ids for j in cand], dtype=bool)
        pos_idx = [cand[k] for k in np.where(is_rel)[0].tolist()]
        neg_idx = [cand[k] for k in np.where(~is_rel)[0].tolist()]
        if not pos_idx or not neg_idx:
            continue

        # sample negatives relative to number of positives
        n_neg = min(len(neg_idx), NEG_PER_POS * len(pos_idx))
        sampled_neg = rng.choice(neg_idx, size=n_neg, replace=False).tolist()

        for j in pos_idx:
            X_rows.append(
                make_feature_row(
                    j,
                    b_norm_map.get(j, 0.0),
                    m_norm_map.get(j, 0.0),
                    s_norm_map.get(j, 0.0),
                    int(b_pos[i, j]),
                    int(m_pos[i, j]),
                    int(s_pos[i, j]),
                )
            )
            y.append(1)
        for j in sampled_neg:
            X_rows.append(
                make_feature_row(
                    j,
                    b_norm_map.get(j, 0.0),
                    m_norm_map.get(j, 0.0),
                    s_norm_map.get(j, 0.0),
                    int(b_pos[i, j]),
                    int(m_pos[i, j]),
                    int(s_pos[i, j]),
                )
            )
            y.append(0)

    return np.asarray(X_rows, dtype=np.float32), np.asarray(y, dtype=np.int8)


def rank_query(
    clf: LogisticRegression,
    i: int,
    corpus_ids: List[str],
    bm25_r: np.ndarray,
    bm25_s: np.ndarray,
    minilm_r: np.ndarray,
    minilm_s: np.ndarray,
    specter_r: np.ndarray,
    specter_s: np.ndarray,
) -> np.ndarray:
    n_docs = len(corpus_ids)
    depth = bm25_r.shape[1]
    cand = np.array(sorted(set(bm25_r[i].tolist()) | set(minilm_r[i].tolist()) | set(specter_r[i].tolist())), dtype=np.int64)

    b_pos = np.full(n_docs, depth + 1, dtype=np.int16); b_pos[bm25_r[i]] = np.arange(1, depth + 1)
    m_pos = np.full(n_docs, depth + 1, dtype=np.int16); m_pos[minilm_r[i]] = np.arange(1, depth + 1)
    s_pos = np.full(n_docs, depth + 1, dtype=np.int16); s_pos[specter_r[i]] = np.arange(1, depth + 1)

    b_norm = minmax_1d(bm25_s[i]); b_norm_map = {int(idx): float(v) for idx, v in zip(bm25_r[i], b_norm)}
    m_norm = minmax_1d(minilm_s[i]); m_norm_map = {int(idx): float(v) for idx, v in zip(minilm_r[i], m_norm)}
    s_norm = minmax_1d(specter_s[i]); s_norm_map = {int(idx): float(v) for idx, v in zip(specter_r[i], s_norm)}

    X = np.asarray(
        [
            make_feature_row(
                int(j),
                b_norm_map.get(int(j), 0.0),
                m_norm_map.get(int(j), 0.0),
                s_norm_map.get(int(j), 0.0),
                int(b_pos[int(j)]),
                int(m_pos[int(j)]),
                int(s_pos[int(j)]),
            )
            for j in cand
        ],
        dtype=np.float32,
    )
    proba = clf.predict_proba(X)[:, 1].astype(np.float32)

    # tie-breaker: reciprocal-rank prior
    prior = (1.0 / b_pos[cand]) + (1.0 / m_pos[cand]) + (1.0 / s_pos[cand])
    final = proba + 0.01 * prior.astype(np.float32)
    order = np.argsort(-final, kind="stable")
    ranked = cand[order]
    ranked = ranked[:TOP_K]
    if len(ranked) < TOP_K:
        seen = set(ranked.tolist())
        filler = [j for j in bm25_r[i].tolist() if j not in seen]
        ranked = np.array((ranked.tolist() + filler)[:TOP_K], dtype=np.int64)
    return ranked


def main() -> None:
    data = data_paths()
    if not data["using_held_out_queries"]:
        raise RuntimeError(
            "Submission mode requires held-out queries. "
            "Add `held_out_queries.parquet` to `data/`, project root, or `starter_kit/`."
        )

    held_queries, corpus = load_queries_corpus(data["queries_path"], data["corpus_path"])
    corpus_ids = corpus["doc_id"].astype(str).tolist()

    train_queries, train_corpus, qrels = load_public_train_data()
    if train_corpus["doc_id"].astype(str).tolist() != corpus_ids:
        raise ValueError("Expected same corpus order for public and held-out pipelines.")

    corpus_rich = corpus.apply(lambda r: i4.format_rich_text(r, i4.MAX_FULLTEXT_CHARS), axis=1).tolist()
    train_query_rich = train_queries.apply(lambda r: i4.format_rich_text(r, i4.MAX_FULLTEXT_CHARS), axis=1).tolist()
    held_query_rich = held_queries.apply(lambda r: i4.format_rich_text(r, i4.MAX_FULLTEXT_CHARS), axis=1).tolist()

    print("BM25 retrieval for train/held …")
    train_b_r, train_b_s = bm25_rank_and_score(train_query_rich, corpus_rich, top_k=POOL_K)
    held_b_r, held_b_s = bm25_rank_and_score(held_query_rich, corpus_rich, top_k=POOL_K)

    print(f"Loading {i4.MINILM_MODEL} …")
    minilm = SentenceTransformer(i4.MINILM_MODEL)
    print("Encoding corpus with MiniLM …")
    corpus_emb_m = i4.encode_minilm(minilm, corpus_rich)
    print("Encoding train queries with MiniLM …")
    train_emb_m = i4.encode_minilm(minilm, train_query_rich)
    print("Encoding held-out queries with MiniLM …")
    held_emb_m = i4.encode_minilm(minilm, held_query_rich)
    train_m_r, train_m_s = dense_rank_and_score(train_emb_m, corpus_emb_m, top_k=POOL_K)
    held_m_r, held_m_s = dense_rank_and_score(held_emb_m, corpus_emb_m, top_k=POOL_K)

    print(f"Loading {SPECTER2_MODEL} …")
    specter = SentenceTransformer(SPECTER2_MODEL)
    sep = specter.tokenizer.sep_token or "[SEP]"
    corpus_specter = corpus.apply(lambda r: format_specter(r, sep), axis=1).tolist()
    train_q_s = train_queries.apply(lambda r: format_specter(r, sep), axis=1).tolist()
    held_q_s = held_queries.apply(lambda r: format_specter(r, sep), axis=1).tolist()
    corpus_emb_s = load_or_cache_specter_corpus(specter, corpus_specter, corpus_ids)
    print("Encoding train queries with SPECTER2 …")
    train_emb_s = encode_specter(specter, train_q_s)
    print("Encoding held-out queries with SPECTER2 …")
    held_emb_s = encode_specter(specter, held_q_s)
    train_s_r, train_s_s = dense_rank_and_score(train_emb_s, corpus_emb_s, top_k=POOL_K)
    held_s_r, held_s_s = dense_rank_and_score(held_emb_s, corpus_emb_s, top_k=POOL_K)

    print("Building balanced training set …")
    train_qids = train_queries["doc_id"].astype(str).tolist()
    X, y = build_balanced_train(
        qids=train_qids,
        corpus_ids=corpus_ids,
        qrels=qrels,
        bm25_r=train_b_r,
        bm25_s=train_b_s,
        minilm_r=train_m_r,
        minilm_s=train_m_s,
        specter_r=train_s_r,
        specter_s=train_s_s,
    )
    print(f"Train rows: {len(X)} | positives: {int(y.sum())} | neg/pos≈{(len(y)-int(y.sum()))/max(1,int(y.sum())):.1f}")
    if len(X) == 0 or y.sum() == 0:
        raise RuntimeError("Training set is empty or has no positives.")

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    )
    clf.fit(X, y)

    held_qids = held_queries["doc_id"].astype(str).tolist()
    submission: Dict[str, List[str]] = {}
    for i, qid in enumerate(held_qids):
        ranked = rank_query(
            clf=clf,
            i=i,
            corpus_ids=corpus_ids,
            bm25_r=held_b_r,
            bm25_s=held_b_s,
            minilm_r=held_m_r,
            minilm_s=held_m_s,
            specter_r=held_s_r,
            specter_s=held_s_s,
        )
        submission[qid] = [corpus_ids[int(j)] for j in ranked]

    paths = {**data, **iteration_submission_paths(ITERATION_NAME)}
    validate_submission(submission=submission, expected_query_ids=held_qids, top_k=TOP_K)
    validate_doc_ids_in_corpus(submission=submission, corpus_doc_ids=corpus_ids)
    save_submission(submission=submission, output_file=paths["output_file"])
    create_submission_zip(submission_file=paths["output_file"], zip_file=paths["zip_file"])

    print("Iteration 11b submission generated.")
    print("Method: logistic regression learned fusion over normalized BM25 + MiniLM + SPECTER2 features.")
    print(f"Saved to: {paths['zip_file']}")


if __name__ == "__main__":
    main()

