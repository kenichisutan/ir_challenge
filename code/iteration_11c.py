from __future__ import annotations

"""
Iteration 11c — pairwise learned fusion (query-aware ranking).

Train a linear ranker on pairwise preferences per query:
  (q, pos) should score > (q, neg)
by fitting a classifier on feature differences: phi(q,pos) - phi(q,neg).

At inference, score candidates with w·phi and sort.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import SGDClassifier

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

ITERATION_NAME = "iteration_11c"
TOP_K = DEFAULT_TOP_K
POOL_K = 700
SPECTER2_MODEL = "allenai/specter2_base"
SPECTER_BATCH = 32
NEG_PER_POS = 8
PAIRS_PER_QUERY_CAP = 300
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


def build_pos_vec(rank_idx: np.ndarray, n_docs: int) -> np.ndarray:
    depth = rank_idx.shape[0]
    pos = np.full(n_docs, depth + 1, dtype=np.int16)
    pos[rank_idx] = np.arange(1, depth + 1, dtype=np.int16)
    return pos


def feat_matrix_for_query(
    i: int,
    cand: np.ndarray,
    n_docs: int,
    bm25_r: np.ndarray,
    bm25_s: np.ndarray,
    minilm_r: np.ndarray,
    minilm_s: np.ndarray,
    specter_r: np.ndarray,
    specter_s: np.ndarray,
) -> np.ndarray:
    depth = bm25_r.shape[1]
    pos_b = build_pos_vec(bm25_r[i], n_docs)
    pos_m = build_pos_vec(minilm_r[i], n_docs)
    pos_s = build_pos_vec(specter_r[i], n_docs)

    b_norm = minmax_1d(bm25_s[i]); b_norm_map = {int(idx): float(v) for idx, v in zip(bm25_r[i], b_norm)}
    m_norm = minmax_1d(minilm_s[i]); m_norm_map = {int(idx): float(v) for idx, v in zip(minilm_r[i], m_norm)}
    s_norm = minmax_1d(specter_s[i]); s_norm_map = {int(idx): float(v) for idx, v in zip(specter_r[i], s_norm)}

    X = np.zeros((len(cand), 8), dtype=np.float32)
    for k, j in enumerate(cand):
        jb = int(j)
        pb = int(pos_b[jb]); pm = int(pos_m[jb]); ps = int(pos_s[jb])
        rb = 1.0 / pb; rm = 1.0 / pm; rs = 1.0 / ps
        X[k] = np.array(
            [
                b_norm_map.get(jb, 0.0),
                m_norm_map.get(jb, 0.0),
                s_norm_map.get(jb, 0.0),
                rb,
                rm,
                rs,
                rb + rm + rs,
                float(min(pb, pm, ps)),
            ],
            dtype=np.float32,
        )
    return X


def build_pairwise_train(
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
    X_diffs: List[np.ndarray] = []
    y: List[int] = []

    for i, qid in enumerate(qids):
        rel_ids = qrels.get(qid, set())
        if not rel_ids:
            continue
        cand = np.array(sorted(set(bm25_r[i].tolist()) | set(minilm_r[i].tolist()) | set(specter_r[i].tolist())), dtype=np.int64)
        if len(cand) == 0:
            continue
        rel_mask = np.array([corpus_ids[int(j)] in rel_ids for j in cand], dtype=bool)
        pos = cand[rel_mask]
        neg = cand[~rel_mask]
        if len(pos) == 0 or len(neg) == 0:
            continue

        Xq = feat_matrix_for_query(i, cand, n_docs, bm25_r, bm25_s, minilm_r, minilm_s, specter_r, specter_s)
        # index maps from doc idx to row in Xq
        row_by_doc = {int(doc): r for r, doc in enumerate(cand.tolist())}

        pairs_added = 0
        for p in pos:
            n_neg = min(len(neg), NEG_PER_POS)
            sampled = rng.choice(neg, size=n_neg, replace=False)
            for n in sampled:
                dp = Xq[row_by_doc[int(p)]]
                dn = Xq[row_by_doc[int(n)]]
                X_diffs.append(dp - dn)
                y.append(1)
                # add reverse pair for balance
                X_diffs.append(dn - dp)
                y.append(0)
                pairs_added += 2
                if pairs_added >= PAIRS_PER_QUERY_CAP:
                    break
            if pairs_added >= PAIRS_PER_QUERY_CAP:
                break

    X = np.asarray(X_diffs, dtype=np.float32)
    yv = np.asarray(y, dtype=np.int8)
    return X, yv


def rank_query(
    clf: SGDClassifier,
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
    cand = np.array(sorted(set(bm25_r[i].tolist()) | set(minilm_r[i].tolist()) | set(specter_r[i].tolist())), dtype=np.int64)
    Xq = feat_matrix_for_query(i, cand, n_docs, bm25_r, bm25_s, minilm_r, minilm_s, specter_r, specter_s)
    scores = clf.decision_function(Xq).astype(np.float32)
    order = np.argsort(-scores, kind="stable")
    ranked = cand[order][:TOP_K]
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

    print("Building pairwise training set …")
    train_qids = train_queries["doc_id"].astype(str).tolist()
    X, y = build_pairwise_train(
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
    print(f"Pairwise rows: {len(X)} | y_mean={float(y.mean()):.3f}")
    if len(X) == 0:
        raise RuntimeError("Pairwise training set is empty.")

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=30,
        tol=1e-4,
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

    print("Iteration 11c submission generated.")
    print("Method: pairwise SGD ranker over normalized BM25 + MiniLM + SPECTER2 features.")
    print(f"Saved to: {paths['zip_file']}")


if __name__ == "__main__":
    main()

