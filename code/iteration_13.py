from __future__ import annotations

"""
Iteration 13 — chunk-level dense retrieval + doc aggregation fused into Iteration 12.

New capability: retrieve over chunks of `full_text` with MiniLM, aggregate chunk
scores to document scores (max), and fuse with doc-level hybrid.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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

ITERATION_NAME = "iteration_13"
TOP_K = DEFAULT_TOP_K
DEPTH = 400
RRF_K = 40.0

# chunk retrieval parameters
CHUNK_WORDS = 220
CHUNK_STRIDE = 180
MAX_CHUNKS_PER_DOC = 8
CHUNK_TOPK = 2500  # retrieve over chunks, then aggregate to docs

# interpolation weights (sum=1)
W_TFIDF = 0.44
W_MINILM_DOC = 0.36
W_SPECTER2 = 0.10
W_MINILM_CHUNK = 0.10

GAMMA_RRF = 0.35

SPECTER2_MODEL = "allenai/specter2_base"
SPECTER_BATCH = 32


def chunk_emb_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "embeddings" / "chunk_minilm_rich_v1"


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
    s = np.zeros(n_docs, dtype=np.float32)
    for r, j in enumerate(rank_idx, start=1):
        s[int(j)] += 1.0 / (k + r)
    return s


def split_words(text: str) -> List[str]:
    return str(text).replace("\n", " ").lower().split()


def build_chunk_texts(corpus: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    chunk_texts: List[str] = []
    chunk_to_doc: List[int] = []
    for di, row in enumerate(corpus.itertuples(index=False)):
        title = getattr(row, "title", "") or ""
        abstract = getattr(row, "abstract", "") or ""
        full = getattr(row, "full_text", "") or ""
        head = f"{title} {abstract}".strip()
        words = split_words(full)
        if not words:
            chunk_texts.append(head)
            chunk_to_doc.append(di)
            continue

        made = 0
        start = 0
        while start < len(words) and made < MAX_CHUNKS_PER_DOC:
            chunk = " ".join(words[start : start + CHUNK_WORDS])
            chunk_texts.append(f"{head} {chunk}".strip() if head else chunk)
            chunk_to_doc.append(di)
            start += CHUNK_STRIDE
            made += 1
    return chunk_texts, np.asarray(chunk_to_doc, dtype=np.int32)


def load_or_build_chunk_embeddings(
    model: SentenceTransformer, corpus: pd.DataFrame, corpus_ids: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    d = chunk_emb_dir()
    d.mkdir(parents=True, exist_ok=True)
    emb_path = d / "chunk_embeddings.npy"
    map_path = d / "chunk_to_doc.npy"
    ids_path = d / "corpus_ids.json"

    if emb_path.exists() and map_path.exists() and ids_path.exists():
        with ids_path.open("r", encoding="utf-8") as f:
            saved = [str(x) for x in json.load(f)]
        if saved == corpus_ids:
            print(f"Loading cached chunk embeddings from {emb_path} …")
            return np.load(emb_path), np.load(map_path)

    print("Building chunk texts …")
    chunk_texts, chunk_to_doc = build_chunk_texts(corpus)
    print(f"Encoding {len(chunk_texts)} chunks with MiniLM …")
    chunk_emb = model.encode(
        chunk_texts,
        batch_size=96,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    np.save(emb_path, chunk_emb)
    np.save(map_path, chunk_to_doc)
    with ids_path.open("w", encoding="utf-8") as f:
        json.dump(corpus_ids, f)
    return chunk_emb, chunk_to_doc


def aggregate_chunk_scores_to_docs(chunk_scores: np.ndarray, chunk_to_doc: np.ndarray, n_docs: int) -> np.ndarray:
    doc_scores = np.full(n_docs, -1e9, dtype=np.float32)
    np.maximum.at(doc_scores, chunk_to_doc, chunk_scores)
    return doc_scores


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

    tfidf_ranks = i4.rank_with_tfidf_topk(query_rich, corpus_rich, top_k=DEPTH)

    print(f"Loading {i4.MINILM_MODEL} …")
    minilm = SentenceTransformer(i4.MINILM_MODEL)

    # doc-level MiniLM
    print("Encoding corpus with MiniLM (doc-level rich) …")
    corpus_emb_m = i4.encode_minilm(minilm, corpus_rich)
    print("Encoding queries with MiniLM (doc-level rich) …")
    query_emb_m = i4.encode_minilm(minilm, query_rich)
    minilm_sim = (query_emb_m @ corpus_emb_m.T).astype(np.float32)
    minilm_doc_ranks = np.argsort(-minilm_sim, axis=1, kind="stable")[:, :DEPTH]

    # chunk-level MiniLM
    chunk_emb, chunk_to_doc = load_or_build_chunk_embeddings(minilm, corpus, corpus_ids)

    # SPECTER2 doc-level
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

    submission: Dict[str, List[str]] = {}
    for i, qid in enumerate(query_ids):
        q = query_emb_m[i]
        chunk_scores_all = (chunk_emb @ q).astype(np.float32)
        top_chunk_idx = np.argsort(-chunk_scores_all, kind="stable")[:CHUNK_TOPK]
        chunk_doc_scores = aggregate_chunk_scores_to_docs(
            chunk_scores_all[top_chunk_idx], chunk_to_doc[top_chunk_idx], n_docs
        )
        chunk_doc_ranks = np.argsort(-chunk_doc_scores, kind="stable")[:DEPTH]

        cand = sorted(
            set(tfidf_ranks[i].tolist())
            | set(minilm_doc_ranks[i].tolist())
            | set(specter_ranks[i].tolist())
            | set(chunk_doc_ranks.tolist())
        )
        cand = np.asarray(cand, dtype=np.int64)

        tf_pos = {int(d): r + 1 for r, d in enumerate(tfidf_ranks[i].tolist())}
        tf_score = np.asarray([1.0 / tf_pos.get(int(d), DEPTH + 1) for d in cand], dtype=np.float32)
        tf_score = minmax(tf_score)

        m_doc = minmax(minilm_sim[i, cand])
        s_doc = minmax(specter_sim[i, cand])
        m_chunk = minmax(chunk_doc_scores[cand])

        interp = (
            (W_TFIDF * tf_score)
            + (W_MINILM_DOC * m_doc)
            + (W_SPECTER2 * s_doc)
            + (W_MINILM_CHUNK * m_chunk)
        )

        rrf_full = (
            rrf_from_rank(tfidf_ranks[i], n_docs, RRF_K)
            + rrf_from_rank(minilm_doc_ranks[i], n_docs, RRF_K)
            + rrf_from_rank(specter_ranks[i], n_docs, RRF_K)
            + rrf_from_rank(chunk_doc_ranks, n_docs, RRF_K)
        )
        rrf_c = minmax(rrf_full[cand])

        final = (1.0 - GAMMA_RRF) * interp + GAMMA_RRF * rrf_c
        order = np.argsort(-final, kind="stable")
        ranked = cand[order][:TOP_K]

        if len(ranked) < TOP_K:
            seen = set(ranked.tolist())
            filler = [int(j) for j in tfidf_ranks[i].tolist() if int(j) not in seen]
            ranked = np.asarray((ranked.tolist() + filler)[:TOP_K], dtype=np.int64)

        submission[qid] = [corpus_ids[int(j)] for j in ranked]

    validate_submission(submission=submission, expected_query_ids=query_ids, top_k=TOP_K)
    validate_doc_ids_in_corpus(submission=submission, corpus_doc_ids=corpus_ids)
    save_submission(submission=submission, output_file=paths["output_file"])
    create_submission_zip(submission_file=paths["output_file"], zip_file=paths["zip_file"])

    print("Iteration 13 submission generated.")
    print(
        "Method: Iteration12 signals + MiniLM chunk leg | "
        f"interp_w=({W_TFIDF},{W_MINILM_DOC},{W_SPECTER2},{W_MINILM_CHUNK}), gamma_rrf={GAMMA_RRF}"
    )
    print(f"Saved to: {paths['zip_file']}")


if __name__ == "__main__":
    main()

