from __future__ import annotations

"""
Iteration 4.1 — BM25 chunked sparse + SPECTER2 dense + hybrid fusion + CE rerank.

Implements selected team feedback:
1) Sparse: BM25 over body chunks (full text) rather than plain TF-IDF.
2) Dense: domain encoder `allenai/specter2_base`.
3) Hybrid: score interpolation + weighted RRF.
4) Refinement: cross-encoder reranking with paragraph-level max scoring.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

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

ITERATION_NAME = "iteration_4_1"
TOP_K = DEFAULT_TOP_K
FUSION_DEPTH = 300
RRF_K = 40.0

SPECTER2_MODEL = "allenai/specter2_base"
SPECTER_BATCH = 32
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_BATCH = 32
RERANK_M = 30

INTERP_ALPHA = 0.60  # sparse-heavy interpolation on normalized raw scores.
RRF_WEIGHT = 0.35    # mix in rank-only evidence.
CE_LAMBDA = 0.45     # blend CE paragraph-level score with hybrid prior for top-M.

CHUNK_WORDS = 220
CHUNK_STRIDE = 180
MAX_CHUNKS_PER_DOC = 6
MAX_PARAS_FOR_CE = 4
MAX_PARA_CHARS = 1200


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def safe_text(x: object) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).replace("\n", " ").strip()


def rich_query_text(row: pd.Series) -> str:
    return f"{safe_text(row.get('title'))} {safe_text(row.get('abstract'))}".strip()


def specter_ta_text(row: pd.Series, sep: str) -> str:
    t = safe_text(row.get("title"))
    a = safe_text(row.get("abstract"))
    if t and a:
        return f"{t} {sep} {a}"
    return t or a


def split_body_chunks(full_text: str, chunk_words: int, stride: int, max_chunks: int) -> List[str]:
    words = tokenize(full_text)
    if not words:
        return []
    out: List[str] = []
    i = 0
    while i < len(words) and len(out) < max_chunks:
        chunk = words[i : i + chunk_words]
        if not chunk:
            break
        out.append(" ".join(chunk))
        i += stride
    return out


def minmax(v: np.ndarray) -> np.ndarray:
    lo = float(v.min())
    hi = float(v.max())
    if hi - lo < 1e-12:
        return np.zeros_like(v, dtype=np.float32)
    return ((v - lo) / (hi - lo)).astype(np.float32)


def rrf_scores(rank_lists: List[np.ndarray], n_docs: int, k: float) -> np.ndarray:
    scores = np.zeros(n_docs, dtype=np.float32)
    for ranks in rank_lists:
        for r, idx in enumerate(ranks, start=1):
            scores[int(idx)] += 1.0 / (k + r)
    return scores


def build_chunk_bm25_index(corpus: pd.DataFrame) -> Tuple[BM25Okapi, np.ndarray]:
    chunk_texts: List[List[str]] = []
    chunk_to_doc: List[int] = []
    for i, row in enumerate(corpus.itertuples(index=False)):
        title = safe_text(getattr(row, "title", ""))
        abstract = safe_text(getattr(row, "abstract", ""))
        full = safe_text(getattr(row, "full_text", ""))
        head = f"{title} {abstract}".strip()
        chunks = split_body_chunks(full, CHUNK_WORDS, CHUNK_STRIDE, MAX_CHUNKS_PER_DOC)
        if not chunks:
            chunks = [head]
        for c in chunks:
            chunk_to_doc.append(i)
            chunk_texts.append(tokenize(f"{head} {c}".strip()))
    return BM25Okapi(chunk_texts, k1=1.2, b=0.75), np.asarray(chunk_to_doc, dtype=np.int64)


def bm25_doc_scores(bm25: BM25Okapi, chunk_to_doc: np.ndarray, query: str, n_docs: int) -> np.ndarray:
    cs = bm25.get_scores(tokenize(query))
    ds = np.full(n_docs, -1e9, dtype=np.float32)
    np.maximum.at(ds, chunk_to_doc, cs.astype(np.float32))
    return ds


def specter_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "embeddings" / "allenai_specter2_base"


def load_or_cache_specter_corpus(model: SentenceTransformer, texts: List[str], ids: List[str]) -> np.ndarray:
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
    emb = model.encode(
        texts,
        batch_size=SPECTER_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    np.save(npy, emb)
    with idf.open("w", encoding="utf-8") as f:
        json.dump(ids, f)
    return emb


def ce_passages_for_doc(row: pd.Series) -> List[str]:
    title = safe_text(row.get("title"))
    abstract = safe_text(row.get("abstract"))
    full = safe_text(row.get("full_text"))
    base = f"{title}. {abstract}".strip()
    paras = [p.strip() for p in re.split(r"\n\s*\n+", full) if p.strip()]
    if not paras:
        paras = split_body_chunks(full, chunk_words=180, stride=160, max_chunks=MAX_PARAS_FOR_CE)
    sel = paras[:MAX_PARAS_FOR_CE]
    out = [base] if base else []
    out.extend([(f"{title}. {p}" if title else p)[:MAX_PARA_CHARS] for p in sel])
    return out[: MAX_PARAS_FOR_CE + 1]


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
    n_docs = len(corpus)

    print("Building BM25 chunk index …")
    bm25, chunk_to_doc = build_chunk_bm25_index(corpus)

    print(f"Loading {SPECTER2_MODEL} …")
    specter = SentenceTransformer(SPECTER2_MODEL)
    sep = specter.tokenizer.sep_token or "[SEP]"
    corpus_specter = corpus.apply(lambda r: specter_ta_text(r, sep), axis=1).tolist()
    query_specter = queries.apply(lambda r: specter_ta_text(r, sep), axis=1).tolist()
    corpus_emb = load_or_cache_specter_corpus(specter, corpus_specter, corpus_ids)
    query_emb = specter.encode(
        query_specter,
        batch_size=SPECTER_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    print("Loading cross-encoder reranker …")
    ce = CrossEncoder(CE_MODEL)

    submission: Dict[str, List[str]] = {}
    for qi, qid in enumerate(query_ids):
        q_text = rich_query_text(queries.iloc[qi])

        sparse_scores = bm25_doc_scores(bm25, chunk_to_doc, q_text, n_docs)
        dense_scores = (query_emb[qi] @ corpus_emb.T).astype(np.float32)

        sparse_n = minmax(sparse_scores)
        dense_n = minmax(dense_scores)
        interp = INTERP_ALPHA * sparse_n + (1.0 - INTERP_ALPHA) * dense_n

        sparse_rank = np.argsort(-sparse_scores, kind="stable")[:FUSION_DEPTH]
        dense_rank = np.argsort(-dense_scores, kind="stable")[:FUSION_DEPTH]
        rrf = rrf_scores([sparse_rank, dense_rank], n_docs=n_docs, k=RRF_K)
        rrf_n = minmax(rrf)

        hybrid = (1.0 - RRF_WEIGHT) * interp + RRF_WEIGHT * rrf_n
        base_rank = np.argsort(-hybrid, kind="stable")[:TOP_K]

        # Cross-encoder rerank top-M docs using max paragraph score.
        m = min(RERANK_M, len(base_rank))
        ce_doc_scores = np.full(m, -1e6, dtype=np.float32)
        pair_doc_idx: List[int] = []
        pairs: List[List[str]] = []
        for j in range(m):
            doc_idx = int(base_rank[j])
            doc_row = corpus.iloc[doc_idx]
            passages = ce_passages_for_doc(doc_row)
            for p in passages:
                pairs.append([q_text, p])
                pair_doc_idx.append(j)
        if pairs:
            p_scores = ce.predict(pairs, batch_size=CE_BATCH, show_progress_bar=False)
            for s, j in zip(np.asarray(p_scores, dtype=np.float32), pair_doc_idx):
                if s > ce_doc_scores[j]:
                    ce_doc_scores[j] = float(s)

        ce_n = minmax(ce_doc_scores)
        prior_n = minmax(hybrid[base_rank[:m]])
        blended_top = (1.0 - CE_LAMBDA) * prior_n + CE_LAMBDA * ce_n
        top_order = np.argsort(-blended_top, kind="stable")
        reranked_top = base_rank[:m][top_order]
        final_idx = np.concatenate([reranked_top, base_rank[m:TOP_K]])

        submission[qid] = [corpus_ids[int(k)] for k in final_idx]
        if (qi + 1) % 10 == 0:
            print(f"Processed {qi + 1}/{len(query_ids)} queries …")

    validate_submission(submission=submission, expected_query_ids=query_ids, top_k=TOP_K)
    validate_doc_ids_in_corpus(submission=submission, corpus_doc_ids=corpus_ids)
    save_submission(submission=submission, output_file=paths["output_file"])
    create_submission_zip(submission_file=paths["output_file"], zip_file=paths["zip_file"])

    print("Iteration 4.1 submission generated.")
    print(
        "Method: BM25 chunk sparse + SPECTER2 dense + interpolation+RRF + CE paragraph rerank "
        f"(top-{RERANK_M})"
    )
    print(f"Saved to: {paths['zip_file']}")


if __name__ == "__main__":
    main()
