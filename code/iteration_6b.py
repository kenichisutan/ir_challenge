from __future__ import annotations

"""
Iteration 6b — Iteration 4 sparse + MiniLM (rich) + SPECTER2 dense, triple RRF.

- TF-IDF and MiniLM use the same **rich** strings as Iteration 4 (`format_rich_text`, 2000-char full_text cap).
- SPECTER2 (`allenai/specter2_base`) uses **title + tokenizer.sep_token + abstract** (HF / Allen AI convention for citation embeddings).
- Corpus SPECTER2 vectors are cached under `data/embeddings/allenai_specter2_base/` after the first run.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

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

ITERATION_NAME = "iteration_6b"
TOP_K = DEFAULT_TOP_K
# Match Iteration 4 fusion hyperparameters (single new leg = SPECTER2).
FUSION_DEPTH = i4.FUSION_DEPTH
RRF_K = i4.RRF_K
SPECTER2_MODEL = "allenai/specter2_base"
SPECTER_ENCODE_BATCH = 32


def challenge_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def specter2_emb_dir() -> Path:
    return challenge_dir() / "data" / "embeddings" / "allenai_specter2_base"


def format_specter2_title_sep_abstract(row: pd.Series, sep: str) -> str:
    title = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    if title and abstract:
        return f"{title} {sep} {abstract}".strip()
    return title or abstract


def rrf_fuse_many(rankings: List[np.ndarray], k: float, top_n: int) -> List[np.ndarray]:
    nq = rankings[0].shape[0]
    out: List[np.ndarray] = []
    for i in range(nq):
        scores: Dict[int, float] = {}
        for ranks in rankings:
            for rank, doc_idx in enumerate(ranks[i], start=1):
                idx = int(doc_idx)
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
        ordered = sorted(scores.keys(), key=lambda d: (-scores[d], d))
        if len(ordered) < top_n:
            raise ValueError(f"RRF produced only {len(ordered)} docs; need {top_n}")
        out.append(np.array(ordered[:top_n], dtype=np.int64))
    return out


def encode_specter2(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=SPECTER_ENCODE_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return emb.astype(np.float32)


def load_or_build_specter2_corpus(
    model: SentenceTransformer,
    corpus_texts_specter: List[str],
    corpus_ids: List[str],
) -> np.ndarray:
    emb_dir = specter2_emb_dir()
    emb_dir.mkdir(parents=True, exist_ok=True)
    npy_path = emb_dir / "corpus_embeddings.npy"
    ids_path = emb_dir / "corpus_ids.json"

    if npy_path.exists() and ids_path.exists():
        with ids_path.open("r", encoding="utf-8") as f:
            saved = [str(x) for x in json.load(f)]
        if saved == corpus_ids:
            print(f"Loading cached SPECTER2 corpus embeddings from {npy_path} …")
            return np.load(npy_path)

    print("Encoding corpus with SPECTER2 (title [SEP] abstract) — caching …")
    emb = encode_specter2(model, corpus_texts_specter)
    np.save(npy_path, emb)
    with ids_path.open("w", encoding="utf-8") as f:
        json.dump(corpus_ids, f)
    return emb


def build_submission_arrays(
    queries: pd.DataFrame,
    corpus: pd.DataFrame,
    minilm: SentenceTransformer,
    specter2: SentenceTransformer,
    sep: str,
    top_k: int,
    fusion_depth: int,
) -> List[np.ndarray]:
    query_rich = queries.apply(lambda r: i4.format_rich_text(r, i4.MAX_FULLTEXT_CHARS), axis=1).tolist()
    corpus_rich = corpus.apply(lambda r: i4.format_rich_text(r, i4.MAX_FULLTEXT_CHARS), axis=1).tolist()

    tfidf_ranks = i4.rank_with_tfidf_topk(query_rich, corpus_rich, top_k=fusion_depth)

    print("Encoding corpus with MiniLM (rich text) …")
    corpus_emb_minilm = i4.encode_minilm(minilm, corpus_rich)
    print("Encoding queries with MiniLM (rich text) …")
    query_emb_minilm = i4.encode_minilm(minilm, query_rich)
    dense_minilm_ranks = i4.rank_dense_topk(query_emb_minilm, corpus_emb_minilm, top_k=fusion_depth)

    corpus_ids = corpus["doc_id"].astype(str).tolist()
    query_specter = queries.apply(lambda r: format_specter2_title_sep_abstract(r, sep), axis=1).tolist()
    corpus_specter = corpus.apply(lambda r: format_specter2_title_sep_abstract(r, sep), axis=1).tolist()

    corpus_emb_specter = load_or_build_specter2_corpus(specter2, corpus_specter, corpus_ids)
    print("Encoding queries with SPECTER2 …")
    query_emb_specter = encode_specter2(specter2, query_specter)
    specter_ranks = i4.rank_dense_topk(query_emb_specter, corpus_emb_specter, top_k=fusion_depth)

    return rrf_fuse_many(
        [tfidf_ranks, dense_minilm_ranks, specter_ranks],
        k=RRF_K,
        top_n=top_k,
    )


def main() -> None:
    data = data_paths()
    if not data["using_held_out_queries"]:
        raise RuntimeError(
            "Submission mode requires held-out queries. "
            "Add `held_out_queries.parquet` to `data/`, project root, or `starter_kit/`."
        )

    paths = {**data, **iteration_submission_paths(ITERATION_NAME)}
    queries, corpus = load_queries_corpus(paths["queries_path"], paths["corpus_path"])
    parquet_ids = corpus["doc_id"].astype(str).tolist()

    print(f"Loading {i4.MINILM_MODEL} …")
    minilm = SentenceTransformer(i4.MINILM_MODEL)

    print(f"Loading {SPECTER2_MODEL} …")
    specter2 = SentenceTransformer(SPECTER2_MODEL)
    sep = specter2.tokenizer.sep_token
    if not sep:
        sep = " [SEP] "
    print(f"SPECTER2 query format: title + '{sep}' + abstract")

    fused = build_submission_arrays(
        queries, corpus, minilm, specter2, sep, TOP_K, FUSION_DEPTH
    )

    qids = queries["doc_id"].astype(str).tolist()
    submission: Dict[str, List[str]] = {}
    for i, qid in enumerate(qids):
        submission[qid] = [parquet_ids[int(j)] for j in fused[i]]

    validate_submission(submission=submission, expected_query_ids=qids, top_k=TOP_K)
    validate_doc_ids_in_corpus(submission=submission, corpus_doc_ids=parquet_ids)
    save_submission(submission=submission, output_file=paths["output_file"])
    create_submission_zip(submission_file=paths["output_file"], zip_file=paths["zip_file"])

    print("Iteration 6b submission generated.")
    print(f"Queries: {len(queries)} | Corpus: {len(corpus)} | Top-K: {TOP_K}")
    print(
        f"Method: rich TF-IDF + rich MiniLM + SPECTER2 (TA) | "
        f"RRF k={RRF_K}, depth={FUSION_DEPTH}"
    )
    print(f"Query source: {paths['queries_source']}")
    print(f"Saved to: {paths['output_file']}")
    print(f"Zipped to: {paths['zip_file']}")


if __name__ == "__main__":
    main()
