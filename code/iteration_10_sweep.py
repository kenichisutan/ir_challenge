from __future__ import annotations

"""
Iteration 10 weight sweep.

Generates multiple weighted-RRF submissions in one run using shared retrieval
artifacts (TF-IDF rank list, MiniLM rank list, SPECTER2 rank list).
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

TOP_K = DEFAULT_TOP_K
FUSION_DEPTH = 300
RRF_K = 40.0
SPECTER2_MODEL = "allenai/specter2_base"
SPECTER_ENCODE_BATCH = 32

WEIGHT_CONFIGS: Sequence[Tuple[str, float, float, float]] = [
    ("iteration_10a", 0.50, 0.40, 0.10),
    ("iteration_10b", 0.48, 0.42, 0.10),
    ("iteration_10c", 0.52, 0.38, 0.10),
    ("iteration_10d", 0.46, 0.44, 0.10),
]


def specter_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "embeddings" / "allenai_specter2_base"


def format_specter(row: pd.Series, sep: str) -> str:
    title = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    if title and abstract:
        return f"{title} {sep} {abstract}".strip()
    return title or abstract


def encode_specter(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=SPECTER_ENCODE_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return emb.astype(np.float32)


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
    emb = encode_specter(model, texts)
    np.save(npy, emb)
    with idf.open("w", encoding="utf-8") as f:
        json.dump(ids, f)
    return emb


def weighted_rrf(rankings: List[np.ndarray], weights: List[float], k: float, top_n: int) -> List[np.ndarray]:
    nq = rankings[0].shape[0]
    out: List[np.ndarray] = []
    for i in range(nq):
        scores: Dict[int, float] = {}
        for w, ranks in zip(weights, rankings):
            for rank, doc_idx in enumerate(ranks[i], start=1):
                j = int(doc_idx)
                scores[j] = scores.get(j, 0.0) + w / (k + rank)
        ordered = sorted(scores.keys(), key=lambda d: (-scores[d], d))
        out.append(np.array(ordered[:top_n], dtype=np.int64))
    return out


def main() -> None:
    data = data_paths()
    if not data["using_held_out_queries"]:
        raise RuntimeError(
            "Submission mode requires held-out queries. "
            "Add `held_out_queries.parquet` to `data/`, project root, or `starter_kit/`."
        )

    queries, corpus = load_queries_corpus(data["queries_path"], data["corpus_path"])
    corpus_ids = corpus["doc_id"].astype(str).tolist()
    query_ids = queries["doc_id"].astype(str).tolist()

    query_rich = queries.apply(lambda r: i4.format_rich_text(r, i4.MAX_FULLTEXT_CHARS), axis=1).tolist()
    corpus_rich = corpus.apply(lambda r: i4.format_rich_text(r, i4.MAX_FULLTEXT_CHARS), axis=1).tolist()

    print("Computing TF-IDF ranks once …")
    tfidf_ranks = i4.rank_with_tfidf_topk(query_rich, corpus_rich, top_k=FUSION_DEPTH)

    print(f"Loading {i4.MINILM_MODEL} …")
    minilm = SentenceTransformer(i4.MINILM_MODEL)
    print("Encoding corpus with MiniLM (rich text) …")
    corpus_emb_minilm = i4.encode_minilm(minilm, corpus_rich)
    print("Encoding queries with MiniLM (rich text) …")
    query_emb_minilm = i4.encode_minilm(minilm, query_rich)
    minilm_ranks = i4.rank_dense_topk(query_emb_minilm, corpus_emb_minilm, top_k=FUSION_DEPTH)

    print(f"Loading {SPECTER2_MODEL} …")
    specter = SentenceTransformer(SPECTER2_MODEL)
    sep = specter.tokenizer.sep_token or "[SEP]"
    query_specter = queries.apply(lambda r: format_specter(r, sep), axis=1).tolist()
    corpus_specter = corpus.apply(lambda r: format_specter(r, sep), axis=1).tolist()
    corpus_emb_s = load_or_cache_specter_corpus(specter, corpus_specter, corpus_ids)
    print("Encoding queries with SPECTER2 …")
    query_emb_s = encode_specter(specter, query_specter)
    specter_ranks = i4.rank_dense_topk(query_emb_s, corpus_emb_s, top_k=FUSION_DEPTH)

    for run_name, w_tfidf, w_minilm, w_specter in WEIGHT_CONFIGS:
        print(f"Building {run_name} with weights ({w_tfidf}, {w_minilm}, {w_specter}) …")
        fused = weighted_rrf(
            [tfidf_ranks, minilm_ranks, specter_ranks],
            [w_tfidf, w_minilm, w_specter],
            k=RRF_K,
            top_n=TOP_K,
        )

        submission: Dict[str, List[str]] = {}
        for i, qid in enumerate(query_ids):
            submission[qid] = [corpus_ids[int(j)] for j in fused[i]]

        validate_submission(submission=submission, expected_query_ids=query_ids, top_k=TOP_K)
        validate_doc_ids_in_corpus(submission=submission, corpus_doc_ids=corpus_ids)

        paths = {**data, **iteration_submission_paths(run_name)}
        save_submission(submission=submission, output_file=paths["output_file"])
        create_submission_zip(submission_file=paths["output_file"], zip_file=paths["zip_file"])
        print(f"Saved: {paths['zip_file']}")

    print("Sweep completed.")


if __name__ == "__main__":
    main()
