from __future__ import annotations

"""
Iteration 8 — dense-only SPECTER2 baseline.

Untried direction: remove sparse/hybrid fusion and test a pure citation-oriented
dense retriever.
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

ITERATION_NAME = "iteration_8"
TOP_K = DEFAULT_TOP_K
SPECTER2_MODEL = "allenai/specter2_base"
ENCODE_BATCH = 32


def emb_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "embeddings" / "allenai_specter2_base"


def format_ta_sep(row: pd.Series, sep: str) -> str:
    title = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    if title and abstract:
        return f"{title} {sep} {abstract}".strip()
    return title or abstract


def encode(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=ENCODE_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return emb.astype(np.float32)


def load_or_cache_corpus(model: SentenceTransformer, corpus_texts: List[str], corpus_ids: List[str]) -> np.ndarray:
    d = emb_dir()
    d.mkdir(parents=True, exist_ok=True)
    npy = d / "corpus_embeddings.npy"
    ids = d / "corpus_ids.json"

    if npy.exists() and ids.exists():
        with ids.open("r", encoding="utf-8") as f:
            saved = [str(x) for x in json.load(f)]
        if saved == corpus_ids:
            print(f"Loading cached SPECTER2 corpus embeddings from {npy} …")
            return np.load(npy)

    print("Encoding corpus with SPECTER2 and caching …")
    emb = encode(model, corpus_texts)
    np.save(npy, emb)
    with ids.open("w", encoding="utf-8") as f:
        json.dump(corpus_ids, f)
    return emb


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

    print(f"Loading {SPECTER2_MODEL} …")
    model = SentenceTransformer(SPECTER2_MODEL)
    sep = model.tokenizer.sep_token or "[SEP]"

    corpus_texts = corpus.apply(lambda r: format_ta_sep(r, sep), axis=1).tolist()
    query_texts = queries.apply(lambda r: format_ta_sep(r, sep), axis=1).tolist()

    corpus_emb = load_or_cache_corpus(model, corpus_texts, corpus_ids)
    print("Encoding queries with SPECTER2 …")
    query_emb = encode(model, query_texts)

    sim = query_emb @ corpus_emb.T
    ranks = np.argsort(-sim, axis=1, kind="stable")[:, :TOP_K]

    submission: Dict[str, List[str]] = {}
    for i, qid in enumerate(query_ids):
        submission[qid] = [corpus_ids[int(j)] for j in ranks[i]]

    validate_submission(submission=submission, expected_query_ids=query_ids, top_k=TOP_K)
    validate_doc_ids_in_corpus(submission=submission, corpus_doc_ids=corpus_ids)
    save_submission(submission=submission, output_file=paths["output_file"])
    create_submission_zip(submission_file=paths["output_file"], zip_file=paths["zip_file"])

    print("Iteration 8 submission generated.")
    print("Method: dense-only SPECTER2 (title [SEP] abstract)")
    print(f"Saved to: {paths['zip_file']}")


if __name__ == "__main__":
    main()
