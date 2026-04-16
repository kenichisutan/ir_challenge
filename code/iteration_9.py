from __future__ import annotations

"""
Iteration 9 — hybrid + safer cross-encoder rerank.

Untried variant over 6a:
- Rerank only top-50 candidates (smaller damage surface).
- Blend CE score with base hybrid rank prior instead of hard overwrite.
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

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

ITERATION_NAME = "iteration_9"
TOP_K = DEFAULT_TOP_K
RERANK_M = 50
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_BATCH = 32
LAMBDA_CE = 0.35


def minmax(x: np.ndarray) -> np.ndarray:
    lo = float(x.min())
    hi = float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def rerank_blended(
    query_texts: List[str],
    corpus_texts: List[str],
    candidates: np.ndarray,
    ce: CrossEncoder,
) -> np.ndarray:
    nq, k = candidates.shape
    out = np.zeros_like(candidates)
    base_prior = np.array([1.0 / (i4.RRF_K + (r + 1)) for r in range(k)], dtype=np.float32)

    for i in range(nq):
        row = candidates[i]
        m = min(RERANK_M, k)
        pairs = [[query_texts[i], corpus_texts[int(doc_idx)]] for doc_idx in row[:m]]
        ce_scores = ce.predict(pairs, batch_size=CE_BATCH, show_progress_bar=False)
        ce_norm = minmax(np.asarray(ce_scores, dtype=np.float32))

        final = base_prior.copy()
        final[:m] = (1.0 - LAMBDA_CE) * base_prior[:m] + LAMBDA_CE * ce_norm
        order = np.argsort(-final, kind="stable")
        out[i] = row[order]
    return out


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

    query_texts = queries.apply(lambda r: i4.format_rich_text(r, i4.MAX_FULLTEXT_CHARS), axis=1).tolist()
    corpus_texts = corpus.apply(lambda r: i4.format_rich_text(r, i4.MAX_FULLTEXT_CHARS), axis=1).tolist()

    print(f"Loading bi-encoder {i4.MINILM_MODEL} …")
    bi = SentenceTransformer(i4.MINILM_MODEL)
    print("Building Iteration 4 candidate set …")
    candidates = i4.compute_iter4_fused_indices(queries, corpus, bi, top_k=TOP_K, fusion_depth=i4.FUSION_DEPTH)

    print(f"Loading cross-encoder {CE_MODEL} …")
    ce = CrossEncoder(CE_MODEL)
    print(f"Reranking top-{RERANK_M} with CE + blended prior …")
    reranked = rerank_blended(query_texts, corpus_texts, candidates, ce)

    submission: Dict[str, List[str]] = {}
    for i, qid in enumerate(query_ids):
        submission[qid] = [corpus_ids[int(j)] for j in reranked[i]]

    validate_submission(submission=submission, expected_query_ids=query_ids, top_k=TOP_K)
    validate_doc_ids_in_corpus(submission=submission, corpus_doc_ids=corpus_ids)
    save_submission(submission=submission, output_file=paths["output_file"])
    create_submission_zip(submission_file=paths["output_file"], zip_file=paths["zip_file"])

    print("Iteration 9 submission generated.")
    print(
        f"Method: Iteration 4 candidates + CE top-{RERANK_M} blend "
        f"(lambda={LAMBDA_CE}, model={CE_MODEL})"
    )
    print(f"Saved to: {paths['zip_file']}")


if __name__ == "__main__":
    main()
