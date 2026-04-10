"""
Shared helpers for challenge submissions: paths, loading, validation, JSON export, zip.

Used by iteration scripts under `code/`. Resolve the challenge root from this file
(`scripts/` -> parent is `ir_challenge/`).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

DEFAULT_TOP_K = 100


def challenge_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def data_paths() -> dict[str, Path]:
    root = challenge_dir()
    data = root / "data"
    candidates = [
        data / "held_out_queries.parquet",
        root / "held_out_queries.parquet",
        root / "starter_kit" / "held_out_queries.parquet",
    ]
    held_out = next((p for p in candidates if p.exists()), None)
    queries = held_out if held_out is not None else data / "queries.parquet"
    queries_source = str(queries.relative_to(root)) if queries.exists() else str(queries)
    return {
        "challenge_dir": root,
        "data_dir": data,
        "queries_path": queries,
        "queries_source": queries_source,
        "using_held_out_queries": held_out is not None,
        "corpus_path": data / "corpus.parquet",
        "sample_submission_path": data / "sample_submission.json",
    }


def iteration_submission_paths(iteration_name: str) -> dict[str, Path]:
    """
    iteration_name: folder under submissions/, e.g. "iteration_1".
    Writes submission_data.json and <iteration_name>.zip in that folder.
    """
    root = challenge_dir()
    out = root / "submissions" / iteration_name
    return {
        "output_dir": out,
        "output_file": out / "submission_data.json",
        "zip_file": out / f"{iteration_name}.zip",
    }


def format_title_abstract(row: pd.Series) -> str:
    title = "" if pd.isna(row.get("title")) else str(row.get("title"))
    abstract = "" if pd.isna(row.get("abstract")) else str(row.get("abstract"))
    return f"{title} {abstract}".strip()


def load_queries_corpus(queries_path: Path, corpus_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    queries = pd.read_parquet(queries_path)
    corpus = pd.read_parquet(corpus_path)
    return queries, corpus


def validate_submission(
    submission: Dict[str, List[str]],
    expected_query_ids: List[str],
    top_k: int = DEFAULT_TOP_K,
) -> None:
    submission_query_ids = set(submission.keys())
    expected_query_id_set = set(expected_query_ids)
    if submission_query_ids != expected_query_id_set:
        missing = expected_query_id_set - submission_query_ids
        extra = submission_query_ids - expected_query_id_set
        raise ValueError(f"Submission query IDs mismatch. Missing={len(missing)}, Extra={len(extra)}")

    for qid in expected_query_ids:
        ranked_docs = submission[qid]
        if len(ranked_docs) != top_k:
            raise ValueError(f"Query {qid}: expected {top_k} documents, got {len(ranked_docs)}")
        if not all(isinstance(doc_id, str) for doc_id in ranked_docs):
            raise ValueError(f"Query {qid}: all document IDs must be strings")
        if len(set(ranked_docs)) != top_k:
            raise ValueError(f"Query {qid}: duplicate document IDs found in top-{top_k}")


def validate_doc_ids_in_corpus(submission: Dict[str, List[str]], corpus_doc_ids: List[str]) -> None:
    corpus_set = set(corpus_doc_ids)
    invalid = 0
    for ranked_docs in submission.values():
        for doc_id in ranked_docs:
            if doc_id not in corpus_set:
                invalid += 1
    if invalid > 0:
        raise ValueError(f"Submission contains {invalid} doc IDs not found in corpus")


def load_sample_query_ids(sample_submission_path: Path) -> List[str]:
    with sample_submission_path.open("r", encoding="utf-8") as f:
        sample = json.load(f)
    return [str(qid) for qid in sample.keys()]


def save_submission(submission: Dict[str, List[str]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)


def create_submission_zip(submission_file: Path, zip_file: Path, arcname: str = "submission_data.json") -> None:
    zip_file.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_file, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(submission_file, arcname=arcname)
