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
    return {
        "challenge_dir": root,
        "data_dir": data,
        "queries_path": data / "queries.parquet",
        "corpus_path": data / "corpus.parquet",
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


def save_submission(submission: Dict[str, List[str]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)


def create_submission_zip(submission_file: Path, zip_file: Path, arcname: str = "submission_data.json") -> None:
    zip_file.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_file, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(submission_file, arcname=arcname)
