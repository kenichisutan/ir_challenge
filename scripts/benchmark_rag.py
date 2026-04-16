from __future__ import annotations

"""
Benchmark + lightweight RAG-style inspection on public data.

What it does:
1) Scores a submission against `data/qrels.json` on public queries.
2) Writes a markdown report with MAP, Recall@100, NDCG@10.
3) Adds a small extractive RAG view: top retrieved snippets per query.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def load_submission(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): [str(x) for x in v] for k, v in raw.items()}


def ap_at_k(pred: Sequence[str], rel: set[str], k: int = 100) -> float:
    hit = 0
    s = 0.0
    for i, doc_id in enumerate(pred[:k], start=1):
        if doc_id in rel:
            hit += 1
            s += hit / i
    denom = max(1, min(len(rel), k))
    return s / denom


def recall_at_k(pred: Sequence[str], rel: set[str], k: int = 100) -> float:
    if not rel:
        return 0.0
    return len(set(pred[:k]).intersection(rel)) / len(rel)


def ndcg_at_k(pred: Sequence[str], rel: set[str], k: int = 10) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(pred[:k], start=1):
        if doc_id in rel:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return 0.0 if idcg == 0.0 else dcg / idcg


def score_submission(submission: Dict[str, List[str]], qrels: Dict[str, List[str]]) -> Tuple[float, float, float]:
    qids = [qid for qid in qrels.keys() if qid in submission]
    if not qids:
        raise ValueError("No overlapping query IDs between submission and qrels.")
    aps = []
    recs = []
    ndcgs = []
    for qid in qids:
        rel = set(qrels[qid])
        pred = submission[qid]
        aps.append(ap_at_k(pred, rel, 100))
        recs.append(recall_at_k(pred, rel, 100))
        ndcgs.append(ndcg_at_k(pred, rel, 10))
    n = len(qids)
    return sum(aps) / n, sum(recs) / n, sum(ndcgs) / n


def build_public_tfidf_submission(queries: pd.DataFrame, corpus: pd.DataFrame, top_k: int = 100) -> Dict[str, List[str]]:
    def ta(row: pd.Series) -> str:
        t = "" if pd.isna(row.get("title")) else str(row.get("title"))
        a = "" if pd.isna(row.get("abstract")) else str(row.get("abstract"))
        return f"{t} {a}".strip()

    q_texts = queries.apply(ta, axis=1).tolist()
    c_texts = corpus.apply(ta, axis=1).tolist()

    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.95, stop_words="english")
    X = vec.fit_transform(c_texts)
    Q = vec.transform(q_texts)
    sim = Q @ X.T
    ranks = sim.toarray().argsort(axis=1)[:, ::-1][:, :top_k]

    cids = corpus["doc_id"].astype(str).tolist()
    qids = queries["doc_id"].astype(str).tolist()
    out: Dict[str, List[str]] = {}
    for i, qid in enumerate(qids):
        out[qid] = [cids[int(j)] for j in ranks[i]]
    return out


def safe_text(x: object) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip().replace("\n", " ")


def build_rag_rows(
    submission: Dict[str, List[str]],
    qrels: Dict[str, List[str]],
    queries: pd.DataFrame,
    corpus: pd.DataFrame,
    n_queries: int,
    per_query_docs: int,
) -> List[str]:
    q_map = {str(row["doc_id"]): row for _, row in queries.iterrows()}
    c_map = {str(row["doc_id"]): row for _, row in corpus.iterrows()}
    lines: List[str] = []
    shown = 0
    for qid, rel_docs in qrels.items():
        if shown >= n_queries or qid not in submission or qid not in q_map:
            continue
        q = q_map[qid]
        lines.append(f"### Query {qid}")
        lines.append(f"- Question: {safe_text(q.get('title'))} {safe_text(q.get('abstract'))}".strip())
        lines.append(f"- #Relevant in qrels: {len(rel_docs)}")
        top_docs = submission[qid][:per_query_docs]
        lines.append("- Retrieved context:")
        for rank, doc_id in enumerate(top_docs, start=1):
            d = c_map.get(doc_id)
            if d is None:
                lines.append(f"  - {rank}. {doc_id} (missing in corpus)")
                continue
            title = safe_text(d.get("title"))
            abstract = safe_text(d.get("abstract"))
            snippet = abstract[:320] + ("..." if len(abstract) > 320 else "")
            rel_mark = " [relevant]" if doc_id in set(rel_docs) else ""
            lines.append(f"  - {rank}. {doc_id}{rel_mark}: {title} :: {snippet}")
        lines.append("")
        shown += 1
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark retrieval + lightweight RAG inspection.")
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path("submissions/iteration_4/submission_data.json"),
        help="Path to submission_data.json to benchmark on public qrels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/benchmark_rag_report.md"),
        help="Output markdown report path.",
    )
    parser.add_argument("--n_queries", type=int, default=10, help="How many queries to include in RAG section.")
    parser.add_argument("--docs_per_query", type=int, default=3, help="Retrieved docs shown per query.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    qrels_path = root / "data" / "qrels.json"
    queries_path = root / "data" / "queries.parquet"
    corpus_path = root / "data" / "corpus.parquet"

    with qrels_path.open("r", encoding="utf-8") as f:
        qrels = {str(k): [str(x) for x in v] for k, v in json.load(f).items()}
    submission = load_submission(root / args.submission if not args.submission.is_absolute() else args.submission)
    queries = pd.read_parquet(queries_path)
    corpus = pd.read_parquet(corpus_path)

    overlap = set(submission.keys()).intersection(set(qrels.keys()))
    source_label = str(args.submission)
    if not overlap:
        print("No qrels overlap with submission IDs; generating public TF-IDF baseline for benchmarking.")
        submission = build_public_tfidf_submission(queries, corpus, top_k=100)
        source_label = "auto-generated public TF-IDF baseline"

    map_v, rec_v, ndcg_v = score_submission(submission, qrels)
    rag_lines = build_rag_rows(
        submission=submission,
        qrels=qrels,
        queries=queries,
        corpus=corpus,
        n_queries=args.n_queries,
        per_query_docs=args.docs_per_query,
    )

    out = root / args.output if not args.output.is_absolute() else args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    report = [
        "# Retrieval Benchmark + RAG Inspection",
        "",
        f"- Submission source: `{source_label}`",
        f"- mean_average_precision: **{map_v:.4f}**",
        f"- recall_at_100: **{rec_v:.4f}**",
        f"- ndcg_at_10: **{ndcg_v:.4f}**",
        "",
        "## Qualitative RAG Context View",
        "",
        *rag_lines,
    ]
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote report to: {out}")
    print(f"MAP={map_v:.4f} | Recall@100={rec_v:.4f} | NDCG@10={ndcg_v:.4f}")


if __name__ == "__main__":
    main()
