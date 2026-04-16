from __future__ import annotations

"""
Iteration 14 (train) — fine-tune a MiniLM bi-encoder on public qrels.

Objective: MultipleNegativesRankingLoss (in-batch negatives).
We train on (query, positive_doc) pairs constructed from `data/qrels.json`.

Output model is saved under `data/models/iter14_minilm_mnrl_v1/`.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.optim import AdamW
from torch.utils.data import DataLoader

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

import iteration_4 as i4

from submission_utils import challenge_dir, load_queries_corpus

MODEL_BASE = i4.MINILM_MODEL
OUT_DIRNAME = "iter14_minilm_mnrl_v1"

MAX_TRAIN_PAIRS = 6000  # cap for speed; bump if you have time
EPOCHS = 1
BATCH_SIZE = 16
LR = 2e-5


def model_out_dir() -> Path:
    return challenge_dir() / "data" / "models" / OUT_DIRNAME


def load_qrels(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): [str(x) for x in v] for k, v in raw.items()}


def main() -> None:
    root = challenge_dir()
    qrels_path = root / "data" / "qrels.json"
    queries_path = root / "data" / "queries.parquet"
    corpus_path = root / "data" / "corpus.parquet"

    if not (qrels_path.exists() and queries_path.exists() and corpus_path.exists()):
        raise FileNotFoundError("Expected `data/qrels.json`, `data/queries.parquet`, `data/corpus.parquet` under ir_challenge/")

    qrels = load_qrels(qrels_path)
    queries, corpus = load_queries_corpus(queries_path, corpus_path)

    query_by_id = {str(r.doc_id): r for r in queries.itertuples(index=False)}
    doc_row_by_id = {str(r.doc_id): r for r in corpus.itertuples(index=False)}

    pairs: List[InputExample] = []
    for qid, pos_ids in qrels.items():
        qrow = query_by_id.get(qid)
        if qrow is None:
            continue
        qtext = i4.format_rich_text(pd.Series(qrow._asdict()), i4.MAX_FULLTEXT_CHARS)
        for did in pos_ids:
            drow = doc_row_by_id.get(did)
            if drow is None:
                continue
            dtext = i4.format_rich_text(pd.Series(drow._asdict()), i4.MAX_FULLTEXT_CHARS)
            if not qtext or not dtext:
                continue
            pairs.append(InputExample(texts=[qtext, dtext]))

    if not pairs:
        raise RuntimeError("No training pairs constructed from qrels.")

    rng = np.random.default_rng(0)
    if len(pairs) > MAX_TRAIN_PAIRS:
        idx = rng.choice(len(pairs), size=MAX_TRAIN_PAIRS, replace=False)
        pairs = [pairs[int(i)] for i in idx.tolist()]

    out_dir = model_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model {MODEL_BASE} …")
    model = SentenceTransformer(MODEL_BASE)
    model.train()

    train_loss = losses.MultipleNegativesRankingLoss(model)
    train_loader = DataLoader(
        pairs,
        shuffle=True,
        batch_size=BATCH_SIZE,
        drop_last=True,
        collate_fn=model.smart_batching_collate,
    )

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    print(f"Training pairs: {len(pairs)} | batches/epoch: {len(train_loader)} | total_steps: {total_steps}")

    device = model.device
    global_step = 0
    for epoch in range(EPOCHS):
        for sentence_features, labels in train_loader:
            sentence_features = [
                {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sf.items()}
                for sf in sentence_features
            ]
            labels = labels.to(device)

            loss_val = train_loss(sentence_features, labels)
            loss_val.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if global_step % 25 == 0:
                print(f"step {global_step}/{total_steps} loss={float(loss_val.detach().cpu()):.4f}")

    model.save(str(out_dir))
    print(f"Saved fine-tuned model to: {out_dir}")


if __name__ == "__main__":
    main()

