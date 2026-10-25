"""
Прогноз модели над историей событий как признак для LightGBM.

Модель обучена на ранней части train (scripts/train_seq.py --mode stack), её прогноз есть для поздней части
train (день события ≥ stack_day), valid и test. LightGBM обучается только на поздней части train, поэтому
прогноз на его train не переобучен. Valid и test — те же строки, что у сравнения ручных и автоматических признаков.

Варианты (3 seed, одинаковые параметры):
  hand_all            — все 246 ручных признаков;
  hand_all+seq        — то же + прогноз модели;
  hand_bsbs_30        — первые 30 ручных по BSBS;
  hand_bsbs_30+seq    — то же + прогноз модели.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

from src.eval.lgbm_eval import BASE_PARAMS, Data, fit_matrices  # noqa: E402

PARAMS = BASE_PARAMS | {"learning_rate": 0.1}


def main() -> None:
    out = ROOT / "experiments" / "seq"
    preds = pl.read_parquet(out / "preds_stack.parquet")
    hand = Data(ROOT / "data" / "features" / "handcrafted_shift0.parquet", query_ids=preds["query_id"].to_numpy())
    scores = dict(zip(preds["query_id"].to_list(), preds["seq_score"].to_list()))
    ids = {p: pl.read_parquet(ROOT / "data" / "features" / "handcrafted_shift0.parquet", columns=["query_id", "part"])
           .filter(pl.col("query_id").is_in(preds["query_id"].implode()) & (pl.col("part") == p)).sort("query_id")["query_id"].to_numpy()
           for p in ("train", "valid", "test")}
    seq_col = {p: np.array([scores[q] for q in ids[p]], dtype=np.float32)[:, None] for p in ids}
    print({p: len(v) for p, v in ids.items()}, flush=True)

    bsbs = pd.read_csv(ROOT / "experiments" / "selection" / "rankings" / "ppop.csv").sort_values("Individual_Rank")["Feature"].tolist()[:30]
    variants = {
        "hand_all": hand.features,
        "hand_bsbs_30": bsbs,
    }
    rows = []
    for seed in (0, 1, 2):
        for name, columns in variants.items():
            idx = hand.columns(columns)
            for with_seq in (False, True):
                matrices = {p: hand.parts[p]["X"][:, idx] for p in ("train", "valid", "test")}
                names = list(columns)
                if with_seq:
                    matrices = {p: np.hstack([m, seq_col[p]]) for p, m in matrices.items()}
                    names = names + ["seq_score"]
                _, info = fit_matrices(hand, matrices, names, PARAMS, seed)
                variant = name + ("+seq" if with_seq else "")
                rows.append({"variant": variant, "seed": seed, "n_features": len(names),
                             "valid_logloss": info["valid"]["logloss"],
                             **{f"test_{k}": v for k, v in info["test"].items()}, "fit_seconds": info["fit_seconds"]})
                print(f"[stack] {variant:<18} seed {seed}: test logloss {info['test']['logloss']:.4f} auc {info['test']['auc']:.4f} "
                      f"gauc {info['test']['gauc']:.4f} ndcg {info['test']['ndcg@10']:.4f}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "stack_results.csv", index=False)
    summary = frame.groupby("variant")[["valid_logloss", "test_logloss", "test_auc", "test_gauc", "test_ndcg@10"]].agg(["mean", "std"])
    print(summary.round(4).to_string(), flush=True)
    seq_only = json.loads((out / "report_stack.json").read_text(encoding="utf-8"))["test"]
    print("seq model alone (stack, test):", {k: round(v, 4) for k, v in seq_only.items()}, flush=True)


if __name__ == "__main__":
    main()
