"""
Автоматические признаки против ручных: одна итоговая модель, одни строки, одно разбиение.

Варианты (LightGBM с одинаковыми параметрами, early stopping по valid):
  hand_all            — все ручные признаки;
  auto_all            — весь автоматический пул;
  auto_<метод>_k      — первые k автоматических признаков по рейтингу (BSBS или gain LightGBM);
  hand_<метод>_k      — то же для ручных признаков (рейтинг построен на train ручного набора);
  hand_plus_auto_k    — все ручные + первые k автоматических по BSBS, которых нет среди ручных по имени.
Рейтинги строятся только по train. Выбор — по valid, test — итог.

python scripts/compare_auto.py
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

from src.eval.lgbm_eval import BASE_PARAMS, META, Data, fit, fit_matrices  # noqa: E402
from src.models.linear_selection import LinearPPOP, loo_alpha, ridge_spectrum, scale_channels  # noqa: E402
from src.selection.preprocess import PREPROCESSING_NAME, LinearPreprocessor  # noqa: E402

PARAMS = BASE_PARAMS | {"learning_rate": 0.1}
ALPHAS = np.logspace(-3, 8, 56)
KS = [10, 30, 50, 100, 200]


def bsbs_ranking(path: Path, n_select: int, out: Path) -> tuple[list[str], dict]:
    """BSBS (обратный пакетный отбор на Ridge) по train автоматического пула."""
    cache = out / "ranking_auto_bsbs.csv"
    meta_path = out / "ranking_auto_bsbs.json"
    if cache.exists():
        return pd.read_csv(cache).sort_values("Individual_Rank")["Feature"].tolist(), json.loads(meta_path.read_text())
    frame = pl.scan_parquet(path).filter(pl.col("part") == "train").collect()
    names = [c for c in frame.columns if c not in META]
    sample = frame.sample(n=min(n_select, frame.height), seed=0)
    del frame
    X_raw = sample.select(names).to_numpy()
    y = sample["label"].to_numpy().astype(np.float64)
    del sample
    started = time.perf_counter()
    X = LinearPreprocessor(names).fit(X_raw).transform(X_raw)
    del X_raw
    (scaled,) = scale_channels(X, method="zscore")
    u, s2, _ = ridge_spectrum(scaled)
    alpha = loo_alpha(u, s2, y - y.mean(), ALPHAS)
    del u, scaled
    with contextlib.redirect_stdout(io.StringIO()):
        selector = LinearPPOP(alpha=alpha, step=0.1, cv=5).fit(pd.DataFrame(X, columns=names), y)
    ranking = selector.ranking_frame()
    ranking.to_csv(cache, index=False)
    selector.history_.to_csv(out / "ranking_auto_bsbs_history.csv", index=False)
    meta = {"alpha": alpha, "seconds": round(time.perf_counter() - started, 1), "rows": len(y),
            "preprocessing": PREPROCESSING_NAME}
    meta_path.write_text(json.dumps(meta, indent=2))
    return ranking["Feature"].tolist(), meta


def window_of(name: str) -> str:
    match = re.search(r"__(\d+[hd]|all)$", name)
    return match.group(1) if match else ("last_k" if "__last" in name and "_listen__" in name else "other")


def key_of(name: str) -> str:
    return name.split("__")[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", type=Path, default=ROOT / "data" / "features" / "auto_shift0.parquet")
    parser.add_argument("--hand", type=Path, default=ROOT / "data" / "features" / "handcrafted_shift0.parquet")
    parser.add_argument("--hand-rankings", type=Path, default=ROOT / "experiments" / "selection" / "rankings")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "auto")
    parser.add_argument("--n-select", type=int, default=300_000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    bsbs, bsbs_meta = bsbs_ranking(args.auto, args.n_select, args.out)
    print(f"[bsbs] auto pool {len(bsbs)} features, alpha {bsbs_meta['alpha']:g}, {bsbs_meta['seconds']}s", flush=True)

    results_path = args.out / "results.csv"
    results = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    done = set(results.get("variant", []))

    def record(variant: str, info: dict, n: int) -> None:
        nonlocal results
        row = {"variant": variant, "n_features": n, "best_iteration": info["best_iteration"],
               "fit_seconds": round(info["fit_seconds"], 1),
               **{f"valid_{m}": v for m, v in info["valid"].items()}, **{f"test_{m}": v for m, v in info["test"].items()}}
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
        results.to_csv(results_path, index=False)
        done.add(variant)
        print(f"[eval] {variant:<22} n={n:<4} valid logloss {row['valid_logloss']:.4f} auc {row['valid_auc']:.4f} | "
              f"test logloss {row['test_logloss']:.4f} auc {row['test_auc']:.4f} gauc {row['test_gauc']:.4f}"
              f"  ({row['fit_seconds']:.0f}s)", flush=True)

    query_ids = np.load(args.auto.with_suffix(".query_ids.npy"))

    # Ручные признаки на тех же строках.
    hand = Data(args.hand, query_ids=query_ids)
    if "hand_all" not in done:
        _, info = fit(hand, hand.features, PARAMS, seed=0)
        record("hand_all", info, len(hand.features))
    for method in ("ppop", "lgbm_gain"):
        order = pd.read_csv(args.hand_rankings / f"{method}.csv").sort_values("Individual_Rank")["Feature"].tolist()
        for k in KS[:-1]:
            variant = f"hand_{'bsbs' if method == 'ppop' else 'gain'}_{k}"
            if variant not in done:
                _, info = fit(hand, order[:k], PARAMS, seed=0)
                record(variant, info, k)
    hand_names = list(hand.features)
    hand_matrices = {part: hand.parts[part]["X"] for part in ("train", "valid", "test")}

    # Автоматический пул.
    auto = Data(args.auto, query_ids=query_ids)
    assert all((auto.parts[p]["y"] == hand.parts[p]["y"]).all() for p in ("train", "valid", "test")), "строки не совпадают"
    gain_path = args.out / "ranking_auto_gain.csv"
    if "auto_all" not in done or not gain_path.exists():
        booster, info = fit(auto, auto.features, PARAMS, seed=0)
        if "auto_all" not in done:
            record("auto_all", info, len(auto.features))
        pd.DataFrame({"Feature": auto.features, "Score": booster.feature_importance("gain")}) \
            .sort_values("Score", ascending=False).assign(Individual_Rank=lambda d: np.arange(1, len(d) + 1)).to_csv(gain_path, index=False)
    gain = pd.read_csv(gain_path)["Feature"].tolist()
    for name, order in (("bsbs", bsbs), ("gain", gain)):
        for k in KS:
            variant = f"auto_{name}_{k}"
            if variant not in done:
                _, info = fit(auto, order[:k], PARAMS, seed=0)
                record(variant, info, k)

    # Ручные + автоматические, которых нет среди ручных.
    extra = [f for f in bsbs if f not in set(hand_names)]
    for k in (30, 100):
        variant = f"hand_plus_auto_{k}"
        if variant not in done:
            idx = auto.columns(extra[:k])
            matrices = {p: np.hstack([hand_matrices[p], auto.parts[p]["X"][:, idx]]) for p in ("train", "valid", "test")}
            _, info = fit_matrices(auto, matrices, hand_names + extra[:k], PARAMS, seed=0)
            record(variant, info, len(hand_names) + k)
            del matrices

    # Какие окна и связи выбирает отбор.
    summary = {}
    for name, order in (("bsbs", bsbs), ("gain", gain)):
        for k in (30, 100):
            summary[f"{name}_top{k}_windows"] = dict(Counter(window_of(f) for f in order[:k]).most_common())
            summary[f"{name}_top{k}_keys"] = dict(Counter(key_of(f) for f in order[:k]).most_common())
    summary["pool_windows"] = dict(Counter(window_of(f) for f in auto.features).most_common())
    summary["pool_size"] = len(auto.features)
    summary["hand_size"] = len(hand_names)
    summary["auto_not_in_hand"] = len(extra)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"total {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
