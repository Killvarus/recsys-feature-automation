"""
Бейзлайн: эвристики и LightGBM на ручных признаках, абляции по группам.

python scripts/train_baseline.py                   # эвристики + LightGBM, 3 seed
python scripts/train_baseline.py --ablation        # + без каждой группы признаков (1 seed)
python scripts/train_baseline.py --grid            # подбор гиперпараметров по valid

Test используется только для итоговых чисел: early stopping и выбор параметров — по valid.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

from src.eval.lgbm_eval import BASE_PARAMS, Data, fit, score  # noqa: E402
from src.features.handcrafted import feature_group  # noqa: E402

def heuristics(data: Data) -> dict:
    """Простые правила без обучения: насколько сильны отдельные сигналы."""
    prior = float(data.parts["train"]["y"].mean())
    results = {}
    rules = {
        "prior": None,
        "user_algo_listened_rate": "u__listen_algo__mean_listened__all",
        "item_listened_rate": "i__listen__mean_listened__all",
        "user_item_last_ratio": "ui__listen__last_ratio",
        "user_last20_listened": "u__last20_listen__mean_listened",
    }
    for name, column in rules.items():
        if column is None:
            prediction = np.full(len(data.parts["test"]["y"]), prior)
            results[name] = score(data, "test", prediction) | {"auc": 0.5, "gauc": 0.5}
            continue
        values = data.parts["test"]["X"][:, data.columns([column])[0]].astype(np.float64)
        prediction = np.where(np.isnan(values), prior, values)
        results[name] = score(data, "test", prediction)
    return results


def summarize(runs: list[dict], part: str) -> dict:
    keys = runs[0][part].keys()
    return {k: (float(np.mean([r[part][k] for r in runs])), float(np.std([r[part][k] for r in runs]))) for k in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=ROOT / "data" / "features" / "handcrafted_shift0.parquet")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "baseline")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--skip-main", action="store_true", help="не обучать основную модель (результат берётся из report.json)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "report.json"

    def save() -> None:
        # Отчёт пишется после каждого этапа: прерванный запуск не теряет посчитанное.
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    started = time.perf_counter()
    data = Data(args.features)
    print(f"features {len(data.features)}, rows " + ", ".join(f"{k} {len(v['y']):,}" for k, v in data.parts.items())
          + f", load {time.perf_counter() - started:.1f}s", flush=True)
    report: dict = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    report["features"] = args.features.name

    report["heuristics"] = heuristics(data)
    for name, metrics in report["heuristics"].items():
        print(f"[heuristic] {name:<26} " + "  ".join(f"{k} {v:.4f}" for k, v in metrics.items()), flush=True)

    params = dict(BASE_PARAMS)
    if args.grid:
        grid = []
        for num_leaves in (63, 255):
            for min_data in (100, 1000):
                trial = params | {"num_leaves": num_leaves, "min_data_in_leaf": min_data, "learning_rate": 0.1}
                _, info = fit(data, data.features, trial, seed=0)
                info.pop("test_pred")
                grid.append({"num_leaves": num_leaves, "min_data_in_leaf": min_data, **info["valid"], "best_iteration": info["best_iteration"]})
                print(f"[grid] leaves {num_leaves} min_data {min_data}: valid logloss {info['valid']['logloss']:.5f} auc {info['valid']['auc']:.5f}", flush=True)
        best = min(grid, key=lambda g: g["logloss"])
        params |= {"num_leaves": best["num_leaves"], "min_data_in_leaf": best["min_data_in_leaf"]}
        report["grid"] = grid
    report["params"] = params
    save()

    if not args.skip_main:
        train_main(data, params, args, report)
        save()

    if args.ablation:
        ablation = report.setdefault("ablation", {})
        fast = params | {"learning_rate": 0.1}
        jobs = [("all", data.features)]
        for group in sorted({feature_group(f) for f in data.features}):
            jobs.append((f"without_{group}", [f for f in data.features if feature_group(f) != group]))
            jobs.append((f"only_{group}", [f for f in data.features if feature_group(f) == group]))
        for name, columns in jobs:
            if name in ablation and ablation[name].get("n_features") == len(columns):
                continue  # уже посчитано в прерванном запуске
            _, info = fit(data, columns, fast, seed=0)
            info.pop("test_pred")
            ablation[name] = info
            save()
            print(f"[ablation] {name:<26} test auc {info['test']['auc']:.4f} gauc {info['test']['gauc']:.4f}"
                  f" logloss {info['test']['logloss']:.4f}", flush=True)

    print(f"saved {report_path}, total {time.perf_counter() - started:.0f}s", flush=True)


def train_main(data: Data, params: dict, args, report: dict) -> None:
    runs = []
    importance = None
    for seed in args.seeds:
        booster, info = fit(data, data.features, params, seed)
        np.save(args.out / f"test_pred_lgbm_all_seed{seed}.npy", info.pop("test_pred"))
        runs.append(info)
        gain = booster.feature_importance("gain")
        importance = gain if importance is None else importance + gain
        print(f"[lgbm all] seed {seed}: test " + "  ".join(f"{k} {v:.4f}" for k, v in info["test"].items())
              + f"  iters {info['best_iteration']}  fit {info['fit_seconds']:.0f}s", flush=True)
    report["lgbm_all"] = {"runs": runs, "test": summarize(runs, "test"), "valid": summarize(runs, "valid")}
    pd.DataFrame({"feature": data.features, "group": [feature_group(f) for f in data.features], "gain": importance / len(args.seeds)}) \
        .sort_values("gain", ascending=False).to_csv(args.out / "importance_gain.csv", index=False)



if __name__ == "__main__":
    main()
