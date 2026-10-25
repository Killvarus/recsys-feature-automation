"""
Сравнение методов отбора признаков одной итоговой моделью.

Протокол:
  - рейтинг или преобразование строится только по train (ранняя остановка нейросети — по valid);
  - проверка — LightGBM с одними и теми же параметрами на первых k признаках рейтинга
    (или на k компонентах PCA/PLS), early stopping по valid;
  - базовые линии: все признаки и случайные k (3 seed);
  - лучший вариант выбирается по valid, test — только для итоговых чисел.

Методы и их предобработка:
  random_s*, lgbm_gain, permutation         — сырые признаки (деревьям предобработка не нужна);
  corr, ppop, rfe_loo, nn_weights, gbc_builtin, pca, pls — LinearPreprocessor (PREPROCESSING_NAME).

Рейтинги кэшируются в experiments/selection/rankings/, результаты дописываются в results.csv —
прерванный запуск продолжается с места остановки.

python scripts/compare_selection.py
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.eval.lgbm_eval import BASE_PARAMS, Data, fit, fit_matrices  # noqa: E402
from src.models.linear_selection import LinearPPOP, LinearRFE, loo_alpha, ridge_spectrum, scale_channels  # noqa: E402
from src.preprocessing.pca import PCATransformer  # noqa: E402
from src.preprocessing.pls import PLSTransformer  # noqa: E402
from src.selection.boosting import builtin_importance_ranking, permutation_ranking  # noqa: E402
from src.selection.nn_weights import nn_weight_ranking  # noqa: E402
from src.selection.preprocess import PREPROCESSING_NAME, LinearPreprocessor  # noqa: E402

DAY = 86_400
PARAMS = BASE_PARAMS | {"learning_rate": 0.1}  # одна итоговая модель для всех вариантов
ALPHAS = np.logspace(-3, 8, 56)  # n ~ 3·10⁵: собственные числа XᵀX порядка n, сетка должна доходить до таких alpha


def _sample(n: int, size: int, seed: int) -> np.ndarray:
    return np.sort(np.random.default_rng(seed).choice(n, size=min(size, n), replace=False))


def _frame(names: list[str], scores: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"Feature": names, "Score": scores}).sort_values(["Score", "Feature"], ascending=[False, True])
    frame["Individual_Rank"] = np.arange(1, len(frame) + 1)
    return frame.reset_index(drop=True)


def _quiet(func):
    with contextlib.redirect_stdout(io.StringIO()):
        return func()


def build_rankings(data: Data, out: Path, args) -> tuple[dict[str, list[str]], LinearPreprocessor, np.ndarray, np.ndarray]:
    rank_dir = out / "rankings"
    rank_dir.mkdir(parents=True, exist_ok=True)
    names = data.features
    train = data.parts["train"]
    sel = _sample(len(train["y"]), args.n_select, 0)
    y_sel = train["y"][sel].astype(np.float64)
    pre = LinearPreprocessor(names).fit(train["X"][sel])
    X_sel = pre.transform(train["X"][sel])
    meta_path = out / "rankings_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    rankings: dict[str, list[str]] = {}

    def cached(name: str, preprocessing: str, builder) -> None:
        path = rank_dir / f"{name}.csv"
        if not path.exists():
            started = time.perf_counter()
            frame = builder()
            meta[name] = {"seconds": round(time.perf_counter() - started, 1), "preprocessing": preprocessing,
                          "rows": int(args.n_select)}
            frame.to_csv(path, index=False)
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[ranking] {name:<12} {meta[name]['seconds']:7.1f}s  top5: {frame['Feature'].head(5).tolist()}", flush=True)
        rankings[name] = pd.read_csv(path).sort_values("Individual_Rank")["Feature"].tolist()

    for seed in range(3):
        cached(f"random_s{seed}", "none", lambda seed=seed: _frame(names, np.random.default_rng(100 + seed).permutation(len(names)).astype(float)))

    def corr():
        centered = X_sel - X_sel.mean(axis=0)
        yc = y_sel - y_sel.mean()
        denominator = np.sqrt((centered**2).sum(axis=0) * (yc @ yc))
        return _frame(names, np.abs(centered.T @ yc) / np.where(denominator == 0, 1, denominator))

    cached("corr", PREPROCESSING_NAME, corr)

    # alpha для BSBS — LOO на train на всех признаках.
    if "alpha_all_features" not in meta:
        (scaled,) = scale_channels(X_sel, method="zscore")
        u, s2, _ = ridge_spectrum(scaled)
        meta["alpha_all_features"] = loo_alpha(u, s2, y_sel - y_sel.mean(), ALPHAS)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[alpha] LOO on train, all features: {meta['alpha_all_features']:g}", flush=True)
        del u, scaled
    alpha = float(meta["alpha_all_features"])

    frame_sel = pd.DataFrame(X_sel, columns=names)
    cached("ppop", PREPROCESSING_NAME,
           lambda: _quiet(lambda: LinearPPOP(alpha=alpha, step=0.1, cv=5).fit(frame_sel, y_sel)).ranking_frame())
    cached("rfe_loo", PREPROCESSING_NAME,
           lambda: _quiet(lambda: LinearRFE(alpha="loo", step=0.1).fit(frame_sel, y_sel)).ranking_frame())

    def nn():
        valid = data.parts["valid"]
        vs = _sample(len(valid["y"]), args.n_nn_valid, 1)
        frame, infos = nn_weight_ranking(names, X_sel, y_sel, pre.transform(valid["X"][vs]), valid["y"][vs].astype(np.float64),
                                         runs=5, top_n=30, threshold=3, hidden=32)
        print(f"  nn runs: {infos}", flush=True)
        return frame

    cached("nn_weights", PREPROCESSING_NAME, nn)
    cached("gbc_builtin", PREPROCESSING_NAME,
           lambda: builtin_importance_ranking(names, X_sel[: args.n_gbc], y_sel[: args.n_gbc]))

    def lgbm_gain():
        # Важности — из бейзлайна ТОГО ЖЕ разбиения: бейзлайн другого разбиения видел бы его valid/test.
        gain = pd.read_csv(args.baseline_dir / "importance_gain.csv").set_index("feature")["gain"]
        return _frame(names, gain.reindex(names).fillna(0.0).to_numpy())

    cached("lgbm_gain", "none", lgbm_gain)

    def permutation():
        day = train["timestamp"] // DAY
        early = np.flatnonzero(day < args.permutation_split_day)
        late = np.flatnonzero(day >= args.permutation_split_day)
        fit_idx = early[_sample(len(early), args.n_select, 2)]
        eval_idx = late[_sample(len(late), args.n_permutation_eval, 3)]
        return permutation_ranking(names, train["X"][fit_idx], train["y"][fit_idx], train["X"][eval_idx], train["y"][eval_idx])

    cached("permutation", "none", permutation)
    return rankings, pre, X_sel, y_sel


def _transform_chunks(data: Data, pre: LinearPreprocessor, transformer, names: list[str], chunk: int = 300_000) -> dict:
    matrices = {}
    for part in ("train", "valid", "test"):
        X = data.parts[part]["X"]
        pieces = [transformer.transform(pd.DataFrame(pre.transform(X[i:i + chunk]), columns=names)).to_numpy(np.float32)
                  for i in range(0, len(X), chunk)]
        matrices[part] = np.vstack(pieces)
    return matrices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=ROOT / "data" / "features" / "handcrafted_shift0.parquet")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "selection")
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20, 30, 50, 100])
    parser.add_argument("--component-ks", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--n-select", type=int, default=300_000, help="строк train для построения рейтингов")
    parser.add_argument("--n-nn-valid", type=int, default=200_000)
    parser.add_argument("--n-gbc", type=int, default=100_000)
    parser.add_argument("--n-permutation-eval", type=int, default=150_000)
    parser.add_argument("--permutation-split-day", type=int, default=281, help="граница ранней и поздней части train")
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "experiments" / "baseline",
                        help="бейзлайн того же разбиения (importance_gain.csv для рейтинга lgbm_gain)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    data = Data(args.features)
    print(f"features {len(data.features)}, load {time.perf_counter() - started:.1f}s", flush=True)
    rankings, pre, X_sel, y_sel = build_rankings(data, args.out, args)

    results_path = args.out / "results.csv"
    results = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    done = set(zip(results.get("method", []), results.get("k", [])))

    def record(method: str, k: int, info: dict) -> None:
        nonlocal results
        row = {"method": method, "k": k, "best_iteration": info["best_iteration"], "fit_seconds": round(info["fit_seconds"], 1),
               **{f"valid_{m}": v for m, v in info["valid"].items()}, **{f"test_{m}": v for m, v in info["test"].items()}}
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
        results.to_csv(results_path, index=False)
        done.add((method, k))
        print(f"[eval] {method:<12} k={k:<4} valid auc {row['valid_auc']:.4f} logloss {row['valid_logloss']:.4f} | "
              f"test auc {row['test_auc']:.4f} gauc {row['test_gauc']:.4f}  ({row['fit_seconds']:.0f}s)", flush=True)

    all_k = len(data.features)
    if ("all", all_k) not in done:
        _, info = fit(data, data.features, PARAMS, seed=0)
        record("all", all_k, info)
    for k in args.ks:
        for method, order in rankings.items():
            if (method, k) not in done:
                _, info = fit(data, order[:k], PARAMS, seed=0)
                record(method, k, info)

    names = data.features
    for method, factory in (("pca", lambda: PCATransformer(n_components=max(args.component_ks)).fit(pd.DataFrame(X_sel, columns=names))),
                            ("pls", lambda: PLSTransformer(n_components=max(args.component_ks)).fit(pd.DataFrame(X_sel, columns=names), pd.Series(y_sel)))):
        if all((method, k) in done for k in args.component_ks):
            continue
        t0 = time.perf_counter()
        transformer = factory()
        matrices = _transform_chunks(data, pre, transformer, names)
        print(f"[{method}] fit+transform {time.perf_counter() - t0:.0f}s", flush=True)
        for k in args.component_ks:
            if (method, k) not in done:
                comp = {part: m[:, :k] for part, m in matrices.items()}
                _, info = fit_matrices(data, comp, [f"{method}{i + 1}" for i in range(k)], PARAMS, seed=0)
                record(method, k, info)
        del matrices

    summary = results.pivot_table(index="method", columns="k", values=["valid_auc", "test_auc"])
    summary.to_csv(args.out / "summary_auc.csv")
    best = results.loc[results.groupby("method")["valid_logloss"].idxmin()].sort_values("valid_logloss")
    best.to_csv(args.out / "best_by_valid.csv", index=False)
    print(best[["method", "k", "valid_logloss", "valid_auc", "test_logloss", "test_auc", "test_gauc", "test_ndcg@10"]].to_string(index=False))
    print(f"total {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
