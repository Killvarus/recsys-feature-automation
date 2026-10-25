"""
Общая итоговая модель для всех сравнений: LightGBM с фиксированными параметрами,
early stopping по valid, метрики на valid и test. Бейзлайн, абляции и проверка
рейтингов отбора признаков используют одну и ту же функцию fit.
"""
from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from src.eval.metrics import evaluate

META = ["query_id", "uid", "item_id", "artist_id", "album_id", "timestamp", "session", "event_ts", "eval_session",
        "track_length_seconds", "part", "label"]

BASE_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "num_threads": 6,
    "verbose": -1,
}


class Data:
    def __init__(self, path: Path, query_ids=None, columns: list[str] | None = None):
        """query_ids — оставить только эти примеры; columns — только эти признаки (экономия памяти)."""
        lazy = pl.scan_parquet(path)
        if query_ids is not None:
            lazy = lazy.filter(pl.col("query_id").is_in(pl.Series(query_ids).implode()))
        if columns is not None:
            lazy = lazy.select([c for c in META if c in lazy.collect_schema().names()] + list(columns))
        self.features = [c for c in lazy.collect_schema().names() if c not in META]
        self.parts = {}
        for part in ("train", "valid", "test"):
            # По одной части: в памяти не держится весь кадр вместе с копией в numpy.
            sub = lazy.filter(pl.col("part") == part).collect().sort("query_id")
            self.parts[part] = {
                "X": sub.select(self.features).to_numpy(),
                "y": sub["label"].to_numpy(),
                "uid": sub["uid"].to_numpy(),
                "session": sub["eval_session"].to_numpy(),  # настоящая сессия — только для NDCG
                "timestamp": sub["timestamp"].to_numpy(),  # момент решения
            }
            del sub

    def columns(self, names: list[str]) -> list[int]:
        index = {n: i for i, n in enumerate(self.features)}
        return [index[n] for n in names]


def score(data: Data, part: str, prediction: np.ndarray, probabilities: bool = True) -> dict:
    p = data.parts[part]
    return evaluate(p["uid"], p["session"], p["y"], prediction, probabilities=probabilities)


def fit(data: Data, columns: list[str], params: dict, seed: int, rounds: int = 5000) -> tuple[lgb.Booster, dict]:
    """LightGBM на столбцах columns исходных признаков."""
    idx = data.columns(columns)
    matrices = {part: data.parts[part]["X"][:, idx] for part in ("train", "valid", "test")}
    return fit_matrices(data, matrices, columns, params, seed, rounds)


def fit_matrices(data: Data, matrices: dict, names: list[str], params: dict, seed: int,
                 rounds: int = 5000) -> tuple[lgb.Booster, dict]:
    """LightGBM на готовых матрицах (например, компонентах PCA/PLS) с метками и группами из data."""
    columns = names
    train = lgb.Dataset(matrices["train"], data.parts["train"]["y"], feature_name=columns, free_raw_data=True)
    valid = lgb.Dataset(matrices["valid"], data.parts["valid"]["y"], reference=train)
    started = time.perf_counter()
    booster = lgb.train(
        params | {"seed": seed},
        train,
        num_boost_round=rounds,
        valid_sets=[valid],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    test_pred = booster.predict(matrices["test"], num_iteration=booster.best_iteration)
    predict_seconds = time.perf_counter() - started
    valid_pred = booster.predict(matrices["valid"], num_iteration=booster.best_iteration)
    info = {
        "test_pred": test_pred,
        "valid": score(data, "valid", valid_pred),
        "test": score(data, "test", test_pred),
        "best_iteration": booster.best_iteration,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_features": len(columns),
    }
    return booster, info


