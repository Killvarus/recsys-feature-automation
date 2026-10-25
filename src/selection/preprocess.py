"""
Предобработка признаков для методов, которым важен масштаб: линейный ППОП/RFE, веса нейросети, PCA, PLS.
LightGBM получает сырые признаки — деревьям предобработка не нужна.

Все параметры считаются только по train (fit), к valid и test применяется transform.

Правила по имени признака («log1p + заполнение»):
  счётчики и суммы (__cnt__, __sum_), время с события (since_), длина трека, пауза  → log1p(x);
  net_like / net_dislike (могут быть < 0)                                           → sign·log1p(|x|);
  доли и средние (mean_, last_, rate, share, trend), флаги, час, день               → как есть.
Пропуски:
  since_* (события не было)  → максимум по train после log1p («очень давно»);
  остальные (доля без событий) → среднее по train.
Затем обрезка по квантилям train [0.001, 0.999]. Стандартизацию (z-score) делает каждый метод сам.

Рейтинг признаков, полученный на этой предобработке, всегда публикуется вместе с её именем: PREPROCESSING_NAME.
"""
from __future__ import annotations

import numpy as np

PREPROCESSING_NAME = "log1p_counts+fill_since_max+fill_mean+clip_q001"

_LOG_MARKERS = ("__cnt__", "__sum_", "since_", "ctx__track_length", "prev__gap")
_SIGNED_LOG = ("__net_like", "__net_dislike")


def column_rule(name: str) -> str:
    if any(m in name for m in _SIGNED_LOG):
        return "signed_log"
    if any(m in name for m in _LOG_MARKERS):
        return "log"
    return "raw"


class LinearPreprocessor:
    def __init__(self, names: list[str], clip_quantile: float = 0.001):
        self.names = list(names)
        self.rules = [column_rule(n) for n in self.names]
        self.clip_quantile = clip_quantile

    def _transform_values(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64, copy=True)
        for j, rule in enumerate(self.rules):
            if rule == "log":
                X[:, j] = np.log1p(np.clip(X[:, j], 0.0, None))
            elif rule == "signed_log":
                X[:, j] = np.sign(X[:, j]) * np.log1p(np.abs(X[:, j]))
        return X

    def fit(self, X: np.ndarray) -> "LinearPreprocessor":
        values = self._transform_values(X)
        fill = np.nanmean(values, axis=0)
        since = np.array(["since_" in n for n in self.names])
        fill[since] = np.nanmax(values[:, since], axis=0)
        self.fill_ = np.nan_to_num(fill, nan=0.0)
        filled = np.where(np.isnan(values), self.fill_, values)
        self.low_, self.high_ = np.quantile(filled, [self.clip_quantile, 1 - self.clip_quantile], axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = self._transform_values(X)
        values = np.where(np.isnan(values), self.fill_, values)
        return np.clip(values, self.low_, self.high_)
