"""
Важность признаков в градиентном бустинге — только как источник рейтинга.
Проверка рейтинга делается той же итоговой моделью (LightGBM на первых k), что и у остальных методов.

(а) Встроенная важность: GradientBoostingClassifier.feature_importances_ (суммарное уменьшение
    неоднородности в разбиениях). У HistGradientBoosting в sklearn такого атрибута нет.
    Обучается на train; GradientBoosting однопоточный, поэтому на подвыборке.
(б) Permutation importance: HistGradientBoostingClassifier обучается на ранней части train,
    признаки перемешиваются на поздней части train (по времени). Validation не используется:
    она нужна для выбора k, и рейтинг, построенный на ней, завысил бы метрики на ней же.
    Важность — рост LogLoss при перемешивании признака (среднее по n_repeats).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance


def _ranking(names: list[str], scores: np.ndarray, extra: dict | None = None) -> pd.DataFrame:
    frame = pd.DataFrame({"Feature": names, "Score": scores, **(extra or {})})
    frame = frame.sort_values(["Score", "Feature"], ascending=[False, True]).reset_index(drop=True)
    frame["Individual_Rank"] = np.arange(1, len(frame) + 1)
    return frame


def builtin_importance_ranking(names: list[str], X: np.ndarray, y: np.ndarray, n_estimators: int = 150,
                               max_depth: int = 4, learning_rate: float = 0.1, subsample: float = 0.8,
                               seed: int = 0) -> pd.DataFrame:
    """X без пропусков (GradientBoosting их не принимает) — после LinearPreprocessor."""
    model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                                       subsample=subsample, random_state=seed)
    model.fit(X, y)
    return _ranking(names, model.feature_importances_)


def permutation_ranking(names: list[str], X_fit: np.ndarray, y_fit: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray,
                        n_repeats: int = 3, seed: int = 0) -> pd.DataFrame:
    """X может содержать NaN (HistGradientBoosting их поддерживает)."""
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, early_stopping=True,
                                           validation_fraction=0.1, random_state=seed)
    model.fit(X_fit, y_fit)
    result = permutation_importance(model, X_eval, y_eval, scoring="neg_log_loss", n_repeats=n_repeats,
                                    random_state=seed, n_jobs=1)
    return _ranking(names, result.importances_mean, {"Std": result.importances_std})
