import numpy as np
import pytest

from src.selection.nn_weights import nn_weight_importance, nn_weight_ranking
from src.selection.preprocess import LinearPreprocessor, column_rule


def _data(n=3000, p=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)) * rng.uniform(0.1, 100, size=p)  # разный масштаб признаков
    logit = 2.0 * X[:, 0] / X[:, 0].std() - 1.5 * X[:, 1] / X[:, 1].std()
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(float)
    return X[: n // 2], y[: n // 2], X[n // 2:], y[n // 2:]


def test_matrix_importance_equals_original_triple_loop():
    # Исходная формула: importance[i] = Σ_k Σ_j |W2[k,j]|·|W1[j,i]|.
    rng = np.random.default_rng(1)
    w1, w2 = rng.normal(size=(7, 5)), rng.normal(size=(1, 7))
    loop = np.array([sum(abs(w2[k, j]) * abs(w1[j, i]) for k in range(1) for j in range(7)) for i in range(5)])
    np.testing.assert_allclose((np.abs(w2) @ np.abs(w1)).sum(axis=0), loop)


def test_nn_ranking_finds_informative_features_despite_scale():
    X_train, y_train, X_valid, y_valid = _data()
    frame, infos = nn_weight_ranking([f"f{i}" for i in range(6)], X_train, y_train, X_valid, y_valid,
                                     runs=3, top_n=2, threshold=2, hidden=8, num_epochs=30, patience=3, learning_rate=1e-2)
    assert set(frame["Feature"].head(2)) == {"f0", "f1"}
    assert sorted(frame["Individual_Rank"]) == list(range(1, 7))
    assert all(1 <= info["best_epoch"] <= info["epochs_run"] for info in infos)


def test_best_epoch_weights_are_used():
    X_train, y_train, X_valid, y_valid = _data(seed=2)
    importance, info = nn_weight_importance(X_train, y_train, X_valid, y_valid, hidden=4, num_epochs=15, patience=15)
    assert info["epochs_run"] == 15
    assert importance.sum() == pytest.approx(1.0)


def test_preprocessor_fits_on_train_only_and_fills_since_with_max():
    names = ["u__listen__cnt__all", "ua__listen__since_last", "u__listen__mean_ratio__all", "ui__net_like"]
    assert [column_rule(n) for n in names] == ["log", "log", "raw", "signed_log"]
    train = np.array([[0, 10, 0.5, -2], [9, np.nan, np.nan, 3], [99, 1000, 0.7, 0]], dtype=float)
    other = np.array([[1e6, np.nan, np.nan, 5]], dtype=float)
    pre = LinearPreprocessor(names, clip_quantile=0.0).fit(train)
    out = pre.transform(other)
    assert out[0, 0] == pytest.approx(np.log1p(99))          # обрезано по максимуму train
    assert out[0, 1] == pytest.approx(np.log1p(1000))        # «не было события» = самое давнее на train
    assert out[0, 2] == pytest.approx(0.6)                   # среднее по train
    assert out[0, 3] == pytest.approx(np.log1p(3))           # обрезано по train


def test_constant_feature_gets_zero_importance():
    X_train, y_train, X_valid, y_valid = _data(seed=3)
    # log1p(1) на 300 тыс. строк: среднее ≠ значению в последнем знаке, std ~1e-16, а не 0.
    X_train = np.vstack([X_train] * 200)
    y_train = np.concatenate([y_train] * 200)
    X_train[:, 5] = np.log1p(1.0)
    X_valid[:, 5] = np.log1p(1.0)
    importance, _ = nn_weight_importance(X_train, y_train, X_valid, y_valid, hidden=4, num_epochs=1)
    assert importance[5] == 0.0
