"""
PLS (Partial Least Squares) для признаков.

Функционал:
- Применение PLS с заданным количеством компонент
- Анализ explained variance
- Визуализация
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class PLSTransformer:
    """PLS — самостоятельный регрессионный метод (не требует нейросети)."""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.pls: Optional[PLSRegression] = None
        self.n_components_used: Optional[int] = None
        self.explained_variance_ratio: Optional[np.ndarray] = None
        self.cumulative_variance: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler_X.fit_transform(X_imputed)

        if y.ndim == 1 or y.shape[1] == 1:
            y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1))
        else:
            y_scaled = self.scaler_y.fit_transform(y)

        n_components = min(self.n_components, X.shape[1], X.shape[0])
        self.n_components_used = n_components

        self.pls = PLSRegression(n_components=n_components, scale=False)
        self.pls.fit(X_scaled, y_scaled)

        # Explained variance in X-space (approximation via scores)
        x_scores = self.pls.x_scores_
        total_var = np.sum(np.var(X_scaled, axis=0))
        explained_var = np.array([np.var(x_scores[:, i]) for i in range(n_components)])
        self.explained_variance_ratio = explained_var / total_var
        self.cumulative_variance = np.cumsum(self.explained_variance_ratio)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """PLS-проекция (скрытые переменные)."""
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler_X.transform(X_imputed)
        X_pls = self.pls.transform(X_scaled)
        columns = [f'PLS{i+1}' for i in range(self.n_components_used)]
        return pd.DataFrame(X_pls, index=X.index, columns=columns)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Предсказание таргета в оригинальном масштабе."""
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler_X.transform(X_imputed)
        y_scaled = self.pls.predict(X_scaled)
        return self.scaler_y.inverse_transform(y_scaled)

    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    def get_info(self) -> Dict:
        return {
            'n_components': self.n_components_used,
            'total_variance_explained': float(self.cumulative_variance[-1]) if self.cumulative_variance is not None else None,
            'explained_variance_ratio': self.explained_variance_ratio.tolist() if self.explained_variance_ratio is not None else None,
            'cumulative_variance': self.cumulative_variance.tolist() if self.cumulative_variance is not None else None,
        }


def apply_pls_to_data(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    target_columns: List[str],
    n_components: int = 2,
    feature_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PLSTransformer]:
    """Применение PLS к train/valid/test."""
    feature_cols = feature_columns or [c for c in train.columns if c not in target_columns]

    X_train = train[feature_cols]
    X_valid = valid[feature_cols]
    X_test = test[feature_cols]

    y_train = train[target_columns]

    transformer = PLSTransformer(n_components=n_components)
    X_train_pls = transformer.fit_transform(X_train, y_train)
    X_valid_pls = transformer.transform(X_valid)
    X_test_pls = transformer.transform(X_test)

    train_pls = X_train_pls.copy()
    valid_pls = X_valid_pls.copy()
    test_pls = X_test_pls.copy()

    for col in target_columns:
        train_pls[col] = train[col].values
        valid_pls[col] = valid[col].values
        test_pls[col] = test[col].values

    return train_pls, valid_pls, test_pls, transformer


def plot_pls_comparison(
    results: List[Dict],
    metric: str = 'r2_mean',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """График сравнения качества при разном количестве PLS компонент."""
    df = pd.DataFrame(results)
    df = df.sort_values('n_components')

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df['n_components'], df[metric], 'g-o', markersize=8, linewidth=2, label=metric)

    std_col = metric.replace('_mean', '_std')
    if std_col in df.columns:
        ax.fill_between(df['n_components'], df[metric] - df[std_col], df[metric] + df[std_col], alpha=0.2, color='green')

    ax.set_xlabel('Number of PLS Components', fontsize=12)
    ax.set_ylabel(metric.replace('_mean', '').upper(), fontsize=12)
    ax.set_title(f'{metric.replace("_mean", "").upper()} vs Number of PLS Components', fontsize=13)
    ax.grid(True, alpha=0.3)

    if 'original_n_features' in df.columns:
        ax.axvline(x=df['original_n_features'].iloc[0], color='red', linestyle='--',
                   label=f"Original features ({int(df['original_n_features'].iloc[0])})")
        ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig