"""
PCA (метод главных компонент) для признаков

Функционал:
- Применение PCA с заданным количеством компонент
- Анализ explained variance
- Визуализация вклада компонент
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import json


# =============================================================================
# PCA ТРАНСФОРМАЦИЯ
# =============================================================================
class PCATransformer:
    """
    Класс для применения PCA к данным.
    
    Автоматически масштабирует данные перед PCA.
    Обрабатывает пропущенные значения (NaN).
    """
    
    def __init__(self, n_components: int = None, variance_ratio: float = None):
        """
        Args:
            n_components: количество компонент (если указано)
            variance_ratio: доля дисперсии для сохранения (0.0-1.0)
                           Если указано, n_components игнорируется
        """
        self.n_components = n_components
        self.variance_ratio = variance_ratio
        
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.pca = None
        self.n_components_used = None
        self.explained_variance_ratio = None
        self.cumulative_variance = None
        
    def fit(self, X: pd.DataFrame):
        """
        Обучение PCA на данных.
        
        Args:
            X: DataFrame с признаками
        """
        # Обработка пропущенных значений (NaN)
        X_imputed = self.imputer.fit_transform(X)

        # Масштабирование
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        # Определяем количество компонент
        if self.variance_ratio is not None:
            # Сначала PCA со всеми компонентами
            pca_full = PCA()
            pca_full.fit(X_scaled)
            
            # Находим количество компонент для заданной дисперсии
            cumsum = np.cumsum(pca_full.explained_variance_ratio_)
            self.n_components_used = np.argmax(cumsum >= self.variance_ratio) + 1
            self.n_components_used = min(self.n_components_used, X.shape[1])
        else:
            self.n_components_used = min(self.n_components, X.shape[1]) if self.n_components else X.shape[1]
        
        # PCA с нужным количеством компонент
        self.pca = PCA(n_components=self.n_components_used)
        self.pca.fit(X_scaled)
        
        # Сохраняем статистику
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        self.cumulative_variance = np.cumsum(self.explained_variance_ratio)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Трансформация данных.
        
        Returns:
            DataFrame с PCA компонентами
        """
        # Обработка пропущенных значений (NaN)
        X_imputed = self.imputer.transform(X)

        X_scaled = self.scaler.transform(X_imputed)
        X_pca = self.pca.transform(X_scaled)
        
        # Создаём DataFrame с понятными именами колонок
        columns = [f'PC{i+1}' for i in range(self.n_components_used)]
        return pd.DataFrame(X_pca, index=X.index, columns=columns)
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Обучение и трансформация за один шаг."""
        self.fit(X)
        return self.transform(X)
    
    def get_info(self) -> Dict:
        """Информация о PCA трансформации."""
        return {
            'n_components': self.n_components_used,
            'total_variance_explained': float(self.cumulative_variance[-1]) if self.cumulative_variance is not None else None,
            'explained_variance_ratio': self.explained_variance_ratio.tolist() if self.explained_variance_ratio is not None else None,
            'cumulative_variance': self.cumulative_variance.tolist() if self.cumulative_variance is not None else None
        }


def apply_pca_to_data(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    target_columns: List[str],
    n_components: int = None,
    variance_ratio: float = None,
    feature_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PCATransformer]:
    """
    Применение PCA к train/valid/test данным.
    
    Args:
        train, valid, test: исходные DataFrame
        target_columns: список целевых переменных (таргет для обучения)
        n_components: количество компонент
        variance_ratio: доля дисперсии для сохранения
        feature_columns: признаки; по умолчанию все столбцы, кроме target_columns
    
    Returns:
        (train_pca, valid_pca, test_pca, transformer)
    """
    feature_cols = feature_columns or [c for c in train.columns if c not in target_columns]
    
    X_train = train[feature_cols]
    X_valid = valid[feature_cols]
    X_test = test[feature_cols]
    
    # PCA
    transformer = PCATransformer(n_components=n_components, variance_ratio=variance_ratio)
    X_train_pca = transformer.fit_transform(X_train)
    X_valid_pca = transformer.transform(X_valid)
    X_test_pca = transformer.transform(X_test)
    
    # Добавляем целевые переменные
    train_pca = X_train_pca.copy()
    valid_pca = X_valid_pca.copy()
    test_pca = X_test_pca.copy()
    
    for col in target_columns:
        train_pca[col] = train[col].values
        valid_pca[col] = valid[col].values
        test_pca[col] = test[col].values
    
    return train_pca, valid_pca, test_pca, transformer


# =============================================================================
# АНАЛИЗ PCA
# =============================================================================
def analyze_pca_variance(transformer: PCATransformer) -> pd.DataFrame:
    """
    Анализ explained variance PCA.
    
    Returns:
        DataFrame с информацией о каждой компоненте
    """
    info = transformer.get_info()
    
    df = pd.DataFrame({
        'component': range(1, transformer.n_components_used + 1),
        'explained_variance': info['explained_variance_ratio'],
        'cumulative_variance': info['cumulative_variance']
    })
    
    return df


def find_optimal_n_components(
    transformer: PCATransformer,
    thresholds: List[float] = [0.80, 0.85, 0.90, 0.95, 0.99]
) -> Dict[float, int]:
    """
    Найти оптимальное количество компонент для разных порогов дисперсии.
    
    Returns:
        dict: {threshold: n_components}
    """
    cumsum = transformer.cumulative_variance
    result = {}
    
    for threshold in thresholds:
        n_comp = np.argmax(cumsum >= threshold) + 1
        result[threshold] = int(n_comp)
    
    return result


# =============================================================================
# ВИЗУАЛИЗАЦИЯ
# =============================================================================
def plot_explained_variance(
    transformer: PCATransformer,
    n_components_to_show: int = 50,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    График explained variance.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    n_show = min(n_components_to_show, transformer.n_components_used)
    
    # Individual variance
    axes[0].bar(
        range(1, n_show + 1),
        transformer.explained_variance_ratio[:n_show],
        color='steelblue',
        alpha=0.7
    )
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('Individual Explained Variance')
    axes[0].grid(True, alpha=0.3)
    
    # Cumulative variance
    axes[1].plot(
        range(1, n_show + 1),
        transformer.cumulative_variance[:n_show],
        'b-o',
        markersize=4,
        linewidth=2
    )
    
    # Добавляем горизонтальные линии для порогов
    for threshold in [0.80, 0.90, 0.95, 0.99]:
        n_comp = np.argmax(transformer.cumulative_variance >= threshold) + 1
        if n_comp <= n_show:
            axes[1].axhline(y=threshold, color='red', linestyle='--', alpha=0.5)
            axes[1].axvline(x=n_comp, color='red', linestyle='--', alpha=0.5)
            axes[1].text(n_comp, threshold, f' {threshold:.0%}', fontsize=9)
    
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Explained Variance')
    axes[1].set_title('Cumulative Explained Variance')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_pca_comparison(
    results: List[Dict],
    metric: str = 'r2_mean',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    График сравнения качества модели при разном количестве PCA компонент.
    
    Args:
        results: список результатов с n_components и metric
    """
    df = pd.DataFrame(results)
    df = df.sort_values('n_components')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Основной график
    ax.plot(
        df['n_components'],
        df[metric],
        'b-o',
        markersize=8,
        linewidth=2,
        label=metric
    )
    
    # Error bars если есть std
    std_col = metric.replace('_mean', '_std')
    if std_col in df.columns:
        ax.fill_between(
            df['n_components'],
            df[metric] - df[std_col],
            df[metric] + df[std_col],
            alpha=0.2,
            color='blue'
        )
    
    ax.set_xlabel('Number of PCA Components', fontsize=12)
    ax.set_ylabel(metric.replace('_mean', '').upper(), fontsize=12)
    ax.set_title(f'{metric.replace("_mean", "").upper()} vs Number of PCA Components', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    # Добавляем вертикальную линию для исходного количества признаков
    if 'original_n_features' in df.columns:
        ax.axvline(x=df['original_n_features'].iloc[0], color='red', linestyle='--', 
                   label=f"Original features ({int(df['original_n_features'].iloc[0])})")
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig
