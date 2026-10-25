"""
Эталонный ППОП с честным переобучением модели.

Здесь он нужен только тестам: быстрый LinearPPOP (BSBS) сверяется с ним на маленьких данных.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

# =============================================================================
# TRUE BACKWARD FEATURE SELECTION
# =============================================================================
class TrueBackwardFeatureSelection:
    """
    ППОП: обратный пакетный отбор с переобучением модели.

    На каждой итерации для каждого оставшегося признака обучается модель без него.
    Score — качество такой модели (больше — лучше). Чем выше score, тем меньше
    признак нужен, поэтому выбрасывается пакет с наибольшими score.
    Ранг 1 получает признак, продержавшийся дольше всех.

    n_features_to_drop: целое — фиксированный пакет, дробь из (0, 1) — доля
    оставшихся признаков (см. linear_selection.packet_size).
    """

    def __init__(self, estimator, n_features_to_drop=100, cv=5,
                 scoring='neg_mean_squared_error', random_state=42):
        from .linear_selection import check_step

        check_step(n_features_to_drop)
        self.estimator = estimator
        self.n_features_to_drop = n_features_to_drop
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.ranking_ = None
        self.individual_scores_ = {}
        self.elapsed_seconds_ = None
        self._X_val = None
        self._y_val = None

    def fit(self, X, y, X_val=None, y_val=None):
        from .linear_selection import packet_size

        features = X.columns.tolist()
        n_features = len(features)
        current_features = features.copy()
        self.individual_scores_ = {}
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val и y_val нужно передавать вместе")
        self._X_val = X_val
        self._y_val = y_val
        eliminated = []

        print(f"Запуск ППОП, признаков: {n_features}")

        iteration = 0
        started = time.perf_counter()

        with tqdm(total=n_features, desc="Backward Elimination") as pbar:
            while current_features:
                iteration += 1
                feature_scores = self._train_models_sequential(X[current_features], y)
                # Без этих признаков модель лучше всего — они нужны меньше всего.
                least_needed = sorted(current_features, key=lambda f: (-feature_scores[f], str(f)))
                size = packet_size(len(current_features), self.n_features_to_drop)
                features_to_drop = least_needed[:size]
                for feature in features_to_drop:
                    self.individual_scores_[feature] = feature_scores[feature]
                eliminated.append(features_to_drop)
                dropped = set(features_to_drop)
                current_features = [f for f in current_features if f not in dropped]
                pbar.update(len(features_to_drop))

        feature_group_ranks = {}
        feature_individual_ranks = {}
        next_rank = 1
        for group, packet in enumerate(reversed(eliminated), 1):
            # Внутри пакета первым идёт признак, без которого модель хуже всего.
            most_needed = sorted(packet, key=lambda f: (self.individual_scores_[f], str(f)))
            for offset, feature in enumerate(most_needed):
                feature_group_ranks[feature] = group
                feature_individual_ranks[feature] = next_rank + offset
            next_rank += len(packet)

        self.ranking_ = {
            'group_rank': feature_group_ranks,
            'individual_rank': feature_individual_ranks,
            'scores': self.individual_scores_
        }
        self.elapsed_seconds_ = time.perf_counter() - started

        print(f"Завершено. Итераций: {iteration}. Время ППОП: {self.elapsed_seconds_:.1f} с")
        return self

    def _train_models_sequential(self, X, y):
        feature_scores = {}
        for feature in tqdm(X.columns.tolist(), desc="Оценка признаков", leave=False):
            feature_scores[feature] = self._evaluate_model(X.drop(columns=[feature]), y)
        return feature_scores

    def _evaluate_model(self, X, y):
        from sklearn.dummy import DummyRegressor
        from sklearn.metrics import mean_squared_error

        # Без последнего признака остаётся только среднее по train.
        model = clone(self.estimator) if X.shape[1] else DummyRegressor()
        if self._X_val is not None:
            model.fit(X, y)
            prediction = model.predict(self._X_val.loc[:, X.columns])
            return -float(mean_squared_error(self._y_val, prediction))
        if self.cv is not None:
            scores = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring, n_jobs=1)
            return float(np.mean(scores))
        model.fit(X, y)
        prediction = model.predict(X)
        return -float(mean_squared_error(y, prediction))
    
    def get_ranking_df(self):
        if self.ranking_ is None:
            raise ValueError("Сначала выполните fit()")
        
        features = list(self.individual_scores_.keys())
        
        data = {
            'Feature': features,
            'Group_Rank': [self.ranking_['group_rank'][f] for f in features],
            'Individual_Rank': [self.ranking_['individual_rank'][f] for f in features],
            'Score': [self.individual_scores_[f] for f in features]
        }
        
        df_ranking = pd.DataFrame(data)
        df_ranking = df_ranking.sort_values('Individual_Rank').reset_index(drop=True)
        
        return df_ranking
    
    def save_ranking(self, filename, output_dir="Results"):
        if self.ranking_ is None:
            raise ValueError("Сначала выполните fit()")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        df_ranking = self.get_ranking_df()
        file_path = output_path / f"{filename}.csv"
        
        df_ranking.to_csv(file_path, index=False)
        
        print(f"✅ Ранжирование сохранено: {file_path}")
        
        return file_path

