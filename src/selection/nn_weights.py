"""
Отбор по весам нейросети с одним скрытым слоем (ReLU).

Особенности реализации:
  - входы стандартизуются (z-score по train) до обучения, иначе веса разных признаков несравнимы;
  - берутся веса лучшей эпохи по validation, а не последней;
  - hidden и top_n — параметры;
  - результат — рейтинг всех признаков (ранг 1 — самый важный), а не срез из N;
  - функция потерь — BCE с логитами: цель бинарная;
  - важность считается матрично: |W2|·|W1| вместо тройного цикла (та же формула);
  - признак без разброса на train получает важность 0. После стандартизации его вход равен нулю,
    градиент по его весам нулевой, и |W1| остаётся случайным с инициализации — без этого формула
    ставила такие признаки в верх рейтинга (у нас prev__gap и u__listen__since_last, оба ≡ 1 с).

Важность признака i = Σ_k Σ_j |W2[k,j]|·|W1[j,i]|.
Дискретный вариант: runs запусков с разными seed; признак засчитывается в запуске, если попал в top_n.
Рейтинг — по числу попаданий (по убыванию), при равенстве — по средней важности. Признаки,
попавшие в top_n хотя бы threshold раз, отмечены столбцом Selected.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SingleHiddenLayer(nn.Module):
    def __init__(self, input_dim: int, hidden: int, output_dim: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, output_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def _constant(X: np.ndarray) -> np.ndarray:
    """Постоянные столбцы: max == min. Проверка std == 0 не годится: у 3·10⁵ одинаковых
    float-значений среднее отличается от них на ~1e-16, и std выходит ~1e-16, а не 0."""
    return X.max(axis=0) == X.min(axis=0)


def _standardize(X_train: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    constant = _constant(X_train)
    mean[constant] = X_train[0, constant]  # постоянный столбец превращается в нули
    std[constant | (std == 0.0)] = 1.0
    return [((X - mean) / std).astype(np.float32) for X in (X_train, *others)]


def nn_weight_importance(
    X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray,
    hidden: int = 32, batch_size: int = 1024, learning_rate: float = 1e-3, num_epochs: int = 100,
    patience: int = 5, seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Нормированная важность всех признаков и сведения об обучении (лучшая эпоха, лучшая ошибка)."""
    torch.manual_seed(seed)
    constant = _constant(X_train)
    X_train, X_valid = _standardize(X_train, X_valid)
    train = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.float32)[:, None])),
                       batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    X_val = torch.from_numpy(X_valid)
    y_val = torch.from_numpy(y_valid.astype(np.float32)[:, None])

    model = SingleHiddenLayer(X_train.shape[1], hidden)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_loss, best_epoch, best_state, waited = float("inf"), -1, copy.deepcopy(model.state_dict()), 0
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(X_val), y_val))
        if val_loss < best_loss:
            best_loss, best_epoch, best_state, waited = val_loss, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)

    w1 = model.fc1.weight.detach().numpy()  # hidden × input
    w2 = model.fc2.weight.detach().numpy()  # output × hidden
    importance = (np.abs(w2) @ np.abs(w1)).sum(axis=0)
    importance[constant] = 0.0
    if importance.sum() > 0:
        importance = importance / importance.sum()
    return importance, {"best_epoch": best_epoch + 1, "best_val_loss": best_loss, "epochs_run": epoch + 1}


def nn_weight_ranking(
    names: list[str], X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray,
    runs: int = 5, top_n: int = 30, threshold: int = 3, **kwargs,
) -> tuple[pd.DataFrame, list[dict]]:
    """Дискретный вариант по runs запускам. Возвращает рейтинг всех признаков и сведения о запусках."""
    importances, infos = [], []
    for run in range(runs):
        importance, info = nn_weight_importance(X_train, y_train, X_valid, y_valid, seed=run, **kwargs)
        importances.append(importance)
        infos.append(info)
    importances = np.array(importances)
    in_top = np.zeros(len(names), dtype=int)
    for importance in importances:
        in_top[np.argsort(-importance)[:top_n]] += 1
    mean_importance = importances.mean(axis=0)
    frame = pd.DataFrame({"Feature": names, "TopCount": in_top, "Score": mean_importance})
    frame = frame.sort_values(["TopCount", "Score"], ascending=[False, False]).reset_index(drop=True)
    frame["Individual_Rank"] = np.arange(1, len(frame) + 1)
    frame["Selected"] = frame["TopCount"] >= threshold
    return frame, infos
