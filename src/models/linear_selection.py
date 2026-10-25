"""
BSBS (Batch Sequential Backward Selection, последовательный обратный пакетный отбор; в коде — LinearPPOP) и RFE для линейной регрессии: МНК (alpha=0) и Ridge (alpha>0).

Признаки стандартизуются по обучающей части, y центрируется. Так модель
совпадает с Pipeline(StandardScaler(), Ridge(alpha)), а при alpha=0 — с LinearRegression.
Вместо стандартизации можно взять другую нормировку по каналам (scaling, см. scale_channels).

Пусть A = XᵀX + alpha·I на текущем наборе признаков и β = A⁻¹Xᵀy. Решение той же
задачи без признака j: β₋ⱼ = β − A⁻¹[:, j]·βⱼ / A⁻¹[j, j]. Это точная формула, не
приближение, поэтому N моделей на N−1 признаке считаются из одного обращения A.

ППОП выбрасывает пакет признаков, без которых MSE на оценочной выборке выросла
меньше всего (или упала). RFE выбрасывает пакет с наименьшими |β| на train.
В обоих методах ранг 1 получает признак, продержавшийся дольше всех.

Пакет задаётся как в sklearn.feature_selection.RFE: целое — число признаков,
дробь из (0, 1) — доля от оставшихся (округление вверх, не меньше одного).

alpha="loo": штраф подбирается заново на каждой итерации (в RFE — при каждом
переобращении матрицы) по leave-one-out ошибке Ridge на всём train, как в RidgeCV.
Оптимальный штраф для 100 признаков другой, чем для 5427.
alpha="loo1se": то же, но берётся самый большой штраф, чья LOO-ошибка не больше
минимальной плюс одна её стандартная ошибка (правило одной SE, как lambda.1se
в glmnet). При тысяче скоррелированных каналов минимум LOO даёт почти МНК,
и решения о выбрасывании становятся шумом; правило 1SE держит штраф выше.
alpha="loo_schedule": правило 1SE, пока признаков больше switch_at, и минимум LOO,
когда их switch_at или меньше. Минимум лучше выбирает первые сотни каналов,
1SE не даёт отбору развалиться на тысячах скоррелированных каналов.
"""
from __future__ import annotations

import math
import time
from numbers import Integral, Real
from pathlib import Path
from typing import Hashable

import numpy as np
import pandas as pd
from scipy.linalg import lapack
from scipy.linalg.blas import dger
from sklearn.model_selection import KFold


def check_step(step) -> None:
    if isinstance(step, bool):
        raise ValueError("step должен быть целым >= 1 или дробью из (0, 1)")
    if isinstance(step, Integral):
        if step < 1:
            raise ValueError("целый step должен быть >= 1")
        return
    if isinstance(step, Real) and 0.0 < float(step) < 1.0:
        return
    raise ValueError("step должен быть целым >= 1 или дробью из (0, 1)")


def packet_size(n_remaining: int, step, min_step: int = 1) -> int:
    """Сколько признаков выбросить, когда их осталось n_remaining."""
    check_step(step)
    if isinstance(step, Integral):
        size = int(step)
    else:
        size = math.ceil(float(step) * n_remaining)
    return min(n_remaining, max(size, int(min_step), 1))


LOO_ALPHAS = np.logspace(-3, 6, 46)
LOO_MODES = ("loo", "loo1se", "loo_schedule")


def _check_alpha(alpha) -> float | str:
    if isinstance(alpha, str):
        if alpha not in LOO_MODES:
            raise ValueError('alpha должен быть числом >= 0, "loo", "loo1se" или "loo_schedule"')
        return alpha
    alpha = float(alpha)
    if not alpha >= 0.0:
        raise ValueError("alpha должен быть >= 0")
    return alpha


def ridge_spectrum(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    U, s² и V из SVD X = U·S·Vᵀ через собственное разложение меньшей из XᵀX и XXᵀ.
    Нулевые сингулярные числа отброшены. V возвращается только при k ≤ n.
    """
    n, k = X.shape
    if k <= n:
        s2, v = np.linalg.eigh(X.T @ X)
    else:
        s2, u = np.linalg.eigh(X @ X.T)
    s2 = np.clip(s2, 0.0, None)
    keep = s2 > s2.max() * 1e-12
    s2 = s2[keep]
    if k <= n:
        v = v[:, keep]
        return (X @ v) / np.sqrt(s2), s2, v
    return u[:, keep], s2, None


def loo_alpha(
    u: np.ndarray, s2: np.ndarray, target: np.ndarray, alphas: np.ndarray = LOO_ALPHAS, rule: str = "min"
) -> float:
    """
    alpha по leave-one-out ошибке Ridge (центрированный y, свободный член).
    rule="min" — наименьшая ошибка, rule="1se" — самый большой alpha с ошибкой
    не больше минимума плюс стандартная ошибка этого минимума.

    При d = s² / (s² + alpha): остаток e = y − U·diag(d)·Uᵀy, рычаг hᵢ = Σⱼ Uᵢⱼ²dⱼ + 1/n,
    ошибка без i-й строки eᵢ / (1 − hᵢ). Части вне столбцов U не зависят от alpha
    и считаются один раз: при малом alpha вычитание 1 − hᵢ напрямую теряет точность.
    """
    n = len(target)
    projected = u.T @ target
    u_squared = u**2
    outside = target - u @ projected
    outside_leverage = 1.0 - 1.0 / n - u_squared.sum(axis=1)
    if rule not in ("min", "1se"):
        raise ValueError('rule должен быть "min" или "1se"')
    means, errors = [], []
    for alpha in alphas:
        shrink = alpha / (s2 + alpha)
        residual = u @ (shrink * projected) + outside
        one_minus_leverage = u_squared @ shrink + outside_leverage
        squared = (residual / one_minus_leverage) ** 2
        means.append(float(squared.mean()))
        errors.append(float(squared.std(ddof=1) / np.sqrt(n)))
    best = int(np.argmin(means))
    if rule == "min":
        return float(alphas[best])
    limit = means[best] + errors[best]
    return float(max(alpha for alpha, mean in zip(alphas, means) if mean <= limit))


def _check_switch_at(switch_at) -> int:
    if isinstance(switch_at, bool) or not isinstance(switch_at, Integral) or switch_at < 0:
        raise ValueError("switch_at должен быть целым >= 0")
    return int(switch_at)


def _loo_rule(alpha: str, n_features: int, switch_at: int) -> str:
    """Правило выбора alpha для набора из n_features признаков."""
    if alpha == "loo_schedule":
        return "1se" if n_features > switch_at else "min"
    return "1se" if alpha == "loo1se" else "min"


def _as_target(y, n_rows: int) -> np.ndarray:
    target = np.asarray(y, dtype=float).reshape(-1)
    if len(target) != n_rows:
        raise ValueError("Длины X и y не совпадают")
    return target


SCALINGS = ("zscore", "pareto", "robust", "center")


def scale_channels(X_fit: np.ndarray, *others: np.ndarray, method: str = "zscore") -> list[np.ndarray]:
    """
    Нормировка по каналам: сдвиг и масштаб каждого столбца считаются по X_fit.
      zscore — минус среднее, деление на std (ddof=0), как StandardScaler;
      pareto — минус среднее, деление на корень из std;
      robust — минус среднее, деление на межквартильный размах (сдвиг для Ridge
               со свободным членом не важен, а быстрые формулы требуют
               центрирования средним);
      center — только минус среднее, масштаб исходный.
    Нулевой масштаб заменяется на 1.
    """
    shift, scale = X_fit.mean(axis=0), _channel_scale(X_fit, method)
    return [(X - shift) / scale for X in (X_fit, *others)]


def scale_only(X_fit: np.ndarray, *others: np.ndarray, method: str = "zscore") -> list[np.ndarray]:
    """
    Только деление каждого канала на его масштаб по X_fit, без вычитания среднего
    (масштабы те же, что в scale_channels). Нужна, когда после неё каналы
    нормируются по строкам: при центрировании сумма и максимум строки теряют смысл.
    """
    scale = _channel_scale(X_fit, method)
    return [X / scale for X in (X_fit, *others)]


def _channel_scale(X_fit: np.ndarray, method: str) -> np.ndarray:
    if method == "zscore":
        scale = X_fit.std(axis=0)
    elif method == "pareto":
        scale = np.sqrt(X_fit.std(axis=0))
    elif method == "robust":
        q25, q75 = np.percentile(X_fit, [25, 75], axis=0)
        scale = q75 - q25
    elif method == "center":
        scale = np.ones(X_fit.shape[1])
    else:
        raise ValueError(f"scaling должен быть одним из {SCALINGS}")
    return np.where(scale == 0.0, 1.0, scale)


def _standardize(X_fit: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """Как StandardScaler: среднее и std (ddof=0) по X_fit, у константы масштаб 1."""
    return scale_channels(X_fit, *others, method="zscore")


def _check_scaling(scaling: str) -> str:
    if scaling not in SCALINGS:
        raise ValueError(f"scaling должен быть одним из {SCALINGS}")
    return scaling


def _inverse(gram: np.ndarray, alpha: float, n_rows: int) -> np.ndarray:
    """(XᵀX + alpha·I)⁻¹ через Холецкого. При alpha=0 нужна невырожденная XᵀX."""
    p = gram.shape[0]
    if alpha == 0.0 and p >= n_rows:
        raise ValueError(
            f"МНК без штрафа не определён: признаков {p}, строк {n_rows}. "
            "После центрирования ранг XᵀX не больше n−1, решений бесконечно много. "
            "Нужен alpha > 0 или меньше признаков."
        )
    matrix = np.array(gram, dtype=float, order="F", copy=True)
    matrix[np.diag_indices(p)] += alpha
    factor, info = lapack.dpotrf(matrix, lower=0, overwrite_a=1)
    if info != 0:
        raise ValueError(
            f"XᵀX + alpha·I вырождена (dpotrf info={info}). "
            "Признаки линейно зависимы, нужен alpha > 0."
        )
    inverse, info = lapack.dpotri(factor, lower=0, overwrite_c=1)
    if info != 0:
        raise ValueError(f"не удалось обратить матрицу (dpotri info={info})")
    upper = np.triu(inverse)
    return upper + np.triu(inverse, 1).T


def _ordered(positions: np.ndarray, values: np.ndarray, names: list, descending: bool) -> list[int]:
    """Позиции по значению, при равенстве — по имени признака."""
    sign = -1.0 if descending else 1.0
    return sorted(range(len(positions)), key=lambda i: (sign * values[i], str(names[positions[i]])))


def _ranks(names: list, packets: list[list[int]], scores: dict[int, float]) -> pd.DataFrame:
    """Последний выброшенный пакет — группа 1, внутри пакета по убыванию score."""
    rows = []
    next_rank = 1
    for group, packet in enumerate(reversed(packets), 1):
        best_first = sorted(packet, key=lambda i: (-scores[i], str(names[i])))
        for offset, i in enumerate(best_first):
            rows.append(
                {
                    "Feature": names[i],
                    "Group_Rank": group,
                    "Individual_Rank": next_rank + offset,
                    "Score": scores[i],
                }
            )
        next_rank += len(packet)
    return pd.DataFrame(rows)


class _SelectorBase:
    ranking_: pd.DataFrame | None
    history_: pd.DataFrame | None
    elapsed_seconds_: float | None

    def ranking_frame(self) -> pd.DataFrame:
        if self.ranking_ is None:
            raise RuntimeError("Сначала вызовите fit()")
        return self.ranking_.copy()

    def save_ranking(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.ranking_frame().to_csv(path, index=False)
        return path


class _Split:
    """Одна пара (обучение, оценка), масштабированная по обучающей части."""

    def __init__(
        self, X_fit: np.ndarray, y_fit: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray, scaling: str = "zscore"
    ):
        X_fit, X_eval = scale_channels(X_fit, X_eval, method=scaling)
        y_mean = y_fit.mean()
        self.n_fit = len(y_fit)
        self.n_eval = len(y_eval)
        self.gram = X_fit.T @ X_fit
        self.xty = X_fit.T @ (y_fit - y_mean)
        self.X_eval = X_eval
        self.y_eval = y_eval - y_mean

    def sse_without_each(self, columns: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
        """SSE на оценочной части для модели на columns и для каждой модели без одного признака."""
        inverse = _inverse(self.gram[np.ix_(columns, columns)], alpha, self.n_fit)
        beta = inverse @ self.xty[columns]
        X_eval = self.X_eval[:, columns]
        residual = self.y_eval - X_eval @ beta
        shift = X_eval @ inverse
        coef = beta / np.diag(inverse)
        base = float(residual @ residual)
        # Остаток без признака j: residual + shift[:, j]·coef[j].
        sse = base + 2.0 * coef * (residual @ shift) + coef**2 * np.einsum("ij,ij->j", shift, shift)
        return sse, base


class LinearPPOP(_SelectorBase):
    """
    ППОП на МНК или Ridge.

    Оценочная выборка: если в fit переданы X_eval, y_eval — MSE считается на них
    (например, на validation). Иначе — K-fold на переданном train, cv фолдов.
    Score признака — на сколько выросла MSE без него в момент выбрасывания.

    reshuffle=True: на каждой итерации свои случайные фолды. Иначе тысячи решений
    принимаются по одному и тому же разбиению, и отбор подстраивается под его шум.
    """

    def __init__(
        self,
        alpha: float | str = 0.0,
        step=0.1,
        min_step: int = 1,
        cv: int = 5,
        random_state: int = 0,
        reshuffle: bool = False,
        switch_at: int = 500,
        scaling: str = "zscore",
    ):
        check_step(step)
        if cv < 2:
            raise ValueError("cv должен быть >= 2")
        self.alpha = _check_alpha(alpha)
        self.switch_at = _check_switch_at(switch_at)
        self.step = step
        self.min_step = int(min_step)
        self.cv = int(cv)
        self.random_state = random_state
        self.reshuffle = bool(reshuffle)
        self.scaling = _check_scaling(scaling)
        self.ranking_ = None
        self.history_ = None
        self.elapsed_seconds_ = None

    def _splits(self, X: np.ndarray, y: np.ndarray, X_eval, y_eval, seed=None) -> list[_Split]:
        if X_eval is not None:
            evaluation = np.asarray(X_eval, dtype=float)
            return [_Split(X, y, evaluation, _as_target(y_eval, len(X_eval)), self.scaling)]
        seed = self.random_state if seed is None else seed
        folds = KFold(n_splits=self.cv, shuffle=True, random_state=seed)
        return [_Split(X[fit], y[fit], X[held], y[held], self.scaling) for fit, held in folds.split(X)]

    def fit(self, X: pd.DataFrame, y, X_eval: pd.DataFrame | None = None, y_eval=None) -> "LinearPPOP":
        if (X_eval is None) != (y_eval is None):
            raise ValueError("X_eval и y_eval нужно передавать вместе")
        if X_eval is not None and list(X_eval.columns) != list(X.columns):
            raise ValueError("Колонки train и оценочной выборки должны совпадать и идти в одном порядке")
        names: list[Hashable] = list(X.columns)
        started = time.perf_counter()
        values = X.to_numpy(dtype=float)
        target = _as_target(y, len(X))
        reshuffle = self.reshuffle and X_eval is None
        splits = None if reshuffle else self._splits(values, target, X_eval, y_eval)
        n_eval = len(target) if reshuffle else sum(split.n_eval for split in splits)
        if isinstance(self.alpha, str):
            (scaled,) = scale_channels(values, method=self.scaling)
            centered = target - target.mean()

        remaining = np.arange(len(names))
        packets: list[list[int]] = []
        scores: dict[int, float] = {}
        history = []
        while remaining.size:
            step_started = time.perf_counter()
            if isinstance(self.alpha, str):
                u, s2, _ = ridge_spectrum(scaled[:, remaining])
                alpha = loo_alpha(u, s2, centered, rule=_loo_rule(self.alpha, remaining.size, self.switch_at))
            else:
                alpha = self.alpha
            sse = np.zeros(remaining.size)
            base = 0.0
            if reshuffle:
                # Новые фолды строятся только по оставшимся признакам.
                seed = None if self.random_state is None else self.random_state + len(history)
                current = self._splits(values[:, remaining], target, None, None, seed)
                columns = np.arange(remaining.size)
            else:
                current, columns = splits, remaining
            for split in current:
                part, part_base = split.sse_without_each(columns, alpha)
                sse += part
                base += part_base
            mse_full = base / n_eval
            increase = sse / n_eval - mse_full
            size = packet_size(remaining.size, self.step, self.min_step)
            # Без них MSE выросла меньше всего — они нужны меньше всего.
            drop = _ordered(remaining, increase, names, descending=False)[:size]
            for i in drop:
                scores[int(remaining[i])] = float(increase[i])
            packets.append([int(remaining[i]) for i in drop])
            history.append(
                {
                    "n_features": int(remaining.size),
                    "alpha": alpha,
                    "mse": mse_full,
                    "dropped": size,
                    "seconds": time.perf_counter() - step_started,
                }
            )
            print(
                f"PPOP {len(history)}: features {remaining.size}, alpha {alpha:g}, mse {mse_full:.6g}, "
                f"dropped {size}, step_s {history[-1]['seconds']:.1f}",
                flush=True,
            )
            remaining = np.delete(remaining, drop)

        self.ranking_ = _ranks(names, packets, scores)
        self.history_ = pd.DataFrame(history)
        self.elapsed_seconds_ = time.perf_counter() - started
        return self


class LinearRFE(_SelectorBase):
    """
    RFE на МНК или Ridge. Важность — |β| после масштабирования train (scaling).

    step=1 — классический RFE. Обратная матрица после выбрасывания обновляется
    формулой понижения ранга, раз в refresh_every выброшенных признаков
    пересчитывается заново, чтобы не копилась ошибка округления.
    При alpha="loo_schedule" матрица пересчитывается и в момент, когда признаков
    осталось switch_at или меньше: с этого шага alpha берётся по минимуму LOO.
    """

    def __init__(
        self,
        alpha: float | str = 700.0,
        step=1,
        min_step: int = 1,
        refresh_every: int = 256,
        switch_at: int = 500,
        scaling: str = "zscore",
    ):
        check_step(step)
        if refresh_every < 1:
            raise ValueError("refresh_every должен быть >= 1")
        self.scaling = _check_scaling(scaling)
        self.alpha = _check_alpha(alpha)
        self.switch_at = _check_switch_at(switch_at)
        self.step = step
        self.min_step = int(min_step)
        self.refresh_every = int(refresh_every)
        self.ranking_ = None
        self.history_ = None
        self.elapsed_seconds_ = None

    def fit(self, X: pd.DataFrame, y) -> "LinearRFE":
        names: list[Hashable] = list(X.columns)
        started = time.perf_counter()
        (scaled,) = scale_channels(X.to_numpy(dtype=float), method=self.scaling)
        target = _as_target(y, len(X))
        gram = scaled.T @ scaled
        xty = scaled.T @ (target - target.mean())
        n_rows = len(target)

        def rule_for(n_features: int) -> str | None:
            if not isinstance(self.alpha, str):
                return None
            return _loo_rule(self.alpha, n_features, self.switch_at)

        def choose_alpha(columns: np.ndarray) -> float:
            if not isinstance(self.alpha, str):
                return self.alpha
            u, s2, _ = ridge_spectrum(scaled[:, columns])
            return loo_alpha(u, s2, target - target.mean(), rule=rule_for(columns.size))

        # block — признаки, по которым построена inverse; active — ещё не выброшенные из block.
        block = np.arange(len(names))
        alpha = choose_alpha(block)
        rule = rule_for(block.size)
        inverse = _inverse(gram, alpha, n_rows)
        active = np.ones(block.size, dtype=bool)
        since_refresh = 0
        packets: list[list[int]] = []
        scores: dict[int, float] = {}
        history = []
        while active.any():
            step_started = time.perf_counter()
            # У выброшенных строки и столбцы inverse обнулены, их β тоже 0.
            used_alpha = alpha
            beta = inverse @ xty[block]
            positions = np.flatnonzero(active)
            weights = np.abs(beta[positions])
            size = packet_size(positions.size, self.step, self.min_step)
            drop = [positions[i] for i in _ordered(block[positions], weights, names, descending=False)[:size]]
            for j in drop:
                scores[int(block[j])] = float(abs(beta[j]))
            packets.append([int(block[j]) for j in drop])
            active[drop] = False
            since_refresh += size
            if active.any():
                switched = rule_for(int(active.sum())) != rule
                if since_refresh >= self.refresh_every or switched:
                    block = block[active]
                    active = np.ones(block.size, dtype=bool)
                    alpha = choose_alpha(block)
                    rule = rule_for(block.size)
                    inverse = _inverse(gram[np.ix_(block, block)], alpha, n_rows)
                    since_refresh = 0
                else:
                    inverse = self._downdate(inverse, drop)
            history.append(
                {
                    "n_features": int(positions.size),
                    "alpha": used_alpha,
                    "dropped": size,
                    "smallest_abs_coef": float(weights.min()),
                    "seconds": time.perf_counter() - step_started,
                }
            )
            if len(history) % 250 == 1 or not active.any() or size > 1:
                print(
                    f"RFE {len(history)}: features {positions.size}, alpha {used_alpha:g}, dropped {size}, "
                    f"smallest_abs_coef {weights.min():.6g}, step_s {history[-1]['seconds']:.2f}",
                    flush=True,
                )

        self.ranking_ = _ranks(names, packets, scores)
        self.history_ = pd.DataFrame(history)
        self.elapsed_seconds_ = time.perf_counter() - started
        return self

    @staticmethod
    def _downdate(inverse: np.ndarray, drop: list[int]) -> np.ndarray:
        """A⁻¹ без признаков drop: A⁻¹ − A⁻¹[:, D]·(A⁻¹[D, D])⁻¹·A⁻¹[D, :]. Строки D становятся нулевыми."""
        if len(drop) == 1:
            j = drop[0]
            column = inverse[:, j].copy()
            # inverse симметрична, транспонированный вид F-непрерывен — dger правит её на месте.
            updated = dger(-1.0 / column[j], column, column, a=inverse.T, overwrite_a=1)
            return inverse if np.shares_memory(updated, inverse) else np.ascontiguousarray(updated.T)
        columns = inverse[:, drop]
        return inverse - columns @ np.linalg.solve(inverse[np.ix_(drop, drop)], columns.T)
