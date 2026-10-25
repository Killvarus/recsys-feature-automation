"""
Метрики: качество вероятностей и качество ранжирования выданных треков.

LogLoss, ROC-AUC — по всем примерам.
GAUC — AUC внутри пользователя, среднее с весом по числу примеров; пользователи
       только с одним классом не участвуют. Убирает выигрыш от «какой пользователь
       вообще чаще дослушивает» и оставляет персональное упорядочивание треков.
NDCG@10 — внутри сессии (пользователь, сессия) треки упорядочиваются по прогнозу,
       релевантность — дослушан или нет. Берутся сессии с обоими классами: в остальных
       любой порядок даёт одно и то же значение. Равные прогнозы — в случайном порядке.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score


def _group_auc(frame: pl.DataFrame, group: list[str]) -> float:
    # AUC через ранги внутри группы (формула Манна — Уитни), средние ранги при равенстве.
    ranked = frame.with_columns(pl.col("score").rank("average").over(group).alias("_rank"))
    stats = ranked.group_by(group).agg(
        pl.len().alias("n"),
        pl.col("label").sum().alias("pos"),
        pl.col("_rank").filter(pl.col("label") == 1).sum().alias("rank_sum"),
    ).filter((pl.col("pos") > 0) & (pl.col("pos") < pl.col("n")))
    neg = stats["n"] - stats["pos"]
    auc = (stats["rank_sum"] - stats["pos"] * (stats["pos"] + 1) / 2) / (stats["pos"] * neg)
    return float((auc * stats["n"]).sum() / stats["n"].sum())


def _ndcg(frame: pl.DataFrame, group: list[str], k: int) -> tuple[float, float]:
    usable = frame.with_columns(pl.col("label").sum().over(group).alias("_pos"), pl.len().over(group).alias("_n"))
    usable = usable.filter((pl.col("_pos") > 0) & (pl.col("_pos") < pl.col("_n")))
    discount = 1.0 / (pl.int_range(1, pl.len() + 1).over(group).cast(pl.Float64) + 1.0).log(2)
    # Равные прогнозы упорядочиваются случайно (фиксированный seed), а не по порядку строк:
    # иначе постоянный прогноз «ранжирует» треки по времени внутри сессии.
    tiebreak = np.random.default_rng(0).permutation(usable.height)
    usable = usable.with_columns(pl.Series("_tb", tiebreak))
    by_score = usable.sort([*group, "score", "_tb"], descending=[False] * len(group) + [True, False]).with_columns(_d=discount)
    dcg = by_score.filter(pl.int_range(1, pl.len() + 1).over(group) <= k).group_by(group).agg((pl.col("label") * pl.col("_d")).sum().alias("dcg"))
    ideal = usable.sort([*group, "label"], descending=[False] * len(group) + [True]).with_columns(_d=discount)
    idcg = ideal.filter(pl.int_range(1, pl.len() + 1).over(group) <= k).group_by(group).agg((pl.col("label") * pl.col("_d")).sum().alias("idcg"))
    joined = dcg.join(idcg, on=group)
    share = usable.select(group).unique().height / max(frame.select(group).unique().height, 1)
    return float((joined["dcg"] / joined["idcg"]).mean()), share


def evaluate(uid, session, label, score, k: int = 10, probabilities: bool = True) -> dict[str, float]:
    label = np.asarray(label).astype(np.int8)
    score = np.asarray(score, dtype=np.float64)
    frame = pl.DataFrame({"uid": np.asarray(uid), "session": np.asarray(session), "label": label, "score": score})
    ndcg, share = _ndcg(frame, ["uid", "session"], k)
    result = {
        "logloss": float(log_loss(label, np.clip(score, 1e-6, 1 - 1e-6))) if probabilities else float("nan"),
        "auc": float(roc_auc_score(label, score)),
        "gauc": _group_auc(frame, ["uid"]),
        f"ndcg@{k}": ndcg,
        "ndcg_sessions_share": share,
    }
    return result
