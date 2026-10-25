"""
Ручной набор признаков для задачи «дослушает ли пользователь трек, поставленный рекомендациями».

Группы (префикс имени признака → группа):
  u     — пользователь: активность и доля дослушанного в окнах, лайки;
  i, a, al — трек, артист, альбом: популярность и отклик всех пользователей выборки;
  ui    — прямая связь пользователь × трек;
  ua, ual — косвенная связь через артиста и альбом;
  us, u__last, prev — контекст: текущая сессия, последние k прослушиваний, предыдущий трек;
  ctx   — время суток, день недели, длина трека.
"""
from __future__ import annotations

import gc
import time
from pathlib import Path

import polars as pl

from .engine import DAY, HOUR, Agg, compute, last_k, previous_event

U, I, A, AL = ("uid",), ("item_id",), ("artist_id",), ("album_id",)
UI, UA, UAL, US = ("uid", "item_id"), ("uid", "artist_id"), ("uid", "album_id"), ("uid", "session")
W = 7 * DAY
M = 30 * DAY

SPECS: list[Agg] = [
    # Пользователь
    Agg(U, "listen", (HOUR, DAY, W, M, None), ("listened", "ratio", "algo"), last=True, first=True),
    Agg(U, "listen_algo", (DAY, W, M, None), ("listened", "ratio")),
    Agg(U, "listen_org", (W, None), ("listened",)),
    Agg(U, "like", (W, M, None)),
    Agg(U, "dislike", (M, None)),
    Agg(U, "unlike", (None,)),
    # Трек, артист, альбом
    Agg(I, "listen", (DAY, W, M, None), ("listened", "ratio", "algo"), first=True),
    Agg(I, "listen_algo", (W, None), ("listened",)),
    Agg(I, "like", (W, None)),
    Agg(I, "dislike", (None,)),
    Agg(A, "listen", (W, M, None), ("listened", "algo"), first=True),
    Agg(A, "like", (None,)),
    Agg(AL, "listen", (M, None), ("listened",)),
    # Прямая связь
    Agg(UI, "listen", (DAY, W, M, None), ("listened", "ratio", "algo"), last=True, first=True),
    Agg(UI, "like", (None,), last=True),
    Agg(UI, "unlike", (None,), last=True),
    Agg(UI, "dislike", (None,)),
    Agg(UI, "undislike", (None,)),
    # Косвенные связи
    Agg(UA, "listen", (DAY, W, M, None), ("listened", "ratio"), last=True, first=True),
    Agg(UA, "listen_algo", (M, None), ("listened",)),
    Agg(UA, "like", (None,)),
    Agg(UA, "dislike", (None,)),
    Agg(UAL, "listen", (M, None), ("listened",), last=True),
    # Текущая сессия до t
    Agg(US, "listen", (None,), ("listened", "algo"), first=True),
]

LAST_K = (5, 20, 100)
LAST_K_VALUES = ("listened", "ratio", "algo")


def _derived(columns: list[str] | None = None) -> list[pl.Expr]:
    def ratio(a: str, b: str, name: str) -> pl.Expr:
        return pl.when(pl.col(b) > 0).then(pl.col(a) / pl.col(b)).otherwise(None).alias(name)

    return [
        (pl.col("ui__like__cnt__all") - pl.col("ui__unlike__cnt__all")).alias("ui__net_like"),
        (pl.col("ui__dislike__cnt__all") - pl.col("ui__undislike__cnt__all")).alias("ui__net_dislike"),
        ratio("u__like__cnt__all", "u__listen__cnt__all", "u__like_rate__all"),
        ratio("i__like__cnt__all", "i__listen__cnt__all", "i__like_rate__all"),
        ratio("i__dislike__cnt__all", "i__listen__cnt__all", "i__dislike_rate__all"),
        ratio("a__like__cnt__all", "a__listen__cnt__all", "a__like_rate__all"),
        ratio("ua__like__cnt__all", "ua__listen__cnt__all", "ua__like_rate__all"),
        ratio("ua__listen__cnt__30d", "u__listen__cnt__30d", "ua__share_of_user__30d"),
        ratio("ua__listen__cnt__all", "u__listen__cnt__all", "ua__share_of_user__all"),
        ratio("ui__listen__cnt__all", "u__listen__cnt__all", "ui__share_of_user__all"),
        ratio("i__listen__cnt__1d", "i__listen__cnt__7d", "i__trend_1d_7d"),
        (pl.col("timestamp") % DAY // HOUR).cast(pl.Float64).alias("ctx__hour"),
        (pl.col("timestamp") // DAY % 7).cast(pl.Float64).alias("ctx__dow"),
        pl.col("track_length_seconds").cast(pl.Float64).alias("ctx__track_length"),
    ]


def _float32(part: pl.DataFrame) -> pl.DataFrame:
    return part.with_columns([pl.col(c).cast(pl.Float32) for c in part.columns if c != "query_id"])


def _sharded(func, queries: pl.DataFrame, events: pl.DataFrame, n_shards: int) -> pl.DataFrame:
    """
    Признаки, у которых в ключе есть пользователь, не зависят от других пользователей:
    считаем по частям пользователей, чтобы не держать в памяти состояния всех 46 млн событий.
    """
    parts = []
    for shard in range(n_shards):
        mask = pl.col("uid") % n_shards == shard
        parts.append(_float32(func(queries.filter(mask), events.filter(mask))))
        gc.collect()
    return pl.concat(parts)


def build(queries: pl.DataFrame, events: pl.DataFrame, parts_dir: Path, specs: list[Agg] = SPECS,
          n_shards: int = 4, verbose: bool = True, last_k_ks: tuple[int, ...] = LAST_K,
          last_k_values: tuple[str, ...] = LAST_K_VALUES, derived=None, keep_sums: bool = True) -> pl.DataFrame:
    """
    queries — целевые события (targets) со столбцом query_id. Возвращает queries + признаки (Float32).
    Каждая группа признаков сразу пишется в parts_dir, сборка — в конце, когда события уже не нужны.
    derived — функция, возвращающая выражения производных признаков (по умолчанию ручные отношения);
    keep_sums=False отбрасывает суммы (__sum_): при наличии счётчика и среднего они избыточны.
    """
    parts_dir.mkdir(parents=True, exist_ok=True)
    derived = derived or _derived
    jobs = [(spec.prefix, "uid" in spec.key, lambda q, e, spec=spec: compute(q, e, spec)) for spec in specs]
    jobs += [("u__last_k", True, lambda q, e: last_k(q, e, last_k_ks, last_k_values)),
             ("prev", True, previous_event)]
    paths = []
    for name, by_user, func in jobs:
        path = parts_dir / f"{name}.parquet"
        paths.append(path)
        if path.exists():  # уже посчитано в прошлом запуске (те же queries)
            continue
        started = time.perf_counter()
        part = _sharded(func, queries, events, n_shards) if by_user else _float32(func(queries, events))
        if not keep_sums:
            part = part.select([c for c in part.columns if "__sum_" not in c])
        part.write_parquet(path)
        if verbose:
            print(f"  {name:<18} {part.width - 1:>3} features  {time.perf_counter() - started:6.1f}s", flush=True)
        del part
        gc.collect()
    del events
    gc.collect()
    frame = queries
    for path in paths:
        frame = frame.join(pl.read_parquet(path), on="query_id", how="left")
    frame = frame.with_columns(derived(frame.columns))
    meta = set(queries.columns)
    return frame.with_columns([pl.col(c).cast(pl.Float32) for c in frame.columns if c not in meta])


def feature_group(name: str) -> str:
    """Группа признака для абляций."""
    head = name.split("__")[0]
    if head in ("us", "prev") or name.startswith("u__last"):
        return "context"
    return {"u": "user", "i": "item", "a": "artist_album", "al": "artist_album", "ui": "user_item",
            "ua": "user_artist_album", "ual": "user_artist_album", "ctx": "context"}.get(head, "other")
