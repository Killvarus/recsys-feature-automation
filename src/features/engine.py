"""
Признаки на момент события (point-in-time) через накопленные суммы и as-of join.

Признак задаётся спецификацией Agg: ключ (user, item, user×artist, …), фильтр событий,
окна и столбцы-значения. Для события запроса в момент t учитываются только события
с timestamp < t: события из того же 5-секундного интервала, включая само целевое,
не видны. Окно w — полуинтервал [t − w, t).

Как считается. Для каждой группы ключа события сортируются по времени и считаются
накопленные count и суммы значений. Строки с одинаковым (ключ, timestamp) схлопываются
в последнюю, поэтому накопленное значение в строке включает все события этого момента.
Запрос делает as-of join (последняя строка с timestamp ≤ t − 1) — это сумма до t;
второй join на t − w − 1 — сумма до начала окна. Разность — агрегат по окну.
Стоимость — сортировка и join, без циклов по строкам и без материализации пар.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

HOUR = 3_600
DAY = 86_400

FILTERS: dict[str, pl.Expr] = {
    "listen": pl.col("event_type") == "listen",
    "listen_algo": (pl.col("event_type") == "listen") & (pl.col("is_organic") == 0),
    "listen_org": (pl.col("event_type") == "listen") & (pl.col("is_organic") == 1),
    "like": pl.col("event_type") == "like",
    "unlike": pl.col("event_type") == "unlike",
    "dislike": pl.col("event_type") == "dislike",
    "undislike": pl.col("event_type") == "undislike",
    "any": pl.lit(True),
}

# Значения, которые можно суммировать по событиям (только для прослушиваний).
# Все значения целые: накопленные суммы считаются в Int64 точно и не зависят от других групп.
# Проценты переводятся в долю делением при выводе (SCALE).
VALUES: dict[str, pl.Expr] = {
    "listened": (pl.col("played_ratio_pct") >= 50).cast(pl.Int64),
    "ratio": pl.col("played_ratio_pct").clip(0, 100).cast(pl.Int64),
    "algo": (pl.col("is_organic") == 0).cast(pl.Int64),
}

SCALE = {"ratio": 0.01}

KEY_NAMES = {
    ("uid",): "u",
    ("item_id",): "i",
    ("artist_id",): "a",
    ("album_id",): "al",
    ("uid", "item_id"): "ui",
    ("uid", "artist_id"): "ua",
    ("uid", "album_id"): "ual",
    ("uid", "session"): "us",
}


def window_name(window: int | None) -> str:
    if window is None:
        return "all"
    if window % DAY == 0:
        return f"{window // DAY}d"
    if window % HOUR == 0:
        return f"{window // HOUR}h"
    return f"{window}s"


@dataclass(frozen=True)
class Agg:
    """
    key      — столбцы ключа;
    events   — имя фильтра из FILTERS;
    windows  — окна в секундах, None — вся история;
    values   — имена из VALUES: для каждого окна даётся сумма и среднее на событие;
    last     — время с последнего события ключа и значения в нём;
    first    — время с первого события ключа (возраст связи).
    """

    key: tuple[str, ...]
    events: str
    windows: tuple[int | None, ...] = (None,)
    values: tuple[str, ...] = ()
    last: bool = False
    first: bool = False
    name: str = field(default="", compare=False)

    @property
    def prefix(self) -> str:
        return self.name or f"{KEY_NAMES.get(self.key, '_'.join(self.key))}__{self.events}"


def _boundaries(key: list[str]) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """
    Для таблицы, отсортированной по (key, timestamp): начало группы ключа,
    начало момента (ключ, timestamp) и последняя строка момента.
    Без group_by и .over(): на миллионе групп они требуют гигабайты памяти.
    """
    new_group = pl.any_horizontal([pl.col(c) != pl.col(c).shift(1) for c in key]).fill_null(True)
    new_moment = (new_group | (pl.col("timestamp") != pl.col("timestamp").shift(1))).fill_null(True)
    next_group = pl.any_horizontal([pl.col(c) != pl.col(c).shift(-1) for c in key]).fill_null(True)
    last_in_moment = (next_group | (pl.col("timestamp") != pl.col("timestamp").shift(-1))).fill_null(True)
    return new_group, new_moment, last_in_moment


def _within_group_cumsum(column: str, new_group: pl.Expr) -> pl.Expr:
    """Накопленная сумма внутри группы = общая накопленная сумма минус её значение перед группой."""
    total = pl.col(column).cum_sum()
    before = pl.when(new_group).then(total - pl.col(column)).otherwise(None).forward_fill()
    return total - before


def _state_table(events: pl.DataFrame, spec: Agg) -> pl.DataFrame:
    """Накопленные суммы по группе ключа, одна строка на (ключ, timestamp)."""
    key = list(spec.key)
    new_group, new_moment, last_in_moment = _boundaries(key)
    table = (
        events.filter(FILTERS[spec.events])
        .select(*key, "timestamp", "event_idx", *[VALUES[v].alias(v) for v in spec.values])
        .sort(*key, "timestamp", "event_idx")
        .with_columns(pl.lit(1, pl.Int64).alias("_one"), new_group.alias("_new_group"), new_moment.alias("_new_moment"),
                      last_in_moment.alias("_keep"))
        .with_columns(
            _within_group_cumsum("_one", pl.col("_new_group")).cast(pl.UInt32).alias("_cnt"),
            *[_within_group_cumsum(v, pl.col("_new_group")).alias(f"_cum_{v}") for v in spec.values],
        )
    )
    extra = []
    if spec.last:
        # Значение «последнего события» при нескольких событиях в один момент — их среднее:
        # разность накопленных сумм на конце момента и перед его началом.
        def before_moment(column: str) -> pl.Expr:
            return pl.when(pl.col("_new_moment")).then(pl.col(column) - pl.col(f"_m_{column}")).otherwise(None).forward_fill()

        table = table.with_columns(pl.col("_one").alias("_m__cnt"), *[pl.col(v).alias(f"_m__cum_{v}") for v in spec.values])
        n_moment = pl.col("_cnt").cast(pl.Float64) - before_moment("_cnt")
        extra = [((pl.col(f"_cum_{v}") - before_moment(f"_cum_{v}")).cast(pl.Float64) / n_moment).alias(f"_last_{v}") for v in spec.values]
    if spec.first:
        extra.append(pl.when(pl.col("_new_group")).then(pl.col("timestamp")).otherwise(None).forward_fill().alias("_first_ts"))
    keep = [*key, "timestamp", "_cnt", *[f"_cum_{v}" for v in spec.values]]
    state = table.with_columns(extra).filter(pl.col("_keep")).select(*keep, *[e.meta.output_name() for e in extra])
    return state.with_columns(pl.col("timestamp").alias("_state_ts")).sort("timestamp")


def _as_of(queries: pl.DataFrame, state: pl.DataFrame, key: list[str], shift: int) -> pl.DataFrame:
    """Строка состояния с последним timestamp ≤ t − 1 − shift для каждого запроса."""
    left = queries.with_columns((pl.col("timestamp").cast(pl.Int64) - 1 - shift).alias("_probe")).sort("_probe")
    right = state.with_columns(pl.col("timestamp").cast(pl.Int64).alias("_probe")).drop("timestamp")
    return left.join_asof(right, on="_probe", by=key, strategy="backward", check_sortedness=False).drop("_probe")


def compute(queries: pl.DataFrame, events: pl.DataFrame, spec: Agg) -> pl.DataFrame:
    """
    queries: столбцы query_id, timestamp и столбцы ключа.
    Возвращает query_id и признаки спецификации (null, если событий не было).
    """
    key = list(spec.key)
    state = _state_table(events, spec)
    base = queries.select("query_id", "timestamp", *key)
    p = spec.prefix
    end = _as_of(base, state, key, 0)
    out = [pl.col("query_id")]
    for window in spec.windows:
        w = window_name(window)
        if window is None:
            cnt = pl.col("_cnt").fill_null(0)
            sums = {v: pl.col(f"_cum_{v}").fill_null(0) for v in spec.values}
        else:
            start = _as_of(base, state.select(*key, "timestamp", "_cnt", *[f"_cum_{v}" for v in spec.values]), key, window)
            start = start.select("query_id", *[pl.col(c).alias(f"{c}@{w}") for c in ["_cnt", *[f"_cum_{v}" for v in spec.values]]])
            end = end.join(start, on="query_id", how="left")
            cnt = pl.col("_cnt").fill_null(0) - pl.col(f"_cnt@{w}").fill_null(0)
            sums = {v: pl.col(f"_cum_{v}").fill_null(0) - pl.col(f"_cum_{v}@{w}").fill_null(0) for v in spec.values}
        out.append(cnt.cast(pl.Float64).alias(f"{p}__cnt__{w}"))
        for v, total in sums.items():
            total = total.cast(pl.Float64) * SCALE.get(v, 1.0)
            out.append(total.alias(f"{p}__sum_{v}__{w}"))
            out.append(pl.when(cnt > 0).then(total / cnt).otherwise(None).alias(f"{p}__mean_{v}__{w}"))
    if spec.last:
        since = (pl.col("timestamp").cast(pl.Int64) - pl.col("_state_ts").cast(pl.Int64)).cast(pl.Float64)
        out.append(since.alias(f"{p}__since_last"))
        out += [(pl.col(f"_last_{v}") * SCALE.get(v, 1.0)).alias(f"{p}__last_{v}") for v in spec.values]
    if spec.first:
        since_first = (pl.col("timestamp").cast(pl.Int64) - pl.col("_first_ts").cast(pl.Int64)).cast(pl.Float64)
        out.append(pl.when(pl.col("_state_ts").is_not_null()).then(since_first).otherwise(None).alias(f"{p}__since_first"))
    return end.select(out)


def last_k(queries: pl.DataFrame, events: pl.DataFrame, ks: tuple[int, ...], values: tuple[str, ...],
           events_filter: str = "listen") -> pl.DataFrame:
    """
    Средние значения по последним k событиям пользователя (до t, не по времени, а по числу).
    Нужен номер события r в истории пользователя: агрегат по последним k —
    разность накопленных сумм в строках r и r − k.
    """
    new_group, _, last_in_moment = _boundaries(["uid"])
    rows = (
        events.filter(FILTERS[events_filter])
        .select("uid", "timestamp", "event_idx", *[VALUES[v].alias(v) for v in values])
        .sort("uid", "timestamp", "event_idx")
        .with_columns(pl.lit(1, pl.Int64).alias("_one"), new_group.alias("_new_group"), last_in_moment.alias("_keep"))
        .with_columns(
            _within_group_cumsum("_one", pl.col("_new_group")).cast(pl.UInt32).alias("_r"),
            *[_within_group_cumsum(v, pl.col("_new_group")).alias(f"_cum_{v}") for v in values],
        )
    )
    cum_cols = [f"_cum_{v}" for v in values]
    state = rows.filter(pl.col("_keep")).select("uid", "timestamp", "_r", *cum_cols).sort("timestamp")
    end = _as_of(queries.select("query_id", "timestamp", "uid"), state, ["uid"], 0)
    by_row = rows.select("uid", "_r", *cum_cols)
    out = [pl.col("query_id")]
    for k in ks:
        lookup = by_row.select("uid", (pl.col("_r") + k).alias("_r"), *[pl.col(c).alias(f"{c}@{k}") for c in cum_cols])
        end = end.join(lookup, on=["uid", "_r"], how="left")
        n = pl.min_horizontal(pl.col("_r"), pl.lit(k)).cast(pl.Float64)
        for v in values:
            total = (pl.col(f"_cum_{v}") - pl.col(f"_cum_{v}@{k}").fill_null(0)).cast(pl.Float64) * SCALE.get(v, 1.0)
            out.append(pl.when(pl.col("_r") > 0).then(total / n).otherwise(None).alias(f"u__last{k}_{events_filter}__mean_{v}"))
    return end.select(out)


def previous_event(queries: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """
    Последнее прослушивание пользователя до t: тот же трек или артист, отклик, органика.
    prev__gap — время от него до t; если t — момент решения (конец предыдущего трека), он почти всегда ≈ 1 с.
    """
    state = (
        events.filter(FILTERS["listen"])
        .select("uid", "timestamp", "event_idx", "item_id", "artist_id", "played_ratio_pct", "is_organic")
        .sort("uid", "timestamp", "event_idx")
        .filter(_boundaries(["uid"])[2])
        .rename({"item_id": "_item", "artist_id": "_artist", "played_ratio_pct": "_ratio", "is_organic": "_organic"})
        .with_columns(pl.col("timestamp").alias("_state_ts"))
        .sort("timestamp")
        .drop("event_idx")
    )
    joined = _as_of(queries.select("query_id", "timestamp", "uid", "item_id", "artist_id"), state, ["uid"], 0)
    return joined.select(
        "query_id",
        (pl.col("timestamp").cast(pl.Int64) - pl.col("_state_ts").cast(pl.Int64)).cast(pl.Float64).alias("prev__gap"),
        (pl.col("_item") == pl.col("item_id")).cast(pl.Float64).alias("prev__same_item"),
        (pl.col("_artist") == pl.col("artist_id")).cast(pl.Float64).alias("prev__same_artist"),
        (pl.col("_ratio").cast(pl.Float64) / 100).alias("prev__ratio"),
        pl.col("_organic").cast(pl.Float64).alias("prev__organic"),
    )
