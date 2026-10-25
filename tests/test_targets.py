"""Момент решения: признаки не должны зависеть от того, сколько длилось целевое прослушивание."""
import numpy as np
import polars as pl

from src.data.yambda import DAY, SESSION_GAP, Split, decision_points, targets
from src.features.handcrafted import SPECS
from src.features.engine import compute, last_k, previous_event


def _events(seed=0, n_users=4, per_user=400):
    rng = np.random.default_rng(seed)
    rows = []
    for uid in range(n_users):
        t = 280 * DAY
        for _ in range(per_user):
            # Конец прослушивания: предыдущий конец + длительность (иногда перерыв между сессиями).
            t += int(rng.integers(1, 60)) * 5 + (int(rng.integers(1, 20)) * 3600 if rng.uniform() < 0.05 else 0)
            rows.append((uid, t, int(rng.integers(0, 30)), int(rng.integers(0, 2)), int(rng.integers(0, 101)), 200,
                         "listen" if rng.uniform() < 0.9 else "like"))
    frame = pl.DataFrame(rows, schema=["uid", "timestamp", "item_id", "is_organic", "played_ratio_pct",
                                       "track_length_seconds", "event_type"], orient="row")
    frame = frame.with_columns(
        pl.col("uid").cast(pl.UInt32), pl.col("timestamp").cast(pl.UInt32), pl.col("item_id").cast(pl.UInt32),
        pl.col("is_organic").cast(pl.UInt8), pl.col("played_ratio_pct").cast(pl.UInt16),
        (pl.col("item_id") // 5).cast(pl.Int32).alias("artist_id"), (pl.col("item_id") // 2).cast(pl.Int32).alias("album_id"),
    ).with_row_index("event_idx").sort("uid", "timestamp", "event_idx")
    gap = pl.col("timestamp").diff().over("uid").fill_null(SESSION_GAP + 1) > SESSION_GAP
    return frame.with_columns(session=gap.cast(pl.UInt32).cum_sum().over("uid"))


SPLIT = Split(train=(280, 281), valid=(281, 282), test=(282, 400))


def _features(events, query_ids):
    queries = targets(events, SPLIT).rename({"event_idx": "query_id"}).filter(pl.col("query_id").is_in(query_ids))
    parts = [compute(queries, events, spec) for spec in SPECS if spec.events != "any"]
    parts += [last_k(queries, events, (5,), ("listened",)), previous_event(queries, events)]
    frame = queries.select("query_id", "timestamp")
    for part in parts:
        frame = frame.join(part, on="query_id", how="left")
    return frame.sort("query_id")


def test_decision_time_is_end_of_previous_listen():
    events = _events()
    points = decision_points(events)
    listens = events.filter(pl.col("event_type") == "listen").select("uid", "timestamp").unique().sort("uid", "timestamp")
    expected = listens.with_columns((pl.col("timestamp").shift(1).over("uid").cast(pl.Int64) + 1).alias("expected"))
    joined = points.join(expected, on=["uid", "timestamp"])
    assert joined.filter(pl.col("decision_ts") != pl.col("expected")).height == 0
    assert joined.filter(pl.col("decision_ts") >= pl.col("timestamp").cast(pl.Int64)).height == 0


def test_features_do_not_depend_on_duration_of_target_listen():
    events = _events(seed=1)
    queries = targets(events, SPLIT)
    rng = np.random.default_rng(2)
    nxt = events.with_columns(pl.col("timestamp").shift(-1).over("uid").alias("_next"))
    candidates = nxt.join(queries.select("event_idx"), on="event_idx").filter(
        pl.col("_next").is_not_null() & (pl.col("_next") - pl.col("timestamp") >= 20))
    # По одному прослушиванию: удлиняем его (конец сдвигается позже, но раньше следующего события)
    # и проверяем, что его собственные признаки не изменились. Сдвигать несколько сразу нельзя:
    # сдвинутый трек попадает в прошлое другого и законно меняет его признаки.
    for row in candidates.sample(15, seed=3).iter_rows(named=True):
        shift = int(rng.integers(1, (row["_next"] - row["timestamp"]) // 5)) * 5
        longer = events.with_columns(
            pl.when(pl.col("event_idx") == row["event_idx"]).then(pl.col("timestamp") + shift)
            .otherwise(pl.col("timestamp")).alias("timestamp")
        ).sort("uid", "timestamp", "event_idx")
        a, b = _features(events, [row["event_idx"]]), _features(longer, [row["event_idx"]])
        assert a.height == 1
        assert a.equals(b), row["event_idx"]
