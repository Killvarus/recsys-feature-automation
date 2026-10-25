"""Движок признаков против наивного пересчёта и проверка, что будущее не просачивается."""
import numpy as np
import polars as pl
import pytest

from src.features.engine import DAY, FILTERS, SCALE, VALUES, Agg, compute, last_k, previous_event


def _events(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    # Мало пользователей и треков — много повторов; время кратно 5 с и часто совпадает.
    ts = rng.integers(0, 40 * DAY // 5, size=n) * 5
    ts[rng.uniform(size=n) < 0.3] = ts[0]
    item = rng.integers(0, 40, size=n)
    frame = pl.DataFrame(
        {
            "uid": rng.integers(0, 6, size=n).astype(np.uint32),
            "timestamp": ts.astype(np.uint32),
            "item_id": item.astype(np.uint32),
            "artist_id": (item // 4).astype(np.int64),
            "album_id": (item // 2).astype(np.int64),
            "is_organic": rng.integers(0, 2, size=n).astype(np.uint8),
            "played_ratio_pct": rng.integers(0, 130, size=n).astype(np.uint16),
            "event_type": rng.choice(["listen"] * 6 + ["like", "unlike", "dislike"], size=n),
        }
    ).with_row_index("event_idx")
    frame = frame.sort("uid", "timestamp", "event_idx")
    return frame.with_columns(session=(pl.col("timestamp").diff().over("uid").fill_null(10**9) > 1800).cast(pl.UInt32).cum_sum().over("uid"))


def _queries(events, n=150, seed=1):
    listens = events.filter(pl.col("event_type") == "listen")
    return listens.sample(n, seed=seed).rename({"event_idx": "query_id"})


def _naive(events, query, spec):
    t = query["timestamp"]
    past = events.filter(FILTERS[spec.events] & (pl.col("timestamp") < t))
    for column in spec.key:
        past = past.filter(pl.col(column) == query[column])
    past = past.with_columns([(VALUES[v] * SCALE.get(v, 1.0)).alias(v) for v in spec.values])
    result = {}
    for window in spec.windows:
        part = past if window is None else past.filter(pl.col("timestamp") >= t - window)
        w = "all" if window is None else f"{window // DAY}d" if window % DAY == 0 else f"{window // 3600}h"
        result[f"{spec.prefix}__cnt__{w}"] = float(part.height)
        for v in spec.values:
            total = float(part[v].sum()) if part.height else 0.0
            result[f"{spec.prefix}__sum_{v}__{w}"] = total
            result[f"{spec.prefix}__mean_{v}__{w}"] = total / part.height if part.height else None
    if spec.last:
        if past.height:
            last_ts = past["timestamp"].max()
            result[f"{spec.prefix}__since_last"] = float(t - last_ts)
            for v in spec.values:
                result[f"{spec.prefix}__last_{v}"] = float(past.filter(pl.col("timestamp") == last_ts)[v].mean())
        else:
            result[f"{spec.prefix}__since_last"] = None
            for v in spec.values:
                result[f"{spec.prefix}__last_{v}"] = None
    if spec.first:
        result[f"{spec.prefix}__since_first"] = float(t - past["timestamp"].min()) if past.height else None
    return result


SPECS = [
    Agg(("uid",), "listen", (3600, DAY, 7 * DAY, None), ("listened", "ratio"), last=True, first=True),
    Agg(("item_id",), "listen_algo", (7 * DAY, None), ("listened",)),
    Agg(("uid", "artist_id"), "listen", (DAY, None), ("listened", "algo"), last=True, first=True),
    Agg(("uid", "item_id"), "like", (None,), (), last=True),
    Agg(("uid", "session"), "listen", (None,), ("listened",)),
]


def _assert_close(actual, expected, label):
    if expected is None:
        assert actual is None, label
    else:
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-9), label


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.prefix)
def test_aggregates_match_naive(spec):
    events = _events()
    queries = _queries(events)
    fast = {row["query_id"]: row for row in compute(queries, events, spec).iter_rows(named=True)}
    assert len(fast) == queries.height
    for query in queries.iter_rows(named=True):
        for name, expected in _naive(events, query, spec).items():
            _assert_close(fast[query["query_id"]][name], expected, f"{name} q={query['query_id']}")


def test_last_k_and_previous_event_match_naive():
    events = _events(seed=2)
    queries = _queries(events, seed=3)
    fast = {r["query_id"]: r for r in last_k(queries, events, (1, 5), ("listened", "algo")).iter_rows(named=True)}
    prev = {r["query_id"]: r for r in previous_event(queries, events).iter_rows(named=True)}
    listens = events.filter(pl.col("event_type") == "listen").with_columns([(VALUES[v] * SCALE.get(v, 1.0)).alias(v) for v in ("listened", "algo")])
    for q in queries.iter_rows(named=True):
        past = listens.filter((pl.col("uid") == q["uid"]) & (pl.col("timestamp") < q["timestamp"])).sort("timestamp", "event_idx")
        for k in (1, 5):
            tail = past.tail(k)
            for v in ("listened", "algo"):
                expected = float(tail[v].mean()) if tail.height else None
                _assert_close(fast[q["query_id"]][f"u__last{k}_listen__mean_{v}"], expected, f"last{k} {v}")
        if past.height:
            last = past.row(-1, named=True)
            assert prev[q["query_id"]]["prev__gap"] == q["timestamp"] - last["timestamp"]
            assert prev[q["query_id"]]["prev__same_item"] == float(last["item_id"] == q["item_id"])
        else:
            assert prev[q["query_id"]]["prev__gap"] is None


def test_future_events_do_not_change_features():
    events = _events(seed=4)
    cutoff = int(events["timestamp"].quantile(0.6)) // 5 * 5
    queries = _queries(events, n=300, seed=5).filter(pl.col("timestamp") <= cutoff)
    rng = np.random.default_rng(9)
    future = pl.col("timestamp") >= cutoff
    n = events.height
    # Будущее (и сам момент cutoff) переписано случайно: другие треки, отклики, типы событий.
    scrambled = events.with_columns(
        pl.when(future).then(pl.Series(rng.integers(0, 40, n).astype(np.uint32))).otherwise(pl.col("item_id")).alias("item_id"),
        pl.when(future).then(pl.Series(rng.integers(0, 130, n).astype(np.uint16))).otherwise(pl.col("played_ratio_pct")).alias("played_ratio_pct"),
        pl.when(future).then(pl.lit("listen")).otherwise(pl.col("event_type")).alias("event_type"),
    ).with_columns((pl.col("item_id") // 4).cast(pl.Int64).alias("artist_id"))
    for spec in SPECS[:3]:
        a = compute(queries, events, spec).sort("query_id")
        b = compute(queries, scrambled, spec).sort("query_id")
        assert a.equals(b), spec.prefix
    a = last_k(queries, events, (5,), ("listened",)).sort("query_id")
    b = last_k(queries, scrambled, (5,), ("listened",)).sort("query_id")
    assert a.equals(b)
