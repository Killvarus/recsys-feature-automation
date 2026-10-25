"""История для модели над последовательностью: только события до момента решения."""
import numpy as np
import polars as pl

from src.data.yambda import targets
from src.seq.history import History
from test_targets import SPLIT, _events


def test_window_matches_naive_and_is_strictly_before_decision():
    events = _events(seed=4)
    history = History(events, vocab_cutoff=10**9, min_item_count=1, min_artist_count=1)
    queries = targets(events, SPLIT).sample(100, seed=0)
    uid = queries["uid"].to_numpy()
    t = queries["timestamp"].to_numpy()
    batch = history.batch(uid, t, length=16)
    for row in range(len(uid)):
        past = events.filter((pl.col("uid") == uid[row]) & (pl.col("timestamp").cast(pl.Int64) < t[row]))
        expected = past["timestamp"].to_numpy()[-16:]
        got = batch["last_ts"][row][batch["mask"][row]]
        np.testing.assert_array_equal(got, expected)
        assert (got < t[row]).all()


def test_lengthening_target_listen_does_not_change_history():
    events = _events(seed=5)
    queries = targets(events, SPLIT).sample(10, seed=1)
    for q in queries.iter_rows(named=True):
        longer = events.with_columns(
            pl.when(pl.col("event_idx") == q["event_idx"]).then(pl.col("timestamp") + 25)
            .otherwise(pl.col("timestamp")).alias("timestamp")
        ).sort("uid", "timestamp", "event_idx")
        a = History(events, 10**9, 1, 1).batch(np.array([q["uid"]]), np.array([q["timestamp"]]), 16)
        b = History(longer, 10**9, 1, 1).batch(np.array([q["uid"]]), np.array([q["timestamp"]]), 16)
        for key in ("item", "type", "ratio", "mask", "last_ts"):
            np.testing.assert_array_equal(a[key], b[key])
