"""
История пользователя для модели над последовательностью событий.

Все события (прослушивания, лайки, дизлайки и их отмены) лежат в общих массивах, отсортированных по
(пользователь, время). Для примера с моментом решения t история — последние L событий пользователя
с timestamp < t (тот же момент решения, что у признаков бустинга: конец предыдущего прослушивания).
Окно находится двоичным поиском по ключу uid·2³² + timestamp, без циклов по примерам; признаки событий
собираются для пакета на лету.

Словари треков и артистов строятся только по событиям до начала valid (vocab_cutoff): редкие и
новые объекты получают общий индекс 1, 0 — заполнитель.
"""
from __future__ import annotations

import numpy as np
import polars as pl

TYPES = {"listen": 1, "like": 2, "dislike": 3, "unlike": 4, "undislike": 5}
N_TYPES = 6
N_DT = 26  # корзины log2(1 + давность в секундах)


def _vocab(values: pl.Series, early: pl.Series, min_count: int) -> dict:
    counts = pl.DataFrame({"v": values, "early": early}).filter(pl.col("early")).group_by("v").len()
    kept = counts.filter(pl.col("len") >= min_count)["v"].sort()
    return dict(zip(kept.to_list(), range(2, len(kept) + 2)))


class History:
    def __init__(self, events: pl.DataFrame, vocab_cutoff: int, min_item_count: int = 5, min_artist_count: int = 3):
        early = events["timestamp"] < vocab_cutoff
        self.item_vocab = _vocab(events["item_id"], early, min_item_count)
        self.artist_vocab = _vocab(events["artist_id"], early, min_artist_count)
        self.n_items = len(self.item_vocab) + 2
        self.n_artists = len(self.artist_vocab) + 2
        frame = events.select(
            (pl.col("uid").cast(pl.Int64) * (1 << 32) + pl.col("timestamp").cast(pl.Int64)).alias("key"),
            pl.col("timestamp").cast(pl.Int64),
            self.map_items(pl.col("item_id")).alias("item"),
            self.map_artists(pl.col("artist_id")).alias("artist"),
            pl.col("event_type").cast(pl.String).replace_strict(TYPES, return_dtype=pl.Int8).alias("type"),
            pl.when(pl.col("event_type") == "listen").then(pl.col("played_ratio_pct").clip(0, 100)).otherwise(0)
            .cast(pl.UInt8).alias("ratio"),
            pl.col("is_organic").cast(pl.UInt8).alias("organic"),
        )
        # load_events уже сортирует по (uid, timestamp, event_idx); проверяем, т.к. двоичный поиск на это опирается.
        assert frame["key"].is_sorted(), "события должны быть отсортированы по (uid, timestamp)"
        self.key = frame["key"].to_numpy()
        self.ts = frame["timestamp"].to_numpy()
        self.item = frame["item"].to_numpy().astype(np.int32)
        self.artist = frame["artist"].to_numpy().astype(np.int32)
        self.type = frame["type"].to_numpy().astype(np.int64)
        self.ratio = frame["ratio"].to_numpy()
        self.organic = frame["organic"].to_numpy()

    def map_items(self, expr: pl.Expr) -> pl.Expr:
        return expr.replace_strict(self.item_vocab, default=1, return_dtype=pl.Int32)

    def map_artists(self, expr: pl.Expr) -> pl.Expr:
        return expr.replace_strict(self.artist_vocab, default=1, return_dtype=pl.Int32)

    def window(self, uid: np.ndarray, t: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
        """Индексы последних length событий пользователя с timestamp < t и маска настоящих (не заполнитель)."""
        uid = uid.astype(np.int64)
        end = np.searchsorted(self.key, uid * (1 << 32) + t.astype(np.int64), side="left")
        start = np.searchsorted(self.key, uid * (1 << 32), side="left")
        idx = end[:, None] - length + np.arange(length)[None, :]
        mask = idx >= start[:, None]
        return np.where(mask, idx, 0), mask

    def batch(self, uid: np.ndarray, t: np.ndarray, length: int) -> dict[str, np.ndarray]:
        """Признаки событий истории: хронологически, самое свежее — последним, заполнитель — в начале."""
        idx, mask = self.window(uid, t, length)
        dt = np.maximum(t[:, None].astype(np.int64) - self.ts[idx], 0)
        return {
            "item": np.where(mask, self.item[idx], 0),
            "artist": np.where(mask, self.artist[idx], 0),
            "type": np.where(mask, self.type[idx], 0),
            "dt": np.where(mask, np.minimum(np.log2(1.0 + dt).astype(np.int64), N_DT - 1), 0),
            "ratio": np.where(mask, self.ratio[idx] / 100.0, 0.0).astype(np.float32),
            "organic": np.where(mask, self.organic[idx], 0).astype(np.float32),
            "mask": mask,
            "last_ts": np.where(mask, self.ts[idx], -1),
        }
