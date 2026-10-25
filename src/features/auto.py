"""
Автоматическая генерация признаков: перебор по грамматике вместо ручного выбора.

Признак = связь (ключ) × тип событий × окно × статистика. Человек задаёт только словарь:
  связи        — пользователь; трек, артист, альбом; пары пользователя с треком, артистом, альбомом;
                 текущая сессия;
  типы событий — прослушивание (все / алгоритмические / органические), лайк, дизлайк;
  окна         — для пользовательских связей 1 ч, 6 ч, 1 д, 3 д, 7 д, 14 д, 30 д, 90 д, всё время;
                 для трека, артиста, альбома (популярность) 1 д, 7 д, 30 д, 90 д, всё время;
  статистики   — число событий, доля дослушанных, средняя доля прослушанного, доля алгоритмических,
                 время с последнего и первого события, последние k прослушиваний (k = 3 … 200).
Генератор строит все сочетания и затем автоматические отношения:
  доля связи в активности пользователя: cnt(пара, w) / cnt(пользователь, w) для каждой пары и окна;
  тренд: интенсивность в коротком окне / в соседнем длинном (для пользователя и трека);
  доля лайков и дизлайков: cnt(лайк) / cnt(прослушивание) для каждой связи.
Ручные решения «какое окно» и «какую связь» заменяются полным перебором; что из этого нужно,
решает отбор признаков на train.
"""
from __future__ import annotations

import polars as pl

from .engine import DAY, HOUR, Agg, window_name

U, I, A, AL = ("uid",), ("item_id",), ("artist_id",), ("album_id",)
UI, UA, UAL, US = ("uid", "item_id"), ("uid", "artist_id"), ("uid", "album_id"), ("uid", "session")

USER_WINDOWS: tuple[int | None, ...] = (HOUR, 6 * HOUR, DAY, 3 * DAY, 7 * DAY, 14 * DAY, 30 * DAY, 90 * DAY, None)
ITEM_WINDOWS: tuple[int | None, ...] = (DAY, 7 * DAY, 30 * DAY, 90 * DAY, None)
LAST_K = (3, 5, 10, 20, 50, 100, 200)
LAST_K_VALUES = ("listened", "ratio", "algo")

EVENT_VALUES = {
    "listen": ("listened", "ratio", "algo"),
    "listen_algo": ("listened", "ratio"),
    "listen_org": ("listened", "ratio"),
    "like": (),
    "dislike": (),
}


def specs() -> list[Agg]:
    out = []
    for key in (U, UI, UA, UAL):
        for events, values in EVENT_VALUES.items():
            out.append(Agg(key, events, USER_WINDOWS, values, last=events == "listen", first=events == "listen"))
    for key in (I, A, AL):
        for events, values in EVENT_VALUES.items():
            out.append(Agg(key, events, ITEM_WINDOWS, values, first=events == "listen"))
    out.append(Agg(US, "listen", (None,), ("listened", "ratio", "algo"), first=True))
    return out


def _ratio(a: str, b: str, name: str, scale: float = 1.0) -> pl.Expr:
    return pl.when(pl.col(b) > 0).then(pl.col(a) / pl.col(b) * scale).otherwise(None).alias(name)


def derived(columns: list[str]) -> list[pl.Expr]:
    have = set(columns)
    out = []
    for pair in ("ui", "ua", "ual"):
        for w in USER_WINDOWS:
            a, b = f"{pair}__listen__cnt__{window_name(w)}", f"u__listen__cnt__{window_name(w)}"
            if a in have and b in have:
                out.append(_ratio(a, b, f"{pair}__share_of_user__{window_name(w)}"))
    for key, windows in (("u", USER_WINDOWS), ("i", ITEM_WINDOWS)):
        finite = [w for w in windows if w is not None]
        for short, long in zip(finite, finite[1:]):
            a, b = f"{key}__listen__cnt__{window_name(short)}", f"{key}__listen__cnt__{window_name(long)}"
            out.append(_ratio(a, b, f"{key}__trend__{window_name(short)}_{window_name(long)}", long / short))
    for key in ("u", "i", "a", "al", "ui", "ua", "ual"):
        for events in ("like", "dislike"):
            a, b = f"{key}__{events}__cnt__all", f"{key}__listen__cnt__all"
            if a in have and b in have:
                out.append(_ratio(a, b, f"{key}__{events}_rate__all"))
    out += [
        (pl.col("timestamp") % DAY // HOUR).cast(pl.Float64).alias("ctx__hour"),
        (pl.col("timestamp") // DAY % 7).cast(pl.Float64).alias("ctx__dow"),
        pl.col("track_length_seconds").cast(pl.Float64).alias("ctx__track_length"),
    ]
    return out
