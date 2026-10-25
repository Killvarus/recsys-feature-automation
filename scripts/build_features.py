"""
Сборка ручных признаков для train / valid / test.

python scripts/build_features.py                 # основное разбиение
python scripts/build_features.py --shift-days 7  # то же, сдвинутое на неделю назад
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import polars as pl  # noqa: E402

from src.data.yambda import Split, load_events, targets  # noqa: E402
from src.features.handcrafted import build  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shift-days", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=1.0, help="доля целевых событий train (история не режется)")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "features")
    parser.add_argument("--resume", action="store_true", help="не пересчитывать уже сохранённые части (только для тех же queries)")
    args = parser.parse_args()

    started = time.perf_counter()
    events = load_events()
    split = Split().shifted(args.shift_days)
    print(f"events {events.height:,}, load {time.perf_counter() - started:.1f}s; split {split}", flush=True)

    queries = targets(events, split).rename({"event_idx": "query_id"})
    if args.train_frac < 1.0:
        train = queries.filter(pl.col("part") == "train").sample(fraction=args.train_frac, seed=0)
        queries = pl.concat([train, queries.filter(pl.col("part") != "train")])
    print(queries.group_by("part").agg(pl.len(), pl.col("label").mean().alias("pos_rate")).sort("part"), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    parts_dir = args.out / f"parts_shift{args.shift_days}"
    if parts_dir.exists() and not args.resume:
        shutil.rmtree(parts_dir)
    features = build(queries, events, parts_dir)
    del events
    path = args.out / f"handcrafted_shift{args.shift_days}.parquet"
    features.write_parquet(path)
    print(f"saved {path} shape {features.shape}, total {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
