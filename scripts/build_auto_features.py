"""
Пул автоматически сгенерированных признаков (src/features/auto.py).

Строки — подмножество тех же примеров, что у ручных признаков (data/features/handcrafted_shift0.parquet):
train — n_train случайных примеров из train ручного набора; valid и test — все примеры доли пользователей
users_frac (сессии и пользователи целиком, чтобы GAUC и NDCG считались так же). Ручные признаки
для сравнения берутся на тех же строках.

python scripts/build_auto_features.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.data.yambda import Split, load_events, targets  # noqa: E402
from src.features import auto  # noqa: E402
from src.features.handcrafted import build  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handcrafted", type=Path, default=ROOT / "data" / "features" / "handcrafted_shift0.parquet")
    parser.add_argument("--n-train", type=int, default=400_000)
    parser.add_argument("--users-frac", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "features" / "auto_shift0.parquet")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    rows = pl.read_parquet(args.handcrafted, columns=["query_id", "uid", "part"])
    users = rows["uid"].unique().sort()
    kept_users = users.sample(fraction=args.users_frac, seed=0)
    train = rows.filter(pl.col("part") == "train").sample(n=min(args.n_train, rows.filter(pl.col("part") == "train").height), seed=0)
    other = rows.filter((pl.col("part") != "train") & pl.col("uid").is_in(kept_users.implode()))
    chosen = pl.concat([train, other])["query_id"]

    events = load_events()
    queries = targets(events, Split()).rename({"event_idx": "query_id"}).filter(pl.col("query_id").is_in(chosen.implode()))
    print(queries.group_by("part").agg(pl.len(), pl.col("label").mean().alias("pos_rate")).sort("part"), flush=True)
    specs = auto.specs()
    print(f"specs {len(specs)}, load {time.perf_counter() - started:.0f}s", flush=True)

    parts_dir = args.out.parent / f"parts_{args.out.stem}"
    if parts_dir.exists() and not args.resume:
        shutil.rmtree(parts_dir)
    features = build(queries, events, parts_dir, specs=specs, last_k_ks=auto.LAST_K, last_k_values=auto.LAST_K_VALUES,
                     derived=auto.derived, keep_sums=False)
    del events
    features.write_parquet(args.out)
    n_features = features.width - queries.width
    print(f"saved {args.out} rows {features.height:,} features {n_features}, total {time.perf_counter() - started:.0f}s", flush=True)
    np.save(args.out.with_suffix(".query_ids.npy"), features["query_id"].to_numpy())


if __name__ == "__main__":
    main()
