"""
Модель над историей событий (src/seq): обучение и прогноз.

--mode standalone: обучение на тех же 400 тыс. примерах train, что у сравнения ручных и автоматических признаков
                   (data/features/auto_shift0.parquet), early stopping по valid, метрики на test тех же строк.
--mode stack:      обучение на всех примерах train с днём события < stack_day; прогноз для примеров train
                   ручного набора с днём ≥ stack_day и для valid/test. Прогноз затем — признак для LightGBM,
                   обученного только на поздней части train (scripts/compare_seq.py): так прогноз на его train
                   не переобучен. Словарь треков — только по событиям до stack_day.

python scripts/train_seq.py --mode standalone
python scripts/train_seq.py --mode stack
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.data.yambda import DAY, Split, load_events, targets  # noqa: E402
from src.eval.metrics import evaluate  # noqa: E402
from src.seq.history import History  # noqa: E402
from src.seq.model import SeqModel  # noqa: E402

META = ["query_id", "uid", "item_id", "artist_id", "timestamp", "event_ts", "eval_session", "track_length_seconds",
        "part", "label"]


def as_arrays(frame: pl.DataFrame, history: History) -> dict[str, np.ndarray]:
    frame = frame.with_columns(
        history.map_items(pl.col("item_id")).alias("cand_item"),
        history.map_artists(pl.col("artist_id")).alias("cand_artist"),
        (pl.col("track_length_seconds").cast(pl.Float64).log1p() / 6.0).cast(pl.Float32).alias("cand_len"),
    )
    return {c: frame[c].to_numpy() for c in ["query_id", "uid", "timestamp", "eval_session", "label",
                                             "cand_item", "cand_artist", "cand_len"]}


def tensors(history: History, q: dict, idx: np.ndarray, length: int) -> dict[str, torch.Tensor]:
    b = history.batch(q["uid"][idx], q["timestamp"][idx], length)
    out = {k: torch.from_numpy(np.ascontiguousarray(b[k])).long() for k in ("item", "artist", "type", "dt")}
    out["ratio"] = torch.from_numpy(b["ratio"])
    out["organic"] = torch.from_numpy(b["organic"])
    out["mask"] = torch.from_numpy(b["mask"])
    out["cand_item"] = torch.from_numpy(q["cand_item"][idx].astype(np.int64))
    out["cand_artist"] = torch.from_numpy(q["cand_artist"][idx].astype(np.int64))
    out["cand_len"] = torch.from_numpy(q["cand_len"][idx].astype(np.float32))
    # Явные флаги совпадения с кандидатом (редкие объекты с общим индексом 1 совпадением не считаются).
    cand_item, cand_artist = q["cand_item"][idx][:, None], q["cand_artist"][idx][:, None]
    out["same_item"] = torch.from_numpy(((b["item"] == cand_item) & (cand_item > 1)).astype(np.float32))
    out["same_artist"] = torch.from_numpy(((b["artist"] == cand_artist) & (cand_artist > 1)).astype(np.float32))
    return out


@torch.no_grad()
def predict(model: SeqModel, history: History, q: dict, length: int, batch: int = 8192) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(q["uid"]), batch):
        idx = np.arange(s, min(s + batch, len(q["uid"])))
        out.append(torch.sigmoid(model(tensors(history, q, idx, length))).numpy())
    return np.concatenate(out)


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train(model: SeqModel, history: History, q_train: dict, q_valid: dict, args) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    n = len(q_train["uid"])
    y = torch.from_numpy(q_train["label"].astype(np.float32))
    best, best_state, waited, log = float("inf"), None, 0, []
    for epoch in range(args.epochs):
        model.train()
        started = time.perf_counter()
        order = rng.permutation(n)
        total = 0.0
        for step, s in enumerate(range(0, n, args.batch)):
            idx = order[s:s + args.batch]
            loss = F.binary_cross_entropy_with_logits(model(tensors(history, q_train, idx, args.length)), y[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(idx)
        valid_loss = logloss(q_valid["label"], predict(model, history, q_valid, args.length))
        log.append({"epoch": epoch + 1, "train_logloss": total / n, "valid_logloss": valid_loss,
                    "seconds": round(time.perf_counter() - started, 1)})
        print(f"[seq] epoch {epoch + 1}: train {total / n:.4f} valid {valid_loss:.4f} ({log[-1]['seconds']}s)", flush=True)
        if valid_loss < best - 1e-5:
            best, best_state, waited = valid_loss, copy.deepcopy(model.state_dict()), 0
        else:
            waited += 1
            if waited >= args.patience:
                break
    model.load_state_dict(best_state)
    return {"epochs": log, "best_valid_logloss": best}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["standalone", "stack"], required=True)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stack-day", type=int, default=277)
    parser.add_argument("--match-flags", action="store_true", help="флаги совпадения истории с кандидатом (версия v2)")
    parser.add_argument("--tag", default="", help="суффикс файлов результата")
    parser.add_argument("--limit", type=int, default=0, help="для отладки: не больше стольких примеров train")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "seq")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(6)

    started = time.perf_counter()
    events = load_events()
    split = Split()
    cutoff = (args.stack_day if args.mode == "stack" else split.valid[0]) * DAY
    history = History(events, vocab_cutoff=cutoff)
    print(f"history: {len(history.key):,} events, items {history.n_items:,}, artists {history.n_artists:,}, "
          f"{time.perf_counter() - started:.0f}s", flush=True)

    auto = pl.read_parquet(ROOT / "data" / "features" / "auto_shift0.parquet", columns=META)
    q_valid = as_arrays(auto.filter(pl.col("part") == "valid"), history)
    q_test = as_arrays(auto.filter(pl.col("part") == "test"), history)
    if args.mode == "standalone":
        train_frame = auto.filter(pl.col("part") == "train")
    else:
        all_targets = targets(events, split).rename({"event_idx": "query_id"})
        train_frame = all_targets.filter((pl.col("part") == "train") & (pl.col("event_ts") // DAY < args.stack_day))
    del events
    if args.limit:
        train_frame = train_frame.sample(n=min(args.limit, train_frame.height), seed=0)
    q_train = as_arrays(train_frame.select(META), history)
    print(f"train {len(q_train['uid']):,}, valid {len(q_valid['uid']):,}, test {len(q_test['uid']):,}", flush=True)

    model = SeqModel(history.n_items, history.n_artists, args.length, d=args.d, match_flags=args.match_flags)
    n_params = sum(p.numel() for p in model.parameters())
    fit_started = time.perf_counter()
    info = train(model, history, q_train, q_valid, args)
    fit_seconds = time.perf_counter() - fit_started

    t0 = time.perf_counter()
    p_test = predict(model, history, q_test, args.length)
    predict_seconds = time.perf_counter() - t0
    p_valid = predict(model, history, q_valid, args.length)
    report = {
        "mode": args.mode, "params": n_params, "fit_seconds": round(fit_seconds), "predict_seconds_test": round(predict_seconds, 1),
        "train_rows": len(q_train["uid"]), **info,
        "valid": evaluate(q_valid["uid"], q_valid["eval_session"], q_valid["label"], p_valid),
        "test": evaluate(q_test["uid"], q_test["eval_session"], q_test["label"], p_test),
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    print(f"[seq {args.mode}] test " + "  ".join(f"{k} {v:.4f}" for k, v in report["test"].items()), flush=True)
    frames = [pl.DataFrame({"query_id": q_valid["query_id"], "seq_score": p_valid}),
              pl.DataFrame({"query_id": q_test["query_id"], "seq_score": p_test})]
    if args.mode == "stack":
        hand = pl.read_parquet(ROOT / "data" / "features" / "handcrafted_shift0.parquet", columns=META)
        late = hand.filter((pl.col("part") == "train") & (pl.col("event_ts") // DAY >= args.stack_day))
        q_late = as_arrays(late, history)
        frames.append(pl.DataFrame({"query_id": q_late["query_id"], "seq_score": predict(model, history, q_late, args.length)}))
        report["late_train_rows"] = len(q_late["uid"])
    pl.concat(frames).write_parquet(args.out / f"preds_{args.mode}{args.tag}.parquet")
    (args.out / f"report_{args.mode}{args.tag}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved, total {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
