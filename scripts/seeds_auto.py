"""Дополнительные seed итоговой модели для ключевых вариантов сравнения ручных и автоматических признаков."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.eval.lgbm_eval import BASE_PARAMS, Data, fit, fit_matrices  # noqa: E402

PARAMS = BASE_PARAMS | {"learning_rate": 0.1}
out = ROOT / "experiments" / "auto"
ids = np.load(ROOT / "data" / "features" / "auto_shift0.query_ids.npy")
hand = Data(ROOT / "data" / "features" / "handcrafted_shift0.parquet", query_ids=ids)
auto = Data(ROOT / "data" / "features" / "auto_shift0.parquet", query_ids=ids)
bsbs = pd.read_csv(out / "ranking_auto_bsbs.csv").sort_values("Individual_Rank")["Feature"].tolist()
extra = [f for f in bsbs if f not in set(hand.features)][:30]
rows = []
for seed in (0, 1, 2):
    variants = {
        "hand_all": lambda s: fit(hand, hand.features, PARAMS, s),
        "auto_bsbs_100": lambda s: fit(auto, bsbs[:100], PARAMS, s),
        "hand_plus_auto_30": lambda s: fit_matrices(
            auto, {p: np.hstack([hand.parts[p]["X"], auto.parts[p]["X"][:, auto.columns(extra)]]) for p in ("train", "valid", "test")},
            hand.features + extra, PARAMS, s),
    }
    for name, run in variants.items():
        _, info = run(seed)
        rows.append({"variant": name, "seed": seed, **{f"test_{k}": v for k, v in info["test"].items()},
                     "valid_logloss": info["valid"]["logloss"]})
        print(rows[-1], flush=True)
frame = pd.DataFrame(rows)
frame.to_csv(out / "seeds.csv", index=False)
print(frame.groupby("variant")[["valid_logloss", "test_logloss", "test_auc", "test_gauc", "test_ndcg@10"]].agg(["mean", "std"]).round(4).to_string())
