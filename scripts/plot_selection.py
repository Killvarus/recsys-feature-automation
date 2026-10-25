"""Кривые «число признаков → качество на test» для методов отбора (experiments/selection/results.csv)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "selection"
HIGHLIGHT = {"ppop": ("#2a78d6", "BSBS"), "permutation": ("#eb6834", "permutation"), "pls": ("#1baf7a", "PLS (компоненты)")}
MUTED = "#b8b7b2"

r = pd.read_csv(OUT / "results.csv")
random = r[r.method.str.startswith("random")]
full = r[r.method == "all"].iloc[0]
others = sorted(set(r.method) - set(HIGHLIGHT) - {"all"} - set(random.method))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, metric, title in ((axes[0], "test_auc", "ROC-AUC на test"), (axes[1], "test_gauc", "GAUC на test")):
    band = random.groupby("k")[metric].agg(["min", "max"])
    ax.fill_between(band.index, band["min"], band["max"], color=MUTED, alpha=0.35, linewidth=0, label="случайные k (3 seed)")
    labels: dict[float, list[str]] = {}
    for m in others:
        s = r[r.method == m].sort_values("k")
        ax.plot(s.k, s[metric], color=MUTED, linewidth=1.2)
        labels.setdefault(round(float(s[metric].iloc[0]), 3), []).append(m)
    if metric == "test_auc":  # подписи серых линий — только слева, совпадающие значения одной строкой
        for y, names in labels.items():
            ax.annotate(", ".join(names), (5, y), fontsize=7, color="#77766f", xytext=(-6, 0),
                        textcoords="offset points", ha="right", va="center")
    for m, (color, label) in HIGHLIGHT.items():
        s = r[r.method == m].sort_values("k")
        ax.plot(s.k, s[metric], color=color, linewidth=2, marker="o", markersize=5, label=label)
    ax.axhline(full[metric], color="#0b0b0b", linestyle="--", linewidth=1, label=f"все {int(full.k)} признаков")
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 30, 50, 100])
    ax.set_xticklabels(["5", "10", "20", "30", "50", "100"])
    ax.set_xlim(3.2, 120)
    low = full[metric] - (0.035 if metric == "test_auc" else 0.08)
    ax.set_ylim(low, full[metric] + 0.006)
    ax.set_xlabel("k — первых признаков рейтинга (компонент)")
    ax.set_title(title, loc="left", fontsize=11)
    ax.grid(color="#e6e5e1", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].legend(fontsize=8, loc="lower right", frameon=False)
fig.suptitle("Методы отбора: LightGBM на первых k признаках (серые линии — остальные методы)", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
path = OUT / "selection_curves.png"
fig.savefig(path, dpi=140)
print(path)
