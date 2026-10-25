"""
Модель над историей событий: трансформер + внимание по кандидату (в духе DIN / BST).

Токен события = трек + артист + тип события + давность (корзина log2 секунд) + позиция с конца
+ доля прослушанного и флаг органики (вариант match_flags — ещё флаги «тот же трек / артист, что кандидат»). Один слой трансформера обрабатывает историю, затем кандидат
(трек, артист, длина) взвешивает события: какие из прошлых важны именно для этого трека. Голова — MLP над
[кандидат, взвешенная история, последнее событие, их произведение]. Ручных признаков нет.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .history import N_DT, N_TYPES


class SeqModel(nn.Module):
    def __init__(self, n_items: int, n_artists: int, length: int, d: int = 64, heads: int = 2, layers: int = 1,
                 dropout: float = 0.1, match_flags: bool = False):
        super().__init__()
        self.item = nn.Embedding(n_items, d, padding_idx=0)
        self.artist = nn.Embedding(n_artists, d, padding_idx=0)
        self.event_type = nn.Embedding(N_TYPES, d, padding_idx=0)
        self.dt = nn.Embedding(N_DT, d)
        self.pos = nn.Embedding(length, d)
        # доля прослушанного, органика (+ флаги «тот же трек / артист, что кандидат», если match_flags)
        self.match_flags = match_flags
        self.side = nn.Linear(4 if match_flags else 2, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.candidate = nn.Sequential(nn.Linear(d + 1, d), nn.ReLU(), nn.Linear(d, d))
        self.query = nn.Linear(d, d, bias=False)
        self.head = nn.Sequential(nn.Linear(4 * d, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))
        self.d = d
        self.register_buffer("positions", torch.arange(length), persistent=False)

    def _side(self, b: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        side = [b["ratio"], b["organic"]]
        return side + [b["same_item"], b["same_artist"]] if self.match_flags else side

    def forward(self, b: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = b["mask"]
        tokens = (self.item(b["item"]) + self.artist(b["artist"]) + self.event_type(b["type"]) + self.dt(b["dt"])
                  + self.pos(self.positions)[None] + self.side(torch.stack(self._side(b), -1)))
        hidden = self.encoder(tokens, src_key_padding_mask=~mask)
        cand = self.candidate(torch.cat([self.item(b["cand_item"]) + self.artist(b["cand_artist"]),
                                         b["cand_len"][:, None]], -1))
        scores = (self.query(hidden) * cand[:, None, :]).sum(-1) / math.sqrt(self.d)
        weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=1)
        pooled = (weights[..., None] * hidden).sum(1)
        last = hidden[:, -1]  # самое свежее событие (у примера всегда есть хотя бы предыдущее прослушивание)
        return self.head(torch.cat([cand, pooled, last, cand * pooled], -1)).squeeze(-1)
