import numpy as np
import pytest
from sklearn.metrics import ndcg_score, roc_auc_score

from src.eval.metrics import evaluate


def test_gauc_and_ndcg_match_sklearn_per_group():
    rng = np.random.default_rng(0)
    n = 400
    uid = rng.integers(0, 5, n)
    session = rng.integers(0, 4, n)
    label = (rng.uniform(size=n) < 0.6).astype(int)
    smooth = label * 0.3 + rng.uniform(size=n)
    score = np.round(smooth, 1)  # равные прогнозы — для AUC
    result = evaluate(uid, session, label, score, k=10, probabilities=False)
    exact = evaluate(uid, session, label, smooth, k=10, probabilities=False)

    aucs, weights, ndcgs = [], [], []
    for u in np.unique(uid):
        m = uid == u
        if 0 < label[m].sum() < m.sum():
            aucs.append(roc_auc_score(label[m], score[m]))
            weights.append(m.sum())
        for s in np.unique(session[m]):
            g = m & (session == s)
            if 0 < label[g].sum() < g.sum():
                ndcgs.append(ndcg_score([label[g]], [smooth[g]], k=10))
    assert result["auc"] == pytest.approx(roc_auc_score(label, score))
    assert result["gauc"] == pytest.approx(np.average(aucs, weights=weights))
    assert exact["ndcg@10"] == pytest.approx(np.mean(ndcgs))
