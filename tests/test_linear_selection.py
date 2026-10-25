import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.feature_selection import TrueBackwardFeatureSelection
from src.models.linear_selection import LinearPPOP, LinearRFE, packet_size


def _data(n_train=60, n_valid=30, p=8, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_train + n_valid, p))
    # Соседние признаки скоррелированы (как окна 7 д / 30 д) и разного масштаба.
    X = (base + 0.6 * np.roll(base, 1, axis=1)) * rng.uniform(0.5, 20.0, size=p)
    weights = rng.normal(size=p) * (rng.uniform(size=p) > 0.3)
    y = X / X.std(axis=0) @ weights + rng.normal(scale=0.5, size=n_train + n_valid)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    y = pd.Series(y, name="y")
    return X.iloc[:n_train], y.iloc[:n_train], X.iloc[n_train:], y.iloc[n_train:]


def _model(alpha):
    if alpha == 0.0:
        return make_pipeline(StandardScaler(), LinearRegression())
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def test_packet_size_fixed_and_fraction():
    assert packet_size(100, 7) == 7
    assert packet_size(5, 7) == 5
    assert packet_size(100, 0.1) == 10
    assert packet_size(95, 0.1) == 10
    assert packet_size(9, 0.1) == 1
    assert packet_size(100, 0.1, min_step=20) == 20
    for bad in (0, -1, 1.0, 0.0, 1.5, True):
        with pytest.raises(ValueError):
            packet_size(10, bad)


@pytest.mark.parametrize("alpha", [0.0, 5.0])
def test_ppop_scores_match_refitting_on_validation(alpha):
    X_train, y_train, X_valid, y_valid = _data()
    p = X_train.shape[1]
    selector = LinearPPOP(alpha=alpha, step=p).fit(X_train, y_train, X_valid, y_valid)
    full = _model(alpha).fit(X_train, y_train)
    mse_full = mean_squared_error(y_valid, full.predict(X_valid))
    assert selector.history_["mse"].iloc[0] == pytest.approx(mse_full, rel=1e-9)
    scores = selector.ranking_frame().set_index("Feature")["Score"]
    for feature in X_train.columns:
        reduced = _model(alpha).fit(X_train.drop(columns=[feature]), y_train)
        mse = mean_squared_error(y_valid, reduced.predict(X_valid.drop(columns=[feature])))
        assert scores[feature] == pytest.approx(mse - mse_full, rel=1e-7, abs=1e-10)


def test_ppop_cv_pools_folds_of_train():
    X_train, y_train, _, _ = _data(n_train=80)
    p = X_train.shape[1]
    selector = LinearPPOP(alpha=3.0, step=p, cv=4, random_state=1).fit(X_train, y_train)
    folds = list(KFold(n_splits=4, shuffle=True, random_state=1).split(X_train))

    def pooled_mse(columns):
        sse = 0.0
        for fit, held in folds:
            model = _model(3.0).fit(X_train.iloc[fit][columns], y_train.iloc[fit])
            sse += float(((y_train.iloc[held] - model.predict(X_train.iloc[held][columns])) ** 2).sum())
        return sse / len(X_train)

    columns = list(X_train.columns)
    mse_full = pooled_mse(columns)
    scores = selector.ranking_frame().set_index("Feature")["Score"]
    for feature in columns:
        expected = pooled_mse([c for c in columns if c != feature]) - mse_full
        assert scores[feature] == pytest.approx(expected, rel=1e-7, abs=1e-10)


def test_ppop_matches_generic_refitting_class():
    X_train, y_train, X_valid, y_valid = _data(p=10)
    fast = LinearPPOP(alpha=2.0, step=3).fit(X_train, y_train, X_valid, y_valid)
    slow = TrueBackwardFeatureSelection(_model(2.0), n_features_to_drop=3, cv=None)
    slow.fit(X_train, y_train, X_val=X_valid, y_val=y_valid)
    fast_frame = fast.ranking_frame()
    slow_frame = slow.get_ranking_df()
    assert fast_frame["Feature"].tolist() == slow_frame["Feature"].tolist()
    assert fast_frame["Group_Rank"].tolist() == slow_frame["Group_Rank"].tolist()


def test_ppop_fraction_packets_shrink():
    X_train, y_train, X_valid, y_valid = _data(p=12)
    selector = LinearPPOP(alpha=1.0, step=0.25).fit(X_train, y_train, X_valid, y_valid)
    assert selector.history_["n_features"].tolist() == [12, 9, 6, 4, 3, 2, 1]
    assert selector.history_["dropped"].tolist() == [3, 3, 2, 1, 1, 1, 1]
    frame = selector.ranking_frame()
    assert sorted(frame["Individual_Rank"]) == list(range(1, 13))
    assert frame["Group_Rank"].iloc[0] == 1


def test_ols_refuses_more_features_than_rows():
    X_train, y_train, X_valid, y_valid = _data(n_train=6, p=8)
    with pytest.raises(ValueError, match="МНК"):
        LinearPPOP(alpha=0.0, step=1).fit(X_train, y_train, X_valid, y_valid)
    with pytest.raises(ValueError, match="МНК"):
        LinearRFE(alpha=0.0, step=1).fit(X_train, y_train)


@pytest.mark.parametrize("refresh_every", [1, 3, 1000])
def test_rfe_step_one_matches_sklearn(refresh_every):
    X_train, y_train, _, _ = _data(n_train=50, p=15, seed=3)
    ours = LinearRFE(alpha=4.0, step=1, refresh_every=refresh_every).fit(X_train, y_train)
    scaled = StandardScaler().fit_transform(X_train)
    reference = RFE(Ridge(alpha=4.0), n_features_to_select=1, step=1).fit(scaled, y_train)
    expected = dict(zip(X_train.columns, reference.ranking_))
    frame = ours.ranking_frame()
    assert dict(zip(frame["Feature"], frame["Individual_Rank"])) == expected


def test_rfe_fraction_matches_refitting_loop():
    X_train, y_train, _, _ = _data(n_train=50, p=15, seed=4)
    ours = LinearRFE(alpha=4.0, step=0.3, refresh_every=1000).fit(X_train, y_train)
    scaled = pd.DataFrame(StandardScaler().fit_transform(X_train), columns=X_train.columns)
    remaining = list(X_train.columns)
    dropped_order = []
    while remaining:
        coef = np.abs(Ridge(alpha=4.0).fit(scaled[remaining], y_train).coef_)
        size = packet_size(len(remaining), 0.3)
        packet = [remaining[i] for i in np.argsort(coef, kind="stable")[:size]]
        dropped_order.append(set(packet))
        remaining = [f for f in remaining if f not in packet]
    frame = ours.ranking_frame()
    groups = [set(frame.loc[frame["Group_Rank"] == g, "Feature"]) for g in sorted(frame["Group_Rank"].unique())]
    assert groups == list(reversed(dropped_order))


def test_loo_alpha_is_chosen_per_iteration():
    X_train, y_train, X_valid, y_valid = _data(n_train=60, p=12)
    selector = LinearPPOP(alpha="loo", step=0.25).fit(X_train, y_train, X_valid, y_valid)
    assert selector.history_["alpha"].gt(0).all()
    scaled = StandardScaler().fit_transform(X_train)
    reference = RidgeCV(alphas=np.logspace(-3, 6, 46)).fit(scaled, y_train)
    assert selector.history_["alpha"].iloc[0] == pytest.approx(reference.alpha_)
    rfe = LinearRFE(alpha="loo", step=0.25).fit(X_train, y_train)
    assert rfe.history_["alpha"].iloc[0] == pytest.approx(reference.alpha_)


def test_rfe_with_loo_alpha_matches_refitting_with_the_same_alphas():
    X_train, y_train, _, _ = _data(n_train=50, p=15, seed=5)
    ours = LinearRFE(alpha="loo", step=1, refresh_every=1).fit(X_train, y_train)
    scaled = pd.DataFrame(StandardScaler().fit_transform(X_train), columns=X_train.columns)
    remaining = list(X_train.columns)
    dropped = []
    for alpha in ours.history_["alpha"]:
        coef = np.abs(Ridge(alpha=alpha).fit(scaled[remaining], y_train).coef_)
        dropped.append(remaining.pop(int(np.argmin(coef))))
    order = ours.ranking_frame().sort_values("Individual_Rank", ascending=False)["Feature"].tolist()
    assert order == dropped
    assert ours.history_["alpha"].nunique() > 1


def test_one_se_rule_never_picks_smaller_alpha_than_minimum():
    from src.models.linear_selection import loo_alpha, ridge_spectrum

    X_train, y_train, _, _ = _data(n_train=60, p=30, seed=6)
    scaled = StandardScaler().fit_transform(X_train)
    u, s2, _ = ridge_spectrum(scaled)
    target = y_train.to_numpy() - y_train.mean()
    best = loo_alpha(u, s2, target, rule="min")
    one_se = loo_alpha(u, s2, target, rule="1se")
    assert one_se >= best
    selector = LinearRFE(alpha="loo1se", step=0.25).fit(X_train, y_train)
    assert selector.history_["alpha"].iloc[0] == pytest.approx(one_se)


def test_ppop_reshuffle_uses_new_folds_each_iteration():
    X_train, y_train, _, _ = _data(n_train=80, p=10, seed=7)
    fixed = LinearPPOP(alpha=3.0, step=10, cv=4, random_state=1, reshuffle=True).fit(X_train, y_train)
    plain = LinearPPOP(alpha=3.0, step=10, cv=4, random_state=1).fit(X_train, y_train)
    # Одна итерация: при том же seed результаты совпадают.
    pd.testing.assert_frame_equal(fixed.ranking_frame(), plain.ranking_frame())
    several = LinearPPOP(alpha=3.0, step=0.3, cv=4, random_state=1, reshuffle=True).fit(X_train, y_train)
    assert sorted(several.ranking_frame()["Individual_Rank"]) == list(range(1, 11))
    assert several.history_["mse"].iloc[0] == pytest.approx(plain.history_["mse"].iloc[0])


def test_rfe_loo_schedule_switches_rule_at_threshold():
    from src.models.linear_selection import loo_alpha, ridge_spectrum

    X_train, y_train, _, _ = _data(n_train=60, p=30, seed=6)
    ours = LinearRFE(alpha="loo_schedule", step=1, switch_at=12, refresh_every=5).fit(X_train, y_train)
    scaled = pd.DataFrame(StandardScaler().fit_transform(X_train), columns=X_train.columns)
    target = y_train.to_numpy() - y_train.mean()
    order = ours.ranking_frame().sort_values("Individual_Rank", ascending=False)["Feature"].tolist()
    remaining = list(X_train.columns)
    since_refresh = 0
    for n_features, alpha in zip(ours.history_["n_features"], ours.history_["alpha"]):
        # Пересчёт раз в 5 шагов и сразу при 12 признаках, когда меняется правило.
        if n_features in (30, 12) or since_refresh == 5:
            u, s2, _ = ridge_spectrum(scaled[remaining].to_numpy())
            expected = loo_alpha(u, s2, target, rule="1se" if n_features > 12 else "min")
            if n_features <= 12:
                reference = RidgeCV(alphas=np.logspace(-3, 6, 46)).fit(scaled[remaining], y_train)
                assert expected == pytest.approx(reference.alpha_)
            since_refresh = 0
        assert alpha == pytest.approx(expected)
        since_refresh += 1
        coef = np.abs(Ridge(alpha=alpha).fit(scaled[remaining], y_train).coef_)
        assert order[30 - n_features] == remaining.pop(int(np.argmin(coef)))
    assert ours.history_["alpha"].nunique() > 1


@pytest.mark.parametrize("switch_at, plain", [(1000, "loo"), (0, "loo1se")])
def test_loo_schedule_reduces_to_one_rule(switch_at, plain):
    X_train, y_train, X_valid, y_valid = _data(n_train=60, p=20, seed=8)
    for fit in (
        lambda alpha, **kw: LinearRFE(alpha=alpha, step=0.2, **kw).fit(X_train, y_train),
        lambda alpha, **kw: LinearPPOP(alpha=alpha, step=0.2, **kw).fit(X_train, y_train, X_valid, y_valid),
    ):
        scheduled = fit("loo_schedule", switch_at=switch_at)
        reference = fit(plain)
        pd.testing.assert_frame_equal(scheduled.ranking_frame(), reference.ranking_frame())
        assert scheduled.history_["alpha"].tolist() == reference.history_["alpha"].tolist()


def test_ppop_loo_schedule_matches_refitting_with_the_same_alphas():
    from src.models.linear_selection import loo_alpha, ridge_spectrum

    X_train, y_train, X_valid, y_valid = _data(n_train=60, p=12, seed=9)
    ours = LinearPPOP(alpha="loo_schedule", step=0.25, switch_at=6).fit(X_train, y_train, X_valid, y_valid)
    scaled = pd.DataFrame(StandardScaler().fit_transform(X_train), columns=X_train.columns)
    target = y_train.to_numpy() - y_train.mean()
    frame = ours.ranking_frame().set_index("Feature")
    remaining = list(X_train.columns)
    for n_features, alpha, mse in zip(ours.history_["n_features"], ours.history_["alpha"], ours.history_["mse"]):
        u, s2, _ = ridge_spectrum(scaled[remaining].to_numpy())
        assert alpha == pytest.approx(loo_alpha(u, s2, target, rule="1se" if n_features > 6 else "min"))
        full = _model(alpha).fit(X_train[remaining], y_train)
        assert mse == pytest.approx(mean_squared_error(y_valid, full.predict(X_valid[remaining])), rel=1e-9)
        packet = frame.index[frame["Group_Rank"] == frame.loc[remaining, "Group_Rank"].max()].tolist()
        for feature in packet:
            kept = [f for f in remaining if f != feature]
            if kept:
                predicted = _model(alpha).fit(X_train[kept], y_train).predict(X_valid[kept])
            else:
                predicted = np.full(len(y_valid), y_train.mean())
            increase = mean_squared_error(y_valid, predicted) - mse
            assert frame.loc[feature, "Score"] == pytest.approx(increase, rel=1e-7, abs=1e-10)
        remaining = [f for f in remaining if f not in packet]
    assert ours.history_["alpha"].nunique() > 1


def test_bad_switch_at_is_rejected():
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError):
            LinearRFE(alpha="loo_schedule", switch_at=bad)


class _ChannelScaler:
    """Нормировка по каналам как в scale_channels, в виде шага sklearn-конвейера."""

    def __init__(self, method):
        self.method = method

    def fit(self, X, y=None):
        from src.models.linear_selection import scale_channels

        self.X_fit_ = np.asarray(X, dtype=float)
        scale_channels(self.X_fit_, method=self.method)
        return self

    def transform(self, X):
        from src.models.linear_selection import scale_channels

        return scale_channels(self.X_fit_, np.asarray(X, dtype=float), method=self.method)[1]

    def get_params(self, deep=True):
        return {"method": self.method}

    def set_params(self, **params):
        self.method = params.get("method", self.method)
        return self


@pytest.mark.parametrize("scaling", ["pareto", "robust", "center"])
def test_ppop_with_channel_scaling_matches_refitting(scaling):
    X_train, y_train, X_valid, y_valid = _data()
    p = X_train.shape[1]
    selector = LinearPPOP(alpha=5.0, step=p, scaling=scaling).fit(X_train, y_train, X_valid, y_valid)

    def mse(columns):
        model = make_pipeline(_ChannelScaler(scaling), Ridge(alpha=5.0)).fit(X_train[columns], y_train)
        return mean_squared_error(y_valid, model.predict(X_valid[columns]))

    columns = list(X_train.columns)
    full = mse(columns)
    scores = selector.ranking_frame().set_index("Feature")["Score"]
    for feature in columns:
        expected = mse([c for c in columns if c != feature]) - full
        assert scores[feature] == pytest.approx(expected, rel=1e-7, abs=1e-10)


@pytest.mark.parametrize("scaling", ["pareto", "robust", "center"])
def test_rfe_with_channel_scaling_matches_sklearn(scaling):
    from src.models.linear_selection import scale_channels

    X_train, y_train, _, _ = _data(n_train=50, p=15, seed=3)
    ours = LinearRFE(alpha=4.0, step=1, scaling=scaling).fit(X_train, y_train)
    (scaled,) = scale_channels(X_train.to_numpy(dtype=float), method=scaling)
    reference = RFE(Ridge(alpha=4.0), n_features_to_select=1, step=1).fit(scaled, y_train)
    frame = ours.ranking_frame()
    assert dict(zip(frame["Feature"], frame["Individual_Rank"])) == dict(zip(X_train.columns, reference.ranking_))


def test_scaling_changes_rfe_order_and_unknown_scaling_fails():
    X_train, y_train, _, _ = _data(n_train=50, p=15, seed=3)
    z = LinearRFE(alpha=4.0, step=1).fit(X_train, y_train).ranking_frame()["Feature"].tolist()
    c = LinearRFE(alpha=4.0, step=1, scaling="center").fit(X_train, y_train).ranking_frame()["Feature"].tolist()
    assert z != c
    with pytest.raises(ValueError):
        LinearRFE(scaling="minmax")


def test_scale_only_divides_by_the_same_scale_without_centering():
    from src.models.linear_selection import scale_channels, scale_only

    X_train, _, X_valid, _ = _data()
    train, valid = X_train.to_numpy(), X_valid.to_numpy()
    for method in ("zscore", "pareto", "robust"):
        only_train, only_valid = scale_only(train, valid, method=method)
        full_train, full_valid = scale_channels(train, valid, method=method)
        shift = train.mean(axis=0) / (train / only_train)[0]
        np.testing.assert_allclose(only_train - shift, full_train, atol=1e-9)
        np.testing.assert_allclose(only_valid - shift, full_valid, atol=1e-9)
