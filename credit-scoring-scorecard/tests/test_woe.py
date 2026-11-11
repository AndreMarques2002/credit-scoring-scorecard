import numpy as np
import pandas as pd

from credit_scoring.woe import WoEBinner


def _toy(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(-2.5 + 1.2 * x)))
    y = (rng.random(n) < p).astype(int)
    x_missing = x.copy()
    x_missing[rng.random(n) < 0.1] = np.nan
    return pd.DataFrame({"x": x_missing, "noise": rng.normal(size=n)}), y


def test_bad_rate_is_monotonic_and_iv_ranks_signal_first():
    X, y = _toy()
    b = WoEBinner(max_bins=6).fit(X, y)
    t = b.bins_["x"]["table"]
    rates = t.loc[t["bin"] != "Missing", "bad_rate"].to_numpy()
    assert np.all(np.diff(rates) >= -1e-12)          # monotônica crescente
    assert b.iv_.index[0] == "x"
    assert b.iv_["x"] > 10 * b.iv_["noise"]


def test_respects_max_bins_and_min_size():
    X, y = _toy()
    b = WoEBinner(max_bins=5, min_bin_frac=0.08).fit(X, y)
    t = b.bins_["x"]["table"]
    regular = t[t["bin"] != "Missing"]
    assert len(regular) <= 5
    assert (regular["pct"] >= 0.08 - 1e-9).all()


def test_missing_gets_own_bin_and_transform_handles_nan():
    X, y = _toy()
    b = WoEBinner().fit(X, y)
    assert "Missing" in set(b.bins_["x"]["table"]["bin"])
    W = b.transform(X)
    assert W.shape == X.shape and not W.isna().any().any()


def test_woe_sign_convention_higher_is_safer():
    X, y = _toy()
    b = WoEBinner().fit(X, y)
    W = b.transform(X)
    # x alto => mais risco => WoE (ln bons/maus) menor
    assert np.corrcoef(X["x"].fillna(0), W["x"])[0, 1] < 0
