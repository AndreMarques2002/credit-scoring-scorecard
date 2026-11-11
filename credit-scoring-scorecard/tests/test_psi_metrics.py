import numpy as np

from credit_scoring.metrics import band_table, evaluate, ks_statistic
from credit_scoring.psi import psi, psi_status


def test_psi_identical_is_zero_and_shift_is_large():
    rng = np.random.default_rng(1)
    a = rng.normal(size=20000)
    assert psi(a, a) < 1e-9
    assert psi(a, rng.normal(size=20000)) < 0.1
    shifted = psi(a, rng.normal(loc=1.0, size=20000))
    assert shifted > 0.25 and psi_status(shifted) == "ação"


def test_psi_handles_nan_as_own_bin():
    rng = np.random.default_rng(2)
    a = rng.normal(size=5000)
    b = a.copy()
    b[:1500] = np.nan
    assert psi(a, b) > 0.1


def test_ks_and_auc_for_perfect_and_random_scores():
    y = np.array([0] * 500 + [1] * 500)
    assert ks_statistic(y, y.astype(float)) == 1.0
    assert evaluate(y, y.astype(float))["auc"] == 1.0
    rng = np.random.default_rng(3)
    assert abs(evaluate(y, rng.random(1000))["auc"] - 0.5) < 0.08


def test_band_table_structure():
    rng = np.random.default_rng(4)
    score = rng.normal(600, 50, 10000)
    y = (rng.random(10000) < 1 / (1 + np.exp((score - 560) / 20))).astype(int)
    t = band_table(y, score)
    assert len(t) == 10 and t["n"].sum() == 10000
    assert abs(t["cum_bad_capture"].iloc[-1] - 1.0) < 1e-9
    assert t["bad_rate"].is_monotonic_decreasing
