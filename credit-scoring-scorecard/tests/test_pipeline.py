import numpy as np
from scipy.stats import spearmanr

from credit_scoring.config import ScoreScale
from credit_scoring.data import clean, make_synthetic, split
from credit_scoring.scorecard import Scorecard
from credit_scoring.train import run


def test_clean_removes_invalid_age_and_special_codes():
    df = make_synthetic(n=3000, seed=0)
    out = clean(df)
    assert (out["age"] >= 18).all()
    assert (out["NumberOfTimes90DaysLate"].dropna() < 90).all()


def test_points_sum_equals_score():
    df = clean(make_synthetic(n=8000, seed=1))
    X_tr, X_te, y_tr, _ = split(df)
    sc = Scorecard(scale=ScoreScale()).fit(X_tr, y_tr)

    scores = sc.score(X_te)
    table = sc.points_table()
    # reconstrói o score somando os pontos da faixa de cada variável
    W = sc.binner_.transform(X_te)[sc.features_]
    k = len(sc.features_)
    base = (sc.offset_ - sc.factor_ * sc.intercept_) / k
    rebuilt = sum(base - sc.factor_ * sc.coef_[f] * W[f] for f in sc.features_)
    assert np.allclose(scores, rebuilt)
    assert set(table["feature"]) == set(sc.features_)


def test_score_is_higher_for_lower_risk():
    df = clean(make_synthetic(n=8000, seed=2))
    X_tr, X_te, y_tr, _ = split(df)
    sc = Scorecard().fit(X_tr, y_tr)
    # score é função monotônica decrescente da PD (a relação não é linear, por isso Spearman)
    assert spearmanr(sc.score(X_te), sc.predict_proba_bad(X_te)).correlation < -0.999
    assert (sc.coef_ < 0).all()  # sinais coerentes com WoE = ln(bons/maus)


def test_end_to_end_smoke(tmp_path):
    m = run(make_synthetic(n=6000, seed=3), tmp_path / "reports", tmp_path / "models")
    assert m["scorecard"]["test"]["auc"] > 0.7
    assert (tmp_path / "reports" / "results.md").exists()
    assert (tmp_path / "reports" / "figures" / "roc_ks.png").exists()
    assert (tmp_path / "models" / "scorecard.joblib").exists()
