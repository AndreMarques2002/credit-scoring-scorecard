"""Modelo challenger (XGBoost) para comparar com o scorecard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def fit_challenger(X: pd.DataFrame, y, seed: int = 42) -> XGBClassifier:
    """XGBoost com early stopping em validação interna (15% do treino).

    Não usa scale_pos_weight, para manter probabilidades aproximadamente calibradas.
    O XGBoost trata NaN nativamente.
    """
    X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=seed)
    model = XGBClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=4, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=10, eval_metric="auc",
        early_stopping_rounds=50, tree_method="hist", random_state=seed, n_jobs=-1,
    )
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    return model


def global_importance(model: XGBClassifier, X: pd.DataFrame, max_rows: int = 5000) -> pd.Series:
    """Média do |efeito| de cada variável na saída (contribuições SHAP nativas do XGBoost)."""
    sample = X.sample(min(len(X), max_rows), random_state=0)
    contrib = model.get_booster().predict(xgb.DMatrix(sample), pred_contribs=True)[:, :-1]
    return pd.Series(np.abs(contrib).mean(axis=0), index=X.columns).sort_values(ascending=False)
