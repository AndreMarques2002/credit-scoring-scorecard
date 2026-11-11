"""Métricas de risco de crédito: AUC, Gini, KS, PR-AUC, Brier e tabela por faixas."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def ks_statistic(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def evaluate(y, p_bad) -> dict:
    auc = roc_auc_score(y, p_bad)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y, p_bad),
        "pr_auc": float(average_precision_score(y, p_bad)),
        "brier": float(brier_score_loss(y, p_bad)),
        "bad_rate": float(np.mean(y)),
        "n": int(len(y)),
    }


def band_table(y, score, n_bands: int = 10) -> pd.DataFrame:
    """Faixas de score (decis). Banda 1 = pior score (maior risco).

    Em um bom modelo a taxa de maus decresce da banda 1 para a última.
    Com poucas faixas de WoE, muitos scores se repetem; por isso o número de faixas
    efetivas pode ser menor que `n_bands`.
    """
    df = pd.DataFrame({"y": np.asarray(y), "score": np.asarray(score)})
    # Scores idênticos ficam na mesma faixa (duplicates="drop" pode gerar < n_bands faixas).
    df["band"] = pd.qcut(df["score"], n_bands, labels=False, duplicates="drop") + 1
    g = df.groupby("band").agg(score_min=("score", "min"), score_max=("score", "max"),
                               n=("y", "size"), bad=("y", "sum"))
    g["bad_rate"] = g["bad"] / g["n"]
    g["cum_bad_capture"] = g["bad"].cumsum() / g["bad"].sum()
    g["lift"] = g["bad_rate"] / df["y"].mean()
    return g.reset_index()
