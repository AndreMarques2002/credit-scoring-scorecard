"""Simulação ILUSTRATIVA de política de corte (cutoff) por lucro esperado."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BusinessAssumptions


def cutoff_analysis(y, score, assumptions: BusinessAssumptions | None = None,
                    n_cutoffs: int = 60) -> pd.DataFrame:
    """Para cada corte, aprova quem tem score >= corte e calcula lucro por proponente."""
    a = assumptions or BusinessAssumptions()
    y, score = np.asarray(y), np.asarray(score)
    cuts = np.unique(np.quantile(score, np.linspace(0, 0.95, n_cutoffs)))
    rows = []
    for c in cuts:
        approved = score >= c
        n_app = int(approved.sum())
        bad_app = int(y[approved].sum())
        good_app = n_app - bad_app
        profit = good_app * a.profit_per_good - bad_app * a.loss_per_bad
        rows.append({
            "cutoff": float(c),
            "approval_rate": n_app / len(y),
            "bad_rate_approved": bad_app / n_app if n_app else np.nan,
            "profit_per_applicant": profit / len(y),
        })
    return pd.DataFrame(rows)


def best_cutoff(table: pd.DataFrame) -> pd.Series:
    return table.loc[table["profit_per_applicant"].idxmax()]
