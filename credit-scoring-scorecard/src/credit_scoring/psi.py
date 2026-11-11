"""Population Stability Index (PSI) para monitorar drift de score e de variáveis."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PSI_ALERT, PSI_STABLE


def psi(expected, actual, bins: int = 10, eps: float = 1e-4) -> float:
    """PSI entre a distribuição de referência (`expected`) e a atual (`actual`).

    Faixas definidas pelos quantis da referência; NaN vira uma faixa própria.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    e_nan, a_nan = np.isnan(expected), np.isnan(actual)
    ev, av = expected[~e_nan], actual[~a_nan]
    if ev.size == 0 or av.size == 0:
        return 0.0

    inner = np.unique(np.quantile(ev, np.linspace(0, 1, bins + 1)[1:-1]))
    e_cnt = np.bincount(np.searchsorted(inner, ev, side="right"), minlength=len(inner) + 1)
    a_cnt = np.bincount(np.searchsorted(inner, av, side="right"), minlength=len(inner) + 1)
    e_cnt = np.append(e_cnt, e_nan.sum()).astype(float)
    a_cnt = np.append(a_cnt, a_nan.sum()).astype(float)

    e_pct = np.clip(e_cnt / e_cnt.sum(), eps, None)
    a_pct = np.clip(a_cnt / a_cnt.sum(), eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def psi_status(value: float) -> str:
    if value < PSI_STABLE:
        return "estável"
    if value < PSI_ALERT:
        return "atenção"
    return "ação"


def feature_psi(reference: pd.DataFrame, current: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    rows = [{"feature": c, "psi": psi(reference[c], current[c], bins)} for c in reference.columns]
    out = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    out["status"] = out["psi"].map(psi_status)
    return out
