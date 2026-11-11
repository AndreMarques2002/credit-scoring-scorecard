"""Scorecard clássico: WoE + regressão logística + conversão em pontos."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .config import ScoreScale
from .woe import WoEBinner

log = logging.getLogger(__name__)


class Scorecard:
    """Pipeline completo: binning WoE -> seleção por IV/correlação -> logística -> pontos."""

    def __init__(self, scale: ScoreScale | None = None, min_iv: float = 0.02,
                 max_corr: float = 0.80, C: float = 1.0, max_bins: int = 6):
        self.scale = scale or ScoreScale()
        self.min_iv = min_iv
        self.max_corr = max_corr
        self.C = C
        self.max_bins = max_bins

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y) -> "Scorecard":
        y = np.asarray(y, dtype=int)
        self.binner_ = WoEBinner(max_bins=self.max_bins).fit(X, y)
        W = self.binner_.transform(X)

        self.features_, self.dropped_ = self._select(W)
        self.model_ = LogisticRegression(C=self.C, max_iter=1000).fit(W[self.features_], y)
        self.coef_ = pd.Series(self.model_.coef_[0], index=self.features_)
        self.intercept_ = float(self.model_.intercept_[0])

        # Com WoE = ln(bons/maus), o esperado é coeficiente NEGATIVO em P(mau).
        wrong = self.coef_[self.coef_ > 0]
        if len(wrong):
            log.warning("Coeficientes com sinal inesperado (possível multicolinearidade): %s",
                        list(wrong.index))
        return self

    def _select(self, W: pd.DataFrame):
        iv = self.binner_.iv_
        dropped = {c: f"IV {v:.3f} < {self.min_iv}" for c, v in iv.items() if v < self.min_iv}
        selected: list[str] = []
        for c in iv.index:
            if c in dropped:
                continue
            corr = max((abs(W[c].corr(W[s])) for s in selected), default=0.0)
            if corr > self.max_corr:
                dropped[c] = f"correlação {corr:.2f} > {self.max_corr} com variável de maior IV"
            else:
                selected.append(c)
        if not selected:
            raise ValueError("Nenhuma variável passou nos filtros de IV/correlação.")
        return selected, dropped

    # ------------------------------------------------------------- predict
    @property
    def factor_(self) -> float:
        return self.scale.pdo / np.log(2)

    @property
    def offset_(self) -> float:
        return self.scale.base_score - self.factor_ * np.log(self.scale.base_odds)

    def _woe(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.binner_.transform(X)[self.features_]

    def predict_proba_bad(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self._woe(X))[:, 1]

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Score = offset + factor * ln(odds de bom). Maior score => menor risco."""
        logit = self._woe(X).to_numpy() @ self.coef_.to_numpy() + self.intercept_
        return self.offset_ + self.factor_ * (-logit)

    # -------------------------------------------------------------- tables
    def points_table(self) -> pd.DataFrame:
        """Pontos por faixa de cada variável (a soma dos pontos = score)."""
        k = len(self.features_)
        base = (self.offset_ - self.factor_ * self.intercept_) / k
        rows = []
        for feat in self.features_:
            t = self.binner_.bins_[feat]["table"].copy()
            t["points"] = base - self.factor_ * self.coef_[feat] * t["woe"]
            t.insert(0, "feature", feat)
            rows.append(t)
        out = pd.concat(rows, ignore_index=True)
        out["points"] = out["points"].round(1)
        return out[["feature", "bin", "n", "pct", "bad_rate", "woe", "points"]]
