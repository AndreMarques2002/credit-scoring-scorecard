"""Binning monotônico com WoE (Weight of Evidence) e IV (Information Value).

Convenção: WoE = ln(%bons / %maus). WoE maior => menor risco.
Valores ausentes ganham uma faixa própria ("Missing").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin

SMOOTHING = 0.5  # evita ln(0) em faixas sem bons ou sem maus


class WoEBinner(BaseEstimator, TransformerMixin):
    """Discretiza cada variável em faixas e as converte em WoE.

    Passos por variável:
      1. pré-binning por quantis (`pre_bins` faixas);
      2. fusão iterativa de faixas adjacentes até: (a) nenhuma faixa menor que
         `min_bin_frac` da base, (b) taxa de maus monotônica (se `monotonic`),
         (c) no máximo `max_bins` faixas;
      3. cálculo de WoE e IV por faixa.
    """

    def __init__(self, max_bins: int = 6, min_bin_frac: float = 0.05,
                 pre_bins: int = 20, monotonic: bool = True):
        self.max_bins = max_bins
        self.min_bin_frac = min_bin_frac
        self.pre_bins = pre_bins
        self.monotonic = monotonic

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y, dtype=int)
        self.columns_ = list(X.columns)
        self.bins_: dict[str, dict] = {}
        for col in self.columns_:
            self.bins_[col] = self._fit_column(X[col].to_numpy(dtype=float), y)
        self.iv_ = pd.Series({c: self.bins_[c]["iv"] for c in self.columns_}).sort_values(ascending=False)
        return self

    def _fit_column(self, x: np.ndarray, y: np.ndarray) -> dict:
        n_total = len(y)
        miss = np.isnan(x)
        xv, yv = x[~miss], y[~miss]

        edges = self._initial_edges(xv)
        idx = np.searchsorted(edges, xv, side="left")
        k = len(edges) + 1
        n = np.bincount(idx, minlength=k).astype(float)
        bad = np.bincount(idx, weights=yv, minlength=k).astype(float)

        direction = self._direction(xv, yv)
        edges, n, bad = self._merge(edges, n, bad, direction, n_total)

        good = n - bad
        g_tot, b_tot = float((y == 0).sum()), float((y == 1).sum())

        def woe(g, b):
            return np.log(((g + SMOOTHING) / g_tot) / ((b + SMOOTHING) / b_tot))

        woe_bins = woe(good, bad)
        iv_bins = (good / g_tot - bad / b_tot) * woe_bins

        miss_n = int(miss.sum())
        miss_bad = float(y[miss].sum())
        miss_good = miss_n - miss_bad
        if miss_n > 0:
            woe_miss = float(woe(miss_good, miss_bad))
            iv_miss = float((miss_good / g_tot - miss_bad / b_tot) * woe_miss)
        else:
            woe_miss, iv_miss = 0.0, 0.0

        labels = self._labels(edges)
        rows = [
            {"bin": lab, "n": int(m), "pct": m / n_total, "bad": int(b),
             "bad_rate": (b / m if m else np.nan), "woe": float(w), "iv": float(i)}
            for lab, m, b, w, i in zip(labels, n, bad, woe_bins, iv_bins)
        ]
        if miss_n > 0:
            rows.append({"bin": "Missing", "n": miss_n, "pct": miss_n / n_total, "bad": int(miss_bad),
                         "bad_rate": miss_bad / miss_n, "woe": woe_miss, "iv": iv_miss})
        table = pd.DataFrame(rows)
        return {"edges": edges, "woe": woe_bins, "woe_missing": woe_miss,
                "iv": float(iv_bins.sum() + iv_miss), "table": table}

    def _initial_edges(self, xv: np.ndarray) -> np.ndarray:
        if xv.size == 0:
            return np.array([])
        qs = np.linspace(0, 1, self.pre_bins + 1)[1:-1]
        edges = np.unique(np.quantile(xv, qs))
        return edges[edges < xv.max()]

    @staticmethod
    def _direction(xv: np.ndarray, yv: np.ndarray) -> int:
        if xv.size < 3 or np.unique(xv).size < 2 or np.unique(yv).size < 2:
            return 0
        corr = spearmanr(xv, yv).correlation
        if corr is None or np.isnan(corr):
            return 0
        return 1 if corr >= 0 else -1

    def _merge(self, edges, n, bad, direction, n_total):
        edges, n, bad = list(edges), list(n), list(bad)
        min_n = self.min_bin_frac * n_total
        while len(n) > 1:
            rate = [b / m if m > 0 else 0.0 for b, m in zip(bad, n)]
            small = [i for i, m in enumerate(n) if m < min_n]
            if small:
                i = min(small, key=lambda j: n[j])
                cands = [j for j in (i - 1, i + 1) if 0 <= j < len(n)]
                j = min(cands, key=lambda j: abs(rate[j] - rate[i]))
                lo = min(i, j)
            elif self.monotonic and direction != 0 and any(
                direction * (rate[i + 1] - rate[i]) < 0 for i in range(len(n) - 1)
            ):
                lo = next(i for i in range(len(n) - 1) if direction * (rate[i + 1] - rate[i]) < 0)
            elif len(n) > self.max_bins:
                lo = int(np.argmin([abs(rate[i + 1] - rate[i]) for i in range(len(n) - 1)]))
            else:
                break
            n[lo] += n[lo + 1]
            bad[lo] += bad[lo + 1]
            del n[lo + 1], bad[lo + 1], edges[lo]
        return np.array(edges), np.array(n), np.array(bad)

    @staticmethod
    def _labels(edges: np.ndarray) -> list[str]:
        if len(edges) == 0:
            return ["(-inf, inf)"]
        fmt = lambda v: f"{v:.4g}"  # noqa: E731
        labels = [f"(-inf, {fmt(edges[0])}]"]
        labels += [f"({fmt(a)}, {fmt(b)}]" for a, b in zip(edges[:-1], edges[1:])]
        labels.append(f"({fmt(edges[-1])}, inf)")
        return labels

    # ------------------------------------------------------------ transform
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in self.columns_:
            info = self.bins_[col]
            x = X[col].to_numpy(dtype=float)
            miss = np.isnan(x)
            idx = np.searchsorted(info["edges"], np.where(miss, 0.0, x), side="left")
            w = info["woe"][idx]
            out[col] = np.where(miss, info["woe_missing"], w)
        return pd.DataFrame(out, index=X.index)

    def iv_table(self) -> pd.DataFrame:
        """Todas as faixas de todas as variáveis, para auditoria."""
        frames = [t.assign(feature=c) for c, t in ((c, self.bins_[c]["table"]) for c in self.columns_)]
        return pd.concat(frames, ignore_index=True)[["feature", "bin", "n", "pct", "bad", "bad_rate", "woe", "iv"]]
