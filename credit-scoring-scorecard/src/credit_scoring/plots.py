"""Gráficos salvos em reports/figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .metrics import ks_statistic  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_iv(iv: pd.Series, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    iv.sort_values().plot.barh(ax=ax, color="#2b6cb0")
    ax.axvline(0.02, ls="--", c="gray", lw=1)
    ax.set_title("Information Value por variável (linha: corte 0,02)")
    ax.set_xlabel("IV")
    _save(fig, path)


def plot_roc_ks(y, probs: dict[str, np.ndarray], path: Path):
    y = np.asarray(y)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, label=name)
    axes[0].plot([0, 1], [0, 1], "--", c="gray", lw=1)
    axes[0].set(title="Curva ROC (teste)", xlabel="FPR", ylabel="TPR")
    axes[0].legend()

    name, p = next(iter(probs.items()))
    order = np.argsort(-p)
    ys = y[order]
    x = np.arange(1, len(ys) + 1) / len(ys)
    cum_bad = np.cumsum(ys) / ys.sum()
    cum_good = np.cumsum(1 - ys) / (1 - ys).sum()
    axes[1].plot(x, cum_bad, label="% acumulado de maus")
    axes[1].plot(x, cum_good, label="% acumulado de bons")
    k = np.argmax(cum_bad - cum_good)
    axes[1].vlines(x[k], cum_good[k], cum_bad[k], colors="k", ls=":")
    axes[1].set(title=f"KS – {name} (KS = {ks_statistic(y, p):.3f})",
                xlabel="População ordenada do maior ao menor risco")
    axes[1].legend()
    _save(fig, path)


def plot_calibration(y, probs: dict[str, np.ndarray], path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, p in probs.items():
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=name)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "--", c="gray", lw=1)
    ax.set(title="Calibração (teste)", xlabel="PD prevista", ylabel="Taxa observada de maus")
    ax.legend()
    _save(fig, path)


def plot_band_bad_rate(table: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(table["band"].astype(str), table["bad_rate"] * 100, color="#c05621")
    ax.set(title="Taxa de maus por faixa de score (1 = pior score)", xlabel="Faixa", ylabel="% maus")
    _save(fig, path)


def plot_score_distribution(train_scores, test_scores, psi_value: float, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(min(train_scores.min(), test_scores.min()),
                       max(train_scores.max(), test_scores.max()), 40)
    ax.hist(train_scores, bins=bins, alpha=0.55, density=True, label="treino")
    ax.hist(test_scores, bins=bins, alpha=0.55, density=True, label="teste")
    ax.set(title=f"Distribuição do score (PSI treino vs. teste = {psi_value:.4f})", xlabel="Score")
    ax.legend()
    _save(fig, path)


def plot_profit_cutoff(table: pd.DataFrame, best: pd.Series, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(table["cutoff"], table["profit_per_applicant"], c="#2f855a")
    ax.axvline(best["cutoff"], ls="--", c="k", lw=1)
    ax.set(title="Lucro por proponente vs. corte (premissas ilustrativas)",
           xlabel="Corte de score (aprova se score ≥ corte)", ylabel="Lucro por proponente (u.r.)")
    _save(fig, path)


def plot_importance(series: pd.Series, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    series.sort_values().plot.barh(ax=ax, color="#6b46c1")
    ax.set_title(title)
    _save(fig, path)
