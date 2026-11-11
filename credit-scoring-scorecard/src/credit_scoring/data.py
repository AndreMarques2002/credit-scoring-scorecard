"""Carga, limpeza e geração de dados sintéticos."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FEATURES, PAST_DUE_COLUMNS, RANDOM_STATE, TARGET, TEST_SIZE


def load_raw(path: str | Path) -> pd.DataFrame:
    """Lê o cs-training.csv do Give Me Some Credit."""
    df = pd.read_csv(path)
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=unnamed)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no arquivo: {sorted(missing)}")
    return df[FEATURES + [TARGET]]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Regras determinísticas de limpeza (sem estatísticas aprendidas => sem vazamento).

    - remove idades inválidas (< 18);
    - trata 96/98 nas contagens de atraso como código especial => NaN
      (o binning cria uma faixa própria para "Missing").
    """
    out = df.copy()
    out = out[out["age"] >= 18]
    for col in PAST_DUE_COLUMNS:
        out.loc[out[col] >= 90, col] = np.nan
    return out.reset_index(drop=True)


def split(df: pd.DataFrame, seed: int = RANDOM_STATE):
    """Split estratificado treino/teste. O dataset não tem data, então não há split temporal."""
    X, y = df[FEATURES], df[TARGET].astype(int)
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=seed)


def make_synthetic(n: int = 30_000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Gera dados sintéticos com o mesmo esquema do Give Me Some Credit.

    Serve para testes, CI e demonstração. NÃO usar para reportar resultados.
    """
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(52, 14.5, n), 21, 100).round()
    util = np.where(rng.random(n) < 0.7, rng.beta(0.6, 3.0, n), rng.beta(2.0, 2.0, n) * 1.2)
    util_c = np.clip(util, 0, 1.5)

    late30 = rng.poisson(0.05 + 0.5 * util_c**2).astype(float)
    late60 = rng.poisson(0.02 + 0.25 * util_c**2).astype(float)
    late90 = rng.poisson(0.01 + 0.20 * util_c**2).astype(float)

    income = rng.lognormal(8.7, 0.6, n)
    no_income = rng.random(n) < 0.2
    income[no_income] = np.nan
    debt = rng.lognormal(-1.0, 1.0, n)
    debt[no_income] = rng.lognormal(6.0, 1.5, no_income.sum())

    open_lines = rng.poisson(8, n).astype(float)
    real_estate = rng.poisson(1.0, n).astype(float)
    dependents = rng.poisson(0.8, n).astype(float)
    dependents[rng.random(n) < 0.025] = np.nan

    z = (
        -3.9
        + 1.8 * util_c
        - 0.03 * (age - 52)
        + 0.45 * np.minimum(late30, 5)
        + 0.35 * np.minimum(late60, 5)
        + 0.70 * np.minimum(late90, 5)
        + 0.15 * np.minimum(np.nan_to_num(dependents), 4)
    )
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    df = pd.DataFrame(
        {
            "RevolvingUtilizationOfUnsecuredLines": util,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": late30,
            "DebtRatio": debt,
            "MonthlyIncome": income,
            "NumberOfOpenCreditLinesAndLoans": open_lines,
            "NumberOfTimes90DaysLate": late90,
            "NumberRealEstateLoansOrLines": real_estate,
            "NumberOfTime60-89DaysPastDueNotWorse": late60,
            "NumberOfDependents": dependents,
            "SeriousDlqin2yrs": y,
        }
    )
    # Imita as anomalias conhecidas do dataset real: códigos 96/98 e uma idade inválida.
    codes = rng.random(n) < 0.003
    df.loc[codes, "NumberOfTimes90DaysLate"] = 98.0
    df.loc[0, "age"] = 0.0
    return df
