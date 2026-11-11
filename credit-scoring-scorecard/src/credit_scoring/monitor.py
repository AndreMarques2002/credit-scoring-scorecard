"""Monitoramento de drift: compara uma nova base com a referência do treino via PSI.

Uso:
    python -m credit_scoring.monitor --data data/raw/nova_safra.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from .data import clean, load_raw
from .psi import feature_psi, psi, psi_status


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="CSV com o mesmo esquema do treino")
    p.add_argument("--model", type=Path, default=Path("models/scorecard.joblib"))
    args = p.parse_args(argv)

    art = joblib.load(args.model)
    sc = art["scorecard"]
    new = clean(load_raw(args.data))
    X_new = new[sc.binner_.columns_]

    score_psi = psi(art["reference_scores"], sc.score(X_new))
    print(f"PSI do score: {score_psi:.4f} -> {psi_status(score_psi)}\n")
    print(feature_psi(art["reference_features"], X_new).round(4).to_string(index=False))
    print("\nRegra prática: PSI < 0,10 estável | 0,10–0,25 atenção | > 0,25 ação (reavaliar/recalibrar).")


if __name__ == "__main__":
    main()
