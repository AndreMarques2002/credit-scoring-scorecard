"""Pipeline de treino e avaliação.

Uso:
    python -m credit_scoring.train --data data/raw/cs-training.csv
    python -m credit_scoring.train --synthetic        # demonstração, NÃO reportar resultados
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from . import plots
from .business import best_cutoff, cutoff_analysis
from .challenger import fit_challenger, global_importance
from .config import RANDOM_STATE, TARGET, BusinessAssumptions, ScoreScale
from .data import clean, load_raw, make_synthetic, split
from .metrics import band_table, evaluate
from .psi import feature_psi, psi, psi_status
from .scorecard import Scorecard

log = logging.getLogger("credit_scoring")


def run(df: pd.DataFrame, out_dir: Path, model_dir: Path, seed: int = RANDOM_STATE,
        synthetic: bool = False) -> dict:
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = clean(df)
    X_tr, X_te, y_tr, y_te = split(df, seed)
    log.info("Treino: %d linhas (%.2f%% maus) | Teste: %d linhas (%.2f%% maus)",
             len(X_tr), 100 * y_tr.mean(), len(X_te), 100 * y_te.mean())

    # --- Scorecard (modelo principal) ---
    sc = Scorecard(scale=ScoreScale()).fit(X_tr, y_tr)
    p_sc_tr, p_sc_te = sc.predict_proba_bad(X_tr), sc.predict_proba_bad(X_te)
    s_tr, s_te = sc.score(X_tr), sc.score(X_te)

    # --- Challenger ---
    xgb_model = fit_challenger(X_tr, y_tr, seed)
    p_xgb_tr, p_xgb_te = xgb_model.predict_proba(X_tr)[:, 1], xgb_model.predict_proba(X_te)[:, 1]

    metrics = {
        "synthetic_data": synthetic,
        "scorecard": {"train": evaluate(y_tr, p_sc_tr), "test": evaluate(y_te, p_sc_te)},
        "xgboost": {"train": evaluate(y_tr, p_xgb_tr), "test": evaluate(y_te, p_xgb_te),
                    "best_iteration": int(xgb_model.best_iteration)},
        "selected_features": sc.features_,
        "dropped_features": sc.dropped_,
        "coefficients": sc.coef_.round(4).to_dict(),
    }

    # --- Faixas de score, PSI e política de corte ---
    bands = band_table(y_te, s_te)
    psi_score = psi(s_tr, s_te)
    fpsi = feature_psi(X_tr, X_te)
    cuts = cutoff_analysis(y_te, s_te, BusinessAssumptions())
    best = best_cutoff(cuts)
    metrics["psi_score_train_vs_test"] = {"psi": psi_score, "status": psi_status(psi_score)}
    metrics["best_cutoff"] = {k: float(v) for k, v in best.items()}
    metrics["monotonic_bands"] = bool(bands["bad_rate"].is_monotonic_decreasing)

    # --- Artefatos ---
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    sc.points_table().to_csv(out_dir / "scorecard_points.csv", index=False)
    sc.binner_.iv_table().to_csv(out_dir / "iv_table.csv", index=False)
    bands.to_csv(out_dir / "score_bands.csv", index=False)
    fpsi.to_csv(out_dir / "feature_psi.csv", index=False)
    cuts.to_csv(out_dir / "cutoff_analysis.csv", index=False)

    imp = global_importance(xgb_model, X_te)
    plots.plot_iv(sc.binner_.iv_, fig_dir / "iv_por_variavel.png")
    plots.plot_roc_ks(y_te, {"Scorecard": p_sc_te, "XGBoost": p_xgb_te}, fig_dir / "roc_ks.png")
    plots.plot_calibration(y_te, {"Scorecard": p_sc_te, "XGBoost": p_xgb_te}, fig_dir / "calibracao.png")
    plots.plot_band_bad_rate(bands, fig_dir / "taxa_maus_por_faixa.png")
    plots.plot_score_distribution(s_tr, s_te, psi_score, fig_dir / "distribuicao_score.png")
    plots.plot_profit_cutoff(cuts, best, fig_dir / "lucro_vs_corte.png")
    plots.plot_importance(imp, fig_dir / "xgb_importancia.png", "XGBoost: média do |efeito| por variável")

    joblib.dump({"scorecard": sc, "reference_scores": s_tr, "reference_features": X_tr.reset_index(drop=True)},
                model_dir / "scorecard.joblib")
    (out_dir / "results.md").write_text(render_results(metrics, bands, best, fpsi))
    return metrics


def render_results(m: dict, bands: pd.DataFrame, best: pd.Series, fpsi: pd.DataFrame) -> str:
    def row(name, d):
        return (f"| {name} | {d['auc']:.4f} | {d['gini']:.4f} | {d['ks']:.4f} | "
                f"{d['pr_auc']:.4f} | {d['brier']:.4f} |")

    head = "| Modelo / base | AUC | Gini | KS | PR-AUC | Brier |\n|---|---|---|---|---|---|"
    lines = []
    if m["synthetic_data"]:
        lines.append("> ⚠️ **Gerado com dados SINTÉTICOS (demonstração). Não use estes números como resultado.**\n")
    lines += [
        "## Desempenho\n", head,
        row("Scorecard – treino", m["scorecard"]["train"]),
        row("Scorecard – teste", m["scorecard"]["test"]),
        row("XGBoost – treino", m["xgboost"]["train"]),
        row("XGBoost – teste", m["xgboost"]["test"]),
        "\n## Faixas de score (teste)\n",
        bands.assign(bad_rate=(bands["bad_rate"] * 100).round(2),
                     cum_bad_capture=(bands["cum_bad_capture"] * 100).round(1),
                     lift=bands["lift"].round(2))
             .rename(columns={"bad_rate": "% maus", "cum_bad_capture": "% maus capturados (acum.)"})
             .to_markdown(index=False),
        f"\nTaxa de maus monotônica entre faixas: **{'sim' if m['monotonic_bands'] else 'não'}**",
        "\n## Estabilidade (PSI treino vs. teste)\n",
        f"PSI do score: **{m['psi_score_train_vs_test']['psi']:.4f}** "
        f"({m['psi_score_train_vs_test']['status']})\n",
        fpsi.round(4).to_markdown(index=False),
        "\n## Política de corte (premissas ilustrativas)\n",
        f"Corte de maior lucro: **{best['cutoff']:.0f} pontos** → aprovação de "
        f"**{best['approval_rate']:.1%}**, taxa de maus entre aprovados de "
        f"**{best['bad_rate_approved']:.2%}**.",
        "\n## Variáveis\n",
        f"Selecionadas: {', '.join(m['selected_features'])}",
        "\nDescartadas:\n" + "\n".join(f"- {k}: {v}" for k, v in m["dropped_features"].items())
        if m["dropped_features"] else "Nenhuma variável descartada.",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path, help="Caminho do cs-training.csv (Give Me Some Credit)")
    src.add_argument("--synthetic", action="store_true", help="Usa dados sintéticos (apenas demonstração)")
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    df = make_synthetic(seed=args.seed) if args.synthetic else load_raw(args.data)
    metrics = run(df, args.out_dir, args.model_dir, args.seed, synthetic=args.synthetic)

    t = metrics["scorecard"]["test"]
    x = metrics["xgboost"]["test"]
    print(f"\nScorecard (teste): AUC={t['auc']:.4f} Gini={t['gini']:.4f} KS={t['ks']:.4f}")
    print(f"XGBoost   (teste): AUC={x['auc']:.4f} Gini={x['gini']:.4f} KS={x['ks']:.4f}")
    print(f"Relatório em {args.out_dir}/results.md")


if __name__ == "__main__":
    main()
