"""Script de experimentação comparativa ponta-a-ponta: PETR4 vs. ITUB4.

Executa o protocolo experimental completo:
1. Ingestão e alinhamento de cotações B3 e dados oficiais do BACEN.
2. Engenharia de variáveis estacionárias de desconto de ciclos e macroeconomia.
3. Rotulagem de excesso de retorno sobre o CDI em horizonte de 60 pregões.
4. Validação Walk-Forward temporal com 4 janelas expansivas e Purging de 60 barras.
5. Treinamento comparativo de LightGBM (Calibrado), XGBoost (Calibrado),
   Regressão Logística e Dummy Classifier.
6. Cálculo de métricas preditivas, Brier Score, simulação de backtest com taxas B3
   e geração de gráficos SHAP para o artigo IEEE.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.ticker_loader import TickerDataLoader
from src.evaluation.risk_evaluator import RiskEvaluator
from src.features.pipeline import FeaturePipeline
from src.labeling.tactical import TacticalLabeler
from src.models.engine import ModelEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configurações do experimento
TICKERS = ["PETR4", "ITUB4"]
START_DATE = "2015-01-01"
END_DATE = "2024-06-01"
HORIZON = 60
N_SPLITS = 4
FIGURES_DIR = "paper/figures"


def run_pipeline_for_ticker(ticker: str) -> Dict[str, Dict[str, float]]:
    """Executa o pipeline completo para um ticker da B3."""
    logger.info("==================================================")
    logger.info("INICIANDO EXPERIMENTO PARA O ATIVO: %s", ticker)
    logger.info("==================================================")

    # 1. Ingestão e alinhamento
    loader = TickerDataLoader(start_date=START_DATE, end_date=END_DATE)
    df_raw = loader.get_aligned_dataset(ticker)

    # 2. Engenharia de atributos
    feat_pipe = FeaturePipeline()
    df_feat = feat_pipe.compute_features(df_raw)

    # 3. Rotulagem tática
    labeler = TacticalLabeler(horizon=HORIZON)
    df_labeled = labeler.generate_tactical_labels(df_feat)

    # 4. Limpeza de warm-up (remoção de NaNs iniciais devido às médias de 500 dias)
    feature_cols = feat_pipe.get_feature_names()
    valid_mask = df_labeled[feature_cols].notna().all(axis=1) & df_labeled["target"].notna()
    df_clean = df_labeled[valid_mask].reset_index(drop=True)

    logger.info("Amostras limpas e rotuladas disponíveis para %s: %d", ticker, len(df_clean))

    X_all = df_clean[feature_cols].values
    y_all = df_clean["target"].values

    models = {
        "LightGBM (Calibrado)": ("lightgbm", True),
        "XGBoost (Calibrado)": ("xgboost", True),
        "Regressão Logística": ("logistic", False),
        "Dummy (Baseline)": ("dummy", False),
    }

    evaluator = RiskEvaluator(transaction_cost=0.0003)
    results_by_model: Dict[str, Dict[str, float]] = {}

    splits = list(labeler.get_walk_forward_splits(df_clean, n_splits=N_SPLITS, min_train_ratio=0.45))

    for model_name, (m_type, calibrate) in models.items():
        logger.info("Avaliando modelo: %s...", model_name)
        all_oos_y_true: List[float] = []
        all_oos_probs: List[float] = []
        all_oos_dfs: List[pd.DataFrame] = []

        last_engine = None
        last_X_test = None

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, y_train = X_all[train_idx], y_all[train_idx]
            X_test, y_test = X_all[test_idx], y_all[test_idx]

            engine = ModelEngine(model_type=m_type, calibrate=calibrate, seed=42)
            engine.fit(X_train, y_train)

            probs = engine.predict_proba(X_test)

            all_oos_y_true.extend(y_test)
            all_oos_probs.extend(probs)
            all_oos_dfs.append(df_clean.iloc[test_idx])

            if fold_idx == len(splits) - 1:
                last_engine = engine
                last_X_test = X_test

        y_true_arr = np.array(all_oos_y_true)
        probs_arr = np.array(all_oos_probs)

        metrics = evaluator.compute_predictive_metrics(y_true_arr, probs_arr, threshold=0.5)

        # Backtest fora-da-amostra
        df_oos_concat = pd.concat(all_oos_dfs).reset_index(drop=True)
        bt_res = evaluator.run_backtest(df_oos_concat, probs_arr, threshold=0.52)

        metrics["sharpe_ratio"] = bt_res["sharpe_ratio"]
        metrics["max_drawdown"] = bt_res["max_drawdown"]
        metrics["total_return_strat"] = bt_res["total_return_strat"]
        metrics["total_return_asset"] = bt_res["total_return_asset"]
        metrics["total_return_cdi"] = bt_res["total_return_cdi"]

        results_by_model[model_name] = metrics

        # Se for o LightGBM, gerar gráfico SHAP e gráfico de backtest
        if model_name == "LightGBM (Calibrado)" and last_engine is not None and last_X_test is not None:
            _generate_shap_plot(last_engine, last_X_test, feature_cols, ticker)
            _generate_backtest_plot(bt_res["backtest_df"], ticker)

    return results_by_model


def _generate_shap_plot(engine: ModelEngine, X: np.ndarray, feature_names: List[str], ticker: str) -> None:
    """Gera e salva o SHAP summary plot para o ativo."""
    import shap

    os.makedirs(FIGURES_DIR, exist_ok=True)
    shap_vals, _ = engine.explain_shap(X, feature_names)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X, feature_names=feature_names, show=False)
    plt.title(f"Importância de Atributos via SHAP — {ticker} (B3)", fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(FIGURES_DIR, f"shap_summary_{ticker}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info("Gráfico SHAP salvo em: %s", plot_path)


def _generate_backtest_plot(df_bt: pd.DataFrame, ticker: str) -> None:
    """Gera e salva o gráfico de evolução patrimonial do backtest."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.figure(figsize=(10, 5))
    dates = pd.to_datetime(df_bt["Date"])

    plt.plot(dates, df_bt["cum_strat"], label="Estratégia Tática (ML + CDI)", color="#1f77b4", linewidth=2.0)
    plt.plot(dates, df_bt["cum_asset"], label=f"Buy & Hold {ticker}", color="#ff7f0e", linestyle="--", alpha=0.8)
    plt.plot(dates, df_bt["cum_cdi"], label="Taxa CDI (Benchmark)", color="#2ca02c", linestyle=":", linewidth=1.8)

    plt.title(f"Simulação de Carteira Out-of-Sample — {ticker} (Walk-Forward com Purging)", fontsize=12)
    plt.xlabel("Data", fontsize=10)
    plt.ylabel("Retorno Acumulado (Base 1.0)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()

    plot_path = os.path.join(FIGURES_DIR, f"backtest_{ticker}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info("Gráfico de Backtest salvo em: %s", plot_path)


def main() -> None:
    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for ticker in TICKERS:
        res = run_pipeline_for_ticker(ticker)
        all_results[ticker] = res

    # Exibição resumida dos resultados
    print("\n=======================================================")
    print("RESUMO CONSOLIDADO DOS EXPERIMENTOS ACADÊMICOS (IEEE)")
    print("=======================================================")

    for ticker, models in all_results.items():
        print(f"\n--- ATIVO: {ticker} ---")
        df_res = pd.DataFrame(models).T
        print(df_res[["roc_auc", "pr_auc", "brier_score", "f1_macro", "sharpe_ratio", "max_drawdown"]].round(4))


if __name__ == "__main__":
    main()
