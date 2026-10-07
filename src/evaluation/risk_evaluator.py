"""Módulo de avaliação estatística, calibração e simulação financeira de carteira.

Este módulo implementa a classe RiskEvaluator, que consolida métricas
de discriminação (ROC-AUC, PR-AUC), calibração (Brier Score, Log-Loss)
e realiza backtesting quantitativo auditado sob fricções de mercado.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class RiskEvaluator:
    """Avaliador unificado de desempenho estatístico e econômico.

    Attributes:
        transaction_cost: Custo percentual proporcional por troca de posição
                          (emolumentos + corretagem, padrão 0.03% = 0.0003).
    """

    def __init__(self, transaction_cost: float = 0.0003) -> None:
        self.transaction_cost = transaction_cost

    def compute_predictive_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Calcula métricas de discriminação e calibração estatística.

        Args:
            y_true: Rótulos reais binários (0 ou 1).
            y_prob: Probabilidades previstas calibradas P(Y=1|X).
            threshold: Limiar para métricas baseadas em corte rígido.

        Returns:
            Dicionário com ROC-AUC, PR-AUC, Brier Score, Log-Loss, F1 e Acurácia.
        """
        y_pred = (y_prob >= threshold).astype(int)

        metrics: Dict[str, float] = {}

        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = float("nan")

        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            metrics["pr_auc"] = float("nan")

        metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))

        # Evitar log(0) ou log(1) limitando probabilidades
        clipped_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
        metrics["log_loss"] = float(log_loss(y_true, clipped_prob))

        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

        return metrics

    def run_backtest(
        self,
        df_oos: pd.DataFrame,
        probabilities: np.ndarray,
        threshold: float = 0.52,
    ) -> Dict[str, Any]:
        """Simula a execução financeira tática fora-da-amostra (Out-of-Sample).

        Regra de Decisão:
            Se P(Y=1|X) >= threshold -> Aloca 100% no ativo.
            Se P(Y=1|X) < threshold  -> Aloca 100% no CDI pós-fixado (risco zero).

        Fricção:
            Dedução de transaction_cost sempre que a posição mudar (turnover).

        Args:
            df_oos: DataFrame fora-da-amostra com 'Close' e 'cdi_daily_rate'.
            probabilities: Série de probabilidades calibradas correspondente a df_oos.
            threshold: Limiar mínimo de convicção para compra do ativo.

        Returns:
            Dicionário com séries temporais de patrimônio e métricas de risco.
        """
        df = df_oos.copy().reset_index(drop=True)
        df["prob"] = probabilities

        # Retorno diário da ação e da taxa livre de risco
        df["asset_ret_daily"] = df["Close"].pct_change().fillna(0.0)
        df["cdi_ret_daily"] = df["cdi_daily_rate"].fillna(0.0)

        # Posição tática desejada no dia t
        df["target_position"] = (df["prob"] >= threshold).astype(int)

        # A execução da compra/venda ocorre no fechamento, gerando retorno a partir do dia seguinte
        df["actual_position"] = df["target_position"].shift(1).fillna(0)

        # Mudança de posição para cálculo de custos de transação
        position_change = (df["actual_position"] != df["actual_position"].shift(1).fillna(0)).astype(int)
        df["cost"] = position_change * self.transaction_cost

        # Retorno líquido diário da estratégia
        df["strat_ret_daily"] = (
            (df["actual_position"] * df["asset_ret_daily"])
            + ((1.0 - df["actual_position"]) * df["cdi_ret_daily"])
            - df["cost"]
        )

        # Curvas de patrimônio acumulado (evolução de R$ 1,00)
        df["cum_strat"] = (1.0 + df["strat_ret_daily"]).cumprod()
        df["cum_asset"] = (1.0 + df["asset_ret_daily"]).cumprod()
        df["cum_cdi"] = (1.0 + df["cdi_ret_daily"]).cumprod()

        # Métricas de risco anualizadas
        n_days = len(df)
        years = max(n_days / 252.0, 0.1)

        total_return_strat = float(df["cum_strat"].iloc[-1] - 1.0)
        total_return_asset = float(df["cum_asset"].iloc[-1] - 1.0)
        total_return_cdi = float(df["cum_cdi"].iloc[-1] - 1.0)

        # Excesso de retorno da estratégia sobre o CDI diário
        excess_ret_daily = df["strat_ret_daily"] - df["cdi_ret_daily"]
        sharpe_strat = (
            float((excess_ret_daily.mean() / (excess_ret_daily.std() + 1e-9)) * np.sqrt(252))
            if excess_ret_daily.std() > 0
            else 0.0
        )

        # Maximum Drawdown (MDD)
        rolling_max = df["cum_strat"].cummax()
        drawdown_series = (df["cum_strat"] - rolling_max) / rolling_max
        max_drawdown = float(drawdown_series.min())

        return {
            "total_return_strat": total_return_strat,
            "total_return_asset": total_return_asset,
            "total_return_cdi": total_return_cdi,
            "sharpe_ratio": sharpe_strat,
            "max_drawdown": max_drawdown,
            "backtest_df": df,
        }
