"""Módulo de rotulagem tática e gerador de validação temporal com Purging e Embargo.

Este módulo implementa a classe TacticalLabeler, responsável por rotular
a superação da taxa CDI em horizonte de 60 pregões e gerar dobras Walk-Forward
estritamente isentas de vazamento temporal (data leakage).
"""

from __future__ import annotations

import logging
from typing import Generator, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TacticalLabeler:
    """Rotulador de retornos táticos sobre a taxa livre de risco (CDI).

    Attributes:
        horizon: Horizonte preditivo em pregões úteis (padrão: 60 pregões / ~3 meses).
        embargo_days: Dias de carência pós-teste para amortecer persistência estocástica.
    """

    def __init__(self, horizon: int = 60, embargo_days: int = 15) -> None:
        self.horizon = horizon
        self.embargo_days = embargo_days

    def generate_tactical_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula o retorno do ativo, o CDI composto futuro e o alvo binário y.

        Args:
            df: DataFrame ordenado por data contendo 'Close' e 'cdi_daily_factor'.

        Returns:
            DataFrame com as colunas 'asset_return_h', 'cdi_compounded_h',
            'excess_return_h' e 'target' (1 se superou CDI, 0 caso contrário).
        """
        data = df.copy()
        close = data["Close"]
        cdi_factor = data["cdi_daily_factor"]
        h = self.horizon

        # 1. Retorno futuro do ativo no horizonte h: P(t+h) / P(t) - 1
        data["asset_return_h"] = (close.shift(-h) / close) - 1.0

        # 2. Retorno composto do CDI no horizonte futuro [t+1, t+h]
        # Usamos log para soma rolante reversa precisa
        log_cdi = np.log(cdi_factor)
        # rolling shift(-h)
        rolling_sum_log = log_cdi.iloc[::-1].rolling(window=h).sum().iloc[::-1]
        data["cdi_compounded_h"] = np.exp(rolling_sum_log.shift(-h)) - 1.0

        # 3. Excesso de retorno sobre o CDI
        data["excess_return_h"] = data["asset_return_h"] - data["cdi_compounded_h"]

        # 4. Alvo binário: 1 se excess_return > 0, 0 se <= 0 (NaN onde não há realização futura)
        valid_mask = data["excess_return_h"].notna()
        data["target"] = np.nan
        data.loc[valid_mask, "target"] = (data.loc[valid_mask, "excess_return_h"] > 0.0).astype(int)

        logger.info(
            "Rotulagem concluída: %d amostras rotuladas no horizonte de %d pregões.",
            valid_mask.sum(),
            h,
        )
        return data

    def get_walk_forward_splits(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        min_train_ratio: float = 0.4,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Gera divisões temporais expansivas com Purging obrigatório.

        Args:
            df: DataFrame com amostras rotuladas (target notna).
            n_splits: Número de dobras temporais fora-da-amostra.
            min_train_ratio: Proporção mínima inicial do dataset para o primeiro treino.

        Yields:
            Tuplas (train_indices, test_indices) com indexação posicional de inteiros.
        """
        # Apenas amostras que possuem target realizado
        labeled_indices = np.where(df["target"].notna())[0]
        n_samples = len(labeled_indices)

        min_train_samples = int(n_samples * min_train_ratio)
        remaining_samples = n_samples - min_train_samples
        test_size = remaining_samples // n_splits

        for fold in range(n_splits):
            test_start = min_train_samples + (fold * test_size)
            test_end = test_start + test_size if fold < n_splits - 1 else n_samples

            # Fim do treino antes do purge
            raw_train_end = test_start

            # PURGING: Elimina do conjunto de treino qualquer observação cujo horizonte
            # preditivo alcance o início do conjunto de teste (i + horizon >= test_start)
            purged_train_end = max(0, raw_train_end - self.horizon)

            train_idx = labeled_indices[:purged_train_end]
            test_idx = labeled_indices[test_start:test_end]

            logger.info(
                "Fold %d/%d: Treino Purged [%d amostras] | Teste OOS [%d amostras] | Purged: %d barras.",
                fold + 1,
                n_splits,
                len(train_idx),
                len(test_idx),
                self.horizon,
            )
            yield train_idx, test_idx
