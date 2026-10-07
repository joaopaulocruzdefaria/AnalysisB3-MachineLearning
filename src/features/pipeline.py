"""Módulo de engenharia de atributos e pipeline de transformações estacionárias.

Este módulo implementa a classe FeaturePipeline, responsável por calcular
as métricas quantitativas de desconto de ciclos, indicadores técnicos de momentum
e volatilidade, e covariáveis macroeconômicas oficiais do BACEN.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Pipeline para extração e engenharia de variáveis preditivas financeiras.

    Todas as variáveis são estritamente estacionárias e calculadas exclusivamente
    com dados passados conhecidos até o fechamento de cada pregão t.
    """

    def __init__(self) -> None:
        self.feature_names: List[str] = [
            "zscore_sma_200",
            "zscore_sma_500",
            "drawdown_52w",
            "bollinger_pct_b_long",
            "spread_asset_cdi_past60",
            "rsi_14",
            "stoch_k_14",
            "macd_normalized",
            "atr_ratio_14",
            "volume_zscore_20",
            "selic_annualized",
            "selic_delta_60",
            "usdbRL_return_60",
            "usdbRL_vol_20",
        ]

    def _compute_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calcula o RSI clássico de Wilder normalizado no intervalo [0, 1]."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi / 100.0

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcula o Average True Range (ATR) normalizado pelo preço de fechamento."""
        high = df["High"]
        low = df["Low"]
        close_prev = df["Close"].shift(1)

        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr / (df["Close"] + 1e-9)

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processa o DataFrame alinhado e calcula todas as features preditivas.

        Args:
            df: DataFrame contendo colunas Date, Open, High, Low, Close, Adj_Close,
                Volume, selic_over_diaria, dolar_ptax, cdi_daily_rate, cdi_daily_factor.

        Returns:
            DataFrame com as novas colunas de features adicionadas.
        """
        data = df.copy()
        close = data["Close"]

        logger.info("Calculando métricas de desconto histórico de ciclos...")
        # 1. Z-scores de médias móveis de longo prazo (200 e 500 pregões)
        sma_200 = close.rolling(200).mean()
        std_200 = close.rolling(200).std()
        data["zscore_sma_200"] = (close - sma_200) / (std_200 + 1e-9)

        sma_500 = close.rolling(500).mean()
        std_500 = close.rolling(500).std()
        data["zscore_sma_500"] = (close - sma_500) / (std_500 + 1e-9)

        # 2. Drawdown em relação à máxima de 52 semanas (252 pregões)
        rolling_max_52w = close.rolling(252).max()
        data["drawdown_52w"] = (close - rolling_max_52w) / (rolling_max_52w + 1e-9)

        # 3. Posição percentual (%b) no canal de Bandas de Bollinger de 200 períodos
        bb_upper = sma_200 + (2.0 * std_200)
        bb_lower = sma_200 - (2.0 * std_200)
        data["bollinger_pct_b_long"] = (close - bb_lower) / ((bb_upper - bb_lower) + 1e-9)

        # 4. Excesso de retorno passado do ativo frente ao CDI nos últimos 60 pregões
        asset_ret_past60 = close.pct_change(60)
        cdi_factor = data["cdi_daily_factor"]
        # Retorno acumulado composto do CDI nos últimos 60 pregões
        cdi_compounded_past60 = cdi_factor.rolling(60).apply(np.prod, raw=True) - 1.0
        data["spread_asset_cdi_past60"] = asset_ret_past60 - cdi_compounded_past60

        logger.info("Calculando indicadores de momentum e volatilidade...")
        # 5. RSI 14
        data["rsi_14"] = self._compute_rsi(close, period=14)

        # 6. Estocástico %K 14
        low_14 = data["Low"].rolling(14).min()
        high_14 = data["High"].rolling(14).max()
        data["stoch_k_14"] = (close - low_14) / ((high_14 - low_14) + 1e-9)

        # 7. MACD normalizado por preço
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        data["macd_normalized"] = (macd_line - signal_line) / (close + 1e-9)

        # 8. ATR ratio 14
        data["atr_ratio_14"] = self._compute_atr(data, period=14)

        # 9. Z-score de volume 20
        vol = data["Volume"]
        vol_sma20 = vol.rolling(20).mean()
        vol_std20 = vol.rolling(20).std()
        data["volume_zscore_20"] = (vol - vol_sma20) / (vol_std20 + 1e-9)

        logger.info("Calculando covariáveis macroeconômicas (BACEN)...")
        # 10. Selic anualizada (%)
        selic_daily = data["cdi_daily_rate"]
        data["selic_annualized"] = ((1.0 + selic_daily) ** 252) - 1.0

        # 11. Delta da Selic nos últimos 60 pregões
        data["selic_delta_60"] = data["selic_annualized"].pct_change(60)

        # 12. Retorno do Dólar PTAX nos últimos 60 pregões
        dolar = data["dolar_ptax"]
        data["usdbRL_return_60"] = dolar.pct_change(60)

        # 13. Volatilidade realizada do dólar em 20 pregões
        dolar_log_ret = np.log(dolar / dolar.shift(1))
        data["usdbRL_vol_20"] = dolar_log_ret.rolling(20).std() * np.sqrt(252)

        return data

    def get_feature_names(self) -> List[str]:
        """Retorna os nomes de todas as features preditivas geradas pelo pipeline."""
        return list(self.feature_names)
