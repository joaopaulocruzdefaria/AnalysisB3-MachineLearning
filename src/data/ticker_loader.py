"""Módulo de ingestão e carregamento de dados históricos da B3 e indicadores macroeconômicos.

Este módulo implementa a classe TickerDataLoader, responsável por coletar cotações
históricas (OHLCV) de qualquer ação da B3 e sincronizá-las com as séries oficiais
do Banco Central do Brasil (SGS/BACEN: Selic Over diária e Câmbio USD/BRL).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class TickerDataLoader:
    """Carregador modular e agnóstico de dados para ações da B3 e macroeconomia.

    Attributes:
        data_dir: Diretório raiz para armazenamento de dados brutos e processados.
        start_date: Data inicial padrão para coleta (YYYY-MM-DD).
        end_date: Data final padrão para coleta (YYYY-MM-DD).
    """

    def __init__(
        self,
        data_dir: str = "data",
        start_date: str = "2014-01-01",
        end_date: Optional[str] = None,
    ) -> None:
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.start_date = start_date
        self.end_date = end_date or datetime.today().strftime("%Y-%m-%d")

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def _format_ticker(self, ticker: str) -> str:
        """Assegura o sufixo .SA para ações brasileiras no Yahoo Finance."""
        ticker = ticker.strip().upper()
        if not ticker.endswith(".SA") and not ticker.startswith("^") and "=" not in ticker:
            return f"{ticker}.SA"
        return ticker

    def fetch_stock_ohlcv(
        self,
        ticker: str,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """Coleta série histórica diária (OHLCV) de um ativo via Yahoo Finance.

        Args:
            ticker: Código da ação (ex: 'PETR4', 'ITUB4' ou 'PETR4.SA').
            force_reload: Se True, ignora o cache local e refaz o download.

        Returns:
            DataFrame com colunas padronizadas: Date, Open, High, Low, Close, Adj_Close, Volume.
        """
        import yfinance as yf

        yf_ticker = self._format_ticker(ticker)
        clean_symbol = ticker.replace(".SA", "").replace("^", "").replace("=X", "")
        cached_file = os.path.join(self.raw_dir, f"{clean_symbol}_daily.csv")

        if os.path.exists(cached_file) and not force_reload:
            logger.info("Carregando cotações de %s do cache local: %s", clean_symbol, cached_file)
            df = pd.read_csv(cached_file, parse_dates=["Date"])
            return df

        logger.info("Baixando cotações de %s [%s a %s] via Yahoo Finance...", yf_ticker, self.start_date, self.end_date)
        df_yf = yf.download(yf_ticker, start=self.start_date, end=self.end_date, progress=False, auto_adjust=False)

        if df_yf.empty:
            raise ValueError(f"Nenhum dado encontrado para o ticker {yf_ticker}.")

        if isinstance(df_yf.columns, pd.MultiIndex):
            df_yf.columns = [col[0] for col in df_yf.columns]

        df_yf = df_yf.reset_index()

        rename_map = {
            "Date": "Date",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Adj Close": "Adj_Close",
            "Volume": "Volume",
        }
        df_clean = df_yf.rename(columns=rename_map)
        df_clean["Date"] = pd.to_datetime(df_clean["Date"])
        df_clean["Ticker"] = clean_symbol

        cols = ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj_Close", "Volume"]
        df_clean = df_clean[[c for c in cols if c in df_clean.columns]]

        df_clean.to_csv(cached_file, index=False)
        logger.info("Salvo cache de %s com %d registros em %s.", clean_symbol, len(df_clean), cached_file)
        return df_clean

    def fetch_bacen_sgs(
        self,
        series_code: int,
        series_name: str,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """Coleta série temporal oficial da API REST do Banco Central do Brasil (SGS).

        Args:
            series_code: Código numérico da série no SGS (ex: 11 para Selic, 1 para Dólar PTAX).
            series_name: Nome semântico para a coluna de valor (ex: 'selic_over_diaria').
            force_reload: Se True, refaz a requisição web ignorando o cache local.

        Returns:
            DataFrame com colunas ['Date', series_name].
        """
        cached_file = os.path.join(self.raw_dir, f"bacen_{series_code}_{series_name}.csv")

        if os.path.exists(cached_file) and not force_reload:
            logger.info("Carregando série SGS %d (%s) do cache local...", series_code, series_name)
            df = pd.read_csv(cached_file, parse_dates=["Date"])
            return df

        start_dt = datetime.strptime(self.start_date, "%Y-%m-%d").strftime("%d/%m/%Y")
        end_dt = datetime.strptime(self.end_date, "%Y-%m-%d").strftime("%d/%m/%Y")

        url = (
            f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_code}/dados"
            f"?formato=json&dataInicial={start_dt}&dataFinal={end_dt}"
        )
        logger.info("Consultando BACEN SGS %d (%s)...", series_code, series_name)
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        data_json = response.json()
        if not data_json:
            raise ValueError(f"Série SGS {series_code} retornou vazia para o período especificado.")

        df = pd.DataFrame(data_json)
        df["Date"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
        df[series_name] = pd.to_numeric(df["valor"], errors="coerce")
        df = df[["Date", series_name]].sort_values("Date").dropna()

        df.to_csv(cached_file, index=False)
        logger.info("Série BACEN SGS %d (%s) salva em %s (%d registros).", series_code, series_name, cached_file, len(df))
        return df

    def get_aligned_dataset(
        self,
        ticker: str,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """Cruza e alinha as cotações da ação com a taxa Selic e o Dólar PTAX.

        Args:
            ticker: Código da ação da B3 (ex: 'PETR4', 'ITUB4').
            force_reload: Se True, recarrega todas as fontes de dados.

        Returns:
            DataFrame perfeitamente alinhado contendo preços e séries macroeconômicas.
        """
        clean_symbol = ticker.replace(".SA", "").replace("^", "").replace("=X", "")
        processed_file = os.path.join(self.processed_dir, f"{clean_symbol}_aligned.csv")

        if os.path.exists(processed_file) and not force_reload:
            logger.info("Carregando dataset alinhado de %s do cache processado.", clean_symbol)
            return pd.read_csv(processed_file, parse_dates=["Date"])

        stock_df = self.fetch_stock_ohlcv(ticker, force_reload=force_reload)
        selic_df = self.fetch_bacen_sgs(11, "selic_over_diaria", force_reload=force_reload)
        ptax_df = self.fetch_bacen_sgs(1, "dolar_ptax", force_reload=force_reload)

        # Merge ordenado pela data dos pregões da ação
        aligned = pd.merge(stock_df, selic_df, on="Date", how="inner")
        aligned = pd.merge(aligned, ptax_df, on="Date", how="inner")
        aligned = aligned.sort_values("Date").reset_index(drop=True)

        # Fator diário do CDI: taxa Selic SGS 11 é dada em % a.d.
        aligned["cdi_daily_rate"] = aligned["selic_over_diaria"] / 100.0
        aligned["cdi_daily_factor"] = 1.0 + aligned["cdi_daily_rate"]

        aligned.to_csv(processed_file, index=False)
        logger.info("Dataset alinhado para %s salvo com %d pregões úteis.", clean_symbol, len(aligned))
        return aligned
