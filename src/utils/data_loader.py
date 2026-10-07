"""Módulo de ingestão e carregamento de dados históricos da B3 e indicadores macroeconômicos.

Este módulo realiza o download de séries temporais diárias (OHLCV) de ações
negociadas na bolsa brasileira (B3) e benchmarks (Ibovespa e USD/BRL) via
Yahoo Finance e fontes públicas oficiais, armazenando os dados brutos de forma
reprodutível em data/raw/.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Lista padrão das ações mais líquidas do Ibovespa (B3)
DEFAULT_B3_TICKERS: List[str] = [
    "PETR4.SA",  # Petrobras PN
    "VALE3.SA",  # Vale ON
    "ITUB4.SA",  # Itaú Unibanco PN
    "BBDC4.SA",  # Bradesco PN
    "BBAS3.SA",  # Banco do Brasil ON
    "WEGE3.SA",  # WEG ON
    "ABEV3.SA",  # Ambev ON
    "RENT3.SA",  # Localiza ON
    "PRIO3.SA",  # PetroRio ON
    "MGLU3.SA",  # Magazine Luiza ON
]

# Benchmarks de mercado e câmbio
BENCHMARK_TICKERS: Dict[str, str] = {
    "^BVSP": "IBOVESPA",
    "USDBRL=X": "USD_BRL",
}


@dataclass(frozen=True)
class TickerDataConfig:
    """Configuração para requisição de série temporal de um ativo.

    Attributes:
        ticker: Símbolo do ativo (ex: 'PETR4.SA' ou '^BVSP').
        start_date: Data inicial no formato 'YYYY-MM-DD'.
        end_date: Data final no formato 'YYYY-MM-DD'.
        output_dir: Diretório de destino do arquivo salvo.
    """

    ticker: str
    start_date: str = "2015-01-01"
    end_date: str = "2024-06-01"
    output_dir: str = "data/raw"


def download_b3_ticker(
    ticker: str,
    start_date: str = "2015-01-01",
    end_date: str = "2024-06-01",
    output_dir: str = "data/raw",
    force: bool = False,
) -> Optional[str]:
    """Baixa o histórico diário (OHLCV) de uma ação da B3 e salva em CSV.

    Args:
        ticker: Símbolo do ticker no formato Yahoo Finance (ex: 'PETR4.SA').
        start_date: Data de início das cotações.
        end_date: Data de fim das cotações.
        output_dir: Diretório de destino.
        force: Se True, sobrescreve arquivo local existente.

    Returns:
        Caminho do arquivo CSV salvo ou None em caso de falha.
    """
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError as err:
        logger.error("Dependências ausentes (yfinance/pandas). Ative o ambiente virtual: %s", err)
        return None

    os.makedirs(output_dir, exist_ok=True)
    clean_symbol = ticker.replace(".SA", "").replace("^", "").replace("=X", "")
    destination_path = os.path.join(output_dir, f"{clean_symbol}_daily.csv")

    if os.path.exists(destination_path) and not force:
        logger.info("Arquivo já disponível em cache local: %s", destination_path)
        return destination_path

    logger.info("Baixando histórico de %s [%s a %s]...", ticker, start_date, end_date)
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if df.empty:
            logger.warning("Nenhum dado retornado para o ticker %s.", ticker)
            return None

        # Achatar MultiIndex se gerado pelo yfinance moderno
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        df = df.reset_index()
        # Normalização de nomes de colunas
        rename_map = {
            "Date": "Date",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Adj Close": "Adj_Close",
            "Volume": "Volume",
        }
        df = df.rename(columns=rename_map)
        df["Ticker"] = clean_symbol

        df.to_csv(destination_path, index=False)
        logger.info("Ticker %s salvo com sucesso em %s (%d registros).", ticker, destination_path, len(df))
        return destination_path
    except Exception as exc:
        logger.error("Erro inesperado ao baixar ticker %s: %s", ticker, exc)
        return None


def fetch_b3_dataset(
    tickers: Optional[List[str]] = None,
    include_benchmarks: bool = True,
    start_date: str = "2015-01-01",
    end_date: str = "2024-06-01",
    output_dir: str = "data/raw",
) -> List[str]:
    """Baixa um conjunto completo de ações da B3 e benchmarks.

    Args:
        tickers: Lista de tickers das ações. Se None, usa DEFAULT_B3_TICKERS.
        include_benchmarks: Se True, inclui Ibovespa (^BVSP) e USD/BRL (USDBRL=X).
        start_date: Data de início das cotações.
        end_date: Data de término das cotações.
        output_dir: Diretório de destino para os arquivos CSV.

    Returns:
        Lista de caminhos de arquivos CSV baixados com sucesso.
    """
    if tickers is None:
        tickers = DEFAULT_B3_TICKERS

    all_symbols = list(tickers)
    if include_benchmarks:
        all_symbols.extend(BENCHMARK_TICKERS.keys())

    downloaded_paths: List[str] = []
    for symbol in all_symbols:
        path = download_b3_ticker(symbol, start_date=start_date, end_date=end_date, output_dir=output_dir)
        if path:
            downloaded_paths.append(path)

    logger.info("Download concluído para %d de %d ativos.", len(downloaded_paths), len(all_symbols))
    return downloaded_paths


if __name__ == "__main__":
    fetch_b3_dataset()
