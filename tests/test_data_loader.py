"""Testes unitários para o módulo de carregamento de dados da B3."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.utils.data_loader import DEFAULT_B3_TICKERS, BENCHMARK_TICKERS, download_b3_ticker


class TestB3DataLoader(unittest.TestCase):
    """Testes para o utilitário data_loader da B3."""

    def test_default_tickers_config(self):
        """Garante que as ações mais líquidas da B3 e benchmarks estão configurados."""
        self.assertIn("PETR4.SA", DEFAULT_B3_TICKERS)
        self.assertIn("VALE3.SA", DEFAULT_B3_TICKERS)
        self.assertIn("ITUB4.SA", DEFAULT_B3_TICKERS)
        self.assertIn("^BVSP", BENCHMARK_TICKERS)
        self.assertIn("USDBRL=X", BENCHMARK_TICKERS)

    def test_download_b3_ticker_cached(self):
        """Verifica se o loader detecta arquivo já existente e não refaz download."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "PETR4_daily.csv")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("Date,Open,High,Low,Close,Adj_Close,Volume,Ticker\n2023-01-02,20,21,19,20.5,20.5,1000,PETR4\n")

            result = download_b3_ticker("PETR4.SA", output_dir=tmp_dir, force=False)
            self.assertEqual(result, file_path)

    @patch("yfinance.download")
    def test_download_b3_ticker_mock(self, mock_yf_download):
        """Testa o fluxo de download mockando o yfinance."""
        import pandas as pd

        fake_df = pd.DataFrame(
            {
                "Open": [30.0, 31.0],
                "High": [32.0, 33.0],
                "Low": [29.0, 30.0],
                "Close": [31.5, 32.5],
                "Adj Close": [31.5, 32.5],
                "Volume": [1000000, 1200000],
            },
            index=pd.date_range("2023-01-01", periods=2, freq="D", name="Date"),
        )
        mock_yf_download.return_value = fake_df

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = download_b3_ticker("ITUB4.SA", output_dir=tmp_dir, force=True)
            self.assertIsNotNone(result)
            self.assertTrue(os.path.exists(result))

            saved_df = pd.read_csv(result)
            self.assertEqual(len(saved_df), 2)
            self.assertIn("ITUB4", saved_df["Ticker"].values)


if __name__ == "__main__":
    unittest.main()
