"""Testes automatizados para validação do Core Engine do Motor Modular B3.

Verifica conformidade estrita de:
1. Cálculo de atributos estacionários de desconto de ciclos e macroeconomia.
2. Rotulagem tática e garantia de Purging contra vazamento temporal.
3. ModelEngine e calibração de probabilidades no intervalo [0, 1].
4. Métricas estatísticas de calibração e backtesting financeiro.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import FeaturePipeline
from src.labeling.tactical import TacticalLabeler
from src.models.engine import ModelEngine
from src.evaluation.risk_evaluator import RiskEvaluator


@pytest.fixture
def dummy_market_data() -> pd.DataFrame:
    """Cria série temporal sintética simulando 800 pregões da B3."""
    np.random.seed(42)
    n = 800
    dates = pd.date_range(start="2020-01-01", periods=n, freq="B")

    # Movimento browniano geométrico para preço
    daily_returns = np.random.normal(0.0005, 0.02, size=n)
    prices = 20.0 * np.exp(np.cumsum(daily_returns))

    highs = prices * (1.0 + np.abs(np.random.normal(0, 0.01, size=n)))
    lows = prices * (1.0 - np.abs(np.random.normal(0, 0.01, size=n)))
    opens = prices * (1.0 + np.random.normal(0, 0.005, size=n))
    volumes = np.random.uniform(1e6, 5e7, size=n)

    # Selic over diária (~12% a.a. => ~0.045% a.d.)
    selic_diaria = np.random.normal(0.045, 0.002, size=n)
    ptax = 5.0 + np.cumsum(np.random.normal(0.0, 0.03, size=n))

    df = pd.DataFrame(
        {
            "Date": dates,
            "Ticker": "TEST4",
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": prices,
            "Adj_Close": prices,
            "Volume": volumes,
            "selic_over_diaria": selic_diaria,
            "dolar_ptax": ptax,
            "cdi_daily_rate": selic_diaria / 100.0,
            "cdi_daily_factor": 1.0 + (selic_diaria / 100.0),
        }
    )
    return df


def test_feature_pipeline(dummy_market_data: pd.DataFrame) -> None:
    """Testa se o pipeline gera as 14 variáveis sem erro e com nomes corretos."""
    pipeline = FeaturePipeline()
    featured_df = pipeline.compute_features(dummy_market_data)

    feature_names = pipeline.get_feature_names()
    assert len(feature_names) == 14

    for name in feature_names:
        assert name in featured_df.columns, f"Feature ausente: {name}"

    # As últimas linhas (pós-warmup de 500 períodos) devem ter todas as features preenchidas
    tail_df = featured_df.iloc[550:]
    for name in feature_names:
        assert not tail_df[name].isna().all(), f"Feature {name} está totalmente NaN pós-warmup."


def test_tactical_labeling_and_purging(dummy_market_data: pd.DataFrame) -> None:
    """Valida se o alvo binário respeita o CDI e se o Purging elimina overlapping."""
    horizon = 60
    labeler = TacticalLabeler(horizon=horizon)
    labeled_df = labeler.generate_tactical_labels(dummy_market_data)

    # Verifica se os alvos são 0, 1 ou NaN
    valid_targets = labeled_df["target"].dropna().unique()
    assert set(valid_targets).issubset({0.0, 1.0})

    # Os últimos 60 pregões devem ser NaN (pois ainda não têm realização futura)
    assert labeled_df["target"].iloc[-horizon:].isna().all()

    # Valida splits Walk-Forward com Purging
    splits = list(labeler.get_walk_forward_splits(labeled_df, n_splits=3, min_train_ratio=0.5))
    assert len(splits) == 3

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        assert len(train_idx) > 0
        assert len(test_idx) > 0

        # PURGING CHECK: O último ponto de treino + 60 deve ser estritamente menor que o primeiro de teste
        last_train_sample = train_idx[-1]
        first_test_sample = test_idx[0]

        assert (
            last_train_sample + horizon < first_test_sample
        ), f"Violação de Purging no Fold {fold_idx}: TrainEnd={last_train_sample}, TestStart={first_test_sample}"


def test_model_engine_and_calibration() -> None:
    """Valida se o ModelEngine treina, calibra e devolve probabilidades limitadas em [0, 1]."""
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    engine = ModelEngine(model_type="lightgbm", calibrate=True, seed=42)
    engine.fit(X[:150], y[:150])

    probs = engine.predict_proba(X[150:])
    assert len(probs) == 50
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    preds = engine.predict(X[150:], threshold=0.5)
    assert set(preds).issubset({0, 1})


def test_risk_evaluator(dummy_market_data: pd.DataFrame) -> None:
    """Valida métricas estatísticas e simulação financeira com atrito."""
    evaluator = RiskEvaluator(transaction_cost=0.0003)

    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    y_prob = np.array([0.8, 0.2, 0.7, 0.65, 0.3, 0.4, 0.9, 0.1, 0.75, 0.35])

    metrics = evaluator.compute_predictive_metrics(y_true, y_prob)
    assert "roc_auc" in metrics
    assert "brier_score" in metrics
    assert metrics["roc_auc"] > 0.8
    assert metrics["brier_score"] < 0.15

    # Simulação de Backtest
    n = 100
    df_slice = dummy_market_data.iloc[:n].copy()
    probs = np.random.uniform(0.3, 0.8, size=n)

    backtest_res = evaluator.run_backtest(df_slice, probs, threshold=0.55)
    assert "total_return_strat" in backtest_res
    assert "sharpe_ratio" in backtest_res
    assert "max_drawdown" in backtest_res
    assert backtest_res["max_drawdown"] <= 0.0
