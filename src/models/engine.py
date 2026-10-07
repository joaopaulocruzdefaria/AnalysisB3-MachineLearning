"""Módulo de modelagem preditiva, calibração probabilística e explicabilidade SHAP.

Este módulo implementa a classe ModelEngine, que encapsula os algoritmos de
Gradient Boosting (LightGBM, XGBoost) e baselines, aplicando calibração de probabilidades
(Platt Scaling / Isotonic) para suporte à decisão tática.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


class ModelEngine:
    """Motor unificado de modelagem preditiva e calibração para ações da B3.

    Attributes:
        model_type: Tipo do algoritmo ('lightgbm', 'xgboost', 'logistic', 'dummy').
        calibrate: Se True, aplica calibração probabilística (Platt Scaling).
        calibration_method: 'sigmoid' (Platt) ou 'isotonic'.
        seed: Semente pseudoaleatória para reprodutibilidade.
    """

    def __init__(
        self,
        model_type: str = "lightgbm",
        calibrate: bool = True,
        calibration_method: str = "sigmoid",
        seed: int = 42,
    ) -> None:
        self.model_type = model_type.lower()
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.seed = seed
        self.model: Optional[BaseEstimator] = None
        self.is_fitted: bool = False

    def _build_base_estimator(self) -> BaseEstimator:
        """Instancia o estimador base de acordo com a especificação."""
        if self.model_type == "lightgbm":
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                n_estimators=100,
                learning_rate=0.03,
                max_depth=4,
                num_leaves=15,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                random_state=self.seed,
                verbose=-1,
            )

        elif self.model_type == "xgboost":
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=100,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=self.seed,
            )

        elif self.model_type == "logistic":
            return Pipeline(
                [
                    ("scaler", RobustScaler()),
                    ("clf", LogisticRegression(penalty="l2", C=0.1, random_state=self.seed, max_iter=1000)),
                ]
            )

        elif self.model_type == "dummy":
            return DummyClassifier(strategy="prior", random_state=self.seed)

        else:
            raise ValueError(f"Tipo de modelo desconhecido: '{self.model_type}'. Opções: lightgbm, xgboost, logistic, dummy.")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> ModelEngine:
        """Treina o estimador e calibra as probabilidades preservando ordem temporal.

        Args:
            X_train: Matriz de features do conjunto de treino (já pós-purged).
            y_train: Vetor binário de rótulos target.

        Returns:
            self para encadeamento.
        """
        base_estimator = self._build_base_estimator()

        if not self.calibrate or self.model_type == "dummy":
            logger.info("Treinando modelo %s sem etapa de calibração adicional.", self.model_type)
            base_estimator.fit(X_train, y_train)
            self.model = base_estimator
        else:
            # Para manter estrita ordenação temporal sem vazamento na calibração,
            # reservamos os últimos 20% do treino como conjunto de calibração
            n_samples = len(X_train)
            split_idx = int(n_samples * 0.8)

            logger.info(
                "Ajustando estimador base %s em %d amostras e calibrando em %d amostras com split temporal...",
                self.model_type,
                split_idx,
                n_samples - split_idx,
            )

            temporal_cv = [(np.arange(split_idx), np.arange(split_idx, n_samples))]
            calibrator = CalibratedClassifierCV(
                estimator=base_estimator,
                method=self.calibration_method,
                cv=temporal_cv,
            )
            calibrator.fit(X_train, y_train)
            self.model = calibrator

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Gera probabilidades calibradas P(Y=1|X) para superação do CDI.

        Args:
            X: Matriz de atributos de teste.

        Returns:
            Vetor unidimensional contendo P(Y=1|X) para cada observação.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("O modelo precisa ser treinado com fit() antes de gerar predições.")

        proba = self.model.predict_proba(X)
        # Probabilidade da classe positiva (superar CDI, coluna 1)
        return proba[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Classifica em 0 ou 1 baseado em um limiar de corte de probabilidade."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def explain_shap(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Any]:
        """Calcula os valores SHAP para interpretabilidade via TreeSHAP.

        Args:
            X: Matriz de features para cálculo das contribuições.
            feature_names: Nomes legíveis das colunas.

        Returns:
            Tupla contendo (shap_values, explainer).
        """
        import shap

        if not self.is_fitted or self.model is None:
            raise RuntimeError("Modelo não treinado.")

        # Extrair estimador subjacente se estiver envelopado por CalibratedClassifierCV
        underlying = self.model
        if isinstance(self.model, CalibratedClassifierCV):
            underlying = self.model.calibrated_classifiers_[0].estimator

        if hasattr(underlying, "named_steps"):
            underlying = underlying.named_steps["clf"]

        explainer = shap.TreeExplainer(underlying)
        shap_values = explainer.shap_values(X)

        # Se for lista (duas classes em versões antigas de shap), selecionar classe positiva
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]

        return shap_values, explainer
