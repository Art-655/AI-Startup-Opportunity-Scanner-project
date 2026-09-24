"""XGBoost-based startup success prediction with SHAP explanations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class SuccessPredictor:
    """Train, persist, and explain an XGBoost binary success classifier."""

    def __init__(
        self,
        model_path: PathLike = "models/success_predictor.joblib",
        reports_path: PathLike = "reports/shap_summary.png",
        **model_parameters: Any,
    ) -> None:
        """Initialize a predictor configuration.

        Args:
            model_path: Location used to save and load the trained model.
            reports_path: Location used for the SHAP summary plot.
            **model_parameters: Optional XGBClassifier parameters.
        """
        self.model_path = Path(model_path)
        self.reports_path = Path(reports_path)
        self.model_parameters = model_parameters
        self.model: Optional[Any] = None
        self.feature_columns: Sequence[str] = ()

    def train(self, features: pd.DataFrame, target: pd.Series) -> Any:
        """Train an XGBoost classifier and save it with joblib.

        ``scale_pos_weight`` is calculated as negative examples divided by
        positive examples unless explicitly supplied in ``model_parameters``.
        Categorical columns are one-hot encoded and the resulting feature
        columns are stored with the model for consistent prediction.

        Args:
            features: Training predictors.
            target: Binary target where ``1`` represents startup success.

        Returns:
            The fitted XGBClassifier.

        Raises:
            ValueError: If the training data is empty or the target is not
                binary with both classes present.
            ImportError: If XGBoost or joblib is unavailable.
        """
        try:
            target_values = self._normalize_target(target)
            self._validate_training_data(features, target_values)
            xgboost = self._import_xgboost()
            encoded_features = self._prepare_features(features)
            positive_count = int((target_values == 1).sum())
            negative_count = int((target_values == 0).sum())
            parameters: Dict[str, Any] = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "random_state": 42,
                "n_jobs": 1,
                "scale_pos_weight": negative_count / positive_count,
            }
            parameters.update(self.model_parameters)
            self.model = xgboost.XGBClassifier(**parameters)
            self.model.fit(encoded_features, target_values)
            self.feature_columns = tuple(encoded_features.columns)
            self._save_model()
            logger.info(
                "Trained success predictor on %d rows and saved it to %s",
                len(encoded_features),
                self.model_path,
            )
            return self.model
        except (ImportError, TypeError, ValueError) as error:
            logger.exception("Unable to train success predictor: %s", error)
            raise

    def predict_probability(self, features: pd.DataFrame) -> np.ndarray:
        """Return the predicted probability of success for each input row.

        Args:
            features: Predictor columns in the same format used for training.

        Returns:
            One success probability per input row.

        Raises:
            RuntimeError: If no trained model has been loaded.
        """
        try:
            self._ensure_model()
            encoded_features = self._prepare_features(features)
            aligned_features = encoded_features.reindex(
                columns=self.feature_columns, fill_value=0.0
            )
            return np.asarray(self.model.predict_proba(aligned_features)[:, 1])
        except (TypeError, ValueError, RuntimeError) as error:
            logger.exception("Unable to predict success probabilities: %s", error)
            raise

    def get_shap_summary_plot(self, features: pd.DataFrame) -> Path:
        """Generate and save a SHAP feature-importance summary plot.

        Args:
            features: Rows to explain in the same format used for training.

        Returns:
            Path to the saved PNG file.

        Raises:
            ImportError: If SHAP or matplotlib is unavailable.
            RuntimeError: If no trained model has been loaded.
        """
        try:
            self._ensure_model()
            shap = self._import_shap()
            import matplotlib.pyplot as plt

            encoded_features = self._prepare_features(features).reindex(
                columns=self.feature_columns, fill_value=0.0
            )
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(encoded_features)
            if isinstance(shap_values, list):
                shap_values = shap_values[-1]
            self.reports_path.parent.mkdir(parents=True, exist_ok=True)
            plt.figure()
            shap.summary_plot(
                shap_values,
                encoded_features,
                show=False,
                plot_size="auto",
            )
            plt.tight_layout()
            plt.savefig(self.reports_path, dpi=160, bbox_inches="tight")
            plt.close()
            logger.info("Saved SHAP summary plot to %s", self.reports_path)
            return self.reports_path
        except (ImportError, RuntimeError, TypeError, ValueError) as error:
            logger.exception("Unable to create SHAP summary plot: %s", error)
            raise

    @classmethod
    def load(cls, model_path: PathLike) -> "SuccessPredictor":
        """Load a persisted predictor from a joblib file."""
        try:
            import joblib

            bundle = joblib.load(model_path)
            predictor = cls(model_path=model_path)
            predictor.model = bundle["model"]
            predictor.feature_columns = tuple(bundle["feature_columns"])
            return predictor
        except (ImportError, OSError, KeyError, TypeError, ValueError) as error:
            logger.exception("Unable to load success predictor: %s", error)
            raise

    @staticmethod
    def _import_xgboost() -> Any:
        """Import XGBoost with a clear dependency error boundary."""
        try:
            import xgboost

            return xgboost
        except ImportError as error:
            raise ImportError("XGBoost is required to train SuccessPredictor") from error

    @staticmethod
    def _import_shap() -> Any:
        """Import SHAP with a clear dependency error boundary."""
        try:
            import shap

            return shap
        except ImportError as error:
            raise ImportError("SHAP is required to explain SuccessPredictor") from error

    def _save_model(self) -> None:
        """Persist model state and preprocessing metadata with joblib."""
        try:
            import joblib

            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {"model": self.model, "feature_columns": self.feature_columns},
                self.model_path,
            )
        except (ImportError, OSError, TypeError, ValueError) as error:
            logger.exception("Unable to save success predictor: %s", error)
            raise

    def _ensure_model(self) -> None:
        """Ensure a fitted model is available for prediction or explanation."""
        if self.model is None or not self.feature_columns:
            raise RuntimeError("Train or load a SuccessPredictor before using it")

    @staticmethod
    def _validate_training_data(
        features: pd.DataFrame, target: pd.Series
    ) -> None:
        """Validate the shape and classes of training data."""
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")
        if not isinstance(target, pd.Series):
            raise TypeError("target must be a pandas Series")
        if features.empty or target.empty:
            raise ValueError("features and target must not be empty")
        if len(features) != len(target):
            raise ValueError("features and target must have the same row count")
        numeric_target = pd.to_numeric(target, errors="coerce")
        if numeric_target.isna().any() or not set(numeric_target.unique()).issubset({0, 1}):
            raise ValueError("target must contain only binary values 0 and 1")
        if numeric_target.nunique() != 2:
            raise ValueError("target must contain both classes 0 and 1")

    @staticmethod
    def _normalize_target(target: pd.Series) -> pd.Series:
        """Normalize numeric or startup outcome labels to binary values."""
        if not isinstance(target, pd.Series):
            raise TypeError("target must be a pandas Series")
        numeric_target = pd.to_numeric(target, errors="coerce")
        if numeric_target.notna().all():
            return numeric_target.astype(int)
        labels = target.astype("string").str.strip().str.casefold()
        if labels.isna().any() or labels.eq("").any():
            raise ValueError("target contains empty labels")
        return labels.ne("failure").astype(int)

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features and fill numeric missing values."""
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")
        if features.empty:
            raise ValueError("features must not be empty")
        prepared = features.copy()
        categorical_columns = list(
            prepared.select_dtypes(include=["object", "category", "bool"]).columns
        )
        for column in categorical_columns:
            prepared[column] = prepared[column].astype("string").fillna("__missing__")
        prepared = pd.get_dummies(
            prepared, columns=categorical_columns, dtype=float
        )
        for column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        if prepared.isna().any().any():
            prepared = prepared.fillna(prepared.median(numeric_only=True)).fillna(0.0)
        if not np.isfinite(prepared.to_numpy(dtype=float)).all():
            raise ValueError("features must contain finite numeric values")
        return prepared.astype(float)