"""Ensemble model system for arbitrage detection.

Multi-model ensemble with specialized models for different arbitrage types,
based on arXiv paper 2512.02436 "Semantic Trading".
"""

import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod

import numpy as np

from src.ml.feature_builder_v2 import AdvancedFeatureBuilder, FeatureSet

logger = logging.getLogger(__name__)

# Model storage
MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
ENSEMBLE_MODEL_FILE = MODEL_DIR / "ensemble_model.pkl"


@dataclass
class PredictionResult:
    """Result from model prediction."""
    probability: float  # Probability of profitable opportunity
    confidence: float   # Model confidence in prediction
    model_name: str     # Which model made prediction
    features_used: int  # Number of features used
    raw_scores: Dict[str, float] = field(default_factory=dict)  # Per-model scores


class BaseModel(ABC):
    """Base class for arbitrage detection models."""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_fitted = False
        self.feature_names: List[str] = []

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train the model."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)


class GradientBoostingModel(BaseModel):
    """Gradient boosting model (sklearn)."""

    def __init__(self, name: str = "gradient_boosting", **kwargs):
        super().__init__(name)
        self.params = {
            "n_estimators": kwargs.get("n_estimators", 100),
            "max_depth": kwargs.get("max_depth", 5),
            "learning_rate": kwargs.get("learning_rate", 0.1),
            "subsample": kwargs.get("subsample", 0.8),
            "random_state": 42,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import cross_val_score

            self.model = GradientBoostingClassifier(**self.params)
            self.model.fit(X, y)
            self.is_fitted = True

            # Cross-validation score
            cv_scores = cross_val_score(self.model, X, y, cv=3, scoring='roc_auc')

            return {
                "train_score": self.model.score(X, y),
                "cv_score": cv_scores.mean(),
                "cv_std": cv_scores.std()
            }
        except ImportError:
            logger.warning("sklearn not available")
            return {"error": "sklearn_not_available"}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.full(len(X), 0.5)
        proba = self.model.predict_proba(X)
        return proba[:, 1] if proba.ndim > 1 else proba


class XGBoostModel(BaseModel):
    """XGBoost model."""

    def __init__(self, name: str = "xgboost", **kwargs):
        super().__init__(name)
        self.params = {
            "n_estimators": kwargs.get("n_estimators", 200),
            "max_depth": kwargs.get("max_depth", 6),
            "learning_rate": kwargs.get("learning_rate", 0.1),
            "subsample": kwargs.get("subsample", 0.8),
            "colsample_bytree": kwargs.get("colsample_bytree", 0.8),
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss",
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        try:
            from xgboost import XGBClassifier
            from sklearn.model_selection import cross_val_score

            self.model = XGBClassifier(**self.params)
            self.model.fit(X, y)
            self.is_fitted = True

            cv_scores = cross_val_score(self.model, X, y, cv=3, scoring='roc_auc')

            return {
                "train_score": self.model.score(X, y),
                "cv_score": cv_scores.mean(),
                "cv_std": cv_scores.std()
            }
        except ImportError:
            logger.warning("xgboost not available, falling back to sklearn")
            return GradientBoostingModel(self.name).fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.full(len(X), 0.5)
        proba = self.model.predict_proba(X)
        return proba[:, 1] if proba.ndim > 1 else proba


class LightGBMModel(BaseModel):
    """LightGBM model."""

    def __init__(self, name: str = "lightgbm", **kwargs):
        super().__init__(name)
        self.params = {
            "n_estimators": kwargs.get("n_estimators", 200),
            "num_leaves": kwargs.get("num_leaves", 31),
            "learning_rate": kwargs.get("learning_rate", 0.1),
            "subsample": kwargs.get("subsample", 0.8),
            "random_state": 42,
            "verbose": -1,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        try:
            from lightgbm import LGBMClassifier
            from sklearn.model_selection import cross_val_score

            self.model = LGBMClassifier(**self.params)
            self.model.fit(X, y)
            self.is_fitted = True

            cv_scores = cross_val_score(self.model, X, y, cv=3, scoring='roc_auc')

            return {
                "train_score": self.model.score(X, y),
                "cv_score": cv_scores.mean(),
                "cv_std": cv_scores.std()
            }
        except ImportError:
            logger.warning("lightgbm not available, falling back to sklearn")
            return GradientBoostingModel(self.name).fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.full(len(X), 0.5)
        proba = self.model.predict_proba(X)
        return proba[:, 1] if proba.ndim > 1 else proba


class RandomForestModel(BaseModel):
    """Random Forest model."""

    def __init__(self, name: str = "random_forest", **kwargs):
        super().__init__(name)
        self.params = {
            "n_estimators": kwargs.get("n_estimators", 100),
            "max_depth": kwargs.get("max_depth", 10),
            "min_samples_split": kwargs.get("min_samples_split", 5),
            "random_state": 42,
            "n_jobs": -1,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score

            self.model = RandomForestClassifier(**self.params)
            self.model.fit(X, y)
            self.is_fitted = True

            cv_scores = cross_val_score(self.model, X, y, cv=3, scoring='roc_auc')

            return {
                "train_score": self.model.score(X, y),
                "cv_score": cv_scores.mean(),
                "cv_std": cv_scores.std()
            }
        except ImportError:
            logger.warning("sklearn not available")
            return {"error": "sklearn_not_available"}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.full(len(X), 0.5)
        proba = self.model.predict_proba(X)
        return proba[:, 1] if proba.ndim > 1 else proba


class LogisticModel(BaseModel):
    """Logistic regression model (fast baseline)."""

    def __init__(self, name: str = "logistic", **kwargs):
        super().__init__(name)
        self.params = {
            "C": kwargs.get("C", 1.0),
            "max_iter": kwargs.get("max_iter", 1000),
            "random_state": 42,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score

            self.model = LogisticRegression(**self.params)
            self.model.fit(X, y)
            self.is_fitted = True

            cv_scores = cross_val_score(self.model, X, y, cv=3, scoring='roc_auc')

            return {
                "train_score": self.model.score(X, y),
                "cv_score": cv_scores.mean(),
                "cv_std": cv_scores.std()
            }
        except ImportError:
            logger.warning("sklearn not available")
            return {"error": "sklearn_not_available"}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.full(len(X), 0.5)
        proba = self.model.predict_proba(X)
        return proba[:, 1] if proba.ndim > 1 else proba


class ArbitrageEnsemble:
    """
    Ensemble of models for arbitrage detection.

    Combines multiple models using stacking or averaging.
    """

    def __init__(self, use_stacking: bool = True):
        """
        Initialize ensemble.

        Args:
            use_stacking: If True, use stacking with meta-learner.
                         If False, use simple averaging.
        """
        self.use_stacking = use_stacking
        self.feature_builder = AdvancedFeatureBuilder()

        # Base models
        self.base_models: Dict[str, BaseModel] = {
            "gradient_boosting": GradientBoostingModel(),
            "random_forest": RandomForestModel(),
            "logistic": LogisticModel(),
        }

        # Try to add XGBoost and LightGBM
        try:
            import xgboost
            self.base_models["xgboost"] = XGBoostModel()
        except ImportError:
            pass

        try:
            import lightgbm
            self.base_models["lightgbm"] = LightGBMModel()
        except ImportError:
            pass

        # Meta-learner for stacking
        self.meta_learner: Optional[LogisticModel] = None
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.model_weights: Dict[str, float] = {}

    def _save(self) -> None:
        """Save ensemble to disk."""
        try:
            data = {
                "base_models": self.base_models,
                "meta_learner": self.meta_learner,
                "feature_names": self.feature_names,
                "model_weights": self.model_weights,
                "use_stacking": self.use_stacking,
            }
            with open(ENSEMBLE_MODEL_FILE, 'wb') as f:
                pickle.dump(data, f)
            logger.info("Saved ensemble model")
        except Exception as e:
            logger.error(f"Error saving ensemble: {e}")

    def _load(self) -> bool:
        """Load ensemble from disk."""
        if not ENSEMBLE_MODEL_FILE.exists():
            return False

        try:
            with open(ENSEMBLE_MODEL_FILE, 'rb') as f:
                data = pickle.load(f)

            self.base_models = data.get("base_models", self.base_models)
            self.meta_learner = data.get("meta_learner")
            self.feature_names = data.get("feature_names", [])
            self.model_weights = data.get("model_weights", {})
            self.use_stacking = data.get("use_stacking", True)
            self.is_fitted = True

            logger.info("Loaded ensemble model")
            return True
        except Exception as e:
            logger.warning(f"Error loading ensemble: {e}")
            return False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train the ensemble.

        Args:
            X: Feature matrix
            y: Target labels (1=profitable, 0=not profitable)
            validation_split: Fraction for validation

        Returns:
            Training metrics
        """
        logger.info(f"Training ensemble on {len(X)} samples with {X.shape[1]} features")

        self.feature_names = self.feature_builder.get_feature_names()
        metrics = {"base_models": {}}

        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )

        # Train base models
        oof_predictions = np.zeros((len(X_train), len(self.base_models)))

        for i, (name, model) in enumerate(self.base_models.items()):
            logger.info(f"Training {name}...")
            model_metrics = model.fit(X_train, y_train)
            metrics["base_models"][name] = model_metrics

            # Get OOF predictions for stacking
            if self.use_stacking:
                oof_predictions[:, i] = model.predict_proba(X_train)

        # Train meta-learner for stacking
        if self.use_stacking:
            logger.info("Training meta-learner...")
            self.meta_learner = LogisticModel("meta_learner")
            self.meta_learner.fit(oof_predictions, y_train)

        # Calculate model weights based on validation performance
        val_predictions = {}
        for name, model in self.base_models.items():
            val_pred = model.predict_proba(X_val)
            val_predictions[name] = val_pred

            # Calculate AUC
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y_val, val_pred)
                self.model_weights[name] = auc
            except:
                self.model_weights[name] = 0.5

        # Normalize weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            self.model_weights = {k: v/total_weight for k, v in self.model_weights.items()}

        # Calculate ensemble validation score
        ensemble_pred = self.predict_proba(X_val)
        try:
            from sklearn.metrics import roc_auc_score, accuracy_score
            metrics["ensemble_auc"] = roc_auc_score(y_val, ensemble_pred)
            metrics["ensemble_accuracy"] = accuracy_score(y_val, ensemble_pred >= 0.5)
        except:
            pass

        metrics["model_weights"] = self.model_weights
        metrics["n_models"] = len(self.base_models)

        self.is_fitted = True
        self._save()

        logger.info(f"Ensemble trained. Validation AUC: {metrics.get('ensemble_auc', 'N/A')}")
        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using ensemble.

        Args:
            X: Feature matrix

        Returns:
            Array of probabilities
        """
        if not self.is_fitted:
            # Try to load saved model
            if not self._load():
                logger.warning("Ensemble not fitted, returning 0.5")
                return np.full(len(X), 0.5)

        # Get predictions from all base models
        base_predictions = np.zeros((len(X), len(self.base_models)))

        for i, (name, model) in enumerate(self.base_models.items()):
            if model.is_fitted:
                base_predictions[:, i] = model.predict_proba(X)
            else:
                base_predictions[:, i] = 0.5

        # Combine predictions
        if self.use_stacking and self.meta_learner is not None and self.meta_learner.is_fitted:
            # Use meta-learner
            return self.meta_learner.predict_proba(base_predictions)
        else:
            # Weighted average
            weights = np.array([self.model_weights.get(name, 1.0) for name in self.base_models.keys()])
            weights = weights / weights.sum()
            return np.average(base_predictions, axis=1, weights=weights)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def predict_single(self, item: Dict) -> PredictionResult:
        """
        Predict for a single opportunity.

        Args:
            item: Market/opportunity dict

        Returns:
            PredictionResult with probability and model info
        """
        # Build features
        fs = self.feature_builder.build_features(item)
        X = fs.to_vector().reshape(1, -1)

        # Get ensemble prediction
        proba = self.predict_proba(X)[0]

        # Get per-model scores
        raw_scores = {}
        for name, model in self.base_models.items():
            if model.is_fitted:
                raw_scores[name] = float(model.predict_proba(X)[0])

        # Calculate confidence (agreement between models)
        if len(raw_scores) > 1:
            scores = list(raw_scores.values())
            confidence = 1.0 - np.std(scores)  # Higher agreement = higher confidence
        else:
            confidence = 0.7

        return PredictionResult(
            probability=float(proba),
            confidence=float(confidence),
            model_name="ensemble",
            features_used=len(self.feature_names),
            raw_scores=raw_scores
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the ensemble."""
        return {
            "is_fitted": self.is_fitted,
            "use_stacking": self.use_stacking,
            "n_base_models": len(self.base_models),
            "base_models": list(self.base_models.keys()),
            "model_weights": self.model_weights,
            "n_features": len(self.feature_names),
        }


# Convenience functions
_ensemble: Optional[ArbitrageEnsemble] = None


def get_ensemble() -> ArbitrageEnsemble:
    """Get or create singleton ensemble."""
    global _ensemble
    if _ensemble is None:
        _ensemble = ArbitrageEnsemble()
        _ensemble._load()  # Try to load saved model
    return _ensemble


def train_ensemble(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Train the ensemble model."""
    ensemble = get_ensemble()
    return ensemble.fit(X, y)


def predict_opportunity(item: Dict) -> PredictionResult:
    """Predict if an opportunity is profitable."""
    ensemble = get_ensemble()
    return ensemble.predict_single(item)


def predict_batch(items: List[Dict]) -> np.ndarray:
    """Predict probabilities for multiple items."""
    ensemble = get_ensemble()
    X = ensemble.feature_builder.build_features_batch(items)
    return ensemble.predict_proba(X)
