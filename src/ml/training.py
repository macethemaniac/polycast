"""Training utilities for ML models.

Includes time-series cross-validation and hyperparameter optimization.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Iterator, Callable
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CVFold:
    """A single cross-validation fold."""
    fold_idx: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_start: Optional[float] = None
    train_end: Optional[float] = None
    test_start: Optional[float] = None
    test_end: Optional[float] = None


@dataclass
class CVResult:
    """Cross-validation results."""
    scores: List[float]
    mean_score: float
    std_score: float
    fold_details: List[Dict[str, Any]] = field(default_factory=list)

    def __str__(self):
        return f"CV Score: {self.mean_score:.4f} (+/- {self.std_score:.4f})"


class TimeSeriesSplit:
    """
    Time-series cross-validation splitter.

    Respects temporal ordering and includes gap to prevent lookahead.
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_size: float = 0.6,
        gap_size: int = 0,
        expanding: bool = True
    ):
        """
        Initialize time-series splitter.

        Args:
            n_splits: Number of CV folds
            train_size: Initial training set size (fraction)
            gap_size: Number of samples to skip between train and test (prevent leakage)
            expanding: If True, training set expands. If False, sliding window.
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.gap_size = gap_size
        self.expanding = expanding

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None
    ) -> Iterator[CVFold]:
        """
        Generate train/test splits.

        Args:
            X: Feature matrix
            y: Labels (optional, not used for splitting)
            timestamps: Sample timestamps for proper ordering

        Yields:
            CVFold objects with train/test indices
        """
        n_samples = len(X)

        # Sort by timestamp if provided
        if timestamps is not None:
            sorted_indices = np.argsort(timestamps)
        else:
            sorted_indices = np.arange(n_samples)

        # Calculate fold sizes
        initial_train_size = int(n_samples * self.train_size)
        test_size = (n_samples - initial_train_size - self.gap_size * self.n_splits) // self.n_splits

        if test_size < 10:
            logger.warning(f"Test size {test_size} is small, consider fewer splits")

        for fold_idx in range(self.n_splits):
            if self.expanding:
                # Expanding window: train set grows
                train_end = initial_train_size + fold_idx * test_size
                train_start = 0
            else:
                # Sliding window: fixed train size
                train_start = fold_idx * test_size
                train_end = train_start + initial_train_size

            # Gap between train and test
            test_start = train_end + self.gap_size
            test_end = test_start + test_size

            # Ensure we don't exceed data
            if test_end > n_samples:
                test_end = n_samples

            train_indices = sorted_indices[train_start:train_end]
            test_indices = sorted_indices[test_start:test_end]

            if len(test_indices) == 0:
                continue

            # Get timestamps for fold bounds
            fold = CVFold(
                fold_idx=fold_idx,
                train_indices=train_indices,
                test_indices=test_indices
            )

            if timestamps is not None:
                fold.train_start = float(timestamps[sorted_indices[train_start]])
                fold.train_end = float(timestamps[sorted_indices[train_end - 1]])
                fold.test_start = float(timestamps[sorted_indices[test_start]])
                fold.test_end = float(timestamps[sorted_indices[min(test_end - 1, n_samples - 1)]])

            yield fold

    def get_n_splits(self) -> int:
        return self.n_splits


class WalkForwardValidator:
    """
    Walk-forward validation for time-series data.

    More realistic for trading strategies.
    """

    def __init__(
        self,
        train_period_days: int = 30,
        test_period_days: int = 7,
        step_days: int = 7,
        gap_days: int = 1
    ):
        """
        Initialize walk-forward validator.

        Args:
            train_period_days: Training window size in days
            test_period_days: Test window size in days
            step_days: How much to move forward each fold
            gap_days: Gap between train end and test start
        """
        self.train_period = train_period_days * 86400  # Convert to seconds
        self.test_period = test_period_days * 86400
        self.step = step_days * 86400
        self.gap = gap_days * 86400

    def split(
        self,
        X: np.ndarray,
        timestamps: np.ndarray
    ) -> Iterator[CVFold]:
        """Generate walk-forward splits based on timestamps."""
        min_ts = timestamps.min()
        max_ts = timestamps.max()

        current_train_end = min_ts + self.train_period
        fold_idx = 0

        while current_train_end + self.gap + self.test_period <= max_ts:
            train_mask = (timestamps >= (current_train_end - self.train_period)) & (timestamps < current_train_end)
            test_mask = (timestamps >= (current_train_end + self.gap)) & (timestamps < (current_train_end + self.gap + self.test_period))

            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            if len(train_indices) > 0 and len(test_indices) > 0:
                yield CVFold(
                    fold_idx=fold_idx,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    train_start=current_train_end - self.train_period,
                    train_end=current_train_end,
                    test_start=current_train_end + self.gap,
                    test_end=current_train_end + self.gap + self.test_period
                )
                fold_idx += 1

            current_train_end += self.step


def cross_validate(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: Any,
    scoring: str = "roc_auc",
    timestamps: Optional[np.ndarray] = None
) -> CVResult:
    """
    Perform cross-validation with given splitter.

    Args:
        model: Model to evaluate (must have fit and predict_proba)
        X: Feature matrix
        y: Labels
        cv: Cross-validation splitter
        scoring: Metric to use ("roc_auc", "accuracy", "precision", "recall", "f1")
        timestamps: Optional timestamps for time-series CV

    Returns:
        CVResult with scores
    """
    from sklearn.base import clone

    scores = []
    fold_details = []

    for fold in cv.split(X, y, timestamps):
        # Clone model for fresh training
        model_clone = clone(model)

        # Train
        X_train = X[fold.train_indices]
        y_train = y[fold.train_indices]
        X_test = X[fold.test_indices]
        y_test = y[fold.test_indices]

        model_clone.fit(X_train, y_train)

        # Score
        if scoring == "roc_auc":
            from sklearn.metrics import roc_auc_score
            y_pred = model_clone.predict_proba(X_test)
            if y_pred.ndim > 1:
                y_pred = y_pred[:, 1]
            score = roc_auc_score(y_test, y_pred)
        elif scoring == "accuracy":
            score = model_clone.score(X_test, y_test)
        elif scoring == "precision":
            from sklearn.metrics import precision_score
            y_pred = model_clone.predict(X_test)
            score = precision_score(y_test, y_pred, zero_division=0)
        elif scoring == "recall":
            from sklearn.metrics import recall_score
            y_pred = model_clone.predict(X_test)
            score = recall_score(y_test, y_pred, zero_division=0)
        elif scoring == "f1":
            from sklearn.metrics import f1_score
            y_pred = model_clone.predict(X_test)
            score = f1_score(y_test, y_pred, zero_division=0)
        else:
            score = model_clone.score(X_test, y_test)

        scores.append(score)
        fold_details.append({
            "fold": fold.fold_idx,
            "score": score,
            "n_train": len(fold.train_indices),
            "n_test": len(fold.test_indices),
            "train_start": fold.train_start,
            "test_end": fold.test_end,
        })

    return CVResult(
        scores=scores,
        mean_score=float(np.mean(scores)),
        std_score=float(np.std(scores)),
        fold_details=fold_details
    )


@dataclass
class HyperparamResult:
    """Result from hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    all_trials: List[Dict[str, Any]] = field(default_factory=list)
    n_trials: int = 0


class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna or grid search."""

    def __init__(self, use_optuna: bool = True):
        """
        Initialize optimizer.

        Args:
            use_optuna: If True, use Optuna. If False, use grid search.
        """
        self.use_optuna = use_optuna
        self._optuna_available = False

        try:
            import optuna
            self._optuna_available = True
        except ImportError:
            if use_optuna:
                logger.warning("Optuna not available, falling back to grid search")
            self.use_optuna = False

    def optimize_gradient_boosting(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 50,
        cv_splits: int = 3,
        timestamps: Optional[np.ndarray] = None
    ) -> HyperparamResult:
        """Optimize GradientBoosting hyperparameters."""
        if self.use_optuna and self._optuna_available:
            return self._optimize_optuna_gb(X, y, n_trials, cv_splits, timestamps)
        else:
            return self._optimize_grid_gb(X, y, cv_splits, timestamps)

    def _optimize_optuna_gb(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int,
        cv_splits: int,
        timestamps: Optional[np.ndarray]
    ) -> HyperparamResult:
        """Optimize using Optuna."""
        import optuna
        from sklearn.ensemble import GradientBoostingClassifier

        cv = TimeSeriesSplit(n_splits=cv_splits)
        trials = []

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": 42,
            }

            model = GradientBoostingClassifier(**params)
            result = cross_validate(model, X, y, cv, "roc_auc", timestamps)

            trials.append({
                "params": params,
                "score": result.mean_score,
                "std": result.std_score
            })

            return result.mean_score

        # Run optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        return HyperparamResult(
            best_params=study.best_params,
            best_score=study.best_value,
            all_trials=trials,
            n_trials=n_trials
        )

    def _optimize_grid_gb(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_splits: int,
        timestamps: Optional[np.ndarray]
    ) -> HyperparamResult:
        """Optimize using grid search."""
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import ParameterGrid

        cv = TimeSeriesSplit(n_splits=cv_splits)

        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.8, 1.0],
        }

        best_score = -np.inf
        best_params = {}
        trials = []

        for params in ParameterGrid(param_grid):
            params["random_state"] = 42
            model = GradientBoostingClassifier(**params)

            try:
                result = cross_validate(model, X, y, cv, "roc_auc", timestamps)
                score = result.mean_score

                trials.append({
                    "params": params,
                    "score": score,
                    "std": result.std_score
                })

                if score > best_score:
                    best_score = score
                    best_params = params
            except Exception as e:
                logger.debug(f"Grid search iteration failed: {e}")

        return HyperparamResult(
            best_params=best_params,
            best_score=best_score,
            all_trials=trials,
            n_trials=len(trials)
        )

    def optimize_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 50,
        cv_splits: int = 3,
        timestamps: Optional[np.ndarray] = None
    ) -> HyperparamResult:
        """Optimize XGBoost hyperparameters."""
        try:
            from xgboost import XGBClassifier
        except ImportError:
            logger.warning("XGBoost not available")
            return HyperparamResult(best_params={}, best_score=0.0)

        if self.use_optuna and self._optuna_available:
            import optuna

            cv = TimeSeriesSplit(n_splits=cv_splits)
            trials = []

            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 2, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "gamma": trial.suggest_float("gamma", 0, 1),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
                    "random_state": 42,
                    "use_label_encoder": False,
                    "eval_metric": "logloss",
                }

                model = XGBClassifier(**params)
                result = cross_validate(model, X, y, cv, "roc_auc", timestamps)

                trials.append({"params": params, "score": result.mean_score})
                return result.mean_score

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

            return HyperparamResult(
                best_params=study.best_params,
                best_score=study.best_value,
                all_trials=trials,
                n_trials=n_trials
            )
        else:
            # Simple grid search for XGBoost
            return self._optimize_grid_gb(X, y, cv_splits, timestamps)


# Convenience functions
def time_series_cv(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    timestamps: Optional[np.ndarray] = None
) -> CVResult:
    """Run time-series cross-validation."""
    cv = TimeSeriesSplit(n_splits=n_splits)
    return cross_validate(model, X, y, cv, timestamps=timestamps)


def optimize_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "gradient_boosting",
    n_trials: int = 50,
    timestamps: Optional[np.ndarray] = None
) -> HyperparamResult:
    """Optimize model hyperparameters."""
    optimizer = HyperparameterOptimizer()

    if model_type == "xgboost":
        return optimizer.optimize_xgboost(X, y, n_trials, timestamps=timestamps)
    else:
        return optimizer.optimize_gradient_boosting(X, y, n_trials, timestamps=timestamps)
