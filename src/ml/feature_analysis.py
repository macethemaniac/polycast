"""Feature importance analysis for ML models.

Provides SHAP values, permutation importance, and feature correlation analysis.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Cache directory
ANALYSIS_DIR = Path(__file__).resolve().parents[2] / "data" / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class FeatureImportance:
    """Feature importance results."""
    feature_name: str
    importance: float
    std: float = 0.0
    rank: int = 0


@dataclass
class FeatureAnalysisResult:
    """Complete feature analysis result."""
    importances: List[FeatureImportance] = field(default_factory=list)
    correlations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def get_top_features(self, n: int = 10) -> List[FeatureImportance]:
        """Get top N most important features."""
        sorted_imp = sorted(self.importances, key=lambda x: -x.importance)
        return sorted_imp[:n]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "importances": {fi.feature_name: fi.importance for fi in self.importances},
            "top_features": [fi.feature_name for fi in self.get_top_features(10)],
            "statistics": self.statistics
        }


class FeatureAnalyzer:
    """Analyzes feature importance and relationships."""

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def compute_permutation_importance(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 10
    ) -> List[FeatureImportance]:
        """
        Compute permutation importance for a trained model.

        Shuffles each feature and measures performance degradation.
        """
        try:
            from sklearn.inspection import permutation_importance

            result = permutation_importance(
                model, X, y,
                n_repeats=n_repeats,
                random_state=42,
                n_jobs=-1
            )

            importances = []
            for i, fn in enumerate(self.feature_names):
                importances.append(FeatureImportance(
                    feature_name=fn,
                    importance=float(result.importances_mean[i]),
                    std=float(result.importances_std[i]),
                    rank=0
                ))

            # Assign ranks
            importances.sort(key=lambda x: -x.importance)
            for i, fi in enumerate(importances):
                fi.rank = i + 1

            return importances

        except ImportError:
            logger.warning("sklearn not available for permutation importance")
            return self._fallback_importance(X, y)

    def compute_tree_importance(self, model: Any) -> List[FeatureImportance]:
        """
        Get feature importance from tree-based models.

        Works with XGBoost, LightGBM, RandomForest, etc.
        """
        try:
            # Try different attribute names
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
            elif hasattr(model, 'get_score'):
                # XGBoost
                score = model.get_score(importance_type='gain')
                imp = np.array([score.get(f'f{i}', 0) for i in range(len(self.feature_names))])
            else:
                return []

            importances = []
            for i, fn in enumerate(self.feature_names):
                importances.append(FeatureImportance(
                    feature_name=fn,
                    importance=float(imp[i]) if i < len(imp) else 0.0,
                    rank=0
                ))

            # Assign ranks
            importances.sort(key=lambda x: -x.importance)
            for i, fi in enumerate(importances):
                fi.rank = i + 1

            return importances

        except Exception as e:
            logger.warning(f"Could not extract tree importance: {e}")
            return []

    def compute_shap_importance(
        self,
        model: Any,
        X: np.ndarray,
        max_samples: int = 1000
    ) -> List[FeatureImportance]:
        """
        Compute SHAP values for model explanation.

        SHAP provides consistent feature attributions.
        """
        try:
            import shap

            # Subsample if too large
            if len(X) > max_samples:
                indices = np.random.choice(len(X), max_samples, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X

            # Create explainer based on model type
            model_type = type(model).__name__.lower()

            if 'tree' in model_type or 'forest' in model_type or 'xgb' in model_type or 'lgb' in model_type:
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.KernelExplainer(model.predict_proba, X_sample[:100])

            shap_values = explainer.shap_values(X_sample)

            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Take positive class

            # Mean absolute SHAP value per feature
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

            importances = []
            for i, fn in enumerate(self.feature_names):
                importances.append(FeatureImportance(
                    feature_name=fn,
                    importance=float(mean_abs_shap[i]) if i < len(mean_abs_shap) else 0.0,
                    rank=0
                ))

            # Assign ranks
            importances.sort(key=lambda x: -x.importance)
            for i, fi in enumerate(importances):
                fi.rank = i + 1

            return importances

        except ImportError:
            logger.warning("shap not available")
            return []
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return []

    def _fallback_importance(self, X: np.ndarray, y: np.ndarray) -> List[FeatureImportance]:
        """Fallback importance based on correlation with target."""
        importances = []

        for i, fn in enumerate(self.feature_names):
            if i < X.shape[1]:
                # Correlation with target
                corr = np.corrcoef(X[:, i], y)[0, 1]
                importance = abs(corr) if not np.isnan(corr) else 0.0
            else:
                importance = 0.0

            importances.append(FeatureImportance(
                feature_name=fn,
                importance=importance,
                rank=0
            ))

        # Assign ranks
        importances.sort(key=lambda x: -x.importance)
        for i, fi in enumerate(importances):
            fi.rank = i + 1

        return importances

    def compute_feature_statistics(self, X: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compute basic statistics for each feature."""
        stats = {}

        for i, fn in enumerate(self.feature_names):
            if i < X.shape[1]:
                col = X[:, i]
                stats[fn] = {
                    "mean": float(np.mean(col)),
                    "std": float(np.std(col)),
                    "min": float(np.min(col)),
                    "max": float(np.max(col)),
                    "median": float(np.median(col)),
                    "missing_rate": float(np.isnan(col).sum() / len(col)),
                }
            else:
                stats[fn] = {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0, "missing_rate": 1}

        return stats

    def compute_feature_correlations(
        self,
        X: np.ndarray,
        threshold: float = 0.7
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute feature correlations.

        Returns pairs with correlation above threshold.
        """
        correlations = {}

        try:
            corr_matrix = np.corrcoef(X.T)

            for i, fn_i in enumerate(self.feature_names):
                if i >= X.shape[1]:
                    continue

                high_corr = {}
                for j, fn_j in enumerate(self.feature_names):
                    if j >= X.shape[1] or i >= j:
                        continue

                    corr = corr_matrix[i, j]
                    if not np.isnan(corr) and abs(corr) >= threshold:
                        high_corr[fn_j] = float(corr)

                if high_corr:
                    correlations[fn_i] = high_corr

        except Exception as e:
            logger.warning(f"Correlation computation failed: {e}")

        return correlations

    def analyze(
        self,
        model: Optional[Any],
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        use_shap: bool = False
    ) -> FeatureAnalysisResult:
        """
        Complete feature analysis.

        Args:
            model: Trained model (optional)
            X: Feature matrix
            y: Target values (optional, needed for permutation importance)
            use_shap: Whether to compute SHAP values (slower but more accurate)

        Returns:
            FeatureAnalysisResult with all analysis
        """
        result = FeatureAnalysisResult()

        # Compute importances
        if model is not None:
            # Try tree importance first (fast)
            result.importances = self.compute_tree_importance(model)

            # If tree importance failed and we have labels, use permutation
            if not result.importances and y is not None:
                result.importances = self.compute_permutation_importance(model, X, y)

            # Optionally compute SHAP (slow but accurate)
            if use_shap and len(result.importances) == 0:
                result.importances = self.compute_shap_importance(model, X)

        # Fallback to correlation-based importance
        if not result.importances and y is not None:
            result.importances = self._fallback_importance(X, y)

        # Compute statistics
        result.statistics = self.compute_feature_statistics(X)

        # Compute correlations
        result.correlations = self.compute_feature_correlations(X)

        return result

    def select_features(
        self,
        importances: List[FeatureImportance],
        min_importance: float = 0.01,
        max_features: int = 30
    ) -> List[str]:
        """
        Select most important features.

        Args:
            importances: Feature importance list
            min_importance: Minimum importance threshold
            max_features: Maximum features to select

        Returns:
            List of selected feature names
        """
        selected = []

        for fi in sorted(importances, key=lambda x: -x.importance):
            if fi.importance < min_importance:
                break
            if len(selected) >= max_features:
                break
            selected.append(fi.feature_name)

        return selected


def analyze_features(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str]
) -> FeatureAnalysisResult:
    """
    Convenience function to analyze features.

    Args:
        model: Trained model
        X: Feature matrix
        y: Target values
        feature_names: List of feature names

    Returns:
        FeatureAnalysisResult
    """
    analyzer = FeatureAnalyzer(feature_names)
    return analyzer.analyze(model, X, y)


def get_top_features(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    n: int = 10
) -> List[str]:
    """Get names of top N most important features."""
    analyzer = FeatureAnalyzer(feature_names)
    result = analyzer.analyze(model, X)
    return [fi.feature_name for fi in result.get_top_features(n)]
