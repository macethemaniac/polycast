"""Model monitoring and drift detection.

Tracks model performance over time and detects concept drift.
"""

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Storage
MONITORING_DIR = Path(__file__).resolve().parents[2] / "data" / "monitoring"
MONITORING_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_LOG_FILE = MONITORING_DIR / "predictions.jsonl"
METRICS_LOG_FILE = MONITORING_DIR / "metrics.jsonl"


@dataclass
class PredictionLog:
    """Single prediction record."""
    timestamp: float
    model_name: str
    prediction: float
    actual: Optional[float] = None
    features: Dict[str, float] = field(default_factory=dict)
    market_id: str = ""
    opportunity_type: str = ""

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "prediction": self.prediction,
            "actual": self.actual,
            "features": self.features,
            "market_id": self.market_id,
            "opportunity_type": self.opportunity_type,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "PredictionLog":
        return cls(**d)


@dataclass
class DriftAlert:
    """Alert for detected drift."""
    alert_type: str  # "psi", "accuracy", "distribution"
    severity: str  # "low", "medium", "high"
    metric_name: str
    current_value: float
    baseline_value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)
    details: str = ""

    def to_dict(self) -> Dict:
        return {
            "alert_type": self.alert_type,
            "severity": self.severity,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "baseline_value": self.baseline_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
            "details": self.details,
        }


class ModelMonitor:
    """
    Monitors model performance over time.

    Tracks predictions, actual outcomes, and performance metrics.
    """

    def __init__(
        self,
        window_size: int = 1000,
        baseline_size: int = 500
    ):
        """
        Initialize monitor.

        Args:
            window_size: Number of recent predictions to keep
            baseline_size: Number of predictions for baseline comparison
        """
        self.window_size = window_size
        self.baseline_size = baseline_size

        self.predictions: deque = deque(maxlen=window_size)
        self.baseline_predictions: List[PredictionLog] = []
        self.metrics_history: deque = deque(maxlen=1000)
        self.alerts: List[DriftAlert] = []

        self._load_history()

    def _load_history(self) -> None:
        """Load prediction history from disk."""
        if PREDICTION_LOG_FILE.exists():
            try:
                with open(PREDICTION_LOG_FILE, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            pred = PredictionLog.from_dict(data)
                            self.predictions.append(pred)

                            # Build baseline from first N predictions
                            if len(self.baseline_predictions) < self.baseline_size:
                                self.baseline_predictions.append(pred)

                logger.info(f"Loaded {len(self.predictions)} prediction records")
            except Exception as e:
                logger.error(f"Error loading prediction history: {e}")

    def _save_prediction(self, pred: PredictionLog) -> None:
        """Save prediction to disk."""
        try:
            with open(PREDICTION_LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(pred.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")

    def log_prediction(
        self,
        prediction: float,
        model_name: str = "ensemble",
        features: Optional[Dict[str, float]] = None,
        market_id: str = "",
        opportunity_type: str = ""
    ) -> None:
        """Log a new prediction."""
        pred = PredictionLog(
            timestamp=time.time(),
            model_name=model_name,
            prediction=prediction,
            features=features or {},
            market_id=market_id,
            opportunity_type=opportunity_type,
        )

        self.predictions.append(pred)
        self._save_prediction(pred)

        # Update baseline if needed
        if len(self.baseline_predictions) < self.baseline_size:
            self.baseline_predictions.append(pred)

    def log_actual(self, market_id: str, actual: float) -> None:
        """Log actual outcome for a prediction."""
        for pred in reversed(self.predictions):
            if pred.market_id == market_id and pred.actual is None:
                pred.actual = actual
                break

    def get_recent_predictions(self, n: int = 100) -> List[PredictionLog]:
        """Get most recent predictions."""
        return list(self.predictions)[-n:]

    def get_labeled_predictions(self) -> List[PredictionLog]:
        """Get predictions with actual outcomes."""
        return [p for p in self.predictions if p.actual is not None]

    def calculate_accuracy(self, threshold: float = 0.5) -> float:
        """Calculate accuracy on labeled predictions."""
        labeled = self.get_labeled_predictions()
        if not labeled:
            return 0.0

        correct = sum(
            1 for p in labeled
            if (p.prediction >= threshold) == (p.actual >= threshold)
        )
        return correct / len(labeled)

    def calculate_auc(self) -> float:
        """Calculate AUC on labeled predictions."""
        labeled = self.get_labeled_predictions()
        if len(labeled) < 2:
            return 0.5

        try:
            from sklearn.metrics import roc_auc_score
            y_true = [p.actual for p in labeled]
            y_pred = [p.prediction for p in labeled]

            # Need both classes
            if len(set(y_true)) < 2:
                return 0.5

            return roc_auc_score(y_true, y_pred)
        except ImportError:
            return 0.5

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate current performance metrics."""
        labeled = self.get_labeled_predictions()

        metrics = {
            "total_predictions": len(self.predictions),
            "labeled_predictions": len(labeled),
            "accuracy": self.calculate_accuracy(),
            "auc": self.calculate_auc(),
        }

        # Prediction distribution stats
        if self.predictions:
            preds = [p.prediction for p in self.predictions]
            metrics["pred_mean"] = float(np.mean(preds))
            metrics["pred_std"] = float(np.std(preds))
            metrics["pred_min"] = float(np.min(preds))
            metrics["pred_max"] = float(np.max(preds))

        # Calibration (if we have labeled data)
        if labeled:
            actuals = [p.actual for p in labeled]
            metrics["actual_positive_rate"] = sum(1 for a in actuals if a > 0.5) / len(actuals)

        # Log metrics
        metrics["timestamp"] = time.time()
        self.metrics_history.append(metrics)

        return metrics

    def get_metrics_trend(self, metric_name: str, n: int = 10) -> List[float]:
        """Get trend of a specific metric."""
        return [
            m.get(metric_name, 0)
            for m in list(self.metrics_history)[-n:]
        ]


class DriftDetector:
    """
    Detects concept drift in model predictions.

    Uses PSI, accuracy degradation, and distribution shifts.
    """

    # Thresholds
    PSI_LOW = 0.1
    PSI_MEDIUM = 0.2
    PSI_HIGH = 0.25

    ACCURACY_DROP_LOW = 0.05
    ACCURACY_DROP_MEDIUM = 0.10
    ACCURACY_DROP_HIGH = 0.15

    def __init__(self, monitor: ModelMonitor):
        self.monitor = monitor

    def calculate_psi(
        self,
        baseline: List[float],
        current: List[float],
        n_bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI measures distribution shift between baseline and current.
        """
        if not baseline or not current:
            return 0.0

        # Create bins
        min_val = min(min(baseline), min(current))
        max_val = max(max(baseline), max(current))
        bins = np.linspace(min_val, max_val, n_bins + 1)

        # Calculate histograms
        baseline_hist, _ = np.histogram(baseline, bins=bins)
        current_hist, _ = np.histogram(current, bins=bins)

        # Normalize
        baseline_pct = baseline_hist / len(baseline)
        current_pct = current_hist / len(current)

        # Avoid division by zero
        baseline_pct = np.clip(baseline_pct, 0.0001, 1)
        current_pct = np.clip(current_pct, 0.0001, 1)

        # Calculate PSI
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
        return float(psi)

    def detect_prediction_drift(self, window_size: int = 500) -> Optional[DriftAlert]:
        """Detect drift in prediction distribution."""
        if len(self.monitor.baseline_predictions) < 100:
            return None

        baseline_preds = [p.prediction for p in self.monitor.baseline_predictions]
        recent_preds = [p.prediction for p in list(self.monitor.predictions)[-window_size:]]

        if len(recent_preds) < 100:
            return None

        psi = self.calculate_psi(baseline_preds, recent_preds)

        if psi > self.PSI_HIGH:
            severity = "high"
        elif psi > self.PSI_MEDIUM:
            severity = "medium"
        elif psi > self.PSI_LOW:
            severity = "low"
        else:
            return None

        return DriftAlert(
            alert_type="psi",
            severity=severity,
            metric_name="prediction_distribution",
            current_value=psi,
            baseline_value=0.0,
            threshold=self.PSI_LOW,
            details=f"PSI={psi:.3f} indicates {'significant' if psi > self.PSI_MEDIUM else 'moderate'} distribution shift"
        )

    def detect_accuracy_drift(self, window_size: int = 200) -> Optional[DriftAlert]:
        """Detect drift in model accuracy."""
        labeled = self.monitor.get_labeled_predictions()
        if len(labeled) < window_size:
            return None

        # Baseline accuracy (first half)
        baseline = labeled[:len(labeled)//2]
        baseline_acc = sum(
            1 for p in baseline
            if (p.prediction >= 0.5) == (p.actual >= 0.5)
        ) / len(baseline)

        # Current accuracy (recent)
        current = labeled[-window_size:]
        current_acc = sum(
            1 for p in current
            if (p.prediction >= 0.5) == (p.actual >= 0.5)
        ) / len(current)

        drop = baseline_acc - current_acc

        if drop > self.ACCURACY_DROP_HIGH:
            severity = "high"
        elif drop > self.ACCURACY_DROP_MEDIUM:
            severity = "medium"
        elif drop > self.ACCURACY_DROP_LOW:
            severity = "low"
        else:
            return None

        return DriftAlert(
            alert_type="accuracy",
            severity=severity,
            metric_name="accuracy",
            current_value=current_acc,
            baseline_value=baseline_acc,
            threshold=self.ACCURACY_DROP_LOW,
            details=f"Accuracy dropped from {baseline_acc:.1%} to {current_acc:.1%}"
        )

    def detect_feature_drift(
        self,
        feature_name: str,
        window_size: int = 500
    ) -> Optional[DriftAlert]:
        """Detect drift in a specific feature."""
        baseline = [
            p.features.get(feature_name, 0)
            for p in self.monitor.baseline_predictions
            if feature_name in p.features
        ]

        recent = [
            p.features.get(feature_name, 0)
            for p in list(self.monitor.predictions)[-window_size:]
            if feature_name in p.features
        ]

        if len(baseline) < 50 or len(recent) < 50:
            return None

        psi = self.calculate_psi(baseline, recent)

        if psi > self.PSI_MEDIUM:
            severity = "high" if psi > self.PSI_HIGH else "medium"
            return DriftAlert(
                alert_type="feature_drift",
                severity=severity,
                metric_name=f"feature:{feature_name}",
                current_value=psi,
                baseline_value=0.0,
                threshold=self.PSI_MEDIUM,
                details=f"Feature '{feature_name}' distribution shifted (PSI={psi:.3f})"
            )

        return None

    def detect_all(self) -> List[DriftAlert]:
        """Run all drift detection checks."""
        alerts = []

        # Prediction drift
        pred_alert = self.detect_prediction_drift()
        if pred_alert:
            alerts.append(pred_alert)

        # Accuracy drift
        acc_alert = self.detect_accuracy_drift()
        if acc_alert:
            alerts.append(acc_alert)

        # Feature drift (check key features)
        key_features = ["yes_price", "edge_pct", "similarity_score", "log_volume"]
        for feature in key_features:
            feature_alert = self.detect_feature_drift(feature)
            if feature_alert:
                alerts.append(feature_alert)

        return alerts


class PerformanceTracker:
    """
    Tracks model performance over time with rolling windows.
    """

    def __init__(self, monitor: ModelMonitor):
        self.monitor = monitor
        self.detector = DriftDetector(monitor)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        metrics = self.monitor.calculate_metrics()
        drift_alerts = self.detector.detect_all()

        return {
            "metrics": metrics,
            "drift_alerts": [a.to_dict() for a in drift_alerts],
            "health_status": self._determine_health(drift_alerts),
            "recommendations": self._get_recommendations(drift_alerts),
        }

    def _determine_health(self, alerts: List[DriftAlert]) -> str:
        """Determine overall model health."""
        if not alerts:
            return "healthy"

        severities = [a.severity for a in alerts]
        if "high" in severities:
            return "degraded"
        elif "medium" in severities:
            return "warning"
        return "healthy"

    def _get_recommendations(self, alerts: List[DriftAlert]) -> List[str]:
        """Generate recommendations based on alerts."""
        recommendations = []

        for alert in alerts:
            if alert.alert_type == "accuracy":
                recommendations.append("Consider retraining model with recent data")
            elif alert.alert_type == "psi":
                recommendations.append("Market conditions may have changed significantly")
            elif alert.alert_type == "feature_drift":
                recommendations.append(f"Monitor feature '{alert.metric_name}' closely")

        if not recommendations:
            recommendations.append("Model performance is stable")

        return recommendations


# Convenience functions
_monitor: Optional[ModelMonitor] = None
_tracker: Optional[PerformanceTracker] = None


def get_monitor() -> ModelMonitor:
    """Get or create singleton monitor."""
    global _monitor
    if _monitor is None:
        _monitor = ModelMonitor()
    return _monitor


def get_tracker() -> PerformanceTracker:
    """Get or create singleton tracker."""
    global _tracker
    if _tracker is None:
        _tracker = PerformanceTracker(get_monitor())
    return _tracker


def log_prediction(prediction: float, **kwargs) -> None:
    """Log a model prediction."""
    monitor = get_monitor()
    monitor.log_prediction(prediction, **kwargs)


def check_model_health() -> Dict[str, Any]:
    """Check overall model health."""
    tracker = get_tracker()
    return tracker.get_performance_summary()
