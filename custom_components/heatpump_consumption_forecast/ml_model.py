"""Local machine-learning model for Heat Pump Consumption Forecast.

The ML model is deliberately optional at runtime:
- it only becomes active after 30 completed daily samples,
- it is trained and stored locally in /config/.storage,
- the rule model remains the mandatory fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging
import pickle

_LOGGER = logging.getLogger(__name__)

ML_MINIMUM_DAYS = 30
ML_OPTIMIZED_DAYS = 90

FEATURE_NAMES = (
    "avg_temperature_c",
    "persons",
    "occupied_units",
    "occupied_area_sqm",
    "month",
    "weekday",
    "heating_basis_kwh",
    "dhw_basis_kwh",
)


@dataclass
class HeatPumpMLPrediction:
    """Prediction result returned by the ML model."""

    value: float | None
    available: bool
    error: str | None = None


def _optional_float(value: Any) -> float | None:
    """Return optional float."""
    try:
        return float(str(value).replace(",", ".")) if value is not None else None
    except (TypeError, ValueError):
        return None


def completed_training_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return completed samples with a valid final target."""
    valid: list[dict[str, Any]] = []
    for sample in samples or []:
        if not isinstance(sample, dict) or not sample.get("completed"):
            continue
        target = _optional_float(sample.get("actual_total_kwh_final"))
        if target is None or not 0 <= target <= 500:
            continue
        valid.append(sample)
    return valid


def build_ml_features(
    *,
    avg_temperature_c: float | None,
    persons: float,
    occupied_units: int,
    occupied_area_sqm: float,
    month: int,
    weekday: int,
    heating_basis_kwh: float,
    dhw_basis_kwh: float,
) -> dict[str, float]:
    """Build the feature dictionary used for training and prediction."""
    return {
        "avg_temperature_c": float(avg_temperature_c if avg_temperature_c is not None else 10.0),
        "persons": float(persons or 0.0),
        "occupied_units": float(occupied_units or 0),
        "occupied_area_sqm": float(occupied_area_sqm or 0.0),
        "month": float(month or 1),
        "weekday": float(weekday or 0),
        "heating_basis_kwh": float(heating_basis_kwh or 0.0),
        "dhw_basis_kwh": float(dhw_basis_kwh or 0.0),
    }


def _features_from_sample(sample: dict[str, Any]) -> dict[str, float] | None:
    """Build features from one stored training sample."""
    try:
        return build_ml_features(
            avg_temperature_c=_optional_float(sample.get("avg_temperature_c") or sample.get("current_temperature_c")),
            persons=float(sample.get("persons") or 0.0),
            occupied_units=int(sample.get("occupied_units") or 0),
            occupied_area_sqm=float(sample.get("occupied_area_sqm") or sample.get("total_area_sqm") or 0.0),
            month=int(sample.get("month") or 1),
            weekday=int(sample.get("weekday") or 0),
            heating_basis_kwh=float(sample.get("estimated_heating_basis_kwh") or 0.0),
            dhw_basis_kwh=float(sample.get("estimated_dhw_basis_kwh") or 0.0),
        )
    except (TypeError, ValueError):
        return None


def _feature_vector(features: dict[str, float]) -> list[float]:
    """Convert a feature dictionary to a stable vector."""
    return [float(features.get(name, 0.0) or 0.0) for name in FEATURE_NAMES]


class HeatPumpMLModel:
    """Wrapper around a local RandomForestRegressor."""

    def __init__(self) -> None:
        """Initialize wrapper."""
        self._model: Any | None = None
        self._metadata: dict[str, Any] = {}

    def train_or_load(self, samples: list[dict[str, Any]], model_path: Path) -> dict[str, Any]:
        """Train the model if enough samples exist, otherwise keep waiting."""
        completed = completed_training_samples(samples)
        completed_count = len(completed)
        base_status = {
            "completed_days": completed_count,
            "minimum_days": ML_MINIMUM_DAYS,
            "recommended_days": ML_OPTIMIZED_DAYS,
            "model_file": str(model_path),
            "features": list(FEATURE_NAMES),
            "algorithm": "RandomForestRegressor",
        }

        if completed_count < ML_MINIMUM_DAYS:
            self._load_existing_model(model_path)
            return {
                **base_status,
                "status": "Wartet auf Trainingsdaten",
                "active": False,
                "optimized": False,
                "model_file_exists": model_path.exists(),
                "message": f"Sammelt Daten ({completed_count}/{ML_MINIMUM_DAYS})",
            }

        try:
            from sklearn.ensemble import RandomForestRegressor
        except Exception as err:  # noqa: BLE001
            self._load_existing_model(model_path)
            return {
                **base_status,
                "status": "Wartet auf Trainingsdaten",
                "active": False,
                "optimized": False,
                "model_file_exists": model_path.exists(),
                "last_error": f"scikit-learn nicht verfügbar: {err}",
            }

        x_values: list[list[float]] = []
        y_values: list[float] = []
        for sample in completed:
            features = _features_from_sample(sample)
            target = _optional_float(sample.get("actual_total_kwh_final"))
            if features is None or target is None:
                continue
            if not 0 <= target <= 500:
                continue
            x_values.append(_feature_vector(features))
            y_values.append(float(target))

        if len(x_values) < ML_MINIMUM_DAYS:
            self._load_existing_model(model_path)
            return {
                **base_status,
                "status": "Wartet auf Trainingsdaten",
                "active": False,
                "optimized": False,
                "usable_samples": len(x_values),
                "model_file_exists": model_path.exists(),
                "message": "Nicht genug verwendbare abgeschlossene Trainingsdaten.",
            }

        estimator_count = 80 if len(x_values) < ML_OPTIMIZED_DAYS else 140
        model = RandomForestRegressor(
            n_estimators=estimator_count,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=1,
        )
        model.fit(x_values, y_values)

        self._model = model
        self._metadata = {
            **base_status,
            "status": "Optimiert" if completed_count >= ML_OPTIMIZED_DAYS else "Aktiv",
            "active": True,
            "optimized": completed_count >= ML_OPTIMIZED_DAYS,
            "usable_samples": len(x_values),
            "n_estimators": estimator_count,
        }
        self._save_model(model_path)
        return {**self._metadata, "model_file_exists": model_path.exists()}

    def predict(self, features: dict[str, float]) -> HeatPumpMLPrediction:
        """Predict total daily kWh from features."""
        if self._model is None:
            return HeatPumpMLPrediction(value=None, available=False, error="ML-Modell nicht geladen")
        try:
            value = float(self._model.predict([_feature_vector(features)])[0])
        except Exception as err:  # noqa: BLE001
            return HeatPumpMLPrediction(value=None, available=False, error=str(err))
        return HeatPumpMLPrediction(value=round(value, 3), available=True)

    def _save_model(self, model_path: Path) -> None:
        """Persist model and metadata locally."""
        try:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = model_path.with_suffix(".pkl.tmp")
            with tmp_path.open("wb") as file_obj:
                pickle.dump({"model": self._model, "metadata": self._metadata}, file_obj)
            tmp_path.replace(model_path)
        except Exception as err:  # noqa: BLE001
            _LOGGER.debug("Could not save ML model %s: %s", model_path, err)

    def _load_existing_model(self, model_path: Path) -> None:
        """Load an existing model if present."""
        if self._model is not None or not model_path.exists():
            return
        try:
            with model_path.open("rb") as file_obj:
                payload = pickle.load(file_obj)
            if isinstance(payload, dict):
                self._model = payload.get("model")
                self._metadata = payload.get("metadata") or {}
        except Exception as err:  # noqa: BLE001
            _LOGGER.debug("Could not load ML model %s: %s", model_path, err)
