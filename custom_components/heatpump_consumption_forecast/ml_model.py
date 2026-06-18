"""Local dependency-free machine-learning model for Heat Pump Consumption Forecast.

This module intentionally uses only the Python standard library.
It stores a compact local similarity model in model.pkl and keeps the
existing rule model as mandatory fallback in sensor.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging
import math
import pickle
from datetime import datetime

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

ALGORITHM_NAME = "Lokales Ähnlichkeitsmodell"


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


def _stats(vectors: list[list[float]]) -> tuple[list[float], list[float]]:
    """Return per-feature mean and spread for normalization."""
    if not vectors:
        return [0.0 for _ in FEATURE_NAMES], [1.0 for _ in FEATURE_NAMES]

    means: list[float] = []
    spreads: list[float] = []
    for idx in range(len(FEATURE_NAMES)):
        values = [row[idx] for row in vectors]
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / max(len(values), 1)
        spread = math.sqrt(variance)
        # Avoid zero division. A minimum spread keeps stable features from dominating.
        means.append(mean_value)
        spreads.append(max(spread, 1.0))
    return means, spreads


class HeatPumpMLModel:
    """Small local weighted-similarity model.

    The model keeps the completed training samples as normalized feature vectors.
    For a prediction it finds similar historical days and returns a weighted
    average of their final daily consumption.

    This is not a Random Forest, but it is deterministic, transparent, fast,
    HACS-friendly, Raspberry-Pi-friendly and requires no external packages.
    """

    def __init__(self) -> None:
        """Initialize wrapper."""
        self._model: dict[str, Any] | None = None
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
            "model_file_exists": model_path.exists(),
            "features": list(FEATURE_NAMES),
            "algorithm": ALGORITHM_NAME,
            "external_dependencies": [],
        }

        if completed_count < ML_MINIMUM_DAYS:
            self._load_existing_model(model_path)
            return {
                **base_status,
                "status": "Wartet auf Trainingsdaten",
                "active": False,
                "optimized": False,
                "model_file_exists": model_path.exists(),
                "usable_samples": completed_count,
                "last_training": None,
                "message": f"Sammelt Daten ({completed_count}/{ML_MINIMUM_DAYS})",
            }

        vectors: list[list[float]] = []
        targets: list[float] = []
        source_dates: list[str | None] = []
        for sample in completed:
            features = _features_from_sample(sample)
            target = _optional_float(sample.get("actual_total_kwh_final"))
            if features is None or target is None or not 0 <= target <= 500:
                continue
            vectors.append(_feature_vector(features))
            targets.append(float(target))
            source_dates.append(sample.get("date"))

        if len(vectors) < ML_MINIMUM_DAYS:
            self._load_existing_model(model_path)
            return {
                **base_status,
                "status": "Wartet auf Trainingsdaten",
                "active": False,
                "optimized": False,
                "usable_samples": len(vectors),
                "model_file_exists": model_path.exists(),
                "last_training": None,
                "message": "Nicht genug verwendbare abgeschlossene Trainingsdaten.",
            }

        means, spreads = _stats(vectors)
        self._model = {
            "vectors": vectors,
            "targets": targets,
            "means": means,
            "spreads": spreads,
            "source_dates": source_dates,
            "feature_names": list(FEATURE_NAMES),
        }
        self._metadata = {
            **base_status,
            "status": "Optimiert" if completed_count >= ML_OPTIMIZED_DAYS else "Aktiv",
            "active": True,
            "optimized": completed_count >= ML_OPTIMIZED_DAYS,
            "usable_samples": len(vectors),
            "model_type": "weighted_similarity",
            "nearest_neighbors": min(12, len(vectors)),
            "last_training": datetime.now().isoformat(),
        }
        self._save_model(model_path)
        return {**self._metadata, "model_file_exists": model_path.exists()}

    def predict(self, features: dict[str, float]) -> HeatPumpMLPrediction:
        """Predict total daily kWh from features."""
        if self._model is None:
            return HeatPumpMLPrediction(value=None, available=False, error="ML-Modell nicht geladen")

        vectors = self._model.get("vectors") or []
        targets = self._model.get("targets") or []
        means = self._model.get("means") or []
        spreads = self._model.get("spreads") or []
        if not vectors or not targets or len(vectors) != len(targets):
            return HeatPumpMLPrediction(value=None, available=False, error="ML-Modell enthält keine gültigen Daten")

        query = _feature_vector(features)
        distances: list[tuple[float, float]] = []
        for vector, target in zip(vectors, targets, strict=False):
            distance = 0.0
            for idx, value in enumerate(query):
                spread = float(spreads[idx] if idx < len(spreads) else 1.0) or 1.0
                # Some feature weights. Temperature, heating and DHW basis are the strongest signals.
                weight = {
                    "avg_temperature_c": 1.35,
                    "heating_basis_kwh": 1.25,
                    "dhw_basis_kwh": 1.15,
                    "occupied_area_sqm": 0.85,
                    "persons": 0.75,
                    "occupied_units": 0.55,
                    "month": 0.35,
                    "weekday": 0.25,
                }.get(FEATURE_NAMES[idx], 1.0)
                normalized_delta = (float(value) - float(vector[idx])) / spread
                distance += weight * normalized_delta * normalized_delta
            distances.append((math.sqrt(distance), float(target)))

        if not distances:
            return HeatPumpMLPrediction(value=None, available=False, error="Keine Vergleichstage gefunden")

        distances.sort(key=lambda item: item[0])
        neighbors = distances[: min(12, len(distances))]
        weighted_sum = 0.0
        weight_sum = 0.0
        for distance, target in neighbors:
            weight = 1.0 / (0.15 + distance)
            weighted_sum += target * weight
            weight_sum += weight

        if weight_sum <= 0:
            return HeatPumpMLPrediction(value=None, available=False, error="Ungültige Gewichtung")

        value = weighted_sum / weight_sum
        if not 0 <= value <= 500:
            return HeatPumpMLPrediction(value=None, available=False, error="Unplausible ML-Prognose")
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
