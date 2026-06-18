"""Sensors for Heat Pump Consumption Forecast."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any
import json
import logging
import os
import re

from homeassistant.components import recorder
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN, UnitOfEnergy
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from .const import (
    CONF_DHW_DAILY_ENERGY_SENSOR,
    CONF_FIXED_PERSONS,
    CONF_HEATING_DAILY_ENERGY_SENSOR,
    CONF_HEATING_THRESHOLD_TEMP,
    CONF_HEATPUMP_DAILY_ENERGY_SENSOR,
    CONF_HEATPUMP_TOTAL_ENERGY_SENSOR,
    CONF_OUTDOOR_TEMP_SENSOR,
    CONF_UNITS,
    CONF_WEATHER_ENTITY,
    DOMAIN,
    SCAN_INTERVAL_MINUTES,
)
from .ml_model import (
    HeatPumpMLModel,
    ML_MINIMUM_DAYS,
    ML_OPTIMIZED_DAYS,
    build_ml_features,
    completed_training_samples,
)

_LOGGER = logging.getLogger(__name__)
WH_UNITS = {"wh", "watt hour", "watt-hours"}
KWH_UNITS = {"kwh", "kilowatt hour", "kilowatt-hours"}
MWH_UNITS = {"mwh", "megawatt hour", "megawatt-hours"}
ENERGY_UNITS = WH_UNITS | KWH_UNITS | MWH_UNITS
STORAGE_VERSION = 1
STORAGE_KEY_PREFIX = f"{DOMAIN}_training_data"
STORAGE_DIR_NAME = DOMAIN
TRAINING_DATA_FILE = "training_data.json"
HEATING_CURVE_FILE = "heating_curve.json"
MODEL_FILE = "model.pkl"
MAX_TRAINING_SAMPLES = 370


@dataclass
class OccupancyInfo:
    """Occupancy summary for one day."""

    persons: float = 0.0
    occupied_units: int = 0
    total_units: int = 0
    occupied_area_sqm: float = 0.0
    total_area_sqm: float = 0.0
    unit_details: list[dict[str, Any]] | None = None


def _to_float(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", ".")) if value is not None else None
    except (TypeError, ValueError):
        return None


def _float_state(hass: HomeAssistant, entity_id: str | None) -> float | None:
    state = hass.states.get(entity_id) if entity_id else None
    return _to_float(state.state) if state else None


def _energy_kwh(hass: HomeAssistant, entity_id: str | None) -> float | None:
    value = _float_state(hass, entity_id)
    if value is None:
        return None
    state = hass.states.get(entity_id) if entity_id else None
    unit = str(state.attributes.get("unit_of_measurement", "") if state else "").strip().lower()
    if unit in WH_UNITS:
        return value / 1000.0
    if unit in MWH_UNITS:
        return value * 1000.0
    if unit and unit not in ENERGY_UNITS:
        return None
    return value


def _history_state_kwh(state: Any) -> float | None:
    value = _to_float(getattr(state, "state", None))
    if value is None:
        return None
    unit = str(getattr(state, "attributes", {}).get("unit_of_measurement", "")).strip().lower()
    if unit in WH_UNITS:
        return value / 1000.0
    if unit in MWH_UNITS:
        return value * 1000.0
    if unit and unit not in ENERGY_UNITS:
        return None
    return value


def _avg(values: list[float], days: int) -> float | None:
    relevant = values[-days:]
    return float(mean(relevant)) if relevant else None


def _parse_persons_from_text(text: str | None) -> float | None:
    if not text:
        return None
    normalized = str(text).replace("\n", " ")
    adults = re.search(r"(\d+(?:[,.]\d+)?)\s*(?:erw\.?|erwachsene?|adult|adults)", normalized, re.I)
    children = re.search(r"(\d+(?:[,.]\d+)?)\s*(?:kind(?:er)?|child|children)", normalized, re.I)
    if adults or children:
        return (_to_float(adults.group(1)) or 0.0 if adults else 0.0) + (_to_float(children.group(1)) or 0.0 if children else 0.0)
    persons = re.search(r"(\d+(?:[,.]\d+)?)\s*(?:pers\.?|personen?|person|persons|guests?|gäste|gaeste)", normalized, re.I)
    return _to_float(persons.group(1)) if persons else None


def _unit_area(unit: dict[str, Any]) -> float:
    return float(unit.get("area_sqm") or 0)


def _total_area(units: list[dict[str, Any]]) -> float:
    return sum(_unit_area(unit) for unit in units)


def _fixed_unit_persons(units: list[dict[str, Any]]) -> float:
    return sum(float(unit.get("fixed_persons") or 0) for unit in units)


def _current_occupancy(hass: HomeAssistant, units: list[dict[str, Any]], fallback_persons: float = 0.0) -> OccupancyInfo:
    info = OccupancyInfo(total_units=len(units), total_area_sqm=_total_area(units), unit_details=[])
    for unit in units:
        entity_id = unit.get("calendar_entity")
        state = hass.states.get(entity_id) if entity_id else None
        occupied = False
        persons = float(unit.get("fixed_persons") or 0)
        if entity_id:
            occupied = state is not None and state.state in {"on", "true"}
            if occupied and state:
                parsed = None
                for key in ("description", "message", "summary"):
                    parsed = _parse_persons_from_text(state.attributes.get(key))
                    if parsed is not None:
                        break
                persons = float(parsed if parsed is not None else persons)
            else:
                persons = 0.0
        else:
            occupied = persons > 0
        area = _unit_area(unit)
        if occupied:
            info.occupied_units += 1
            info.occupied_area_sqm += area
        info.persons += persons
        info.unit_details.append({"name": unit.get("name"), "occupied": occupied, "persons": persons, "area_sqm": area})
    if info.persons <= 0 and fallback_persons > 0:
        info.persons = fallback_persons
    return info


def _heating_degree_factor(avg_temp: float | None, threshold: float) -> float:
    if avg_temp is None:
        return 1.0
    heating_degrees = max(0.0, threshold - float(avg_temp))
    return 0.0 if heating_degrees <= 0 else max(0.15, min(2.8, heating_degrees / 8.0))


def _occupancy_factor(persons: float) -> float:
    return 1.0 + min(max(float(persons or 0), 0), 100) * 0.018


def _area_factor(area: float) -> float:
    return 1.0 if area <= 0 else max(0.75, min(1.35, 0.85 + (area / 500.0)))


def _label_source(source: str | None) -> str:
    return {
        "daily_history": "Tageshistorie",
        "daily": "Tageswert",
        "accumulated_delta": "Gesamtzähler-Differenz",
        "fallback": "Ersatzwert",
        "fallback_guardrail_selected_daily_too_high": "Ersatzwert nach Plausibilitätsprüfung",
    }.get(str(source or ""), str(source or "unbekannt"))


def _label_split_source(source: str | None) -> str:
    return {
        "total_minus_heating": "Gesamtverbrauch minus Heizverbrauch",
        "dedicated_sensors": "separate Verbrauchssensoren",
        "total_minus_dhw": "Gesamtverbrauch minus Warmwasserverbrauch",
        "estimated_split": "geschätzte Aufteilung",
        "dhw_guardrail_total_minus_heating": "Warmwasser abgeleitet: Gesamt minus Heizung",
        "heating_guardrail_total_minus_dhw": "Heizung abgeleitet: Gesamt minus Warmwasser",
    }.get(str(source or ""), str(source or "unbekannt"))


def _learn_heating_curve(temp_series: dict[str, float], heating_series: dict[str, float], threshold: float) -> dict[str, Any]:
    pairs: list[tuple[float, float]] = []
    for day, temp in (temp_series or {}).items():
        heat = (heating_series or {}).get(day)
        temp_f = _to_float(temp)
        heat_f = _to_float(heat)
        if temp_f is not None and heat_f is not None and -40 <= temp_f <= 50 and 0 <= heat_f <= 200:
            pairs.append((temp_f, heat_f))
    buckets = {"unter_5": [], "5_bis_10": [], "10_bis_15": [], "15_bis_heizgrenze": [], "ueber_heizgrenze": []}
    for temp, heat in pairs:
        if temp < 5:
            buckets["unter_5"].append(heat)
        elif temp < 10:
            buckets["5_bis_10"].append(heat)
        elif temp < 15:
            buckets["10_bis_15"].append(heat)
        elif temp < threshold:
            buckets["15_bis_heizgrenze"].append(heat)
        else:
            buckets["ueber_heizgrenze"].append(heat)
    paired_days = len(pairs)
    optimized = paired_days >= 90
    active = paired_days >= 30
    status = "Optimiert" if optimized else "Aktiv" if active else "Wird aufgebaut" if paired_days else "Keine Heizkurvendaten"
    return {
        "available": bool(pairs),
        "active": active,
        "ready": active,
        "status": status,
        "paired_days": paired_days,
        "minimum_days": 30,
        "recommended_days": 90,
        "optimized": optimized,
        "progress_percent": min(100, round((paired_days / 30) * 100)) if paired_days else 0,
        "threshold_c": threshold,
        "bucket_avgs_kwh": {name: round(float(mean(vals)), 2) if vals else None for name, vals in buckets.items()},
        "bucket_counts": {name: len(vals) for name, vals in buckets.items()},
        "pairs_last_30": [{"temperature_c": round(t, 2), "heating_kwh": round(h, 3)} for t, h in pairs[-30:]],
    }


def _heating_from_curve(avg_temp: float | None, curve: dict[str, Any], fallback: float, threshold: float) -> tuple[float, str]:
    if avg_temp is None or not curve.get("active"):
        return fallback * _heating_degree_factor(avg_temp, threshold), "regelbasis"
    if float(avg_temp) >= threshold:
        return 0.0, "heizkurve"
    buckets = curve.get("bucket_avgs_kwh") or {}
    temp = float(avg_temp)
    key = "unter_5" if temp < 5 else "5_bis_10" if temp < 10 else "10_bis_15" if temp < 15 else "15_bis_heizgrenze"
    value = buckets.get(key)
    if value is None:
        populated = [v for v in buckets.values() if v is not None]
        value = float(mean(populated)) if populated else fallback * _heating_degree_factor(avg_temp, threshold)
    return float(value or 0.0), "heizkurve"


def _plausible_forecast(value: float | None, rule_value: float | None) -> bool:
    value_f = _to_float(value)
    if value_f is None or not 0 <= value_f <= 500:
        return False
    if rule_value is None:
        return True
    rule_f = max(float(rule_value or 0), 0.1)
    return value_f <= max(80.0, rule_f * 3.0) and value_f >= max(0.0, rule_f * 0.15)


class HeatPumpForecastCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Coordinator calculating forecast values."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.entry = entry
        self._last_total_kwh: float | None = None
        self._estimated_today_from_total = 0.0
        self._training_samples: list[dict[str, Any]] | None = None
        self._storage_dir = Path(hass.config.path(".storage", STORAGE_DIR_NAME))
        self._training_data_path = self._storage_dir / TRAINING_DATA_FILE
        self._heating_curve_path = self._storage_dir / HEATING_CURVE_FILE
        self._model_path = self._storage_dir / MODEL_FILE
        self._ml_model = HeatPumpMLModel()
        super().__init__(hass, _LOGGER, name=DOMAIN, update_interval=timedelta(minutes=SCAN_INTERVAL_MINUTES))

    def _storage_paths_attributes(self) -> dict[str, str]:
        return {
            "Speicherordner": str(self._storage_dir),
            "Trainingsdaten_Datei": str(self._training_data_path),
            "Heizkurven_Datei": str(self._heating_curve_path),
            "ML_Modell_Datei": str(self._model_path),
        }

    async def _async_legacy_store_load(self) -> list[dict[str, Any]]:
        try:
            from homeassistant.helpers.storage import Store
            stored = await Store(self.hass, STORAGE_VERSION, f"{STORAGE_KEY_PREFIX}_{self.entry.entry_id}").async_load()
        except Exception as err:  # noqa: BLE001
            _LOGGER.debug("Could not load legacy training store: %s", err)
            return []
        raw = stored.get("samples", []) if isinstance(stored, dict) else []
        return [sample for sample in raw if isinstance(sample, dict)] if isinstance(raw, list) else []

    async def _async_load_training_samples(self) -> list[dict[str, Any]]:
        if self._training_samples is not None:
            return self._training_samples

        def _read_file() -> dict[str, Any] | None:
            if not self._training_data_path.exists():
                return None
            with self._training_data_path.open("r", encoding="utf-8") as file_obj:
                return json.load(file_obj)

        stored = None
        try:
            stored = await self.hass.async_add_executor_job(_read_file)
        except Exception as err:  # noqa: BLE001
            _LOGGER.debug("Could not load training data file %s: %s", self._training_data_path, err)
        raw = stored.get("samples", []) if isinstance(stored, dict) else []
        samples = [sample for sample in raw if isinstance(sample, dict)] if isinstance(raw, list) else []
        if not samples:
            samples = await self._async_legacy_store_load()
        self._training_samples = samples[-MAX_TRAINING_SAMPLES:]
        if samples and stored is None:
            await self._async_save_training_samples(self._training_samples)
        return self._training_samples

    async def _async_save_training_samples(self, samples: list[dict[str, Any]]) -> None:
        payload = {"version": 1, "domain": DOMAIN, "entry_id": self.entry.entry_id, "updated_at": dt_util.now().isoformat(), "samples": samples[-MAX_TRAINING_SAMPLES:]}

        def _write_file() -> None:
            os.makedirs(self._storage_dir, exist_ok=True)
            tmp_path = self._training_data_path.with_suffix(".json.tmp")
            with tmp_path.open("w", encoding="utf-8") as file_obj:
                json.dump(payload, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
            os.replace(tmp_path, self._training_data_path)

        await self.hass.async_add_executor_job(_write_file)

    async def _async_store_heating_curve(self, heating_curve: dict[str, Any]) -> None:
        payload = {"version": 1, "domain": DOMAIN, "entry_id": self.entry.entry_id, "updated_at": dt_util.now().isoformat(), "heating_curve": heating_curve}

        def _write_file() -> None:
            os.makedirs(self._storage_dir, exist_ok=True)
            tmp_path = self._heating_curve_path.with_suffix(".json.tmp")
            with tmp_path.open("w", encoding="utf-8") as file_obj:
                json.dump(payload, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
            os.replace(tmp_path, self._heating_curve_path)

        try:
            await self.hass.async_add_executor_job(_write_file)
        except Exception as err:  # noqa: BLE001
            _LOGGER.debug("Could not save heating curve file %s: %s", self._heating_curve_path, err)

    def _finalize_previous_training_samples(self, samples: list[dict[str, Any]], today_date: str, total_series: dict[str, Any], heating_series: dict[str, Any], dhw_series: dict[str, Any]) -> list[dict[str, Any]]:
        now_iso = dt_util.now().isoformat()
        for sample in samples:
            sample_date = sample.get("date")
            if not sample_date or sample_date == today_date:
                sample["completed"] = False
                sample.setdefault("actual_total_kwh_final", None)
                sample.setdefault("actual_heating_kwh_final", None)
                sample.setdefault("actual_dhw_kwh_final", None)
                sample.setdefault("forecast_error_kwh", None)
                sample.setdefault("forecast_abs_error_kwh", None)
                continue
            final_total_f = _to_float((total_series or {}).get(sample_date))
            if final_total_f is None:
                sample["completed"] = False
                sample.setdefault("completion_pending_reason", "Tagesendwert noch nicht im Recorder verfügbar")
                continue
            final_heating_f = _to_float((heating_series or {}).get(sample_date))
            final_dhw_f = _to_float((dhw_series or {}).get(sample_date))
            if final_dhw_f is None and final_heating_f is not None:
                final_dhw_f = max(final_total_f - final_heating_f, 0.0)
            if final_heating_f is None and final_dhw_f is not None:
                final_heating_f = max(final_total_f - final_dhw_f, 0.0)
            forecast_today_f = _to_float(sample.get("forecast_today_kwh"))
            sample["completed"] = True
            sample["completed_at"] = now_iso
            sample.pop("completion_pending_reason", None)
            sample["actual_total_kwh_final"] = round(final_total_f, 3)
            sample["actual_heating_kwh_final"] = round(final_heating_f, 3) if final_heating_f is not None else None
            sample["actual_dhw_kwh_final"] = round(final_dhw_f, 3) if final_dhw_f is not None else None
            if forecast_today_f is not None:
                error = final_total_f - forecast_today_f
                sample["forecast_error_kwh"] = round(error, 3)
                sample["forecast_abs_error_kwh"] = round(abs(error), 3)
        return samples

    async def _async_store_training_sample(self, sample: dict[str, Any], total_series: dict[str, Any] | None = None, heating_series: dict[str, Any] | None = None, dhw_series: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        samples = await self._async_load_training_samples()
        date_key = sample.get("date")
        if not date_key:
            return samples
        samples = self._finalize_previous_training_samples(samples, str(date_key), total_series or {}, heating_series or {}, dhw_series or {})
        for idx, existing in enumerate(samples):
            if existing.get("date") == date_key:
                samples[idx] = sample
                break
        else:
            samples.append(sample)
        samples = samples[-MAX_TRAINING_SAMPLES:]
        self._training_samples = samples
        await self._async_save_training_samples(samples)
        return samples

    async def _async_get_forecast_temperatures(self, weather_entity: str | None) -> dict[int, float | None]:
        result: dict[int, float | None] = {1: None, 2: None}
        if not weather_entity:
            return result
        forecast: list[dict[str, Any]] = []
        try:
            response = await self.hass.services.async_call("weather", "get_forecasts", {"entity_id": [weather_entity], "type": "daily"}, blocking=True, return_response=True)
            forecast = (response or {}).get(weather_entity, {}).get("forecast") or []
        except Exception as err:  # noqa: BLE001
            _LOGGER.debug("Could not fetch daily weather forecast via service: %s", err)
        if not forecast:
            state = self.hass.states.get(weather_entity)
            forecast = state.attributes.get("forecast") or [] if state else []
        for offset in (1, 2):
            if len(forecast) > offset:
                high = _to_float(forecast[offset].get("temperature"))
                low = _to_float(forecast[offset].get("templow") or forecast[offset].get("temperature_low"))
                result[offset] = round((high + low) / 2.0, 2) if high is not None and low is not None else high
        return result

    async def _async_get_calendar_occupancy_for_day(self, units: list[dict[str, Any]], offset_days: int, fallback_persons: float = 0.0) -> OccupancyInfo | None:
        calendar_entities = []
        for unit in units:
            entity_id = unit.get("calendar_entity")
            state = self.hass.states.get(entity_id) if entity_id else None
            if entity_id and state is not None and state.state not in {STATE_UNAVAILABLE, STATE_UNKNOWN}:
                calendar_entities.append(entity_id)
        if not calendar_entities:
            return None
        day = datetime.now().date() + timedelta(days=offset_days)
        try:
            response = await self.hass.services.async_call(
                "calendar",
                "get_events",
                {"entity_id": calendar_entities, "start_date_time": datetime.combine(day, datetime.min.time()).isoformat(), "end_date_time": datetime.combine(day + timedelta(days=1), datetime.min.time()).isoformat()},
                blocking=True,
                return_response=True,
            )
        except Exception as err:  # noqa: BLE001
            _LOGGER.debug("Could not fetch calendar events for day offset %s: %s", offset_days, err)
            return None
        info = OccupancyInfo(total_units=len(units), total_area_sqm=_total_area(units), unit_details=[])
        found_any = False
        for unit in units:
            entity_id = unit.get("calendar_entity")
            area = _unit_area(unit)
            events = (response.get(entity_id, {}) or {}).get("events") or [] if response and entity_id else []
            occupied = bool(events)
            persons = 0.0
            if occupied:
                found_any = True
                parsed_values = [_parse_persons_from_text(" ".join(str(event.get(key) or "") for key in ("description", "summary", "message"))) for event in events]
                parsed_values = [float(v) for v in parsed_values if v is not None]
                persons = max(parsed_values) if parsed_values else float(unit.get("fixed_persons") or 0)
                info.occupied_units += 1
                info.occupied_area_sqm += area
            info.persons += persons
            info.unit_details.append({"name": unit.get("name"), "occupied": occupied, "persons": persons, "area_sqm": area})
        if info.persons <= 0 and fallback_persons > 0:
            info.persons = fallback_persons
        return info if found_any else None

    async def _async_get_temperature_history_stats(self, entity_id: str | None, days: int = 30) -> dict[str, Any]:
        empty = {"available": False, "days": 0, "values": [], "series": {}, "avg_7": None, "avg_14": None, "avg_30": None, "today_avg": None}
        if not entity_id:
            return empty
        now = dt_util.now()

        def _read_history() -> list[Any]:
            try:
                from homeassistant.components.recorder import history
                return list((history.get_significant_states(self.hass, now - timedelta(days=days + 1), now, entity_ids=[entity_id], significant_changes_only=False, minimal_response=False, no_attributes=False)).get(entity_id, []) or [])
            except Exception as err:  # noqa: BLE001
                _LOGGER.debug("Could not read temperature history for %s: %s", entity_id, err)
                return []

        states = await recorder.get_instance(self.hass).async_add_executor_job(_read_history)
        grouped: dict[Any, list[float]] = {}
        for state in states:
            value = _to_float(getattr(state, "state", None))
            changed = getattr(state, "last_changed", None) or getattr(state, "last_updated", None)
            if value is None or value < -50 or value > 60 or changed is None:
                continue
            grouped.setdefault(dt_util.as_local(changed).date(), []).append(value)
        today = now.date()
        series = {day.isoformat(): round(float(mean(vals)), 2) for day, vals in sorted(grouped.items()) if day != today and vals}
        completed = list(series.values())
        today_vals = grouped.get(today, [])
        today_avg = round(float(mean(today_vals)), 2) if today_vals else None
        if not completed:
            return {**empty, "today_avg": today_avg}
        return {"available": True, "days": len(completed), "values": completed[-30:], "series": dict(list(series.items())[-30:]), "avg_7": round(_avg(completed, 7), 2), "avg_14": round(_avg(completed, 14), 2), "avg_30": round(_avg(completed, 30), 2), "today_avg": today_avg}

    async def _async_get_daily_history_stats(self, entity_id: str | None, days: int = 30) -> dict[str, Any]:
        empty = {"available": False, "days": 0, "values": [], "avg_7": None, "avg_14": None, "avg_30": None, "basis": None, "series": {}}
        if not entity_id:
            return empty
        now = dt_util.now()

        def _read_history() -> list[Any]:
            try:
                from homeassistant.components.recorder import history
                return list((history.get_significant_states(self.hass, now - timedelta(days=days + 1), now, entity_ids=[entity_id], significant_changes_only=False, minimal_response=False, no_attributes=False)).get(entity_id, []) or [])
            except Exception as err:  # noqa: BLE001
                _LOGGER.debug("Could not read recorder history for %s: %s", entity_id, err)
                return []

        states = await recorder.get_instance(self.hass).async_add_executor_job(_read_history)
        daily_max: dict[Any, float] = {}
        current_day = now.date()
        for state in states:
            value = _history_state_kwh(state)
            changed = getattr(state, "last_changed", None) or getattr(state, "last_updated", None)
            if value is None or value < 0 or value > 500 or changed is None:
                continue
            day = dt_util.as_local(changed).date()
            if day != current_day:
                daily_max[day] = max(daily_max.get(day, 0.0), value)
        series = {day.isoformat(): round(v, 3) for day, v in sorted(daily_max.items()) if v > 0}
        values = list(series.values())
        if not values:
            return empty
        avg_7 = _avg(values, 7)
        avg_14 = _avg(values, 14)
        avg_30 = _avg(values, 30)
        basis = avg_7 or avg_14 or avg_30
        return {"available": True, "days": len(values), "values": values[-30:], "series": dict(list(series.items())[-30:]), "avg_7": round(avg_7, 2), "avg_14": round(avg_14, 2), "avg_30": round(avg_30, 2), "basis": round(basis, 2) if basis is not None else None}

    async def _async_train_ml(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        def _train() -> dict[str, Any]:
            return self._ml_model.train_or_load(samples, self._model_path)
        try:
            return await self.hass.async_add_executor_job(_train)
        except Exception as err:  # noqa: BLE001
            return {"status": "Wartet auf Trainingsdaten", "active": False, "optimized": False, "completed_days": len(completed_training_samples(samples)), "minimum_days": ML_MINIMUM_DAYS, "recommended_days": ML_OPTIMIZED_DAYS, "model_file": str(self._model_path), "last_error": str(err)}

    async def _async_update_data(self) -> dict[str, Any]:
        config = {**self.entry.data, **self.entry.options}
        daily_entity = config.get(CONF_HEATPUMP_DAILY_ENERGY_SENSOR)
        total_entity = config.get(CONF_HEATPUMP_TOTAL_ENERGY_SENSOR)
        current_temp = _float_state(self.hass, config.get(CONF_OUTDOOR_TEMP_SENSOR))
        forecast_temps = await self._async_get_forecast_temperatures(config.get(CONF_WEATHER_ENTITY))
        units: list[dict[str, Any]] = list(config.get(CONF_UNITS) or [])
        fallback_persons = float(config.get(CONF_FIXED_PERSONS, 0) or 0)
        current_occ = _current_occupancy(self.hass, units, fallback_persons=fallback_persons)
        fixed_persons = _fixed_unit_persons(units)
        if current_occ.persons <= 0 and fixed_persons > 0:
            current_occ.persons = fixed_persons
        tomorrow_occ = await self._async_get_calendar_occupancy_for_day(units, 1, fallback_persons=current_occ.persons) or current_occ
        day_after_occ = await self._async_get_calendar_occupancy_for_day(units, 2, fallback_persons=current_occ.persons) or tomorrow_occ

        heating_entity = config.get(CONF_HEATING_DAILY_ENERGY_SENSOR)
        dhw_entity = config.get(CONF_DHW_DAILY_ENERGY_SENSOR)
        heating_threshold = float(config.get(CONF_HEATING_THRESHOLD_TEMP, 17.0) or 17.0)
        today_so_far_kwh = _energy_kwh(self.hass, daily_entity)
        today_heating_so_far_kwh = _energy_kwh(self.hass, heating_entity)
        today_dhw_so_far_kwh = _energy_kwh(self.hass, dhw_entity)

        history_stats = await self._async_get_daily_history_stats(daily_entity)
        heating_history = await self._async_get_daily_history_stats(heating_entity)
        dhw_history = await self._async_get_daily_history_stats(dhw_entity)
        temp_history = await self._async_get_temperature_history_stats(config.get(CONF_OUTDOOR_TEMP_SENSOR))
        heating_curve = _learn_heating_curve(temp_history.get("series") or {}, heating_history.get("series") or {}, heating_threshold)
        await self._async_store_heating_curve(heating_curve)

        today_avg_temp = temp_history.get("today_avg") if temp_history.get("today_avg") is not None else current_temp
        tomorrow_avg_temp = forecast_temps.get(1) if forecast_temps.get(1) is not None else today_avg_temp
        day_after_avg_temp = forecast_temps.get(2) if forecast_temps.get(2) is not None else tomorrow_avg_temp

        source = "daily_history" if history_stats.get("basis") is not None else "daily"
        baseline_kwh = history_stats.get("basis") or today_so_far_kwh
        if baseline_kwh is None or baseline_kwh < 0:
            total_kwh = _energy_kwh(self.hass, total_entity)
            source = "accumulated_delta"
            if total_kwh is not None:
                if self._last_total_kwh is not None and total_kwh >= self._last_total_kwh:
                    delta = total_kwh - self._last_total_kwh
                    if 0 <= delta < 100:
                        self._estimated_today_from_total += delta
                self._last_total_kwh = total_kwh
            baseline_kwh = self._estimated_today_from_total if self._estimated_today_from_total > 0 else None
            today_so_far_kwh = baseline_kwh
        if baseline_kwh is None or baseline_kwh <= 0:
            baseline_kwh = 10.0
            source = "fallback"
        if baseline_kwh > 500:
            baseline_kwh = 10.0
            source = "fallback_guardrail_selected_daily_too_high"

        heating_baseline = heating_history.get("basis")
        dhw_baseline = dhw_history.get("basis")
        split_source = "estimated_split"
        if heating_baseline is not None and dhw_baseline is None:
            dhw_baseline = round(max(float(baseline_kwh) - float(heating_baseline or 0), 0.0), 2)
            split_source = "total_minus_heating"
        elif heating_baseline is not None and dhw_baseline is not None:
            split_source = "dedicated_sensors"
        elif heating_baseline is None and dhw_baseline is not None:
            heating_baseline = round(max(float(baseline_kwh) - float(dhw_baseline or 0), 0.0), 2)
            split_source = "total_minus_dhw"
        else:
            ref = temp_history.get("avg_7") if temp_history.get("avg_7") is not None else today_avg_temp
            share = max(0.0, min(0.75, _heating_degree_factor(ref, heating_threshold) / 2.8))
            heating_baseline = round(float(baseline_kwh) * share, 2)
            dhw_baseline = round(max(float(baseline_kwh) - float(heating_baseline or 0), float(baseline_kwh) * 0.25), 2)
        heating_baseline = max(0.0, float(heating_baseline or 0.0))
        dhw_baseline = max(0.0, float(dhw_baseline or 0.0))
        if dhw_baseline > max(20.0, float(baseline_kwh) * 3.0):
            dhw_baseline = round(max(float(baseline_kwh) - heating_baseline, 0.0), 2)
            split_source = "dhw_guardrail_total_minus_heating"
        if heating_baseline > max(100.0, float(baseline_kwh) * 3.0):
            heating_baseline = round(max(float(baseline_kwh) - dhw_baseline, 0.0), 2)
            split_source = "heating_guardrail_total_minus_dhw"

        def _rule_forecast_for(avg_temp: float | None, occ: OccupancyInfo) -> dict[str, Any]:
            heat_active = avg_temp is None or avg_temp < heating_threshold
            learned_heat, heating_method = _heating_from_curve(avg_temp, heating_curve, heating_baseline, heating_threshold)
            heating_part = learned_heat * _area_factor(occ.occupied_area_sqm or occ.total_area_sqm)
            dhw_part = dhw_baseline * _occupancy_factor(occ.persons)
            if not heat_active:
                heating_part = 0.0
            total = heating_part + dhw_part
            return {"total_kwh": round(total, 2), "heating_kwh": round(heating_part, 2), "dhw_kwh": round(dhw_part, 2), "heating_active": bool(heat_active), "heating_factor": round(_heating_degree_factor(avg_temp, heating_threshold), 3), "heating_method": heating_method}

        today_rule_calc = _rule_forecast_for(today_avg_temp, current_occ)
        tomorrow_rule_calc = _rule_forecast_for(tomorrow_avg_temp, tomorrow_occ)
        day_after_rule_calc = _rule_forecast_for(day_after_avg_temp, day_after_occ)
        today_floor = today_so_far_kwh if today_so_far_kwh is not None and 0 <= today_so_far_kwh < 500 else 0.0
        rule_today = round(max(today_floor, today_rule_calc["total_kwh"]), 2)
        rule_tomorrow = tomorrow_rule_calc["total_kwh"]
        rule_day_after = day_after_rule_calc["total_kwh"]

        now_local = dt_util.now()
        remaining_day_fraction = max(0.0, min(1.0, (86400 - (now_local.hour * 3600 + now_local.minute * 60 + now_local.second)) / 86400))
        minimum_rest_dhw = dhw_baseline * remaining_day_fraction * _occupancy_factor(current_occ.persons)
        rest_today = round(max(today_rule_calc["total_kwh"] - today_floor, minimum_rest_dhw, 0.0), 2)

        today_date = dt_util.now().date().isoformat()
        training_sample = {
            "date": today_date, "updated_at": dt_util.now().isoformat(),
            "actual_total_kwh_so_far": round(today_so_far_kwh, 3) if today_so_far_kwh is not None else None,
            "actual_heating_kwh_so_far": round(today_heating_so_far_kwh, 3) if today_heating_so_far_kwh is not None else None,
            "actual_dhw_kwh_so_far": round(today_dhw_so_far_kwh, 3) if today_dhw_so_far_kwh is not None else None,
            "completed": False, "actual_total_kwh_final": None, "actual_heating_kwh_final": None, "actual_dhw_kwh_final": None,
            "forecast_error_kwh": None, "forecast_abs_error_kwh": None,
            "estimated_dhw_basis_kwh": round(dhw_baseline, 3), "estimated_heating_basis_kwh": round(heating_baseline, 3),
            "avg_temperature_c": today_avg_temp, "current_temperature_c": current_temp, "heating_threshold_c": heating_threshold,
            "persons": current_occ.persons, "occupied_units": current_occ.occupied_units, "total_units": current_occ.total_units,
            "occupied_area_sqm": current_occ.occupied_area_sqm, "total_area_sqm": current_occ.total_area_sqm,
            "weekday": dt_util.now().weekday(), "month": dt_util.now().month,
            "forecast_today_kwh": rule_today, "rest_today_kwh": rest_today, "forecast_tomorrow_kwh": rule_tomorrow,
            "forecast_day_after_tomorrow_kwh": rule_day_after, "source": source, "split_source": split_source,
        }
        training_samples = await self._async_store_training_sample(training_sample, history_stats.get("series") or {}, heating_history.get("series") or {}, dhw_history.get("series") or {})
        completed_samples = completed_training_samples(training_samples)
        pending_samples = [s for s in training_samples if not s.get("completed")]
        ml_status = await self._async_train_ml(training_samples)

        def _ml_value(day_offset: int, avg_temp: float | None, occ: OccupancyInfo, rule_value: float) -> tuple[float, str, dict[str, Any] | None]:
            if not ml_status.get("active"):
                return rule_value, "Regelmodell", None
            day = dt_util.now().date() + timedelta(days=day_offset)
            features = build_ml_features(avg_temperature_c=avg_temp, persons=occ.persons, occupied_units=occ.occupied_units, occupied_area_sqm=occ.occupied_area_sqm or occ.total_area_sqm, month=day.month, weekday=day.weekday(), heating_basis_kwh=heating_baseline, dhw_basis_kwh=dhw_baseline)
            prediction = self._ml_model.predict(features)
            if _plausible_forecast(prediction.value, rule_value):
                return round(float(prediction.value), 2), "ML-Modell", {"features": features, **asdict(prediction)}
            return rule_value, "ML + Fallback", {"features": features, **asdict(prediction), "fallback_rule_kwh": rule_value}

        tomorrow, tomorrow_model, tomorrow_ml = _ml_value(1, tomorrow_avg_temp, tomorrow_occ, rule_tomorrow)
        day_after, day_after_model, day_after_ml = _ml_value(2, day_after_avg_temp, day_after_occ, rule_day_after)
        forecast_model = "ML-Modell" if tomorrow_model == "ML-Modell" and day_after_model == "ML-Modell" else ("ML + Fallback" if ml_status.get("active") else "Regelmodell")

        completed_count = len(completed_samples)
        confidence_label, confidence_stage = ("Unzureichend", 1) if completed_count < 7 else ("Schwach", 2) if completed_count < 15 else ("Ausreichend", 3) if completed_count < 30 else ("Gut", 4) if completed_count < 90 else ("Sehr gut", 5)
        confidence = min(90, 30 + (25 if source == "daily_history" else 15) + (10 if temp_history.get("available") else 5 if current_temp is not None else 0) + (10 if forecast_temps.get(1) is not None else 0) + (5 if forecast_temps.get(2) is not None else 0) + (5 if units else 0) + (5 if heating_entity or dhw_entity else 0) + (5 if ml_status.get("active") else 0))
        reason_summary = " + ".join([forecast_model, _label_source(source), "Wetter", "Personen" if current_occ.total_units > 0 else "Heizgrenze", "Heizkurve" if heating_curve.get("active") else "Heizgrenze"])
        reason_text = f"v0.9.0 | Prognosemodell: {forecast_model} | ML-Status: {ml_status.get('status')} | Datenbasis: {_label_source(source)} | Basis {baseline_kwh:.2f} kWh | Morgen Regelmodell {rule_tomorrow:.2f} kWh, final {tomorrow:.2f} kWh | Übermorgen Regelmodell {rule_day_after:.2f} kWh, final {day_after:.2f} kWh"

        reason_structured = {
            "version": "Basis v0.9.0", "source": source, "split_source": split_source, "forecast_model": forecast_model, "ml_status": ml_status,
            "today_so_far_kwh": round(today_so_far_kwh, 2) if today_so_far_kwh is not None else None,
            "today_heating_so_far_kwh": round(today_heating_so_far_kwh, 2) if today_heating_so_far_kwh is not None else None,
            "today_dhw_so_far_kwh": round(today_dhw_so_far_kwh, 2) if today_dhw_so_far_kwh is not None else None,
            "history": history_stats, "heating_history": heating_history, "dhw_history": dhw_history, "temperature_history": temp_history, "heating_curve": heating_curve,
            "baseline_kwh": round(baseline_kwh, 2), "heating_baseline_kwh": round(heating_baseline, 2), "dhw_baseline_kwh": round(dhw_baseline, 2),
            "remaining_day_fraction": round(remaining_day_fraction, 3), "minimum_rest_dhw_kwh": round(minimum_rest_dhw, 2), "heating_threshold_c": heating_threshold,
            "today": {"temperature_c": current_temp, "avg_temperature_c": today_avg_temp, "forecast": today_rule_calc, **asdict(current_occ)},
            "tomorrow": {"avg_temperature_c": tomorrow_avg_temp, "rule_forecast": tomorrow_rule_calc, "ml": tomorrow_ml, "selected_model": tomorrow_model, **asdict(tomorrow_occ)},
            "day_after_tomorrow": {"avg_temperature_c": day_after_avg_temp, "rule_forecast": day_after_rule_calc, "ml": day_after_ml, "selected_model": day_after_model, **asdict(day_after_occ)},
        }
        reason_structured["training"] = {"enabled": True, **self._storage_paths_attributes(), "sample_count": len(training_samples), "ml_daily_sample_count": len(training_samples), "completed_sample_count": len(completed_samples), "ml_completed_daily_sample_count": len(completed_samples), "pending_sample_count": len(pending_samples), "latest_sample": training_sample, "last_7_samples": training_samples[-7:], "completed_last_7_samples": completed_samples[-7:]}
        return {
            "today_kwh": rule_today, "tomorrow_kwh": tomorrow, "day_after_tomorrow_kwh": day_after, "rest_today_kwh": rest_today,
            "rule_tomorrow_kwh": rule_tomorrow, "rule_day_after_tomorrow_kwh": rule_day_after,
            "confidence": confidence_label, "confidence_internal_score": confidence, "confidence_stage": confidence_stage,
            "confidence_completed_days": completed_count, "confidence_minimum_days": 30, "confidence_recommended_days": 90,
            "reason": reason_text, "reason_summary": reason_summary, "reason_structured": reason_structured,
            "training_sample_count": len(training_samples), "training_completed_sample_count": len(completed_samples), "training_pending_sample_count": len(pending_samples),
            "training_latest_sample": training_sample, "training_last_7_samples": training_samples[-7:], "training_completed_last_7_samples": completed_samples[-7:],
            "storage_paths": self._storage_paths_attributes(),
            "storage_status": {"Speicherstatus": "OK", "Trainingsdaten_Datei_vorhanden": self._training_data_path.exists(), "Heizkurven_Datei_vorhanden": self._heating_curve_path.exists(), "ML_Modell_Datei_vorhanden": self._model_path.exists(), "Trainingsdaten_Datei": str(self._training_data_path), "Heizkurven_Datei": str(self._heating_curve_path), "ML_Modell_Datei": str(self._model_path), "Gesammelte_Tagesdaten_ML": len(training_samples), "Abgeschlossene_Tagesdaten_ML": len(completed_samples), "Offene_Tagesdaten_ML": len(pending_samples)},
            "heating_curve": heating_curve, "ml_status": ml_status, "forecast_model": forecast_model,
        }


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    coordinator = HeatPumpForecastCoordinator(hass, entry)
    await coordinator.async_config_entry_first_refresh()
    async_add_entities([
        HeatPumpForecastSensor(coordinator, entry, "tomorrow_kwh", "Verbrauch morgen", "mdi:calendar-today"),
        HeatPumpForecastSensor(coordinator, entry, "day_after_tomorrow_kwh", "Verbrauch übermorgen", "mdi:calendar-arrow-right"),
        HeatPumpRestTodaySensor(coordinator, entry),
        HeatPumpConfidenceSensor(coordinator, entry),
        HeatPumpReasonSensor(coordinator, entry),
        HeatPumpTrainingSamplesSensor(coordinator, entry),
        HeatPumpLastTrainingSampleSensor(coordinator, entry),
        HeatPumpTrainingStatusSensor(coordinator, entry),
        HeatPumpDataQualitySensor(coordinator, entry),
        HeatPumpStorageStatusSensor(coordinator, entry),
        HeatPumpHeatingCurveStatusSensor(coordinator, entry),
        HeatPumpHeatingCurveSensor(coordinator, entry),
        HeatPumpMLStatusSensor(coordinator, entry),
        HeatPumpForecastModelSensor(coordinator, entry),
    ])


class HeatPumpBaseSensor(SensorEntity):
    _attr_has_entity_name = True

    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        self.coordinator = coordinator
        self.entry = entry
        self._attr_device_info = {"identifiers": {(DOMAIN, entry.entry_id)}, "name": entry.title, "manufacturer": "MTTPoll", "model": "Lokale Wärmepumpen-Verbrauchsprognose"}

    @property
    def available(self) -> bool:
        return self.coordinator.last_update_success

    async def async_update(self) -> None:
        await self.coordinator.async_request_refresh()


class HeatPumpForecastSensor(HeatPumpBaseSensor):
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry, key: str, name: str, icon: str) -> None:
        super().__init__(coordinator, entry)
        self.key = key
        self._attr_translation_key = key
        self._attr_name = name
        self._attr_icon = icon
        self._attr_unique_id = f"{entry.entry_id}_{key}"

    @property
    def native_value(self) -> float | None:
        return self.coordinator.data.get(self.key)


class HeatPumpRestTodaySensor(HeatPumpForecastSensor):
    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry, "rest_today_kwh", "Rest-Tagesprognose", "mdi:weather-sunset-down")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        structured = self.coordinator.data.get("reason_structured") or {}
        return {"Heute_bisher_kWh": structured.get("today_so_far_kwh"), "Prognose_heute_gesamt_kWh": self.coordinator.data.get("today_kwh"), "Rest_bis_Mitternacht_kWh": self.coordinator.data.get("rest_today_kwh"), "Berechnet_um": dt_util.now().isoformat()}


class HeatPumpStringSensor(HeatPumpBaseSensor):
    _attr_icon = "mdi:information-outline"

    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry, key: str, name: str, icon: str) -> None:
        super().__init__(coordinator, entry)
        self.key = key
        self._attr_name = name
        self._attr_icon = icon
        self._attr_translation_key = key
        self._attr_unique_id = f"{entry.entry_id}_{key}"

    @property
    def native_value(self) -> str | int | None:
        return self.coordinator.data.get(self.key)


class HeatPumpConfidenceSensor(HeatPumpStringSensor):
    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry, "confidence", "Prognosegüte", "mdi:chart-bell-curve")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return {"Stufe": self.coordinator.data.get("confidence_stage"), "Abgeschlossene_Tagesdaten": self.coordinator.data.get("confidence_completed_days"), "Minimum_für_gute_Prognose": self.coordinator.data.get("confidence_minimum_days"), "Empfohlen_für_sehr_gute_Prognose": self.coordinator.data.get("confidence_recommended_days"), "Interner_Basiswert": self.coordinator.data.get("confidence_internal_score"), "ML_Status": self.coordinator.data.get("ml_status"), "Prognosemodell": self.coordinator.data.get("forecast_model")}


class HeatPumpReasonSensor(HeatPumpStringSensor):
    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry, "reason_summary", "Prognosegrundlage", "mdi:text-box-search-outline")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        structured = self.coordinator.data.get("reason_structured") or {}
        return {"Kurzfassung": self.coordinator.data.get("reason_summary"), "Details": self.coordinator.data.get("reason"), "Version": structured.get("version"), "Datenbasis": _label_source(structured.get("source")), "Aufteilung": _label_split_source(structured.get("split_source")), "Prognosemodell": structured.get("forecast_model"), "ML_Status": structured.get("ml_status"), "Heute": structured.get("today"), "Morgen": structured.get("tomorrow"), "Übermorgen": structured.get("day_after_tomorrow"), "Lernspeicher": structured.get("training"), "Heizkurve": structured.get("heating_curve")}


class HeatPumpTrainingSamplesSensor(HeatPumpStringSensor):
    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry, "training_sample_count", "Gesammelte Tagesdaten", "mdi:database-clock-outline")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return {"Gesammelte_Tagesdaten_ML": self.coordinator.data.get("training_sample_count"), "Abgeschlossene_Tagesdaten_ML": self.coordinator.data.get("training_completed_sample_count"), "Offene_Tagesdaten_ML": self.coordinator.data.get("training_pending_sample_count"), "Aktueller_Datensatz_ML": self.coordinator.data.get("training_latest_sample"), "Letzte_7_Datensätze_ML": self.coordinator.data.get("training_last_7_samples"), "Letzte_7_abgeschlossene_Datensätze_ML": self.coordinator.data.get("training_completed_last_7_samples"), **(self.coordinator.data.get("storage_paths") or {})}


class HeatPumpLastTrainingSampleSensor(HeatPumpStringSensor):
    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry, "last_training_sample", "Letzter Trainingsdatensatz", "mdi:clipboard-text-clock-outline")

    @property
    def native_value(self) -> str | None:
        sample = self.coordinator.data.get("training_latest_sample") or {}
        return sample.get("date") or "Nicht verfügbar"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return self.coordinator.data.get("training_latest_sample") or {"Hinweis": "Noch kein Trainingsdatensatz gespeichert."}


class HeatPumpTrainingStatusSensor(HeatPumpStringSensor):
    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry, "training_status", "Trainingsstatus", "mdi:progress-clock")

    @property
    def native_value(self) -> str:
        completed = int(self.coordinator.data.get("training_completed_sample_count") or 0)
        if completed < ML_MINIMUM_DAYS:
            return f"Sammelt Daten ({completed}/{ML_MINIMUM_DAYS})"
        if completed < ML_OPTIMIZED_DAYS:
            return "ML darf trainiert werden"
        return "ML-Nachtraining möglich"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return {"Abgeschlossene_Tagesdaten_ML": self.coordinator.data.get("training_completed_sample_count"), "Minimum_ML": ML_MINIMUM_DAYS, "Optimiert_ab": ML_OPTIMIZED_DAYS, "ML_Status": self.coordinator.data.get("ml_status")}


class HeatPumpDataQualitySensor(HeatPumpStringSensor):
    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry, "data_quality", "Datenqualität", "mdi:database-check-outline")

    @property
    def native_value(self) -> str | None:
        return self.coordinator.data.get("confidence")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        structured = self.coordinator.data.get("reason_structured") or {}
        return {
            "Abgeschlossene_Tagesdaten": self.coordinator.data.get("confidence_completed_days"),
            "Temperaturhistorie_Tage": (structured.get("temperature_history") or {}).get("days"),
            "Verbrauchshistorie_Tage": (structured.get("history") or {}).get("days"),
            "Heizkurve": self.coordinator.data.get("heating_curve"),
            "ML_Status": self.coordinator.data.get("ml_status"),
        }


class HeatPumpStorageStatusSensor(HeatPumpStringSensor):
    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry, "storage_status", "Speicherstatus", "mdi:folder-check-outline")

    @property
    def native_value(self) -> str:
        return (self.coordinator.data.get("storage_status") or {}).get("Speicherstatus", "Unbekannt")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return self.coordinator.data.get("storage_status") or {}


class HeatPumpHeatingCurveStatusSensor(HeatPumpStringSensor):
    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry, "heating_curve_status", "Heizkurvenstatus", "mdi:chart-bell-curve-cumulative")

    @property
    def native_value(self) -> str:
        return (self.coordinator.data.get("heating_curve") or {}).get("status", "Wird aufgebaut")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return self.coordinator.data.get("heating_curve") or {}


class HeatPumpHeatingCurveSensor(HeatPumpHeatingCurveStatusSensor):
    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        HeatPumpStringSensor.__init__(self, coordinator, entry, "heating_curve", "Erlernte Heizkurve", "mdi:chart-line")


class HeatPumpMLStatusSensor(HeatPumpStringSensor):
    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry, "ml_status", "ML-Status", "mdi:brain")

    @property
    def native_value(self) -> str:
        return (self.coordinator.data.get("ml_status") or {}).get("status", "Wartet auf Trainingsdaten")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return self.coordinator.data.get("ml_status") or {}


class HeatPumpForecastModelSensor(HeatPumpStringSensor):
    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry, "forecast_model", "Prognosemodell", "mdi:source-branch-check")

    @property
    def native_value(self) -> str:
        return self.coordinator.data.get("forecast_model") or "Regelmodell"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        structured = self.coordinator.data.get("reason_structured") or {}
        return {"Prognosemodell": self.coordinator.data.get("forecast_model"), "ML_Status": self.coordinator.data.get("ml_status"), "Regelmodell_morgen_kWh": self.coordinator.data.get("rule_tomorrow_kwh"), "Regelmodell_übermorgen_kWh": self.coordinator.data.get("rule_day_after_tomorrow_kwh"), "Morgen": structured.get("tomorrow"), "Übermorgen": structured.get("day_after_tomorrow"), "Hinweis": "ML wird nur verwendet, wenn genug Daten vorhanden sind und die Prognose plausibel ist. Das Regelmodell bleibt immer Fallback."}
