"""Sensors for Heat Pump Consumption Forecast."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from statistics import mean
import logging
import re
from typing import Any

from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorStateClass
from homeassistant.components import recorder
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfEnergy, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.helpers.storage import Store

from .const import (
    CONF_FIXED_PERSONS,
    CONF_HEATPUMP_DAILY_ENERGY_SENSOR,
    CONF_HEATING_DAILY_ENERGY_SENSOR,
    CONF_DHW_DAILY_ENERGY_SENSOR,
    CONF_HEATING_THRESHOLD_TEMP,
    CONF_HEATPUMP_TOTAL_ENERGY_SENSOR,
    CONF_OUTDOOR_TEMP_SENSOR,
    CONF_UNITS,
    CONF_WEATHER_ENTITY,
    DOMAIN,
    SCAN_INTERVAL_MINUTES,
)

_LOGGER = logging.getLogger(__name__)
WH_UNITS = {"wh", "watt hour", "watt-hours"}
KWH_UNITS = {"kwh", "kilowatt hour", "kilowatt-hours"}
MWH_UNITS = {"mwh", "megawatt hour", "megawatt-hours"}
ENERGY_UNITS = WH_UNITS | KWH_UNITS | MWH_UNITS
STORAGE_VERSION = 1
STORAGE_KEY_PREFIX = f"{DOMAIN}_training_data"
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


def _float_state(hass: HomeAssistant, entity_id: str | None) -> float | None:
    """Return entity state as float."""
    if not entity_id:
        return None
    state = hass.states.get(entity_id)
    if state is None:
        return None
    try:
        return float(str(state.state).replace(",", "."))
    except (TypeError, ValueError):
        return None


def _energy_kwh(hass: HomeAssistant, entity_id: str | None) -> float | None:
    """Return an energy sensor value normalized to kWh."""
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
    """Return a historic state value normalized to kWh."""
    try:
        value = float(str(state.state).replace(",", "."))
    except (AttributeError, TypeError, ValueError):
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
    """Average of the last n values."""
    relevant = values[-days:]
    if not relevant:
        return None
    return float(mean(relevant))


def _parse_persons_from_text(text: str | None) -> float | None:
    """Parse person counts from calendar summary/description text."""
    if not text:
        return None
    normalized = str(text).replace("\n", " ").strip()

    adults = 0.0
    children = 0.0
    adult_match = re.search(r"(\d+(?:[,.]\d+)?)\s*(?:erw\.?|erwachsene?|adult|adults)", normalized, flags=re.IGNORECASE)
    child_match = re.search(r"(\d+(?:[,.]\d+)?)\s*(?:kind(?:er)?|child|children)", normalized, flags=re.IGNORECASE)
    if adult_match:
        adults = float(adult_match.group(1).replace(",", "."))
    if child_match:
        children = float(child_match.group(1).replace(",", "."))
    if adult_match or child_match:
        return adults + children

    patterns = (
        r"(\d+(?:[,.]\d+)?)\s*(?:pers\.?|personen?|person|persons|guests?|gäste|gaeste)",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(",", "."))
            except (TypeError, ValueError):
                return None
    return None


def _unit_area(unit: dict[str, Any]) -> float:
    """Return unit area."""
    return float(unit.get("area_sqm") or 0)


def _total_area(units: list[dict[str, Any]]) -> float:
    """Return total configured living area."""
    return sum(_unit_area(unit) for unit in units)


def _fixed_unit_persons(units: list[dict[str, Any]]) -> float:
    """Return sum of fixed persons configured per unit."""
    return sum(float(unit.get("fixed_persons") or 0) for unit in units)


def _parse_current_calendar_unit(hass: HomeAssistant, unit: dict[str, Any]) -> tuple[bool, float]:
    """Return whether a configured unit calendar is currently occupied and its persons."""
    entity_id = unit.get("calendar_entity")
    if not entity_id:
        persons = float(unit.get("fixed_persons") or 0)
        return persons > 0, persons

    state = hass.states.get(entity_id)
    if state is None or state.state not in {"on", "true"}:
        return False, 0.0

    text_parts = [
        state.attributes.get("description"),
        state.attributes.get("message"),
        state.attributes.get("summary"),
    ]
    persons = None
    for text in text_parts:
        persons = _parse_persons_from_text(text)
        if persons is not None:
            break
    if persons is None:
        persons = float(unit.get("fixed_persons") or 0)
    return True, float(persons or 0)


def _current_occupancy(hass: HomeAssistant, units: list[dict[str, Any]], fallback_persons: float = 0.0) -> OccupancyInfo:
    """Return current occupancy summary from states."""
    info = OccupancyInfo(total_units=len(units), total_area_sqm=_total_area(units), unit_details=[])
    for unit in units:
        occupied, persons = _parse_current_calendar_unit(hass, unit)
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
    """Return heating demand factor based on average outside temperature and heating threshold."""
    if avg_temp is None:
        return 1.0
    heating_degrees = max(0.0, float(threshold) - float(avg_temp))
    # No space-heating demand above threshold. Below threshold, demand rises smoothly.
    if heating_degrees <= 0:
        return 0.0
    return max(0.15, min(2.8, heating_degrees / 8.0))


def _total_temperature_factor(avg_temp: float | None, threshold: float) -> float:
    """Fallback factor for systems without separate heating/DHW sensors."""
    if avg_temp is None:
        return 1.0
    heating_factor = _heating_degree_factor(avg_temp, threshold)
    # Keep a warm-water base load above threshold, add space-heating influence below threshold.
    return max(0.55, min(2.5, 0.65 + 0.55 * heating_factor))


def _occupancy_factor(persons: float) -> float:
    """Return deterministic occupancy factor."""
    return 1.0 + min(max(float(persons or 0), 0), 100) * 0.018


def _area_factor(area: float) -> float:
    """Return deterministic area factor. Uses occupied area when available."""
    if area <= 0:
        return 1.0
    return max(0.75, min(1.35, 0.85 + (area / 500.0)))


def _label_source(source: str | None) -> str:
    """Return German label for forecast data source."""
    return {
        "daily_history": "Tageshistorie",
        "daily": "Tageswert",
        "accumulated_delta": "Gesamtzähler-Differenz",
        "fallback": "Ersatzwert",
        "fallback_guardrail_selected_daily_too_high": "Ersatzwert nach Plausibilitätsprüfung",
    }.get(str(source or ""), str(source or "unbekannt"))


def _label_split_source(source: str | None) -> str:
    """Return German label for heating/DHW split source."""
    return {
        "total_minus_heating": "Gesamtverbrauch minus Heizverbrauch",
        "dedicated_sensors": "separate Verbrauchssensoren",
        "total_minus_dhw": "Gesamtverbrauch minus Warmwasserverbrauch",
        "estimated_split": "geschätzte Aufteilung",
        "dhw_guardrail_total_minus_heating": "Warmwasser abgeleitet: Gesamt minus Heizung",
        "heating_guardrail_total_minus_dhw": "Heizung abgeleitet: Gesamt minus Warmwasser",
    }.get(str(source or ""), str(source or "unbekannt"))


def _learn_heating_curve(temp_series: dict[str, float], heating_series: dict[str, float], threshold: float) -> dict[str, Any]:
    """Learn a simple temperature-to-heating-kWh curve from paired daily history."""
    pairs: list[tuple[float, float]] = []
    for day, temp in (temp_series or {}).items():
        heat = (heating_series or {}).get(day)
        if temp is None or heat is None:
            continue
        try:
            temp_f = float(temp)
            heat_f = float(heat)
        except (TypeError, ValueError):
            continue
        if -40 <= temp_f <= 50 and 0 <= heat_f <= 200:
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

    bucket_avgs = {name: round(float(mean(vals)), 2) if vals else None for name, vals in buckets.items()}
    bucket_counts = {name: len(vals) for name, vals in buckets.items()}
    paired_days = len(pairs)
    active = paired_days >= 30
    ready = active
    optimized = paired_days >= 90
    if optimized:
        status = "Optimiert"
    elif active:
        status = "Aktiv"
    elif paired_days > 0:
        status = "Wird aufgebaut"
    else:
        status = "Keine Heizkurvendaten"
    return {
        "available": bool(pairs),
        "active": active,
        "ready": ready,
        "status": status,
        "paired_days": paired_days,
        "minimum_days": 30,
        "recommended_days": 90,
        "optimized": optimized,
        "progress_percent": min(100, round((paired_days / 30) * 100)) if paired_days else 0,
        "threshold_c": threshold,
        "bucket_avgs_kwh": bucket_avgs,
        "bucket_counts": bucket_counts,
        "pairs_last_30": [{"temperature_c": round(t, 2), "heating_kwh": round(h, 3)} for t, h in pairs[-30:]],
    }


def _heating_from_curve(avg_temp: float | None, curve: dict[str, Any], fallback_heating_baseline: float, threshold: float) -> tuple[float, str]:
    """Return heating kWh from learned curve if available, else fallback baseline."""
    if avg_temp is None or not curve.get("active"):
        return float(fallback_heating_baseline or 0.0) * _heating_degree_factor(avg_temp, threshold), "regelbasis"
    buckets = curve.get("bucket_avgs_kwh") or {}
    temp = float(avg_temp)
    if temp >= threshold:
        return 0.0, "heizkurve"
    if temp < 5:
        value = buckets.get("unter_5")
    elif temp < 10:
        value = buckets.get("5_bis_10")
    elif temp < 15:
        value = buckets.get("10_bis_15")
    else:
        value = buckets.get("15_bis_heizgrenze")
    if value is None:
        # Fallback to nearest populated cooler bucket or deterministic factor.
        populated = [v for v in buckets.values() if v is not None]
        if populated:
            value = float(mean(populated))
        else:
            value = float(fallback_heating_baseline or 0.0) * _heating_degree_factor(avg_temp, threshold)
    return float(value or 0.0), "heizkurve"


class HeatPumpForecastCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Coordinator calculating basic forecast values."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize coordinator."""
        self.entry = entry
        self._last_total_kwh: float | None = None
        self._estimated_today_from_total = 0.0
        self._store: Store = Store(hass, STORAGE_VERSION, f"{STORAGE_KEY_PREFIX}_{entry.entry_id}")
        self._training_samples: list[dict[str, Any]] | None = None
        super().__init__(hass, _LOGGER, name=DOMAIN, update_interval=timedelta(minutes=SCAN_INTERVAL_MINUTES))

    async def _async_load_training_samples(self) -> list[dict[str, Any]]:
        """Load persisted daily training samples."""
        if self._training_samples is not None:
            return self._training_samples
        try:
            stored = await self._store.async_load()
        except Exception as err:  # noqa: BLE001
            _LOGGER.debug("Could not load training store: %s", err)
            stored = None
        samples = []
        if isinstance(stored, dict):
            raw_samples = stored.get("samples", [])
            if isinstance(raw_samples, list):
                samples = [sample for sample in raw_samples if isinstance(sample, dict)]
        self._training_samples = samples[-MAX_TRAINING_SAMPLES:]
        return self._training_samples

    async def _async_store_training_sample(self, sample: dict[str, Any]) -> list[dict[str, Any]]:
        """Persist or update one daily learning sample.

        v0.7 stores one record per date. The current day is overwritten during the
        day and becomes a completed sample after midnight. This is intentionally
        simple and prepares the data foundation for the later ML model.
        """
        samples = await self._async_load_training_samples()
        date_key = sample.get("date")
        if not date_key:
            return samples
        replaced = False
        for idx, existing in enumerate(samples):
            if existing.get("date") == date_key:
                samples[idx] = sample
                replaced = True
                break
        if not replaced:
            samples.append(sample)
        samples = samples[-MAX_TRAINING_SAMPLES:]
        self._training_samples = samples
        try:
            await self._store.async_save({"samples": samples})
        except Exception as err:  # noqa: BLE001
            _LOGGER.debug("Could not save training sample: %s", err)
        return samples

    async def _async_get_forecast_temperatures(self, weather_entity: str | None) -> dict[int, float | None]:
        """Try to get daily forecast temperatures: 1=tomorrow, 2=day after tomorrow."""
        result: dict[int, float | None] = {1: None, 2: None}
        if not weather_entity:
            return result

        forecast: list[dict[str, Any]] = []
        try:
            response = await self.hass.services.async_call(
                "weather",
                "get_forecasts",
                {"entity_id": [weather_entity], "type": "daily"},
                blocking=True,
                return_response=True,
            )
            forecast = (response or {}).get(weather_entity, {}).get("forecast") or []
        except Exception as err:  # noqa: BLE001
            _LOGGER.debug("Could not fetch daily weather forecast via service: %s", err)

        if not forecast:
            state = self.hass.states.get(weather_entity)
            if state is not None:
                forecast = state.attributes.get("forecast") or []

        for offset in (1, 2):
            if len(forecast) > offset:
                high = forecast[offset].get("temperature")
                low = forecast[offset].get("templow") or forecast[offset].get("temperature_low")
                try:
                    high_f = float(high) if high is not None else None
                    low_f = float(low) if low is not None else None
                    if high_f is not None and low_f is not None:
                        result[offset] = round((high_f + low_f) / 2.0, 2)
                    elif high_f is not None:
                        result[offset] = high_f
                except (TypeError, ValueError):
                    result[offset] = None
        return result

    async def _async_get_calendar_occupancy_for_day(self, units: list[dict[str, Any]], offset_days: int, fallback_persons: float = 0.0) -> OccupancyInfo | None:
        """Read persons/occupied units/occupied area for one future day from configured calendars."""
        calendar_entities = []
        for unit in units:
            entity_id = unit.get("calendar_entity")
            if not entity_id:
                continue
            state = self.hass.states.get(entity_id)
            if state is None or state.state in {STATE_UNAVAILABLE, STATE_UNKNOWN}:
                _LOGGER.debug("Calendar entity %s is not available yet; occupancy for offset %s will be retried later", entity_id, offset_days)
                continue
            calendar_entities.append(entity_id)
        if not calendar_entities:
            return None

        day = datetime.now().date() + timedelta(days=offset_days)
        start = datetime.combine(day, datetime.min.time()).isoformat()
        end = datetime.combine(day + timedelta(days=1), datetime.min.time()).isoformat()

        try:
            response = await self.hass.services.async_call(
                "calendar",
                "get_events",
                {"entity_id": calendar_entities, "start_date_time": start, "end_date_time": end},
                blocking=True,
                return_response=True,
            )
        except Exception as err:  # noqa: BLE001
            _LOGGER.debug("Could not fetch calendar events for day offset %s: %s", offset_days, err)
            return None

        if not response:
            return None

        info = OccupancyInfo(total_units=len(units), total_area_sqm=_total_area(units), unit_details=[])
        found_any = False
        for unit in units:
            entity_id = unit.get("calendar_entity")
            area = _unit_area(unit)
            occupied = False
            unit_persons = 0.0
            events = (response.get(entity_id, {}) or {}).get("events") or [] if entity_id else []
            if events:
                occupied = True
                found_any = True
                parsed_values: list[float] = []
                for event in events:
                    text = " ".join(str(event.get(key) or "") for key in ("description", "summary", "message"))
                    parsed = _parse_persons_from_text(text)
                    if parsed is not None:
                        parsed_values.append(float(parsed))
                if parsed_values:
                    unit_persons = max(parsed_values)
                else:
                    unit_persons = float(unit.get("fixed_persons") or 0)

            if occupied:
                info.occupied_units += 1
                info.occupied_area_sqm += area
            info.persons += unit_persons
            info.unit_details.append({"name": unit.get("name"), "occupied": occupied, "persons": unit_persons, "area_sqm": area})

        if info.persons <= 0 and fallback_persons > 0:
            info.persons = fallback_persons
        return info if found_any else None


    async def _async_get_temperature_history_stats(self, entity_id: str | None, days: int = 30) -> dict[str, Any]:
        """Read recorder history for outside temperature and calculate daily averages."""
        empty = {"available": False, "days": 0, "values": [], "series": {}, "avg_7": None, "avg_14": None, "avg_30": None, "today_avg": None}
        if not entity_id:
            return empty

        now = dt_util.now()
        start = now - timedelta(days=days + 1)
        end = now

        def _read_history() -> list[Any]:
            try:
                from homeassistant.components.recorder import history

                states_by_entity = history.get_significant_states(
                    self.hass,
                    start,
                    end,
                    entity_ids=[entity_id],
                    significant_changes_only=False,
                    minimal_response=False,
                    no_attributes=False,
                )
                return list(states_by_entity.get(entity_id, []) or [])
            except Exception as err:  # noqa: BLE001
                _LOGGER.debug("Could not read temperature history for %s: %s", entity_id, err)
                return []

        states = await recorder.get_instance(self.hass).async_add_executor_job(_read_history)
        if not states:
            return empty

        current_day = now.date()
        grouped: dict[Any, list[float]] = {}
        for state in states:
            try:
                value = float(str(state.state).replace(",", "."))
            except (AttributeError, TypeError, ValueError):
                continue
            if value < -50 or value > 60:
                continue
            changed = getattr(state, "last_changed", None) or getattr(state, "last_updated", None)
            if changed is None:
                continue
            day = dt_util.as_local(changed).date()
            grouped.setdefault(day, []).append(value)

        series = {day.isoformat(): round(float(mean(vals)), 2) for day, vals in sorted(grouped.items()) if day != current_day and vals}
        completed = list(series.values())
        today_vals = grouped.get(current_day, [])
        today_avg = round(float(mean(today_vals)), 2) if today_vals else None
        if not completed:
            return {**empty, "today_avg": today_avg}

        avg_7 = _avg(completed, 7)
        avg_14 = _avg(completed, 14)
        avg_30 = _avg(completed, 30)
        return {
            "available": True,
            "days": len(completed),
            "values": completed[-30:],
            "series": dict(list(series.items())[-30:]),
            "avg_7": round(avg_7, 2) if avg_7 is not None else None,
            "avg_14": round(avg_14, 2) if avg_14 is not None else None,
            "avg_30": round(avg_30, 2) if avg_30 is not None else None,
            "today_avg": today_avg,
        }

    async def _async_get_daily_history_stats(self, entity_id: str | None, days: int = 30) -> dict[str, Any]:
        """Read recorder history for a daily energy sensor and calculate daily maxima.

        Daily energy sensors usually reset to 0 once per day and then count upwards.
        Therefore the highest value of a completed day is treated as that day's
        consumption. The current day is excluded from the historical averages.
        """
        empty = {
            "available": False,
            "days": 0,
            "values": [],
            "avg_7": None,
            "avg_14": None,
            "avg_30": None,
            "basis": None,
            "series": {},
        }
        if not entity_id:
            return empty

        now = dt_util.now()
        start = now - timedelta(days=days + 1)
        end = now

        def _read_history() -> list[Any]:
            try:
                from homeassistant.components.recorder import history

                states_by_entity = history.get_significant_states(
                    self.hass,
                    start,
                    end,
                    entity_ids=[entity_id],
                    significant_changes_only=False,
                    minimal_response=False,
                    no_attributes=False,
                )
                return list(states_by_entity.get(entity_id, []) or [])
            except Exception as err:  # noqa: BLE001
                _LOGGER.debug("Could not read recorder history for %s: %s", entity_id, err)
                return []

        states = await recorder.get_instance(self.hass).async_add_executor_job(_read_history)
        if not states:
            return empty

        current_day = now.date()
        daily_max: dict[Any, float] = {}
        for state in states:
            value = _history_state_kwh(state)
            if value is None or value < 0 or value > 500:
                continue
            changed = getattr(state, "last_changed", None) or getattr(state, "last_updated", None)
            if changed is None:
                continue
            local_changed = dt_util.as_local(changed)
            day = local_changed.date()
            if day == current_day:
                continue
            daily_max[day] = max(daily_max.get(day, 0.0), value)

        series = {day.isoformat(): round(v, 3) for day, v in sorted(daily_max.items()) if v > 0}
        values = list(series.values())
        if not values:
            return empty

        avg_7 = _avg(values, 7)
        avg_14 = _avg(values, 14)
        avg_30 = _avg(values, 30)
        basis = avg_7 or avg_14 or avg_30
        return {
            "available": True,
            "days": len(values),
            "values": values[-30:],
            "series": dict(list(series.items())[-30:]),
            "avg_7": round(avg_7, 2) if avg_7 is not None else None,
            "avg_14": round(avg_14, 2) if avg_14 is not None else None,
            "avg_30": round(avg_30, 2) if avg_30 is not None else None,
            "basis": round(basis, 2) if basis is not None else None,
        }

    async def _async_update_data(self) -> dict[str, Any]:
        """Calculate forecast data."""
        config = {**self.entry.data, **self.entry.options}
        daily_entity = config.get(CONF_HEATPUMP_DAILY_ENERGY_SENSOR)
        total_entity = config.get(CONF_HEATPUMP_TOTAL_ENERGY_SENSOR)
        current_temp = _float_state(self.hass, config.get(CONF_OUTDOOR_TEMP_SENSOR))
        forecast_temps = await self._async_get_forecast_temperatures(config.get(CONF_WEATHER_ENTITY))
        tomorrow_temp = forecast_temps.get(1)
        day_after_temp = forecast_temps.get(2)
        units: list[dict[str, Any]] = list(config.get(CONF_UNITS) or [])

        fixed_persons = _fixed_unit_persons(units)
        fallback_persons = float(config.get(CONF_FIXED_PERSONS, 0) or 0)
        current_occ = _current_occupancy(self.hass, units, fallback_persons=fallback_persons)
        if current_occ.persons <= 0 and fixed_persons > 0:
            current_occ.persons = fixed_persons

        tomorrow_occ = await self._async_get_calendar_occupancy_for_day(units, 1, fallback_persons=current_occ.persons)
        day_after_occ = await self._async_get_calendar_occupancy_for_day(units, 2, fallback_persons=current_occ.persons)
        if tomorrow_occ is None:
            tomorrow_occ = current_occ
        if day_after_occ is None:
            day_after_occ = tomorrow_occ

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

        today_avg_temp = temp_history.get("today_avg") if temp_history.get("today_avg") is not None else current_temp
        tomorrow_avg_temp = tomorrow_temp if tomorrow_temp is not None else today_avg_temp
        day_after_avg_temp = day_after_temp if day_after_temp is not None else tomorrow_avg_temp

        source = "daily_history" if history_stats.get("basis") is not None else "daily"
        baseline_kwh = history_stats.get("basis")

        if baseline_kwh is None:
            baseline_kwh = today_so_far_kwh

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

        # v0.6.1 split model:
        # - Prefer a real heating daily kWh sensor if available.
        # - DHW/base load can be derived from total heat-pump consumption minus heating consumption.
        # - A dedicated DHW kWh sensor is optional, but temperature sensors are rejected by unit validation above.
        heating_baseline = heating_history.get("basis")
        dhw_baseline = dhw_history.get("basis")
        split_source = "estimated_split"

        if heating_baseline is not None and dhw_baseline is None:
            dhw_baseline = round(max(float(baseline_kwh) - float(heating_baseline or 0.0), 0.0), 2)
            split_source = "total_minus_heating"
        elif heating_baseline is not None and dhw_baseline is not None:
            split_source = "dedicated_sensors"
        elif heating_baseline is None and dhw_baseline is not None:
            heating_baseline = round(max(float(baseline_kwh) - float(dhw_baseline or 0.0), 0.0), 2)
            split_source = "total_minus_dhw"
        else:
            # Estimate historical heating share from last known outside temperature average.
            avg_temp_reference = temp_history.get("avg_7") if temp_history.get("avg_7") is not None else today_avg_temp
            ref_heat_factor = _heating_degree_factor(avg_temp_reference, heating_threshold)
            estimated_heating_share = max(0.0, min(0.75, ref_heat_factor / 2.8))
            heating_baseline = round(float(baseline_kwh) * estimated_heating_share, 2)
            dhw_baseline = round(max(float(baseline_kwh) - float(heating_baseline or 0.0), float(baseline_kwh) * 0.25), 2)

        # Guardrails: derived parts must be plausible and must not exceed the total baseline wildly.
        heating_baseline = max(0.0, float(heating_baseline or 0.0))
        dhw_baseline = max(0.0, float(dhw_baseline or 0.0))
        if dhw_baseline > max(20.0, float(baseline_kwh) * 3.0):
            # Most likely a wrong entity, e.g. a hot-water tank temperature sensor.
            dhw_baseline = round(max(float(baseline_kwh) - heating_baseline, 0.0), 2)
            split_source = "dhw_guardrail_total_minus_heating"
        if heating_baseline > max(100.0, float(baseline_kwh) * 3.0):
            heating_baseline = round(max(float(baseline_kwh) - dhw_baseline, 0.0), 2)
            split_source = "heating_guardrail_total_minus_dhw"

        def _forecast_for(avg_temp: float | None, occ: OccupancyInfo) -> dict[str, Any]:
            heat_active = avg_temp is None or avg_temp < heating_threshold
            heat_factor = _heating_degree_factor(avg_temp, heating_threshold)
            learned_heat, heating_method = _heating_from_curve(avg_temp, heating_curve, float(heating_baseline or 0.0), heating_threshold)
            heating_part = learned_heat * _area_factor(occ.occupied_area_sqm or occ.total_area_sqm)
            dhw_part = float(dhw_baseline or 0.0) * _occupancy_factor(occ.persons)
            if not heat_active:
                heating_part = 0.0
            total = heating_part + dhw_part
            return {
                "total_kwh": round(total, 2),
                "heating_kwh": round(heating_part, 2),
                "dhw_kwh": round(dhw_part, 2),
                "heating_active": bool(heat_active),
                "heating_factor": round(heat_factor, 3),
                "heating_method": heating_method,
            }

        today_calc = _forecast_for(today_avg_temp, current_occ)
        tomorrow_calc = _forecast_for(tomorrow_avg_temp, tomorrow_occ)
        day_after_calc = _forecast_for(day_after_avg_temp, day_after_occ)

        today_floor = today_so_far_kwh if today_so_far_kwh is not None and 0 <= today_so_far_kwh < 500 else 0.0
        today = round(max(today_floor, today_calc["total_kwh"]), 2)
        tomorrow = tomorrow_calc["total_kwh"]
        day_after = day_after_calc["total_kwh"]
        # Rest-Tagesprognose: if today's real consumption already exceeds the
        # calculated total day forecast, do not show 0.00 kWh. A heat pump can still
        # have DHW/base-load demand until midnight. Use a conservative remaining
        # DHW share based on the remaining part of the day.
        now_local = dt_util.now()
        seconds_done = (now_local.hour * 3600) + (now_local.minute * 60) + now_local.second
        remaining_day_fraction = max(0.0, min(1.0, (86400 - seconds_done) / 86400))
        minimum_rest_dhw = float(dhw_baseline or 0.0) * remaining_day_fraction * _occupancy_factor(current_occ.persons)
        rest_today = round(max(today_calc["total_kwh"] - today_floor, minimum_rest_dhw, 0.0), 2)

        confidence = 30
        if source == "daily_history":
            confidence = 55
            history_days = int(history_stats.get("days") or 0)
            if history_days >= 7:
                confidence += 5
            if history_days >= 14:
                confidence += 5
        elif source == "daily":
            confidence = 45
        elif source == "accumulated_delta":
            confidence = 35
        if temp_history.get("available"):
            confidence += 10
        elif current_temp is not None:
            confidence += 5
        if tomorrow_temp is not None:
            confidence += 10
        if day_after_temp is not None:
            confidence += 5
        if units:
            confidence += 5
        if heating_entity or dhw_entity:
            confidence += 5

        reason_structured = {
            "version": "Basis v0.8.0",
            "source": source,
            "split_source": split_source,
            "today_so_far_kwh": round(today_so_far_kwh, 2) if today_so_far_kwh is not None else None,
            "today_heating_so_far_kwh": round(today_heating_so_far_kwh, 2) if today_heating_so_far_kwh is not None else None,
            "today_dhw_so_far_kwh": round(today_dhw_so_far_kwh, 2) if today_dhw_so_far_kwh is not None else None,
            "history": history_stats,
            "heating_history": heating_history,
            "dhw_history": dhw_history,
            "temperature_history": temp_history,
            "heating_curve": heating_curve,
            "baseline_kwh": round(baseline_kwh, 2),
            "heating_baseline_kwh": round(float(heating_baseline or 0), 2),
            "dhw_baseline_kwh": round(float(dhw_baseline or 0), 2),
            "remaining_day_fraction": round(remaining_day_fraction, 3),
            "minimum_rest_dhw_kwh": round(minimum_rest_dhw, 2),
            "heating_threshold_c": heating_threshold,
            "today": {"temperature_c": current_temp, "avg_temperature_c": today_avg_temp, "forecast": today_calc, **asdict(current_occ)},
            "tomorrow": {"avg_temperature_c": tomorrow_avg_temp, "forecast": tomorrow_calc, **asdict(tomorrow_occ)},
            "day_after_tomorrow": {"avg_temperature_c": day_after_avg_temp, "forecast": day_after_calc, **asdict(day_after_occ)},
        }
        source_label = _label_source(source)
        split_label = _label_split_source(split_source)
        reason_text = (
            f"v0.8.0 | Datenbasis: {source_label} | "
            f"Basis {baseline_kwh:.2f} kWh, Ø7 {history_stats.get('avg_7')} kWh | "
            f"Temperatur Ø heute {today_avg_temp}°C, Heizgrenze {heating_threshold:.1f}°C | "
            f"Warmwasser {float(dhw_baseline or 0):.2f} kWh + Heizung {float(heating_baseline or 0):.2f} kWh "
            f"({split_label}) | "
            f"Heute: {current_occ.persons:.0f} Pers., {current_occ.occupied_units}/{current_occ.total_units} WE, "
            f"Heizung {'aktiv' if today_calc['heating_active'] else 'inaktiv'} | "
            f"Morgen: Temperatur Ø {tomorrow_avg_temp}°C, {tomorrow_occ.persons:.0f} Pers., "
            f"Heizung {'aktiv' if tomorrow_calc['heating_active'] else 'inaktiv'} | "
            f"Übermorgen: Temperatur Ø {day_after_avg_temp}°C, {day_after_occ.persons:.0f} Pers., "
            f"Heizung {'aktiv' if day_after_calc['heating_active'] else 'inaktiv'}"
        )

        # Short dashboard status: keep it readable and explain which data sources are used.
        # The exact occupancy numbers remain in the sensor attributes.
        reason_parts = [source_label, "Wetter"]
        if (current_occ.total_units or 0) > 0:
            reason_parts.append("Personen")
        if heating_curve.get("active"):
            reason_parts.append("Heizkurve")
        elif heating_threshold is not None:
            reason_parts.append("Heizgrenze")
        reason_summary = " + ".join(reason_parts)

        # v0.7 learning store: persist a daily feature snapshot for later ML training.
        today_date = dt_util.now().date().isoformat()
        training_sample = {
            "date": today_date,
            "updated_at": dt_util.now().isoformat(),
            "actual_total_kwh_so_far": round(today_so_far_kwh, 3) if today_so_far_kwh is not None else None,
            "actual_heating_kwh_so_far": round(today_heating_so_far_kwh, 3) if today_heating_so_far_kwh is not None else None,
            "actual_dhw_kwh_so_far": round(today_dhw_so_far_kwh, 3) if today_dhw_so_far_kwh is not None else None,
            "estimated_dhw_basis_kwh": round(float(dhw_baseline or 0), 3),
            "estimated_heating_basis_kwh": round(float(heating_baseline or 0), 3),
            "avg_temperature_c": today_avg_temp,
            "current_temperature_c": current_temp,
            "heating_threshold_c": heating_threshold,
            "persons": current_occ.persons,
            "occupied_units": current_occ.occupied_units,
            "total_units": current_occ.total_units,
            "occupied_area_sqm": current_occ.occupied_area_sqm,
            "total_area_sqm": current_occ.total_area_sqm,
            "weekday": dt_util.now().weekday(),
            "month": dt_util.now().month,
            "forecast_today_kwh": today,
            "rest_today_kwh": rest_today,
            "forecast_tomorrow_kwh": tomorrow,
            "forecast_day_after_tomorrow_kwh": day_after,
            "source": source,
            "split_source": split_source,
        }
        training_samples = await self._async_store_training_sample(training_sample)
        completed_training_samples = [s for s in training_samples if s.get("date") != today_date]

        reason_structured["training"] = {
            "enabled": True,
            "sample_count": len(training_samples),
            "ml_daily_sample_count": len(training_samples),
            "completed_sample_count": len(completed_training_samples),
            "ml_completed_daily_sample_count": len(completed_training_samples),
            "latest_sample": training_sample,
            "last_7_samples": training_samples[-7:],
        }

        completed_count_for_quality = len(completed_training_samples)
        if completed_count_for_quality < 7:
            confidence_label = "Unzureichend"
            confidence_stage = 1
        elif completed_count_for_quality < 15:
            confidence_label = "Schwach"
            confidence_stage = 2
        elif completed_count_for_quality < 30:
            confidence_label = "Ausreichend"
            confidence_stage = 3
        elif completed_count_for_quality < 90:
            confidence_label = "Gut"
            confidence_stage = 4
        else:
            confidence_label = "Sehr gut"
            confidence_stage = 5

        return {
            "today_kwh": today,
            "tomorrow_kwh": tomorrow,
            "day_after_tomorrow_kwh": day_after,
            "rest_today_kwh": rest_today,
            "confidence": confidence_label,
            "confidence_internal_score": min(confidence, 80),
            "confidence_stage": confidence_stage,
            "confidence_completed_days": completed_count_for_quality,
            "confidence_minimum_days": 30,
            "confidence_recommended_days": 90,
            "reason": reason_text,
            "reason_summary": reason_summary,
            "reason_structured": reason_structured,
            "training_sample_count": len(training_samples),
            "training_completed_sample_count": len(completed_training_samples),
            "training_latest_sample": training_sample,
            "training_last_7_samples": training_samples[-7:],
            "heating_curve": heating_curve,
        }


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    """Set up sensors from config entry."""
    coordinator = HeatPumpForecastCoordinator(hass, entry)
    await coordinator.async_config_entry_first_refresh()

    async_add_entities(
        [
            HeatPumpForecastSensor(coordinator, entry, "tomorrow_kwh", "Verbrauch morgen", "mdi:calendar-today"),
            HeatPumpForecastSensor(coordinator, entry, "day_after_tomorrow_kwh", "Verbrauch übermorgen", "mdi:calendar-arrow-right"),
            HeatPumpRestTodaySensor(coordinator, entry),
            HeatPumpConfidenceSensor(coordinator, entry),
            HeatPumpReasonSensor(coordinator, entry),
            HeatPumpTrainingSamplesSensor(coordinator, entry),
            HeatPumpLastTrainingSampleSensor(coordinator, entry),
            HeatPumpTrainingStatusSensor(coordinator, entry),
            HeatPumpDataQualitySensor(coordinator, entry),
            HeatPumpHeatingCurveStatusSensor(coordinator, entry),
            HeatPumpHeatingCurveSensor(coordinator, entry),
        ]
    )


class HeatPumpBaseSensor(SensorEntity):
    """Base sensor."""

    _attr_has_entity_name = True

    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        """Initialize base sensor."""
        self.coordinator = coordinator
        self.entry = entry
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": entry.title,
            "manufacturer": "MTTPoll",
            "model": "Lokale Wärmepumpen-Verbrauchsprognose",
        }

    @property
    def available(self) -> bool:
        """Return availability."""
        return self.coordinator.last_update_success

    async def async_update(self) -> None:
        """Update entity."""
        await self.coordinator.async_request_refresh()


class HeatPumpForecastSensor(HeatPumpBaseSensor):
    """Energy forecast sensor."""

    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry, key: str, name: str, icon: str) -> None:
        """Initialize sensor."""
        super().__init__(coordinator, entry)
        self.key = key
        self._attr_translation_key = key
        self._attr_name = name
        self._attr_icon = icon
        self._attr_unique_id = f"{entry.entry_id}_{key}"

    @property
    def native_value(self) -> float | None:
        """Return forecast value."""
        return self.coordinator.data.get(self.key)




class HeatPumpRestTodaySensor(HeatPumpBaseSensor):
    """Remaining forecast for today until midnight."""

    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:weather-sunset-down"
    _attr_name = "Rest-Tagesprognose"
    _attr_translation_key = "rest_today_kwh"

    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        """Initialize sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_rest_today_kwh"

    @property
    def native_value(self) -> float | None:
        """Return expected remaining consumption from now until 23:59."""
        return self.coordinator.data.get("rest_today_kwh")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return details for the remaining day forecast."""
        structured = self.coordinator.data.get("reason_structured") or {}
        today_so_far = structured.get("today_so_far_kwh")
        today_forecast = self.coordinator.data.get("today_kwh")
        return {
            "Heute_bisher_kWh": today_so_far,
            "Prognose_heute_gesamt_kWh": today_forecast,
            "Rest_bis_Mitternacht_kWh": self.coordinator.data.get("rest_today_kwh"),
            "Berechnet_um": dt_util.now().isoformat(),
            "Hinweis": "Erwarteter zusätzlicher Wärmepumpenverbrauch von jetzt bis 23:59 Uhr.",
        }


class HeatPumpConfidenceSensor(HeatPumpBaseSensor):
    """Forecast quality sensor."""

    _attr_icon = "mdi:chart-bell-curve"
    _attr_name = "Prognosegüte"
    _attr_translation_key = "confidence"

    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        """Initialize sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_confidence"

    @property
    def native_value(self) -> str | None:
        """Return qualitative forecast quality."""
        return self.coordinator.data.get("confidence") or "Unzureichend"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return forecast quality details without exposing an error percentage as state."""
        return {
            "Stufe": self.coordinator.data.get("confidence_stage"),
            "Abgeschlossene_Tagesdaten": self.coordinator.data.get("confidence_completed_days"),
            "Minimum_für_gute_Prognose": self.coordinator.data.get("confidence_minimum_days"),
            "Empfohlen_für_sehr_gute_Prognose": self.coordinator.data.get("confidence_recommended_days"),
            "Interner_Basiswert": self.coordinator.data.get("confidence_internal_score"),
            "Hinweis": "Die Prognosegüte wird in fünf Stufen angezeigt: Unzureichend, Schwach, Ausreichend, Gut, Sehr gut. Prozentuale Fehlerwerte werden nicht als Sensorwert angezeigt.",
        }


class HeatPumpReasonSensor(HeatPumpBaseSensor):
    """Human readable reason/debug sensor."""

    _attr_icon = "mdi:text-box-search-outline"
    _attr_name = "Prognosegrundlage"
    _attr_translation_key = "reason"

    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        """Initialize sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_reason"

    @property
    def native_value(self) -> str | None:
        """Return short reason summary. Full details are in attributes."""
        return self.coordinator.data.get("reason_summary") or "Tageshistorie + Wetter"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return structured debug attributes."""
        structured = self.coordinator.data.get("reason_structured") or {}
        attrs = {
            "Kurzfassung": self.coordinator.data.get("reason_summary"),
            "Details": self.coordinator.data.get("reason"),
        }
        if structured:
            attrs.update({
                "Version": structured.get("version"),
                "Datenbasis": _label_source(structured.get("source")),
                "Aufteilung": _label_split_source(structured.get("split_source")),
                "Heute_bisher_kWh": structured.get("today_so_far_kwh"),
                "Durchschnitt_7_Tage_kWh": (structured.get("history") or {}).get("avg_7"),
                "Durchschnitt_14_Tage_kWh": (structured.get("history") or {}).get("avg_14"),
                "Basis_kWh": structured.get("baseline_kwh"),
                "Warmwasser_Basis_kWh": structured.get("dhw_baseline_kwh"),
                "Heizung_Basis_kWh": structured.get("heating_baseline_kwh"),
                "Heizgrenze_Grad_C": structured.get("heating_threshold_c"),
                "Heute": structured.get("today"),
                "Morgen": structured.get("tomorrow"),
                "Übermorgen": structured.get("day_after_tomorrow"),
                "Lernspeicher": structured.get("training"),
                "Heizkurve": structured.get("heating_curve"),
            })
        return attrs


class HeatPumpTrainingSamplesSensor(HeatPumpBaseSensor):
    """Training data counter sensor."""

    _attr_icon = "mdi:database-clock-outline"
    _attr_name = "Gesammelte Tagesdaten"
    _attr_translation_key = "training_samples"

    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        """Initialize sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_training_samples"

    @property
    def native_value(self) -> int | None:
        """Return number of stored learning samples."""
        return self.coordinator.data.get("training_sample_count")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return training store details."""
        return {
            "Gesammelte_Tagesdaten_ML": self.coordinator.data.get("training_sample_count"),
            "Abgeschlossene_Tagesdaten_ML": self.coordinator.data.get("training_completed_sample_count"),
            "Aktueller_Datensatz_ML": self.coordinator.data.get("training_latest_sample"),
            "Letzte_7_Datensätze_ML": self.coordinator.data.get("training_last_7_samples"),
            "Hinweis": "Sammelt neue Tagesdaten für das spätere ML-Modell. Das ist getrennt von den historischen Tagen, die aus dem Home-Assistant-Recorder für die Heizkurve gelesen werden.",
        }


class HeatPumpLastTrainingSampleSensor(HeatPumpBaseSensor):
    """Diagnostic sensor for the latest stored training sample."""

    _attr_icon = "mdi:clipboard-text-clock-outline"
    _attr_name = "Letzter Trainingsdatensatz"
    _attr_translation_key = "last_training_sample"

    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        """Initialize sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_last_training_sample"

    @property
    def native_value(self) -> str | None:
        """Return the date of the latest stored sample."""
        sample = self.coordinator.data.get("training_latest_sample") or {}
        return sample.get("date") or "Nicht verfügbar"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the latest sample in a user-readable structure."""
        sample = self.coordinator.data.get("training_latest_sample") or {}
        if not sample:
            return {"Hinweis": "Noch kein Trainingsdatensatz gespeichert."}
        weekday_names = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
        month_names = [
            "Januar", "Februar", "März", "April", "Mai", "Juni",
            "Juli", "August", "September", "Oktober", "November", "Dezember",
        ]
        weekday = sample.get("weekday")
        month = sample.get("month")
        return {
            "Datum": sample.get("date"),
            "Aktualisiert": sample.get("updated_at"),
            "Gesamtverbrauch_bisher_kWh": sample.get("actual_total_kwh_so_far"),
            "Heizverbrauch_bisher_kWh": sample.get("actual_heating_kwh_so_far"),
            "Warmwasser_bisher_kWh": sample.get("actual_dhw_kwh_so_far"),
            "Warmwasser_Basis_kWh": sample.get("estimated_dhw_basis_kwh"),
            "Heizung_Basis_kWh": sample.get("estimated_heating_basis_kwh"),
            "Temperatur_Durchschnitt_Grad_C": sample.get("avg_temperature_c"),
            "Temperatur_aktuell_Grad_C": sample.get("current_temperature_c"),
            "Heizgrenze_Grad_C": sample.get("heating_threshold_c"),
            "Personen": sample.get("persons"),
            "Belegte_WE": sample.get("occupied_units"),
            "WE_gesamt": sample.get("total_units"),
            "Belegte_Fläche_m2": sample.get("occupied_area_sqm"),
            "Gesamtfläche_m2": sample.get("total_area_sqm"),
            "Wochentag": weekday_names[int(weekday)] if isinstance(weekday, int) and 0 <= weekday <= 6 else weekday,
            "Monat": month_names[int(month) - 1] if isinstance(month, int) and 1 <= month <= 12 else month,
            "Prognose_heute_gesamt_kWh": sample.get("forecast_today_kwh"),
            "Rest_Tagesprognose_kWh": sample.get("rest_today_kwh"),
            "Prognose_morgen_kWh": sample.get("forecast_tomorrow_kwh"),
            "Prognose_übermorgen_kWh": sample.get("forecast_day_after_tomorrow_kwh"),
            "Datenbasis": _label_source(sample.get("source")),
            "Aufteilung": _label_split_source(sample.get("split_source")),
        }


class HeatPumpTrainingStatusSensor(HeatPumpBaseSensor):
    """Diagnostic sensor that shows whether enough data is available for ML."""

    _attr_icon = "mdi:brain"
    _attr_name = "Trainingsstatus"
    _attr_translation_key = "training_status"

    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        """Initialize sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_training_status"

    @property
    def native_value(self) -> str | None:
        """Return training readiness."""
        total = int(self.coordinator.data.get("training_sample_count") or 0)
        completed = int(self.coordinator.data.get("training_completed_sample_count") or 0)
        minimum = 30
        if completed >= minimum:
            return "ML bereit"
        visible_count = max(total, completed)
        return f"Sammelt Daten ({visible_count}/{minimum})"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return readiness details."""
        total = int(self.coordinator.data.get("training_sample_count") or 0)
        completed = int(self.coordinator.data.get("training_completed_sample_count") or 0)
        minimum = 30
        recommended = 90
        visible_count = max(total, completed)
        return {
            "Gesammelte_Tagesdaten_ML": visible_count,
            "Datensätze_ML_gesamt": total,
            "Abgeschlossene_Tagesdaten": completed,
            "Minimum_für_ML": minimum,
            "Empfohlen_für_ML": recommended,
            "Fortschritt_Prozent": min(100, round((visible_count / minimum) * 100)) if minimum else 0,
            "ML_bereit": completed >= minimum,
            "ML_aktiv": False,
            "Hinweis": "Die Integration sammelt Tagesdaten. ML wird erst aktiviert, wenn genügend abgeschlossene Tagesdatensätze vorhanden sind.",
        }


class HeatPumpDataQualitySensor(HeatPumpBaseSensor):
    """Diagnostic sensor that summarizes training data quality."""

    _attr_icon = "mdi:database-check-outline"
    _attr_name = "Datenqualität"
    _attr_translation_key = "data_quality"

    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        """Initialize sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_data_quality"

    def _quality_flags(self) -> dict[str, Any]:
        """Return simple quality flags for the latest training sample."""
        sample = self.coordinator.data.get("training_latest_sample") or {}
        has_temperature = sample.get("avg_temperature_c") is not None or sample.get("current_temperature_c") is not None
        has_consumption = sample.get("actual_total_kwh_so_far") is not None or sample.get("estimated_dhw_basis_kwh") is not None
        has_heating = sample.get("actual_heating_kwh_so_far") is not None or sample.get("estimated_heating_basis_kwh") is not None
        has_occupancy = (sample.get("total_units") or 0) > 0 or sample.get("persons") is not None
        missing = 0
        for ok in (has_temperature, has_consumption, has_occupancy):
            if not ok:
                missing += 1
        return {
            "temperature": has_temperature,
            "consumption": has_consumption,
            "heating": has_heating,
            "occupancy": has_occupancy,
            "missing": missing,
        }

    @property
    def native_value(self) -> str | None:
        """Return quality summary as percentage and label."""
        flags = self._quality_flags()
        score = 100
        score -= int(flags["missing"] or 0) * 25
        if not flags["heating"]:
            score -= 15
        score = max(0, min(100, score))
        if score >= 85:
            label = "Gut"
        elif score >= 60:
            label = "Eingeschränkt"
        else:
            label = "Fehlerhaft"
        return f"{score} % ({label})"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return quality details."""
        flags = self._quality_flags()
        samples = self.coordinator.data.get("training_last_7_samples") or []
        invalid = 0
        for sample in samples:
            if sample.get("actual_total_kwh_so_far") is None and sample.get("estimated_dhw_basis_kwh") is None:
                invalid += 1
        return {
            "Temperatur_vorhanden": flags["temperature"],
            "Verbrauch_vorhanden": flags["consumption"],
            "Heizdaten_vorhanden": flags["heating"],
            "Belegung_vorhanden": flags["occupancy"],
            "Fehlende_Datenbereiche": flags["missing"],
            "Ungültige_ML_Datensätze_letzte_7": invalid,
            "Gültige_ML_Datensätze_letzte_7": max(0, len(samples) - invalid),
            "Fehlende_Temperaturdaten": not flags["temperature"],
            "Fehlende_Verbrauchsdaten": not flags["consumption"],
            "Fehlende_Belegungsdaten": not flags["occupancy"],
            "Hinweis": "Diese Bewertung prüft die Datenbasis. Sie ist noch keine ML-Genauigkeitsbewertung.",
        }


class HeatPumpHeatingCurveStatusSensor(HeatPumpBaseSensor):
    """Diagnostic sensor for learned heating curve readiness."""

    _attr_icon = "mdi:chart-bell-curve-cumulative"
    _attr_name = "Heizkurvenstatus"
    _attr_translation_key = "heating_curve_status"

    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        """Initialize sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_heating_curve_status"

    @property
    def native_value(self) -> str | None:
        curve = self.coordinator.data.get("heating_curve") or {}
        return curve.get("status") or "Keine Heizkurvendaten"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        curve = self.coordinator.data.get("heating_curve") or {}
        return {
            "Aktiv": curve.get("active"),
            "Bereit": curve.get("ready"),
            "Optimiert": curve.get("optimized"),
            "Heizkurve_bereit": curve.get("ready"),
            "Historische_Tage_Heizkurve": curve.get("paired_days"),
            "Minimum_Tage": curve.get("minimum_days"),
            "Empfohlen_Tage": curve.get("recommended_days"),
            "Heizgrenze_Grad_C": curve.get("threshold_c"),
            "Fortschritt_Prozent": curve.get("progress_percent"),
            "Hinweis": "Lernt den Zusammenhang zwischen Tagesdurchschnittstemperatur und Heizverbrauch aus historischen Recorder-Daten. Diese historischen Tage sind getrennt von den neu gesammelten ML-Tagesdaten.",
        }


class HeatPumpHeatingCurveSensor(HeatPumpBaseSensor):
    """Sensor exposing the learned heating curve buckets."""

    _attr_icon = "mdi:thermometer-lines"
    _attr_name = "Erlernte Heizkurve"
    _attr_translation_key = "heating_curve"

    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        """Initialize sensor."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_heating_curve"

    @property
    def native_value(self) -> str | None:
        curve = self.coordinator.data.get("heating_curve") or {}
        if not curve.get("available"):
            return "Keine Daten"
        return curve.get("status") or "Wird aufgebaut"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        curve = self.coordinator.data.get("heating_curve") or {}
        return {
            "Heizgrenze_Grad_C": curve.get("threshold_c"),
            "Historische_Tage_Heizkurve": curve.get("paired_days"),
            "Aktiv": curve.get("active"),
            "Optimiert": curve.get("optimized"),
            "Heizkurve_bereit": curve.get("ready"),
            "Minimum_Tage": curve.get("minimum_days"),
            "Empfohlen_Tage": curve.get("recommended_days"),
            "Fortschritt_Prozent": curve.get("progress_percent"),
            "Unter_5_Grad_kWh": (curve.get("bucket_avgs_kwh") or {}).get("unter_5"),
            "5_bis_10_Grad_kWh": (curve.get("bucket_avgs_kwh") or {}).get("5_bis_10"),
            "10_bis_15_Grad_kWh": (curve.get("bucket_avgs_kwh") or {}).get("10_bis_15"),
            "15_bis_Heizgrenze_kWh": (curve.get("bucket_avgs_kwh") or {}).get("15_bis_heizgrenze"),
            "Über_Heizgrenze_kWh": (curve.get("bucket_avgs_kwh") or {}).get("ueber_heizgrenze"),
            "Anzahl_je_Bereich": curve.get("bucket_counts"),
            "Letzte_30_Historienpaare": curve.get("pairs_last_30"),
        }
