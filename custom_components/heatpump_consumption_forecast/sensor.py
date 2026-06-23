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
    DHW_DAILY_ENERGY_MODE_NONE,
    CONF_FIXED_PERSONS,
    CONF_HEATING_DAILY_ENERGY_SENSOR,
    CONF_HEATING_THRESHOLD_TEMP,
    CONF_HEATING_CURVE_FLOW_WARM,
    CONF_HEATING_CURVE_FLOW_MID,
    CONF_HEATING_CURVE_FLOW_COLD,
    CONF_HEATING_CURVE_SAVING_PERCENT_PER_C,
    CONF_HEATING_CURVE_SIMULATION_ENABLED,
    CONF_DHW_TARGET_TEMP,
    CONF_DHW_TANK_VOLUME_L,
    CONF_DHW_LITERS_PER_PERSON,
    DEFAULT_DHW_TARGET_TEMP,
    DEFAULT_DHW_TANK_VOLUME_L,
    DEFAULT_DHW_LITERS_PER_PERSON,
    DEFAULT_COLD_WATER_TEMP,
    RUNTIME_HEATING_CURVE_DELTA_C,
    RUNTIME_DHW_TARGET_DELTA_C,
    DEFAULT_HEATING_CURVE_FLOW_WARM,
    DEFAULT_HEATING_CURVE_FLOW_MID,
    DEFAULT_HEATING_CURVE_FLOW_COLD,
    DEFAULT_HEATING_CURVE_SAVING_PERCENT_PER_C,
    HEATING_CURVE_OUTDOOR_WARM,
    HEATING_CURVE_OUTDOOR_MID,
    HEATING_CURVE_OUTDOOR_COLD,
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
        "no_dhw_meter_total_minus_heating": "Kein Warmwasserzähler: Gesamt minus Heizung",
        "no_dhw_meter_total_as_dhw": "Kein Warmwasserzähler: Gesamtverbrauch als Warmwasser",
        "summer_total_as_dhw": "Sommerbetrieb: Gesamtverbrauch als Warmwasser",
        "estimated_no_dhw_meter_total_minus_heating": "Kein Warmwasserzähler: Heizung geschätzt, Warmwasser abgeleitet",
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





def _interpolate_flow_temperature(
    outdoor_temp_c: float | None,
    flow_cold: float,
    flow_mid: float,
    flow_warm: float,
) -> float | None:
    """Calculate target flow temperature from three heating-curve support points."""
    temp = _to_float(outdoor_temp_c)
    if temp is None:
        return None

    x_cold = float(HEATING_CURVE_OUTDOOR_COLD)
    x_mid = float(HEATING_CURVE_OUTDOOR_MID)
    x_warm = float(HEATING_CURVE_OUTDOOR_WARM)

    def _lin(x_value: float, x_a: float, y_a: float, x_b: float, y_b: float) -> float:
        return y_a + (x_value - x_a) * (y_b - y_a) / (x_b - x_a)

    if temp >= x_warm:
        flow = flow_warm
    elif temp > x_mid:
        flow = _lin(temp, x_mid, flow_mid, x_warm, flow_warm)
    elif temp > x_cold:
        flow = _lin(temp, x_cold, flow_cold, x_mid, flow_mid)
    else:
        flow = flow_cold
    return round(float(flow), 2)


def _build_heating_curve_simulation(
    *,
    enabled: bool,
    avg_temp_today: float | None,
    avg_temp_tomorrow: float | None,
    avg_temp_day_after: float | None,
    heating_threshold: float,
    flow_cold: float,
    flow_mid: float,
    flow_warm: float,
    saving_percent_per_c: float,
    heating_curve_delta_c: float,
    dhw_target_delta_c: float,
    dhw_target_temp_c: float,
    dhw_tank_volume_l: float,
    dhw_liters_per_person: float,
    today_rule: dict[str, Any],
    tomorrow_rule: dict[str, Any],
    day_after_rule: dict[str, Any],
    today_persons: float,
    tomorrow_persons: float,
    day_after_persons: float,
) -> dict[str, Any]:
    """Build interactive heating and DHW energy simulation.

    Heating-curve changes are applied only to the heating share and only below
    the heating threshold. DHW target-temperature changes are applied only to
    the DHW share. The DHW calculation is based on learned daily DHW energy,
    configured tank volume and target temperature, so occupancy and learned
    draw-off behavior remain part of the forecast.
    """
    flow_cold = float(flow_cold)
    flow_mid = float(flow_mid)
    flow_warm = float(flow_warm)
    saving_percent_per_c = max(0.0, min(10.0, float(saving_percent_per_c or 0.0)))
    heating_curve_delta_c = max(-5.0, min(5.0, float(heating_curve_delta_c or 0.0)))
    dhw_target_delta_c = max(-10.0, min(10.0, float(dhw_target_delta_c or 0.0)))
    dhw_target_temp_c = max(25.0, min(75.0, float(dhw_target_temp_c or DEFAULT_DHW_TARGET_TEMP)))
    dhw_tank_volume_l = max(0.0, min(2000.0, float(dhw_tank_volume_l or 0.0)))
    dhw_liters_per_person = max(0.0, min(250.0, float(dhw_liters_per_person or DEFAULT_DHW_LITERS_PER_PERSON)))
    cold_water_temp_c = float(DEFAULT_COLD_WATER_TEMP)

    def _dhw_change_for_day(dhw_kwh: float, persons: float) -> dict[str, Any]:
        persons = max(0.0, float(persons or 0.0))
        daily_liters = persons * dhw_liters_per_person
        demand_based_cycles = (daily_liters / dhw_tank_volume_l) if dhw_tank_volume_l > 0 else None
        if dhw_kwh <= 0 or dhw_target_delta_c == 0:
            return {
                "saving_kwh_day": 0.0,
                "additional_kwh_day": 0.0,
                "change_kwh_day": 0.0,
                "estimated_tank_cycles_per_day": 0.0,
                "estimated_tank_cycles_per_day_by_persons": round(demand_based_cycles, 2) if demand_based_cycles is not None else None,
                "persons": round(persons, 2),
                "liters_per_person": dhw_liters_per_person,
                "daily_liters": round(daily_liters, 1),
                "tank_volume_l": dhw_tank_volume_l,
                "energy_per_full_tank_charge_kwh": 0.0,
                "energy_change_per_full_tank_kwh": 0.0,
            }
        effective_delta = max(1.0, dhw_target_temp_c - cold_water_temp_c)
        new_effective_delta = max(1.0, dhw_target_temp_c + dhw_target_delta_c - cold_water_temp_c)
        new_dhw_kwh = dhw_kwh * (new_effective_delta / effective_delta)
        change_kwh = new_dhw_kwh - dhw_kwh
        energy_per_full_tank = dhw_tank_volume_l * 0.001163 * effective_delta if dhw_tank_volume_l > 0 else None
        energy_change_per_tank = dhw_tank_volume_l * 0.001163 * abs(dhw_target_delta_c) if dhw_tank_volume_l > 0 else None
        estimated_cycles = (dhw_kwh / energy_per_full_tank) if energy_per_full_tank and energy_per_full_tank > 0 else None
        relative_change_percent = (dhw_target_delta_c / effective_delta * 100.0) if effective_delta > 0 else 0.0
        return {
            "saving_kwh_day": round(max(-change_kwh, 0.0), 3),
            "additional_kwh_day": round(max(change_kwh, 0.0), 3),
            "change_kwh_day": round(change_kwh, 3),
            "estimated_tank_cycles_per_day": round(estimated_cycles, 2) if estimated_cycles is not None else None,
            "estimated_tank_cycles_per_day_by_persons": round(demand_based_cycles, 2) if demand_based_cycles is not None else None,
            "persons": round(persons, 2),
            "liters_per_person": dhw_liters_per_person,
            "daily_liters": round(daily_liters, 1),
            "tank_volume_l": dhw_tank_volume_l,
            "learned_dhw_kwh_day": round(dhw_kwh, 3),
            "target_temp_c": dhw_target_temp_c,
            "target_delta_c": round(dhw_target_delta_c, 2),
            "cold_water_reference_c": cold_water_temp_c,
            "temperature_hub_c": round(effective_delta, 2),
            "relative_temp_change_percent": round(relative_change_percent, 2),
            "energy_per_full_tank_charge_kwh": round(energy_per_full_tank, 3) if energy_per_full_tank is not None else None,
            "energy_change_per_full_tank_kwh": round(energy_change_per_tank, 3) if energy_change_per_tank is not None else None,
        }

    def _scenario_for_day(label: str, avg_temp: float | None, rule: dict[str, Any], persons: float) -> dict[str, Any]:
        base_flow = _interpolate_flow_temperature(avg_temp, flow_cold, flow_mid, flow_warm)
        total_kwh = float(rule.get("total_kwh") or 0.0)
        heating_kwh = float(rule.get("heating_kwh") or 0.0)
        dhw_kwh = float(rule.get("dhw_kwh") or 0.0)
        heating_active = bool(rule.get("heating_active")) and avg_temp is not None and float(avg_temp) < float(heating_threshold)

        # A negative heating delta lowers the flow temperature and saves energy.
        # A positive heating delta raises the curve and adds consumption.
        if heating_active and heating_kwh > 0 and heating_curve_delta_c != 0:
            heating_change_kwh = heating_kwh * (saving_percent_per_c / 100.0) * heating_curve_delta_c
        else:
            heating_change_kwh = 0.0
        dhw_change = _dhw_change_for_day(dhw_kwh, persons)
        total_change_kwh = heating_change_kwh + float(dhw_change.get("change_kwh_day") or 0.0)
        new_forecast = max(0.0, total_kwh + total_change_kwh)
        net_saving = max(-total_change_kwh, 0.0)
        net_additional = max(total_change_kwh, 0.0)

        simulations: dict[str, Any] = {}
        for delta_c in (-3, -2, -1, 1, 2, 3):
            if heating_active and heating_kwh > 0:
                change_kwh = heating_kwh * (saving_percent_per_c / 100.0) * delta_c
            else:
                change_kwh = 0.0
            simulations[f"{'minus' if delta_c < 0 else 'plus'}_{abs(delta_c)}c"] = {
                "flow_target_c": round(base_flow + delta_c, 2) if base_flow is not None else None,
                "delta_flow_c": delta_c,
                "saving_kwh_day": round(max(-change_kwh, 0.0), 3),
                "additional_kwh_day": round(max(change_kwh, 0.0), 3),
                "change_kwh_day": round(change_kwh, 3),
                "saving_percent_heating": round(saving_percent_per_c * abs(delta_c), 2) if heating_active else 0.0,
                "new_forecast_kwh": round(max(0.0, total_kwh + change_kwh), 3),
            }

        return {
            "label": label,
            "avg_temperature_c": avg_temp,
            "heating_threshold_c": heating_threshold,
            "summer_mode": not heating_active,
            "current_flow_target_c": base_flow,
            "heating_share_kwh": round(heating_kwh, 3),
            "dhw_share_kwh": round(dhw_kwh, 3),
            "base_forecast_kwh": round(total_kwh, 3),
            "selected_simulation": {
                "heating_curve_delta_c": round(heating_curve_delta_c, 2),
                "dhw_target_delta_c": round(dhw_target_delta_c, 2),
                "simulated_flow_target_c": round(base_flow + heating_curve_delta_c, 2) if base_flow is not None else None,
                "simulated_dhw_target_temp_c": round(dhw_target_temp_c + dhw_target_delta_c, 2),
                "heating_change_kwh_day": round(heating_change_kwh, 3),
                "heating_saving_kwh_day": round(max(-heating_change_kwh, 0.0), 3),
                "heating_additional_kwh_day": round(max(heating_change_kwh, 0.0), 3),
                "dhw_change_kwh_day": dhw_change.get("change_kwh_day"),
                "dhw_saving_kwh_day": dhw_change.get("saving_kwh_day"),
                "dhw_additional_kwh_day": dhw_change.get("additional_kwh_day"),
                "total_change_kwh_day": round(total_change_kwh, 3),
                "total_saving_kwh_day": round(net_saving, 3),
                "total_additional_kwh_day": round(net_additional, 3),
                "new_forecast_kwh": round(new_forecast, 3),
                "change_percent_total": round((total_change_kwh / total_kwh * 100.0), 2) if total_kwh > 0 else 0.0,
                "dhw_details": dhw_change,
                "dhw_demand": {
                    "persons": dhw_change.get("persons"),
                    "liters_per_person": dhw_change.get("liters_per_person"),
                    "daily_liters": dhw_change.get("daily_liters"),
                    "tank_volume_l": dhw_change.get("tank_volume_l"),
                    "estimated_tank_cycles_per_day_by_persons": dhw_change.get("estimated_tank_cycles_per_day_by_persons"),
                    "learned_dhw_kwh_day": dhw_change.get("learned_dhw_kwh_day"),
                    "target_temp_c": dhw_change.get("target_temp_c"),
                    "cold_water_reference_c": dhw_change.get("cold_water_reference_c"),
                    "temperature_hub_c": dhw_change.get("temperature_hub_c"),
                    "relative_temp_change_percent": dhw_change.get("relative_temp_change_percent"),
                    "simulated_dhw_change_kwh": dhw_change.get("change_kwh_day"),
                },
            },
            "simulation": simulations,
            "note": "Heizkurven-Effekt ist 0 kWh, weil die Heizgrenze erreicht oder überschritten ist." if not heating_active else "Heizkurve wirkt nur auf Heizanteil; Warmwasser wird separat simuliert.",
        }

    today = _scenario_for_day("today", avg_temp_today, today_rule, today_persons)
    tomorrow = _scenario_for_day("tomorrow", avg_temp_tomorrow, tomorrow_rule, tomorrow_persons)
    day_after = _scenario_for_day("day_after_tomorrow", avg_temp_day_after, day_after_rule, day_after_persons)
    tomorrow_selected = tomorrow["selected_simulation"]

    return {
        "enabled": bool(enabled),
        "available": bool(enabled),
        "algorithm": "Interaktive Energie-Simulation: 3-Punkt-Heizkurve + Warmwasser-Solltemperatur",
        "heating_curve_points": {
            "cold": {"outside_c": HEATING_CURVE_OUTDOOR_COLD, "flow_c": flow_cold},
            "mid": {"outside_c": HEATING_CURVE_OUTDOOR_MID, "flow_c": flow_mid},
            "warm": {"outside_c": HEATING_CURVE_OUTDOOR_WARM, "flow_c": flow_warm},
        },
        "saving_factor_percent_per_c": saving_percent_per_c,
        "dhw": {
            "target_temp_c": dhw_target_temp_c,
            "target_delta_c": dhw_target_delta_c,
            "tank_volume_l": dhw_tank_volume_l,
            "liters_per_person": dhw_liters_per_person,
            "cold_water_reference_c": cold_water_temp_c,
        },
        "runtime_controls": {
            "heating_curve_delta_c": heating_curve_delta_c,
            "dhw_target_delta_c": dhw_target_delta_c,
            "note": "Diese Werte kommen aus Number-Entitäten und können im laufenden Betrieb geändert werden.",
        },
        "important_note": "Heizkurve wirkt nur auf Heizanteil. Warmwasser-Solltemperatur wirkt nur auf Warmwasseranteil. Im Sommerbetrieb ist der Heizkurven-Effekt 0 kWh.",
        "today": today,
        "tomorrow": tomorrow,
        "day_after_tomorrow": day_after,
        "summary": {
            "primary_basis": "tomorrow",
            "basis_label": "Prognose morgen",
            "basis_forecast_kwh": tomorrow.get("base_forecast_kwh"),
            "selected_effect_kwh": tomorrow_selected["total_change_kwh_day"],
            "effect_type": "saving" if float(tomorrow_selected["total_change_kwh_day"] or 0.0) < 0 else "additional_consumption" if float(tomorrow_selected["total_change_kwh_day"] or 0.0) > 0 else "no_change",
            "new_forecast_kwh": tomorrow_selected["new_forecast_kwh"],
            "heating_curve_delta_c": round(heating_curve_delta_c, 2),
            "dhw_delta_c": round(dhw_target_delta_c, 2),
            "simulation_summary": f"Prognose morgen {float(tomorrow.get('base_forecast_kwh') or 0.0):.2f} kWh -> neue Prognose {float(tomorrow_selected['new_forecast_kwh'] or 0.0):.2f} kWh ({float(tomorrow_selected['total_change_kwh_day'] or 0.0):+.3f} kWh)",
            "tomorrow_selected_total_change_kwh": tomorrow_selected["total_change_kwh_day"],
            "tomorrow_selected_total_saving_kwh": tomorrow_selected["total_saving_kwh_day"],
            "tomorrow_selected_total_additional_kwh": tomorrow_selected["total_additional_kwh_day"],
            "tomorrow_selected_new_forecast_kwh": tomorrow_selected["new_forecast_kwh"],
            "tomorrow_summer_mode": tomorrow["summer_mode"],
        },
    }


def _safe_error_percent(actual: float | None, forecast: float | None) -> float | None:
    """Return absolute forecast error in percent."""
    actual_f = _to_float(actual)
    forecast_f = _to_float(forecast)
    if actual_f is None or forecast_f is None or actual_f <= 0:
        return None
    return abs(actual_f - forecast_f) / actual_f * 100.0


def _forecast_quality_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate forecast-quality metrics from completed samples."""
    rows: list[dict[str, Any]] = []
    for sample in samples or []:
        if not sample.get("completed"):
            continue
        actual = _to_float(sample.get("actual_total_kwh_final"))
        forecast = _to_float(sample.get("forecast_today_kwh"))
        if actual is None or forecast is None or actual <= 0:
            continue
        error = actual - forecast
        rows.append({
            "date": sample.get("date"),
            "actual_kwh": round(actual, 3),
            "forecast_kwh": round(forecast, 3),
            "error_kwh": round(error, 3),
            "abs_error_kwh": round(abs(error), 3),
            "error_percent": round(abs(error) / actual * 100.0, 2),
        })

    def _window(window_rows: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(window_rows)
        if not count:
            return {"sample_count": 0, "mae_kwh": None, "mape_percent": None, "rmse_kwh": None}
        abs_errors = [float(row["abs_error_kwh"]) for row in window_rows]
        pct_errors = [float(row["error_percent"]) for row in window_rows]
        sq_errors = [float(row["error_kwh"]) ** 2 for row in window_rows]
        return {
            "sample_count": count,
            "mae_kwh": round(sum(abs_errors) / count, 3),
            "mape_percent": round(sum(pct_errors) / count, 2),
            "rmse_kwh": round((sum(sq_errors) / count) ** 0.5, 3),
        }

    all_metrics = _window(rows)
    last_7 = _window(rows[-7:])
    last_30 = _window(rows[-30:])
    last_90 = _window(rows[-90:])
    active_window = "last_30" if last_30.get("sample_count") else "all"
    mape = last_30.get("mape_percent") if last_30.get("sample_count") else all_metrics.get("mape_percent")
    accuracy_percent = round(max(0.0, min(100.0, 100.0 - float(mape))), 2) if mape is not None else None
    if mape is None:
        quality_label = "Keine Bewertung"
        quality_description = "Noch keine abgeschlossenen Vergleichsdaten vorhanden."
    elif mape <= 5:
        quality_label = "Exzellent"
        quality_description = "Die Prognose liegt sehr nah am tatsächlichen Verbrauch."
    elif mape <= 10:
        quality_label = "Sehr gut"
        quality_description = "Die Prognose ist sehr zuverlässig."
    elif mape <= 15:
        quality_label = "Gut"
        quality_description = "Die Prognose ist gut nutzbar."
    elif mape <= 25:
        quality_label = "Ausreichend"
        quality_description = "Die Prognose ist brauchbar, sollte aber weiter beobachtet werden."
    elif mape <= 40:
        quality_label = "Schwach"
        quality_description = "Die Prognose weicht noch deutlich ab. Weitere Trainingsdaten sind sinnvoll."
    else:
        quality_label = "Unzureichend"
        quality_description = "Die Prognose ist noch nicht belastbar."
    summary = (
        f"{quality_label}: {accuracy_percent:.2f}% Genauigkeit, MAPE {float(mape):.2f}%"
        if mape is not None and accuracy_percent is not None
        else quality_description
    )
    return {
        "available": bool(rows),
        "sample_count": len(rows),
        "quality_label": quality_label,
        "quality_description": quality_description,
        "accuracy_percent": accuracy_percent,
        "mape_used_percent": round(float(mape), 2) if mape is not None else None,
        "active_window": active_window,
        "summary": summary,
        "latest": rows[-1] if rows else None,
        "last_7": last_7,
        "last_30": last_30,
        "last_90": last_90,
        "all": all_metrics,
        "history_last_30": rows[-30:],
    }



def _learning_analysis_series(
    samples: list[dict[str, Any]],
    heating_curve: dict[str, Any],
    forecast_quality: dict[str, Any],
    ml_status: dict[str, Any],
    model_selection: dict[str, Any],
) -> dict[str, Any]:
    """Build compact chart-friendly learning analysis series for HA dashboards."""
    completed = [s for s in samples or [] if isinstance(s, dict) and s.get("completed")]
    completed_sorted = sorted(completed, key=lambda item: str(item.get("date") or ""))

    learning_progress_series: list[dict[str, Any]] = []
    for idx, sample in enumerate(completed_sorted[-90:], start=max(1, len(completed_sorted[-90:]) - len(completed_sorted[-90:]) + 1)):
        learning_progress_series.append({
            "date": sample.get("date"),
            "completed_days": max(0, len(completed_sorted) - len(completed_sorted[-90:])) + idx,
            "forecast_model": sample.get("selected_model") or sample.get("forecast_model") or sample.get("source") or "Regelmodell",
            "ml_status": "Optimiert" if max(0, len(completed_sorted) - len(completed_sorted[-90:])) + idx >= ML_OPTIMIZED_DAYS else "Aktiv" if max(0, len(completed_sorted) - len(completed_sorted[-90:])) + idx >= ML_MINIMUM_DAYS else "Wartet auf Trainingsdaten",
        })

    forecast_error_series = (forecast_quality or {}).get("history_last_30") or []

    heating_curve_series: list[dict[str, Any]] = []
    for pair in (heating_curve or {}).get("pairs_last_30") or []:
        heating_curve_series.append({
            "temperature_c": pair.get("temperature_c"),
            "heating_kwh": pair.get("heating_kwh"),
        })

    ml_quality_series = [
        {"window": "last_7", **((forecast_quality or {}).get("last_7") or {})},
        {"window": "last_30", **((forecast_quality or {}).get("last_30") or {})},
        {"window": "last_90", **((forecast_quality or {}).get("last_90") or {})},
        {"window": "all", **((forecast_quality or {}).get("all") or {})},
    ]

    model_selection_series = []
    if learning_progress_series:
        for row in learning_progress_series[-30:]:
            completed_days = int(row.get("completed_days") or 0)
            model_selection_series.append({
                "date": row.get("date"),
                "completed_days": completed_days,
                "model": "Regelmodell" if completed_days < ML_MINIMUM_DAYS else (model_selection or {}).get("mode", "ML + Fallback"),
                "ml_allowed": completed_days >= ML_MINIMUM_DAYS,
            })
    else:
        model_selection_series.append({
            "date": None,
            "completed_days": len(completed_sorted),
            "model": (model_selection or {}).get("mode", "Regelmodell"),
            "ml_allowed": bool((model_selection or {}).get("ml_allowed")),
        })

    return {
        "available": bool(completed_sorted or forecast_error_series or heating_curve_series),
        "completed_days": len(completed_sorted),
        "learning_progress_series": learning_progress_series,
        "forecast_error_series": forecast_error_series,
        "heating_curve_series": heating_curve_series,
        "ml_quality_series": ml_quality_series,
        "model_selection_series": model_selection_series,
        "dashboard_hint": "Diese Attribute sind für ApexCharts, Plotly oder History-Dashboard-Karten gedacht.",
    }


def _select_model_strategy(ml_status: dict[str, Any], quality: dict[str, Any]) -> dict[str, Any]:
    """Return the automatic model selection strategy."""
    if not ml_status.get("active"):
        return {
            "mode": "Regelmodell",
            "ml_allowed": False,
            "reason": "ML wartet auf ausreichend abgeschlossene Trainingsdaten.",
        }
    sample_count = int((quality or {}).get("sample_count") or 0)
    if sample_count < 7:
        return {
            "mode": "ML + Fallback",
            "ml_allowed": True,
            "reason": "ML ist aktiv, aber die Qualitätsbewertung hat noch weniger als 7 Vergleichstage.",
        }
    mape = ((quality.get("last_30") or {}).get("mape_percent") or (quality.get("all") or {}).get("mape_percent"))
    if mape is None:
        return {"mode": "ML + Fallback", "ml_allowed": True, "reason": "Noch keine belastbare Fehlerquote verfügbar."}
    if float(mape) <= 25:
        return {"mode": "ML bevorzugt", "ml_allowed": True, "reason": f"ML-Qualität ist gut genug (MAPE {mape}%)."}
    if float(mape) <= 40:
        return {"mode": "ML + Fallback", "ml_allowed": True, "reason": f"ML wird nur mit Plausibilitätsprüfung genutzt (MAPE {mape}%)."}
    return {"mode": "Regelmodell bevorzugt", "ml_allowed": False, "reason": f"Regelmodell wird bevorzugt, weil die bisherige Fehlerquote hoch ist (MAPE {mape}%)."}


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


    def _repair_no_dhw_meter_training_values(self, sample: dict[str, Any], dhw_meter_missing: bool = False) -> dict[str, Any]:
        """Fill derived DHW/heating values for installations without a DHW meter.

        In no-DHW-meter mode DHW is always derived as total minus heating. This
        also repairs older samples so ML training does not learn from null split
        values when the total daily energy is known.
        """
        mode = sample.get("dhw_meter_mode")
        split_source = sample.get("split_source") or sample.get("actual_split_source")
        no_dhw_meter_sample = bool(
            dhw_meter_missing
            or mode == DHW_DAILY_ENERGY_MODE_NONE
            or (isinstance(split_source, str) and split_source.startswith("no_dhw_meter"))
        )
        if not no_dhw_meter_sample:
            return sample

        def _round_or_none(value: Any) -> float | None:
            value_f = _to_float(value)
            return round(value_f, 3) if value_f is not None else None

        sample["dhw_meter_mode"] = DHW_DAILY_ENERGY_MODE_NONE
        sample["split_source"] = "no_dhw_meter_total_minus_heating"

        total_so_far_f = _to_float(sample.get("actual_total_kwh_so_far"))
        heating_so_far_f = _to_float(sample.get("actual_heating_kwh_so_far"))
        if total_so_far_f is not None:
            if heating_so_far_f is None:
                heating_so_far_f = 0.0
            heating_so_far_f = max(0.0, min(total_so_far_f, heating_so_far_f))
            sample["actual_total_kwh_so_far"] = round(total_so_far_f, 3)
            sample["actual_heating_kwh_so_far"] = round(heating_so_far_f, 3)
            sample["actual_dhw_kwh_so_far"] = round(max(total_so_far_f - heating_so_far_f, 0.0), 3)

        total_final_f = _to_float(sample.get("actual_total_kwh_final"))
        heating_final_f = _to_float(sample.get("actual_heating_kwh_final"))
        if total_final_f is not None:
            if heating_final_f is None:
                heating_final_f = _to_float(sample.get("actual_heating_kwh_so_far"))
            if heating_final_f is None:
                heating_final_f = 0.0
            heating_final_f = max(0.0, min(total_final_f, heating_final_f))
            sample["actual_total_kwh_final"] = round(total_final_f, 3)
            sample["actual_heating_kwh_final"] = round(heating_final_f, 3)
            sample["actual_dhw_kwh_final"] = round(max(total_final_f - heating_final_f, 0.0), 3)
            sample["actual_split_source"] = "no_dhw_meter_total_minus_heating"
            if sample.get("completed") is True:
                # For completed days the so-far values should be complete too.
                sample["actual_total_kwh_so_far"] = round(total_final_f, 3)
                sample["actual_heating_kwh_so_far"] = round(heating_final_f, 3)
                sample["actual_dhw_kwh_so_far"] = round(max(total_final_f - heating_final_f, 0.0), 3)

        return sample

    def _finalize_previous_training_samples(self, samples: list[dict[str, Any]], today_date: str, total_series: dict[str, Any], heating_series: dict[str, Any], dhw_series: dict[str, Any], dhw_meter_missing: bool = False) -> list[dict[str, Any]]:
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
            split_source_final = sample.get("split_source")
            sample_avg_temp = _to_float(sample.get("avg_temperature_c"))
            sample_threshold = _to_float(sample.get("heating_threshold_c"))

            if dhw_meter_missing:
                if final_heating_f is None:
                    if sample_avg_temp is not None and sample_threshold is not None and sample_avg_temp >= sample_threshold:
                        final_heating_f = 0.0
                        split_source_final = "no_dhw_meter_total_as_dhw"
                    else:
                        estimated_heating = _to_float(sample.get("estimated_heating_basis_kwh"))
                        final_heating_f = max(0.0, min(final_total_f, estimated_heating or 0.0))
                        split_source_final = "estimated_no_dhw_meter_total_minus_heating"
                final_dhw_f = max(final_total_f - float(final_heating_f or 0.0), 0.0)
                if split_source_final not in ("no_dhw_meter_total_as_dhw", "estimated_no_dhw_meter_total_minus_heating"):
                    split_source_final = "no_dhw_meter_total_minus_heating"
            else:
                if final_dhw_f is None and final_heating_f is not None:
                    final_dhw_f = max(final_total_f - final_heating_f, 0.0)
                    split_source_final = "total_minus_heating"
                if final_heating_f is None and final_dhw_f is not None:
                    final_heating_f = max(final_total_f - final_dhw_f, 0.0)
                    split_source_final = "total_minus_dhw"
                if final_heating_f is None and final_dhw_f is None and sample_avg_temp is not None and sample_threshold is not None and sample_avg_temp >= sample_threshold:
                    final_heating_f = 0.0
                    final_dhw_f = final_total_f
                    split_source_final = "summer_total_as_dhw"

            forecast_today_f = _to_float(sample.get("forecast_today_kwh"))
            sample["completed"] = True
            sample["completed_at"] = now_iso
            sample.pop("completion_pending_reason", None)
            sample["actual_total_kwh_final"] = round(final_total_f, 3)
            sample["actual_heating_kwh_final"] = round(final_heating_f, 3) if final_heating_f is not None else None
            sample["actual_dhw_kwh_final"] = round(final_dhw_f, 3) if final_dhw_f is not None else None
            sample["actual_split_source"] = split_source_final
            if dhw_meter_missing:
                self._repair_no_dhw_meter_training_values(sample, dhw_meter_missing=True)
            if forecast_today_f is not None:
                error = final_total_f - forecast_today_f
                sample["forecast_error_kwh"] = round(error, 3)
                sample["forecast_abs_error_kwh"] = round(abs(error), 3)
        return samples

    async def _async_store_training_sample(self, sample: dict[str, Any], total_series: dict[str, Any] | None = None, heating_series: dict[str, Any] | None = None, dhw_series: dict[str, Any] | None = None, dhw_meter_missing: bool = False) -> list[dict[str, Any]]:
        samples = await self._async_load_training_samples()
        if dhw_meter_missing:
            samples = [self._repair_no_dhw_meter_training_values(existing, dhw_meter_missing=True) for existing in samples]
            sample = self._repair_no_dhw_meter_training_values(sample, dhw_meter_missing=True)
        date_key = sample.get("date")
        if not date_key:
            return samples
        samples = self._finalize_previous_training_samples(samples, str(date_key), total_series or {}, heating_series or {}, dhw_series or {}, dhw_meter_missing=dhw_meter_missing)
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

    async def _async_get_daily_history_stats(self, entity_id: str | None, days: int = 30, include_zero: bool = False) -> dict[str, Any]:
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
        series = {day.isoformat(): round(v, 3) for day, v in sorted(daily_max.items()) if include_zero or v > 0}
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
        runtime_options = self.hass.data.get(DOMAIN, {}).get(self.entry.entry_id, {}).get("runtime_options", {})
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
        # Es gibt keinen separaten Warmwasser-Energiezähler mehr in der UI.
        # Warmwasser wird immer aus Gesamtverbrauch minus Heizverbrauch berechnet.
        dhw_entity = None
        dhw_mode = DHW_DAILY_ENERGY_MODE_NONE
        dhw_meter_missing = True
        heating_threshold = float(config.get(CONF_HEATING_THRESHOLD_TEMP, 17.0) or 17.0)
        heating_curve_simulation_enabled = bool(config.get(CONF_HEATING_CURVE_SIMULATION_ENABLED, True))
        heating_curve_flow_warm = float(config.get(CONF_HEATING_CURVE_FLOW_WARM, DEFAULT_HEATING_CURVE_FLOW_WARM) or DEFAULT_HEATING_CURVE_FLOW_WARM)
        heating_curve_flow_mid = float(config.get(CONF_HEATING_CURVE_FLOW_MID, DEFAULT_HEATING_CURVE_FLOW_MID) or DEFAULT_HEATING_CURVE_FLOW_MID)
        heating_curve_flow_cold = float(config.get(CONF_HEATING_CURVE_FLOW_COLD, DEFAULT_HEATING_CURVE_FLOW_COLD) or DEFAULT_HEATING_CURVE_FLOW_COLD)
        heating_curve_saving_percent_per_c = float(config.get(CONF_HEATING_CURVE_SAVING_PERCENT_PER_C, DEFAULT_HEATING_CURVE_SAVING_PERCENT_PER_C) or DEFAULT_HEATING_CURVE_SAVING_PERCENT_PER_C)
        dhw_target_temp_c = float(config.get(CONF_DHW_TARGET_TEMP, DEFAULT_DHW_TARGET_TEMP) or DEFAULT_DHW_TARGET_TEMP)
        dhw_tank_volume_l = float(config.get(CONF_DHW_TANK_VOLUME_L, DEFAULT_DHW_TANK_VOLUME_L) or DEFAULT_DHW_TANK_VOLUME_L)
        dhw_liters_per_person = float(config.get(CONF_DHW_LITERS_PER_PERSON, DEFAULT_DHW_LITERS_PER_PERSON) or DEFAULT_DHW_LITERS_PER_PERSON)
        heating_curve_delta_c = float(runtime_options.get(RUNTIME_HEATING_CURVE_DELTA_C, 0.0) or 0.0)
        dhw_target_delta_c = float(runtime_options.get(RUNTIME_DHW_TARGET_DELTA_C, 0.0) or 0.0)
        today_so_far_kwh = _energy_kwh(self.hass, daily_entity)
        today_heating_so_far_kwh = _energy_kwh(self.hass, heating_entity)
        if dhw_meter_missing:
            if today_so_far_kwh is not None:
                today_dhw_so_far_kwh = max(float(today_so_far_kwh) - float(today_heating_so_far_kwh or 0.0), 0.0)
            else:
                today_dhw_so_far_kwh = None
        else:
            today_dhw_so_far_kwh = _energy_kwh(self.hass, dhw_entity)

        history_stats = await self._async_get_daily_history_stats(daily_entity)
        heating_history = await self._async_get_daily_history_stats(heating_entity, include_zero=True)
        dhw_history = {"available": False, "days": 0, "values": [], "avg_7": None, "avg_14": None, "avg_30": None, "basis": None, "series": {}} if dhw_meter_missing else await self._async_get_daily_history_stats(dhw_entity, include_zero=True)
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
        if dhw_meter_missing:
            if heating_baseline is None:
                ref = temp_history.get("avg_7") if temp_history.get("avg_7") is not None else today_avg_temp
                if ref is not None and ref >= heating_threshold:
                    heating_baseline = 0.0
                    split_source = "no_dhw_meter_total_as_dhw"
                else:
                    share = max(0.0, min(0.75, _heating_degree_factor(ref, heating_threshold) / 2.8))
                    heating_baseline = round(float(baseline_kwh) * share, 2)
                    split_source = "estimated_no_dhw_meter_total_minus_heating"
            dhw_baseline = round(max(float(baseline_kwh) - float(heating_baseline or 0.0), 0.0), 2)
            if split_source not in ("no_dhw_meter_total_as_dhw", "estimated_no_dhw_meter_total_minus_heating"):
                split_source = "no_dhw_meter_total_minus_heating"
        elif heating_baseline is not None and dhw_baseline is None:
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
        heating_curve_simulation = _build_heating_curve_simulation(
            enabled=heating_curve_simulation_enabled,
            avg_temp_today=today_avg_temp,
            avg_temp_tomorrow=tomorrow_avg_temp,
            avg_temp_day_after=day_after_avg_temp,
            heating_threshold=heating_threshold,
            flow_cold=heating_curve_flow_cold,
            flow_mid=heating_curve_flow_mid,
            flow_warm=heating_curve_flow_warm,
            saving_percent_per_c=heating_curve_saving_percent_per_c,
            heating_curve_delta_c=heating_curve_delta_c,
            dhw_target_delta_c=dhw_target_delta_c,
            dhw_target_temp_c=dhw_target_temp_c,
            dhw_tank_volume_l=dhw_tank_volume_l,
            dhw_liters_per_person=dhw_liters_per_person,
            today_rule=today_rule_calc,
            tomorrow_rule=tomorrow_rule_calc,
            day_after_rule=day_after_rule_calc,
            today_persons=current_occ.persons,
            tomorrow_persons=tomorrow_occ.persons,
            day_after_persons=day_after_occ.persons,
        )
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
            "dhw_meter_mode": dhw_mode,
        }
        training_samples = await self._async_store_training_sample(training_sample, history_stats.get("series") or {}, heating_history.get("series") or {}, dhw_history.get("series") or {}, dhw_meter_missing=dhw_meter_missing)
        completed_samples = completed_training_samples(training_samples)
        pending_samples = [s for s in training_samples if not s.get("completed")]
        ml_status = await self._async_train_ml(training_samples)
        forecast_quality = _forecast_quality_metrics(completed_samples)
        model_selection = _select_model_strategy(ml_status, forecast_quality)
        learning_analysis = _learning_analysis_series(training_samples, heating_curve, forecast_quality, ml_status, model_selection)

        def _ml_value(day_offset: int, avg_temp: float | None, occ: OccupancyInfo, rule_value: float) -> tuple[float, str, dict[str, Any] | None]:
            if not ml_status.get("active") or not model_selection.get("ml_allowed"):
                return rule_value, model_selection.get("mode") or "Regelmodell", None
            day = dt_util.now().date() + timedelta(days=day_offset)
            features = build_ml_features(avg_temperature_c=avg_temp, persons=occ.persons, occupied_units=occ.occupied_units, occupied_area_sqm=occ.occupied_area_sqm or occ.total_area_sqm, month=day.month, weekday=day.weekday(), heating_basis_kwh=heating_baseline, dhw_basis_kwh=dhw_baseline)
            prediction = self._ml_model.predict(features)
            if _plausible_forecast(prediction.value, rule_value):
                return round(float(prediction.value), 2), "ML-Modell", {"features": features, **asdict(prediction)}
            return rule_value, "ML + Fallback", {"features": features, **asdict(prediction), "fallback_rule_kwh": rule_value}

        tomorrow, tomorrow_model, tomorrow_ml = _ml_value(1, tomorrow_avg_temp, tomorrow_occ, rule_tomorrow)
        day_after, day_after_model, day_after_ml = _ml_value(2, day_after_avg_temp, day_after_occ, rule_day_after)
        if not model_selection.get("ml_allowed"):
            forecast_model = model_selection.get("mode") or "Regelmodell"
        else:
            forecast_model = "ML-Modell" if tomorrow_model == "ML-Modell" and day_after_model == "ML-Modell" else "ML + Fallback"

        completed_count = len(completed_samples)
        confidence_label, confidence_stage = ("Unzureichend", 1) if completed_count < 7 else ("Schwach", 2) if completed_count < 15 else ("Ausreichend", 3) if completed_count < 30 else ("Gut", 4) if completed_count < 90 else ("Sehr gut", 5)
        confidence = min(90, 30 + (25 if source == "daily_history" else 15) + (10 if temp_history.get("available") else 5 if current_temp is not None else 0) + (10 if forecast_temps.get(1) is not None else 0) + (5 if forecast_temps.get(2) is not None else 0) + (5 if units else 0) + (5 if heating_entity or dhw_entity else 0) + (5 if ml_status.get("active") else 0))
        reason_summary = " + ".join([forecast_model, _label_source(source), "Wetter", "Personen" if current_occ.total_units > 0 else "Heizgrenze", "Heizkurve" if heating_curve.get("active") else "Heizgrenze"])
        reason_text = f"v1.1.6 | Prognosemodell: {forecast_model} | ML-Status: {ml_status.get('status')} | Datenbasis: {_label_source(source)} | Basis {baseline_kwh:.2f} kWh | Morgen Regelmodell {rule_tomorrow:.2f} kWh, final {tomorrow:.2f} kWh | Übermorgen Regelmodell {rule_day_after:.2f} kWh, final {day_after:.2f} kWh"

        reason_structured = {
            "version": "Basis v1.1.6", "source": source, "split_source": split_source, "forecast_model": forecast_model, "ml_status": ml_status, "forecast_quality": forecast_quality, "model_selection": model_selection, "learning_analysis": learning_analysis, "heating_curve_simulation": heating_curve_simulation,
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
            "heating_curve": heating_curve, "heating_curve_simulation": heating_curve_simulation, "ml_status": ml_status, "forecast_model": forecast_model, "forecast_quality": forecast_quality, "model_selection": model_selection, "learning_analysis": learning_analysis, "heating_curve_simulation": heating_curve_simulation,
        }


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    coordinator = HeatPumpForecastCoordinator(hass, entry)
    hass.data.setdefault(DOMAIN, {}).setdefault(entry.entry_id, {})["coordinator"] = coordinator
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
        HeatPumpForecastEvaluationSensor(coordinator, entry),
        HeatPumpForecastQualityRatingSensor(coordinator, entry),
        HeatPumpForecastAccuracySensor(coordinator, entry),
        HeatPumpMLDiagnosticsSensor(coordinator, entry),
        HeatPumpMLQualitySensor(coordinator, entry),
        HeatPumpModelSelectionSensor(coordinator, entry),
        HeatPumpLearningAnalysisSensor(coordinator, entry),
        HeatPumpHeatingCurveAnalysisSensor(coordinator, entry),
        HeatPumpHeatingCurveSimulationSensor(coordinator, entry),
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
        training = structured.get("training") or {}
        ml_status = structured.get("ml_status") or {}
        today = structured.get("today") or {}
        tomorrow = structured.get("tomorrow") or {}
        day_after = structured.get("day_after_tomorrow") or {}
        # Keep this diagnostic sensor small. Home Assistant recorder drops
        # attributes above 16 KiB, so do not expose full training samples,
        # ML feature dictionaries or full history arrays here. Detailed
        # diagnostics are available in the dedicated diagnostic sensors/files.
        return {
            "Kurzfassung": self.coordinator.data.get("reason_summary"),
            "Details": self.coordinator.data.get("reason"),
            "Version": structured.get("version"),
            "Datenbasis": _label_source(structured.get("source")),
            "Aufteilung": _label_split_source(structured.get("split_source")),
            "Prognosemodell": structured.get("forecast_model"),
            "ML_Status": ml_status.get("status") if isinstance(ml_status, dict) else ml_status,
            "ML_Aktiv": ml_status.get("active") if isinstance(ml_status, dict) else None,
            "Heute_bisher_kWh": structured.get("today_so_far_kwh"),
            "Heizung_bisher_kWh": structured.get("today_heating_so_far_kwh"),
            "Warmwasser_bisher_kWh": structured.get("today_dhw_so_far_kwh"),
            "Heute_Temperatur_C": today.get("avg_temperature_c") if isinstance(today, dict) else None,
            "Morgen_Prognose_kWh": self.coordinator.data.get("tomorrow_kwh"),
            "Morgen_Temperatur_C": tomorrow.get("avg_temperature_c") if isinstance(tomorrow, dict) else None,
            "Übermorgen_Prognose_kWh": self.coordinator.data.get("day_after_tomorrow_kwh"),
            "Übermorgen_Temperatur_C": day_after.get("avg_temperature_c") if isinstance(day_after, dict) else None,
            "Trainingsdaten_gesamt": training.get("sample_count") if isinstance(training, dict) else None,
            "Trainingsdaten_abgeschlossen": training.get("completed_sample_count") if isinstance(training, dict) else None,
            "Trainingsdaten_offen": training.get("pending_sample_count") if isinstance(training, dict) else None,
            "Heizgrenze_C": structured.get("heating_threshold_c"),
            "Basis_kWh": structured.get("baseline_kwh"),
            "Heizbasis_kWh": structured.get("heating_baseline_kwh"),
            "Warmwasserbasis_kWh": structured.get("dhw_baseline_kwh"),
        }


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


class HeatPumpForecastEvaluationSensor(HeatPumpStringSensor):
    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry, "forecast_evaluation", "Prognosebewertung", "mdi:clipboard-check-outline")

    @property
    def native_value(self) -> str:
        quality = self.coordinator.data.get("forecast_quality") or {}
        latest = quality.get("latest") or {}
        if not latest:
            return "Noch keine Bewertung"
        err = latest.get("abs_error_kwh")
        pct = latest.get("error_percent")
        return f"{err} kWh / {pct}%" if err is not None and pct is not None else "Noch keine Bewertung"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return self.coordinator.data.get("forecast_quality") or {}


class HeatPumpForecastQualityRatingSensor(HeatPumpStringSensor):
    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry, "forecast_quality_rating", "Prognosequalität", "mdi:star-check-outline")

    @property
    def native_value(self) -> str:
        quality = self.coordinator.data.get("forecast_quality") or {}
        return quality.get("quality_label") or "Keine Bewertung"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        quality = self.coordinator.data.get("forecast_quality") or {}
        return {
            "Bewertung": quality.get("quality_label"),
            "Beschreibung": quality.get("quality_description"),
            "Genauigkeit_Prozent": quality.get("accuracy_percent"),
            "MAPE_Prozent": quality.get("mape_used_percent"),
            "Aktives_Fenster": quality.get("active_window"),
            "Zusammenfassung": quality.get("summary"),
            "Letzter_Vergleich": quality.get("latest"),
            "Letzte_7_Tage": quality.get("last_7"),
            "Letzte_30_Tage": quality.get("last_30"),
            "Letzte_90_Tage": quality.get("last_90"),
            "Hinweis": "Bewertung anhand abgeschlossener Tagesvergleiche. Niedrige MAPE bedeutet hohe Prognosequalität.",
        }


class HeatPumpForecastAccuracySensor(HeatPumpBaseSensor):
    _attr_native_unit_of_measurement = "%"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:percent-outline"
    _attr_name = "Prognosegenauigkeit"

    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry)
        self._attr_translation_key = "forecast_accuracy"
        self._attr_unique_id = f"{entry.entry_id}_forecast_accuracy"

    @property
    def native_value(self) -> float | None:
        quality = self.coordinator.data.get("forecast_quality") or {}
        return quality.get("accuracy_percent")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        quality = self.coordinator.data.get("forecast_quality") or {}
        active_window = quality.get("active_window") or "last_30"
        active_metrics = quality.get(active_window) or quality.get("all") or {}
        return {
            "quality_rating": quality.get("quality_label"),
            "sample_count": quality.get("sample_count"),
            "mape_percent": quality.get("mape_used_percent"),
            "mae_kwh": active_metrics.get("mae_kwh"),
            "rmse_kwh": active_metrics.get("rmse_kwh"),
            "active_window": active_window,
            "summary": quality.get("summary"),
        }


class HeatPumpMLDiagnosticsSensor(HeatPumpStringSensor):
    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry, "ml_diagnostics", "ML-Diagnose", "mdi:brain")

    @property
    def native_value(self) -> str:
        status = self.coordinator.data.get("ml_status") or {}
        return status.get("status") or "Wartet auf Trainingsdaten"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return {
            "ML_Status": self.coordinator.data.get("ml_status"),
            "Modellwahl": self.coordinator.data.get("model_selection"),
            "Prognosemodell": self.coordinator.data.get("forecast_model"),
            "Trainingsdaten": {
                "Gesammelt": self.coordinator.data.get("training_sample_count"),
                "Abgeschlossen": self.coordinator.data.get("training_completed_sample_count"),
                "Offen": self.coordinator.data.get("training_pending_sample_count"),
            },
            "Speicher": self.coordinator.data.get("storage_status"),
        }


class HeatPumpMLQualitySensor(HeatPumpStringSensor):
    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry, "ml_quality", "ML-Qualität", "mdi:chart-timeline-variant")

    @property
    def native_value(self) -> str:
        quality = self.coordinator.data.get("forecast_quality") or {}
        return quality.get("quality_label") or "Keine Bewertung"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return self.coordinator.data.get("forecast_quality") or {}


class HeatPumpModelSelectionSensor(HeatPumpStringSensor):
    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry, "model_selection", "Automatische Modellwahl", "mdi:source-branch-sync")

    @property
    def native_value(self) -> str:
        selection = self.coordinator.data.get("model_selection") or {}
        return selection.get("mode") or "Regelmodell"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return self.coordinator.data.get("model_selection") or {}


class HeatPumpLearningAnalysisSensor(HeatPumpStringSensor):
    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry, "learning_analysis", "Lernanalyse", "mdi:chart-areaspline")

    @property
    def native_value(self) -> str:
        analysis = self.coordinator.data.get("learning_analysis") or {}
        completed = analysis.get("completed_days") or 0
        return f"{completed} Tage" if completed else "Wird aufgebaut"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return self.coordinator.data.get("learning_analysis") or {}


class HeatPumpHeatingCurveAnalysisSensor(HeatPumpStringSensor):
    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry, "heating_curve_analysis", "Heizkurvenanalyse Verlauf", "mdi:chart-scatter-plot")

    @property
    def native_value(self) -> str:
        curve = self.coordinator.data.get("heating_curve") or {}
        return curve.get("status") or "Wird aufgebaut"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        analysis = self.coordinator.data.get("learning_analysis") or {}
        return {
            "heating_curve_series": analysis.get("heating_curve_series"),
            "heating_curve": self.coordinator.data.get("heating_curve"),
            "Hinweis": "Temperatur/Heizverbrauch-Punkte für grafische Heizkurvenanalyse.",
        }


class HeatPumpHeatingCurveSimulationSensor(HeatPumpStringSensor):
    def __init__(self, coordinator: HeatPumpForecastCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry, "energy_simulation", "Energie-Simulation", "mdi:home-thermometer-outline")

    @property
    def native_value(self) -> float | str | None:
        simulation = self.coordinator.data.get("heating_curve_simulation") or {}
        summary = simulation.get("summary") or {}
        # Sensor state is the selected tomorrow energy effect in kWh.
        # Negative values mean savings, positive values mean additional consumption.
        return summary.get("selected_effect_kwh") or 0.0

    @property
    def native_unit_of_measurement(self) -> str | None:
        return UnitOfEnergy.KILO_WATT_HOUR

    @property
    def device_class(self) -> SensorDeviceClass | None:
        return SensorDeviceClass.ENERGY

    @property
    def state_class(self) -> SensorStateClass | None:
        return SensorStateClass.MEASUREMENT

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        simulation = self.coordinator.data.get("heating_curve_simulation") or {}
        return {
            **simulation,
            "Hinweis": "Interaktive Simulation: Heizkurve wirkt nur auf Heizanteil; Warmwasser-Solltemperatur wirkt nur auf Warmwasseranteil.",
        }


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
