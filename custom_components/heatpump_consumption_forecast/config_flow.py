"""Config flow for Heat Pump Consumption Forecast."""
from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    BUILDING_APARTMENT,
    BUILDING_OTHER,
    BUILDING_RESIDENTIAL,
    BUILDING_VACATION,
    CONF_BUILDING_TYPE,
    CONF_FIXED_PERSONS,
    CONF_HAS_OCCUPANCY_CALENDARS,
    CONF_HEATPUMP_DAILY_ENERGY_SENSOR,
    CONF_HEATING_DAILY_ENERGY_SENSOR,
    CONF_DHW_DAILY_ENERGY_SENSOR,
    CONF_DHW_DAILY_ENERGY_MODE,
    DHW_DAILY_ENERGY_MODE_NONE,
    CONF_DHW_TARGET_TEMP,
    CONF_DHW_TANK_VOLUME_L,
    CONF_DHW_LITERS_PER_PERSON,
    CONF_HEATING_THRESHOLD_TEMP,
    CONF_HEATING_CURVE_FLOW_WARM,
    CONF_HEATING_CURVE_FLOW_MID,
    CONF_HEATING_CURVE_FLOW_COLD,
    CONF_HEATING_CURVE_SAVING_PERCENT_PER_C,
    CONF_HEATING_CURVE_SIMULATION_ENABLED,
    DEFAULT_HEATING_CURVE_FLOW_WARM,
    DEFAULT_HEATING_CURVE_FLOW_MID,
    DEFAULT_HEATING_CURVE_FLOW_COLD,
    DEFAULT_HEATING_CURVE_SAVING_PERCENT_PER_C,
    DEFAULT_DHW_TARGET_TEMP,
    DEFAULT_DHW_TANK_VOLUME_L,
    DEFAULT_DHW_LITERS_PER_PERSON,
    CONF_HEATPUMP_TOTAL_ENERGY_SENSOR,
    CONF_OUTDOOR_TEMP_SENSOR,
    CONF_PERSON_MODEL,
    CONF_UNIT_COUNT,
    CONF_UNITS,
    CONF_WEATHER_ENTITY,
    DEFAULT_NAME,
    DOMAIN,
    PERSON_CALENDAR,
    PERSON_FIXED,
    PERSON_NONE,
)

_LOGGER = logging.getLogger(__name__)


def _find_candidates(
    hass: HomeAssistant,
    words: tuple[str, ...],
    domain: str | None = None,
) -> list[str]:
    """Find likely entity candidates by entity_id and friendly_name."""
    states = hass.states.async_all(domain) if domain else hass.states.async_all()
    candidates: list[str] = []

    for state in states:
        text = f"{state.entity_id} {state.attributes.get('friendly_name', '')}".lower()
        if any(word in text for word in words):
            candidates.append(state.entity_id)

    return sorted(candidates)


def _guess_default(candidates: list[str]) -> str | None:
    """Return first candidate or None."""
    return candidates[0] if candidates else None



class HeatPumpConsumptionForecastConfigFlow(
    config_entries.ConfigFlow,
    domain=DOMAIN,
):
    """Handle a config flow."""

    VERSION = 3

    def __init__(self) -> None:
        """Initialize flow."""
        self._data: dict[str, Any] = {}

    async def async_step_user(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """First step: choose profile."""
        await self.async_set_unique_id(DOMAIN)
        self._abort_if_unique_id_configured()

        if user_input is not None:
            self._data.update(user_input)
            return await self.async_step_units()

        schema = vol.Schema(
            {
                vol.Required(CONF_NAME, default=DEFAULT_NAME): str,
                vol.Required(
                    CONF_BUILDING_TYPE,
                    default=BUILDING_RESIDENTIAL,
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            BUILDING_RESIDENTIAL,
                            BUILDING_VACATION,
                            BUILDING_APARTMENT,
                            BUILDING_OTHER,
                        ],
                        translation_key="building_type",
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Required(CONF_UNIT_COUNT, default=1): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=1,
                        max=30,
                        step=1,
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Required(
                    CONF_PERSON_MODEL,
                    default=PERSON_NONE,
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            PERSON_NONE,
                            PERSON_FIXED,
                            PERSON_CALENDAR,
                        ],
                        translation_key="person_model",
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Required(
                    CONF_HAS_OCCUPANCY_CALENDARS,
                    default=False,
                ): selector.BooleanSelector(),
                vol.Optional(
                    CONF_FIXED_PERSONS,
                    default=2,
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0,
                        max=100,
                        step=1,
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
            }
        )

        return self.async_show_form(step_id="user", data_schema=schema)

    async def async_step_units(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Second step: configure dwelling units."""
        unit_count = int(self._data.get(CONF_UNIT_COUNT, 1) or 1)
        has_calendars = (
            bool(self._data.get(CONF_HAS_OCCUPANCY_CALENDARS))
            or self._data.get(CONF_PERSON_MODEL) == PERSON_CALENDAR
        )

        if user_input is not None:
            units: list[dict[str, Any]] = []

            for idx in range(1, unit_count + 1):
                unit: dict[str, Any] = {
                    "name": user_input.get(f"unit_{idx}_name") or f"WE {idx}",
                    "area_sqm": float(
                        user_input.get(f"unit_{idx}_area_sqm") or 0
                    ),
                    "fixed_persons": float(
                        user_input.get(f"unit_{idx}_fixed_persons") or 0
                    ),
                }

                if has_calendars:
                    unit["calendar_entity"] = user_input.get(f"unit_{idx}_calendar")

                units.append(unit)

            self._data[CONF_UNITS] = units
            return await self.async_step_sensors()

        schema_dict: dict[Any, Any] = {}
        calendar_candidates = _find_candidates(
            self.hass,
            (
                "wohnung",
                "we ",
                "apartment",
                "calendar",
                "kalender",
            ),
            "calendar",
        )

        for idx in range(1, unit_count + 1):
            schema_dict[
                vol.Required(f"unit_{idx}_name", default=f"WE {idx}")
            ] = str

            schema_dict[
                vol.Required(f"unit_{idx}_area_sqm", default=50)
            ] = selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0,
                    max=1000,
                    step=1,
                    unit_of_measurement="m²",
                    mode=selector.NumberSelectorMode.BOX,
                )
            )

            schema_dict[
                vol.Optional(f"unit_{idx}_fixed_persons", default=0)
            ] = selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0,
                    max=30,
                    step=1,
                    mode=selector.NumberSelectorMode.BOX,
                )
            )

            if has_calendars:
                schema_dict[
                    vol.Optional(
                        f"unit_{idx}_calendar",
                        default=(
                            calendar_candidates[idx - 1]
                            if idx - 1 < len(calendar_candidates)
                            else None
                        ),
                    )
                ] = selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["calendar"])
                )

        return self.async_show_form(
            step_id="units",
            data_schema=vol.Schema(schema_dict),
        )

    async def async_step_sensors(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Third step: select sensors."""
        if user_input is not None:
            self._data.update(user_input)
            # Kein separater Warmwasser-Energiezähler im UI:
            # Warmwasser wird immer aus Gesamtverbrauch minus Heizverbrauch berechnet.
            self._data[CONF_DHW_DAILY_ENERGY_MODE] = DHW_DAILY_ENERGY_MODE_NONE
            self._data[CONF_DHW_DAILY_ENERGY_SENSOR] = None
            return self.async_create_entry(
                title=self._data.get(CONF_NAME, DEFAULT_NAME),
                data=self._data,
            )

        daily_candidates = _find_candidates(
            self.hass,
            (
                "daily",
                "dayli",
                "today",
                "tag",
                "täglich",
                "taeglich",
                "heat_pump_energy_daily",
                "verbrauch_heatpump",
            ),
            "sensor",
        )

        total_candidates = _find_candidates(
            self.hass,
            (
                "accumulated",
                "total",
                "gesamt",
                "consumption accumulated",
                "compressor power consumption",
            ),
            "sensor",
        )

        temp_candidates = _find_candidates(
            self.hass,
            (
                "outdoor",
                "outside",
                "ambient",
                "außen",
                "aussen",
                "temperature",
            ),
            "sensor",
        )

        weather_candidates = [
            state.entity_id for state in self.hass.states.async_all("weather")
        ]

        heating_candidates = _find_candidates(
            self.hass,
            (
                "heizen",
                "heating",
                "heat demand",
                "wp heizen",
                "raumheizung",
                "space heating",
            ),
            "sensor",
        )


        schema = vol.Schema(
            {
                vol.Optional(
                    CONF_HEATPUMP_DAILY_ENERGY_SENSOR,
                    default=_guess_default(daily_candidates),
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["sensor"])
                ),
                vol.Optional(
                    CONF_HEATPUMP_TOTAL_ENERGY_SENSOR,
                    default=_guess_default(total_candidates),
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["sensor"])
                ),
                vol.Optional(
                    CONF_OUTDOOR_TEMP_SENSOR,
                    default=_guess_default(temp_candidates),
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["sensor"])
                ),
                vol.Optional(
                    CONF_WEATHER_ENTITY,
                    default=_guess_default(weather_candidates),
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["weather"])
                ),
                vol.Required(
                    CONF_HEATING_THRESHOLD_TEMP,
                    default=17.0,
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=-10,
                        max=30,
                        step=0.5,
                        unit_of_measurement="°C",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(
                    CONF_HEATING_CURVE_SIMULATION_ENABLED,
                    default=True,
                ): selector.BooleanSelector(),
                vol.Optional(
                    CONF_HEATING_CURVE_FLOW_WARM,
                    default=DEFAULT_HEATING_CURVE_FLOW_WARM,
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=10,
                        max=70,
                        step=0.1,
                        unit_of_measurement="°C",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(
                    CONF_HEATING_CURVE_FLOW_MID,
                    default=DEFAULT_HEATING_CURVE_FLOW_MID,
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=10,
                        max=80,
                        step=0.1,
                        unit_of_measurement="°C",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(
                    CONF_HEATING_CURVE_FLOW_COLD,
                    default=DEFAULT_HEATING_CURVE_FLOW_COLD,
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=10,
                        max=90,
                        step=0.1,
                        unit_of_measurement="°C",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(
                    CONF_HEATING_CURVE_SAVING_PERCENT_PER_C,
                    default=DEFAULT_HEATING_CURVE_SAVING_PERCENT_PER_C,
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0,
                        max=10,
                        step=0.05,
                        unit_of_measurement="%/°C",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(
                    CONF_DHW_TARGET_TEMP,
                    default=DEFAULT_DHW_TARGET_TEMP,
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=25,
                        max=75,
                        step=0.1,
                        unit_of_measurement="°C",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(
                    CONF_DHW_TANK_VOLUME_L,
                    default=DEFAULT_DHW_TANK_VOLUME_L,
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0,
                        max=2000,
                        step=10,
                        unit_of_measurement="L",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(
                    CONF_HEATING_DAILY_ENERGY_SENSOR,
                    default=_guess_default(heating_candidates),
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["sensor"])
                ),
            }
        )

        return self.async_show_form(step_id="sensors", data_schema=schema)

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Create the options flow."""
        return HeatPumpConsumptionForecastOptionsFlow()


class HeatPumpConsumptionForecastOptionsFlow(config_entries.OptionsFlow):
    """Options flow for later changes."""

    async def async_step_init(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Manage options."""
        current = {
            **self.config_entry.data,
            **self.config_entry.options,
        }

        if user_input is not None:
            self._options_data = dict(user_input)
            # Kein separater Warmwasser-Energiezähler im UI:
            # Warmwasser wird immer aus Gesamtverbrauch minus Heizverbrauch berechnet.
            self._options_data[CONF_DHW_DAILY_ENERGY_MODE] = DHW_DAILY_ENERGY_MODE_NONE
            self._options_data[CONF_DHW_DAILY_ENERGY_SENSOR] = None
            return self.async_create_entry(title="", data=self._options_data)

        schema = vol.Schema(
            {
                vol.Optional(
                    CONF_HEATPUMP_DAILY_ENERGY_SENSOR,
                    default=current.get(CONF_HEATPUMP_DAILY_ENERGY_SENSOR),
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["sensor"])
                ),
                vol.Optional(
                    CONF_HEATPUMP_TOTAL_ENERGY_SENSOR,
                    default=current.get(CONF_HEATPUMP_TOTAL_ENERGY_SENSOR),
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["sensor"])
                ),
                vol.Optional(
                    CONF_OUTDOOR_TEMP_SENSOR,
                    default=current.get(CONF_OUTDOOR_TEMP_SENSOR),
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["sensor"])
                ),
                vol.Optional(
                    CONF_WEATHER_ENTITY,
                    default=current.get(CONF_WEATHER_ENTITY),
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["weather"])
                ),
                vol.Required(
                    CONF_HEATING_THRESHOLD_TEMP,
                    default=current.get(CONF_HEATING_THRESHOLD_TEMP, 17.0),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=-10,
                        max=30,
                        step=0.5,
                        unit_of_measurement="°C",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(
                    CONF_HEATING_CURVE_SIMULATION_ENABLED,
                    default=current.get(CONF_HEATING_CURVE_SIMULATION_ENABLED, True),
                ): selector.BooleanSelector(),
                vol.Optional(
                    CONF_HEATING_CURVE_FLOW_WARM,
                    default=current.get(CONF_HEATING_CURVE_FLOW_WARM, DEFAULT_HEATING_CURVE_FLOW_WARM),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=10,
                        max=70,
                        step=0.1,
                        unit_of_measurement="°C",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(
                    CONF_HEATING_CURVE_FLOW_MID,
                    default=current.get(CONF_HEATING_CURVE_FLOW_MID, DEFAULT_HEATING_CURVE_FLOW_MID),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=10,
                        max=80,
                        step=0.1,
                        unit_of_measurement="°C",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(
                    CONF_HEATING_CURVE_FLOW_COLD,
                    default=current.get(CONF_HEATING_CURVE_FLOW_COLD, DEFAULT_HEATING_CURVE_FLOW_COLD),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=10,
                        max=90,
                        step=0.1,
                        unit_of_measurement="°C",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(
                    CONF_HEATING_CURVE_SAVING_PERCENT_PER_C,
                    default=current.get(CONF_HEATING_CURVE_SAVING_PERCENT_PER_C, DEFAULT_HEATING_CURVE_SAVING_PERCENT_PER_C),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0,
                        max=10,
                        step=0.05,
                        unit_of_measurement="%/°C",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(
                    CONF_DHW_TARGET_TEMP,
                    default=current.get(CONF_DHW_TARGET_TEMP, DEFAULT_DHW_TARGET_TEMP),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=25,
                        max=75,
                        step=0.1,
                        unit_of_measurement="°C",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(
                    CONF_DHW_TANK_VOLUME_L,
                    default=current.get(CONF_DHW_TANK_VOLUME_L, DEFAULT_DHW_TANK_VOLUME_L),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0,
                        max=2000,
                        step=10,
                        unit_of_measurement="L",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(
                    CONF_HEATING_DAILY_ENERGY_SENSOR,
                    default=current.get(CONF_HEATING_DAILY_ENERGY_SENSOR),
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["sensor"])
                ),
            }
        )

        return self.async_show_form(step_id="init", data_schema=schema)
