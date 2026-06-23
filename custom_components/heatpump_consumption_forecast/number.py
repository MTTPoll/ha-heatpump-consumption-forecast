"""Number entities for Heat Pump Consumption Forecast runtime simulations."""
from __future__ import annotations

from typing import Any

from homeassistant.components.number import NumberEntity, NumberMode
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity

from .const import (
    DOMAIN,
    RUNTIME_DHW_TARGET_DELTA_C,
    RUNTIME_HEATING_CURVE_DELTA_C,
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up runtime simulation number entities."""
    async_add_entities(
        [
            HeatPumpRuntimeDeltaNumber(
                hass,
                entry,
                key=RUNTIME_HEATING_CURVE_DELTA_C,
                name="Heizkurven-Verschiebung",
                icon="mdi:home-thermometer-outline",
                min_value=-5.0,
                max_value=5.0,
                step=0.5,
            ),
            HeatPumpRuntimeDeltaNumber(
                hass,
                entry,
                key=RUNTIME_DHW_TARGET_DELTA_C,
                name="Warmwasser-Solltemperatur-Verschiebung",
                icon="mdi:water-thermometer-outline",
                min_value=-10.0,
                max_value=10.0,
                step=0.5,
            ),
        ]
    )


class HeatPumpRuntimeDeltaNumber(NumberEntity, RestoreEntity):
    """Runtime-adjustable simulation delta."""

    _attr_has_entity_name = True
    _attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
    _attr_mode = NumberMode.SLIDER

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        *,
        key: str,
        name: str,
        icon: str,
        min_value: float,
        max_value: float,
        step: float,
    ) -> None:
        """Initialize number entity."""
        self.hass = hass
        self.entry = entry
        self.key = key
        self._attr_name = name
        self._attr_unique_id = f"{entry.entry_id}_{key}"
        self._attr_icon = icon
        self._attr_native_min_value = min_value
        self._attr_native_max_value = max_value
        self._attr_native_step = step
        self._attr_native_value = 0.0
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": entry.title,
            "manufacturer": "MTTPoll",
            "model": "Lokale Wärmepumpen-Verbrauchsprognose",
        }

    async def async_added_to_hass(self) -> None:
        """Restore previous value and publish it to runtime options."""
        last_state = await self.async_get_last_state()
        if last_state is not None:
            try:
                self._attr_native_value = float(str(last_state.state).replace(",", "."))
            except (TypeError, ValueError):
                self._attr_native_value = 0.0
        self._store_runtime_value(float(self._attr_native_value or 0.0), refresh=False)

    async def async_set_native_value(self, value: float) -> None:
        """Set runtime simulation delta without reloading the integration."""
        value_f = round(float(value), 2)
        self._attr_native_value = value_f
        self._store_runtime_value(value_f, refresh=True)
        self.async_write_ha_state()

    def _store_runtime_value(self, value: float, *, refresh: bool) -> None:
        """Store value in integration runtime data and optionally refresh coordinator."""
        domain_data: dict[str, Any] = self.hass.data.setdefault(DOMAIN, {})
        entry_data: dict[str, Any] = domain_data.setdefault(self.entry.entry_id, {})
        runtime_options: dict[str, Any] = entry_data.setdefault("runtime_options", {})
        runtime_options[self.key] = value
        if refresh:
            coordinator = entry_data.get("coordinator")
            if coordinator is not None:
                self.hass.async_create_task(coordinator.async_request_refresh())
