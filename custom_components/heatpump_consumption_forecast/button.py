"""Button platform for Heat Pump Consumption Forecast."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

from homeassistant.components.button import ButtonEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)
STORAGE_DIR_NAME = DOMAIN
TRAINING_DATA_FILE = "training_data.json"
HEATING_CURVE_FILE = "heating_curve.json"
MODEL_FILE = "model.pkl"


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    """Set up reset buttons."""
    async_add_entities([HeatPumpTrainingResetButton(hass, entry)])


class HeatPumpTrainingResetButton(ButtonEntity):
    """Reset all local learning data."""

    _attr_has_entity_name = True
    _attr_name = "Trainingsdaten zurücksetzen"
    _attr_translation_key = "reset_training_data"
    _attr_icon = "mdi:database-remove-outline"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize button."""
        self.hass = hass
        self.entry = entry
        self._attr_unique_id = f"{entry.entry_id}_reset_training_data"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": entry.title,
            "manufacturer": "MTTPoll",
            "model": "Lokale Wärmepumpen-Verbrauchsprognose",
        }

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return files that will be deleted."""
        storage_dir = Path(self.hass.config.path(".storage", STORAGE_DIR_NAME))
        return {
            "Hinweis": "Löscht Trainingsdaten, Heizkurve und lokales ML-Modell. Danach startet die Lernphase neu.",
            "Trainingsdaten_Datei": str(storage_dir / TRAINING_DATA_FILE),
            "Heizkurven_Datei": str(storage_dir / HEATING_CURVE_FILE),
            "ML_Modell_Datei": str(storage_dir / MODEL_FILE),
        }

    async def async_press(self) -> None:
        """Delete local learning files and reload the config entry."""
        storage_dir = Path(self.hass.config.path(".storage", STORAGE_DIR_NAME))
        paths = [
            storage_dir / TRAINING_DATA_FILE,
            storage_dir / HEATING_CURVE_FILE,
            storage_dir / MODEL_FILE,
        ]

        def _delete_files() -> list[str]:
            deleted: list[str] = []
            for path in paths:
                try:
                    if path.exists():
                        path.unlink()
                        deleted.append(str(path))
                except Exception as err:  # noqa: BLE001
                    _LOGGER.warning("Could not delete %s: %s", path, err)
            return deleted

        deleted = await self.hass.async_add_executor_job(_delete_files)
        _LOGGER.info("Reset Heat Pump Consumption Forecast training data: %s", deleted)
        await self.hass.config_entries.async_reload(self.entry.entry_id)
