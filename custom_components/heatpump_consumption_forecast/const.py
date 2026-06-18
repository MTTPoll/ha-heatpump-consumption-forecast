"Constants for Heat Pump Consumption Forecast."
from __future__ import annotations

DOMAIN = "heatpump_consumption_forecast"
PLATFORMS = ["sensor", "button"]

CONF_BUILDING_TYPE = "building_type"
CONF_PERSON_MODEL = "person_model"
CONF_FIXED_PERSONS = "fixed_persons"
CONF_UNIT_COUNT = "unit_count"
CONF_HAS_OCCUPANCY_CALENDARS = "has_occupancy_calendars"
CONF_UNITS = "units"

CONF_HEATPUMP_DAILY_ENERGY_SENSOR = "heatpump_daily_energy_sensor"
CONF_HEATPUMP_TOTAL_ENERGY_SENSOR = "heatpump_total_energy_sensor"
CONF_OUTDOOR_TEMP_SENSOR = "outdoor_temp_sensor"
CONF_OCCUPANCY_SENSOR = "occupancy_sensor"
CONF_WEATHER_ENTITY = "weather_entity"
CONF_HEATING_DAILY_ENERGY_SENSOR = "heating_daily_energy_sensor"
CONF_DHW_DAILY_ENERGY_SENSOR = "dhw_daily_energy_sensor"
CONF_HEATING_THRESHOLD_TEMP = "heating_threshold_temp"

BUILDING_RESIDENTIAL = "residential"
BUILDING_VACATION = "vacation"
BUILDING_APARTMENT = "apartment_house"
BUILDING_OTHER = "other"

PERSON_NONE = "none"
PERSON_FIXED = "fixed"
PERSON_SENSOR = "sensor"
PERSON_CALENDAR = "calendar"

DEFAULT_NAME = "Heat Pump Consumption Forecast"
SCAN_INTERVAL_MINUTES = 30
