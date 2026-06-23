"Constants for Heat Pump Consumption Forecast."
from __future__ import annotations

DOMAIN = "heatpump_consumption_forecast"
PLATFORMS = ["sensor", "button", "number"]

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
CONF_DHW_DAILY_ENERGY_MODE = "dhw_daily_energy_mode"
DHW_DAILY_ENERGY_MODE_SENSOR = "sensor"
DHW_DAILY_ENERGY_MODE_NONE = "none"
CONF_HEATING_THRESHOLD_TEMP = "heating_threshold_temp"


CONF_HEATING_CURVE_FLOW_WARM = "heating_curve_flow_warm"
CONF_HEATING_CURVE_FLOW_MID = "heating_curve_flow_mid"
CONF_HEATING_CURVE_FLOW_COLD = "heating_curve_flow_cold"
CONF_HEATING_CURVE_SAVING_PERCENT_PER_C = "heating_curve_saving_percent_per_c"
CONF_HEATING_CURVE_SIMULATION_ENABLED = "heating_curve_simulation_enabled"
CONF_DHW_TARGET_TEMP = "dhw_target_temp"
CONF_DHW_TANK_VOLUME_L = "dhw_tank_volume_l"
CONF_DHW_LITERS_PER_PERSON = "dhw_liters_per_person"

RUNTIME_HEATING_CURVE_DELTA_C = "heating_curve_delta_c"
RUNTIME_DHW_TARGET_DELTA_C = "dhw_target_delta_c"

HEATING_CURVE_OUTDOOR_WARM = 22.0
HEATING_CURVE_OUTDOOR_MID = 0.0
HEATING_CURVE_OUTDOOR_COLD = -22.0
DEFAULT_HEATING_CURVE_FLOW_WARM = 31.0
DEFAULT_HEATING_CURVE_FLOW_MID = 40.0
DEFAULT_HEATING_CURVE_FLOW_COLD = 48.0
DEFAULT_HEATING_CURVE_SAVING_PERCENT_PER_C = 2.25
DEFAULT_DHW_TARGET_TEMP = 50.0
DEFAULT_DHW_TANK_VOLUME_L = 200.0
DEFAULT_DHW_LITERS_PER_PERSON = 85.0
DEFAULT_COLD_WATER_TEMP = 10.0

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
