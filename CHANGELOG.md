# Changelog

All notable changes to this project will be documented in this file.

---

## v1.1.6

### Changed

- Removed the Domestic Hot Water daily energy sensor from the configuration UI.
- Domestic Hot Water consumption is now always calculated internally:

```text
DHW = Total daily heat pump consumption - Daily heating consumption
```

- If heating consumption is 0, total daily consumption is treated as Domestic Hot Water consumption.
- Improved support for installations without a dedicated DHW energy meter.
- Added automatic reconstruction of DHW consumption from total and heating consumption.

### Improved

- Training data consistency significantly improved.
- Missing DHW values are automatically reconstructed where possible.
- Eliminated unnecessary `null` values in training samples.
- Completed samples now contain consistent `*_so_far` and `*_final` values.
- Forecast basis diagnostics now use reduced attributes to keep Recorder data compact.

### Fixed

- Fixed missing DHW training values during summer operation.
- Fixed training samples containing incomplete heating/DHW splits.
- Fixed missing `actual_dhw_kwh_so_far` values in completed samples.
- Fixed oversized sensor attributes causing Recorder warnings when attributes exceeded Home Assistant's 16 kB Recorder limit.
- Improved long-term Recorder database stability.

### Notes

For systems without a dedicated Domestic Hot Water energy meter:

```text
DHW = Total Daily Consumption - Heating Daily Consumption
```

This is now the default and recommended operating mode.

---

## v1.0.0

### 🎉 First Public Release

Heat Pump Consumption Forecast is now considered feature complete and ready for public testing and community feedback.

This release introduces a fully local heat pump consumption forecasting system for Home Assistant without any cloud dependency.

---

### Added

#### Forecasting

- Consumption forecast for tomorrow
- Consumption forecast for the day after tomorrow
- Remaining consumption forecast until midnight
- Weather-based consumption adjustments
- Heating threshold support

#### Occupancy Model

- Person-based consumption model
- Calendar-based occupancy support
- Multiple dwelling units
- Individual living area per dwelling unit
- Support for:
  - Residential homes
  - Holiday homes
  - Vacation apartments
  - Multi-family buildings

#### Consumption Analysis

- Heating consumption analysis
- Domestic hot water consumption analysis
- Automatic heating / hot water split
- Historical consumption learning

#### Heating Curve

- Automatic heating curve learning
- Heating curve diagnostics
- Heating curve status sensor
- Learned heating curve sensor

#### Local Machine Learning

- Lightweight local similarity-based ML model
- No external machine-learning dependencies
- No scikit-learn required
- Raspberry Pi friendly
- Fully local operation
- Persistent ML model storage

#### Diagnostics

- Forecast quality sensor
- Data quality sensor
- Training status sensor
- ML status sensor
- Forecast model sensor
- Storage status sensor
- Learning analysis sensor
- Forecast error analysis

#### Storage

Persistent storage support:

```text
/config/.storage/heatpump_consumption_forecast/training_data.json
/config/.storage/heatpump_consumption_forecast/heating_curve.json
/config/.storage/heatpump_consumption_forecast/model.pkl
```

#### Safety

- Rule-based forecast model always available
- Automatic ML fallback handling
- Protection against implausible forecasts
- Automatic recovery from model failures
- Automatic fallback when insufficient training data exists

---

### Machine Learning Logic

#### 0–29 completed daily samples

- Rule model active
- ML collects data only

#### 30–89 completed daily samples

- ML becomes active
- Local similarity model starts learning

#### 90+ completed daily samples

- ML status becomes optimized
- Model quality improves over time

---

### Community Release Notes

This release is intended to gather real-world feedback from different:

- Heat pump systems
- Building types
- Climate regions
- Occupancy patterns

Community feedback will help improve future versions.

---

## v0.9.1

- Removed scikit-learn dependency
- Added lightweight local similarity model
- Added ML status sensor
- Added forecast model sensor
- Added persistent model storage
- Added automatic ML fallback

---

## v0.9.0

- Introduced first ML framework
- Added ML status sensor
- Added forecast model sensor
- Prepared ML training workflow

---

## v0.8.2

- Persistent training data storage
- Persistent heating curve storage
- Storage status sensor

---

## v0.8.0

- Forecast quality evaluation
- Five-level forecast quality rating
- Heating curve preparation improvements

---

## v0.7.x

- Daily training storage
- Training diagnostics
- Remaining daily forecast
- Initial heating curve implementation
- Recorder integration improvements
