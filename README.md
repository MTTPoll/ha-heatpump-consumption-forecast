# Heat Pump Consumption Forecast (ML)

[![HACS](https://img.shields.io/badge/HACS-Custom-blue.svg)](https://hacs.xyz/)
[![Home Assistant](https://img.shields.io/badge/Home%20Assistant-2026.3+-41BDF5.svg)](https://www.home-assistant.io/)
[![Version](https://img.shields.io/badge/version-v1.0.0-green.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

---

# 🇩🇪 Deutsch

## Übersicht

Heat Pump Consumption Forecast erstellt eine vollständig lokale Verbrauchsprognose für Wärmepumpen in Home Assistant.

Die Integration kombiniert:

- Historische Verbrauchsdaten
- Wetterdaten
- Außentemperatur
- Heizgrenze
- Personenmodell
- Kalenderbasierte Belegung
- Wohnflächen
- Heizkurvenanalyse
- Lokales Machine Learning

Es werden keine Cloud-Dienste verwendet.

---

## Highlights

### Verbrauchsprognosen

- Verbrauch morgen
- Verbrauch übermorgen
- Rest-Tagesprognose

### Wetterintegration

- Wettervorhersage
- Außentemperatur
- Heizgrenze

### Belegungsmodell

- Wohnhäuser
- Ferienwohnungen
- Mehrfamilienhäuser
- Mehrere Wohneinheiten
- Individuelle Wohnflächen
- Kalenderbasierte Belegung
- Personenmodell

### Verbrauchsanalyse

- Heizverbrauch
- Warmwasserverbrauch
- Automatische Verbrauchsaufteilung
- Historische Lernfunktion

### Machine Learning

- Lokales Ähnlichkeitsmodell
- Selbstlernend
- Keine externen ML-Abhängigkeiten
- Raspberry Pi kompatibel
- Persistente Datenspeicherung
- Regelmodell als permanenter Fallback

---

## ⚠️ Benötigte Sensoren

### Mindestanforderungen

| Sensor | Beschreibung |
|----------|-------------|
| Tagesverbrauch Wärmepumpe | Verbrauch seit 00:00 Uhr in kWh |
| Außentemperatur | Aktuelle Außentemperatur |
| Wetter-Entity | Wettervorhersage |

### Empfohlen

| Sensor | Nutzen |
|----------|---------|
| Heizverbrauch täglich | Genauere Heizanalyse |
| Warmwasserverbrauch täglich | Genauere Warmwasserprognose |
| Gesamtverbrauch Wärmepumpe | Zusätzliche Plausibilitätsprüfung |

---

## Wichtig: Tagesverbrauchszähler

Die Integration funktioniert am besten mit einem Sensor, der täglich um Mitternacht zurückgesetzt wird.

Beispiel:

```yaml
sensor.heatpump_daily_energy
```

Wenn nur ein Gesamtzähler vorhanden ist:

```yaml
utility_meter:
  heatpump_daily_energy:
    source: sensor.heatpump_total_energy
    cycle: daily
```

---

## ML-Logik

### 0–29 abgeschlossene Tagesdatensätze

- Regelmodell aktiv
- ML sammelt Daten

### 30–89 abgeschlossene Tagesdatensätze

- ML wird aktiv
- Lokales Modell wird erzeugt

### 90+ abgeschlossene Tagesdatensätze

- ML wird optimiert

---

## Sicherheitslogik

Machine Learning ist niemals die einzige Prognosequelle.

Automatischer Rückfall auf das Regelmodell bei:

- Zu wenig Trainingsdaten
- Fehlendem Modell
- Modellfehlern
- Unplausiblen Prognosen
- Fehlenden Eingangsdaten

---

## Persistente Dateien

```text
/config/.storage/heatpump_consumption_forecast/training_data.json
/config/.storage/heatpump_consumption_forecast/heating_curve.json
/config/.storage/heatpump_consumption_forecast/model.pkl
```

---

## Sensoren

### Prognose

- Verbrauch morgen
- Verbrauch übermorgen
- Rest-Tagesprognose

### Qualität

- Prognosegüte
- Datenqualität

### Training

- Gesammelte Tagesdaten
- Trainingsstatus

### Machine Learning

- ML-Status
- Prognosemodell

### Heizkurve

- Heizkurvenstatus
- Erlernte Heizkurve

### Diagnose

- Speicherstatus
- Lernanalyse
- Prognosefehleranalyse

---

## Installation über HACS

Repository hinzufügen:

```text
https://github.com/MTTPoll/ha-heatpump-consumption-forecast
```

Kategorie:

```text
Integration
```

Danach Home Assistant neu starten.

---

# 🇬🇧 English

## Overview

Heat Pump Consumption Forecast provides a fully local heat pump consumption forecast for Home Assistant.

The integration combines:

- Historical energy consumption
- Weather forecast data
- Outdoor temperature
- Heating threshold
- Occupancy model
- Calendar-based occupancy
- Living area
- Heating curve analysis
- Local machine learning

No cloud services are required.

---

## Highlights

### Forecasts

- Tomorrow forecast
- Day-after-tomorrow forecast
- Remaining forecast for today

### Weather Integration

- Weather forecast
- Outdoor temperature
- Heating threshold

### Occupancy Model

- Residential homes
- Holiday homes
- Apartment buildings
- Multiple dwelling units
- Individual living areas
- Calendar occupancy
- Person model

### Consumption Analysis

- Heating energy
- Domestic hot water energy
- Automatic consumption split
- Historical learning

### Machine Learning

- Local similarity model
- Self-learning
- No external ML dependencies
- Raspberry Pi friendly
- Persistent model storage
- Permanent rule-model fallback

---

## ⚠️ Required Sensors

### Minimum Requirements

| Sensor | Description |
|----------|-------------|
| Daily Heat Pump Energy Sensor | Daily energy consumption in kWh |
| Outdoor Temperature Sensor | Current outdoor temperature |
| Weather Entity | Weather forecast |

### Recommended

| Sensor | Benefit |
|----------|---------|
| Daily Heating Energy | Better heating analysis |
| Daily DHW Energy | Better hot water prediction |
| Total Heat Pump Energy Meter | Additional validation |

---

## Important: Daily Energy Sensor

The integration works best with a sensor that resets daily at midnight.

Example:

```yaml
sensor.heatpump_daily_energy
```

If only a total energy meter is available:

```yaml
utility_meter:
  heatpump_daily_energy:
    source: sensor.heatpump_total_energy
    cycle: daily
```

---

## Machine Learning Logic

### 0–29 completed daily samples

- Rule model active
- ML collects data

### 30–89 completed daily samples

- ML becomes active
- Local model is generated

### 90+ completed daily samples

- ML becomes optimized

---

## Safety Logic

Machine learning is never the only forecast source.

Automatic fallback to the rule model when:

- insufficient training data
- missing model
- model failures
- implausible forecasts
- missing input data

---

## Persistent Files

```text
/config/.storage/heatpump_consumption_forecast/training_data.json
/config/.storage/heatpump_consumption_forecast/heating_curve.json
/config/.storage/heatpump_consumption_forecast/model.pkl
```

---

## Community Feedback

This is the first public release.

Feedback regarding:

- Forecast accuracy
- ML behavior
- Heating curve quality
- Occupancy modeling
- Feature requests
- Bugs

is highly appreciated.

---

## License

MIT License
