# Heat Pump Consumption Forecast

[![HACS](https://img.shields.io/badge/HACS-Custom-blue.svg)](https://hacs.xyz/)
[![Home Assistant](https://img.shields.io/badge/Home%20Assistant-2026.3+-41BDF5.svg)](https://www.home-assistant.io/)
[![Version](https://img.shields.io/badge/version-v0.9.1-green.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

---

# 🇩🇪 Deutsch

## Übersicht

**Heat Pump Consumption Forecast** erstellt eine vollständig lokale Verbrauchsprognose für Wärmepumpen in Home Assistant.

Die Integration kombiniert:

- Historische Verbrauchsdaten
- Wetterdaten
- Außentemperatur
- Heizgrenze
- Personenanzahl
- Kalenderbasierte Belegung
- Wohnflächen
- Heizkurvenanalyse
- Lokales Machine Learning

Es werden **keine Cloud-Dienste** verwendet.

---

## Neu in v0.9.1

### Lokales Machine Learning

v0.9.1 verwendet ein leichtgewichtiges lokales Ähnlichkeitsmodell.

Vorteile:

- Keine externen ML-Abhängigkeiten
- Kein scikit-learn
- Raspberry-Pi-freundlich
- HACS-kompatibel
- Vollständig lokal
- Keine Cloud-Anbindung

---

## Hauptfunktionen

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
- Individuelle Flächen
- Kalenderbasierte Belegung
- Personenmodell

### Verbrauchsanalyse

- Heizverbrauch
- Warmwasserverbrauch
- Automatische Verbrauchsaufteilung
- Heizkurvenanalyse

### Machine Learning

- Lokales Ähnlichkeitsmodell
- Selbstlernend
- Persistente Trainingsdaten
- Automatische Modelloptimierung
- Regelmodell als permanenter Fallback

---

## ML-Logik

### 0–29 abgeschlossene Tagesdatensätze

- Regelmodell aktiv
- ML sammelt Daten
- ML-Status: `Wartet auf Trainingsdaten`
- Prognosemodell: `Regelmodell`

### 30–89 abgeschlossene Tagesdatensätze

- ML wird aktiv
- Lokales Modell wird erzeugt
- Prognosemodell:
  - `ML-Modell`
  - oder `ML + Fallback`

### 90+ abgeschlossene Tagesdatensätze

- ML-Status: `Optimiert`
- Modell wird automatisch verbessert
- Regelmodell bleibt immer verfügbar

---

## Sicherheitslogik

ML ist niemals die einzige Prognosequelle.

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
- Letzter Trainingsdatensatz
- Trainingsstatus

### Machine Learning

- ML-Status
- Prognosemodell

### Heizkurve

- Heizkurvenstatus
- Erlernte Heizkurve

### Diagnose

- Speicherstatus

---

## Installation über HACS

1. HACS öffnen
2. Benutzerdefinierte Repositories
3. Repository hinzufügen

```text
https://github.com/MTTPoll/ha-heatpump-consumption-forecast
```

4. Kategorie: **Integration**
5. Installation durchführen
6. Home Assistant neu starten

---

## Manuelle Installation

Ordner kopieren:

```text
custom_components/heatpump_consumption_forecast
```

nach:

```text
/config/custom_components/heatpump_consumption_forecast
```

Home Assistant anschließend neu starten.

---

# 🇬🇧 English

## Overview

**Heat Pump Consumption Forecast** provides a fully local heat pump energy consumption forecast for Home Assistant.

The integration combines:

- Historical energy consumption
- Weather forecast data
- Outdoor temperature
- Heating threshold
- Occupancy information
- Calendar data
- Living area
- Heating curve analysis
- Local machine learning

No cloud services are required.

---

## New in v0.9.1

### Local Machine Learning

Version 0.9.1 uses a lightweight local similarity model.

Benefits:

- No external ML dependencies
- No scikit-learn
- Raspberry Pi friendly
- HACS compatible
- Fully local
- No cloud services

---

## Features

### Consumption Forecasts

- Tomorrow forecast
- Day-after-tomorrow forecast
- Remaining forecast for today

### Weather Integration

- Weather forecast
- Outdoor temperature
- Heating threshold

### Occupancy Model

- Residential homes
- Holiday rentals
- Apartment buildings
- Multiple dwelling units
- Individual living areas
- Calendar-based occupancy
- Person model

### Consumption Analysis

- Heating energy
- Domestic hot water energy
- Automatic consumption split
- Heating curve analysis

### Machine Learning

- Local similarity model
- Self-learning
- Persistent training data
- Automatic optimization
- Rule-based fallback

---

## ML Logic

### 0–29 completed daily samples

- Rule-based model active
- ML collects data only
- ML Status: `Waiting for training data`
- Forecast Model: `Rule Model`

### 30–89 completed daily samples

- ML becomes active
- Local model is generated
- Forecast Model:
  - `ML Model`
  - or `ML + Fallback`

### 90+ completed daily samples

- ML Status: `Optimized`
- Model continuously improves
- Rule-based model always remains available

---

## Safety Logic

Machine learning is never the only forecast source.

Automatic fallback to the rule-based model when:

- Insufficient training data
- Missing model
- Model errors
- Implausible forecasts
- Missing input data

---

## Persistent Files

```text
/config/.storage/heatpump_consumption_forecast/training_data.json
/config/.storage/heatpump_consumption_forecast/heating_curve.json
/config/.storage/heatpump_consumption_forecast/model.pkl
```

---

## Sensors

### Forecast

- Consumption Tomorrow
- Consumption Day After Tomorrow
- Remaining Forecast Today

### Quality

- Forecast Quality
- Data Quality

### Training

- Collected Daily Data
- Latest Training Sample
- Training Status

### Machine Learning

- ML Status
- Forecast Model

### Heating Curve

- Heating Curve Status
- Learned Heating Curve

### Diagnostics

- Storage Status

---

## HACS Installation

Add the repository as a custom repository:

```text
https://github.com/MTTPoll/ha-heatpump-consumption-forecast
```

Category:

```text
Integration
```

Install via HACS and restart Home Assistant.

---

## Development Status

### v0.9.1

- Removed scikit-learn dependency
- Added lightweight local similarity model
- Added ML status sensor
- Added forecast model sensor
- Persistent model storage
- Permanent rule-based fallback

### v0.9.0

- First ML framework
- ML status sensor
- Forecast model sensor

### v0.8.2

- Persistent training data storage
- Persistent heating curve storage
- Storage status sensor

---

## License

MIT License
