# Heat Pump Consumption Forecast (ML)

[![HACS](https://img.shields.io/badge/HACS-Custom-blue.svg)](https://hacs.xyz/)
[![Home Assistant](https://img.shields.io/badge/Home%20Assistant-2026.3+-41BDF5.svg)](https://www.home-assistant.io/)
[![Version](https://img.shields.io/badge/version-v1.1.6-green.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

A fully local heat pump consumption forecast for Home Assistant.

No cloud. No external services. No subscriptions.

Designed for residential homes, holiday homes, vacation apartments, and multi-family buildings.

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
- Wetterbasierte Prognose
- Heizgrenzen-Unterstützung

### Belegungsmodell

- Wohnhäuser
- Ferienwohnungen
- Ferienhäuser
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

### Lokales Machine Learning

- Lokales Ähnlichkeitsmodell
- Selbstlernend
- Keine externen ML-Abhängigkeiten
- Raspberry-Pi kompatibel
- Persistente Datenspeicherung
- Permanenter Regelmodell-Fallback

---

# Installation

## Installation über HACS (Empfohlen)

### Repository hinzufügen

1. HACS öffnen
2. Oben rechts auf die drei Punkte klicken
3. **Benutzerdefinierte Repositories**
4. Repository hinzufügen:

```text
https://github.com/MTTPoll/ha-heatpump-consumption-forecast
```

5. Kategorie auswählen:

```text
Integration
```

6. Hinzufügen
7. Installation starten
8. Home Assistant neu starten

### Integration hinzufügen

Nach dem Neustart:

```text
Einstellungen → Geräte & Dienste
```

Integration hinzufügen:

```text
Heat Pump Consumption Forecast
```

---

## Manuelle Installation

Aktuelle Version herunterladen:

```text
https://github.com/MTTPoll/ha-heatpump-consumption-forecast/releases
```

Ordner:

```text
custom_components/heatpump_consumption_forecast
```

nach:

```text
/config/custom_components/heatpump_consumption_forecast
```

kopieren.

Danach Home Assistant vollständig neu starten.

Anschließend:

```text
Einstellungen → Geräte & Dienste
```

und die Integration hinzufügen.

---

# ⚠️ Vor der Installation

Für eine sinnvolle Prognose sollten folgende Sensoren bereits vorhanden sein.

## Mindestanforderungen

| Sensor | Beschreibung |
|----------|-------------|
| Tagesverbrauch Wärmepumpe | Verbrauch seit 00:00 Uhr in kWh |
| Außentemperatur | Aktuelle Außentemperatur |
| Wetter-Entity | Wettervorhersage |

## Empfohlen

| Sensor | Nutzen |
|----------|---------|
| Heizverbrauch täglich | Genauere Heizanalyse |
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

## Warmwasserverbrauch

Ein separater Warmwasser-Energiezähler ist nicht erforderlich.

Die Integration berechnet den Warmwasserverbrauch automatisch aus:

```text
Warmwasser = Tagesverbrauch Wärmepumpe - Tagesverbrauch Heizung
```

Wenn die Heizung nicht läuft:

```text
Warmwasser = Tagesverbrauch Wärmepumpe
```

Dadurch entstehen auch bei Anlagen ohne separaten Warmwasserzähler vollständige Trainingsdaten.

---

## Erste Einrichtung Checkliste

Vor der Konfiguration sollte vorhanden sein:

✅ Tagesverbrauch Wärmepumpe

✅ Außentemperatur-Sensor

✅ Wetter-Entity

Empfohlen:

✅ Heizverbrauch täglich

✅ Gesamtverbrauch Wärmepumpe

---

## ML-Logik

### 0–29 abgeschlossene Tagesdatensätze

- Regelmodell aktiv
- ML sammelt Daten
- ML-Status: Wartet auf Trainingsdaten
- Prognosemodell: Regelmodell

### 30–89 abgeschlossene Tagesdatensätze

- ML wird aktiv
- Lokales Modell wird erzeugt
- Prognosemodell:
  - ML-Modell
  - ML + Fallback

### 90+ abgeschlossene Tagesdatensätze

- ML wird optimiert
- Heizkurve wird optimiert
- Prognosequalität verbessert sich weiter

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
- Letzter Trainingsdatensatz
- Trainingsstatus

### Machine Learning

- ML-Status
- Prognosemodell

### Heizkurve

- Heizkurvenstatus
- Erlernte Heizkurve

### Analyse

- Lernanalyse
- Prognosefehleranalyse

### Diagnose

- Speicherstatus

---

## Unterstützte Gebäudetypen

- Einfamilienhäuser
- Mehrfamilienhäuser
- Ferienhäuser
- Ferienwohnungen
- Apartmenthäuser

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

### Forecasting

- Tomorrow forecast
- Day-after-tomorrow forecast
- Remaining forecast for today
- Weather-based prediction
- Heating threshold support

### Occupancy Model

- Residential homes
- Holiday homes
- Vacation apartments
- Apartment buildings
- Multiple dwelling units
- Individual living areas
- Calendar-based occupancy
- Person model

### Consumption Analysis

- Heating energy
- Domestic hot water energy
- Automatic consumption split
- Historical learning

### Local Machine Learning

- Local similarity model
- Self-learning
- No external ML dependencies
- Raspberry Pi friendly
- Persistent storage
- Permanent rule-model fallback

---

# Installation

## Installation via HACS (Recommended)

### Add Repository

1. Open HACS
2. Click the three dots in the upper-right corner
3. Select **Custom Repositories**
4. Add:

```text
https://github.com/MTTPoll/ha-heatpump-consumption-forecast
```

5. Select:

```text
Integration
```

6. Add repository
7. Install integration
8. Restart Home Assistant

### Add Integration

After restart:

```text
Settings → Devices & Services
```

Add:

```text
Heat Pump Consumption Forecast
```

---

## Manual Installation

Download the latest release:

```text
https://github.com/MTTPoll/ha-heatpump-consumption-forecast/releases
```

Copy:

```text
custom_components/heatpump_consumption_forecast
```

to:

```text
/config/custom_components/heatpump_consumption_forecast
```

Restart Home Assistant completely.

Then add the integration through:

```text
Settings → Devices & Services
```

---

# ⚠️ Before Installation

For meaningful forecasts the following sensors should already exist.

## Minimum Requirements

| Sensor | Description |
|----------|-------------|
| Daily Heat Pump Energy Sensor | Daily energy consumption in kWh |
| Outdoor Temperature Sensor | Current outdoor temperature |
| Weather Entity | Weather forecast |

## Recommended

| Sensor | Benefit |
|----------|---------|
| Daily Heating Energy Sensor | Better heating analysis |
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

## Domestic Hot Water Consumption

A dedicated Domestic Hot Water energy meter is not required.

The integration automatically calculates DHW consumption from:

```text
DHW = Daily Heat Pump Consumption - Daily Heating Consumption
```

If heating consumption is zero:

```text
DHW = Daily Heat Pump Consumption
```

This allows complete training data even on systems without a dedicated DHW meter.

---

## First-Time Setup Checklist

Required:

✅ Daily heat pump energy sensor

✅ Outdoor temperature sensor

✅ Weather entity

Recommended:

✅ Daily heating energy sensor

✅ Total heat pump energy meter

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
- Heating curve becomes optimized

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

### Analysis

- Learning Analysis
- Forecast Error Analysis

### Diagnostics

- Storage Status

---

## Supported Building Types

- Residential homes
- Multi-family homes
- Holiday homes
- Vacation apartments
- Apartment buildings

---

## Community Feedback

This is the first public release.

Feedback regarding:

- Forecast accuracy
- ML behavior
- Heating curve quality
- Occupancy model quality
- Feature requests
- Bugs

is highly appreciated.

---

## License

MIT License
