# Heat Pump Consumption Forecast

Lokale Wärmepumpen-Verbrauchsprognose für Home Assistant.

Die Integration erstellt eine Verbrauchsprognose für Wärmepumpen anhand von historischen Verbrauchsdaten, Wetterdaten, Heizgrenze und optionaler Belegung bzw. Personenzahl. Sie ist für Wohnhäuser, Ferienhäuser und Mehrfamilienhäuser gedacht.

## Status

Aktuelle Version: **v0.8.0**

> Hinweis: v0.8.0 enthält noch kein aktives ML-Modell. Die Integration sammelt Trainingsdaten, analysiert Datenqualität und bereitet Heizkurve/ML-Auswertung vor.

## Funktionen

- Prognose für morgen und übermorgen
- Rest-Tagesprognose bis Mitternacht
- Tageshistorie aus Home-Assistant-Recorder
- Heizgrenze/Sommerbetrieb-Schwelle
- optionale Belegungskalender je Wohneinheit
- Quadratmeter je Wohneinheit
- Personenanzahl aus Kalenderbeschreibung, z. B. `4 Erw.`
- Trennung von Warmwasser- und Heizanteil, wenn Heizverbrauch vorhanden ist
- Lernspeicher für spätere ML-Auswertung
- Datenqualität und Trainingsstatus
- Heizkurvenstatus und erlernte Heizkurve in Vorbereitung
- Prognosegüte in fünf Stufen: Unzureichend, Schwach, Ausreichend, Gut, Sehr gut

## Installation per HACS als Custom Repository

1. HACS öffnen.
2. Rechts oben auf die drei Punkte klicken.
3. **Benutzerdefinierte Repositories** wählen.
4. Repository-URL einfügen:

   ```text
   https://github.com/MTTPoll/ha-heatpump-consumption-forecast
   ```

5. Kategorie **Integration** auswählen.
6. Repository hinzufügen.
7. Integration über HACS installieren.
8. Home Assistant neu starten.
9. Unter **Einstellungen → Geräte & Dienste → Integration hinzufügen** nach **Heat Pump Consumption Forecast** suchen.

## Manuelle Installation

1. Ordner kopieren:

   ```text
   custom_components/heatpump_consumption_forecast
   ```

2. Nach Home Assistant kopieren:

   ```text
   /config/custom_components/heatpump_consumption_forecast
   ```

3. Home Assistant neu starten.
4. Integration über die UI hinzufügen.

## Benötigte Datenquellen

Mindestens empfohlen:

- Wärmepumpen-Tagesverbrauch in kWh
- Außentemperatur-Sensor
- Wetter-Entity
- Heizgrenze, z. B. 17 °C

Optional:

- Gesamtverbrauchszähler der Wärmepumpe
- Heizverbrauch Tageswert
- Belegungskalender je Wohneinheit
- feste Personenzahlen
- Wohnfläche je Wohneinheit

## Wichtige Sensoren

- **Verbrauch morgen**
- **Verbrauch übermorgen**
- **Rest-Tagesprognose**
- **Prognosegrundlage**
- **Prognosegüte**
- **Gesammelte Tagesdaten**
- **Trainingsstatus**
- **Datenqualität**
- **Erlernte Heizkurve**
- **Heizkurvenstatus**

## Entwicklungsstand

## mit v0.8.2

- Trainingsdaten werden jetzt dauerhaft als JSON gespeichert.
- Speicherort: `/config/.storage/heatpump_consumption_forecast/training_data.json`
- Heizkurvenanalyse wird zusätzlich gespeichert unter: `/config/.storage/heatpump_consumption_forecast/heating_curve.json`
- Geplanter Speicherort für spätere ML-Modelle: `/config/.storage/heatpump_consumption_forecast/model.pkl`
- Migration vorhandener v0.7.x/v0.8.0-Trainingsdaten aus dem bisherigen Home-Assistant-Store vorbereitet.
- Diagnoseattribute zeigen die Speicherpfade an.

Hinweis: v0.8.2 enthält noch kein aktives ML-Modell. Die Integration sammelt weiterhin Trainingsdaten und bereitet die ML-Auswertung vor.

## v0.8.2

- Completed daily training samples are now marked with `completed: true` once final daily recorder values are available.
- Final daily values are stored separately from live `*_so_far` values.
- Internal forecast error values are stored for future quality evaluation, but are not displayed as user-facing percentage values.
- Added diagnostic sensor `Speicherstatus` for persistent JSON storage under `/config/.storage/heatpump_consumption_forecast/`.


## Lizenz

MIT License
