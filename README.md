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
   https://github.com/MTTPoll/heatpump-consumption-forecast
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

Die Integration befindet sich in aktiver Entwicklung. Ziel ist eine robuste lokale Verbrauchsprognose und später ein kleines lokales ML-Modell, das auf den gesammelten Tagesdaten trainiert wird.

## GitHub Desktop Workflow

### Repository erstellen

1. GitHub Desktop öffnen.
2. **File → New repository** wählen.
3. Name eintragen:

   ```text
   heatpump-consumption-forecast
   ```

4. Lokalen Pfad auswählen.
5. Repository erstellen.

### Dateien einfügen

1. Den Inhalt dieses Projektordners in den neuen Repository-Ordner kopieren.
2. Wichtig: `custom_components` muss direkt im Repository-Root liegen.

Korrekte Struktur:

```text
heatpump-consumption-forecast/
├── custom_components/
│   └── heatpump_consumption_forecast/
│       ├── __init__.py
│       ├── config_flow.py
│       ├── const.py
│       ├── manifest.json
│       ├── sensor.py
│       ├── strings.json
│       └── translations/
│           ├── de.json
│           └── en.json
├── hacs.json
├── README.md
├── CHANGELOG.md
├── LICENSE
├── info.md
└── .gitignore
```

### Erster Commit

1. In GitHub Desktop sollten alle Dateien als Änderungen angezeigt werden.
2. Commit-Nachricht eingeben:

   ```text
   Initial release v0.8.0
   ```

3. Auf **Commit to main** klicken.

### Auf GitHub veröffentlichen

1. In GitHub Desktop auf **Publish repository** klicken.
2. Sichtbarkeit wählen: öffentlich oder privat.
3. Repository veröffentlichen.

### Release erstellen

1. Auf GitHub im Browser das Repository öffnen.
2. Rechts auf **Releases** klicken.
3. **Create a new release** wählen.
4. Tag eintragen:

   ```text
   v0.8.0
   ```

5. Titel:

   ```text
   v0.8.0
   ```

6. Beschreibung aus `CHANGELOG.md` übernehmen.
7. Release veröffentlichen.

## Lizenz

MIT License
