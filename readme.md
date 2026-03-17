# Energy-Weather Project

Dieses Projekt untersucht den Zusammenhang zwischen Wetter und Stromverbrauch in der Schweiz. Ziel ist es, die Daten schrittweise aufzubereiten, zu analysieren und anschliessend in Power BI zu visualisieren.

## Projektinhalt

### `src/build_electricity_dataset.py`
Bereitet die Rohdaten zum Stromverbrauch auf und erstellt einen stündlichen Stromdatensatz.

### `src/build_weather_dataset.py`
Bereitet die Wetterrohdaten auf und erstellt einen stündlichen Wetterdatensatz.

### `src/build_dim_location.py`
Erstellt eine einfache Standort- bzw. Regionsdimension für die weitere Analyse.

### `src/build_analytical_base.py`
Führt Strom- und Wetterdaten zu einer gemeinsamen Analytical Base zusammen.

### `src/build_dim_time.py`
Erstellt eine Zeitdimension auf Basis der vorhandenen Zeitstempel aus der Analytical Base.
Dabei werden zentrale zeitbasierte Features generiert (z. B. Stunde, Wochentag, Monat, Wochenende, Saison), die insbesondere für die Analyse von Verbrauchsmustern und die spätere Verwendung in Power BI relevant sind.

## Ordnerstruktur

### `data_raw/`
Enthält die ursprünglichen Rohdaten.

### `data_processed/`
Enthält die aufbereiteten Datensätze für Analyse und Reporting.

### `archive/`
Ablage für ältere Dateien, Zwischenstände oder nicht mehr verwendete Inhalte.

## Ziel
Aufbau einer sauberen Datenbasis für Feature Engineering, Analysen, Prognosen und Visualisierungen in Power BI.