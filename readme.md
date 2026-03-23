# Energy-Weather Project

Dieses Projekt untersucht den Zusammenhang zwischen Wetter und Stromverbrauch in der Schweiz. Ziel ist es, die Daten schrittweise aufzubereiten, anzureichern, zu analysieren und anschliessend in Power BI zu visualisieren.

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

### `src/build_feature_dataset.py`
Erweitert die Analytical Base um zusätzliche Analyse- und Modellierungsfeatures (z. B. temperaturbasierte Kennzahlen wie HDD/CDD, Wetter-Flags sowie zeitliche Lag-Features des Stromverbrauchs).  
Das Ergebnis ist ein Feature-Datensatz, der als zentrale Faktentabelle für Analytics und Power BI dient.

### `src/analyse_correlations.py`
Berechnet Korrelationen zwischen Stromverbrauch und allen relevanten Features über verschiedene Analyse-Sichten (z. B. gesamt, je Kanton, je Saison oder nach Zeitmerkmalen).
Relevantes Output-File für PowerBI ist "correlations_target_long.csv"

### `src/regression_analysis.py`
Führt eine lineare Regressionsanalyse auf Basis des Feature-Datensatzes durch, um den Einfluss von Wetter-, Zeit- und Strukturmerkmalen auf den Stromverbrauch zu quantifizieren.
Es werden drei Modellvarianten berechnet (mit Lag-Features, ohne Lag sowie ohne Lag inkl. Region), um unterschiedliche Einflussfaktoren vergleichbar zu machen.
Der Output umfasst:
-Modellgütemetriken (z. B. R², MAE, RMSE)
-Regressionskoeffizienten zur Interpretation der Feature-Effekte
-Beispielhafte Predictions zur Bewertung der Modellqualität

## Ordnerstruktur

### `data_raw/`
Enthält die ursprünglichen Rohdaten.

### `data_processed/`
Enthält die aufbereiteten Datensätze für Analyse und Reporting.

### `archive/`
Ablage für ältere Dateien, Zwischenstände oder nicht mehr verwendete Inhalte.

## Ziel
Aufbau einer sauberen und erweiterten Datenbasis für Feature Engineering, Analysen, Prognosen und Visualisierungen in Power BI.