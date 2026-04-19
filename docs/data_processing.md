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
Relevantes Output-File für Power BI: `correlations_target_long.csv`

### `src/analyse_regression.py`
Führt eine lineare Regressionsanalyse auf Basis des Feature-Datensatzes durch, um den Einfluss von Wetter-, Zeit- und Strukturmerkmalen auf den Stromverbrauch zu quantifizieren.  
Es werden drei Modellvarianten berechnet (mit Lag-Features, ohne Lag sowie ohne Lag inkl. Region), um unterschiedliche Einflussfaktoren vergleichbar zu machen.

Output-Files:
- `regression_model_metrics.csv` – Modellgütemetriken (R², MAE, RMSE)
- `regression_coefficients.csv` – Regressionskoeffizienten zur Interpretation der Feature-Effekte
- `regression_predictions_sample.csv` – Beispielhafte Predictions zur Bewertung der Modellqualität

### `src/analyse_regression_extended.py`
Erweitert `analyse_regression.py` um zwei Random-Forest-Modelle (Modell D und E) und ergänzt die bestehenden Output-Files um die neuen Ergebnisse. Muss **nach** `analyse_regression.py` ausgeführt werden.

Es werden folgende Modelle trainiert:

| Modell | Algorithmus | Features | Vergleich zu |
|--------|-------------|----------|--------------|
| D | Random Forest | Wetter + Zeit + Kanton | Modell C (LinReg) |
| E | Random Forest | Wetter + Zeit + Lag | Modell A (LinReg) |

Output-Files:
- `regression_model_metrics.csv` – aktualisiert (Modelle D + E ergänzt, Spalten `algorithm` und `model_label` hinzugefügt)
- `regression_coefficients.csv` – aktualisiert (RF Feature Importances für D + E angehängt)
- `regression_residuals.csv` – neu erstellt (Residualanalyse von Modell E für Power BI)

### `src/build_scenario_predictions_LR.py`
Erstellt ein vereinfachtes, erklärbares Prognosemodell auf Basis ausgewählter, fachlich interpretierbarer Features (Kanton, Saison, Tagtyp, Stunde und Temperaturklasse) mittels **linearer Regression**.  
Das Skript bereitet die Daten entsprechend auf, trainiert das Modell und evaluiert dessen Güte.  
Auf Basis des Modells werden anschliessend alle sinnvollen Szenario-Kombinationen generiert und für jede Kombination der erwartete Stromverbrauch prognostiziert.

Output-File: `scenario_predictions.csv`
- Enthält für jede Kombination von Kanton, Saison, Tagtyp, Stunde und Temperaturklasse den geschätzten Stromverbrauch
- **Wird in Power BI verwendet** für die interaktive Szenario-Analyse: Der User wählt via Slicer eine Kombination aus Einflussfaktoren und erhält unmittelbar eine modellbasierte Schätzung des Tagesverbrauchsprofils. Das lineare Modell wurde bewusst gewählt, da es stündliche Verbrauchsprofile realistisch und interpretierbar abbildet — im Gegensatz zum Random Forest, der unter fixen Szenariobedingungen zu stark mittelt und dadurch flache, wenig aussagekräftige Tagesverläufe erzeugt.

### `src/build_scenario_predictions_RF.py`
Variante von `build_scenario_predictions_LR.py` mit einem **Random Forest** als Prognosemodell (analog zu Modell D aus `analyse_regression_extended.py`: RF mit Kanton, ohne Lag-Features).  
Das Modell erzielt global eine höhere Prognosegenauigkeit (R² ≈ 0.93 vs. 0.85), eignet sich jedoch weniger für die Visualisierung typischer Tagesverläufe unter fixen Szenariobedingungen, da die stündliche Variation im RF-Output stark geglättet wird.  
Das Skript dient als Vergleichsgrundlage und Archivzweck.

Output-File: `scenario_predictions_rf.csv`
- Gleiche Struktur wie `scenario_predictions.csv`
- **Nicht in Power BI eingebunden** (siehe Begründung oben)

## Ausführungsreihenfolge

```
build_electricity_dataset.py
build_weather_dataset.py
build_dim_location.py
build_analytical_base.py
build_dim_time.py
build_feature_dataset.py
analyse_correlations.py
analyse_regression.py
analyse_regression_extended.py   ← muss nach analyse_regression.py laufen
build_scenario_predictions_LR.py
build_scenario_predictions_RF.py ← optional, nur für Modellvergleich
```

## Ordnerstruktur

### `data_raw/`
Enthält die ursprünglichen Rohdaten.

### `data_processed/`
Enthält die aufbereiteten Datensätze für Analyse und Reporting.

### `archive/`
Ablage für ältere Dateien, Zwischenstände oder nicht mehr verwendete Inhalte.

## Ziel
Aufbau einer sauberen und erweiterten Datenbasis für Feature Engineering, Analysen, Prognosen und Visualisierungen in Power BI.
