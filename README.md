# Dynamic Flight Path Rerouting Risk Assessment

A high-fidelity agentic AI system designed to predict and prescribe optimal flight trajectories by analyzing real-time transponder data and meteorological hazards. The system fuses deep learning (LSTM) and traditional ML (Random Forest) to assess rerouting risks and recommend safer corridors.

---

## Technical Overview

The system operates on the principle of **Predictive-Prescriptive Analytics**. It doesn't just predict if a flight will be rerouted; it calculates a "Risk Surface" across 3,600 possible altitude/heading combinations to prescribe the safest alternative.

### Core Data Streams
*   **Live Trajectories**: Consumed via the **OpenSky Network API**, providing state vectors for aircraft over the Continental US (CONUS).
*   **Meteorological Fusion**: Integrates three NOAA data products:
    *   **METAR**: High-frequency airport observations (wind, visibility, temp).
    *   **PIREP**: Pilot reports used to quantify actual turbulence intensity (0-6 scale).
    *   **SIGMET**: Hazardous weather polygons (Convective, Turbulence, Icing) defining immediate no-fly zones.

---

## Machine Learning Architecture

The system utilizes a weighted ensemble to maximize both point-in-time physics and historical trend analysis.

### 1. Random Forest (RF) - Local Hazard Model
*   **Purpose**: Analyzes the immediate physical environment (e.g., wind speed, temp-dewpoint spread, SIGMET proximity).
*   **Optimization**: Retrained with a 17-feature schema after removing redundant variables (e.g., wind_shear_index) to prevent logical inversions.
*   **Feature Engineering**: Includes circular heading encoding (Sin/Cos) and seasonal temporal features.

### 2. LSTM (RNN) - Sequential Trajectory Model
*   **Purpose**: Processes 5-point sequences of historical flight data to detect erratic maneuvers or speed changes that correlate with early-stage rerouting decisions.

### 3. Ensemble Calibration
*   **Weights**: Dynamically balanced (0.55 RF / 0.45 LSTM).
*   **Risk Thresholds**: Calibrated for aviation standards:
    *   **Low**: < 10%
    *   **Moderate**: 10% - 25%
    *   **High**: 25% - 50%
    *   **Critical**: > 50%

---

## Physical Meteorological Models

To maintain accuracy in data-sparse regions (e.g., high-altitude cruise), the system implements physical fallbacks:

*   **Altitude-Aware ISA Fallback**: If a METAR station is missing, the system applies the International Standard Atmosphere (ISA) lapse rate:
    *   Standard Temperature: `15.0C - (Altitude / 1000 * 1.98)`
*   **Wind Factor Scaling**: Adjusts surface winds to flight-level estimates based on a standard altitude-wind gradient.
*   **Visibility Bounds**: Enforces standard high-altitude visibility (10sm+) unless convective activity is present.

---

## Project Structure

```text
dynamic-flight-rerouting/
├── 01_collect_opensky.py    # Live transponder data acquisition
├── 02_collect_noaa.py       # Weather product ingestion (AWC)
├── 03_preprocess.py         # Spatial joins & feature engineering
├── 04_train_rf.py           # Random Forest training & hyperparameter tuning
├── 05_train_lstm.py         # Sequential LSTM training
├── 06_ensemble.py           # Model weighting & DeLong test evaluation
├── 07_risk_scoring.py       # Prescriptive grid search engine
├── 08_app.py                # Streamlit Dashboard (Inference Engine)
└── data/
    ├── processed/           # ML-ready datasets & scalers
    └── raw/                 # Raw trajectory and weather cache
```

---

## Installation & Usage

1.  **Environment Setup**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **API Credentials**: Create a `.env` file with your OpenSky Network credentials:
    ```env
    OPENSKY_USER=your_username
    OPENSKY_PASS=your_password
    ```
3.  **Run the Dashboard**:
    ```bash
    streamlit run 08_app.py
    ```

---

## Operational Modes

### Live Flight Lookup
Enter a flight's ICAO24 hex code or Callsign (e.g., `UAL123`). The system will:
1. Fetch the aircraft's current position and state.
2. Join the nearest NOAA METAR and SIGMET data.
3. Generate a real-time risk heatmap and alternative trajectory recommendation.

### Manual Scenario Testing
A "What-if" simulation tool. Adjust wind speed, turbulence intensity, and hazard flags manually to see how the ensemble model reacts to extreme hypothetical conditions.
