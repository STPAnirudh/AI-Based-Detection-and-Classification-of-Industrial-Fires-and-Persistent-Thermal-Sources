# SIH26162 — AI-Based Detection and Classification of Industrial Fires and Persistent Thermal Sources

**Department of Information Technology | Vasavi College of Engineering**  
**V Semester — Theme-Based Project (Batch B13)**  
**SIH Theme:** Disaster Management / AIMLDL  
**Current Milestone:** Review-1 Working Prototype  

---

> [!IMPORTANT]
> ### Review-1 Prototype Disclaimer & Notice
> This repository contains a **lightweight, demonstrable Review-1 prototype** built to prove technical feasibility.
> - **NOT Production Ready:** Built as a collegiate research prototype.
> - **Sample / Demo Data:** Uses NASA FIRMS-compatible CSV schema and sample industrial facility coordinates (Visakhapatnam Industrial Corridor). It does **not** claim to represent live NASA satellite streams or authenticated facility databases.
> - **Exploratory Baseline ML:** The Random Forest classifier is trained on a synthetic demonstration dataset. Accuracy metrics demonstrate pipeline integrity, **not** scientifically validated ground-truth fire detection.
> - **No Confirmed Fires:** Classified events are labelled with exploratory prototype categories (e.g., *Potential Industrial Fire*, *Persistent Thermal Source*) and do **not** constitute confirmed emergency incidents.

---

## 1. Project Objective

Industrial facilities frequently exhibit high-temperature thermal signatures from regular operations (flare stacks, blast furnaces, kilns, coking ovens) as well as accidental fires. Remote sensing systems like NASA FIRMS (VIIRS/MODIS) detect thermal anomalies globally, but satellite observations alone lack infrastructural context.

The objective of this project is to develop an AI-enabled geospatial system that:
1. Ingests satellite thermal anomaly data (NASA FIRMS format).
2. Integrates geospatial context from OpenStreetMap (OSM) and industrial infrastructure records.
3. Classifies hotspots using machine learning into operational thermal sources, vegetation fires, and potential industrial fires.
4. Analyzes historical persistence to identify recurring operational heat sources.
5. Evaluates multi-factor risk scores and presents actionable intelligence on an interactive GIS dashboard.

---

## 2. Review-1 Prototype Scope

The Review-1 milestone delivers a **complete, self-contained local working prototype** that demonstrates the end-to-end flow from data ingestion to interactive GIS visualization without heavy cloud or production dependencies.

### 5 Core Features Implemented:

| Feature | Review-1 Implementation Status |
| :--- | :--- |
| **1. Thermal Event Detection & Visualization** | Loads NASA FIRMS-compatible CSVs (VIIRS/MODIS format), validates coordinates, cleans anomalies, and displays them on an interactive multi-layer Leaflet GIS map. |
| **2. Geospatial Context & Proximity** | Computes great-circle (Haversine) distances between thermal events and nearby industrial facilities. Includes an optional OpenStreetMap / Overpass connector with graceful offline fallback. |
| **3. Baseline ML Classification** | Trains a scikit-learn `RandomForestClassifier` on thermal and spatial features to predict 4 prototype categories (*Potential Industrial Fire*, *Persistent Thermal Source*, *Natural/Vegetation Fire*, *Other/Uncertain*) with prediction confidence percentages. |
| **4. Persistence & Historical Analysis** | Implements an explainable spatial grouping algorithm to cluster recurring hotspots at approximately the same location. Computes occurrence count, observation time window, and persistence level (*High*, *Medium*, *Low*). |
| **5. Risk/Priority Assessment & GIS Dashboard** | Calculates explainable multi-factor risk scores (0–100) mapped to Priority tiers (*HIGH*, *MEDIUM*, *LOW*). Serves an interactive GIS dashboard with category filters, dynamic event inspection drawers, and a live AI prediction simulator. |

---

## 3. Architecture

The system follows a modular, feed-forward architecture designed for future expansion into full-stack FastAPI + React + PostgreSQL/PostGIS.

```
Data Sources (NASA FIRMS CSV, Industrial GIS CSV, OSM Overpass)
                           ↓
Data Ingestion (`src/ingestion.py`)
                           ↓
Data Cleaning & Coordinate Validation (`src/cleaning.py`)
                           ↓
Geospatial Proximity & Context (`src/geospatial.py`)
                           ↓
Persistence & Historical Spatial Grouping (`src/persistence.py`)
                           ↓
Baseline Machine Learning Inference (`src/ml_classifier.py`)
                           ↓
Multi-Factor Risk Scoring Engine (`src/risk_engine.py`)
                           ↓
REST API & Interactive GIS Dashboard (`src/api.py` + `static/`)
```

---

## 4. Technology Stack

- **Language:** Python 3.11+
- **Data Engineering:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn` (Random Forest, feature scaling, evaluation metrics)
- **Geospatial Processing:** Haversine great-circle calculation, coordinate bounding checks, spatial clustering
- **Backend API:** `FastAPI`, `uvicorn`, `pydantic`
- **Frontend / GIS:** HTML5, CSS3 (modern responsive dark theme), `Leaflet.js` (CartoDB Dark, OpenStreetMap, Esri Satellite Imagery layers)
- **Testing:** `pytest`, `httpx` (FastAPI TestClient)

---

## 5. Dataset & Data Source Explanation

```
data/
├── README_DATA.md                 # Detailed dataset schema and origin specifications
├── demo_firms_hotspots.csv        # 30 thermal anomaly records in standard NASA FIRMS VIIRS/MODIS format
├── demo_industrial_facilities.csv # 8 industrial infrastructure locations (Visakhapatnam Corridor)
└── demo_ml_training_data.csv      # 160 synthetic records for baseline ML demonstration
```

### Distinction Between Data Tiers:
- **FIRMS-Compatible Demo Data:** Uses the exact column specifications of NASA FIRMS (`latitude`, `longitude`, `brightness`, `scan`, `track`, `acq_date`, `acq_time`, `satellite`, `instrument`, `confidence`, `bright_t31`, `frp`, `daynight`). Generated across realistic coordinates in the Visakhapatnam industrial corridor.
- **Facility Data:** Realistic industrial locations (Steel Plant, Oil Refinery, Thermal Power Plant, Fertilizer Complex) representing industrial context.
- **Synthetic ML Training Set:** A balanced 160-sample training dataset generated using physical heuristic distributions (`src/generate_training_data.py`) to demonstrate training, feature importance, and probability calibration.
- **Prototype Logic:** The persistence thresholds (>= 6 high, 3-5 medium) and risk heuristic weights (35 class, 25 proximity, 20 FRP, 10 confidence, 10 persistence) are prototype formulas designed for review demonstration.

---

## 6. How to Install & Set Up

### Prerequisites
- Python 3.11+ installed.

### Step 1: Clone the repository
```bash
git clone https://github.com/STPAnirudh/AI-Based-Detection-and-Classification-of-Industrial-Fires-and-Persistent-Thermal-Sources.git
cd AI-Based-Detection-and-Classification-of-Industrial-Fires-and-Persistent-Thermal-Sources
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```
*(Dependencies: pandas, numpy, scikit-learn, fastapi, uvicorn, requests, pytest, httpx)*

---

## 7. How to Run the Prototype

### Option A: Complete Pipeline & Interactive GIS Dashboard (Recommended)
```bash
python run_demo.py
```
This command runs the end-to-end pipeline in terminal, prints summary statistics, and launches the web server at:  
👉 **`http://127.0.0.1:8000`**

Open this URL in any web browser to interact with the map, inspect events, and test the simulator.

### Option B: Terminal CLI Report Only (No Web Server)
```bash
python run_demo.py --cli
```
Runs the 6-stage pipeline and outputs formatted terminal tables showing ingestion counts, clusters, ML classification breakdown, and top risk events.

### Option C: Execute Automated Test Suite
```bash
pytest -v tests
# or
python run_demo.py --test
```
Runs all 18 automated unit and integration tests covering data loading, Haversine math, spatial clustering, Random Forest inference, risk bounds, and REST API endpoints.

---

## 8. Review-1 Feature Demonstration Guide

During Review-1, each of the 5 required features can be demonstrated as follows:

### Feature 1: Thermal Event Detection & Visualization
- **How to demonstrate:** Open `http://127.0.0.1:8000`. Show the interactive Leaflet map with 30 thermal hotspots. Switch between "Carto Dark", "Street Map", and "Satellite Imagery" base layers to view thermal anomalies over terrain.
- **Explain:** Point out the FIRMS-compatible fields in the table below (`brightness`, `frp`, `satellite`, `acq_time_fmt`).

### Feature 2: Geospatial Context & Proximity
- **How to demonstrate:** Click on any hotspot marker (e.g. `EVT_1013` or `EVT_1001`).
- **Explain:** Notice the dashed blue line drawn directly from the thermal event to the nearest industrial facility (`HPCL Visakh Refinery` or `Visakhapatnam Steel Plant`). The details panel displays the calculated Haversine distance (e.g. `0.18 km`) and facility sector.

### Feature 3: Baseline ML Classification
- **How to demonstrate:**
  1. Highlight the color-coded markers (Red = Potential Industrial Fire, Orange = Persistent Thermal Source, Green = Natural/Vegetation Fire, Grey = Other).
  2. Inspect `EVT_1013` (sudden 98 MW FRP near refinery classified as *Potential Industrial Fire* with ~94% confidence).
  3. Inspect `EVT_1017` (located 10 km away in Eastern Ghats hills classified as *Natural/Vegetation Fire*).
  4. Use the **Live AI Simulator** in the right drawer: enter custom FRP (e.g. 90 MW) and proximity (0.3 km) to see on-demand Random Forest predictions.

### Feature 4: Persistence & Historical Analysis
- **How to demonstrate:** Filter by "Persistence: High".
- **Explain:** Events `EVT_1001` through `EVT_1007` all cluster at the same blast furnace / coking oven coordinates across dates from January to March. The system identifies 7 historical detections over a 74-day span and classifies it as a *Persistent Thermal Source* with *High* persistence.

### Feature 5: Risk/Priority & GIS Dashboard
- **How to demonstrate:** Click on `EVT_1013` (High Priority) vs `EVT_1024` (Low Priority).
- **Explain:** Walk through the explainable risk breakdown box:  
  `Score 82.5 [Class: 35.0 pts | Proximity: 25.0 pts | FRP: 20.0 pts | Conf: 9.4 pts | Persistence: 2.0 pts] -> HIGH PRIORITY`.  
  Contrast this with a distant, low-FRP rural anomaly that scores `< 40 -> LOW PRIORITY`.

---

## 9. Current Limitations

1. **Demo Datasets:** Uses simulated NASA FIRMS records; live NASA FIRMS API key authentication will be added in Review-2.
2. **Synthetic Training Distribution:** Baseline ML model is trained on a synthetic feature matrix designed to verify pipeline plumbing, not verified ground truth.
3. **Flat-File Storage:** Datasets are currently stored as CSV files. A relational spatial database (PostgreSQL/PostGIS) is planned for the next milestone.
4. **Spatial Indexing:** Nearest-neighbor search is currently computed using standard distance iteration rather than R-tree / PostGIS spatial indexing.
5. **Heuristic Risk Weights:** Risk scoring weights are rule-based heuristics rather than actuarially or empirically calibrated disaster models.

---

## 10. Future Development Plan (Two-Semester Roadmap)

- [ ] **Milestone 2 (Semester 5):**
  - Integrate live NASA FIRMS API client with automated scheduled polling.
  - Implement PostgreSQL + PostGIS spatial database for indexed geographic queries (`ST_DWithin`, `ST_Distance`).
  - Expand OpenStreetMap Overpass extraction for all industrial land-use polygons in target state corridors.
  - Research and compile ground-truth historical industrial fire incident datasets.
- [ ] **Milestone 3 (Semester 6):**
  - Multi-model evaluation: Benchmark Random Forest against XGBoost, LightGBM, and Gradient Boosting.
  - Integrate Sentinel-2 multi-spectral satellite imagery (SWIR / NIR bands) for visual verification.
  - Develop deep learning CNN module for satellite image patch classification.
  - Build automated incident alerting engine (Email / Webhook / SMS dispatch).
  - Transition frontend to a decoupled React + Leaflet / MapLibre SPA.
  - Containerization via Docker and reproducible deployment pipeline.

---

## 11. Project Mapping Reference

This prototype directly satisfies the outcomes defined in the department project mapping:
- **PO1 (Engineering Knowledge):** Applied mathematical Haversine calculations and statistical ML classification.
- **PO3 (Design & Development):** Designed a modular decision-support pipeline combining remote sensing and GIS.
- **PO5 (Modern Tool Usage):** Utilized Python, scikit-learn, FastAPI, and Leaflet.js.
- **PSO1 & PSO2:** Built an intelligent software system applying AI/ML concepts to disaster risk monitoring.
- **SDG Alignment:** Goal 9 (Industry, Innovation & Infrastructure), Goal 11 (Sustainable Cities), Goal 13 (Climate Action).