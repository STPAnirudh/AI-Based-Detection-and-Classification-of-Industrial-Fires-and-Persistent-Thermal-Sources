# Data Catalog & Specifications (Review-1 Prototype)

> **DISCLAIMER:**  
> All data provided in this directory consists of **DEMO / SYNTHETIC DATASETS** created strictly for demonstrating the functionality of the Review-1 prototype.  
> They are **NOT** live NASA FIRMS satellite data feeds and are **NOT** scientifically validated ground-truth records. Live FIRMS API tokens and authenticated GIS feeds will be integrated in future phases.

---

## 1. `demo_firms_hotspots.csv`
Represents thermal anomaly detections formatted according to the **NASA FIRMS standard CSV specification** (VIIRS / MODIS format).

| Column | Type | Description |
| :--- | :--- | :--- |
| `event_id` | String | Unique identifier for the thermal hotspot detection |
| `latitude` | Float | Hotspot center latitude in decimal degrees (WGS84) |
| `longitude` | Float | Hotspot center longitude in decimal degrees (WGS84) |
| `brightness` | Float | Channel 21/22 or I4 brightness temperature (Kelvin) |
| `scan` | Float | Along-scan pixel size (km) |
| `track` | Float | Along-track pixel size (km) |
| `acq_date` | String | Acquisition date (`YYYY-MM-DD`) |
| `acq_time` | String | Acquisition time in UTC (`HHMM`) |
| `satellite` | String | Satellite platform (`N` = Suomi NPP, `1` = NOAA-20, `T` = Terra, `A` = Aqua) |
| `instrument`| String | Sensor instrument (`VIIRS` or `MODIS`) |
| `confidence`| String/Int | Detection confidence metric (`nominal`, `high`, `low` or 0-100) |
| `version` | String | Data processing version (`2.0NRT`) |
| `bright_t31`| Float | Channel 31 or I5 brightness temperature (Kelvin) |
| `frp` | Float | Fire Radiative Power (Megawatts - MW) |
| `daynight` | String | Observation day/night flag (`D` = Day, `N` = Night) |

---

## 2. `demo_industrial_facilities.csv`
Represents known industrial facilities used to calculate geospatial proximity and provide infrastructure context.

| Column | Type | Description |
| :--- | :--- | :--- |
| `facility_id` | String | Unique identifier for industrial infrastructure |
| `name` | String | Name of the facility |
| `latitude` | Float | Facility center latitude in decimal degrees |
| `longitude` | Float | Facility center longitude in decimal degrees |
| `facility_type`| String | Sector (e.g., `Steel Plant`, `Oil Refinery`, `Power Plant`, `Chemical Complex`) |
| `description` | String | Functional context (e.g., blast furnaces, flare units, tank farm) |

---

## 3. `demo_ml_training_data.csv`
Synthetic training set designed specifically for training the scikit-learn baseline classifier (`RandomForestClassifier`).

### Target Classes:
1. `Potential Industrial Fire`: High thermal intensity (FRP), located in close proximity (<1.5 km) to industrial infrastructure.
2. `Persistent Thermal Source`: Moderate-to-high temperature, occurring repeatedly day and night near industrial operations (flares, kilns, furnaces).
3. `Natural/Vegetation Fire`: Located away from industrial installations (>3.0 km) in vegetation/forest terrain, moderate FRP.
4. `Other/Uncertain`: Ambiguous, low-confidence, or low-FRP anomalies.
