<style>
/* Style pour améliorer la lisibilité et la présentation */
h1 { color: #2E86C1; text-align: center; font-size: 28px; }
h2 { color: #2874A6; font-size: 24px; }
h3 { color: #1B4F72; font-size: 20px; }
h4 { color: #154360; font-size: 18px; }
p, li { font-size: 16px; }
strong { color: #D35400; }
</style>

# 🌾 Drought Monitoring in Béni Mellal-Khénifra (Morocco)

This project focuses on **detecting, characterizing, and predicting drought events** in the semi-arid region of **Béni Mellal-Khénifra, Morocco**, using **satellite-based indicators** (SPI and NDVI) and **machine learning techniques** to support agricultural drought monitoring and early warning systems.

---

## 📍 Study Region

- **Location**: Béni Mellal-Khénifra, Morocco 🇲🇦  
- **Climate**: Semi-arid, highly susceptible to drought events.  
- **Purpose**: Monitor vegetation health (NDVI) and precipitation deficits (SPI) to provide actionable insights for drought management.

---

## 📊 Objectives

1. **Data Collection & Processing**: Acquire and preprocess satellite-derived NDVI (MODIS) and SPI (CHIRPS) data.
2. **Exploratory Analysis**: Analyze spatio-temporal drought dynamics and correlations between SPI and NDVI.
3. **Machine Learning Models**: Develop models for drought classification, prediction, and anomaly detection.
4. **Visualization & Reporting**: Generate maps, time series, heatmaps, and interactive dashboards for drought monitoring.

---

## 📂 Data Sources

| Indicator | Source | Description |
|-----------|--------|-------------|
| **NDVI** | [MODIS (via CGMS-Maroc)](http://www.cgms-maroc.ma/ndvi/) | Normalized Difference Vegetation Index (250m, 10-day) |
| **SPI**  | [CHIRPS](https://www.chc.ucsb.edu/data/chirps) | Standardized Precipitation Index (monthly, derived from rainfall data) |

---

## 🧪 Methodology

### 1. Data Collection
- **NDVI**: Downloaded from CGMS-Maroc (2000–present) and clipped to Béni Mellal-Khénifra using a GeoJSON shapefile.
- **SPI**: Derived from CHIRPS precipitation data (1981–present), processed to compute monthly SPI at a 3-month scale.
- **Preprocessing**: Scaling, masking, cleaning missing values, and computing monthly means.

### 2. Exploratory Data Analysis
- **Time Series**: Visualize NDVI and SPI trends over time.
- **Seasonality**: Analyze monthly and seasonal patterns for both indices.
- **Correlation**: Assess NDVI-SPI relationships using Pearson correlation and cross-correlation with lags.
- **Drought Detection**: Identify drought events (SPI < -1 for ≥3 months) and classify severity (e.g., moderate, severe, extreme).

### 3. Machine Learning Modeling
- **Classification**: Random Forest Classifier to categorize drought conditions (Humide, Normale, Sécheresse modérée, Sécheresse sévère, Sécheresse extrême).
- **Regression**: Random Forest Regressor to predict NDVI based on SPI and temporal features.
- **Feature Engineering**: Include lagged variables (1, 2, 3, 6, 12 months), moving averages, trends, and seasonal indicators.

### 4. Visualization & Outputs
- **Maps**: Monthly NDVI and SPI maps for selected years (e.g., driest and greenest years).
- **Heatmaps**: Temporal drought patterns by year and month.
- **Interactive Dashboard**: Plotly-based dashboard showing SPI/NDVI trends, correlations, and drought classifications.
- **Report**: Comprehensive summary of drought statistics, events, and model performance.

### 5. Evaluation Metrics
- **Classification**: Accuracy, F1-score, confusion matrix.
- **Regression**: RMSE, R², Mean Squared Error (MSE).
- **Feature Importance**: Analyze contributions of lagged SPI/NDVI, seasonal, and trend features.

---

## 📈 Key Outputs
- **CSV Files**: `mean_ndvi_bmk.csv`, `mean_spi_bmk.csv`, `drought_events_bmk.csv`.
- **Visualizations**: Time series plots, seasonal boxplots, correlation scatter plots, drought heatmaps, and event timelines.
- **Models**: Trained Random Forest models for drought classification and NDVI prediction.
- **Dashboard**: Interactive Plotly dashboard for real-time drought monitoring.

---

## 📁 Project Structure

```
drought_monitoring_bmk/
│
├── data/
│   ├── chirps_data/              # Raw CHIRPS precipitation data
│   ├── chirps_beni_mellal/       # Clipped SPI rasters
│   ├── ndvi_data/                # Raw NDVI data
│   ├── ndvi_beni_mellal/         # Clipped NDVI rasters
│   ├── mean_ndvi_bmk.csv         # Monthly NDVI means
│   ├── mean_spi_bmk.csv          # Monthly SPI means
│   ├── drought_events_bmk.csv     # Detected drought events
│   └── beni_mellal_khenifra.geojson  # Region shapefile
├── scripts/
│   └── drought_analysis.py        # Main analysis script
├── outputs/
│   ├── carte_de_sechresse.png     # Drought heatmap
│   └── dashboard.html             # Interactive dashboard
├── README.md                     # Project documentation
└── requirements.txt              # Python dependencies
```

---

## 🛠️ Dependencies
- Python libraries: `pandas`, `numpy`, `geopandas`, `rasterio`, `matplotlib`, `seaborn`, `plotly`, `scikit-learn`, `xgboost`, `scipy`, `requests`, `beautifulsoup4`, `earthaccess`, `pymodis`.
- Install via: `pip install -r requirements.txt`

---

## 🧠 Author
**Adam Daoudi**  
Data Science Student – INSEA (Morocco)  
📅 Summer Internship 2025  
📬 Contact: addaoudi@insea.ac.ma

---
