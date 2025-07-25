# 🌾 Drought Monitoring in Béni Mellal-Khénifra (Morocco)

This project aims to **detect and predict drought events** in the semi-arid region of **Béni Mellal-Khénifra** using **satellite-based indicators** (SPI and NDVI) and **machine learning techniques**.

---

## 📍 Study Region

- **Location**: Béni Mellal-Khénifra, Morocco 🇲🇦  
- **Context**: Semi-arid climate, vulnerable to drought events.
- **Purpose**: Early warning and monitoring of vegetation health and water stress.

---

## 📊 Objectives

- Collect and process satellite data (SPI and NDVI).
- Visualize and analyze drought dynamics over time.
- Build ML models to classify and predict drought conditions.

---

## 📂 Data Sources

| Indicator | Source | Description |
|----------|--------|-------------|
| **NDVI** | [MODIS](https://modis.gsfc.nasa.gov/data/dataprod/mod13.php) | Normalized Difference Vegetation Index (250m, 16-day) |
| **SPI**  | [CHIRPS](https://www.chc.ucsb.edu/data/chirps) | Standardized Precipitation Index from CHIRPS rainfall data (monthly) |

---

## 🧪 Methodology

1. **Data Collection**
   - Download NDVI (2000–present) and SPI (1981–present) rasters.
   - Clip data to the Béni Mellal-Khénifra boundary using shapefiles.

2. **Data Preprocessing**
   - Apply scaling, masking, and NDVI filtering.
   - Compute monthly means and clean missing values.

3. **Exploratory Data Analysis**
   - Time series plots (NDVI & SPI)
   - SPI < -1 → drought condition
   - Correlation analysis NDVI vs SPI

4. **Machine Learning Modeling**
   - Classification (Drought vs No Drought)
   - Regression (Forecast future SPI or NDVI)
   - Anomaly detection

5. **Evaluation & Visualization**
   - Metrics: Accuracy, F1, RMSE, R²
   - Final visual outputs: maps, curves, and CSV exports

---

## 📁 Project Structure

🧠 Author
-
Adam Daoudi\
Data Science Student – INSEA (Morocco)\
📅 Summer Internship 2025\
📬 Contact: addaoudi@insea.ac.ma

