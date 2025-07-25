# Analyse des données géospatiales appliquée au contexte agricole
# Étude spatio-temporelle de l'indice NDVI et SPI - Région Béni Mellal-Khénifra
# Réalisé par : Adam Daoudi | Élève ingénieur en Data Science à l'INSEA

import os
import urllib.request
import time
from datetime import datetime, timedelta
import gzip
import shutil
import geopandas as gpd 
from osgeo import gdal
import rasterio
from rasterio.plot import show
from rasterio.mask import mask
from rasterio import open as rio_open
import glob
from bs4 import BeautifulSoup
import pandas as pd 
import numpy as np
from scipy.stats import norm, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import earthaccess 
from pathlib import Path
import logging
import requests
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# SECTION 1: COLLECTE ET PRÉTRAITEMENT DES DONNÉES
# =============================================================================

def download_chirps_data():
    """Télécharge les données CHIRPS pour le calcul du SPI"""
    base_url = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_monthly/tifs/"
    output_dir = "chirps_data"
    os.makedirs(output_dir, exist_ok=True)
    
    dates = pd.date_range(start="2000-01", end="2024-12", freq="MS")
    formatted_dates = [date.strftime("%Y.%m") for date in dates]
    
    for date in formatted_dates:
        filename = f'chirps-v2.0.{date}.tif.gz'
        url = base_url + filename
        filepath = os.path.join(output_dir, filename)
        
        if os.path.exists(filepath):
            print(f"{filename} déjà présent, on passe.")
            continue
            
        print(f"Téléchargement de {filename}...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"Téléchargé : {filename}")
                break
            except Exception as e:
                print(f"Tentative {attempt + 1} échouée : {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print(f"Échec du téléchargement de {filename}")

def decompress_chirps():
    """Décompresse les fichiers CHIRPS"""
    input_dir = "chirps_data"
    output_dir = "chirps_tifs"
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".tif.gz"):
            gz_path = os.path.join(input_dir, filename)
            tif_filename = filename[:-3]
            tif_path = os.path.join(output_dir, tif_filename)
            
            if os.path.exists(tif_path):
                continue
                
            try:
                with gzip.open(gz_path, 'rb') as f_in, open(tif_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                print(f"Décompressé : {tif_filename}")
            except Exception as e:
                print(f"Erreur avec {filename} : {e}")

def clip_rasters_to_region(input_folder, output_folder, shapefile_path, file_extension=".tif"):
    """Découpe les rasters selon les limites de la région"""
    os.makedirs(output_folder, exist_ok=True)
    gdf = gpd.read_file(shapefile_path)
    
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(file_extension):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            if os.path.exists(output_path):
                continue
                
            try:
                with rasterio.open(input_path) as src:
                    if gdf.crs != src.crs:
                        gdf = gdf.to_crs(src.crs)
                    
                    out_image, out_transform = mask(src, gdf.geometry, crop=True)
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform
                    })
                    
                    with rasterio.open(output_path, "w", **out_meta) as dest:
                        dest.write(out_image)
                        
                print(f"Découpé : {filename}")
            except Exception as e:
                print(f"Erreur avec {filename}: {e}")

def calculate_spi(precipitation_data, scale=3):
    """Calcule l'indice SPI à partir des données de précipitations"""
    # Calcul de la moyenne mobile sur la période spécifiée
    rolling_precip = precipitation_data.rolling(window=scale, min_periods=1).sum()
    
    # Calcul des paramètres de la distribution gamma
    # En pratique, utilisez scipy.stats.gamma.fit() pour une implémentation complète
    mean_precip = rolling_precip.mean()
    std_precip = rolling_precip.std()
    
    # Approximation normale pour simplifier (à améliorer avec distribution gamma)
    spi = (rolling_precip - mean_precip) / std_precip
    return spi

def extract_mean_values(input_folder, index_type, shapefile_path):
    """Extrait les valeurs moyennes des rasters pour la région"""
    gdf = gpd.read_file(shapefile_path)
    results = []
    
    files = sorted([f for f in os.listdir(input_folder) 
                   if f.endswith((".tif", ".tiff"))])
    
    for file in files:
        file_path = os.path.join(input_folder, file)
        
        try:
            with rasterio.open(file_path) as src:
                if gdf.crs != src.crs:
                    gdf = gdf.to_crs(src.crs)
                
                out_image, _ = mask(src, gdf.geometry, crop=True)
                data = out_image[0]
                
                # Nettoyage des données
                data = np.where((data == src.nodata) | np.isnan(data), np.nan, data)
                
                if index_type.upper() == "NDVI":
                    data = data.astype(float) * 0.0001
                    data = np.where((data < -0.2) | (data > 1.0), np.nan, data)
                elif index_type.upper() == "SPI":
                    data = data.astype(float)
                
                mean_value = np.nanmean(data)
                
                # Extraction de la date
                if index_type.upper() == "NDVI":
                    match = re.search(r'(\d{4}\.\d{2}\.\d{2})', file)
                    if match:
                        date_str = match.group(1)
                        date_obj = pd.to_datetime(date_str, format="%Y.%m.%d")
                else:  # SPI/CHIRPS
                    match = re.search(r'(\d{4}\.\d{2})', file)
                    if match:
                        date_str = match.group(1)
                        date_obj = pd.to_datetime(date_str, format="%Y.%m")
                
                if match:
                    results.append({
                        "date": date_obj, 
                        f"mean_{index_type.lower()}": mean_value
                    })
                    
        except Exception as e:
            print(f"Erreur avec {file}: {e}")
    
    df = pd.DataFrame(results)
    df = df.sort_values("date").reset_index(drop=True)
    return df

# =============================================================================
# SECTION 2: ANALYSE EXPLORATOIRE ET VISUALISATION
# =============================================================================

def create_time_series_plots(df, columns, title, save_path=None):
    """Crée des graphiques de séries temporelles"""
    fig, axes = plt.subplots(len(columns), 1, figsize=(15, 6*len(columns)))
    if len(columns) == 1:
        axes = [axes]
    
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    for i, col in enumerate(columns):
        axes[i].plot(df['date'], df[col], color=colors[i], linewidth=2, label=col)
        axes[i].set_title(f'{title} - {col.upper()}', fontsize=14)
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel(col.upper())
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_seasonal_patterns(df, value_col):
    """Analyse les patterns saisonniers"""
    df_copy = df.copy()
    df_copy['month'] = df_copy['date'].dt.month
    df_copy['year'] = df_copy['date'].dt.year
    df_copy['season'] = df_copy['month'].map({
        12: 'Hiver', 1: 'Hiver', 2: 'Hiver',
        3: 'Printemps', 4: 'Printemps', 5: 'Printemps',
        6: 'Été', 7: 'Été', 8: 'Été',
        9: 'Automne', 10: 'Automne', 11: 'Automne'
    })
    
    # Analyse mensuelle
    monthly_stats = df_copy.groupby('month')[value_col].agg(['mean', 'std', 'min', 'max'])
    
    # Analyse saisonnière
    seasonal_stats = df_copy.groupby('season')[value_col].agg(['mean', 'std', 'min', 'max'])
    
    # Visualisation
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Boxplot mensuel
    df_copy.boxplot(column=value_col, by='month', ax=axes[0,0])
    axes[0,0].set_title(f'Distribution mensuelle de {value_col.upper()}')
    axes[0,0].set_xlabel('Mois')
    
    # Boxplot saisonnier
    df_copy.boxplot(column=value_col, by='season', ax=axes[0,1])
    axes[0,1].set_title(f'Distribution saisonnière de {value_col.upper()}')
    
    # Moyennes mensuelles
    monthly_stats['mean'].plot(kind='bar', ax=axes[1,0])
    axes[1,0].set_title(f'Moyennes mensuelles de {value_col.upper()}')
    axes[1,0].set_xlabel('Mois')
    
    # Moyennes saisonnières
    seasonal_stats['mean'].plot(kind='bar', ax=axes[1,1])
    axes[1,1].set_title(f'Moyennes saisonnières de {value_col.upper()}')
    
    plt.suptitle('')
    plt.tight_layout()
    plt.show()
    
    return monthly_stats, seasonal_stats

def correlation_analysis(df):
    """Analyse de corrélation entre NDVI et SPI"""
    # Calcul de la corrélation
    correlation = df[['mean_ndvi', 'mean_spi']].corr()
    
    # Test de significativité
    corr_coef, p_value = pearsonr(df['mean_ndvi'].dropna(), df['mean_spi'].dropna())
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Matrice de corrélation
    sns.heatmap(correlation, annot=True, cmap='RdBu_r', center=0, ax=axes[0])
    axes[0].set_title('Matrice de corrélation NDVI-SPI')
    
    # Scatter plot
    axes[1].scatter(df['mean_spi'], df['mean_ndvi'], alpha=0.6, color='green')
    axes[1].set_xlabel('SPI')
    axes[1].set_ylabel('NDVI')
    axes[1].set_title(f'Relation NDVI-SPI (r={corr_coef:.3f}, p={p_value:.3f})')
    
    # Ligne de régression
    z = np.polyfit(df['mean_spi'].dropna(), df['mean_ndvi'].dropna(), 1)
    p = np.poly1d(z)
    axes[1].plot(df['mean_spi'], p(df['mean_spi']), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.show()
    
    return corr_coef, p_value

# =============================================================================
# SECTION 3: DÉTECTION ET CLASSIFICATION DE LA SÉCHERESSE
# =============================================================================

def classify_drought_severity(spi_values, ndvi_values):
    """Classifie la sévérité de la sécheresse basée sur SPI et NDVI"""
    drought_classes = []
    
    for spi, ndvi in zip(spi_values, ndvi_values):
        if pd.isna(spi) or pd.isna(ndvi):
            drought_classes.append('Données manquantes')
        elif spi >= 0.5:
            drought_classes.append('Humide')
        elif -0.5 <= spi < 0.5:
            drought_classes.append('Normal')
        elif -1.0 <= spi < -0.5:
            drought_classes.append('Sécheresse modérée')
        elif -1.5 <= spi < -1.0:
            drought_classes.append('Sécheresse sévère')
        else:
            drought_classes.append('Sécheresse extrême')
    
    return drought_classes

def detect_drought_events(df, spi_threshold=-1.0, duration_threshold=3):
    """Détecte les événements de sécheresse"""
    df_copy = df.copy()
    df_copy['drought'] = df_copy['mean_spi'] < spi_threshold    
    
    # Identifier les périodes continues de sécheresse
    df_copy['drought_group'] = (df_copy['drought'] != df_copy['drought'].shift()).cumsum()
    
    drought_events = []
    for group_id, group in df_copy.groupby('drought_group'):
        if group['drought'].iloc[0] and len(group) >= duration_threshold:
            drought_events.append({
                'start_date': group['date'].min(),
                'end_date': group['date'].max(),
                'duration_months': len(group),
                'min_spi': group['mean_spi'].min(),
                'avg_spi': group['mean_spi'].mean(),
                'min_ndvi': group['mean_ndvi'].min(),
                'avg_ndvi': group['mean_ndvi'].mean()
            })
    
    return pd.DataFrame(drought_events)

# =============================================================================
# SECTION 4: MODÉLISATION ET PRÉDICTION
# =============================================================================

def prepare_features(df, lag_periods=[1, 2, 3, 6, 12]):
    """Prépare les features pour la modélisation"""
    df_features = df.copy()
    
    # Features temporelles
    df_features['month'] = df_features['date'].dt.month
    df_features['year'] = df_features['date'].dt.year
    df_features['season'] = df_features['month'].map({
        12: 0, 1: 0, 2: 0,  # Hiver
        3: 1, 4: 1, 5: 1,   # Printemps
        6: 2, 7: 2, 8: 2,   # Été
        9: 3, 10: 3, 11: 3  # Automne
    })
    
    # Features de lag
    for lag in lag_periods:
        df_features[f'ndvi_lag_{lag}'] = df_features['mean_ndvi'].shift(lag)
        df_features[f'spi_lag_{lag}'] = df_features['mean_spi'].shift(lag)
    
    # Moyennes mobiles
    for window in [3, 6, 12]:
        df_features[f'ndvi_ma_{window}'] = df_features['mean_ndvi'].rolling(window=window).mean()
        df_features[f'spi_ma_{window}'] = df_features['mean_spi'].rolling(window=window).mean()
    
    # Tendances
    df_features['ndvi_trend'] = df_features['mean_ndvi'].diff()
    df_features['spi_trend'] = df_features['mean_spi'].diff()
    
    return df_features

def train_drought_prediction_model(df):
    """Entraîne un modèle de prédiction de sécheresse"""
    # Préparation des features
    df_features = prepare_features(df)
    
    # Classification de la sécheresse
    df_features['drought_class'] = classify_drought_severity(
        df_features['mean_spi'], df_features['mean_ndvi']
    )
    
    # Encoding des classes
    drought_encoding = {
        'Humide': 0, 'Normal': 1, 'Sécheresse modérée': 2, 
        'Sécheresse sévère': 3, 'Sécheresse extrême': 4
    }
    df_features['drought_encoded'] = df_features['drought_class'].map(drought_encoding)
    
    # Sélection des features
    feature_cols = [col for col in df_features.columns 
                   if col not in ['date', 'drought_class', 'drought_encoded']]
    
    # Suppression des valeurs manquantes
    df_clean = df_features.dropna()
    
    if len(df_clean) == 0:
        print("Pas assez de données après nettoyage")
        return None, None, None
    
    X = df_clean[feature_cols]
    y = df_clean['drought_encoded']
    
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
    )
    
    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entraînement du modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test_scaled)
    
    # Évaluation
    print("=== ÉVALUATION DU MODÈLE DE CLASSIFICATION ===")
    print(f"Accuracy: {model.score(X_test_scaled, y_test):.3f}")
    print("\nRapport de classification:")
    print(classification_report(y_test, y_pred))
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de confusion - Classification de la sécheresse')
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies valeurs')
    plt.show()
    
    # Importance des features
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
    plt.title('Importance des features - Top 15')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()
    
    return model, scaler, feature_importance

def train_ndvi_prediction_model(df):
    """Entraîne un modèle de prédiction NDVI"""
    df_features = prepare_features(df)
    
    # Features pour prédire NDVI
    feature_cols = [col for col in df_features.columns 
                   if col not in ['date', 'mean_ndvi'] and 'ndvi' not in col]
    
    df_clean = df_features.dropna()
    
    if len(df_clean) == 0:
        print("Pas assez de données pour la prédiction NDVI")
        return None, None
    
    X = df_clean[feature_cols]
    y = df_clean['mean_ndvi']
    
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entraînement
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test_scaled)
    
    # Évaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("=== ÉVALUATION DU MODÈLE DE PRÉDICTION NDVI ===")
    print(f"MSE: {mse:.6f}")
    print(f"R²: {r2:.3f}")
    print(f"RMSE: {np.sqrt(mse):.6f}")
    
    # Visualisation des prédictions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('NDVI observé')
    plt.ylabel('NDVI prédit')
    plt.title(f'Prédiction NDVI (R² = {r2:.3f})')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return model, scaler

# =============================================================================
# SECTION 5: ANALYSE SPATIALE ET CLUSTERING
# =============================================================================

def spatial_clustering_analysis(input_folder, n_clusters=5):
    """Analyse de clustering spatial des patterns de sécheresse"""
    files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.tif', '.tiff'))])
    
    # Collecter les données spatiales
    spatial_data = []
    
    for file in files[:12]:  # Limiter pour l'exemple
        file_path = os.path.join(input_folder, file)
        
        try:
            with rasterio.open(file_path) as src:
                data = src.read(1)
                
                # Conversion en coordonnées
                height, width = data.shape
                cols, rows = np.meshgrid(np.arange(width), np.arange(height))
                xs, ys = rasterio.transform.xy(src.transform, rows, cols)
                
                # Aplatir les données
                valid_mask = ~np.isnan(data)
                x_flat = np.array(xs)[valid_mask]
                y_flat = np.array(ys)[valid_mask]
                values_flat = data[valid_mask]
                
                for x, y, val in zip(x_flat, y_flat, values_flat):
                    spatial_data.append({
                        'x': x, 'y': y, 'value': val, 
                        'date': file.split('.')[1] + '.' + file.split('.')[2]
                    })
                    
        except Exception as e:
            print(f"Erreur avec {file}: {e}")
    
    if not spatial_data:
        print("Aucune donnée spatiale collectée")
        return None
    
    df_spatial = pd.DataFrame(spatial_data)
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_spatial['cluster'] = kmeans.fit_predict(df_spatial[['x', 'y', 'value']])
    
    # Visualisation
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df_spatial['x'], df_spatial['y'], 
                         c=df_spatial['cluster'], cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('Clustering spatial des patterns de sécheresse')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()
    
    return df_spatial, kmeans

# =============================================================================
# SECTION 6: RAPPORTS ET SYNTHÈSE
# =============================================================================

def generate_drought_report(df, drought_events):
    """Génère un rapport complet sur la sécheresse"""
    print("=" * 60)
    print("RAPPORT D'ANALYSE DE LA SÉCHERESSE")
    print("Région : Béni Mellal-Khénifra")
    print("=" * 60)
    
    # Statistiques générales
    print(f"\n1. PÉRIODE D'ANALYSE:")
    print(f"   - Début: {df['date'].min().strftime('%Y-%m-%d')}")
    print(f"   - Fin: {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"   - Nombre d'observations: {len(df)}")
    
    # Statistiques SPI
    print(f"\n2. STATISTIQUES SPI:")
    print(f"   - Moyenne: {df['mean_spi'].mean():.3f}")
    print(f"   - Écart-type: {df['mean_spi'].std():.3f}")
    print(f"   - Minimum: {df['mean_spi'].min():.3f}")
    print(f"   - Maximum: {df['mean_spi'].max():.3f}")
    
    # Statistiques NDVI
    print(f"\n3. STATISTIQUES NDVI:")
    print(f"   - Moyenne: {df['mean_ndvi'].mean():.3f}")
    print(f"   - Écart-type: {df['mean_ndvi'].std():.3f}")
    print(f"   - Minimum: {df['mean_ndvi'].min():.3f}")
    print(f"   - Maximum: {df['mean_ndvi'].max():.3f}")
    
    # Événements de sécheresse
    print(f"\n4. ÉVÉNEMENTS DE SÉCHERESSE DÉTECTÉS:")
    print(f"   - Nombre total: {len(drought_events)}")
    
    if len(drought_events) > 0:
        print(f"   - Durée moyenne: {drought_events['duration_months'].mean():.1f} mois")
        print(f"   - Durée maximale: {drought_events['duration_months'].max()} mois")
        print(f"   - SPI minimum enregistré: {drought_events['min_spi'].min():.3f}")
        
        print("\n   Événements les plus sévères:")
        severe_events = drought_events.nsmallest(3, 'min_spi')
        for _, event in severe_events.iterrows():
            print(f"     • {event['start_date'].strftime('%Y-%m')} - "
                  f"{event['end_date'].strftime('%Y-%m')} "
                  f"(SPI min: {event['min_spi']:.3f})")
    
    # Classification des conditions
    drought_classes = classify_drought_severity(df['mean_spi'], df['mean_ndvi'])
    class_counts = pd.Series(drought_classes).value_counts()
    
    print(f"\n5. RÉPARTITION DES CONDITIONS:")
    for condition, count in class_counts.items():
        percentage = (count / len(drought_classes)) * 100
        print(f"   - {condition}: {count} mois ({percentage:.1f}%)")
    
    # Corrélation
    corr_coef, p_value = pearsonr(df['mean_ndvi'].dropna(), df['mean_spi'].dropna())
    print(f"\n6. CORRÉLATION NDVI-SPI:")
    print(f"   - Coefficient de corrélation: {corr_coef:.3f}")
    print(f"   - P-value: {p_value:.6f}")
    print(f"   - Significativité: {'Significative' if p_value < 0.05 else 'Non significative'}")
    
    print("\n" + "=" * 60)
    print("FIN DU RAPPORT")
    print("=" * 60)

def create_interactive_dashboard(df, drought_events):
    """Crée un tableau de bord interactif avec Plotly"""
    
    # Préparation des données
    df_copy = df.copy()
    df_copy['drought_class'] = classify_drought_severity(df_copy['mean_spi'], df_copy['mean_ndvi'])
    df_copy['year'] = df_copy['date'].dt.year
    df_copy['month'] = df_copy['date'].dt.month
    
    # Création des sous-graphiques
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Évolution temporelle SPI', 'Évolution temporelle NDVI',
                       'Relation SPI-NDVI', 'Distribution des classes de sécheresse',
                       'Tendances annuelles', 'Patterns saisonniers'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Évolution temporelle SPI
    fig.add_trace(
        go.Scatter(x=df_copy['date'], y=df_copy['mean_spi'],
                  mode='lines', name='SPI', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Ajout des seuils de sécheresse
    fig.add_hline(y=-1.0, line_dash="dash", line_color="orange", 
                  annotation_text="Sécheresse modérée", row=1, col=1)
    fig.add_hline(y=-1.5, line_dash="dash", line_color="red", 
                  annotation_text="Sécheresse sévère", row=1, col=1)
    
    # 2. Évolution temporelle NDVI
    fig.add_trace(
        go.Scatter(x=df_copy['date'], y=df_copy['mean_ndvi'],
                  mode='lines', name='NDVI', line=dict(color='green')),
        row=1, col=2
    )
    
    # 3. Relation SPI-NDVI
    fig.add_trace(
        go.Scatter(x=df_copy['mean_spi'], y=df_copy['mean_ndvi'],
                  mode='markers', name='SPI-NDVI', 
                  marker=dict(color=df_copy['year'], colorscale='viridis',
                             showscale=True, colorbar=dict(title="Année"))),
        row=2, col=1
    )
    
    # 4. Distribution des classes de sécheresse
    class_counts = df_copy['drought_class'].value_counts()
    fig.add_trace(
        go.Bar(x=class_counts.index, y=class_counts.values,
               name='Classes', marker_color='lightblue'),
        row=2, col=2
    )
    
    # 5. Tendances annuelles
    annual_stats = df_copy.groupby('year').agg({
        'mean_spi': 'mean',
        'mean_ndvi': 'mean'
    }).reset_index()
    
    fig.add_trace(
        go.Scatter(x=annual_stats['year'], y=annual_stats['mean_spi'],
                  mode='lines+markers', name='SPI annuel', line=dict(color='blue')),
        row=3, col=1
    )
    
    # 6. Patterns saisonniers
    seasonal_stats = df_copy.groupby('month').agg({
        'mean_spi': 'mean',
        'mean_ndvi': 'mean'
    }).reset_index()
    
    fig.add_trace(
        go.Scatter(x=seasonal_stats['month'], y=seasonal_stats['mean_spi'],
                  mode='lines+markers', name='SPI saisonnier', line=dict(color='blue')),
        row=3, col=2
    )
    
    # Mise à jour de la mise en page
    fig.update_layout(
        height=1200,
        title_text="Tableau de bord - Analyse de la sécheresse BMK",
        showlegend=False
    )
    
    # Mise à jour des axes
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_xaxes(title_text="SPI", row=2, col=1)
    fig.update_xaxes(title_text="Classes", row=2, col=2)
    fig.update_xaxes(title_text="Année", row=3, col=1)
    fig.update_xaxes(title_text="Mois", row=3, col=2)
    
    fig.update_yaxes(title_text="SPI", row=1, col=1)
    fig.update_yaxes(title_text="NDVI", row=1, col=2)
    fig.update_yaxes(title_text="NDVI", row=2, col=1)
    fig.update_yaxes(title_text="Fréquence", row=2, col=2)
    fig.update_yaxes(title_text="SPI", row=3, col=1)
    fig.update_yaxes(title_text="SPI", row=3, col=2)
    
    fig.show()
    
    return fig

def vulnerability_assessment(df, drought_events):
    """Évalue la vulnérabilité de la région à la sécheresse"""
    
    print("\n" + "=" * 60)
    print("ÉVALUATION DE LA VULNÉRABILITÉ À LA SÉCHERESSE")
    print("=" * 60)
    
    # Calcul des métriques de vulnérabilité
    
    # 1. Fréquence des événements de sécheresse
    total_months = len(df)
    drought_months = len(df[df['mean_spi'] < -1.0])
    drought_frequency = (drought_months / total_months) * 100
    
    # 2. Intensité moyenne des sécheresses
    drought_periods = df[df['mean_spi'] < -1.0]
    avg_drought_intensity = abs(drought_periods['mean_spi'].mean()) if len(drought_periods) > 0 else 0
    
    # 3. Impact sur la végétation
    drought_ndvi_impact = df[df['mean_spi'] < -1.0]['mean_ndvi'].mean() if len(drought_periods) > 0 else 0
    normal_ndvi = df[df['mean_spi'] >= -0.5]['mean_ndvi'].mean()
    ndvi_reduction = ((normal_ndvi - drought_ndvi_impact) / normal_ndvi) * 100 if normal_ndvi > 0 else 0
    
    # 4. Durée moyenne des événements
    avg_duration = drought_events['duration_months'].mean() if len(drought_events) > 0 else 0
    
    # 5. Tendance temporelle
    years = df['date'].dt.year
    spi_trend = np.polyfit(range(len(df)), df['mean_spi'], 1)[0]
    ndvi_trend = np.polyfit(range(len(df)), df['mean_ndvi'], 1)[0]
    
    print(f"\n1. MÉTRIQUES DE VULNÉRABILITÉ:")
    print(f"   • Fréquence de sécheresse: {drought_frequency:.1f}% des mois")
    print(f"   • Intensité moyenne des sécheresses: {avg_drought_intensity:.3f}")
    print(f"   • Durée moyenne des événements: {avg_duration:.1f} mois")
    print(f"   • Réduction NDVI durant sécheresses: {ndvi_reduction:.1f}%")
    
    print(f"\n2. TENDANCES TEMPORELLES:")
    print(f"   • Tendance SPI: {spi_trend:.6f} par mois")
    print(f"   • Tendance NDVI: {ndvi_trend:.6f} par mois")
    
    # Classification de la vulnérabilité
    vulnerability_score = 0
    
    if drought_frequency > 30:
        vulnerability_score += 3
    elif drought_frequency > 20:
        vulnerability_score += 2
    elif drought_frequency > 10:
        vulnerability_score += 1
    
    if avg_drought_intensity > 2.0:
        vulnerability_score += 3
    elif avg_drought_intensity > 1.5:
        vulnerability_score += 2
    elif avg_drought_intensity > 1.0:
        vulnerability_score += 1
    
    if avg_duration > 6:
        vulnerability_score += 3
    elif avg_duration > 4:
        vulnerability_score += 2
    elif avg_duration > 2:
        vulnerability_score += 1
    
    if ndvi_reduction > 15:
        vulnerability_score += 3
    elif ndvi_reduction > 10:
        vulnerability_score += 2
    elif ndvi_reduction > 5:
        vulnerability_score += 1
    
    # Classification finale
    if vulnerability_score >= 9:
        vulnerability_level = "TRÈS ÉLEVÉE"
    elif vulnerability_score >= 6:
        vulnerability_level = "ÉLEVÉE"
    elif vulnerability_score >= 3:
        vulnerability_level = "MODÉRÉE"
    else:
        vulnerability_level = "FAIBLE"
    
    print(f"\n3. NIVEAU DE VULNÉRABILITÉ:")
    print(f"   • Score: {vulnerability_score}/12")
    print(f"   • Niveau: {vulnerability_level}")
    
    # Recommandations
    print(f"\n4. RECOMMANDATIONS:")
    if vulnerability_score >= 6:
        print("   • Mise en place d'un système d'alerte précoce")
        print("   • Développement de cultures résistantes à la sécheresse")
        print("   • Amélioration des systèmes d'irrigation")
        print("   • Diversification des activités agricoles")
    else:
        print("   • Surveillance continue des indicateurs")
        print("   • Préparation de plans de contingence")
        print("   • Formation des agriculteurs aux techniques d'adaptation")
    
    return vulnerability_score, vulnerability_level

# =============================================================================
# SECTION 7: FONCTIONS PRINCIPALES D'EXÉCUTION
# =============================================================================

def main_analysis():
    """Fonction principale d'analyse"""
    
    print("Début de l'analyse géospatiale de la sécheresse - BMK")
    print("=" * 60)
    
    # Paramètres
    shapefile_path = "beni_mellal_khenifra.geojson"
    
    # Vérification de l'existence du shapefile
    if not os.path.exists(shapefile_path):
        print(f"ERREUR: Le fichier {shapefile_path} est introuvable.")
        print("Veuillez vous assurer que le fichier geojson de la région existe.")
        return
    
    try:
        # 1. Collecte des données (optionnel - peut être fait séparément)
        if input("Télécharger les données CHIRPS? (y/n): ").lower() == 'y':
            print("Téléchargement des données CHIRPS...")
            download_chirps_data()
            decompress_chirps()
            clip_rasters_to_region("chirps_tifs", "chirps_beni_mellal", shapefile_path)
            
        if input("Télécharger les données MODIS NDVI? (y/n): ").lower() == 'y':
            print("Téléchargement des données MODIS NDVI...")
            # Code de téléchargement MODIS ici
            clip_rasters_to_region("ndvi_data", "ndvi_beni_mellal", shapefile_path, ".tiff")
        
        # 2. Extraction des moyennes
        print("\n=== EXTRACTION DES DONNÉES ===")
        
        if os.path.exists("chirps_beni_mellal"):
            print("Extraction des moyennes SPI...")
            spi_df = extract_mean_values("chirps_beni_mellal", "SPI", shapefile_path)
            
            # Calcul du SPI réel
            spi_df['mean_spi'] = calculate_spi(spi_df['mean_spi'])
            spi_df.to_csv("mean_spi_bmk.csv", index=False)
            print(f"SPI extrait: {len(spi_df)} observations")
        
        if os.path.exists("ndvi_beni_mellal"):
            print("Extraction des moyennes NDVI...")
            ndvi_df = extract_mean_values("ndvi_beni_mellal", "NDVI", shapefile_path)
            ndvi_df.to_csv("mean_ndvi_bmk.csv", index=False)
            print(f"NDVI extrait: {len(ndvi_df)} observations")
        
        # 3. Chargement des données existantes
        if os.path.exists("mean_spi_bmk.csv") and os.path.exists("mean_ndvi_bmk.csv"):
            spi_df = pd.read_csv("mean_spi_bmk.csv", parse_dates=['date'])
            ndvi_df = pd.read_csv("mean_ndvi_bmk.csv", parse_dates=['date'])
            
            # Conversion NDVI mensuel
            ndvi_df['month'] = ndvi_df['date'].dt.to_period('M')
            monthly_ndvi = ndvi_df.groupby('month')['mean_ndvi'].mean().reset_index()
            monthly_ndvi['date'] = monthly_ndvi['month'].dt.to_timestamp()
            monthly_ndvi.drop('month', axis=1, inplace=True)
            
            # Fusion des données
            merged_df = pd.merge(spi_df, monthly_ndvi, on='date', how='inner')
            merged_df = merged_df.dropna()
            
            print(f"\nDonnées fusionnées: {len(merged_df)} observations")
            print(f"Période: {merged_df['date'].min()} à {merged_df['date'].max()}")
            
            # 4. Analyse exploratoire
            print("\n=== ANALYSE EXPLORATOIRE ===")
            
            # Visualisations temporelles
            create_time_series_plots(merged_df, ['mean_spi', 'mean_ndvi'], 
                                   'Évolution temporelle BMK', 'timeseries_bmk.png')
            
            # Analyse saisonnière
            print("\nAnalyse saisonnière SPI:")
            monthly_spi, seasonal_spi = analyze_seasonal_patterns(merged_df, 'mean_spi')
            
            print("\nAnalyse saisonnière NDVI:")
            monthly_ndvi, seasonal_ndvi = analyze_seasonal_patterns(merged_df, 'mean_ndvi')
            
            # Corrélation
            print("\nAnalyse de corrélation:")
            corr_coef, p_value = correlation_analysis(merged_df)
            
            # 5. Détection des événements de sécheresse
            print("\n=== DÉTECTION DES ÉVÉNEMENTS DE SÉCHERESSE ===")
            drought_events = detect_drought_events(merged_df)
            print(f"Événements détectés: {len(drought_events)}")
            
            if len(drought_events) > 0:
                drought_events.to_csv("drought_events_bmk.csv", index=False)
                print(drought_events.head())
            
            # 6. Modélisation
            print("\n=== MODÉLISATION PRÉDICTIVE ===")
            
            if len(merged_df) > 50:  # Vérification suffisante de données
                print("Entraînement du modèle de classification...")
                drought_model, drought_scaler, feature_importance = train_drought_prediction_model(merged_df)
                
                print("\nEntraînement du modèle de prédiction NDVI...")
                ndvi_model, ndvi_scaler = train_ndvi_prediction_model(merged_df)
            
            # 7. Évaluation de la vulnérabilité
            print("\n=== ÉVALUATION DE LA VULNÉRABILITÉ ===")
            vulnerability_score, vulnerability_level = vulnerability_assessment(merged_df, drought_events)
            
            # 8. Rapport final
            print("\n=== GÉNÉRATION DU RAPPORT ===")
            generate_drought_report(merged_df, drought_events)
            
            # 9. Tableau de bord interactif
            if input("\nCréer un tableau de bord interactif? (y/n): ").lower() == 'y':
                dashboard = create_interactive_dashboard(merged_df, drought_events)
            
            # 10. Sauvegarde des résultats
            merged_df.to_csv("final_analysis_bmk.csv", index=False)
            
            # Résumé des fichiers générés
            print("\n=== FICHIERS GÉNÉRÉS ===")
            output_files = [
                "mean_spi_bmk.csv", "mean_ndvi_bmk.csv", "final_analysis_bmk.csv",
                "drought_events_bmk.csv", "timeseries_bmk.png"
            ]
            
            for file in output_files:
                if os.path.exists(file):
                    print(f"✓ {file}")
                else:
                    print(f"✗ {file} (non créé)")
        
        else:
            print("ERREUR: Fichiers de données SPI/NDVI introuvables.")
            print("Veuillez d'abord extraire les données ou les télécharger.")
    
    except Exception as e:
        print(f"Erreur durant l'analyse: {e}")
        import traceback
        traceback.print_exc()

def quick_analysis():
    """Analyse rapide avec données existantes"""
    
    try:
        # Chargement des données
        if os.path.exists("final_analysis_bmk.csv"):
            df = pd.read_csv("final_analysis_bmk.csv", parse_dates=['date'])
        else:
            print("Fichier d'analyse non trouvé. Exécutez d'abord main_analysis()")
            return
        
        print("=== ANALYSE RAPIDE - BMK ===")
        print(f"Données: {len(df)} observations")
        print(f"Période: {df['date'].min()} à {df['date'].max()}")
        
        # Statistiques rapides
        print(f"\nSPI - Moyenne: {df['mean_spi'].mean():.3f}, Std: {df['mean_spi'].std():.3f}")
        print(f"NDVI - Moyenne: {df['mean_ndvi'].mean():.3f}, Std: {df['mean_ndvi'].std():.3f}")
        
        # Corrélation
        corr = df[['mean_spi', 'mean_ndvi']].corr().iloc[0,1]
        print(f"Corrélation SPI-NDVI: {corr:.3f}")
        
        # Événements de sécheresse
        severe_drought = len(df[df['mean_spi'] < -1.5])
        print(f"Mois de sécheresse sévère: {severe_drought} ({(severe_drought/len(df)*100):.1f}%)")
        
        # Visualisation rapide
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        axes[0].plot(df['date'], df['mean_spi'], 'b-', label='SPI')
        axes[0].axhline(y=-1.0, color='orange', linestyle='--', alpha=0.7, label='Sécheresse modérée')
        axes[0].axhline(y=-1.5, color='red', linestyle='--', alpha=0.7, label='Sécheresse sévère')
        axes[0].set_ylabel('SPI')
        axes[0].set_title('Évolution du SPI - Béni Mellal-Khénifra')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(df['date'], df['mean_ndvi'], 'g-', label='NDVI')
        axes[1].set_ylabel('NDVI')
        axes[1].set_xlabel('Date')
        axes[1].set_title('Évolution du NDVI - Béni Mellal-Khénifra')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('quick_analysis_bmk.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Erreur: {e}")

# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    print("Analyse géospatiale de la sécheresse - Région Béni Mellal-Khénifra")
    print("Réalisé par : Adam Daoudi | INSEA")
    print("=" * 70)
    
    choice = input("""
Choisissez une option:
1. Analyse complète (main_analysis)
2. Analyse rapide (quick_analysis)
3. Quitter

Votre choix (1-3): """)
    
    if choice == "1":
        main_analysis()
    elif choice == "2":
        quick_analysis()
    elif choice == "3":
        print("Au revoir!")
    else:
        print("Choix invalide!")
        
print("Script terminé.")