#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo de Análisis Espacial.

Funciones para:
- Autocorrelación espacial (Moran's I)
- Análisis LISA (Local Indicators of Spatial Association)
- Hot Spot Analysis (Getis-Ord Gi*)
- Matrices de pesos espaciales
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from typing import Tuple, Dict, Optional, Union
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# MATRICES DE PESOS ESPACIALES
# ============================================================================

def create_queen_weights(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Crear matriz de pesos espaciales tipo Queen (contiguidad).
    
    Args:
        gdf: GeoDataFrame con geometrías
    
    Returns:
        Matriz de pesos (n, n)
    """
    n = len(gdf)
    W = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            if gdf.geometry.iloc[i].touches(gdf.geometry.iloc[j]) or \
               gdf.geometry.iloc[i].intersects(gdf.geometry.iloc[j]):
                W[i, j] = 1
                W[j, i] = 1
    
    # Row standardization
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    W_standardized = W / row_sums
    
    return W_standardized


def create_distance_weights(
    gdf: gpd.GeoDataFrame,
    threshold: Optional[float] = None,
    k: Optional[int] = None
) -> np.ndarray:
    """
    Crear matriz de pesos basada en distancia.
    
    Args:
        gdf: GeoDataFrame con geometrías
        threshold: Distancia umbral (si se especifica)
        k: Número de vecinos más cercanos (si se especifica)
    
    Returns:
        Matriz de pesos (n, n)
    """
    from scipy.spatial.distance import cdist
    
    # Obtener centroides
    centroids = np.column_stack([
        gdf.geometry.centroid.x,
        gdf.geometry.centroid.y
    ])
    
    n = len(centroids)
    dist_matrix = cdist(centroids, centroids)
    
    W = np.zeros((n, n))
    
    if k is not None:
        # K vecinos más cercanos
        for i in range(n):
            # Ordenar distancias (excluyendo self)
            sorted_idx = np.argsort(dist_matrix[i])
            neighbors = sorted_idx[1:k+1]  # Excluir el punto mismo
            W[i, neighbors] = 1
    elif threshold is not None:
        # Distancia umbral
        W = (dist_matrix > 0) & (dist_matrix <= threshold)
        W = W.astype(float)
    else:
        # Default: usar 20% de la distancia máxima
        threshold = dist_matrix.max() * 0.2
        W = (dist_matrix > 0) & (dist_matrix <= threshold)
        W = W.astype(float)
    
    # Row standardization
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    W_standardized = W / row_sums
    
    return W_standardized


# ============================================================================
# AUTOCORRELACIÓN ESPACIAL GLOBAL
# ============================================================================

def morans_i(
    values: np.ndarray,
    W: np.ndarray,
    permutations: int = 999
) -> Dict:
    """
    Calcular el índice de Moran I global.
    
    Args:
        values: Valores de la variable (n,)
        W: Matriz de pesos espaciales (n, n)
        permutations: Número de permutaciones para p-value
    
    Returns:
        Dict con I, E(I), Var(I), Z-score, p-value
    """
    n = len(values)
    
    # Estandarizar valores
    z = (values - np.mean(values))
    
    # Calcular Moran's I
    numerator = np.sum(W * np.outer(z, z))
    denominator = np.sum(z ** 2)
    
    # Suma de pesos
    S0 = W.sum()
    
    I = (n / S0) * (numerator / denominator) if denominator != 0 else 0
    
    # Valor esperado bajo hipótesis nula
    E_I = -1 / (n - 1)
    
    # Varianza (normalidad)
    S1 = 0.5 * np.sum((W + W.T) ** 2)
    S2 = np.sum(np.sum(W + W.T, axis=1) ** 2)
    
    A = n * ((n**2 - 3*n + 3) * S1 - n * S2 + 3 * S0**2)
    B = (n**2 - n) * S1 - 2*n * S2 + 6 * S0**2
    C = (n - 1) * (n - 2) * (n - 3) * S0**2
    
    # Kurtosis
    b2 = np.sum(z**4) / (np.sum(z**2)**2 / n)
    
    Var_I = (A - b2 * B) / C - E_I**2 if C != 0 else 0.01
    
    # Z-score
    Z = (I - E_I) / np.sqrt(Var_I) if Var_I > 0 else 0
    
    # P-value (two-tailed)
    p_norm = 2 * (1 - stats.norm.cdf(abs(Z)))
    
    # P-value por permutaciones
    if permutations > 0:
        I_perm = np.zeros(permutations)
        for p in range(permutations):
            z_perm = np.random.permutation(z)
            num_perm = np.sum(W * np.outer(z_perm, z_perm))
            I_perm[p] = (n / S0) * (num_perm / denominator) if denominator != 0 else 0
        
        p_sim = (np.sum(np.abs(I_perm) >= np.abs(I)) + 1) / (permutations + 1)
    else:
        p_sim = p_norm
    
    return {
        'I': I,
        'E_I': E_I,
        'Var_I': Var_I,
        'Z': Z,
        'p_value_norm': p_norm,
        'p_value_sim': p_sim,
        'significant': p_sim < 0.05
    }


# ============================================================================
# LISA (LOCAL INDICATORS OF SPATIAL ASSOCIATION)
# ============================================================================

def local_morans_i(
    values: np.ndarray,
    W: np.ndarray,
    permutations: int = 999
) -> Dict:
    """
    Calcular LISA (Local Moran's I).
    
    Args:
        values: Valores de la variable (n,)
        W: Matriz de pesos espaciales (n, n)
        permutations: Número de permutaciones para p-values
    
    Returns:
        Dict con Is, Zs, p-values, y clasificación de clusters
    """
    n = len(values)
    
    # Estandarizar valores
    z = (values - np.mean(values)) / np.std(values)
    
    # Calcular lag espacial
    lag = W @ z
    
    # Local Moran's I
    m2 = np.sum(z ** 2) / n
    I_local = (z * lag) / m2
    
    # Z-scores y p-values por permutaciones
    Z_local = np.zeros(n)
    p_local = np.zeros(n)
    
    for i in range(n):
        # Obtener vecinos
        neighbors = np.where(W[i] > 0)[0]
        
        if len(neighbors) == 0:
            Z_local[i] = 0
            p_local[i] = 1
            continue
        
        # Permutaciones
        I_perm = np.zeros(permutations)
        for p in range(permutations):
            z_perm = np.random.permutation(z)
            lag_perm = np.sum(W[i] * z_perm)
            I_perm[p] = (z[i] * lag_perm) / m2
        
        # Z-score
        mean_perm = np.mean(I_perm)
        std_perm = np.std(I_perm)
        Z_local[i] = (I_local[i] - mean_perm) / std_perm if std_perm > 0 else 0
        
        # P-value
        p_local[i] = (np.sum(np.abs(I_perm) >= np.abs(I_local[i])) + 1) / (permutations + 1)
    
    # Clasificación de clusters
    clusters = classify_lisa_clusters(z, lag, p_local, alpha=0.05)
    
    return {
        'I_local': I_local,
        'Z_local': Z_local,
        'p_local': p_local,
        'clusters': clusters,
        'z_values': z,
        'lag_values': lag
    }


def classify_lisa_clusters(
    z: np.ndarray,
    lag: np.ndarray,
    p_values: np.ndarray,
    alpha: float = 0.05
) -> np.ndarray:
    """
    Clasificar observaciones en clusters LISA.
    
    Categorías:
    - 0: No significativo
    - 1: High-High (Hot Spot)
    - 2: Low-Low (Cold Spot)
    - 3: High-Low (Outlier)
    - 4: Low-High (Outlier)
    """
    n = len(z)
    clusters = np.zeros(n, dtype=int)
    
    for i in range(n):
        if p_values[i] > alpha:
            clusters[i] = 0  # No significativo
        elif z[i] > 0 and lag[i] > 0:
            clusters[i] = 1  # High-High
        elif z[i] < 0 and lag[i] < 0:
            clusters[i] = 2  # Low-Low
        elif z[i] > 0 and lag[i] < 0:
            clusters[i] = 3  # High-Low
        else:
            clusters[i] = 4  # Low-High
    
    return clusters


LISA_LABELS = {
    0: 'No Significativo',
    1: 'High-High (Hot Spot)',
    2: 'Low-Low (Cold Spot)',
    3: 'High-Low (Outlier)',
    4: 'Low-High (Outlier)'
}

LISA_COLORS = {
    0: '#cccccc',  # Gris
    1: '#d7191c',  # Rojo
    2: '#2c7bb6',  # Azul
    3: '#fdae61',  # Naranja
    4: '#abd9e9'   # Celeste
}


# ============================================================================
# GETIS-ORD Gi*
# ============================================================================

def getis_ord_gi_star(
    values: np.ndarray,
    W: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcular estadístico Getis-Ord Gi* para hot spot analysis.
    
    Args:
        values: Valores de la variable (n,)
        W: Matriz de pesos espaciales binaria (n, n)
    
    Returns:
        gi_star: Valores Gi* (n,)
        z_scores: Z-scores (n,)
    """
    n = len(values)
    x_mean = np.mean(values)
    S = np.std(values)
    
    gi_star = np.zeros(n)
    z_scores = np.zeros(n)
    
    for i in range(n):
        # Suma ponderada
        numerator = np.sum(W[i] * values) - x_mean * np.sum(W[i])
        
        # Calcular denominador
        w_sum = np.sum(W[i])
        w_sq_sum = np.sum(W[i] ** 2)
        
        denominator = S * np.sqrt((n * w_sq_sum - w_sum**2) / (n - 1))
        
        gi_star[i] = numerator / denominator if denominator != 0 else 0
        z_scores[i] = gi_star[i]  # Gi* ya está estandarizado
    
    return gi_star, z_scores


def classify_hotspots(z_scores: np.ndarray) -> np.ndarray:
    """
    Clasificar observaciones según Z-scores en hot/cold spots.
    
    Categorías:
    - 0: No significativo
    - 1: Hot Spot 99% (z > 2.58)
    - 2: Hot Spot 95% (z > 1.96)
    - 3: Hot Spot 90% (z > 1.65)
    - -1: Cold Spot 99% (z < -2.58)
    - -2: Cold Spot 95% (z < -1.96)
    - -3: Cold Spot 90% (z < -1.65)
    """
    n = len(z_scores)
    classification = np.zeros(n, dtype=int)
    
    for i in range(n):
        z = z_scores[i]
        if z > 2.58:
            classification[i] = 1
        elif z > 1.96:
            classification[i] = 2
        elif z > 1.65:
            classification[i] = 3
        elif z < -2.58:
            classification[i] = -1
        elif z < -1.96:
            classification[i] = -2
        elif z < -1.65:
            classification[i] = -3
    
    return classification


HOTSPOT_LABELS = {
    1: 'Hot Spot (99%)',
    2: 'Hot Spot (95%)',
    3: 'Hot Spot (90%)',
    0: 'No Significativo',
    -3: 'Cold Spot (90%)',
    -2: 'Cold Spot (95%)',
    -1: 'Cold Spot (99%)'
}

HOTSPOT_COLORS = {
    1: '#b2182b',
    2: '#ef8a62',
    3: '#fddbc7',
    0: '#f7f7f7',
    -3: '#d1e5f0',
    -2: '#67a9cf',
    -1: '#2166ac'
}


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def spatial_lag(values: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Calcular el lag espacial de una variable."""
    return W @ values


def moran_scatterplot_data(values: np.ndarray, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preparar datos para Moran Scatterplot.
    
    Returns:
        z: Valores estandarizados
        lag: Lag espacial estandarizado
    """
    z = (values - np.mean(values)) / np.std(values)
    lag = W @ z
    return z, lag


if __name__ == "__main__":
    print("Módulo de análisis espacial cargado correctamente")
    
    # Test básico
    np.random.seed(42)
    n = 20
    values = np.random.rand(n) * 100
    
    # Crear matriz de pesos simple (todos conectados)
    W = np.ones((n, n)) - np.eye(n)
    W = W / W.sum(axis=1, keepdims=True)
    
    # Calcular Moran's I
    result = morans_i(values, W, permutations=99)
    print(f"Moran's I: {result['I']:.4f}")
    print(f"Z-score: {result['Z']:.4f}")
    print(f"P-value: {result['p_value_sim']:.4f}")
