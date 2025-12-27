#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo de Geoestadística para Análisis Territorial.

Funciones para:
- Cálculo de semivariogramas
- Interpolación (Kriging, IDW)
- Validación cruzada espacial
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
from typing import Tuple, Dict, Optional, Callable
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# SEMIVARIOGRAMAS
# ============================================================================

def calculate_empirical_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    n_lags: int = 15,
    max_lag: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcular semivariograma empírico.
    
    Args:
        coords: Coordenadas (n, 2)
        values: Valores de la variable (n,)
        n_lags: Número de clases de distancia
        max_lag: Distancia máxima (default: 50% del rango espacial)
    
    Returns:
        lag_centers: Centros de las clases de distancia
        semivariance: Semivarianza para cada lag
        n_pairs: Número de pares en cada lag
    """
    # Calcular matriz de distancias
    dist_matrix = cdist(coords, coords)
    
    # Máxima distancia
    if max_lag is None:
        max_lag = dist_matrix.max() * 0.5
    
    # Definir bins de distancia
    lag_edges = np.linspace(0, max_lag, n_lags + 1)
    lag_centers = (lag_edges[:-1] + lag_edges[1:]) / 2
    
    # Calcular diferencias al cuadrado
    n = len(values)
    semivariance = np.zeros(n_lags)
    n_pairs = np.zeros(n_lags, dtype=int)
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = dist_matrix[i, j]
            # Encontrar el bin correspondiente
            for k in range(n_lags):
                if lag_edges[k] <= dist < lag_edges[k + 1]:
                    semivariance[k] += (values[i] - values[j]) ** 2
                    n_pairs[k] += 1
                    break
    
    # Normalizar
    with np.errstate(divide='ignore', invalid='ignore'):
        semivariance = semivariance / (2 * n_pairs)
        semivariance = np.nan_to_num(semivariance)
    
    return lag_centers, semivariance, n_pairs


# Modelos teóricos de variograma
def spherical_model(h: np.ndarray, nugget: float, sill: float, range_: float) -> np.ndarray:
    """Modelo esférico de variograma."""
    gamma = np.where(
        h <= range_,
        nugget + (sill - nugget) * (1.5 * h / range_ - 0.5 * (h / range_) ** 3),
        sill
    )
    gamma = np.where(h == 0, 0, gamma)
    return gamma


def exponential_model(h: np.ndarray, nugget: float, sill: float, range_: float) -> np.ndarray:
    """Modelo exponencial de variograma."""
    gamma = nugget + (sill - nugget) * (1 - np.exp(-3 * h / range_))
    gamma = np.where(h == 0, 0, gamma)
    return gamma


def gaussian_model(h: np.ndarray, nugget: float, sill: float, range_: float) -> np.ndarray:
    """Modelo gaussiano de variograma."""
    gamma = nugget + (sill - nugget) * (1 - np.exp(-3 * (h / range_) ** 2))
    gamma = np.where(h == 0, 0, gamma)
    return gamma


VARIOGRAM_MODELS = {
    'spherical': spherical_model,
    'exponential': exponential_model,
    'gaussian': gaussian_model
}


def fit_variogram_model(
    lag_centers: np.ndarray,
    semivariance: np.ndarray,
    model_name: str = 'spherical'
) -> Dict:
    """
    Ajustar modelo teórico al semivariograma empírico.
    
    Args:
        lag_centers: Centros de las clases de distancia
        semivariance: Semivarianza empírica
        model_name: Nombre del modelo ('spherical', 'exponential', 'gaussian')
    
    Returns:
        Dict con parámetros del modelo ajustado
    """
    model_func = VARIOGRAM_MODELS[model_name]
    
    # Valores iniciales
    nugget_init = semivariance[0] if semivariance[0] > 0 else 0.1
    sill_init = semivariance.max()
    range_init = lag_centers[np.argmax(semivariance > 0.95 * sill_init)] if any(semivariance > 0.95 * sill_init) else lag_centers[-1] / 2
    
    try:
        popt, pcov = curve_fit(
            model_func,
            lag_centers,
            semivariance,
            p0=[nugget_init, sill_init, range_init],
            bounds=([0, 0, 0], [sill_init * 2, sill_init * 3, lag_centers.max() * 2]),
            maxfev=5000
        )
        
        # Calcular R² del ajuste
        predicted = model_func(lag_centers, *popt)
        ss_res = np.sum((semivariance - predicted) ** 2)
        ss_tot = np.sum((semivariance - np.mean(semivariance)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'model': model_name,
            'nugget': popt[0],
            'sill': popt[1],
            'range': popt[2],
            'r2': r2,
            'function': lambda h: model_func(h, *popt)
        }
    except Exception as e:
        return {
            'model': model_name,
            'nugget': nugget_init,
            'sill': sill_init,
            'range': range_init,
            'r2': 0,
            'error': str(e)
        }


# ============================================================================
# INTERPOLACIÓN
# ============================================================================

def idw_interpolation(
    known_coords: np.ndarray,
    known_values: np.ndarray,
    unknown_coords: np.ndarray,
    power: float = 2.0
) -> np.ndarray:
    """
    Interpolación por Distancia Inversa Ponderada (IDW).
    
    Args:
        known_coords: Coordenadas de puntos conocidos (n, 2)
        known_values: Valores conocidos (n,)
        unknown_coords: Coordenadas a interpolar (m, 2)
        power: Exponente de la distancia
    
    Returns:
        Valores interpolados (m,)
    """
    distances = cdist(unknown_coords, known_coords)
    
    # Evitar división por cero
    distances = np.where(distances == 0, 1e-10, distances)
    
    weights = 1.0 / (distances ** power)
    weights_sum = weights.sum(axis=1, keepdims=True)
    weights_normalized = weights / weights_sum
    
    interpolated = np.dot(weights_normalized, known_values)
    
    return interpolated


def ordinary_kriging(
    known_coords: np.ndarray,
    known_values: np.ndarray,
    unknown_coords: np.ndarray,
    variogram_func: Callable,
    nugget: float = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kriging Ordinario.
    
    Args:
        known_coords: Coordenadas conocidas (n, 2)
        known_values: Valores conocidos (n,)
        unknown_coords: Coordenadas a interpolar (m, 2)
        variogram_func: Función del variograma ajustado
        nugget: Valor del nugget
    
    Returns:
        predictions: Valores predichos (m,)
        variances: Varianzas de kriging (m,)
    """
    n = len(known_coords)
    m = len(unknown_coords)
    
    # Matriz de covarianzas entre puntos conocidos
    dist_known = cdist(known_coords, known_coords)
    gamma_known = variogram_func(dist_known)
    
    # Construir matriz de kriging
    K = np.zeros((n + 1, n + 1))
    K[:n, :n] = gamma_known
    K[n, :n] = 1
    K[:n, n] = 1
    K[n, n] = 0
    
    predictions = np.zeros(m)
    variances = np.zeros(m)
    
    try:
        K_inv = np.linalg.inv(K)
    except np.linalg.LinAlgError:
        K_inv = np.linalg.pinv(K)
    
    for i in range(m):
        # Vector de covarianzas entre punto desconocido y conocidos
        dist_to_known = cdist([unknown_coords[i]], known_coords)[0]
        gamma_unknown = variogram_func(dist_to_known)
        
        k = np.zeros(n + 1)
        k[:n] = gamma_unknown
        k[n] = 1
        
        # Pesos de kriging
        weights = K_inv @ k
        
        # Predicción
        predictions[i] = np.dot(weights[:n], known_values)
        
        # Varianza de kriging
        variances[i] = np.dot(weights, k)
    
    return predictions, variances


# ============================================================================
# VALIDACIÓN CRUZADA
# ============================================================================

def leave_one_out_cv(
    coords: np.ndarray,
    values: np.ndarray,
    method: str = 'kriging',
    variogram_params: Optional[Dict] = None,
    idw_power: float = 2.0
) -> Dict:
    """
    Validación cruzada leave-one-out.
    
    Args:
        coords: Coordenadas (n, 2)
        values: Valores (n,)
        method: 'kriging' o 'idw'
        variogram_params: Parámetros del variograma (para kriging)
        idw_power: Potencia para IDW
    
    Returns:
        Dict con métricas de error
    """
    n = len(values)
    predictions = np.zeros(n)
    
    for i in range(n):
        # Crear conjuntos train/test
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        
        train_coords = coords[mask]
        train_values = values[mask]
        test_coord = coords[i:i+1]
        
        if method == 'idw':
            pred = idw_interpolation(train_coords, train_values, test_coord, idw_power)
        elif method == 'kriging' and variogram_params:
            pred, _ = ordinary_kriging(
                train_coords, train_values, test_coord,
                variogram_params['function'],
                variogram_params.get('nugget', 0)
            )
        else:
            pred = idw_interpolation(train_coords, train_values, test_coord)
        
        predictions[i] = pred[0]
    
    # Calcular métricas
    errors = values - predictions
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    me = np.mean(errors)
    
    # R²
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((values - np.mean(values)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'method': method,
        'predictions': predictions,
        'errors': errors,
        'MAE': mae,
        'RMSE': rmse,
        'ME': me,
        'R2': r2
    }


def create_prediction_grid(
    boundary: gpd.GeoDataFrame,
    resolution: float = 100
) -> np.ndarray:
    """
    Crear grilla de predicción dentro de un polígono límite.
    
    Args:
        boundary: GeoDataFrame con el límite
        resolution: Resolución de la grilla en unidades del CRS
    
    Returns:
        Coordenadas de los puntos de la grilla (m, 2)
    """
    minx, miny, maxx, maxy = boundary.total_bounds
    
    x = np.arange(minx, maxx, resolution)
    y = np.arange(miny, maxy, resolution)
    xx, yy = np.meshgrid(x, y)
    
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Filtrar puntos dentro del límite
    from shapely.geometry import Point
    boundary_union = boundary.unary_union
    
    mask = np.array([boundary_union.contains(Point(p)) for p in grid_points])
    
    return grid_points[mask]


if __name__ == "__main__":
    # Test básico
    print("Módulo de geoestadística cargado correctamente")
    
    # Datos de ejemplo
    np.random.seed(42)
    n = 50
    coords = np.random.rand(n, 2) * 1000
    values = np.sin(coords[:, 0] / 200) + np.cos(coords[:, 1] / 200) + np.random.randn(n) * 0.5
    
    # Calcular variograma
    lags, gamma, npairs = calculate_empirical_variogram(coords, values, n_lags=10)
    print(f"Variograma calculado: {len(lags)} lags")
    
    # Ajustar modelo
    fit = fit_variogram_model(lags, gamma, 'exponential')
    print(f"Modelo ajustado: {fit['model']}, R² = {fit['r2']:.3f}")
