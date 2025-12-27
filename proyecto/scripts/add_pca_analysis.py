#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para agregar análisis PCA espacial.

Implementa:
- Análisis de Componentes Principales (PCA)
- Reducción de dimensionalidad espacial
- Visualización de componentes
- Interpretación de patrones espaciales

Uso:
    python add_pca_analysis.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configuración
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUTS = BASE_DIR / "outputs" / "figures"

OUTPUTS.mkdir(parents=True, exist_ok=True)

print("="*70)
print("ANÁLISIS DE COMPONENTES PRINCIPALES ESPACIALES (PCA)")
print("="*70)

# ============================================================================
# 1. CARGAR DATOS
# ============================================================================
print("\n[1/4] Cargando datos espaciales...")

# Cargar grilla con todas las variables
grid_file = DATA_PROCESSED / "grid_with_topography.gpkg"

if not grid_file.exists():
    grid_file = DATA_PROCESSED / "prepared.gpkg"

if not grid_file.exists():
    print(f"  ✗ No se encontró archivo de grilla")
    sys.exit(1)

try:
    grid = gpd.read_file(grid_file)
    print(f"  ✓ Grilla cargada: {len(grid)} celdas")
except:
    import fiona
    layers = fiona.listlayers(grid_file)
    grid = gpd.read_file(grid_file, layer=layers[0])
    print(f"  ✓ Grilla cargada: {len(grid)} celdas")

# ============================================================================
# 2. PREPARAR VARIABLES PARA PCA
# ============================================================================
print("\n[2/4] Preparando variables para PCA...")

# Seleccionar variables numéricas (excluir geometría y IDs)
numeric_cols = grid.select_dtypes(include=[np.number]).columns.tolist()

# Filtrar columnas relevantes
exclude_patterns = ['cell_id', 'index', 'fid', 'id']
feature_cols = [col for col in numeric_cols 
                if not any(pattern in col.lower() for pattern in exclude_patterns)]

print(f"  → Variables seleccionadas: {len(feature_cols)}")
for col in feature_cols[:10]:  # Mostrar primeras 10
    print(f"     - {col}")
if len(feature_cols) > 10:
    print(f"     ... y {len(feature_cols) - 10} más")

# Extraer datos y manejar NaN
X = grid[feature_cols].copy()
X = X.fillna(X.mean())  # Imputar NaN con media

# Estandarizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"  ✓ Datos preparados: {X_scaled.shape}")

# ============================================================================
# 3. APLICAR PCA
# ============================================================================
print("\n[3/4] Aplicando PCA...")

# Determinar número óptimo de componentes
n_components = min(len(feature_cols), len(grid), 10)
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

print(f"  ✓ PCA aplicado: {n_components} componentes")
print(f"  → Varianza explicada acumulada:")
cumsum_var = np.cumsum(pca.explained_variance_ratio_)
for i, var in enumerate(cumsum_var[:5]):
    print(f"     PC{i+1}: {var*100:.2f}%")

# ============================================================================
# 4. VISUALIZACIONES
# ============================================================================
print("\n[4/4] Generando visualizaciones...")

# Scree Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Varianza explicada
axes[0].bar(range(1, n_components+1), pca.explained_variance_ratio_)
axes[0].plot(range(1, n_components+1), cumsum_var, 'ro-', linewidth=2)
axes[0].set_xlabel('Componente Principal')
axes[0].set_ylabel('Varianza Explicada')
axes[0].set_title('Scree Plot - Varianza por Componente')
axes[0].grid(True, alpha=0.3)

# Varianza acumulada
axes[1].plot(range(1, n_components+1), cumsum_var * 100, 'bo-', linewidth=2)
axes[1].axhline(y=80, color='r', linestyle='--', label='80% varianza')
axes[1].axhline(y=90, color='g', linestyle='--', label='90% varianza')
axes[1].set_xlabel('Número de Componentes')
axes[1].set_ylabel('Varianza Explicada Acumulada (%)')
axes[1].set_title('Varianza Acumulada')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
scree_path = OUTPUTS / "pca_scree_plot.png"
plt.savefig(scree_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Scree plot guardado: {scree_path.name}")

# Biplot (PC1 vs PC2)
fig, ax = plt.subplots(figsize=(12, 10))

# Scatter de observaciones
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                    c=range(len(X_pca)), cmap='viridis',
                    alpha=0.6, s=50)

# Vectores de variables (loadings)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
for i, feature in enumerate(feature_cols):
    if i < 15:  # Mostrar solo primeras 15 para claridad
        ax.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3,
                head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.5)
        ax.text(loadings[i, 0]*3.2, loadings[i, 1]*3.2, feature,
               fontsize=8, ha='center', va='center')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)')
ax.set_title('PCA Biplot - Componentes Principales 1 y 2\nIsla de Pascua',
            fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)

plt.colorbar(scatter, label='Índice de celda')
plt.tight_layout()
biplot_path = OUTPUTS / "pca_biplot.png"
plt.savefig(biplot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Biplot guardado: {biplot_path.name}")

# Mapa espacial de PC1
if 'geometry' in grid.columns:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # PC1
    grid_plot = grid.copy()
    grid_plot['PC1'] = X_pca[:, 0]
    grid_plot.plot(column='PC1', ax=axes[0], cmap='RdYlBu_r',
                  legend=True, edgecolor='black', linewidth=0.5)
    axes[0].set_title(f'Componente Principal 1\n({pca.explained_variance_ratio_[0]*100:.1f}% varianza)',
                     fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # PC2
    grid_plot['PC2'] = X_pca[:, 1]
    grid_plot.plot(column='PC2', ax=axes[1], cmap='RdYlBu_r',
                  legend=True, edgecolor='black', linewidth=0.5)
    axes[1].set_title(f'Componente Principal 2\n({pca.explained_variance_ratio_[1]*100:.1f}% varianza)',
                     fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    spatial_path = OUTPUTS / "pca_spatial_components.png"
    plt.savefig(spatial_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Mapas espaciales guardados: {spatial_path.name}")

# Loadings heatmap
fig, ax = plt.subplots(figsize=(10, max(8, len(feature_cols) * 0.3)))
loadings_df = pd.DataFrame(
    pca.components_[:5].T,
    columns=[f'PC{i+1}' for i in range(5)],
    index=feature_cols
)
sns.heatmap(loadings_df, annot=True, fmt='.2f', cmap='RdBu_r',
           center=0, ax=ax, cbar_kws={'label': 'Loading'})
ax.set_title('Loadings de Variables en Componentes Principales',
            fontsize=12, fontweight='bold')
plt.tight_layout()
loadings_path = OUTPUTS / "pca_loadings_heatmap.png"
plt.savefig(loadings_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Heatmap de loadings guardado: {loadings_path.name}")

# Guardar resultados
pca_results = {
    'explained_variance_ratio': pca.explained_variance_ratio_,
    'explained_variance': pca.explained_variance_,
    'components': pca.components_,
    'feature_names': feature_cols,
    'X_pca': X_pca
}

results_file = OUTPUTS.parent / "pca_results.pkl"
import pickle
with open(results_file, 'wb') as f:
    pickle.dump(pca_results, f)

print("\n" + "="*70)
print("ANÁLISIS PCA COMPLETADO")
print("="*70)
print(f"\nVarianza explicada por primeros 3 componentes: {cumsum_var[2]*100:.2f}%")
print(f"Componentes necesarios para 80% varianza: {np.argmax(cumsum_var >= 0.8) + 1}")
print(f"Componentes necesarios para 90% varianza: {np.argmax(cumsum_var >= 0.9) + 1}")
print("\nArchivos generados:")
print(f"  1. {scree_path}")
print(f"  2. {biplot_path}")
print(f"  3. {spatial_path}")
print(f"  4. {loadings_path}")
print(f"  5. {results_file}")
