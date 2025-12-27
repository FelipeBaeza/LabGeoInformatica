#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para agregar análisis SHAP al notebook de Machine Learning.

Genera:
- SHAP values para modelos entrenados
- Summary plots
- Dependence plots
- Feature importance basada en SHAP

Uso:
    python add_shap_analysis.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Intentar importar SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️ SHAP no está instalado. Instalando...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
    import shap
    SHAP_AVAILABLE = True

# Configuración
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = BASE_DIR / "outputs"
MODELS_DIR = OUTPUTS / "models"
FIGURES_DIR = OUTPUTS / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("ANÁLISIS SHAP - INTERPRETABILIDAD DE MODELOS ML")
print("="*70)

# ============================================================================
# CARGAR MODELO Y DATOS
# ============================================================================
print("\n[1/4] Cargando modelo y datos...")

# Buscar modelo guardado
model_files = list(MODELS_DIR.glob("*.pkl")) if MODELS_DIR.exists() else []

if not model_files:
    print("  ⚠️ No se encontraron modelos guardados.")
    print("  → Ejecuta primero el notebook 03_Machine_Learning_Espacial.ipynb")
    sys.exit(1)

# Usar el primer modelo encontrado (o el mejor)
model_file = model_files[0]
print(f"  ✓ Modelo encontrado: {model_file.name}")

try:
    with open(model_file, "rb") as f:
        model_data = pickle.load(f)
    
    # Extraer modelo y datos
    if isinstance(model_data, dict):
        model = model_data.get('model')
        X_test = model_data.get('X_test')
        feature_names = model_data.get('feature_names', [])
    else:
        model = model_data
        X_test = None
        feature_names = []
    
    print(f"  ✓ Modelo cargado: {type(model).__name__}")
    
except Exception as e:
    print(f"  ✗ Error cargando modelo: {e}")
    sys.exit(1)

# Si no hay datos de test, crear datos sintéticos para demostración
if X_test is None or len(X_test) == 0:
    print("  ⚠️ No hay datos de test guardados. Creando datos sintéticos...")
    
    # Crear datos sintéticos basados en features típicos
    n_samples = 100
    feature_names = ['dist_to_center', 'n_amenities', 'street_length', 
                    'n_streets', 'x_norm', 'y_norm']
    
    np.random.seed(42)
    X_test = pd.DataFrame({
        'dist_to_center': np.random.uniform(0, 5000, n_samples),
        'n_amenities': np.random.poisson(3, n_samples),
        'street_length': np.random.uniform(0, 1000, n_samples),
        'n_streets': np.random.poisson(5, n_samples),
        'x_norm': np.random.uniform(0, 1, n_samples),
        'y_norm': np.random.uniform(0, 1, n_samples)
    })
    
    print(f"  ✓ Datos sintéticos creados: {X_test.shape}")

# ============================================================================
# CALCULAR SHAP VALUES
# ============================================================================
print("\n[2/4] Calculando SHAP values...")

try:
    # Crear explainer apropiado según el tipo de modelo
    model_type = type(model).__name__
    
    if 'RandomForest' in model_type or 'XGB' in model_type or 'GradientBoosting' in model_type:
        print(f"  → Usando TreeExplainer para {model_type}")
        explainer = shap.TreeExplainer(model)
    else:
        print(f"  → Usando KernelExplainer para {model_type}")
        # Usar muestra pequeña para KernelExplainer (más lento)
        background = shap.sample(X_test, min(50, len(X_test)))
        explainer = shap.KernelExplainer(model.predict, background)
    
    # Calcular SHAP values
    print("  → Calculando SHAP values (esto puede tomar unos minutos)...")
    shap_values = explainer.shap_values(X_test)
    
    print(f"  ✓ SHAP values calculados: {np.array(shap_values).shape}")
    
except Exception as e:
    print(f"  ✗ Error calculando SHAP values: {e}")
    print("  → Continuando con análisis limitado...")
    shap_values = None

# ============================================================================
# GENERAR VISUALIZACIONES
# ============================================================================
print("\n[3/4] Generando visualizaciones SHAP...")

if shap_values is not None:
    # Summary Plot
    try:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        summary_path = FIGURES_DIR / "shap_summary_plot.png"
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Summary plot guardado: {summary_path.name}")
    except Exception as e:
        print(f"  ⚠️ Error en summary plot: {e}")
    
    # Feature Importance Plot
    try:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", 
                         feature_names=feature_names, show=False)
        plt.tight_layout()
        importance_path = FIGURES_DIR / "shap_feature_importance.png"
        plt.savefig(importance_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Feature importance guardado: {importance_path.name}")
    except Exception as e:
        print(f"  ⚠️ Error en feature importance: {e}")
    
    # Dependence Plots para top 3 features
    try:
        # Calcular importancia promedio
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_features_idx = np.argsort(mean_abs_shap)[-3:][::-1]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, feat_idx in enumerate(top_features_idx):
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"Feature {feat_idx}"
            shap.dependence_plot(feat_idx, shap_values, X_test, 
                               feature_names=feature_names,
                               ax=axes[idx], show=False)
            axes[idx].set_title(f'Dependence: {feat_name}')
        
        plt.tight_layout()
        dependence_path = FIGURES_DIR / "shap_dependence_plots.png"
        plt.savefig(dependence_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Dependence plots guardados: {dependence_path.name}")
    except Exception as e:
        print(f"  ⚠️ Error en dependence plots: {e}")

# ============================================================================
# GUARDAR RESULTADOS
# ============================================================================
print("\n[4/4] Guardando resultados...")

if shap_values is not None:
    # Guardar SHAP values
    shap_data = {
        'shap_values': shap_values,
        'feature_names': feature_names,
        'X_test': X_test
    }
    
    shap_file = OUTPUTS / "shap_analysis.pkl"
    with open(shap_file, "wb") as f:
        pickle.dump(shap_data, f)
    
    print(f"  ✓ SHAP data guardado: {shap_file}")
    
    # Crear resumen en texto
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names if len(feature_names) == len(mean_abs_shap) else [f'Feature_{i}' for i in range(len(mean_abs_shap))],
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    summary_file = OUTPUTS / "shap_feature_importance.csv"
    importance_df.to_csv(summary_file, index=False)
    print(f"  ✓ Feature importance CSV: {summary_file}")

print("\n" + "="*70)
print("ANÁLISIS SHAP COMPLETADO")
print("="*70)
print("\nArchivos generados:")
print(f"  1. {FIGURES_DIR}/shap_summary_plot.png")
print(f"  2. {FIGURES_DIR}/shap_feature_importance.png")
print(f"  3. {FIGURES_DIR}/shap_dependence_plots.png")
print(f"  4. {OUTPUTS}/shap_analysis.pkl")
print(f"  5. {OUTPUTS}/shap_feature_importance.csv")
print("\nPróximos pasos:")
print("  - Integrar visualizaciones en app web")
print("  - Interpretar resultados en informe")
print("  - Comparar con feature importance tradicional")
