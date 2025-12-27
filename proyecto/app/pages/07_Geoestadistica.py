"""
Página de Geoestadística - Semivariogramas y Kriging
Dashboard interactivo para análisis geoestadístico
"""
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box, Point
import os
from sqlalchemy import create_engine
import sys

# Agregar path de scripts
sys.path.append(str(__file__).replace('/app/pages/07_Geoestadistica.py', '/scripts'))

# Configuracion
st.set_page_config(page_title="Geoestadistica", layout="wide")
st.title("Analisis Geoestadistico")
st.markdown("---")

CRS_UTM = 'EPSG:32719'

DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '55432'),
    'database': os.getenv('POSTGRES_DB', 'geodatabase'),
    'user': os.getenv('POSTGRES_USER', 'geouser'),
    'password': os.getenv('POSTGRES_PASSWORD', 'geopass123'),
}


def get_engine():
    return create_engine(
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )


# ============================================================================
# FUNCIONES DE GEOESTADÍSTICA (incluidas para evitar dependencias)
# ============================================================================

from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit


def calculate_empirical_variogram(coords, values, n_lags=15, max_lag=None):
    """Calcular semivariograma empírico."""
    dist_matrix = cdist(coords, coords)
    
    if max_lag is None:
        max_lag = dist_matrix.max() * 0.5
    
    lag_edges = np.linspace(0, max_lag, n_lags + 1)
    lag_centers = (lag_edges[:-1] + lag_edges[1:]) / 2
    
    n = len(values)
    semivariance = np.zeros(n_lags)
    n_pairs = np.zeros(n_lags, dtype=int)
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = dist_matrix[i, j]
            for k in range(n_lags):
                if lag_edges[k] <= dist < lag_edges[k + 1]:
                    semivariance[k] += (values[i] - values[j]) ** 2
                    n_pairs[k] += 1
                    break
    
    with np.errstate(divide='ignore', invalid='ignore'):
        semivariance = semivariance / (2 * n_pairs)
        semivariance = np.nan_to_num(semivariance)
    
    return lag_centers, semivariance, n_pairs


def spherical_model(h, nugget, sill, range_):
    """Modelo esférico de variograma."""
    gamma = np.where(
        h <= range_,
        nugget + (sill - nugget) * (1.5 * h / range_ - 0.5 * (h / range_) ** 3),
        sill
    )
    gamma = np.where(h == 0, 0, gamma)
    return gamma


def exponential_model(h, nugget, sill, range_):
    """Modelo exponencial de variograma."""
    gamma = nugget + (sill - nugget) * (1 - np.exp(-3 * h / range_))
    gamma = np.where(h == 0, 0, gamma)
    return gamma


def gaussian_model(h, nugget, sill, range_):
    """Modelo gaussiano de variograma."""
    gamma = nugget + (sill - nugget) * (1 - np.exp(-3 * (h / range_) ** 2))
    gamma = np.where(h == 0, 0, gamma)
    return gamma


VARIOGRAM_MODELS = {
    'Esférico': spherical_model,
    'Exponencial': exponential_model,
    'Gaussiano': gaussian_model
}


def fit_variogram_model(lag_centers, semivariance, model_name='Esférico'):
    """Ajustar modelo teórico al semivariograma empírico."""
    model_func = VARIOGRAM_MODELS[model_name]
    
    nugget_init = max(semivariance[0], 0.01)
    sill_init = max(semivariance.max(), 0.01)
    range_init = lag_centers[-1] / 2
    
    try:
        popt, _ = curve_fit(
            model_func,
            lag_centers,
            semivariance,
            p0=[nugget_init, sill_init, range_init],
            bounds=([0, 0, 1], [sill_init * 2, sill_init * 3, lag_centers.max() * 2]),
            maxfev=5000
        )
        
        predicted = model_func(lag_centers, *popt)
        ss_res = np.sum((semivariance - predicted) ** 2)
        ss_tot = np.sum((semivariance - np.mean(semivariance)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'model': model_name,
            'nugget': popt[0],
            'sill': popt[1],
            'range': popt[2],
            'r2': max(0, r2),
            'function': lambda h, p=popt: model_func(h, *p)
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


def idw_interpolation(known_coords, known_values, unknown_coords, power=2.0):
    """Interpolación IDW."""
    distances = cdist(unknown_coords, known_coords)
    distances = np.where(distances == 0, 1e-10, distances)
    weights = 1.0 / (distances ** power)
    weights_sum = weights.sum(axis=1, keepdims=True)
    weights_normalized = weights / weights_sum
    return np.dot(weights_normalized, known_values)


def ordinary_kriging(known_coords, known_values, unknown_coords, variogram_func, nugget=0):
    """Kriging Ordinario simplificado."""
    n = len(known_coords)
    m = len(unknown_coords)
    
    dist_known = cdist(known_coords, known_coords)
    gamma_known = variogram_func(dist_known)
    
    K = np.zeros((n + 1, n + 1))
    K[:n, :n] = gamma_known
    K[n, :n] = 1
    K[:n, n] = 1
    K[n, n] = 0
    
    predictions = np.zeros(m)
    variances = np.zeros(m)
    
    try:
        K_inv = np.linalg.inv(K)
    except:
        K_inv = np.linalg.pinv(K)
    
    for i in range(m):
        dist_to_known = cdist([unknown_coords[i]], known_coords)[0]
        gamma_unknown = variogram_func(dist_to_known)
        
        k = np.zeros(n + 1)
        k[:n] = gamma_unknown
        k[n] = 1
        
        weights = K_inv @ k
        predictions[i] = np.dot(weights[:n], known_values)
        variances[i] = max(0, np.dot(weights, k))
    
    return predictions, variances


# ============================================================================
# CARGAR DATOS
# ============================================================================

@st.cache_data
def load_data():
    """Cargar datos desde PostGIS."""
    data = {}
    tables = {
        'boundary': 'limite_administrativa',
        'buildings': 'area_construcciones',
        'amenities': 'punto_interes',
    }
    
    engine = get_engine()
    for key, table in tables.items():
        try:
            data[key] = gpd.read_postgis(
                f"SELECT * FROM geoanalisis.{table}", engine, geom_col='geometry'
            )
        except Exception as e:
            st.warning(f"Error cargando {table}: {e}")
    
    return data


@st.cache_data
def create_analysis_grid(_boundary_gdf, cell_size=200):
    """Crear grilla para análisis."""
    boundary_utm = _boundary_gdf.to_crs(CRS_UTM)
    minx, miny, maxx, maxy = boundary_utm.total_bounds
    
    cells = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cells.append(box(x, y, x + cell_size, y + cell_size))
            y += cell_size
        x += cell_size
    
    grid = gpd.GeoDataFrame(geometry=cells, crs=CRS_UTM)
    grid = grid[grid.intersects(boundary_utm.unary_union)].reset_index(drop=True)
    grid['cell_id'] = range(len(grid))
    grid['centroid_x'] = grid.geometry.centroid.x
    grid['centroid_y'] = grid.geometry.centroid.y
    
    return grid


@st.cache_data
def count_features_in_grid(_grid, _features_gdf):
    """Contar features por celda usando spatial join."""
    grid = _grid.copy()
    features = _features_gdf.to_crs(CRS_UTM)
    
    if 'Polygon' in str(features.geometry.geom_type.iloc[0]):
        features = features.copy()
        features['geometry'] = features.geometry.centroid
    
    joined = gpd.sjoin(features.reset_index(), grid[['cell_id', 'geometry']], 
                       how='inner', predicate='within')
    counts = joined.groupby('cell_id').size().reset_index(name='count')
    
    grid = grid.merge(counts, on='cell_id', how='left')
    grid['count'] = grid['count'].fillna(0).astype(int)
    
    return grid


# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

try:
    data = load_data()
    
    if not data or 'boundary' not in data:
        st.error("No se encontraron datos. Ejecute primero el script de descarga.")
        st.stop()
    
    # Sidebar
    st.sidebar.header("Configuracion")
    
    cell_size = st.sidebar.slider("Tamaño de celda (m)", 100, 500, 200, 50)
    
    analysis_layer = st.sidebar.selectbox(
        "Capa para análisis:",
        [k for k in data.keys() if k != 'boundary'],
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Semivariograma")
    
    n_lags = st.sidebar.slider("Número de lags", 8, 25, 15)
    model_type = st.sidebar.selectbox("Modelo teórico:", list(VARIOGRAM_MODELS.keys()))
    
    # Crear grilla y calcular variable
    with st.spinner("Creando grilla de análisis..."):
        grid = create_analysis_grid(data['boundary'], cell_size)
        grid = count_features_in_grid(grid, data[analysis_layer])
    
    # Filtrar celdas con datos
    grid_with_data = grid[grid['count'] > 0].copy().reset_index(drop=True)
    
    if len(grid_with_data) < 10:
        st.warning("No hay suficientes celdas con datos para análisis geoestadístico.")
        st.stop()
    
    # Preparar datos para variograma
    coords = np.column_stack([grid_with_data['centroid_x'], grid_with_data['centroid_y']])
    values = grid_with_data['count'].values.astype(float)
    
    # Tabs principales
    tab1, tab2, tab3 = st.tabs(["Semivariograma", "Interpolacion", "Validacion"])
    
    with tab1:
        st.subheader("Análisis de Semivariograma")
        
        # Calcular semivariograma
        with st.spinner("Calculando semivariograma..."):
            lags, gamma, n_pairs = calculate_empirical_variogram(coords, values, n_lags=n_lags)
            fitted = fit_variogram_model(lags, gamma, model_type)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Semivariograma Experimental y Modelo Ajustado**")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Puntos empíricos
            ax.scatter(lags, gamma, c='steelblue', s=80, edgecolor='white', 
                      linewidth=1.5, label='Empírico', zorder=3)
            
            # Modelo ajustado
            h_smooth = np.linspace(0, lags.max(), 100)
            if 'function' in fitted:
                gamma_model = fitted['function'](h_smooth)
                ax.plot(h_smooth, gamma_model, 'r-', linewidth=2.5, 
                       label=f"{model_type} (R²={fitted['r2']:.3f})")
            
            # Líneas de referencia
            if 'sill' in fitted:
                ax.axhline(y=fitted['sill'], color='gray', linestyle='--', 
                          alpha=0.5, label=f"Sill = {fitted['sill']:.2f}")
            if 'range' in fitted:
                ax.axvline(x=fitted['range'], color='green', linestyle=':', 
                          alpha=0.5, label=f"Range = {fitted['range']:.0f}m")
            
            ax.set_xlabel('Distancia (m)', fontsize=12)
            ax.set_ylabel('Semivarianza', fontsize=12)
            ax.set_title(f'Semivariograma - Densidad de {analysis_layer}', fontsize=14, fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, lags.max() * 1.05)
            ax.set_ylim(0, gamma.max() * 1.2)
            
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            st.markdown("**Parámetros del Modelo**")
            
            st.metric("Modelo", model_type)
            st.metric("Nugget", f"{fitted['nugget']:.2f}")
            st.metric("Sill", f"{fitted['sill']:.2f}")
            st.metric("Range", f"{fitted['range']:.0f} m")
            st.metric("R² del ajuste", f"{fitted['r2']:.4f}")
            
            st.markdown("---")
            st.markdown("**Interpretación**")
            st.info(f"""
            - **Nugget**: Variabilidad a distancia 0 (error o micro-escala)
            - **Sill**: Varianza máxima alcanzada
            - **Range**: Distancia donde se alcanza el sill (~{fitted['range']:.0f}m)
            """)
        
        # Tabla de lags
        with st.expander("Ver datos del semivariograma"):
            vario_df = pd.DataFrame({
                'Lag (m)': lags.round(0),
                'Semivarianza': gamma.round(4),
                'N° pares': n_pairs
            })
            st.dataframe(vario_df, use_container_width=True)
    
    with tab2:
        st.subheader("Interpolación Espacial")
        
        method = st.radio(
            "Método de interpolación:",
            ["IDW (Inverse Distance Weighting)", "Kriging Ordinario"],
            horizontal=True
        )
        
        if method == "IDW (Inverse Distance Weighting)":
            power = st.slider("Potencia IDW", 1.0, 4.0, 2.0, 0.5)
        
        if st.button("Ejecutar Interpolacion", type="primary"):
            with st.spinner("Realizando interpolación..."):
                # Crear grilla de predicción
                pred_grid = create_analysis_grid(data['boundary'], cell_size)
                pred_coords = np.column_stack([
                    pred_grid.geometry.centroid.x, 
                    pred_grid.geometry.centroid.y
                ])
                
                if "IDW" in method:
                    predictions = idw_interpolation(coords, values, pred_coords, power)
                    pred_grid['prediction'] = predictions
                    pred_grid['variance'] = np.zeros(len(predictions))
                else:
                    if 'function' in fitted:
                        predictions, variances = ordinary_kriging(
                            coords, values, pred_coords, 
                            fitted['function'], fitted['nugget']
                        )
                        pred_grid['prediction'] = predictions
                        pred_grid['variance'] = variances
                    else:
                        st.error("Error en el variograma ajustado. Intente con otro modelo.")
                        st.stop()
                
                st.session_state['interpolation_result'] = pred_grid
                st.success("Interpolación completada!")
        
        if 'interpolation_result' in st.session_state:
            result = st.session_state['interpolation_result']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Predicción**")
                fig, ax = plt.subplots(figsize=(8, 8))
                data['boundary'].to_crs(CRS_UTM).plot(ax=ax, facecolor='none', 
                                                       edgecolor='black', linewidth=2)
                result.plot(column='prediction', ax=ax, cmap='YlOrRd', legend=True,
                           legend_kwds={'label': 'Predicción'})
                ax.set_title('Valores Predichos', fontweight='bold')
                ax.set_axis_off()
                st.pyplot(fig)
                plt.close(fig)
            
            with col2:
                if 'variance' in result.columns and result['variance'].sum() > 0:
                    st.markdown("**Incertidumbre (Varianza Kriging)**")
                    fig, ax = plt.subplots(figsize=(8, 8))
                    data['boundary'].to_crs(CRS_UTM).plot(ax=ax, facecolor='none', 
                                                           edgecolor='black', linewidth=2)
                    result.plot(column='variance', ax=ax, cmap='Blues', legend=True,
                               legend_kwds={'label': 'Varianza'})
                    ax.set_title('Varianza de Kriging', fontweight='bold')
                    ax.set_axis_off()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.markdown("**Valores Observados**")
                    fig, ax = plt.subplots(figsize=(8, 8))
                    data['boundary'].to_crs(CRS_UTM).plot(ax=ax, facecolor='none', 
                                                           edgecolor='black', linewidth=2)
                    grid_with_data.plot(column='count', ax=ax, cmap='YlOrRd', legend=True,
                                       legend_kwds={'label': 'Observado'})
                    ax.set_title('Valores Observados', fontweight='bold')
                    ax.set_axis_off()
                    st.pyplot(fig)
                    plt.close(fig)
            
            # Estadísticas
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Media predicha", f"{result['prediction'].mean():.2f}")
            with col2:
                st.metric("Mínimo", f"{result['prediction'].min():.2f}")
            with col3:
                st.metric("Máximo", f"{result['prediction'].max():.2f}")
            with col4:
                st.metric("Celdas interpoladas", len(result))
    
    with tab3:
        st.subheader("Validación Cruzada")
        
        st.markdown("""
        La validación cruzada **leave-one-out** evalúa el rendimiento de la interpolación:
        cada punto se predice usando los demás puntos como referencia.
        """)
        
        if st.button("Ejecutar Validacion (Leave-One-Out)", type="primary"):
            with st.spinner("Ejecutando validación cruzada..."):
                n = len(values)
                
                # IDW validation
                idw_predictions = np.zeros(n)
                for i in range(n):
                    mask = np.ones(n, dtype=bool)
                    mask[i] = False
                    idw_predictions[i] = idw_interpolation(
                        coords[mask], values[mask], coords[i:i+1], power=2.0
                    )[0]
                
                idw_errors = values - idw_predictions
                idw_mae = np.mean(np.abs(idw_errors))
                idw_rmse = np.sqrt(np.mean(idw_errors ** 2))
                ss_res = np.sum(idw_errors ** 2)
                ss_tot = np.sum((values - np.mean(values)) ** 2)
                idw_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Kriging validation (si hay función)
                if 'function' in fitted:
                    krig_predictions = np.zeros(n)
                    for i in range(n):
                        mask = np.ones(n, dtype=bool)
                        mask[i] = False
                        pred, _ = ordinary_kriging(
                            coords[mask], values[mask], coords[i:i+1],
                            fitted['function'], fitted['nugget']
                        )
                        krig_predictions[i] = pred[0]
                    
                    krig_errors = values - krig_predictions
                    krig_mae = np.mean(np.abs(krig_errors))
                    krig_rmse = np.sqrt(np.mean(krig_errors ** 2))
                    ss_res_k = np.sum(krig_errors ** 2)
                    krig_r2 = 1 - (ss_res_k / ss_tot) if ss_tot > 0 else 0
                else:
                    krig_mae = krig_rmse = krig_r2 = 0
                
                st.session_state['cv_results'] = {
                    'idw': {'MAE': idw_mae, 'RMSE': idw_rmse, 'R2': idw_r2, 'pred': idw_predictions},
                    'krig': {'MAE': krig_mae, 'RMSE': krig_rmse, 'R2': krig_r2}
                }
                st.success("Validación completada!")
        
        if 'cv_results' in st.session_state:
            cv = st.session_state['cv_results']
            
            st.markdown("### Comparación de Métodos")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**IDW (Inverse Distance Weighting)**")
                st.metric("MAE", f"{cv['idw']['MAE']:.3f}")
                st.metric("RMSE", f"{cv['idw']['RMSE']:.3f}")
                st.metric("R²", f"{cv['idw']['R2']:.4f}")
            
            with col2:
                st.markdown("**Kriging Ordinario**")
                st.metric("MAE", f"{cv['krig']['MAE']:.3f}")
                st.metric("RMSE", f"{cv['krig']['RMSE']:.3f}")
                st.metric("R²", f"{cv['krig']['R2']:.4f}")
            
            # Gráfico de dispersión
            st.markdown("---")
            st.markdown("**Real vs Predicho (IDW)**")
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(values, cv['idw']['pred'], alpha=0.6, c='steelblue', s=40)
            max_val = max(values.max(), cv['idw']['pred'].max())
            ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Línea perfecta')
            ax.set_xlabel('Valor Real')
            ax.set_ylabel('Valor Predicho')
            ax.set_title('Validación Cruzada Leave-One-Out')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
            
            # Interpretación
            best_method = "Kriging" if cv['krig']['R2'] > cv['idw']['R2'] else "IDW"
            st.info(f"""
            **Interpretación**: El método **{best_method}** presenta mejor desempeño según R².
            - Un R² cercano a 1 indica buena predicción
            - RMSE menor indica errores más pequeños
            """)

except Exception as e:
    st.error(f"Error: {e}")
    import traceback
    st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.caption("Análisis Geoestadístico - Isla de Pascua | Laboratorio Integrador 2025")
