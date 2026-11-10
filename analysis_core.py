# Calculations extracted from Proyecto_Final.ipynb; numerical formulas kept unchanged.
import ast
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.base import clone
from sklearn.cluster import KMeans
RESOURCE_DIR = Path(__file__).resolve().parent / "Recursos"


# --- Original notebook cell 7 ---
Path_POBLACION_HISTORICA = RESOURCE_DIR / 'Población_Histórica_Saltillo.xlsx'
POBLACION_HISTORICA = pd.read_excel(Path_POBLACION_HISTORICA)
POBLACION_HISTORICA_COPIA = POBLACION_HISTORICA.copy()


# --- Original notebook cell 8 ---
Datos = POBLACION_HISTORICA_COPIA.copy()
Datos = Datos.sort_values('Año').reset_index(drop=True)
X = (Datos[['Año']].to_numpy(dtype=float) - 2000) / 10
y = Datos['Población'].to_numpy(dtype=float)

def entrenar_gradiente(X_train, y_train, tasa=0.1, iteraciones=5000):
    media_x = X_train.mean(axis=0)
    escala_x = X_train.std(axis=0)
    escala_x[escala_x == 0] = 1
    X_norm = (X_train - media_x) / escala_x
    escala_y = 1000000
    y_norm = y_train / escala_y
    X_bias = np.c_[np.ones(len(X_norm)), X_norm]
    theta = np.zeros(X_bias.shape[1])
    historial = []
    for _ in range(iteraciones):
        predicciones = X_bias @ theta
        errores = predicciones - y_norm
        costo = np.mean(errores ** 2) / 2
        historial.append(costo)
        gradiente = X_bias.T @ errores / len(y_norm)
        theta -= tasa * gradiente
    return {'theta': theta, 'media_x': media_x, 'escala_x': escala_x, 'escala_y': escala_y, 'historial': historial}

def predecir_gradiente(modelo, X_test):
    X_norm = (X_test - modelo['media_x']) / modelo['escala_x']
    X_bias = np.c_[np.ones(len(X_norm)), X_norm]
    return X_bias @ modelo['theta'] * modelo['escala_y']

def crear_modelo(grado, ridge=False, alpha=1.0):
    estimador = Ridge(alpha=alpha) if ridge else LinearRegression()
    return make_pipeline(PolynomialFeatures(degree=grado, include_bias=False), StandardScaler(), estimador)
Modelos = {'Lineal - Mínimos cuadrados': crear_modelo(1), 'Polinomial grado 2': crear_modelo(2), 'Polinomial grado 3': crear_modelo(3)}
Alphas = [0.01, 0.1, 1]

def seleccionar_alpha(X_train, y_train, grado):
    resultados = []
    for alpha in Alphas:
        errores = []
        for i in range(4, len(X_train)):
            modelo = crear_modelo(grado, ridge=True, alpha=alpha)
            modelo.fit(X_train[:i], y_train[:i])
            pred = modelo.predict(X_train[i:i + 1])
            errores.append(abs(y_train[i] - pred[0]))
        if errores:
            resultados.append((np.mean(errores), alpha))
    if not resultados:
        raise ValueError('No hay suficientes datos para seleccionar alpha.')
    return min(resultados)[1]
Anios_Prueba = [2000, 2005, 2010, 2015, 2020]
Nombres_Modelos = ['Lineal - Mínimos cuadrados', 'Lineal - Gradiente descendiente', 'Polinomial grado 2', 'Polinomial grado 3', 'Polinomial grado 2 - Ridge', 'Polinomial grado 3 - Ridge']
Predicciones_Validacion = {nombre: [] for nombre in Nombres_Modelos}
Reales_Validacion = []
for Anio in Anios_Prueba:
    train = Datos['Año'] < Anio
    test = Datos['Año'] == Anio
    X_train, y_train = (X[train], y[train])
    X_test, y_test = (X[test], y[test])
    if len(X_train) < 5 or len(X_test) != 1:
        raise ValueError(f'Datos insuficientes para evaluar {Anio}.')
    Reales_Validacion.append(y_test[0])
    for Nombre, Modelo in Modelos.items():
        modelo = clone(Modelo)
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)[0]
        Predicciones_Validacion[Nombre].append(pred)
    modelo_gd = entrenar_gradiente(X_train, y_train)
    pred_gd = predecir_gradiente(modelo_gd, X_test)[0]
    Predicciones_Validacion['Lineal - Gradiente descendiente'].append(pred_gd)
    for grado in [2, 3]:
        alpha = seleccionar_alpha(X_train, y_train, grado)
        modelo = crear_modelo(grado, ridge=True, alpha=alpha)
        modelo.fit(X_train, y_train)
        Nombre = f'Polinomial grado {grado} - Ridge'
        Predicciones_Validacion[Nombre].append(modelo.predict(X_test)[0])
Resultados = []
for Nombre in Nombres_Modelos:
    Predichos = Predicciones_Validacion[Nombre]
    MAE = mean_absolute_error(Reales_Validacion, Predichos)
    RMSE = np.sqrt(mean_squared_error(Reales_Validacion, Predichos))
    R2 = r2_score(Reales_Validacion, Predichos)
    Resultados.append({'Modelo': Nombre, 'MAE': MAE, 'RMSE': RMSE, 'R²': R2})
Resultados_Modelos = pd.DataFrame(Resultados).sort_values('MAE').reset_index(drop=True)
Mejor_Modelo = Resultados_Modelos.iloc[0]['Modelo']
Modelos_Finales = {}
for Nombre, Modelo in Modelos.items():
    modelo = clone(Modelo)
    modelo.fit(X, y)
    Modelos_Finales[Nombre] = modelo
Modelo_GD_Final = entrenar_gradiente(X, y)
Alphas_Finales = {}
for grado in [2, 3]:
    alpha = seleccionar_alpha(X, y, grado)
    modelo = crear_modelo(grado, ridge=True, alpha=alpha)
    modelo.fit(X, y)
    Nombre = f'Polinomial grado {grado} - Ridge'
    Modelos_Finales[Nombre] = modelo
    Alphas_Finales[Nombre] = alpha

def predecir_modelo(Nombre, X_nuevo):
    if Nombre == 'Lineal - Gradiente descendiente':
        return predecir_gradiente(Modelo_GD_Final, X_nuevo)
    return Modelos_Finales[Nombre].predict(X_nuevo)
X_2025 = np.array([[(2025 - 2000) / 10]])
Predicciones_2025 = {Nombre: float(predecir_modelo(Nombre, X_2025)[0]) for Nombre in Nombres_Modelos}
Tabla_Predicciones = pd.DataFrame({'Modelo': list(Predicciones_2025.keys()), 'Población estimada 2025': list(Predicciones_2025.values())})
Prediccion_Final = Predicciones_2025[Mejor_Modelo]


# --- Original notebook cell 14 ---
Path_SECTORES = RESOURCE_DIR / 'Diccionario_Sectores_Vespertinos.py'
with open(Path_SECTORES, encoding='utf-8') as archivo:
    codigo = ast.parse(archivo.read())
for nodo in codigo.body:
    if isinstance(nodo, ast.Assign):
        if any((isinstance(t, ast.Name) and t.id == 'SectoresVespertinos' for t in nodo.targets)):
            diccionario_original = ast.literal_eval(nodo.value)
            break
else:
    raise ValueError('No se encontró SectoresVespertinos')
SECTORES = {int(sector): list(dict.fromkeys(datos[0])) for sector, datos in diccionario_original.items()}


# --- Original notebook cell 15 ---
Path_ASENTAMIENTOS = RESOURCE_DIR / 'Colonias con Datos Sociodemográficos.xlsx'
ASENTAMIENTOS = pd.read_excel(Path_ASENTAMIENTOS)
ASENTAMIENTOS_COPIA = ASENTAMIENTOS.copy()
ASENTAMIENTOS_COPIA = ASENTAMIENTOS_COPIA.dropna(how='all').copy()
ASENTAMIENTOS_COPIA['CVEASEN'] = pd.to_numeric(ASENTAMIENTOS_COPIA['CVEASEN'], errors='coerce')
ASENTAMIENTOS_COPIA['POBLACIÓN TOTAL'] = pd.to_numeric(ASENTAMIENTOS_COPIA['POBLACIÓN TOTAL'], errors='coerce')
ASENTAMIENTOS_COPIA = ASENTAMIENTOS_COPIA.dropna(subset=['CVEASEN']).copy()
if ASENTAMIENTOS_COPIA['CVEASEN'].duplicated().any():
    raise ValueError('Existen claves CVEASEN duplicadas en el Excel.')
POBLACION_ASENTAMIENTOS = ASENTAMIENTOS_COPIA.set_index('CVEASEN')['POBLACIÓN TOTAL'].to_dict()
RELACIONES = pd.DataFrame([(sector, asentamiento) for sector, asentamientos in SECTORES.items() for asentamiento in asentamientos], columns=['Sector', 'CVEASEN']).drop_duplicates()
RELACIONES['Numero_Sectores'] = RELACIONES.groupby('CVEASEN')['Sector'].transform('nunique')
RELACIONES['Poblacion_Original'] = RELACIONES['CVEASEN'].map(POBLACION_ASENTAMIENTOS)
FALTANTES = RELACIONES[RELACIONES['Poblacion_Original'].isna()]['CVEASEN'].unique()
if len(FALTANTES) > 0:
    raise ValueError(f'Asentamientos sin población: {FALTANTES.tolist()}')
RELACIONES['Poblacion_Asignada'] = RELACIONES['Poblacion_Original'] / RELACIONES['Numero_Sectores']
POBLACION_SECTORES = RELACIONES.groupby('Sector')['Poblacion_Asignada'].sum().reset_index(name='Poblacion_Estimada')


# --- Original notebook cell 16 ---
Path_TONELAJES = RESOURCE_DIR / 'Todos_los_Reportes_de_Tonelaje.xlsx'
REPORTES_TONELAJES = pd.read_excel(Path_TONELAJES, sheet_name='Sheet1')
REPORTES_TONELAJES_COPIA = REPORTES_TONELAJES.copy()
REPORTES_TONELAJES_COPIA['Fecha'] = pd.to_datetime(REPORTES_TONELAJES_COPIA['Fecha'], errors='coerce')
REPORTES_TONELAJES_COPIA['PesoTotal_ton'] = pd.to_numeric(REPORTES_TONELAJES_COPIA['Peso Total'], errors='coerce')
REPORTES_TONELAJES_COPIA['Zona'] = REPORTES_TONELAJES_COPIA['Zona'].astype('string').str.strip().str.title()
REPORTES_TONELAJES_COPIA['Turno'] = REPORTES_TONELAJES_COPIA['Turno'].astype('string').str.strip().str.upper()
REPORTES_TONELAJES_COPIA['Sector'] = np.where(REPORTES_TONELAJES_COPIA['Zona'] == 'Norte', REPORTES_TONELAJES_COPIA['Zona Norte'], REPORTES_TONELAJES_COPIA['Zona Sur'])
REPORTES_TONELAJES_COPIA['Sector'] = pd.to_numeric(REPORTES_TONELAJES_COPIA['Sector'], errors='coerce')
REPORTES_2025 = REPORTES_TONELAJES_COPIA[(REPORTES_TONELAJES_COPIA['Fecha'].dt.year == 2025) & (REPORTES_TONELAJES_COPIA['Turno'] == 'V') & REPORTES_TONELAJES_COPIA['Zona'].isin(['Norte', 'Sur'])].copy()
REPORTES_LIMPIOS = REPORTES_2025.dropna(subset=['Fecha', 'Sector', 'PesoTotal_ton']).copy()
REPORTES_LIMPIOS = REPORTES_LIMPIOS[REPORTES_LIMPIOS['PesoTotal_ton'] >= 0].copy()
REPORTES_LIMPIOS['Sector'] = REPORTES_LIMPIOS['Sector'].astype(int)
REPORTES_LIMPIOS = REPORTES_LIMPIOS[REPORTES_LIMPIOS['Sector'].isin(SECTORES)].copy()
RESUMEN_SECTORES = REPORTES_LIMPIOS.groupby(['Zona', 'Sector']).agg(Toneladas_Recolectadas=('PesoTotal_ton', 'sum'), Dias_Con_Registro=('Fecha', 'nunique'), Registros=('Fecha', 'size')).reset_index()


# --- Original notebook cell 17 ---
if RESUMEN_SECTORES['Sector'].duplicated().any():
    raise ValueError('Hay sectores repetidos en RESUMEN_SECTORES. Revisa su zona de procedencia.')
DATOS_SECTORES = POBLACION_SECTORES.merge(RESUMEN_SECTORES, on='Sector', how='left', validate='one_to_one')
SECTORES_SIN_TONELAJE = DATOS_SECTORES.loc[DATOS_SECTORES['Toneladas_Recolectadas'].isna(), 'Sector'].tolist()
if SECTORES_SIN_TONELAJE:
    raise ValueError('Hay sectores sin registros de tonelaje. Revisa la lista antes de continuar.')
if DATOS_SECTORES['Poblacion_Estimada'].isna().any() or (DATOS_SECTORES['Poblacion_Estimada'] <= 0).any():
    raise ValueError('Hay sectores sin población válida.')
DATOS_SECTORES = DATOS_SECTORES.sort_values('Sector').reset_index(drop=True)


# --- Original notebook cell 18 ---
FECHA_INICIO = REPORTES_LIMPIOS['Fecha'].min().normalize()
FECHA_FIN = REPORTES_LIMPIOS['Fecha'].max().normalize()
DIAS_PERIODO = (FECHA_FIN - FECHA_INICIO).days + 1
DATOS_SECTORES['Kg_Hab_Dia'] = DATOS_SECTORES['Toneladas_Recolectadas'] * 1000 / (DATOS_SECTORES['Poblacion_Estimada'] * DIAS_PERIODO)
TONELADAS_TOTALES = DATOS_SECTORES['Toneladas_Recolectadas'].sum()
POBLACION_TOTAL = DATOS_SECTORES['Poblacion_Estimada'].sum()
TASA_GLOBAL = TONELADAS_TOTALES * 1000 / (POBLACION_TOTAL * DIAS_PERIODO)


# --- Original notebook cell 19 ---
DEMANDA_ASENTAMIENTOS = RELACIONES[['CVEASEN', 'Poblacion_Original']].drop_duplicates(subset=['CVEASEN']).copy()
DEMANDA_ASENTAMIENTOS['Demanda_Toneladas'] = DEMANDA_ASENTAMIENTOS['Poblacion_Original'] * TASA_GLOBAL * 2 / 1000
DEMANDA_ASENTAMIENTOS = DEMANDA_ASENTAMIENTOS.sort_values('CVEASEN').reset_index(drop=True)


# --- Original notebook cell 22 ---
ARCHIVOS_KML = list(RESOURCE_DIR.glob('*.kml'))
if len(ARCHIVOS_KML) != 1:
    raise ValueError(f'Se esperaba un único archivo KML en Recursos. Se encontraron: {[p.name for p in ARCHIVOS_KML]}')
Path_ASENTAMIENTOS_KML = ARCHIVOS_KML[0]
ASENTAMIENTOS_GEO = gpd.read_file(Path_ASENTAMIENTOS_KML)
ASENTAMIENTOS_GEO_COPIA = ASENTAMIENTOS_GEO.copy()


# --- Original notebook cell 23 ---
ASENTAMIENTOS_GEO_COPIA = ASENTAMIENTOS_GEO_COPIA[ASENTAMIENTOS_GEO_COPIA.geometry.notna() & ~ASENTAMIENTOS_GEO_COPIA.geometry.is_empty].copy()
TIPOS_VALIDOS = ['Polygon', 'MultiPolygon']
if not ASENTAMIENTOS_GEO_COPIA.geom_type.isin(TIPOS_VALIDOS).all():
    raise ValueError('El KML contiene geometrías que no son polígonos.')
if ASENTAMIENTOS_GEO_COPIA.crs is None:
    raise ValueError('El KML no tiene un sistema de coordenadas definido.')
CRS_PROYECTADO = ASENTAMIENTOS_GEO_COPIA.estimate_utm_crs()
ASENTAMIENTOS_PROYECTADOS = ASENTAMIENTOS_GEO_COPIA.to_crs(CRS_PROYECTADO)


# --- Original notebook cell 24 ---
CENTROIDES = ASENTAMIENTOS_PROYECTADOS.copy()
CENTROIDES['Centroide'] = CENTROIDES.geometry.centroid
CENTROIDES['Centroide_Dentro'] = [poligono.covers(centroide) for poligono, centroide in zip(CENTROIDES.geometry, CENTROIDES['Centroide'])]
CENTROIDES['Punto_Representativo'] = [centroide if dentro else poligono.representative_point() for poligono, centroide, dentro in zip(CENTROIDES.geometry, CENTROIDES['Centroide'], CENTROIDES['Centroide_Dentro'])]
CENTROIDES['Coordenada_X'] = CENTROIDES['Punto_Representativo'].apply(lambda punto: punto.x)
CENTROIDES['Coordenada_Y'] = CENTROIDES['Punto_Representativo'].apply(lambda punto: punto.y)


# --- Original notebook cell 26 ---
CENTROIDES['CVEASEN'] = pd.to_numeric(CENTROIDES['Name'].astype('string').str[-4:], errors='coerce')
if CENTROIDES['CVEASEN'].isna().any():
    raise ValueError('Existen asentamientos del KML sin identificador válido.')
if CENTROIDES['CVEASEN'].duplicated().any():
    raise ValueError('Existen claves CVEASEN duplicadas en el KML.')
ASENTAMIENTOS_CLUSTER = CENTROIDES.merge(DEMANDA_ASENTAMIENTOS[['CVEASEN', 'Poblacion_Original', 'Demanda_Toneladas']], on='CVEASEN', how='left', validate='one_to_one')
CON_DEMANDA = ASENTAMIENTOS_CLUSTER[ASENTAMIENTOS_CLUSTER['Demanda_Toneladas'].notna()].copy()
SIN_DEMANDA = ASENTAMIENTOS_CLUSTER[ASENTAMIENTOS_CLUSTER['Demanda_Toneladas'].isna()].copy()


# --- Original notebook cell 27 ---
DATOS_CLUSTER = ASENTAMIENTOS_CLUSTER[ASENTAMIENTOS_CLUSTER['Demanda_Toneladas'].notna()].copy()
if DATOS_CLUSTER[['Coordenada_X', 'Coordenada_Y', 'Demanda_Toneladas']].isna().any().any():
    raise ValueError('Existen asentamientos con información incompleta.')
if (DATOS_CLUSTER['Demanda_Toneladas'] < 0).any():
    raise ValueError('Existen asentamientos con demanda negativa.')
COORDENADAS_CLUSTER = DATOS_CLUSTER[['Coordenada_X', 'Coordenada_Y']].to_numpy()
PESOS_CLUSTER = DATOS_CLUSTER['Demanda_Toneladas'].to_numpy()


# --- Original notebook cell 30 ---
RESULTADOS_CLUSTER = []
MODELOS_CLUSTER = {}
for k in range(2, 6):
    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    etiquetas = modelo.fit_predict(COORDENADAS_CLUSTER, sample_weight=PESOS_CLUSTER)
    silueta = silhouette_score(COORDENADAS_CLUSTER, etiquetas)
    cargas = np.bincount(etiquetas, weights=PESOS_CLUSTER, minlength=k)
    carga_ideal = cargas.sum() / k
    desequilibrio = np.max(np.abs(cargas - carga_ideal)) / carga_ideal * 100
    RESULTADOS_CLUSTER.append({'K': k, 'Silhouette': silueta, 'Carga_Min_Ton': cargas.min(), 'Carga_Max_Ton': cargas.max(), 'Desviacion_Max_Pct': desequilibrio, 'Inercia_Ponderada': modelo.inertia_})
    MODELOS_CLUSTER[k] = modelo
RESULTADOS_CLUSTER = pd.DataFrame(RESULTADOS_CLUSTER)


# --- Original notebook cell 32 ---
MODELO_NORTE_SUR = MODELOS_CLUSTER[2]
ZONAS_CLUSTER = DATOS_CLUSTER.copy()
ZONAS_CLUSTER['Cluster'] = MODELO_NORTE_SUR.labels_
PROMEDIO_NORTE = ZONAS_CLUSTER.groupby('Cluster')['Coordenada_Y'].mean()
CLUSTER_NORTE = PROMEDIO_NORTE.idxmax()
CLUSTER_SUR = PROMEDIO_NORTE.idxmin()
ZONAS_CLUSTER['Zona_Propuesta'] = ZONAS_CLUSTER['Cluster'].map({CLUSTER_NORTE: 'Norte', CLUSTER_SUR: 'Sur'})
RESUMEN_ZONAS = ZONAS_CLUSTER.groupby('Zona_Propuesta').agg(Asentamientos=('CVEASEN', 'nunique'), Poblacion=('Poblacion_Original', 'sum'), Demanda_Toneladas=('Demanda_Toneladas', 'sum')).reset_index()
RESUMEN_ZONAS['Porcentaje_Demanda'] = RESUMEN_ZONAS['Demanda_Toneladas'] / RESUMEN_ZONAS['Demanda_Toneladas'].sum() * 100
CARGA_NORTE = RESUMEN_ZONAS.loc[RESUMEN_ZONAS['Zona_Propuesta'] == 'Norte', 'Demanda_Toneladas'].iloc[0]
CARGA_SUR = RESUMEN_ZONAS.loc[RESUMEN_ZONAS['Zona_Propuesta'] == 'Sur', 'Demanda_Toneladas'].iloc[0]
DIFERENCIA_TONELADAS = abs(CARGA_NORTE - CARGA_SUR)
DESEQUILIBRIO_ZONAS = DIFERENCIA_TONELADAS / RESUMEN_ZONAS['Demanda_Toneladas'].sum() * 100


# --- Original notebook cell 34 ---
def componentes_geograficos(geodatos):
    geodatos = geodatos.reset_index(drop=True).copy()
    geometrias = geodatos.geometry
    indice_espacial = geodatos.sindex
    vecinos = {i: set() for i in range(len(geodatos))}
    for i, geometria in enumerate(geometrias):
        candidatos = indice_espacial.query(geometria, predicate='intersects')
        for j in candidatos:
            j = int(j)
            if j <= i:
                continue
            frontera_compartida = geometria.boundary.intersection(geometrias.iloc[j].boundary)
            if frontera_compartida.length > 0.01:
                vecinos[i].add(j)
                vecinos[j].add(i)
    visitados = set()
    componentes = []
    for inicio in range(len(geodatos)):
        if inicio in visitados:
            continue
        pendientes = [inicio]
        visitados.add(inicio)
        componente = []
        while pendientes:
            actual = pendientes.pop()
            componente.append(actual)
            for vecino in vecinos[actual]:
                if vecino not in visitados:
                    visitados.add(vecino)
                    pendientes.append(vecino)
        componentes.append(componente)
    return sorted(componentes, key=len, reverse=True)
RESULTADOS_CONTINUIDAD = []
for zona in ['Norte', 'Sur']:
    subconjunto = ZONAS_CLUSTER[ZONAS_CLUSTER['Zona_Propuesta'] == zona].copy()
    componentes = componentes_geograficos(subconjunto)
    RESULTADOS_CONTINUIDAD.append({'Zona': zona, 'Componentes_Geograficos': len(componentes), 'Asentamientos_Componente_Principal': len(componentes[0]), 'Asentamientos_Aislados': len(subconjunto) - len(componentes[0])})
CONTINUIDAD_ZONAS = pd.DataFrame(RESULTADOS_CONTINUIDAD)
