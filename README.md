# RSU Saltillo — aplicación de Machine Learning

Visualizador del notebook **Proyecto_Final.ipynb** con Streamlit.

## Inicio local

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Publicar en Streamlit Community Cloud

1. Comprueba que puedes publicar los archivos en `Recursos/` (incluidos datos municipales).
2. Sube el contenido de esta carpeta a un repositorio en GitHub. La carpeta `Recursos` debe quedar junto a `app.py`.
3. En [share.streamlit.io](https://share.streamlit.io/), crea una aplicación con ese repositorio, rama principal y archivo de entrada `app.py`.
4. Deja el entorno en una versión moderna de Python compatible (por ejemplo, 3.11) y espera el proceso de instalación.
5. Si el despliegue falla, consulta los registros de la aplicación; comprueba en especial que la ruta de `Recursos/` y las dependencias geoespaciales estén presentes.

El archivo `analysis_core.py` recoge los cálculos del notebook sin las impresiones ni figuras, y la interfaz está implementada en `app.py`. La app muestra los **396 asentamientos vespertinos**, no los 466 asentamientos restantes en los clústeres.

### Precauciones

- Los mapas son agrupamientos exploratorios, **no rutas de recolección**.
- La demanda de dos días se obtiene extrapolando la tasa histórica de recolección vespertina; depende de la completitud de los registros.
- K-Means ponderado no impone continuidad territorial ni equilibrio de tonelaje.
- Revisa los permisos para publicar los archivos de datos antes de hacer el repositorio público.
