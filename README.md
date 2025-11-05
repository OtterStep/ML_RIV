# Proyecto: Preprocesamiento de Datasets con Streamlit

## Descripción

Este proyecto contiene la implementación paso a paso de preprocesamiento de tres datasets distintos utilizando **Python**, **pandas**, **scikit-learn**, **numpy** y **Streamlit**. Cada ejercicio está diseñado para mostrar el flujo completo de limpieza, codificación, escalado y división en conjuntos de entrenamiento y prueba, así como visualizaciones donde corresponde.

Los ejercicios incluidos son:

1. **Ejercicio 1: Titanic** – Preprocesamiento del dataset Titanic para predecir supervivencia.
2. **Ejercicio 2: Student Performance** – Preprocesamiento del dataset de rendimiento estudiantil para predecir la nota final (G3).
3. **Ejercicio 3: Iris** – Preprocesamiento del dataset Iris con estandarización y gráfico de dispersión por clase.

Todos los ejercicios están preparados para mostrar el dataset paso a paso en **Streamlit**, incluyendo información general, valores nulos, codificación de variables categóricas y escalado de variables numéricas.

---

## Estructura del Proyecto

```
mi_proyecto_streamlit/
│
├─ app.py                # Archivo principal que importa y ejecuta los ejercicios en pestañas
├─ ejercicio1.py         # Ejercicio 1: Titanic
├─ ejercicio2.py         # Ejercicio 2: Student Performance
├─ ejercicio3.py         # Ejercicio 3: Iris
├─ data/
│   ├─ Titanic-Dataset.csv
│   └─ student-mat.csv
├─ requirements.txt      # Librerías necesarias
└─ README.md             # Este archivo
```

---

## Instalación

1. **Clonar el repositorio**

```bash
git clone <URL_DEL_REPOSITORIO>
cd mi_proyecto_streamlit
```

2. **Crear un entorno virtual (opcional pero recomendado)**

```bash
python -m venv venv
```

3. **Activar el entorno virtual**

* Windows:

```bash
venv\Scripts\activate
```

* Linux/macOS:

```bash
source venv/bin/activate
```

4. **Instalar dependencias**

```bash
pip install -r requirements.txt
```

---

## Contenido de `requirements.txt`

```text
numpy
pandas
scikit-learn
matplotlib
streamlit
```

---

## Uso

Para ejecutar la aplicación principal con Streamlit:

```bash
streamlit run app.py
```

Esto abrirá la aplicación en tu navegador con una interfaz de pestañas para cada ejercicio:

* **Ejercicio 1:** Preprocesamiento Titanic
* **Ejercicio 2:** Preprocesamiento Student Performance
* **Ejercicio 3:** Preprocesamiento Iris

Cada pestaña muestra:

* Dataset original y transformaciones paso a paso
* Información general del dataset (`info`, `describe`, nulos)
* Tratamiento de valores faltantes
* Codificación de variables categóricas
* Escalado de variables numéricas
* División en entrenamiento y prueba
* Visualizaciones (gráfico de dispersión en Iris y correlación en Student Performance)

---

## Notas

* Asegúrate de tener los archivos **Titanic-Dataset.csv** y **student-mat.csv** en la carpeta `data/`.
* Los datasets se procesan de manera que cualquier valor nulo se rellena automáticamente y se eliminan duplicados o inconsistencias según las instrucciones de cada ejercicio.
* En los ejercicios 1 y 2 se aplica **One Hot Encoding** a las variables categóricas, y las variables numéricas se escalan con **StandardScaler**.
* En el ejercicio 2 se realiza un análisis de correlación entre las notas G1, G2 y G3.

---

## Contribuciones

Si deseas contribuir:

1. Haz un fork del repositorio.
2. Crea una rama con tus mejoras: `git checkout -b mi_rama`.
3. Realiza tus cambios y haz commit: `git commit -am 'Agregué nuevas funciones'`.
4. Envía tus cambios al repositorio remoto: `git push origin mi_rama`.
5. Abre un Pull Request.

---

## Licencia

Este proyecto es de **uso educativo y personal**.
No se permite uso comercial sin autorización.