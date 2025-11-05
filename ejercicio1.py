import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


def preprocesamiento_titanic():
    st.title("🚢 Ejercicio 1: Preprocesamiento del Dataset Titanic (Versión completa paso a paso)")

    # ============================================
    # 1️⃣ CARGA DEL DATASET
    # ============================================
    st.subheader("1️⃣ Cargar el dataset con pandas")
    st.write("Leemos el archivo `Titanic-Dataset.csv` desde la carpeta `data/`.")
    dataset = pd.read_csv("data/Titanic-Dataset.csv")

    st.write("**Vista previa del dataset original:**")
    st.dataframe(dataset.head())

    # ============================================
    # 2️⃣ SELECCIONAR COLUMNAS RELEVANTES
    # ============================================
    st.subheader("2️⃣ Seleccionar columnas relevantes")
    st.write("""
    Mantenemos solo las columnas relevantes para el modelo:
    `Survived`, `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`.
    """)
    dataset = dataset[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    st.dataframe(dataset.head())

    # ============================================
    # 3️⃣ MATRIZ DE VARIABLES (X) Y VECTOR (y)
    # ============================================
    st.subheader("3️⃣ Definición de variables independientes y dependiente")
    X = dataset.iloc[:, 1:].values
    y = dataset.iloc[:, 0].values

    st.write("**Primeras filas de X (variables predictoras):**")
    st.dataframe(pd.DataFrame(X, columns=dataset.columns[1:]).head())
    st.write("**Primeras filas de y (variable objetivo):**")
    st.dataframe(pd.DataFrame(y, columns=["Survived"]).head())

    # ============================================
    # 4️⃣ TRATAMIENTO DE VALORES NULOS
    # ============================================
    st.subheader("4️⃣ Tratamiento de valores faltantes")
    st.write("""
    Reemplazamos los valores faltantes:
    - **Age** → media  
    - **Embarked** → moda
    """)
    st.write("**Valores nulos antes:**")
    st.dataframe(dataset.isnull().sum())

    # Age → media
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    imputer = imputer.fit(X[:, [2]])  # Columna 'Age'
    X[:, [2]] = imputer.transform(X[:, [2]])

    # Embarked → moda
    embarked_col = pd.Series(X[:, 6])
    moda_embarked = embarked_col.mode()[0]
    embarked_col.fillna(moda_embarked, inplace=True)
    X[:, 6] = embarked_col

    st.write("**Valores nulos después (verificación):**")
    df_temp = pd.DataFrame(X, columns=dataset.columns[1:])
    st.dataframe(df_temp.isnull().sum())

    st.write("**Dataset tras reemplazar valores faltantes:**")
    st.dataframe(df_temp.head())

    # ============================================
    # 5️⃣ CODIFICACIÓN DE VARIABLES CATEGÓRICAS
    # ============================================
    st.subheader("5️⃣ Codificación de variables categóricas")
    st.write("""
    - `Sex`: con LabelEncoder (0 = male, 1 = female)  
    - `Embarked`: con OneHotEncoder
    """)

    # Sex
    le_sex = LabelEncoder()
    X[:, 1] = le_sex.fit_transform(X[:, 1])

    st.write("**Después de codificar 'Sex':**")
    st.dataframe(pd.DataFrame(X, columns=dataset.columns[1:]).head())

    # Embarked → OneHotEncoder
    ct = ColumnTransformer(
        [('encoder', OneHotEncoder(categories='auto'), [6])],
        remainder='passthrough'
    )
    X = np.array(ct.fit_transform(X), dtype=np.float64)

    st.write("**Después de aplicar OneHotEncoder a 'Embarked':**")
    st.dataframe(pd.DataFrame(X).head())
    st.markdown("""
    Nota: Las nuevas columnas creadas por OneHotEncoder para 'Embarked' son:
    - Embarked_C
    - Embarked_Q
    - Embarked_S 
                
    Lo que hace OneHotEncoder es crear columnas binarias para cada categoría :D, es método es el más recomendado.
    """)
    # ============================================
    # 6️⃣ DIVISIÓN EN TRAIN Y TEST
    # ============================================
    st.subheader("6️⃣ División del dataset en entrenamiento y prueba")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    st.write(f"🔹 X_train: {X_train.shape}")
    st.write(f"🔹 X_test: {X_test.shape}")
    st.write(f"🔹 y_train: {y_train.shape}")
    st.write(f"🔹 y_test: {y_test.shape}")

    # Mostrar una parte del conjunto de entrenamiento
    st.write("**Primeras filas de X_train:**")
    st.dataframe(pd.DataFrame(X_train).head())

    # ============================================
    # 7️⃣ ESCALADO DE VARIABLES NUMÉRICAS
    # ============================================
    st.subheader("7️⃣ Escalado de variables numéricas")
    st.write("""
    Se aplica `StandardScaler` para que todas las variables numéricas
    estén en una escala comparable.
    """)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    st.write("**Primeras filas de X_train escalado:**")
    st.dataframe(pd.DataFrame(X_train).head())

    # ============================================
    # 8️⃣ CONCLUSIÓN
    # ============================================
    st.subheader("8️⃣ Conclusión del preprocesamiento")
    st.write("""
    ✔️ Cargamos el dataset  
    ✔️ Seleccionamos columnas relevantes  
    ✔️ Tratamos valores faltantes  
    ✔️ Codificamos variables categóricas  
    ✔️ Dividimos en entrenamiento/prueba  
    ✔️ Escalamos las variables numéricas  
    """)
    st.success("🎯 Preprocesamiento completado con éxito.")
    return X_train, X_test, y_train, y_test
