import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def preprocesamiento_student():
   
    # 1️⃣ CARGA DEL DATASET
    st.subheader("1️⃣ Cargar dataset")
    dataset = pd.read_csv("data/student-mat.csv", sep=';')
    st.dataframe(dataset.head())

    # 2️⃣ EXPLORACIÓN INICIAL
    st.subheader("2️⃣ Exploración inicial")
    st.write("Tipos de datos:")
    st.dataframe(dataset.dtypes)
    st.write("Valores nulos:")
    st.dataframe(dataset.isnull().sum())
    st.write("Estadísticas descriptivas:")
    st.dataframe(dataset.describe())

    # 3️⃣ ELIMINAR DUPLICADOS
    st.subheader("3️⃣ Eliminación de duplicados")
    n_duplicados = dataset.duplicated().sum()
    st.write(f"🔹 Duplicados detectados: {n_duplicados}")
    dataset = dataset.drop_duplicates()
    st.dataframe(dataset.head())

    # 4️⃣ VARIABLES CATEGÓRICAS
    st.subheader("4️⃣ Variables categóricas")
    cat_cols = dataset.select_dtypes(include=['object']).columns.tolist()
    st.write(f"Variables categóricas detectadas: {cat_cols}")

    # 5️⃣ SEPARAR X Y y
    st.subheader("5️⃣ Separar variables predictoras y variable objetivo")
    st.write("""
    - **Variable objetivo (y):** `G3` → Nota final del estudiante  
    - **Variables predictoras (X):** Todas las demás columnas excepto `G3`  
    (por ejemplo: school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob, reason, guardian, traveltime, studytime, failures, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic, famrel, freetime, goout, Dalc, Walc, health, absences, G1, G2)
    """)
    X = dataset.drop('G3', axis=1)
    y = dataset['G3'].values

    st.write("**Primeras filas de X (predictoras):**")
    st.dataframe(X.head())
    st.write("**Primeras filas de y (objetivo):**")
    st.dataframe(pd.DataFrame(y, columns=['G3']).head())

    # 6️⃣ ONE HOT ENCODING
    st.subheader("6️⃣ Codificación One Hot Encoding")
    ct = ColumnTransformer(
        transformers=[('encoder', OneHotEncoder(drop='first'), cat_cols)],
        remainder='passthrough'
    )
    X = np.array(ct.fit_transform(X))
    st.write(f"Dimensiones de X tras codificación: {X.shape}")
    st.dataframe(pd.DataFrame(X).head())

    # 7️⃣ NORMALIZACIÓN DE VARIABLES NUMÉRICAS
    st.subheader("7️⃣ Normalización de variables numéricas")
    num_cols = ['age', 'absences', 'G1', 'G2']
    scaler = StandardScaler()
    # Tomamos las últimas columnas como num_cols
    X[:, -len(num_cols):] = scaler.fit_transform(X[:, -len(num_cols):])
    st.dataframe(pd.DataFrame(X).head())

    # 8️⃣ DIVISIÓN EN TRAIN Y TEST
    st.subheader("8️⃣ División entrenamiento/prueba")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    st.write(f"🔹 X_train: {X_train.shape}")
    st.write(f"🔹 X_test: {X_test.shape}")
    st.write(f"🔹 y_train: {y_train.shape}")
    st.write(f"🔹 y_test: {y_test.shape}")
    # 9️⃣ PRUEBA DE PREDICCIÓN (LINEAR REGRESSION)
    st.subheader("9️⃣ Prueba de predicción con LinearRegression")
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write("**Primeros 10 valores predichos vs reales:**")
    st.dataframe(pd.DataFrame({'Real': y_test[:10], 'Predicción': y_pred[:10]}))

    # 10️⃣ CORRELACIÓN ENTRE G1, G2, G3
    st.subheader("🔟 Correlación entre G1, G2 y G3")
    corr = dataset[['G1', 'G2', 'G3']].corr()
    st.dataframe(corr)
    st.write("Mapa de correlación de las notas parciales y la nota final.")

    # Seleccionamos únicamente G1, G2 y G3
    notas = dataset[['G1', 'G2', 'G3']]

    # Calculamos la matriz de correlación
    corr_matrix = notas.corr()

    # Creamos el heatmap
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlación entre G1, G2 y G3")
    st.pyplot(fig)

    st.success("🎯 Preprocesamiento y prueba de predicción completados con éxito.")

    return X_train, X_test, y_train, y_test, model
