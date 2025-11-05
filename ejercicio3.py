import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import io
import numpy as np

def preprocesamiento_iris():
    st.title("🌸 Ejercicio 3: Preprocesamiento del Dataset Iris (Paso a paso)")

    # ============================================
    # 1️⃣ CARGA DEL DATASET
    # ============================================
    st.subheader("1️⃣ Cargar dataset desde sklearn")
    iris = load_iris()
    st.write("Dataset cargado desde `sklearn.datasets`.")

    # ============================================
    # 2️⃣ CONVERTIR A DATAFRAME
    # ============================================
    st.subheader("2️⃣ Convertir a DataFrame y agregar nombres de columnas")
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    st.write("**Primeras filas del dataset:**")
    st.dataframe(df.head())

    # Información inicial
    st.write("**Información general del dataset:**")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.write("**Descripción estadística inicial:**")
    st.dataframe(df.describe())

    st.write("**Valores nulos por columna:**")
    st.dataframe(df.isnull().sum())

    # ============================================
    # 3️⃣ ESTANDARIZACIÓN
    # ============================================
    st.subheader("3️⃣ Estandarización de variables numéricas")
    features = iris.feature_names
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    st.write("**Primeras filas después de estandarizar:**")
    st.dataframe(df.head())

    st.write("**Descripción estadística después del escalado:**")
    st.dataframe(df[features].describe())

    # ============================================
    # 4️⃣ DIVISIÓN EN ENTRENAMIENTO Y PRUEBA
    # ============================================
    st.subheader("4️⃣ División en entrenamiento y prueba")
    X = df[features].values
    y = df['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    st.write(f"🔹 X_train: {X_train.shape}")
    st.write(f"🔹 X_test: {X_test.shape}")
    st.write(f"🔹 y_train: {y_train.shape}")
    st.write(f"🔹 y_test: {y_test.shape}")

    st.write("**Primeras filas de X_train:**")
    st.dataframe(pd.DataFrame(X_train, columns=features).head())

    # ============================================
    # 5️⃣ GRÁFICO DE DISPERSIÓN
    # ============================================
    st.subheader("5️⃣ Gráfico de dispersión: Sepal length vs Petal length")
    plt.figure(figsize=(8,6))
    for target, color, label in zip([0,1,2], ['r','g','b'], iris.target_names):
        plt.scatter(
            df.loc[df['target']==target, 'sepal length (cm)'],
            df.loc[df['target']==target, 'petal length (cm)'],
            c=color,
            label=label
        )
    plt.xlabel("Sepal length (estandarizado)")
    plt.ylabel("Petal length (estandarizado)")
    plt.title("Distribución de Sepal length vs Petal length por clase")
    plt.legend()
    st.pyplot(plt)

    # ============================================
    # 6️⃣ CONCLUSIÓN
    # ============================================
    st.subheader("6️⃣ Conclusión del preprocesamiento")
    st.write("""
    ✔️ Cargamos el dataset Iris  
    ✔️ Convertimos a DataFrame y agregamos nombres de columnas  
    ✔️ Estandarizamos las variables numéricas  
    ✔️ Dividimos en entrenamiento y prueba  
    ✔️ Graficamos la relación entre Sepal length y Petal length diferenciada por clase
    """)
    st.success("🎯 Preprocesamiento completado con éxito.")

    return X_train, X_test, y_train, y_test, df
