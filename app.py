import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import io
from ejercicio1 import preprocesamiento_titanic
from ejercicio2 import preprocesamiento_student
from ejercicio3 import preprocesamiento_iris

# Configuración de la página
st.set_page_config(
    page_title="ML Data Preprocessing",
    page_icon="🤖",
    layout="wide"
)

# Título principal
st.title("🤖 Procesamiento de Datasets en Machine Learning")
st.markdown("**Aplicación de técnicas de preprocesamiento con pandas y scikit-learn**")
st.divider()

# Tabs para cada ejercicio
tab1, tab2, tab3 = st.tabs(["🚢 Titanic", "📚 Student Performance", "🌸 Iris"])

# ============================================
# EJERCICIO 1: TITANIC
# ============================================
with tab1:
   preprocesamiento_titanic()

# ============================================
# EJERCICIO 2: STUDENT PERFORMANCE
# ============================================
with tab2:
    st.header("Ejercicio 2: Student Performance")
    st.markdown("**Objetivo:** Predecir la nota final (G3) de los estudiantes")
    preprocesamiento_student()
# ============================================
# EJERCICIO 3: IRIS
# ============================================
with tab3:
    st.header("Ejercicio 3: Iris Dataset")
    st.markdown("**Objetivo:** Clasificar las especies de iris basándose en características florales")
    preprocesamiento_iris()
# Footer
st.divider()
st.markdown("© 2024 - Aplicación de Preprocesamiento de Datos en ML")