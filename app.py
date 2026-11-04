import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Configuración inicial de la página
st.set_page_config(page_title="Streamlit OCEAN Predictor", layout="centered")

st.title("🎯 Big Five Personality Cluster Predictor")
st.write("Ingresa tus puntajes OCEAN utilizando los deslizadores laterales para descubrir tu clúster, conocer a qué celebridades te asemejas y ver tu comparativa en un gráfico de radar.")

# Cargar los componentes del modelo
try:
    kmeans_cargado = joblib.load('kmeans_model.joblib')
    scaler_cargado = joblib.load('scaler.joblib')
    # Cargamos el dataset original de celebridades
    df_data = pd.read_csv('https://raw.githubusercontent.com/gerrymay/fcaData/refs/heads/master/oceanbig5.csv')
    # Etiquetamos el dataframe original con el modelo entrenado
    scaled_feats = scaler_cargado.transform(df_data[['op', 'co', 'ex', 'ag', 'ne']])
    df_data['cluster'] = kmeans_cargado.predict(scaled_feats)
except Exception as e:
    st.error(f"Error al cargar los datos y modelos: {e}. Por favor, asegúrate de que 'kmeans_model.joblib' y 'scaler.joblib' están en el directorio.")
    st.stop()

# Entradas interactivas de usuario en la barra lateral
st.sidebar.header("Tus Resultados Big Five")
op = st.sidebar.slider("Openness (Apertura)", 0.0, 100.0, 50.0, 0.5)
co = st.sidebar.slider("Conscientiousness (Conciencia)", 0.0, 100.0, 50.0, 0.5)
ex = st.sidebar.slider("Extraversion (Extroversión)", 0.0, 100.0, 50.0, 0.5)
ag = st.sidebar.slider("Agreeableness (Amabilidad)", 0.0, 100.0, 50.0, 0.5)
e_ne = st.sidebar.slider("Neuroticism (Neuroticismo)", 0.0, 100.0, 50.0, 0.5)

# Formatear la entrada del usuario
mis_datos = pd.DataFrame([[op, co, ex, ag, e_ne]], columns=['op', 'co', 'ex', 'ag', 'ne'])

# Escalar y Predecir
mis_datos_escalados = scaler_cargado.transform(mis_datos)
mi_cluster = kmeans_cargado.predict(mis_datos_escalados)[0]

# Mostrar el resultado del Cluster
st.success(f"### 🎉 ¡Has sido asignado al Clúster: {mi_cluster}!")

# Obtener y mostrar las 5 celebridades destacadas de este clúster
st.subheader(f"👥 Famosos en el Clúster {mi_cluster} (Muestra de 5)")
famosos_cluster = df_data[df_data['cluster'] == mi_cluster][['usuario', 'op', 'co', 'ex', 'ag', 'ne']]
st.dataframe(famosos_cluster.head(5))

# Preparación de datos para el radar
features = ['op', 'co', 'ex', 'ag', 'ne']
labels = ['Openness (op)', 'Conscientiousness (co)', 'Extraversion (ex)', 'Agreeableness (ag)', 'Neuroticism (ne)']

# Media del cluster asignado
values_cluster = df_data[df_data['cluster'] == mi_cluster][features].mean().values
values_cluster = np.concatenate((values_cluster, [values_cluster[0]]))

# Datos ingresados
values_user = np.array([op, co, ex, ag, e_ne])
values_user = np.concatenate((values_user, [values_user[0]]))

# Calcular ángulos
num_vars = len(features)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# Creación de la gráfica de Radar
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# Dibujar Clúster
ax.plot(angles, values_cluster, color='teal', linewidth=2, label=f'Promedio Clúster {mi_cluster}')
ax.fill(angles, values_cluster, color='teal', alpha=0.15)

# Dibujar Usuario
ax.plot(angles, values_user, color='tomato', linewidth=2, linestyle='--', label='Mis Resultados')
ax.fill(angles, values_user, color='tomato', alpha=0.15)

# Ajuste de etiquetas y orientación
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)

plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
st.subheader("📊 Comparación en Gráfica de Radar")
st.pyplot(fig)
