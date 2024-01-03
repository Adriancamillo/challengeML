import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

# Cargar datos
df = pd.read_csv('challenge_MLE.csv', delimiter=';')

# Función para la opción "ALL"
def multiselect_all(label, options):
    all_option = "ALL"
    if all_option not in options:
        options = [all_option] + options
    selected = st.sidebar.multiselect(label, options, default=all_option)
    if all_option in selected:
        selected = options[1:]  # Excluir "ALL" de la lista de opciones
    return selected

# Sidebar para filtros
st.sidebar.header('Filtros')
semestre_filtro = multiselect_all('Selecciona el Semestre', list(df['periodo'].unique()))
curso_filtro = multiselect_all('Selecciona el Curso', list(df['course_name'].unique()))

# Búsqueda por Legajo
st.sidebar.header('Búsqueda por Legajo')
legajo_buscado = st.sidebar.text_input('Ingrese el Legajo')

# Aplicar filtros y búsqueda
df_filtrado = df.copy()
if "ALL" not in semestre_filtro:
    df_filtrado = df_filtrado[df_filtrado['periodo'].isin(semestre_filtro)]
if "ALL" not in curso_filtro:
    df_filtrado = df_filtrado[df_filtrado['course_name'].isin(curso_filtro)]
if legajo_buscado:
    df_filtrado = df_filtrado[df_filtrado['legajo'] == legajo_buscado]

# Clustering
columns_for_clustering = ['particion', 'nota_final_materia']
df_clustering = df_filtrado[columns_for_clustering]

# Imputación de valores NaN
imputer = SimpleImputer(strategy='mean')
df_clustering_imputed = imputer.fit_transform(df_clustering)

kmeans = KMeans(n_clusters=3)
df_filtrado['cluster'] = kmeans.fit_predict(df_clustering_imputed)

# Visualización de Clustering
st.header('Análisis de Clustering de Notas Finales a lo Largo del Tiempo')
fig1 = px.scatter(df_filtrado, x='particion', y='nota_final_materia', color='cluster')
st.plotly_chart(fig1)

# Boxplot para Comparaciones
st.header('Comparación de Notas por Curso')
fig2 = px.box(df_filtrado, x='course_name', y='nota_final_materia', color='course_name')
st.plotly_chart(fig2)

# Heatmap de Correlación
if st.checkbox('Mostrar Heatmap de Correlación'):
    corr = df_filtrado.corr()
    fig_corr = px.imshow(corr, text_auto=True)
    st.plotly_chart(fig_corr)

# Gráficos de Líneas para Tendencias Temporales
if st.checkbox('Mostrar Tendencias Temporales'):
    fig_line = px.line(df_filtrado, x='particion', y='nota_final_materia', color='course_name')
    st.plotly_chart(fig_line)

# Comparación Avanzada de Notas
st.header('Comparación Avanzada de Notas')
opcion_comparacion = st.selectbox('Elige el Tipo de Comparación', ['Por Curso', 'Por Semestre'])
if opcion_comparacion == 'Por Curso':
    fig_comp = px.histogram(df_filtrado, x='nota_final_materia', color='course_name', barmode='group')
else:
    fig_comp = px.histogram(df_filtrado, x='nota_final_materia', color='periodo', barmode='group')
st.plotly_chart(fig_comp)

# Resumen Estadístico
if st.checkbox('Mostrar Resumen Estadístico'):
    st.write(df_filtrado.describe())
