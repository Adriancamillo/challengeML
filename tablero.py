import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

# Cargar datos con el delimitador correcto
df = pd.read_csv('challenge_MLE.csv', delimiter=';')

# Función para manejar la opción "ALL"
def multiselect_with_all_option(label, options):
    all_option = ['ALL']
    selected = st.sidebar.multiselect(label, all_option + options, default=all_option)
    if 'ALL' in selected:
        return options
    else:
        return selected

# Sidebar para filtros
st.sidebar.header('Filtros')
semestre_filtro = multiselect_with_all_option('Selecciona el Semestre', df['periodo'].unique())
curso_filtro = multiselect_with_all_option('Selecciona el Curso', df['course_name'].unique())

# Búsqueda por Legajo
st.sidebar.header('Búsqueda por Legajo')
legajo_buscado = st.sidebar.text_input('Ingrese el Legajo')

# Aplicar filtros y búsqueda
df_filtrado = df[df['periodo'].isin(semestre_filtro) & df['course_name'].isin(curso_filtro)]
if legajo_buscado:
    df_filtrado = df_filtrado[df_filtrado['legajo'] == legajo_buscado]

# Seleccionar las columnas para el clustering
columns_for_clustering = ['nota_parcial', 'nota_final_materia', 'particion']
df_clustering = df_filtrado[columns_for_clustering]

# Imputación de valores NaN
imputer = SimpleImputer(strategy='mean')
df_clustering_imputed = imputer.fit_transform(df_clustering)

# Clustering
kmeans = KMeans(n_clusters=3)
df_filtrado['cluster'] = kmeans.fit_predict(df_clustering_imputed)

# Visualización de Clustering
st.header('Análisis de Clustering de Notas')
fig1 = px.scatter(df_filtrado, x='nota_parcial', y='nota_final_materia', color='cluster', hover_data=['particion'])
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
