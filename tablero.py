import streamlit as st
import pandas as pd
import plotly.express as px
from pandas.errors import EmptyDataError
import seaborn as sns
import matplotlib.pyplot as plt

# Función para cargar y preprocesar los datos
@st.cache_data
def load_data(filepath):
    data = pd.read_csv(filepath, delimiter=';')
    data['ass_created_at'] = pd.to_datetime(data['ass_created_at'], unit='s', errors='coerce')
    return data

# Cargar datos
df = load_data('challenge_MLE.csv')

# Sidebar para filtros generales
st.sidebar.header('Filtros Generales')

# Filtro por Curso
all_courses = df['course_name'].unique()
course_options = ['All'] + list(all_courses)
selected_courses = st.sidebar.multiselect('Selecciona el Curso', course_options, default='All')

# Filtro por Rango de Notas
nota_min, nota_max = df['nota_final_materia'].min(), df['nota_final_materia'].max()
selected_notas = st.sidebar.slider('Selecciona el Rango de Notas', nota_min, nota_max, (nota_min, nota_max))

# Búsqueda por Legajo
legajo_busqueda = st.sidebar.text_input('Buscar por Legajo')

# Búsqueda por Palabra Clave en Course Name
keyword = st.sidebar.text_input('Buscar Palabra Clave en Nombre del Curso')

# Aplicar Filtros
df_filtrado = df
if 'All' not in selected_courses:
    df_filtrado = df_filtrado[df_filtrado['course_name'].isin(selected_courses)]
df_filtrado = df_filtrado[df_filtrado['nota_final_materia'].between(selected_notas[0], selected_notas[1])]
if legajo_busqueda:
    df_filtrado = df_filtrado[df_filtrado['legajo'].str.contains(legajo_busqueda)]
if keyword:
    df_filtrado = df_filtrado[df_filtrado['course_name'].str.contains(keyword, case=False)]

# Visualización 1: Rendimiento del Estudiante a lo Largo del Tiempo
st.header('Rendimiento del Estudiante a lo Largo del Tiempo')
fig1 = px.line(df_filtrado, x='particion', y=['nota_final_materia', 'nota_parcial'], color='course_name')
st.plotly_chart(fig1)

# Visualización 2: Distribución de Notas
st.header('Distribución de Notas')
fig2 = px.histogram(df_filtrado, x='nota_final_materia', color='course_name', barmode='overlay')
st.plotly_chart(fig2)

# Visualización 3: Comparación de Rendimiento por Curso
st.header('Comparación de Rendimiento por Curso')
fig3 = px.box(df_filtrado, x='course_name', y='nota_final_materia')
st.plotly_chart(fig3)

# Visualización 4: Análisis Temporal de Tareas y Exámenes
st.header('Análisis Temporal de Tareas y Exámenes')
fig4 = px.line(df, x='ass_created_at', y='assignment_id', color='course_name')
st.plotly_chart(fig4)

# Visualización 5: Rendimiento en Tareas
st.header('Rendimiento en Tareas')
fig5 = px.scatter(df, x='points_possible', y='score', color='course_name')
st.plotly_chart(fig5)

# Visualización 6: Correlaciones entre Notas y Participación en Tareas
st.header('Correlaciones entre Notas y Participación en Tareas')
corr = df[['nota_final_materia', 'score', 'points_possible']].corr()
plt.figure(figsize=(10, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm')
st.pyplot(plt)
