import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv('challenge_MLE.csv', delimiter=';')

# Preprocesamiento básico (como conversión de fechas si es necesario)

# Sidebar para filtros generales
st.sidebar.header('Filtros Generales')
curso_filtro = st.sidebar.multiselect('Selecciona el Curso', df['course_name'].unique(), default=df['course_name'].unique())

# Filtrar por curso
df_filtrado = df[df['course_name'].isin(curso_filtro)]

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
df['ass_created_at'] = pd.to_datetime(df['ass_created_at'])  # Asegúrate de convertir las fechas correctamente
fig4 = px.line(df, x='ass_created_at', y='assignment_id', color='course_name')
st.plotly_chart(fig4)

# Visualización 5: Rendimiento en Tareas
st.header('Rendimiento en Tareas')
fig5 = px.scatter(df, x='points_possible', y='score', color='course_name')
st.plotly_chart(fig5)

# Visualización 6: Correlaciones entre Notas y Participación en Tareas
st.header('Correlaciones entre Notas y Participación en Tareas')
corr = df[['nota_final_materia', 'score', 'points_possible']].corr()
sns.heatmap(corr, annot=True)
st.pyplot(plt)

# Nota: Este es un esquema básico, ajustar y expandir este código según datos y necesidades.
