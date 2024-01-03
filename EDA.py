import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos
file_path = 'challenge_MLE.csv'  # Asegúrate de usar la ruta correcta del archivo
data = pd.read_csv(file_path, sep=';')
data.head()
# Crear el gráfico de puntos
plt.figure(figsize=(12, 8))
plt.scatter(data['periodo'], data['nota_final_materia'], alpha=0.5)

# Añadir títulos y etiquetas
plt.title('Notas Finales de Alumnos por Materia')
plt.xlabel('Nombre de la Materia')
plt.ylabel('Nota Final')
plt.xticks(rotation=45)  # Rota las etiquetas del eje x para mejor lectura

# Mostrar el gráfico
plt.tight_layout()
plt.show()
