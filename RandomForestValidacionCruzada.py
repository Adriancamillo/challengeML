import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Cargar el archivo CSV
file_path = 'challenge_MLE.csv'  # Reemplaza con la ruta a tu archivo CSV
data = pd.read_csv(file_path, delimiter=';')

# Configurar user_uuid, course_uuid, y particion como un índice compuesto
data.set_index(['user_uuid', 'course_uuid', 'particion'], inplace=True)

# Normalización de fechas
data['periodo'] = data['periodo'].str.replace(r'^0', '', regex=True)

# Creación de indicadores binarios para valores faltantes
columns_with_missing_values = data.columns[data.isnull().any()].tolist()
for col in columns_with_missing_values:
    data[col + '_missing'] = data[col].isnull().astype(int)

# Imputación para 'nota_parcial'
data['nota_parcial'] = data['nota_parcial'].fillna(0)

# Codificación de variables categóricas
label_encoders = {}
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Convertir 'nota_final_materia' en una variable binaria (target)
data['target'] = (data['nota_final_materia'] >= 4).astype(int)

# Preparación de los datos para el modelado
X = data.drop(['nota_final_materia', 'target'], axis=1)
y = data['target']

# Imputación de valores faltantes en X
numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_columns:
    X[col] = X[col].fillna(X[col].median())

# Inicializar el clasificador de Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Realizar la validación cruzada de 5 pliegues
cv_scores = cross_val_score(rf_classifier, X, y, cv=5)

# Imprimir los resultados de la validación cruzada
print("Puntajes de Validación Cruzada:", cv_scores)
print("Promedio de Puntaje de Validación Cruzada:", cv_scores.mean())
print("Desviación Estándar de Puntaje de Validación Cruzada:", cv_scores.std())
