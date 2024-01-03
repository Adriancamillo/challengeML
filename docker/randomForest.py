import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Cargar el archivo CSV
file_path = 'challenge_MLE.csv'
data = pd.read_csv(file_path, delimiter=';', low_memory=False)

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

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar y entrenar el clasificador de Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predicción y evaluación del modelo
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Resultados
print("Exactitud:", accuracy)
print("Reporte de Clasificación:\n", class_report)
