import pandas as pd
import numpy as np
import time
import psutil
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix, 
                             roc_curve, roc_auc_score, precision_score, recall_score, f1_score,
                             mean_squared_error, mean_absolute_error)

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

# Medición del tiempo de ejecución total
start_time_total = time.time()

# Medición del tiempo de entrenamiento
start_time_training = time.time()
rf_classifier.fit(X_train, y_train)
training_time = time.time() - start_time_training
print("Tiempo de entrenamiento:", training_time, "segundos")

# Uso de memoria
process = psutil.Process(os.getpid())
memory_usage = process.memory_info().rss / 1024  # Uso de memoria en KB
print("Uso de memoria:", memory_usage, "KB")

# Validación Cruzada
cv_scores = cross_val_score(rf_classifier, X, y, cv=5, scoring='accuracy')
print("Puntuaciones de Validación Cruzada:", cv_scores)
print("Exactitud media (Validación Cruzada):", np.mean(cv_scores))

# Predicción
start_time_prediction = time.time()
y_pred = rf_classifier.predict(X_test)
prediction_time = time.time() - start_time_prediction
print("Tiempo de predicción:", prediction_time, "segundos")

# Métricas de rendimiento
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:\n", conf_matrix)

# Curva ROC y AUC
y_prob = rf_classifier.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)
print("Área bajo la curva ROC (AUC):", roc_auc)

# Importancia de las características
feature_importances = rf_classifier.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("Importancia de las características:\n", feature_importance_df)

# Curvas de Aprendizaje
train_sizes, train_scores, test_scores = learning_curve(rf_classifier, X, y, cv=5, scoring='accuracy', n_jobs=-1)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

# Métricas de regresión (ejemplo)
regression_mse = mean_squared_error(y_test, y_pred)
regression_mae = mean_absolute_error(y_test, y_pred)

# Resultados
print("Exactitud:", accuracy)
print("Reporte de Clasificación:\n", class_report)
print("Precisión:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Gráfico de la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='Curva ROC (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

# Gráfico de Curvas de Aprendizaje
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label='Puntuación de Entrenamiento')
plt.plot(train_sizes, test_mean, label='Puntuación de Validación')
plt.xlabel('Tamaño del Conjunto de Entrenamiento')
plt.ylabel('Puntuación')
plt.title('Curva de Aprendizaje')
plt.legend(loc="lower right")
plt.show()

# Tiempo de ejecución total del script
total_execution_time = time.time() - start_time_total
print("Tiempo de ejecución total del script:", total_execution_time, "segundos")

# Métricas de regresión (ejemplo)
print("Error Cuadrático Medio (MSE):", regression_mse)
print("Error Absoluto Medio (MAE):", regression_mae)
