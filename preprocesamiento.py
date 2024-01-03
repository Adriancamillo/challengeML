## este archivo es un traspaso de un google colab, por lo cual la estructura del archivo esta pensada para correrse en celdas, de correrse secuencialmente puede desencadenar en problemas de rendimientos, muchas gracias

import pandas as pd

# Load the CSV file

file_path = 'file path'

data = pd.read_csv(file_path, delimiter=';', low_memory=False)

# the first few rows of the dataset to understand its structure
data.head()

# date formats:

# Normalizing date formats in the 'periodo' column
data['periodo'] = data['periodo'].str.replace(r'^0', '', regex=True)

# Displaying the unique values in the 'periodo' column to verify the change
data['periodo'].unique()[:10]  # Displaying first 10 unique values for brevity



#EDA:

import matplotlib.pyplot as plt
import seaborn as sns

# Overview of the dataset
data_description = data.describe()

# Distribution of final grades
plt.figure(figsize=(10, 6))
sns.histplot(data['nota_final_materia'], bins=20, kde=True)
plt.title('Distribución de Notas Finales')
plt.xlabel('Nota Final')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()

# Checking for null values in key columns
null_values = data.isnull().sum()

# Displaying the dataset overview and null values in key columns
data_description, null_values.head(10)  # Displaying first 10 columns for brevity



## Preprosesamiento
# Data Cleaning: Creating binary indicators for missing values in relevant columns

# Identifying columns with missing values
columns_with_missing_values = data.columns[data.isnull().any()].tolist()

# Creating binary indicators for these columns
for col in columns_with_missing_values:
    data[col + '_missing'] = data[col].isnull().astype(int)

# Cautious Imputation: Imputing 'nota_parcial' with a minimum grade value (0, FN)
# Assuming a minimum grade of 0 for academic context
data['nota_parcial'] = data['nota_parcial'].fillna(0)

# Display the changes made
data[['nota_parcial'] + [col + '_missing' for col in columns_with_missing_values]].head()

from sklearn.preprocessing import LabelEncoder

# ing de caracteristicas y encoding inicial

# Label Encoding for categorical variables
label_encoders = {}
categorical_columns = data.select_dtypes(include=['object']).columns

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le  # Storing the label encoder for each column

# Display the transformed dataset with encoded categorical variables
data[categorical_columns].head()




##etiquetas: 
from sklearn.feature_selection import SelectKBest, f_classif



# Preparing the dataset for feature selection

X = data.drop('nota_final_materia', axis=1)  # Features

y = data['nota_final_materia']  # Target



# Apply SelectKBest class to extract top 10 best features

bestfeatures = SelectKBest(score_func=f_classif, k=10)

fit = bestfeatures.fit(X, y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)



# Concatenate two dataframes for better visualization 

feature_scores = pd.concat([dfcolumns, dfscores], axis=1)

feature_scores.columns = ['Feature', 'Score']  # Naming the dataframe columns

top_features = feature_scores.nlargest(10, 'Score')  # Top 10 features



top_features  # Displaying the top 10 features




# V2 labels
# Imputation of missing values in the remaining columns
# We will use the median for numerical columns

numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns

# Imputing missing values with the median of each column
for col in numerical_columns:
    X[col] = X[col].fillna(X[col].median())

# Verifying that there are no more NaN values in the dataset
X.isnull().sum().sum()  # Sum of all NaN values in the dataset








##random forest:
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Preparing the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Training the model
rf_classifier.fit(X_train, y_train)

# Predicting the test set results
y_pred = rf_classifier.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

accuracy, class_report  # Displaying the accuracy and classification report
