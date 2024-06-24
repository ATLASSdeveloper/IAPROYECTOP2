import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import cv2
import joblib

# Leer el dataset
df = pd.read_csv('../colores/dataset_colores.csv')
X = df.drop('etiqueta', axis=1)
y = df['etiqueta']

# Dividir el dataset en entrenamiento y prueba
X_entrenamiento, X_testeo, y_entrenamiento, y_testeo = train_test_split(X, y, test_size=0.2, random_state=42)

# Calcular el valor de n_neighbors
num_registros = len(df)
n_neighbors = int(np.ceil(np.log10(num_registros)))
if n_neighbors % 2 == 0:
    n_neighbors += 1

print(f'Número de vecinos (n_neighbors): {n_neighbors}')

# Crear y entrenar el modelo KNN
knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
knn_model.fit(X_entrenamiento, y_entrenamiento)

# Predecir y evaluar el modelo
y_predicha = knn_model.predict(X_testeo)
precision = accuracy_score(y_testeo, y_predicha)
print(f'Precisión del modelo: {precision*100}')
print(classification_report(y_testeo, y_predicha))



# Exportar el modelo entrenado
joblib.dump(knn_model, 'knn_model.pkl')

