import pandas as pd
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import cv2
import joblib

df = pd.read_csv('../colores/dataset_colores.csv')
X = df.drop('etiqueta', axis=1)
y = df['etiqueta']

# SVM
X_entrenamiento, X_testeo, y_entrenamiento, y_testeo = train_test_split(X, y, test_size=0.2, random_state=42)

#svm_model = SVC(kernel='poly', degree=3)
#svm_model = SVC(kernel='poly', degree=3, decision_function_shape='ovr')
svm_model = SVC(kernel='rbf')
svm_model.fit(X_entrenamiento, y_entrenamiento)

y_predicha = svm_model.predict(X_testeo)
precision = accuracy_score(y_testeo, y_predicha)
print(f'Precisión del modelo: {precision*100}')
print(classification_report(y_testeo, y_predicha))

# Exportar el modelo entrenado
joblib.dump(svm_model, 'svm_model.pkl')
