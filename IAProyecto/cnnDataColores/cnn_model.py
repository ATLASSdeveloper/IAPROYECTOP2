import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
import joblib  # Para guardar el class_dict

# Leer el CSV
df = pd.read_csv('../colores/dataset_colores.csv')

# Separar características y etiquetas
X = df.drop('etiqueta', axis=1).values
y = df['etiqueta'].values

# Normalizar las características
X = X / 255.0

# Convertir las etiquetas a categóricas
class_names = np.unique(y)
class_dict = {class_name: i for i, class_name in enumerate(class_names)}
y = np.array([class_dict[label] for label in y])
y = to_categorical(y, num_classes=len(class_dict))

# Guardar class_dict
joblib.dump(class_dict, 'class_dict.pkl')

# Redimensionar las características a (n_samples, 28, 28, 3)
X = X.reshape(-1, 28, 28, 3)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir la arquitectura del modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(class_dict), activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Guardar el modelo entrenado
model.save('cnn_color_model.h5')
print("Modelo CNN guardado exitosamente.")
