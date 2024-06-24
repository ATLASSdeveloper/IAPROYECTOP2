import os
import cv2
import numpy as np
import pandas as pd

# Ruta a la carpeta principal que contiene los subdirectorios
ruta_carpeta_principal = '../colores'

# Listas para almacenar las características y etiquetas
caracteristicas = []
etiquetas = []

# Itera a través de cada subdirectorio en la carpeta principal
for subdir in os.listdir(ruta_carpeta_principal):
    ruta_subdir = os.path.join(ruta_carpeta_principal, subdir)
    
    # Asegúrate de que sea un directorio
    if os.path.isdir(ruta_subdir):
        # Lee las imágenes en el subdirectorio
        for filename in os.listdir(ruta_subdir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                # Cargar la imagen
                img = cv2.imread(os.path.join(ruta_subdir, filename))
                if img is not None:
                    # Redimensionar la imagen a 28x28 píxeles
                    img_redimensionada = cv2.resize(img, (28, 28))
                    
                    # Convertir la imagen redimensionada a un vector (una fila en CSV)
                    vector_img = img_redimensionada.flatten()
                    
                    # Añadir las características y la etiqueta (nombre del subdirectorio)
                    caracteristicas.append(vector_img)
                    etiquetas.append(subdir)  # Usa el nombre del subdirectorio como etiqueta

# Convierte las listas a DataFrame de pandas
df_caracteristicas = pd.DataFrame(caracteristicas)
df_etiquetas = pd.DataFrame(etiquetas, columns=['etiqueta'])

# Combina las características y las etiquetas en un solo DataFrame
df = pd.concat([df_etiquetas, df_caracteristicas], axis=1)

# Guarda el DataFrame en un archivo CSV
df.to_csv('dataset_colores.csv', index=False)

print("Dataset guardado en 'dataset_colores.csv'")

