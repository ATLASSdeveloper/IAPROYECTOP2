# Segmentación de prendas y reconocimiento del color

El proyecto permite segmentar las prendas de vestir que lleva una persona (superior , inferior) y clasificar el color que lleva esta, de acuerdo a modelos que se generaron posterior al entrenamiento utilizando distintos algoritmos CNN, SVM, KNN.

## Índice

- [Características](#características)
- [Tecnologías Usadas](#tecnologías-usadas)
- [Contribución](#contribución)
- [Instalación](#instalación)


## Características

- Segmentación de ropa
- Detección de color
- 3 apis con modelos de predicción del color (CNN, SVM, KNN)

## Tecnologías Usadas

- Python
- Nodejs

## Contribución
- Se trabajo a partir de un repositorio que segmentaba a una persona las prendas (tanto de vestir como accesorios).
- Autor : https://github.com/hugozanini/segformer-clothes-tfjs
- Mi contribución fue la del reconocimiento del color.

## Instalación

Sigue estos pasos para iniciar el proyecto en tu máquina local:

1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/ATLASSdeveloper/IAPROYECTOP2/tree/main

2. **Acceder al reconocimiento de ropa:**
   ```bash
   cd IAProyecto/segformer-clothes-tfjs

3. **Instalar dependencias Nodejs:**
   ```bash
   npm install
   
4. **Iniciar:**
   ```bash
   npm start

5. **Acceder a las apis:**
   ```bash
   cd IAProyecto/api

6. **Instalar dependencias python:**
   ```bash
   pip install Flask Flask-CORS numpy opencv-python tensorflow joblib

7. **Iniciar la api que se quiera:**
   ```bash
   py apiCNN.py
   py apiSVM.py
   py apiKNN.py
