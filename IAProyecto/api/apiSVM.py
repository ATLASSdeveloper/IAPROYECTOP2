from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import cv2
import base64
import io
import joblib

app = Flask(__name__)
CORS(app)

with open('../svmDataColores/svm_model.pkl', 'rb') as f:
    svm_model = joblib.load(f)


@app.route('/predict_color', methods=['POST'])
def predict_color():
    data = request.json
    image_data = data.get('image_data', None)

    if image_data is None:
        return jsonify({'error': 'No se proporcionaron datos de imagen'}), 400
    try:
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'error': f'Error al decodificar la imagen: {str(e)}'}), 400

    image_vector = preprocess_image(image)
    prediction = svm_model.predict(image_vector).tolist()  # Convertir a lista
    return jsonify({'color_label': prediction})


def preprocess_image(data):
    # Aplicar filtro gaussiano
    image_blurred = cv2.GaussianBlur(data, (5, 5), 0)  # (5, 5) es el tamaño del kernel, puedes ajustarlo

    # Redimensionar la imagen
    img_redimensionada = cv2.resize(image_blurred, (28, 28))
    vector_img = img_redimensionada.flatten()
    nueva_imagen_vector = pd.DataFrame([vector_img])
    return nueva_imagen_vector

if __name__ == '__main__':
    app.run(debug=True)
