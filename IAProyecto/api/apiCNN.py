from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
import tensorflow as tf
import joblib  # Para cargar el class_dict

app = Flask(__name__)
CORS(app)

# Cargar el modelo CNN
cnn_model = tf.keras.models.load_model('../cnnDataColores/cnn_color_model.h5')

# Cargar class_dict
class_dict = joblib.load('../cnnDataColores/class_dict.pkl')
# Invertir el class_dict para mapear de índice a etiqueta
inv_class_dict = {v: k for k, v in class_dict.items()}

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
    prediction = cnn_model.predict(image_vector)
    predicted_label_index = np.argmax(prediction, axis=1).tolist()[0]
    predicted_label = inv_class_dict[predicted_label_index]
    
    return jsonify({'color_label': predicted_label})

def preprocess_image(image):
    # Aplicar filtro gaussiano
    image_blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Redimensionar la imagen
    img_resized = cv2.resize(image_blurred, (28, 28))  # Asegúrate de que el tamaño sea el mismo que el usado en el entrenamiento
    img_normalized = img_resized.astype('float32') / 255.0  # Normalizar la imagen
    img_expanded = np.expand_dims(img_normalized, axis=0)  # Expandir dimensiones para que sea (1, 28, 28, 3)
    
    return img_expanded

if __name__ == '__main__':
    app.run(debug=True)
