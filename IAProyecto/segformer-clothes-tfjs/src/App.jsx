import React, { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import { Webcam } from "./utils/webcam";
import { renderBoxes } from "./utils/renderBox";
import "./style/App.css";

const generateClassColors = (numClasses) => {
  const colors = [];
  const hueStep = 360 / numClasses;

  for (let i = 0; i < numClasses; i++) {
    const hue = i * hueStep;
    const color = `hsl(${hue}, 100%, 50%)`;
    colors.push(color);
  }
  colors[0] = `hsl(0%, 0%, 0%)`; // setting the background as None

  return colors;
};

const App = () => {
  const [loading, setLoading] = useState({ loading: true, progress: 0 });
  const [predictions, setPredictions] = useState([]); // Cambiado a lista de predicciones
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const segmentImagesRef = useRef(null); // Referencia al contenedor de imágenes de segmentos
  const webcam = new Webcam();
  const modelName = "clothes_model";

  // Define labels and colors
  const labels = [
    "Background",
    "Hat",
    "Hair",
    "Sunglasses",
    "Upper-clothes",
    "Skirt",
    "Pants",
    "Dress",
    "Belt",
    "Left-shoe",
    "Right-shoe",
    "Face",
    "Left-leg",
    "Right-leg",
    "Left-arm",
    "Right-arm",
    "Bag",
    "Scarf",
  ];

  const colors = generateClassColors(labels.length);

  const detectFrame = async (model) => {
    const model_dim = [512, 512];
    tf.engine().startScope();
    const input = tf.tidy(() => {
      const img = tf.image
        .resizeBilinear(tf.browser.fromPixels(videoRef.current), model_dim)
        .div(255.0)
        .expandDims(0);
      return img;
    });
  
    await model.executeAsync(input).then(async (res) => {
      const detections = res.arraySync()[0];
      const filteredDetections = detections.filter((detection, index) => {
        const label = labels[index + 1]; // +1 para ignorar la etiqueta "Background"
        return labels.includes(label); // Filtrar solo los elementos en la lista 'labels'
      });
  
      const rawImage = videoRef.current;
  
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
  
      const formato = await renderBoxes(canvasRef, filteredDetections, rawImage, segmentImagesRef); // Pasar referencia al contenedor de imágenes
      setPredictions(formato); // Almacenar las predicciones en el estado
      console.log(predictions);
      tf.dispose(res);
    });
  
    requestAnimationFrame(() => detectFrame(model));
    tf.engine().endScope();
  };

  useEffect(() => {
    tf.loadGraphModel(`${window.location.origin}/${modelName}_web_model/model.json`, {
      onProgress: (fractions) => {
        setLoading({ loading: true, progress: fractions });
      },
    }).then(async (segformer) => {
      const dummyInput = tf.ones(segformer.inputs[0].shape);
      await segformer.executeAsync(dummyInput).then((warmupResult) => {
        tf.dispose(warmupResult);
        tf.dispose(dummyInput);

        setLoading({ loading: false, progress: 1 });
        webcam.open(videoRef, () => detectFrame(segformer));
      });
    });
  }, []);
  console.warn = () => {};

  return (
    <div className="App">
      <h2 className="title">Prediccion de colores</h2>
      <div className="content">
        <video autoPlay playsInline muted ref={videoRef} id="frame" />
        <canvas width={640} height={640} ref={canvasRef} />
      </div>
      <div className="predictions">
        <h3>Predicciones:</h3>
        <ul>
          {predictions.map((prediction, index) => (
            <li key={index}>{prediction}</li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default App;
