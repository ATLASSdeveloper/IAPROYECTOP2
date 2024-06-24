export const renderBoxes = async (canvasRef, res, rawImage) => {
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
  const canvas = canvasRef.current;
  const ctx = canvas.getContext('2d');

  const numClasses = res.length;
  const numRows = res[0].length;
  const numCols = res[0][0].length;
  const cellWidth = canvas.width / numCols;
  const cellHeight = canvas.height / numRows;
  const predicciones = [];
  
  ctx.drawImage(rawImage, 0, 0, canvas.width, canvas.height);

  for (let classIdx = 0; classIdx < numClasses; classIdx++) {
    if (classIdx == 0 || classIdx == 1 || classIdx == 2 || classIdx == 3|| classIdx == 5|| classIdx == 7|| classIdx == 8 ||classIdx == 12 || classIdx == 13 ||classIdx == 14 || classIdx == 15 || classIdx == 11 || classIdx == 16 || classIdx == 17) {
      continue;
    }

    let minX = numCols, minY = numRows, maxX = 0, maxY = 0;

    for (let row = 0; row < numRows; row++) {
      for (let col = 0; col < numCols; col++) {
        const maskValue = res[classIdx][row][col];

        if (maskValue > 0) {
          if (col < minX) minX = col;
          if (col > maxX) maxX = col;
          if (row < minY) minY = row;
          if (row > maxY) maxY = row;
        }
      }
    }

    if (minX <= maxX && minY <= maxY) {
      const x = minX * cellWidth;
      const y = minY * cellHeight;
      const width = (maxX - minX + 1) * cellWidth;
      const height = (maxY - minY + 1) * cellHeight;

      // Crear un nuevo canvas para la imagen segmentada
      const classCanvas = document.createElement('canvas');
      classCanvas.width = width;
      classCanvas.height = height;
      const classCtx = classCanvas.getContext('2d');

      // Extraer la parte correspondiente de la imagen original
      classCtx.drawImage(rawImage, x, y, width, height, 0, 0, width, height);

      // Obtener la imagen segmentada como base64
      const classBase64 = classCanvas.toDataURL('image/jpeg', 0.8);

      // Enviar la imagen base64 a la API y obtener la respuesta
      const respuesta = await sendDataToAPI(classBase64, classIdx); // Función para enviar datos a la API

      // Añadir la respuesta a las predicciones
      predicciones.push(`clase : ${labels[classIdx]}; color : ${respuesta}`);

      // Dibujar borde rojo alrededor del segmento en el canvas principal
      ctx.strokeStyle = 'red';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);
    }
  }
  
  return predicciones;
};

const sendDataToAPI = async (imageData, classIdx) => {
  try {
    const response = await fetch('http://127.0.0.1:5000/predict_color', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ image_data: imageData }),
    });

    if (response.ok) {
      const responseData = await response.json(); // Parsear la respuesta JSON
      return responseData.color_label;
    } else {
      const errorData = await response.text();
      return `Error ${response.status}: ${errorData}`;
    }
  } catch (error) {
    return `Error: ${error.message}`;
  }
};
