from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Cargar modelo una sola vez
model = load_model("grape_model.h5")  # O "modelo.h5"
img_size = (160, 160)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = image.resize(img_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    predicted_class = int(np.argmax(prediction))

    return {
    "prediction": predicted_class,
    "probabilities": prediction.tolist()
    }
