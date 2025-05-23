from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Habilitar CORS
origins = [
    "http://localhost:4200"  # Origen de Angular
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo una sola vez
model = load_model("grape_model.h5")
img_size = (64, 64)  # Tamaño esperado por el modelo

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = image.resize(img_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # [1, 64, 64, 3]
    
    prediction = model.predict(img_array)
    predicted_class = int(np.argmax(prediction))

    return {
        "prediction": predicted_class,
        "probabilities": prediction.tolist()
    }

@app.get("/predict")
def read_root():
    return {"status": "API activa", "mensaje": "Modelo de uvas cargado correctamente"}

