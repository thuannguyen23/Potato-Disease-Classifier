from fastapi import FastAPI, File, UploadFile
from uvicorn import run
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf

app = FastAPI()
MODEL = tf.keras.models.load_model("./models/potato_disease_classifier_v1.keras")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/ping")
async def ping():
    return "Hello, My name is Thuan"


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.resize((256, 256))
    image = np.array(image)
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, axis=0)
    predictions = MODEL.predict(image_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {"predicted_class": predicted_class, "confidence": float(confidence)}


if __name__ == "__main__":
    run(app, host="localhost", port=8000)
