# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter

app = FastAPI(title="Plant Disease Detection - TFLite")

# Load TFLite model
MODEL_PATH = "model.tflite"
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Example class labels
CLASS_NAMES = [
    "Apple Scab",
    "Apple Black Rot",
    "Apple Cedar Rust",
    "Apple Healthy"
]

def preprocess_image(image: Image.Image):
    # Resize and normalize image to match model input
    img = image.resize((224, 224))  # change if your model expects different size
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, 224, 224, 3)
    return img_array

def predict(img_array: np.ndarray):
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output[0])
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = float(output[0][predicted_index])
    return predicted_label, confidence

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = preprocess_image(image)
        label, confidence = predict(img_array)

        return JSONResponse(
            content={
                "status": "success",
                "prediction": label,
                "confidence": round(confidence, 4)
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
