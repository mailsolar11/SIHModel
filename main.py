# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = FastAPI(title="Plant Disease Detection API")

MODEL_PATH = "model.tflite"

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class labels (replace with your own)
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___Healthy"
    # Add all classes from your dataset here
]

def preprocess_image(image_bytes):
    # Load image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Resize to model input size
    input_shape = input_details[0]['shape']
    img = img.resize((input_shape[1], input_shape[2]))
    img_array = np.array(img, dtype=np.float32)
    # Normalize if model expects it
    img_array = img_array / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img_array = preprocess_image(image_bytes)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)
        # Run inference
        interpreter.invoke()
        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        # Get predicted class
        predicted_index = np.argmax(output_data)
        predicted_label = CLASS_NAMES[predicted_index]
        confidence = float(np.max(output_data))

        return JSONResponse(
            content={
                "status": "success",
                "predicted_label": predicted_label,
                "confidence": confidence
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/")
def root():
    return {"message": "Plant Disease Detection API is running."}
