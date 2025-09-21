from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Load TFLite model
MODEL_PATH = "model.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Example class labels
CLASS_NAMES = ["Apple Scab", "Apple Black Rot", "Cedar Rust", "Healthy"]

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    image = image.resize(target_size)
    img_array = np.array(image).astype(np.float32)
    img_array = img_array / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image = Image.open(file.file).convert("RGB")
        input_data = preprocess_image(image, target_size=(224, 224))

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        pred_idx = int(np.argmax(output_data))
        confidence = float(np.max(output_data))

        return JSONResponse({
            "status": "success",
            "prediction": CLASS_NAMES[pred_idx],
            "confidence": round(confidence, 4)
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        })
