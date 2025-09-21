from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Load TFLite model once
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image: Image.Image):
    image = image.resize((input_details[0]['shape'][1], input_details[0]['shape'][2]))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile):
    try:
        image = Image.open(file.file).convert("RGB")
        input_data = preprocess_image(image)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        pred_class = int(np.argmax(output_data[0]))
        confidence = float(np.max(output_data[0]))
        return JSONResponse({"status": "success", "prediction": str(pred_class), "confidence": confidence})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})
