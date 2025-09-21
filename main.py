# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import uuid
import os

from utils.class_names import class_names

app = FastAPI()

# Load model at startup
model = load_model("plant_disease_final.h5")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded image
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Preprocess image
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = np.expand_dims(np.array(img)/255.0, axis=0)

    # Predict
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    label = class_names[class_idx]
    confidence = float(pred[0][class_idx])

    # Annotate image
    img_cv = cv2.imread(file_path)
    cv2.putText(img_cv, f"{label} ({confidence:.2f})", (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

    # Save annotated image
    annotated_name = f"annotated_{uuid.uuid4().hex}.jpg"
    annotated_path = os.path.join(UPLOAD_DIR, annotated_name)
    cv2.imwrite(annotated_path, img_cv)

    return {
        "prediction": label,
        "confidence": confidence,
        "annotated_image_url": f"/download/{annotated_name}"
    }

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    return FileResponse(file_path)
