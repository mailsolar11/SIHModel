from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Create FastAPI app
app = FastAPI()

# Load your model once at startup
MODEL_PATH = "plant_disease_final.h5"  # adjust path if needed
model = load_model(MODEL_PATH)

# ✅ Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        img = Image.open(file.file).convert("RGB")
        img = img.resize((224, 224))  # adjust size to match your model
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Run inference
        prediction = model.predict(img_array)
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))

        return JSONResponse({
            "class": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ✅ Run app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Railway provides PORT
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
