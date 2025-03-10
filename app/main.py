from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import tensorflow as tf
import io
import numpy as np
from PIL import Image
import os
import gdown  # type: ignore # For downloading from Google Drive

app = FastAPI()

# Mount the static directory for CSS and JS files
app.mount("/static", StaticFiles(directory="../app/static"), name="static")

# Set up templates directory
templates = Jinja2Templates(directory="../app/templates")

# Previous local model loading is commented out
# MODEL_PATH = "../model/model.h5"
# model = tf.keras.models.load_model(MODEL_PATH)

# Download model from Google Drive and load it
FILE_ID = "1T4RWW8QltMe7lHbng-ntBkgu-2-a8a42"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"
MODEL_LOCAL_PATH = "model.h5"

if not os.path.exists(MODEL_LOCAL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_LOCAL_PATH, quiet=False)
    print("Download complete.")


model = tf.keras.models.load_model(MODEL_LOCAL_PATH)

# Define a mapping for your 5 retinopathy classes.
# Adjust the labels as necessary.
class_labels = {
    0: "No_Dr",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

@app.get("/")
async def read_home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/about")
async def read_about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/contact")
async def read_contact(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})

@app.get("/test")
async def read_test(request: Request):
    return templates.TemplateResponse("test.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image file as bytes.
    contents = await file.read()
    
    # Open the image using PIL and convert to RGB.
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Resize the image to the target size your model expects (e.g., 224x224).
    target_size = (224, 224)  # Update if your model requires a different size.
    image = image.resize(target_size)
    
    # Convert the image to a numpy array and scale pixel values to [0,1].
    image_array = np.array(image) / 255.0
    
    # Expand dimensions to match the model's expected input shape.
    image_array = np.expand_dims(image_array, axis=0)
    
    # Perform prediction.
    predictions = model.predict(image_array)
    
    # Since predictions is a batch (shape: [1, num_classes]), extract the first element.
    raw_preds = predictions.tolist()[0]
    
    # Build a dictionary mapping class names to their probability values.
    detailed_predictions = { class_labels[i]: raw_preds[i] for i in range(len(raw_preds)) }
    
    # Get the index of the class with the highest probability.
    predicted_class_index = int(np.argmax(predictions, axis=1)[0])
    predicted_label = class_labels.get(predicted_class_index, "Unknown")
    
    return {
        "filename": file.filename,
        "prediction": predicted_label,
        "raw_prediction": detailed_predictions
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)
