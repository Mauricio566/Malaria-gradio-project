#The decorator connects your Python function to the internet. 
# Without it, your function only exists on your computer.

from unittest import result
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms

from PIL import Image
import io
import sys
import os

#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pdf.pdf_generator import generate_pdf
from architecture.model_architecture import create_model
from processing.image import image_to_base64
from processing.image import prepare_visualization_data

from gradcam.gradcam_utils import generate_gradcam

# Initialize FastAPI app
app = FastAPI(
    title="Malaria Detection API",
    description="API to detect malaria in blood cell images",
    version="1.0.0"
)
# Ruta absoluta basada en la ubicación de main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("BASE_DIR:", BASE_DIR)

#MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "malaria_detection_model.pth")
#MODEL_PATH = os.path.join(BASE_DIR, "models", "malaria_detection_model.pth")
# Detectar si estamos en Docker

if os.path.exists("/app/models/malaria_detection_model.pth"):
    MODEL_PATH = "/app/models/malaria_detection_model.pth"
else:
    MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "models", "malaria_detection_model.pth")


#MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "models", "malaria_detection_model.pth")
print("MODEL_PATH:", MODEL_PATH)
#model = torch.load(MODEL_PATH, map_location="cpu")

# Load model once at startup
model = create_model()
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=torch.device("cpu"))
)
model.eval()

# Image transformations (same as training)
transformation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                        [0.229, 0.224, 0.225])
])

#def predict_image_from_bytes(image_bytes: bytes, model) -> dict:# -> dict means return a dictionary
#Predicts whether an image contains malaria-infected cells
def predict_image_from_bytes(image_bytes: bytes, model, include_gradcam=True) -> dict:
    try:
        # Convert bytes to PIL Image
        #image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor_imagen = transformation(original_image).unsqueeze(0)# Apply transformations
        
        # Make prediction
        with torch.no_grad():
            output = model(tensor_imagen)#logits or predictions
            prediction = torch.argmax(output, dim=1).item()#Find the index with the highest value
            
            # Get confidence score
            probabilities = torch.softmax(output, dim=1)#Convert logits (raw numbers) to probabilities [0-1] For example: [-2.1, 3.4] → [0.15, 0.85]
            confidence = probabilities.max().item()#Take the highest probability (that of the predicted class). .item(): Converts a tensor to a Python number (example: 0.85) This is the model's confidence level in its prediction.
            
            # Map prediction to label
            predicted_class = "Infectado" if prediction == 0 else "No infectado"
            
            #Returns a dictionary with all the information
            #return {
            #    "prediction": predicted_class,
            #    "confidence": round(confidence * 100, 2),#Confidence in percentage (85.67%)
            #    "class_id": prediction
            #}
            result = {
                "prediction": predicted_class,
                "confidence": round(confidence * 100, 2),#Confidence in percentage (85.67%)
                "class_id": prediction
            }
            
            # Generar Grad-CAM si se solicita
        if include_gradcam:
            try:
                # Generar heatmap
                heatmap = generate_gradcam(model, tensor_imagen, target_class=prediction)
                
                # Preparar visualizaciones
                visualization_data = prepare_visualization_data(original_image, heatmap)
                
                # Agregar visualizaciones al resultado
                result.update({
                    "original_image": visualization_data["original"],
                    "heatmap": visualization_data["heatmap"],
                    "overlay": visualization_data["overlay"],
                    "explanation": f"Las áreas en rojo/amarillo muestran las regiones que más influyeron en la predicción '{predicted_class}'"
                })
                
            except Exception as e:
                print(f"Error generating Grad-CAM: {e}")
                # Continuar sin Grad-CAM
                result["original_image"] = image_to_base64(original_image)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

#When someone visits the root of the site ("/"), run this function.
@app.get("/")
async def root():
    #welcome endpoint
    return {
        "message": "Malaria Detection API",
        "status": "active",
        "features": ["Prediction", "Grad-CAM Visualization", "Explainable AI"],
        "endpoints": {
            "predict": "/predict - POST with image",
            "health": "/health - GET to check status"
        }
    }

#Endpoint. When someone visits /health, run this function
@app.get("/health")# FastAPI reads this
async def health_check(): # and this and automatically creates a button and so on.
    return {"status": "healthy", 
            "model_loaded": True,
            "gradcam_enabled": True
            }

#we are going to send an image
@app.post("/predict")
async def predict_malaria(file: UploadFile = File(...)):#=File(...): Tells FastAPI "expect a required file"
    # Validate file type
    #If the content type EXISTS (not None) AND is NOT an image → error
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail="The file must be an image (jpeg, png, etc.)"
        )
    
    # Read image bytes
    #file.read() may take time (large file)
    try:
        image_bytes = await file.read()#read the entire file and await: "Wait for me to finish reading" (because it may take time)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
    
    # Make prediction
    #result = predict_image_from_bytes(image_bytes, model)
    result = predict_image_from_bytes(image_bytes,model,include_gradcam=True)
    pdf_path = generate_pdf(result, file.filename)
    result["pdf_path"] = pdf_path 
    
    return JSONResponse(
        content={
            "filename": file.filename,
            "result": result,
            "status": "success"
        }
    )

# Run with: uvicorn main:app --reload
#cd D:\malaria_inference_project\python\malaria_clasification\src
#python -m uvicorn inference.main:app --reload
#http://127.0.0.1:8000/docs
#uvicorn (the web server that runs FastAPI)
if __name__ == "__main__":  
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)