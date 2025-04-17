import io
import os
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO

# Configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Mushroom Disease Identification API",
    description="API for identifying diseases in mushrooms using YOLOv8",
    version="1.0.0",
)

# Set up CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model loading is deferred to startup event
model = None

@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = YOLO(MODEL_PATH)
        print(f"Model loaded successfully: {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/")
async def root():
    """Return API information"""
    return {
        "message": "Mushroom Disease Identification API",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/predict", "method": "POST", "description": "Predict diseases from an image"},
            {"path": "/predict-url", "method": "POST", "description": "Predict diseases from an image URL"},
            {"path": "/image/{image_name}", "method": "GET", "description": "Get image directly as blob"}
        ]
    }

@app.get("/health")
async def health_check():
    """Check if the API and model are running correctly"""
    return {
        "status": "ok",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    save_image: bool = Form(False)
):
    """
    Process a single image and return disease detection results
    
    - **file**: The image file to analyze
    - **conf**: Confidence threshold (0-1)
    - **save_image**: Whether to save the annotated image
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Process the image with YOLOv8
        start_time = time.time()
        results = model(image, conf=conf)[0]
        inference_time = time.time() - start_time
        
        # Process detection results
        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            conf_val = float(box.conf[0])
            class_name = results.names[cls]
            bbox = box.xyxy[0].tolist()  # Convert to (x1, y1, x2, y2) format
            
            detections.append({
                "class": class_name,
                "confidence": conf_val,
                "bbox": bbox
            })
        
        # Save annotated image if requested
        image_url = None
        image_name = None
        if save_image:
            # Create a timestamp-based filename
            timestamp = int(time.time())
            image_name = f"result_{timestamp}.jpg"
            result_img_path = RESULTS_DIR / image_name
            results.save(filename=str(result_img_path))
            image_url = f"/results/{image_name}"
            
        # Create response
        return {
            "filename": file.filename,
            "inference_time": inference_time,
            "detections": detections,
            "count": len(detections),
            "image_url": image_url,
            "image_name": image_name,
            "direct_image_url": f"/image/{image_name}" if image_name else None
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/image/{image_name}")
async def get_image(image_name: str):
    """
    Return the image directly as a blob
    
    - **image_name**: Name of the image file
    """
    image_path = RESULTS_DIR / image_name
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    from fastapi.responses import FileResponse
    return FileResponse(image_path, media_type="image/jpeg")

@app.post("/predict-url")
async def predict_url(
    image_url: str = Form(...),
    conf: float = Form(0.25),
    save_image: bool = Form(False)
):
    """
    Process an image from URL and return disease detection results
    
    - **image_url**: URL of the image to analyze
    - **conf**: Confidence threshold (0-1)
    - **save_image**: Whether to save the annotated image
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Download the image
        import requests
        response = requests.get(image_url, timeout=10)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to download image: {response.status_code}")
                
        # Process the image
        img_data = response.content
        image = Image.open(io.BytesIO(img_data))
        
        # Run inference
        start_time = time.time()
        results = model(image, conf=conf)[0]
        inference_time = time.time() - start_time
        
        # Process detections
        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            conf_val = float(box.conf[0])
            class_name = results.names[cls]
            bbox = box.xyxy[0].tolist()
            
            detections.append({
                "class": class_name,
                "confidence": conf_val,
                "bbox": bbox
            })
        
        # Save annotated image if requested
        image_url_result = None
        image_name = None
        if save_image:
            # Create a timestamp-based filename
            timestamp = int(time.time())
            image_name = f"result_{timestamp}.jpg"
            result_img_path = RESULTS_DIR / image_name
            results.save(filename=str(result_img_path))
            image_url_result = f"/results/{image_name}"
        
        # Create response
        return {
            "original_url": image_url,
            "inference_time": inference_time,
            "detections": detections,
            "count": len(detections),
            "image_url": image_url_result,
            "image_name": image_name,
            "direct_image_url": f"/image/{image_name}" if image_name else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Serve static files for result images
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

# Docker specific settings
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)