import os
import io
import traceback
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from PIL import Image

from backend.engine import VisionEngine

# Create FastAPI server
app = FastAPI(
    title="AetherVision API",
    description="Backend services for AetherVision local Computer Vision workbench",
    version="1.0.0"
)

# Setup CORS to allow local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the vision inference engine
engine = VisionEngine()

# Helpers
def load_uploaded_image(file: UploadFile) -> Image.Image:
    try:
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

# Endpoints
@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    task: str = Form(...),
    text_input: str = Form(None)
):
    """
    Analyzes an uploaded image using Florence-2 tasks.
    """
    image = load_uploaded_image(file)
    try:
        result = engine.run_florence_task(image, task, text_input)
        return result
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Florence-2 execution error: {str(e)}")

@app.post("/api/depth")
async def get_depth_map(file: UploadFile = File(...)):
    """
    Estimates depth map using Depth-Anything-V2.
    """
    image = load_uploaded_image(file)
    try:
        result = engine.run_depth_map(image)
        return result
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Depth Anything execution error: {str(e)}")

@app.post("/api/similarity")
async def get_similarity(
    file: UploadFile = File(...),
    candidates: str = Form(...)  # Comma-separated list of candidate labels
):
    """
    Performs zero-shot image classification / description match using CLIP.
    """
    image = load_uploaded_image(file)
    
    # Parse candidate list
    candidate_list = [c.strip() for c in candidates.split(",") if c.strip()]
    if not candidate_list:
        raise HTTPException(status_code=400, detail="Must provide at least one candidate label.")
        
    try:
        result = engine.run_similarity(image, candidate_list)
        return result
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"CLIP execution error: {str(e)}")

import base64
from pydantic import BaseModel
from typing import Optional

class StreamRequest(BaseModel):
    image: str  # base64 string
    modality: str  # "florence", "depth", "clip"
    task: Optional[str] = None
    text_input: Optional[str] = None
    candidates: Optional[str] = None

def decode_base64_image(base64_str: str) -> Image.Image:
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

@app.post("/api/stream")
async def stream_inference(req: StreamRequest):
    """
    Handles rapid base64 frames for webcam/video streaming inference.
    """
    image = decode_base64_image(req.image)
    try:
        if req.modality == "florence":
            if not req.task:
                raise HTTPException(status_code=400, detail="Florence task parameter is required")
            result = engine.run_florence_task(image, req.task, req.text_input)
            return result
        elif req.modality == "depth":
            result = engine.run_depth_map(image)
            return result
        elif req.modality == "clip":
            if not req.candidates:
                raise HTTPException(status_code=400, detail="CLIP candidate labels are required")
            candidate_list = [c.strip() for c in req.candidates.split(",") if c.strip()]
            result = engine.run_similarity(image, candidate_list)
            return result
        else:
            raise HTTPException(status_code=400, detail=f"Invalid modality: {req.modality}")
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Streaming inference error: {str(e)}")

# Serve frontend directory
frontend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(frontend_path):
    # Mount the frontend directory at the root
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    @app.get("/")
    async def root_fallback():
        return HTMLResponse("<h3>AetherVision Frontend folder missing. Backend is running.</h3>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
