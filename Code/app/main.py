"""
RoadGuard Production REST API Service (FastAPI)
Course: CSE471 | Deep Learning for Computer Vision

Provides real-time cloud and edge endpoints for dashcams, municipal servers,
and mobile Android PWA clients.
"""

import os
import sys
import io
import time
import json
import numpy as np
import cv2
from PIL import Image
from typing import List, Optional

# Ensure project root in python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from pydantic import BaseModel

from Code.model.pothole_detector import RoadGuardDetector
from Code.model.severity_engine import PotholeSeverityEngine

app = FastAPI(
    title="RoadGuard AI: Pothole & Road Hazard Detection API",
    description="High-Precision Real-Time Road Defect Localization, Severity Grading, and Telemetry Engine. CSE471 Course Project.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector
detector = RoadGuardDetector()
severity_engine = PotholeSeverityEngine()

# Mount static files for PWA
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
def index_page():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)

    return """
    <html>
        <head><title>RoadGuard AI REST API</title></head>
        <body style="font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; line-height: 1.6;">
            <h1 style="color: #0f172a;">🛣️ RoadGuard AI: Pothole Detection Service</h1>
            <p>Course: CSE471 | Deep Learning for Computer Vision | LPU</p>
            <ul>
                <li><b>Interactive API Docs (Swagger):</b> <a href="/docs">/docs</a></li>
                <li><b>Alternative Redoc:</b> <a href="/redoc">/redoc</a></li>
                <li><b>System Health & Model Status:</b> <a href="/api/v1/health">/api/v1/health</a></li>
                <li><b>Municipal Telemetry Stream:</b> <a href="/api/v1/telemetry">/api/v1/telemetry</a></li>
            </ul>
        </body>
    </html>
    """

@app.get("/api/v1/health")
def health_check():
    return {
        "status": "online",
        "service": "RoadGuard Pothole Detection Engine",
        "course": "CSE471 Deep Learning for Computer Vision",
        "institution": "Lovely Professional University",
        "model_architecture": "YOLOv8n / Anchor-Free Decoupled Head",
        "benchmark_mAP50": 0.924,
        "mean_latency_ms": 18.2,
        "classes": ["Pothole", "Road_Crack", "Manhole_Drain"],
        "timestamp": time.time()
    }

@app.post("/api/v1/detect")
async def detect_road_hazards(
    file: UploadFile = File(...),
    confidence: float = Query(0.30, ge=0.1, le=1.0),
    iou: float = Query(0.45, ge=0.1, le=1.0)
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    detector.conf_threshold = confidence
    detector.iou_threshold = iou

    _, detections, severity_report, latency_ms = detector.predict(image)

    return {
        "success": True,
        "filename": file.filename,
        "latency_ms": latency_ms,
        "num_detections": len(detections),
        "detections": detections,
        "severity_assessment": severity_report
    }

@app.get("/api/v1/telemetry")
def get_telemetry_records(limit: int = 50):
    csv_path = os.path.join(BASE_DIR, "Data", "pothole_telemetry_records.csv")
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        return df.head(limit).to_dict(orient="records")
    return []

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
