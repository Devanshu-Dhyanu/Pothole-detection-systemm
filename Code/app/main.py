"""FastAPI endpoints for road defect inspection."""

import os
import sys
import io
import base64
import secrets
import time
import json
import sqlite3
import hashlib
import hmac
import numpy as np
import cv2
from PIL import Image
from typing import List, Optional

from authlib.integrations.starlette_client import OAuth
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, RedirectResponse
from pydantic import BaseModel
from PIL import UnidentifiedImageError
from starlette.middleware.sessions import SessionMiddleware

# Ensure project root in python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
load_dotenv(os.path.join(BASE_DIR, ".env"))

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
ROADGUARD_SESSION_SECRET = os.getenv("ROADGUARD_SESSION_SECRET", "").strip()
GOOGLE_REDIRECT_URI = os.getenv(
    "GOOGLE_REDIRECT_URI",
    "http://127.0.0.1:8000/auth/google/callback"
).strip()
ROADGUARD_APP_URL = os.getenv("ROADGUARD_APP_URL", "http://127.0.0.1:8000").strip().rstrip("/")
ROADGUARD_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ROADGUARD_ALLOWED_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000"
    ).split(",")
    if origin.strip()
]
GOOGLE_SIGN_IN_ENABLED = all((GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, ROADGUARD_SESSION_SECRET))

oauth = OAuth()
if GOOGLE_SIGN_IN_ENABLED:
    oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"}
    )

from Code.model.pothole_detector import RoadGuardDetector
from Code.model.severity_engine import PotholeSeverityEngine

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}

app = FastAPI(
    title="RoadGuard: Road Defect Detection API",
    description="Image-based road defect detection, severity estimates, and telemetry records.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ROADGUARD_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ACCOUNT_DB = os.path.join(BASE_DIR, "Data", "roadguard_accounts.sqlite3")

def init_account_db():
    os.makedirs(os.path.dirname(ACCOUNT_DB), exist_ok=True)
    with sqlite3.connect(ACCOUNT_DB) as db:
        db.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL UNIQUE, salt BLOB NOT NULL, password_hash BLOB NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")

def hash_password(password, salt=None):
    salt = salt or secrets.token_bytes(16)
    return salt, hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1)

init_account_db()
app.add_middleware(
    SessionMiddleware,
    secret_key=ROADGUARD_SESSION_SECRET or secrets.token_urlsafe(32),
    session_cookie="roadguard_session",
    same_site="lax",
    https_only=ROADGUARD_APP_URL.startswith("https://")
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
        <head><title>RoadGuard API</title></head>
        <body style="font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; line-height: 1.6;">
            <h1>RoadGuard API</h1>
            <ul>
                <li><a href="/docs">API documentation</a></li>
                <li><a href="/api/v1/health">Service health</a></li>
                <li><a href="/api/v1/telemetry">Telemetry records</a></li>
            </ul>
        </body>
    </html>
    """

@app.get("/api/v1/health")
def health_check():
    return {
        "status": "online",
        "service": "RoadGuard API",
        "detection_mode": "model" if detector.model is not None else "image-processing fallback",
        "timestamp": time.time()
    }

@app.get("/auth/me")
def get_auth_state(request: Request):
    user = request.session.get("user")
    return {
        "configured": GOOGLE_SIGN_IN_ENABLED,
        "authenticated": bool(user),
        "user": user
    }

@app.post("/auth/register")
def register_account(payload: dict, request: Request):
    name = str(payload.get("name", "")).strip()[:80]
    email = str(payload.get("email", "")).strip().lower()[:254]
    password = str(payload.get("password", ""))
    if len(name) < 2 or "@" not in email or len(password) < 8 or len(password) > 128:
        raise HTTPException(status_code=400, detail="Enter a name, valid email, and password with at least 8 characters.")
    salt, password_hash = hash_password(password)
    try:
        with sqlite3.connect(ACCOUNT_DB) as db:
            db.execute("INSERT INTO users (name,email,salt,password_hash) VALUES (?,?,?,?)", (name,email,salt,password_hash))
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    request.session.clear()
    request.session["user"] = {"name": name, "email": email}
    return {"authenticated": True, "user": request.session["user"]}

@app.post("/auth/login")
def login_account(payload: dict, request: Request):
    email = str(payload.get("email", "")).strip().lower()[:254]
    password = str(payload.get("password", ""))
    with sqlite3.connect(ACCOUNT_DB) as db:
        row = db.execute("SELECT name,email,salt,password_hash FROM users WHERE email=?", (email,)).fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")
    candidate = hashlib.scrypt(password.encode("utf-8"), salt=row[2], n=2**14, r=8, p=1)
    if not hmac.compare_digest(candidate, row[3]):
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")
    request.session.clear()
    request.session["user"] = {"name": row[0], "email": row[1]}
    return {"authenticated": True, "user": request.session["user"]}

@app.get("/auth/google/login")
async def google_login(request: Request):
    if not GOOGLE_SIGN_IN_ENABLED:
        raise HTTPException(status_code=503, detail="Google sign-in is not configured.")

    google = oauth.create_client("google")
    if google is None:
        raise HTTPException(status_code=503, detail="Google sign-in is unavailable.")

    request.session.clear()
    return await google.authorize_redirect(request, GOOGLE_REDIRECT_URI)

@app.get("/auth/google/callback")
async def google_callback(request: Request):
    if not GOOGLE_SIGN_IN_ENABLED:
        raise HTTPException(status_code=503, detail="Google sign-in is not configured.")
    if request.query_params.get("error"):
        raise HTTPException(status_code=400, detail="Google sign-in was cancelled or denied.")

    google = oauth.create_client("google")
    if google is None:
        raise HTTPException(status_code=503, detail="Google sign-in is unavailable.")

    token = await google.authorize_access_token(request)
    userinfo = token.get("userinfo")
    if not userinfo or not userinfo.get("sub"):
        raise HTTPException(status_code=502, detail="Google did not return a valid sign-in profile.")

    request.session.clear()
    request.session["user"] = {
        "sub": str(userinfo["sub"]),
        "name": str(userinfo.get("name") or ""),
        "email": str(userinfo.get("email") or "")
    }
    return RedirectResponse(ROADGUARD_APP_URL, status_code=303)

@app.post("/auth/logout")
def google_logout(request: Request):
    request.session.clear()
    return {"authenticated": False}

@app.post("/api/v1/detect")
async def detect_road_hazards(
    file: UploadFile = File(...),
    confidence: float = Query(0.30, ge=0.1, le=1.0),
    iou: float = Query(0.45, ge=0.1, le=1.0)
):
    try:
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image is too large. Maximum upload size is 10 MB.")
        if not contents:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        with Image.open(io.BytesIO(contents)) as uploaded_image:
            image_format = uploaded_image.format
            if image_format not in ALLOWED_IMAGE_FORMATS:
                raise HTTPException(status_code=415, detail="Unsupported image format. Use JPG, PNG, or WebP.")
            uploaded_image.verify()

        with Image.open(io.BytesIO(contents)) as uploaded_image:
            image = uploaded_image.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as e:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid, readable image.") from e
    finally:
        await file.close()

    detector.conf_threshold = confidence
    detector.iou_threshold = iou

    annotated_image, detections, severity_report, latency_ms = detector.predict(image)

    encoded, image_buffer = cv2.imencode(
        ".jpg",
        annotated_image,
        [cv2.IMWRITE_JPEG_QUALITY, 88]
    )
    if not encoded:
        raise HTTPException(status_code=500, detail="Unable to encode annotated image")

    return {
        "success": True,
        "filename": file.filename,
        "latency_ms": latency_ms,
        "num_detections": len(detections),
        "detections": detections,
        "severity_assessment": severity_report,
        "annotated_image": f"data:image/jpeg;base64,{base64.b64encode(image_buffer).decode('ascii')}"
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
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
