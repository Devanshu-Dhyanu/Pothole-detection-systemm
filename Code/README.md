# RoadGuard

RoadGuard is a computer-vision project for reviewing road images, displaying defect detections, and estimating severity. It includes a FastAPI service, a Streamlit dashboard, and a browser-based inspection interface.

## Live deployment

Browser inspection app: [https://pothole-detection-systemm.onrender.com](https://pothole-detection-systemm.onrender.com)
API health: [https://pothole-detection-systemm.onrender.com/api/v1/health](https://pothole-detection-systemm.onrender.com/api/v1/health)

## Features

- Inspect JPG, PNG, and WebP images up to 10 MB.
- Review annotated detections and severity estimates.
- View the telemetry records bundled with the project.
- Keep recent scans in browser storage, optionally attach location, and export reports.
- Use a detection model when available; the detector can fall back to image processing.

Estimates and detections are for project demonstration and should not be treated as engineering assessments. Telemetry and repair rates may be sample data.

## Requirements

Install the dependencies from `Code/requirements.txt` using a supported Python environment.

---

## Run locally

### Step 1: Environment Setup
```bash
# Install dependencies
pip install -r Code/requirements.txt
```

### Step 2: Run the Streamlit dashboard
```bash
streamlit run Code/app/streamlit_app.py
```
*Access the dashboard at `http://127.0.0.1:8501`.*

### Step 3: Run the API server
```bash
python Code/app/main.py
# Or using uvicorn:
uvicorn Code.app.main:app --host 127.0.0.1 --port 8000 --reload
```
*Interactive API documentation is available at `http://127.0.0.1:8000/docs`.*

## Optional Google sign-in

Google sign-in is optional in both interfaces. Guests can still use the inspection tools, and scan history remains local to the browser; signing in does not sync scans to an account.

1. In [Google Cloud Console](https://console.cloud.google.com/), create a project, configure the OAuth consent screen, and create an OAuth client ID for a **Web application**.
2. Add these authorized redirect URIs to the OAuth client:
   - `http://127.0.0.1:8000/auth/google/callback`
   - `http://127.0.0.1:8501/oauth2callback`
3. Copy `.env.example` to `.env` and fill in the client ID and client secret. Generate `ROADGUARD_SESSION_SECRET` with:
   ```powershell
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
4. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, fill in the same Google client ID and secret, and generate a different random value for `cookie_secret`.
5. If the consent screen is in testing mode, add your Google account as a test user. Restart both servers after saving the files.

Keep `.env` and `.streamlit/secrets.toml` private; they are excluded from Git. For a deployed site, use HTTPS, register the exact HTTPS callback URLs in Google Cloud, and add the frontend origin to `ROADGUARD_ALLOWED_ORIGINS`.

## Free browser-app deployment (Render)

The repository includes a Render Blueprint (`render.yaml`) for the browser app. It uses `requirements-render.txt` and the checked-in ONNX export so the service does not install PyTorch or Streamlit.

1. Push the project to GitHub and sign in to [Render](https://render.com/).
2. Choose **New → Blueprint**, connect this repository, and deploy the `roadguard-inspection` service on the **Free** plan.
3. Wait for the build to finish, then open the generated `onrender.com` URL. The root page is the browser app; `/api/v1/health` shows the loaded detection mode.
4. Google sign-in is optional. To enable it, set `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` in the Render service environment and add `https://YOUR-SERVICE.onrender.com/auth/google/callback` as an authorized redirect URI in Google Cloud. The app derives its public URL from Render when `ROADGUARD_APP_URL` and `GOOGLE_REDIRECT_URI` are not set.

Free Render services sleep after 15 minutes without traffic and may take about a minute to wake. Their filesystem is temporary: local SQLite accounts are cleared when the service sleeps, restarts, or redeploys. Use this setup for a public demo, not persistent user accounts or production data. Free web services have 512 MB RAM and 0.1 CPU; the deployment uses ONNX Runtime to reduce resource use.

### Step 4: Run Unit & Integration Tests
```bash
python Code/tests/test_pipeline.py
```

---

## API

### Health Check Endpoint
```http
GET /api/v1/health
```
**Response:**
The timestamp is generated for each request.
```json
{
  "status": "online",
  "service": "RoadGuard API",
  "detection_mode": "model",
  "timestamp": 0
}
```

### Road Defect Detection Endpoint
```http
POST /api/v1/detect?confidence=0.30&iou=0.45
Content-Type: multipart/form-data
```
**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
  -F "file=@Data/sample_images/pothole_severe_highway_01.jpg"
```

---

The response’s `detection_mode` is `model` when a detector model is loaded and `image-processing fallback` otherwise. Health responses do not report unverified benchmark measurements.
