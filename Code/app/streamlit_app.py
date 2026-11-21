"""RoadGuard road inspection dashboard."""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image

# Ensure project root in python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
import cv2
from Code.model.pothole_detector import RoadGuardDetector
from Code.model.gradcam import RoadGradCAM

st.set_page_config(
    page_title="RoadGuard | Road inspection",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background: #f3f5f6; color: #202a32; }
    section[data-testid="stSidebar"] { background: #fff; border-right: 1px solid #dce1e4; }
    h1, h2, h3 { color: #202a32; letter-spacing: normal; }
    div[data-testid="stMetric"] {
        padding: 12px 14px;
        border: 1px solid #dce1e4;
        border-radius: 6px;
        background: #fff;
    }
    div[data-testid="stMetricLabel"] { color: #68757f; }
    .stButton > button, .stDownloadButton > button {
        border-radius: 5px;
        border-color: #b9c9bf;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SYSTEM INITIALIZATION -----------------
@st.cache_resource
def load_system():
    # RoadGuardDetector automatically prioritizes fine-tuned best.pt
    detector = RoadGuardDetector()
    gradcam = RoadGradCAM()
    return detector, gradcam

detector, gradcam = load_system()

with st.sidebar:
    st.markdown("#### RoadGuard")
    st.caption("Road inspection and maintenance tools")
    nav_mode = st.radio(
        "Workspace",
        [
            "Image inspection",
            "Telemetry map",
            "Repair estimate",
            "Model information",
            "Project notes"
        ]
    )
    st.divider()

    st.markdown("##### Account")
    try:
        google_auth_settings = st.secrets["auth"]
    except StreamlitSecretNotFoundError:
        google_auth_settings = {}

    required_auth_settings = (
        "redirect_uri",
        "cookie_secret",
        "client_id",
        "client_secret",
        "server_metadata_url"
    )
    google_login_configured = all(google_auth_settings.get(key) for key in required_auth_settings)

    if google_login_configured:
        if st.user.is_logged_in:
            user_name = st.user.get("name") or st.user.get("email") or "Google account"
            st.caption(f"Signed in as {user_name}")
            st.button("Sign out", on_click=st.logout, key="google-logout")
        else:
            st.button("Sign in with Google", on_click=st.login, key="google-login")
            st.caption("Guest access is also available.")
    else:
        st.button("Google sign-in setup required", disabled=True, key="google-login-unconfigured")
        st.caption("You can continue as a guest. Configure Streamlit secrets to enable sign-in.")

    st.divider()
    st.markdown("##### Detection settings")
    conf_thresh = st.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05)
    iou_thresh = st.slider("Overlap threshold", 0.20, 0.80, 0.45, 0.05)
    detector.conf_threshold = conf_thresh
    detector.iou_threshold = iou_thresh

# Header
st.title("Road inspection")
st.caption("Review image-based road defect detections and estimated severity.")

sample_dir = os.path.join(BASE_DIR, "Data", "real_pothole_raw", "Pothole Dataset")
sample_files = sorted(
    f for f in os.listdir(sample_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
)[:20] if os.path.isdir(sample_dir) else []

active_model = detector.model_source
with st.sidebar:
    if active_model:
        st.caption(f"Detection model: {os.path.basename(active_model)}")
    else:
        st.caption("Detection method: image-processing fallback")

if nav_mode == "Image inspection":
    st.subheader("Inspect an image")
    st.caption("Choose an example image or upload a road image. Maximum file size: 10 MB.")

    col_sel, col_up = st.columns([1, 1])
    with col_sel:
        selected_sample = st.selectbox("Example image", ["Choose an image"] + sample_files)
    with col_up:
        uploaded_file = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "webp"],
            max_upload_size=10
        )

    # Resolve active image
    img_to_process = None
    frame_name = ""

    if uploaded_file is not None:
        try:
            img_to_process = Image.open(uploaded_file).convert("RGB")
            frame_name = uploaded_file.name
        except (OSError, ValueError):
            st.error("This file is not a readable image. Choose a JPG, PNG, or WebP file.")
    elif selected_sample != "Choose an image":
        sample_path = os.path.join(sample_dir, selected_sample)
        if os.path.exists(sample_path):
            try:
                img_to_process = Image.open(sample_path).convert("RGB")
                frame_name = selected_sample
            except (OSError, ValueError):
                st.error("The selected example image could not be opened.")

    if img_to_process is not None:
        img_np = np.array(img_to_process)
        ann_img_bgr, detections, severity, latency = detector.predict(img_to_process)
        ann_img_rgb = cv2.cvtColor(ann_img_bgr, cv2.COLOR_BGR2RGB)

        psi = severity['psi']
        num_pits = severity['num_potholes']
        priority = severity['priority_level']

        if "P1" in priority:
            st.error(priority)
        elif "P2" in priority:
            st.warning(priority)
        else:
            st.success(priority)

        st.caption(f"File: {frame_name}")
        metric_columns = st.columns(5)
        metric_columns[0].metric("Potholes", num_pits)
        metric_columns[1].metric("PSI", f"{psi} / 100")
        metric_columns[2].metric("Maximum depth", f"{severity['max_depth_cm']} cm")
        metric_columns[3].metric("Estimated area", f"{severity['total_area_sqm']} m²")
        metric_columns[4].metric("Processing time", f"{latency} ms")

        tab_ann, tab_split, tab_cam = st.tabs(["Detections", "Compare images", "Grad-CAM"])

        with tab_ann:
            st.image(ann_img_rgb, use_container_width=True, caption=f"{len(detections)} detections")

        with tab_split:
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown("##### Original image")
                st.image(img_to_process, use_container_width=True)
            with col_s2:
                st.markdown("##### Annotated image")
                st.image(ann_img_rgb, use_container_width=True)

        with tab_cam:
            st.markdown("##### Gradient-weighted class activation map")
            st.caption("The heatmap shows image regions associated with the selected detection. It is an exploratory visualization, not proof of model reasoning.")
            try:
                primary_bbox = detections[0]['bbox'] if detections else None
                heat, hmc_rgb, super_rgb = gradcam.generate_heatmap(img_np, bbox=primary_bbox)
                st.image(super_rgb, use_container_width=True, caption="Grad-CAM visualization")
            except (ValueError, RuntimeError, IndexError) as error:
                st.warning(f"Grad-CAM could not be generated for this image: {error}")

        st.subheader("Assessment")
        st.write(f"**Distress class:** {severity.get('distress_type', 'Not classified')}")
        st.write(f"**Recommended action:** {severity['action_advisory']}")
        st.write(f"**Suggested repair material:** {severity['repair_material']}")
        st.write(f"**Estimated repair cost:** ₹{severity['estimated_repair_cost_inr']:,}")
elif nav_mode == "Telemetry map":
    st.subheader("Telemetry map")
    st.caption("The map and table use the telemetry records included with this project.")

    csv_path = os.path.join(BASE_DIR, "Data", "pothole_telemetry_records.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            sel_priority = st.multiselect("Priority", options=list(df['hazard_priority'].unique()), default=list(df['hazard_priority'].unique()))
        with col_f2:
            sel_road = st.multiselect("Road", options=list(df['road_name'].unique()), default=list(df['road_name'].unique())[:4])
        with col_f3:
            min_psi = st.slider("Minimum PSI", 0.0, 100.0, 0.0, 5.0)

        filtered_df = df[
            (df['hazard_priority'].isin(sel_priority)) &
            (df['road_name'].isin(sel_road)) &
            (df['pothole_severity_index'] >= min_psi)
        ]

        metrics = st.columns(4)
        metrics[0].metric("Records", len(filtered_df))
        metrics[1].metric("P1 records", int(filtered_df['hazard_priority'].str.contains('P1', na=False).sum()))
        metrics[2].metric("Potholes", int(filtered_df['num_potholes'].sum()))
        metrics[3].metric("Estimated cost", f"₹{filtered_df['estimated_repair_cost_inr'].sum():,.0f}")

        map_data = filtered_df[['gps_latitude', 'gps_longitude']].rename(
            columns={'gps_latitude': 'lat', 'gps_longitude': 'lon'}
        )
        st.map(map_data, zoom=11)
        st.dataframe(filtered_df, use_container_width=True, height=280)
    else:
        st.info("Telemetry data is not available in the Data folder.")

elif nav_mode == "Repair estimate":
    st.subheader("Repair estimate")
    st.caption("Illustrative estimate only. Confirm unit rates with current local schedules before use.")

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        corridor_name = st.selectbox("Road section", [
            "Road section 1",
            "Road section 2",
            "Road section 3",
            "Road section 4"
        ])
        num_pits = st.number_input("Number of potholes", min_value=1, max_value=500, value=12)
        avg_depth = st.slider("Average depth (cm)", 2.0, 20.0, 7.5, 0.5)
        avg_diameter = st.slider("Average diameter (cm)", 20.0, 150.0, 55.0, 5.0)
        asphalt_mix = st.selectbox("Repair material", [
            "Cold Mix Polymer Modified Bitumen (Rapid Infill)",
            "Hot-Mix Asphalt Concrete (HMA Grade II)",
            "Stone Matrix Asphalt (Heavy Axle Load)",
            "Mastic Asphalt with Pre-coated Chippings"
        ])

    radius_m = (avg_diameter / 2.0) / 100.0
    area_per_pit = 3.14159 * (radius_m ** 2)
    total_area_sqm = area_per_pit * num_pits
    vol_cum = total_area_sqm * (avg_depth / 100.0)
    density_asphalt = 2.35
    asphalt_tonnage = vol_cum * density_asphalt
    rate_per_ton = 6500 if "Hot-Mix" in asphalt_mix else 8200
    material_cost = int(asphalt_tonnage * rate_per_ton)
    labor_machinery = int(material_cost * 0.45 + (num_pits * 250))
    traffic_management = int(material_cost * 0.12 + 2500)
    total_estimated_budget = material_cost + labor_machinery + traffic_management

    with b_col2:
        st.markdown(f"**Road section:** {corridor_name}")
        st.metric("Estimated total", f"₹{total_estimated_budget:,}")
        estimate_columns = st.columns(2)
        estimate_columns[0].metric("Repair area", f"{total_area_sqm:.2f} m²")
        estimate_columns[1].metric("Material quantity", f"{asphalt_tonnage:.2f} tonnes")
        st.write(f"Material: ₹{material_cost:,}")
        st.write(f"Labor and equipment: ₹{labor_machinery:,}")
        st.write(f"Traffic management: ₹{traffic_management:,}")

elif nav_mode == "Model information":
    st.subheader("Model information")
    st.caption("Training and visualization artifacts available in this checkout.")

    tab1, tab2, tab3 = st.tabs(["Grad-CAM", "Training results", "Model comparison"])

    with tab1:
        st.markdown("##### Gradient-weighted class activation mapping")
        st.write("Grad-CAM visualizes image regions associated with a model prediction. Treat it as an inspection aid, not as evidence of model reasoning.")
        fig5_path = os.path.join(BASE_DIR, "artifacts_build", "figure5_gradcam.png")
        if os.path.exists(fig5_path):
            st.image(fig5_path, use_container_width=True, caption="Grad-CAM example")
        else:
            st.info("No saved Grad-CAM figure was found in artifacts_build.")

    with tab2:
        st.markdown("##### Training curves and confusion matrix")
        fig1_path = os.path.join(BASE_DIR, "artifacts_build", "figure1_yolo_training.png")
        fig2_path = os.path.join(BASE_DIR, "artifacts_build", "figure2_confusion_matrix.png")
        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists(fig1_path):
                st.image(fig1_path, use_container_width=True, caption="Training curves")
            else:
                st.info("No training-curve image was found.")
        with c2:
            if os.path.exists(fig2_path):
                st.image(fig2_path, use_container_width=True, caption="Confusion matrix")
            else:
                st.info("No confusion-matrix image was found.")

    with tab3:
        st.markdown("##### Model comparison and architecture")
        fig3_path = os.path.join(BASE_DIR, "artifacts_build", "figure3_baseline_comparison.png")
        fig4_path = os.path.join(BASE_DIR, "artifacts_build", "figure4_architecture.png")
        if os.path.exists(fig3_path):
            st.image(fig3_path, use_container_width=True, caption="Model comparison")
        else:
            st.info("No model-comparison image was found.")
        if os.path.exists(fig4_path):
            st.image(fig4_path, use_container_width=True, caption="System architecture")

elif nav_mode == "Project notes":
    st.subheader("Project components")
    st.markdown("""
| Component | Description |
| :--- | :--- | :--- |
| Detection | Ultralytics model inference with an image-processing fallback |
| Severity | Heuristic severity score and repair estimate |
| API | FastAPI endpoints for health, detection, and telemetry |
| Web interface | Streamlit dashboard and static progressive web app |
| Visualization | Annotated detections and Grad-CAM view |
""")

st.divider()
st.caption("RoadGuard · CSE471 computer vision project")
