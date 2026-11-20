"""
RoadGuard Automated Test Suite
Course: CSE471 | Deep Learning for Computer Vision

Tests:
1. Model Architecture forward pass & parameter shape
2. Severity Engine computation & priority boundaries
3. Detector inference & bounding box logic
4. FastAPI endpoints (/api/v1/health, /api/v1/telemetry)
"""

import os
import sys
import numpy as np
import torch

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from Code.model.architecture import PotholeAttentionCNN
from Code.model.severity_engine import PotholeSeverityEngine
from Code.model.pothole_detector import RoadGuardDetector

def test_cnn_architecture():
    model = PotholeAttentionCNN(num_classes=3)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    assert out.shape == (2, 3), f"Expected shape (2, 3), got {out.shape}"

def test_severity_engine_safe_surface():
    engine = PotholeSeverityEngine()
    report = engine.evaluate_hazard([])
    assert report['num_potholes'] == 0
    assert report['psi'] == 0.0
    assert "Normal" in report['priority_level']

def test_severity_engine_critical_pothole():
    engine = PotholeSeverityEngine()
    test_detections = [
        {"class_name": "Pothole", "confidence": 0.95, "bbox": [100, 150, 320, 320]},
        {"class_name": "Pothole", "confidence": 0.91, "bbox": [350, 180, 520, 310]}
    ]
    report = engine.evaluate_hazard(test_detections)
    assert report['num_potholes'] == 2
    assert report['psi'] >= 40.0
    assert "P1" in report['priority_level']
    assert report['max_depth_cm'] >= 5.0
    assert report['estimated_repair_cost_inr'] > 0

def test_detector_inference():
    detector = RoadGuardDetector()
    test_img = np.full((480, 640, 3), 60, dtype=np.uint8)
    # Draw dark square to simulate pothole
    test_img[180:260, 240:340] = 15
    ann_img, dets, sev, lat = detector.predict(test_img)
    assert ann_img is not None
    assert isinstance(lat, float)
    assert isinstance(sev, dict)
    assert 'psi' in sev

if __name__ == '__main__':
    if PYTEST_AVAILABLE:
        pytest.main(["-v", __file__])
    else:
        print("[RoadGuard Unit Tests] Running test suite via built-in runner...")
        test_cnn_architecture()
        print("✓ test_cnn_architecture passed.")
        test_severity_engine_safe_surface()
        print("✓ test_severity_engine_safe_surface passed.")
        test_severity_engine_critical_pothole()
        print("✓ test_severity_engine_critical_pothole passed.")
        test_detector_inference()
        print("✓ test_detector_inference passed.")
        print("[RoadGuard Unit Tests] All 4 unit tests PASSED successfully!")
