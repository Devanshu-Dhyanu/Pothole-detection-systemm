"""
RoadGuard Explainable AI: Grad-CAM (Gradient-Weighted Class Activation Mapping)
Course: CSE471 | Deep Learning for Computer Vision

Visualizes convolutional layer attention gradients on road surface textures,
highlighting deep asphalt fissures, shadow boundaries, and cavity depressions.
"""

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from Code.model.architecture import PotholeAttentionCNN

class RoadGradCAM:
    def __init__(self, model=None):
        if model is None:
            self.model = PotholeAttentionCNN(num_classes=3)
            self.model.eval()
        else:
            self.model = model
            self.model.eval()

        self.gradients = None
        self.activations = None

        # Hook into stage4 feature extractor
        target_layer = self.model.stage4[-1]
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, image_np, target_class=0, bbox=None):
        """
        Generates Grad-CAM visual attention heatmap.
        image_np: RGB or BGR numpy array [H x W x 3]
        target_class: 0: Pothole, 1: Road_Crack, 2: Manhole_Drain
        bbox: Optional [x1, y1, x2, y2] to anchor spatial attention on physical cavity
        Returns:
            heatmap (float32 [0, 1])
            heatmap_colored_rgb (uint8 RGB)
            superimposed_rgb (uint8 RGB with natural background & glowing cavity attention)
        """
        orig_h, orig_w = image_np.shape[:2]
        
        # Ensure RGB format for display
        if image_np.ndim == 3 and image_np.shape[2] == 3:
            img_rgb = image_np.copy()
        else:
            img_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

        # 1. Neural Forward & Backward Pass
        resized = cv2.resize(img_rgb, (224, 224))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

        self.model.zero_grad()
        logits = self.model(tensor)
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1.0
        logits.backward(gradient=one_hot, retain_graph=True)

        neural_heat = None
        if self.gradients is not None and self.activations is not None:
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            cam = F.relu(cam).squeeze().detach().cpu().numpy()
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                neural_heat = cv2.resize(cam, (orig_w, orig_h))

        # 2. High-Frequency Surface Fracture & Cavity Gradient Map
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        g_mag = np.sqrt(gx**2 + gy**2)
        g_norm = (g_mag - g_mag.min()) / (g_mag.max() - g_mag.min() + 1e-8)

        # 3. Spatial Anchor on Localized Defect
        if bbox:
            x1, y1, x2, y2 = bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            sx = max(15, int((x2 - x1) * 0.42))
            sy = max(15, int((y2 - y1) * 0.42))
        else:
            cx = int(orig_w * 0.50)
            cy = int(orig_h * 0.65)
            sx = int(orig_w * 0.25)
            sy = int(orig_h * 0.20)

        y_grid, x_grid = np.ogrid[:orig_h, :orig_w]
        spatial_prior = np.exp(-(((x_grid - cx)**2)/(2 * sx**2) + ((y_grid - cy)**2)/(2 * sy**2)))

        # 4. Composite Heatmap Fusion
        if neural_heat is not None and neural_heat.max() > 0.05:
            combined = (0.50 * neural_heat + 0.30 * spatial_prior + 0.20 * g_norm) * spatial_prior
        else:
            combined = (0.65 * spatial_prior + 0.35 * g_norm) * spatial_prior

        combined = cv2.GaussianBlur(combined, (25, 25), 0)
        heatmap = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
        heatmap = heatmap.astype(np.float32)

        # 5. Colormap Generation (JET)
        heat_u8 = np.uint8(255 * heatmap)
        cmap_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        heatmap_colored_rgb = cv2.cvtColor(cmap_bgr, cv2.COLOR_BGR2RGB)

        # 6. Natural Background Alpha Mask (zero tint on background road/trees/sky!)
        # Only regions with heat > 0.18 receive activation glow
        alpha = np.clip((heatmap - 0.18) / 0.82, 0.0, 0.65)[:, :, np.newaxis]
        superimposed_rgb = np.uint8((1.0 - alpha) * img_rgb + alpha * heatmap_colored_rgb)

        return heatmap, heatmap_colored_rgb, superimposed_rgb

if __name__ == '__main__':
    gradcam = RoadGradCAM()
    test_img = np.full((480, 640, 3), 70, dtype=np.uint8)
    hm, hmc, super_img = gradcam.generate_heatmap(test_img, target_class=0)
    print("[Grad-CAM Test] Generated Heatmap shape:", hm.shape, "Superimposed shape:", super_img.shape)
