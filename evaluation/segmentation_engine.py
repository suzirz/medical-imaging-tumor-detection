import os
import numpy as np
import cv2
import torch
from typing import Dict, Any, Tuple
from models.attention_unet import AttentionUNet

class TumorSegmentationEngine:
    """
    Inference and Metric Analysis Engine for Attention U-Net Semantic Segmentation.
    
    Transforms MRI slices into pixel-accurate lesion masks:
    - Computes pixel-level probability maps via Sigmoid activations
    - Traces sub-millimeter outer neoplastic boundary contours
    - Extracts multi-scale Attention Gate saliency maps
    - Evaluates geometric compactness, lesion centroid, and cross-sectional area
    """
    def __init__(self, model_checkpoint: str = None, mm_per_px: float = 0.47):
        self.mm_per_px = mm_per_px
        self.device = torch.device("cpu")
        self.model = AttentionUNet(in_channels=3, out_channels=1).to(self.device)
        self.is_trained = False
        
        if model_checkpoint is None:
            default_ckpt = "models_checkpoint/attention_unet_best.pth"
            if os.path.exists(default_ckpt):
                model_checkpoint = default_ckpt

        if model_checkpoint and os.path.exists(model_checkpoint):
            try:
                self.model.load_state_dict(torch.load(model_checkpoint, map_location=self.device))
                self.is_trained = True
            except Exception:
                self.is_trained = False
        self.model.eval()

    def segment(
        self,
        mri_rgb: np.ndarray,
        threshold: float = 0.50,
        mask_color: Tuple[int, int, int] = (239, 68, 68),
        alpha: float = 0.45
    ) -> Dict[str, Any]:
        h_orig, w_orig = mri_rgb.shape[:2]
        
        # Preprocessing: resize to (224, 224) and normalize
        resized = cv2.resize(mri_rgb, (224, 224)).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_img = (resized - mean) / std
        tensor_in = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logits, att_maps = self.model(tensor_in, return_attention=True)
            prob_map_224 = torch.sigmoid(logits)[0, 0].cpu().numpy()

        # Resize probability map back to original scan dimensions
        prob_map = cv2.resize(prob_map_224, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
        binary_mask = (prob_map >= threshold).astype(np.uint8)

        # Multi-Scale Attention Gate Visualization (Combine Gate 1 and Gate 2)
        ag1 = att_maps[0][0, 0].cpu().numpy() # (224, 224)
        ag2 = att_maps[1][0, 0].cpu().numpy() # (112, 112)
        ag2_res = cv2.resize(ag2, (224, 224))
        composite_att = (ag1 + ag2_res) / 2.0
        composite_att = cv2.resize(composite_att, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        composite_att = (composite_att - composite_att.min()) / (composite_att.max() - composite_att.min() + 1e-8)

        # Render Attention Gate Heatmap (JET colormap)
        att_heatmap = cv2.applyColorMap((composite_att * 255).astype(np.uint8), cv2.COLORMAP_JET)
        att_heatmap_rgb = cv2.cvtColor(att_heatmap, cv2.COLOR_BGR2RGB)
        att_overlay = cv2.addWeighted(mri_rgb, 0.55, att_heatmap_rgb, 0.45, 0)

        # Render Pixel-Mask Overlay
        overlay = mri_rgb.copy()
        color_mask = np.zeros_like(mri_rgb)
        color_mask[binary_mask == 1] = mask_color

        # Alpha blending where mask is active
        blended = cv2.addWeighted(mri_rgb, 1.0 - alpha, color_mask, alpha, 0)
        overlay[binary_mask == 1] = blended[binary_mask == 1]

        # Find external contours of predicted tumor lesion
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2) # White contour boundary
        cv2.drawContours(overlay, contours, -1, mask_color, 1)       # Colored inner boundary

        # Geometric & Morphometric Calculations
        pixel_count = int(np.sum(binary_mask))
        has_tumor = pixel_count > 25

        if has_tumor and len(contours) > 0:
            # Largest contour
            largest_cnt = max(contours, key=cv2.contourArea)
            area_px = cv2.contourArea(largest_cnt)
            perimeter_px = cv2.arcLength(largest_cnt, True)
            
            # Calibration to mm and cm²
            area_mm2 = area_px * (self.mm_per_px ** 2)
            area_cm2 = area_mm2 / 100.0
            perimeter_mm = perimeter_px * self.mm_per_px
            
            # Compactness / Sphericity index: 4*pi*Area / P^2
            compactness = (4.0 * np.pi * area_px) / (perimeter_px ** 2 + 1e-6)
            compactness = min(max(float(compactness), 0.0), 1.0)
            
            # Centroid
            M = cv2.moments(largest_cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = w_orig // 2, h_orig // 2

            # Draw Centroid Crosshair
            cv2.drawMarker(overlay, (cx, cy), (56, 189, 248), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)

            # Mean Dice Confidence inside the predicted mask
            mean_confidence = float(np.mean(prob_map[binary_mask == 1]))
        else:
            area_px = 0.0
            area_mm2 = 0.0
            area_cm2 = 0.0
            perimeter_mm = 0.0
            compactness = 1.0
            cx, cy = w_orig // 2, h_orig // 2
            mean_confidence = 0.0

        return {
            "has_tumor": has_tumor,
            "binary_mask": binary_mask,
            "mask_overlay": overlay,
            "attention_overlay": att_overlay,
            "probability_map": prob_map,
            "pixel_count": pixel_count,
            "area_mm2": float(area_mm2),
            "area_cm2": float(area_cm2),
            "perimeter_mm": float(perimeter_mm),
            "compactness": float(compactness),
            "centroid": (cx, cy),
            "mean_confidence": float(mean_confidence),
            "threshold_used": threshold
        }
