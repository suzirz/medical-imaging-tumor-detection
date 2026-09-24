import cv2
import numpy as np
from typing import Dict, Any, Tuple

class LesionMorphometryAnalyzer:
    """
    Quantitative Morphometry and Clinical Caliper Measurement Engine.
    
    Extracts anatomical metrics from saliency heatmaps and MRI tissue:
    - Longest lesion diameter (Major Axis in mm)
    - Perpendicular width (Minor Axis in mm)
    - Estimated 2D cross-sectional area (cm²)
    - Anatomical hemisphere and quadrant localization
    - Ratio of tumor volume to intracranial parenchyma (Tumor Burden %)
    - Digital DICOM/PACS-grade caliper measurement overlay visualization
    """
    def __init__(self, mm_per_px: float = 0.47):
        """
        Standard cranial MRI Field-of-View (FOV):
        Typically 240 mm across a 512x512 matrix ~= 0.47 mm/pixel.
        """
        self.mm_per_px = mm_per_px

    def analyze(self, mri_rgb: np.ndarray, heatmap: np.ndarray, is_tumor: bool) -> Dict[str, Any]:
        h, w = mri_rgb.shape[:2]
        
        # Ensure heatmap matches MRI dimensions
        if heatmap.shape[:2] != (h, w):
            heatmap_resized = cv2.resize(heatmap, (w, h))
        else:
            heatmap_resized = heatmap.copy()

        # Brain parenchyma mask
        gray = cv2.cvtColor(mri_rgb, cv2.COLOR_RGB2GRAY)
        _, brain_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)
        total_brain_px = max(cv2.countNonZero(brain_mask), 1)

        caliper_canvas = mri_rgb.copy()
        mid_x, mid_y = w // 2, h // 2

        # Draw anatomical mid-sagittal reference line (falx cerebri)
        for y in range(0, h, 14):
            cv2.line(caliper_canvas, (mid_x, y), (mid_x, min(y + 7, h)), (50, 65, 85), 1)

        if not is_tumor:
            return {
                "has_lesion": False,
                "major_mm": 0.0,
                "minor_mm": 0.0,
                "area_cm2": 0.0,
                "perimeter_mm": 0.0,
                "tumor_burden_pct": 0.0,
                "hemisphere": "Bilateral Symmetrical",
                "quadrant": "Normal Parenchyma",
                "anatomical_location": "No Structural Lesion Detected",
                "caliper_overlay": caliper_canvas
            }

        # Mask heatmap to stay inside intracranial vault
        masked_hm = heatmap_resized * (brain_mask.astype(np.float32) / 255.0)

        # Lesion thresholding
        thresh_val = 0.35
        binary_lesion = (masked_hm > thresh_val).astype(np.uint8) * 255
        
        # Refine with morphological opening to remove spurious pixels
        open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_lesion = cv2.morphologyEx(binary_lesion, cv2.MORPH_OPEN, open_k)

        contours, _ = cv2.findContours(binary_lesion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {
                "has_lesion": False,
                "major_mm": 0.0,
                "minor_mm": 0.0,
                "area_cm2": 0.0,
                "perimeter_mm": 0.0,
                "tumor_burden_pct": 0.0,
                "hemisphere": "Indeterminate",
                "quadrant": "Low Signal Intensity",
                "anatomical_location": "Diffuse / Non-Focal Signal",
                "caliper_overlay": caliper_canvas
            }

        largest_c = max(contours, key=cv2.contourArea)
        area_px = cv2.contourArea(largest_c)

        if area_px < 50:
            return {
                "has_lesion": False,
                "major_mm": 0.0,
                "minor_mm": 0.0,
                "area_cm2": 0.0,
                "perimeter_mm": 0.0,
                "tumor_burden_pct": 0.0,
                "hemisphere": "Indeterminate",
                "quadrant": "Sub-threshold signal",
                "anatomical_location": "Micro-focus (Sub-clinical threshold)",
                "caliper_overlay": caliper_canvas
            }

        # Centroid Calculation via Spatial Moments
        M = cv2.moments(largest_c)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = mid_x, mid_y

        # Minimum bounding rotated rectangle
        rect = cv2.minAreaRect(largest_c)
        (center_x, center_y), (dim_w, dim_h), angle = rect
        major_px = max(dim_w, dim_h)
        minor_px = min(dim_w, dim_h)

        major_mm = major_px * self.mm_per_px
        minor_mm = minor_px * self.mm_per_px
        area_cm2 = (area_px * (self.mm_per_px ** 2)) / 100.0
        perimeter_mm = cv2.arcLength(largest_c, True) * self.mm_per_px
        tumor_burden_pct = (area_px / total_brain_px) * 100.0

        # Anatomical Quadrant & Hemisphere Classification
        dx = cx - mid_x
        dy = cy - mid_y

        if abs(dx) < 18:
            hemisphere = "Central / Midline"
        elif dx > 0:
            hemisphere = "Right Hemisphere"
        else:
            hemisphere = "Left Hemisphere"

        if dy < -25:
            quadrant = "Frontal / Anterior"
        elif dy > 35:
            quadrant = "Occipital / Posterior"
        else:
            quadrant = "Parieto-Temporal"

        anatomical_location = f"{hemisphere} ({quadrant})"

        # --- PACS DIGITAL CALIPER RENDERING ---
        
        # 1. Subtle amber/cyan lesion region fill
        overlay = caliper_canvas.copy()
        cv2.drawContours(overlay, [largest_c], -1, (0, 180, 255), -1)
        caliper_canvas = cv2.addWeighted(overlay, 0.22, caliper_canvas, 0.78, 0)

        # 2. High-precision neon contour boundary
        cv2.drawContours(caliper_canvas, [largest_c], -1, (0, 240, 255), 2)

        # 3. Centroid Crosshairs
        cs = 10
        cv2.line(caliper_canvas, (cx - cs, cy), (cx + cs, cy), (56, 189, 248), 1, cv2.LINE_AA)
        cv2.line(caliper_canvas, (cx, cy - cs), (cx, cy + cs), (56, 189, 248), 1, cv2.LINE_AA)
        cv2.circle(caliper_canvas, (cx, cy), 3, (0, 255, 255), -1)

        # 4. Rotated Caliper Measurement Lines
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        p0, p1, p2, p3 = box[0], box[1], box[2], box[3]
        m1 = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
        m2 = ((p2[0] + p3[0]) // 2, (p2[1] + p3[1]) // 2)
        m3 = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        m4 = ((p3[0] + p0[0]) // 2, (p3[1] + p0[1]) // 2)

        # Major axis caliper (Yellow)
        cv2.line(caliper_canvas, m1, m2, (255, 215, 0), 2, cv2.LINE_AA)
        # Minor axis caliper (Cyan)
        cv2.line(caliper_canvas, m3, m4, (56, 189, 248), 2, cv2.LINE_AA)

        # Caliper Endpoints with target circles
        for pt in [m1, m2, m3, m4]:
            cv2.circle(caliper_canvas, pt, 4, (255, 255, 255), -1)
            cv2.circle(caliper_canvas, pt, 2, (15, 23, 42), -1)

        # 5. Professional Medical HUD Card
        hud_w, hud_h = 245, 96
        hud_bg = caliper_canvas[10:10+hud_h, 10:10+hud_w].copy()
        hud_panel = np.zeros_like(hud_bg)
        cv2.rectangle(hud_panel, (0, 0), (hud_w, hud_h), (11, 15, 26), -1)
        blended_hud = cv2.addWeighted(hud_panel, 0.88, hud_bg, 0.12, 0)
        cv2.rectangle(blended_hud, (0, 0), (hud_w, hud_h), (56, 189, 248), 1)
        caliper_canvas[10:10+hud_h, 10:10+hud_w] = blended_hud

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(caliper_canvas, "LESION MORPHOMETRY", (20, 27), font, 0.44, (56, 189, 248), 1, cv2.LINE_AA)
        cv2.putText(caliper_canvas, f"Major (L): {major_mm:.1f} mm", (20, 47), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(caliper_canvas, f"Minor (W): {minor_mm:.1f} mm", (20, 67), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(caliper_canvas, f"Area: {area_cm2:.2f} cm2 | {anatomical_location[:14]}", (20, 87), font, 0.38, (148, 163, 184), 1, cv2.LINE_AA)

        return {
            "has_lesion": True,
            "major_mm": float(major_mm),
            "minor_mm": float(minor_mm),
            "area_cm2": float(area_cm2),
            "perimeter_mm": float(perimeter_mm),
            "tumor_burden_pct": float(tumor_burden_pct),
            "hemisphere": hemisphere,
            "quadrant": quadrant,
            "anatomical_location": anatomical_location,
            "centroid": (cx, cy),
            "caliper_overlay": caliper_canvas
        }
