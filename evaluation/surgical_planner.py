"""
Neurosurgical Resection Planner & Safe Corridor Simulator (2D Heuristic Prototype).
Calculates approximate Euclidean distances to 2D canonical functional landmarks
on normalized slice coordinates. NOTE: This is a geometric proof-of-concept,
NOT a stereotactic surgical neuronavigation system (no MNI registration or DTI tractography).
"""
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2

class NeurosurgicalPlanner:
    """
    Experimental 2D spatial guidance heuristic mapping lesion centroids against
    canonical coordinates on a 256x256 normalized axial slice.
    """

    def __init__(self, mm_per_px: float = 0.47):
        self.mm_per_px = mm_per_px
        # Canonical anatomical coordinates on normalized 256x256 cranial slice
        self.functional_landmarks = {
            "Primary Motor Strip (Precentral Gyrus)": {"coords": (128, 98), "critical_margin_mm": 10.0, "color": (239, 68, 68)},
            "Broca's Speech Area (Left Inferior Frontal)": {"coords": (85, 118), "critical_margin_mm": 10.0, "color": (168, 85, 247)},
            "Wernicke's Receptive Area (Left Post-Temporal)": {"coords": (78, 155), "critical_margin_mm": 10.0, "color": (59, 130, 246)},
            "Optic Radiation / Visual Pathway": {"coords": (128, 185), "critical_margin_mm": 8.0, "color": (245, 158, 11)}
        }

    def plan_resection(
        self,
        img_np: np.ndarray,
        centroid: Tuple[int, int],
        major_mm: float,
        pathology: str = "Meningioma",
        has_lesion: bool = True
    ) -> Dict[str, Any]:
        """
        Calculates distances to eloquent structures, assigns surgical risk tiers,
        and generates an annotated corridor simulation overlay.
        """
        h, w = img_np.shape[:2]
        cx, cy = centroid
        radius_px = max(10, int((major_mm / self.mm_per_px) / 2.0)) if has_lesion else 0

        # Scale landmarks to image resolution
        sx, sy = w / 256.0, h / 256.0

        landmark_readouts = []
        min_dist_mm = 999.0
        nearest_structure = "None"

        overlay = img_np.copy()
        if overlay.ndim == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)

        for name, data in self.functional_landmarks.items():
            lx, ly = int(data["coords"][0] * sx), int(data["coords"][1] * sy)
            color = data["color"]

            # Compute boundary-to-boundary distance
            center_dist_px = np.sqrt((cx - lx)**2 + (cy - ly)**2)
            edge_dist_px = max(0.0, center_dist_px - radius_px)
            edge_dist_mm = round(edge_dist_px * self.mm_per_px, 1)

            if edge_dist_mm < min_dist_mm:
                min_dist_mm = edge_dist_mm
                nearest_structure = name

            is_critical = edge_dist_mm < data["critical_margin_mm"]
            landmark_readouts.append({
                "structure_name": name,
                "distance_mm": edge_dist_mm,
                "is_critical": is_critical,
                "alert": "CRITICAL PROXIMITY (<10mm)" if is_critical else "Safe Margin"
            })

            # Draw functional zone on overlay
            cv2.circle(overlay, (lx, ly), int(12 * sx), color, 2)
            cv2.putText(overlay, name.split()[0], (lx - 20, ly - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

        # Draw Lesion Boundary and Safe Corridor Vector
        if has_lesion and "normal" not in pathology.lower():
            # Lesion core
            cv2.circle(overlay, (cx, cy), radius_px, (239, 68, 68), 2)
            cv2.drawMarker(overlay, (cx, cy), (239, 68, 68), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

            # Determine entry burr-hole trajectory from nearest skull surface
            # Find vector pointing outward from skull center (128, 128)
            dx = cx - int(128 * sx)
            dy = cy - int(128 * sy)
            norm = max(1.0, np.sqrt(dx**2 + dy**2))
            dir_x, dir_y = dx / norm, dy / norm

            entry_x = int(cx + dir_x * (radius_px + 45))
            entry_y = int(cy + dir_y * (radius_px + 45))

            # Draw Safe Corridor Trajectory Line
            cv2.arrowedLine(overlay, (entry_x, entry_y), (cx, cy), (16, 185, 129), 2, tipLength=0.25)
            cv2.putText(overlay, "BURR-HOLE ENTRY", (entry_x - 30, entry_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (16, 185, 129), 1)

            # Surgical Feasibility Assessment
            if min_dist_mm < 10.0:
                risk_tier = "HIGH RISK — ELOQUENT CORTEX INVOLVEMENT"
                strategy = "Awake Craniotomy with Direct Cortical Stimulation (DCS) & Intraoperative Neuronavigation"
                safety_score = 64
                corridor_name = "Stereotactic Micro-dissection Corridor"
            elif min_dist_mm < 25.0:
                risk_tier = "MODERATE RISK — BORDERLINE MARGIN"
                strategy = "Image-Guided Neuronavigation with DTI Tractography Integration"
                safety_score = 85
                corridor_name = "Trans-sulcal Microsurgical Access"
            else:
                risk_tier = "STANDARD / LOW RISK — CONVEXITY CLEARANCE"
                strategy = "Standard Craniotomy for Complete Gross Total Resection"
                safety_score = 96
                corridor_name = "Direct Transcortical / Convexity Corridor"
        else:
            risk_tier = "N/A — NO SURGICAL LESION DETECTED"
            strategy = "Conservative clinical surveillance. No craniotomy indicated."
            safety_score = 100
            corridor_name = "Non-Surgical"

        return {
            "overlay_image": overlay,
            "min_distance_to_eloquence_mm": min_dist_mm if has_lesion else 0.0,
            "nearest_critical_structure": nearest_structure,
            "surgical_risk_tier": risk_tier,
            "recommended_technique": strategy,
            "resection_safety_index": safety_score,
            "planned_corridor": corridor_name,
            "landmarks_readout": landmark_readouts
        }
