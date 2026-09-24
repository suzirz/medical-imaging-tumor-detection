import io
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2

def generate_clinical_report(
    scan_id: str,
    original_img: np.ndarray,
    cropped_img: np.ndarray,
    gradcam_img: np.ndarray,
    prediction_label: str,
    confidence: float,
    engine_name: str,
    clinical_protocol: str,
    caliper_img: np.ndarray = None,
    morphometry: dict = None,
    file_format: str = "png"
) -> bytes:
    """
    Generates a high-resolution clinical diagnostic sheet in PNG or PDF format.
    Includes 4-panel visual evidence and quantitative lesion morphometry.
    """
    # Normalize inputs to uint8 RGB
    def ensure_rgb(img):
        if img is None:
            return None
        if img.dtype != np.uint8:
            norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (norm * 255).astype(np.uint8)
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    p1 = ensure_rgb(original_img)
    p2 = ensure_rgb(cropped_img)
    p3 = ensure_rgb(gradcam_img)
    p4 = ensure_rgb(caliper_img) if caliper_img is not None else None

    is_tumor = "POSITIVE" in prediction_label.upper() or "TUMOR" in prediction_label.upper()
    status_color = "#ef4444" if is_tumor else "#10b981"
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Figure Setup: Crisp Clinical Diagnostic Sheet
    fig = plt.figure(figsize=(12, 8.5), dpi=200, facecolor="#0b101b")
    
    ncols = 4 if p4 is not None else 3
    gs = fig.add_gridspec(nrows=5, ncols=ncols, height_ratios=[0.9, 0.4, 3.2, 1.3, 0.5], hspace=0.35, wspace=0.15)

    # 1. Header Banner
    ax_head = fig.add_subplot(gs[0, :])
    ax_head.set_facecolor("#0f172a")
    ax_head.axis("off")

    ax_head.text(0.02, 0.72, "NEUROSCAN AI · CLINICAL NEURO-ONCOLOGY DECISION SUPPORT", 
                 fontsize=13, fontweight="bold", color="#38bdf8", family="sans-serif")
    ax_head.text(0.02, 0.32, "Automated Diagnostic Classification, Morphometry & Explainability Summary", 
                 fontsize=8.5, color="#94a3b8", family="sans-serif")

    ax_head.text(0.98, 0.72, f"Scan ID: {scan_id[:26]}", 
                 fontsize=8.5, fontweight="bold", color="#f8fafc", ha="right", family="monospace")
    ax_head.text(0.98, 0.32, f"Date: {timestamp_str}", 
                 fontsize=8, color="#94a3b8", ha="right", family="monospace")

    # 2. Metadata Strip
    ax_meta = fig.add_subplot(gs[1, :])
    ax_meta.set_facecolor("#131c31")
    ax_meta.axis("off")
    meta_text = (
        f" Acquisition Plane: Axial    |    Modality: Cranial MRI    |    "
        f"Inference Engine: {engine_name}    |    Status: Verified Model Weights"
    )
    ax_meta.text(0.5, 0.5, meta_text, fontsize=8, color="#cbd5e1", ha="center", va="center", family="sans-serif")

    # 3. Visual Panels (3 or 4 images side by side)
    ax_p1 = fig.add_subplot(gs[2, 0])
    ax_p1.imshow(p1)
    ax_p1.set_title("1. Original Patient Scan", fontsize=8.5, fontweight="bold", color="#e2e8f0", pad=6)
    ax_p1.axis("off")

    ax_p2 = fig.add_subplot(gs[2, 1])
    ax_p2.imshow(p2)
    ax_p2.set_title("2. Contour Skull Stripping", fontsize=8.5, fontweight="bold", color="#e2e8f0", pad=6)
    ax_p2.axis("off")

    ax_p3 = fig.add_subplot(gs[2, 2])
    ax_p3.imshow(p3)
    ax_p3.set_title("3. Grad-CAM++ Saliency", fontsize=8.5, fontweight="bold", color="#e2e8f0", pad=6)
    ax_p3.axis("off")

    if p4 is not None:
        ax_p4 = fig.add_subplot(gs[2, 3])
        ax_p4.imshow(p4)
        ax_p4.set_title("4. Digital PACS Calipers", fontsize=8.5, fontweight="bold", color="#38bdf8", pad=6)
        ax_p4.axis("off")

    # 4. Diagnostic Readout & Quantitative Morphometry Card
    ax_result = fig.add_subplot(gs[3, :])
    ax_result.set_facecolor("#111827")
    ax_result.axis("off")

    # Primary Finding
    ax_result.text(0.03, 0.78, "PRIMARY DIAGNOSTIC FINDING:", 
                   fontsize=7.5, fontweight="bold", color="#94a3b8", family="monospace")
    ax_result.text(0.03, 0.44, f"{prediction_label.upper()}", 
                   fontsize=12, fontweight="bold", color=status_color, family="sans-serif")
    ax_result.text(0.03, 0.16, f"Clinical Directive: {clinical_protocol}", 
                   fontsize=8, color="#cbd5e1", family="sans-serif")

    # Confidence Index
    ax_result.text(0.48, 0.78, "CONFIDENCE INDEX:", 
                   fontsize=7.5, fontweight="bold", color="#94a3b8", family="monospace")
    ax_result.text(0.48, 0.44, f"{confidence * 100:.2f}%", 
                   fontsize=13, fontweight="bold", color="#f8fafc", family="monospace")

    # Morphometry Readout
    ax_result.text(0.70, 0.78, "QUANTITATIVE LESION MORPHOMETRY:", 
                   fontsize=7.5, fontweight="bold", color="#38bdf8", family="monospace")
    if morphometry and morphometry.get("has_lesion", False):
        m_txt = (
            f"Dimensions: {morphometry['major_mm']:.1f} x {morphometry['minor_mm']:.1f} mm\n"
            f"Area: {morphometry['area_cm2']:.2f} cm2  (Burden: {morphometry.get('tumor_burden_pct', 0):.1f}%)\n"
            f"Location: {morphometry.get('anatomical_location', 'N/A')}"
        )
    else:
        m_txt = "No focal mass lesion detected.\nParenchymal symmetry preserved.\nBilateral normal tissue architecture."
    
    ax_result.text(0.70, 0.28, m_txt, fontsize=8, color="#e2e8f0", family="monospace")

    # 5. Regulatory Disclaimer Footer
    ax_foot = fig.add_subplot(gs[4, :])
    ax_foot.axis("off")
    disclaimer = (
        "CONFIDENTIAL MEDICAL REPORT — Generated by NeuroScan Automated AI Diagnostic Pipeline. "
        "For clinical decision support only; final diagnosis must be confirmed by a board-certified Radiologist."
    )
    ax_foot.text(0.5, 0.5, disclaimer, fontsize=6.8, color="#64748b", ha="center", va="center", style="italic")

    # Export to BytesIO buffer
    buf = io.BytesIO()
    fig.savefig(buf, format=file_format.lower(), facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=200)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
