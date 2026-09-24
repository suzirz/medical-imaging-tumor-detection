"""
NeuroScan AI — Clinical MRI Brain Tumor Detection, Segmentation & Explainability Suite
Main application orchestrator delegating UI presentation to modular components in `ui/`.
"""
import streamlit as st

from ui import (
    apply_clinical_theme,
    render_patient_sidebar,
    render_workstation_tab,
    render_segmentation_tab,
    render_contour_tab,
    render_registry_tab
)

# 1. Configure Streamlit Viewport & Page Metadata
st.set_page_config(
    page_title="NeuroScan AI — Brain Tumor Detection & Diagnostic Pipeline",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Initialize Session State for Multi-Scan Diagnostic History
if "diagnostic_history" not in st.session_state:
    st.session_state.diagnostic_history = []

# 3. Inject High-End Clinical Workstation Design System
apply_clinical_theme()

# 4. Render Patient History & Diagnostic Log Sidebar
render_patient_sidebar()

# 5. Render Main Application Header
st.markdown("""
<div class="clinical-header">
    <div class="header-badge">Clinical Decision Support System · Diagnostic Pipeline</div>
    <h1 class="header-title">NeuroScan MRI Tumor Detection & Explainability Suite</h1>
    <p class="header-subtitle">Deep learning inference with automated skull stripping, contour cropping, and gradient attribution (Grad-CAM++).</p>
</div>
""", unsafe_allow_html=True)

# 6. Render Workflow Navigation Tabs
tabs = st.tabs([
    "Clinical Diagnostic Workstation",
    "Deep Semantic Segmentation (Attention U-Net)",
    "Contour Cropping & Skull Stripping",
    "Neural Architecture & Benchmark Registry"
])

# Tab 1: Clinical Diagnostic Workstation (Consensus, ViT, DenseNet, EfficientNet, Lightweight, PACS Calipers, Reports)
with tabs[0]:
    eval_img, filename = render_workstation_tab()

# Tab 2: Deep Semantic Segmentation (Attention U-Net Pixel-Level Prediction)
with tabs[1]:
    render_segmentation_tab(eval_img, filename)

# Tab 3: Contour Cropping & Morphological Skull Stripping Pipeline
with tabs[2]:
    render_contour_tab(eval_img)

# Tab 4: Neural Architecture & Multi-Model Benchmark Registry
with tabs[3]:
    render_registry_tab()
