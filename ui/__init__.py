"""
UI Module Package for NeuroScan Clinical Workstation.
Separates concerns between styling, patient sidebar records, and individual workflow tabs.
"""
from .styles import apply_clinical_theme
from .sidebar import render_patient_sidebar
from .tab_workstation import render_workstation_tab
from .tab_segmentation import render_segmentation_tab
from .tab_vlm import render_vlm_tab
from .tab_retrieval import render_retrieval_tab
from .tab_contour import render_contour_tab
from .tab_registry import render_registry_tab

__all__ = [
    "apply_clinical_theme",
    "render_patient_sidebar",
    "render_workstation_tab",
    "render_segmentation_tab",
    "render_vlm_tab",
    "render_retrieval_tab",
    "render_contour_tab",
    "render_registry_tab"
]


