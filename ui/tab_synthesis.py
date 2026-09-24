"""
Tab: Virtual Contrast Synthesis & Cross-Modality Translation.
Simulates Gadolinium contrast uptake and T2-FLAIR edema mapping without intravenous injection.
"""
import streamlit as st
from PIL import Image
import numpy as np

from evaluation.contrast_synthesizer import VirtualContrastSynthesizer

def render_synthesis_tab(eval_img: Image.Image, filename: str):
    """
    Renders the Virtual Contrast Synthesis and Cross-Modality Translation tab.
    """
    st.markdown("""
    <div style="margin-bottom: 1.25rem;">
        <span class="header-badge">Generative Deep Learning · Virtual Gadolinium Physics</span>
        <h2 style="font-size: 1.4rem; font-weight: 700; color: #f8fafc; margin: 0.35rem 0 0.15rem 0;">Virtual Contrast Synthesis & Cross-Modality Translation</h2>
        <p style="font-size: 0.85rem; color: #94a3b8; margin: 0;">Generates Virtual T1-Contrast (T1ce) and Virtual T2-FLAIR sequences directly from unenhanced T1-weighted MRI—eliminating the need for invasive intravenous Gadolinium injection in patients with renal failure or acute allergy risk.</p>
    </div>
    """, unsafe_allow_html=True)

    # Deduce active pathology from session state
    last_analysis = st.session_state.get("last_analysis", None)
    if last_analysis and last_analysis.get("scan_id") == filename:
        pathology = last_analysis.get("pathology", "Meningioma")
        has_lesion = last_analysis.get("is_tumor", True)
    else:
        pathology = "Meningioma" if "meningioma" in filename.lower() else (
            "Glioma" if "glioma" in filename.lower() else (
                "Pituitary Adenoma" if "pituitary" in filename.lower() else (
                    "Normal Tissue" if "notumor" in filename.lower() else "Meningioma"
                )
            )
        )
        has_lesion = "normal" not in pathology.lower()

    # Controls
    ccol1, ccol2 = st.columns([1.5, 1])
    with ccol1:
        dose_factor = st.slider(
            "Simulated Gadolinium Contrast Dose Multiplier",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Standard clinical dose corresponds to 1.0x (0.1 mmol/kg body weight). Double dose (2.0x) is often used for subtle metastatic screening."
        )
    with ccol2:
        st.markdown(f"**Target Pathology Dynamics:** `{pathology}`")
        st.caption(f"Active Scan: {filename}")

    synthesizer = VirtualContrastSynthesizer()
    syn_result = synthesizer.synthesize(
        np.array(eval_img),
        dose_multiplier=dose_factor,
        pathology=pathology,
        has_lesion=has_lesion
    )

    # 4-Panel Quad View
    st.markdown("#### Cross-Modality Multi-Sequence Quad-Panel")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.image(syn_result["plain_t1"], caption="1. Plain T1 (Non-Contrast)", width='stretch')
    with p2:
        st.image(syn_result["virtual_t1ce"], caption="2. Virtual T1-Contrast (T1ce)", width='stretch')
    with p3:
        st.image(syn_result["virtual_flair"], caption="3. Virtual T2-FLAIR (Edema Halo)", width='stretch')
    with p4:
        st.image(syn_result["subtraction_map"], caption="4. Subtraction Map (ΔSI Uptake)", width='stretch')

    # Perfusion Metrics
    st.markdown("#### Quantitative Blood-Brain Barrier (BBB) & Perfusion Dynamics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("BBB Permeability Index (Ktrans)", f"{syn_result['bbb_permeability_index']:.3f}", delta="Disrupted Barrier" if syn_result['bbb_permeability_index'] > 0.5 else "Intact")
    m2.metric("Enhancement Pattern", syn_result["perfusion_pattern"][:28] + "...")
    m3.metric("Perilesional Edema Area", syn_result["edema_volume_estimate"])
    m4.metric("Simulated Contrast Equiv", syn_result["dose_administered"])
