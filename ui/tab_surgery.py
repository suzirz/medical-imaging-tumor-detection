"""
Tab: Neurosurgical Resection Planner & Safe Corridor Simulator.
Maps proximity to eloquent cortex (Motor strip, Broca, Wernicke) and guides safe craniotomy access.
"""
import streamlit as st
from PIL import Image
import numpy as np

from evaluation.surgical_planner import NeurosurgicalPlanner

def render_surgery_tab(eval_img: Image.Image, filename: str):
    """
    Renders the Neurosurgical Resection Planner & Safe Corridor Simulator Tab.
    """
    st.markdown("""
    <div style="margin-bottom: 1.25rem;">
        <span class="header-badge">Spatial Surgical AI · Functional Craniotomy Guidance</span>
        <h2 style="font-size: 1.4rem; font-weight: 700; color: #f8fafc; margin: 0.35rem 0 0.15rem 0;">Neurosurgical Resection Planner & Safe Corridor Simulator</h2>
        <p style="font-size: 0.85rem; color: #94a3b8; margin: 0;">Quantifies spatial distance clearances between lesion boundaries and critical eloquent functional cortex (Motor Strip, Broca, Wernicke, Optic Radiations) to simulate optimal craniotomy trajectories.</p>
    </div>
    """, unsafe_allow_html=True)

    # Retrieve active scan context
    last_analysis = st.session_state.get("last_analysis", None)
    if last_analysis and last_analysis.get("scan_id") == filename:
        pathology = last_analysis.get("pathology", "Meningioma")
        has_lesion = last_analysis.get("is_tumor", True)
        major_mm = last_analysis.get("morphometry", {}).get("major_mm", 28.5)
    else:
        pathology = "Meningioma" if "meningioma" in filename.lower() else (
            "Glioma" if "glioma" in filename.lower() else (
                "Pituitary Adenoma" if "pituitary" in filename.lower() else "Meningioma"
            )
        )
        has_lesion = "normal" not in pathology.lower()
        major_mm = 28.5 if has_lesion else 0.0

    # Execute Surgical Planning Engine
    planner = NeurosurgicalPlanner(mm_per_px=0.47)
    
    # Calculate or deduce lesion centroid
    h, w = np.array(eval_img).shape[:2]
    # Default lesion centroid on fronto-parietal region if not in state
    centroid = (int(w * 0.62), int(h * 0.42)) if has_lesion else (int(w * 0.5), int(h * 0.5))

    plan = planner.plan_resection(
        np.array(eval_img),
        centroid=centroid,
        major_mm=major_mm,
        pathology=pathology,
        has_lesion=has_lesion
    )

    # Plan Summary Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Resection Safety Index", f"{plan['resection_safety_index']}%", delta="Feasible Resection" if plan['resection_safety_index'] > 75 else "High-Risk Corridor")
    m2.metric("Clearance to Eloquence", f"{plan['min_distance_to_eloquence_mm']} mm", delta=plan["nearest_critical_structure"].split()[0], delta_color="off")
    m3.metric("Surgical Access Corridor", plan["planned_corridor"])
    m4.metric("Risk Classification", plan["surgical_risk_tier"].split('—')[0].strip())

    # Two-Column Planning Console
    pcol1, pcol2 = st.columns([1.1, 1], gap="large")

    with pcol1:
        st.markdown("#### Intraoperative Surgical Map & Entry Trajectory")
        st.image(
            plan["overlay_image"],
            caption="Green Vector: Recommended Safe Entry Burr-Hole Corridor | Red/Blue Circles: Eloquent Cortex Zones",
            width='stretch'
        )

    with pcol2:
        st.markdown("#### Recommended Surgical Strategy & Protocol")
        st.info(f"**Operative Technique Directive:**\n\n{plan['recommended_technique']}")

        st.markdown("#### Millimeter Clearance to Functional Landmarks")
        for lm in plan["landmarks_readout"]:
            alert_badge = "badge-neg" if not lm["is_critical"] else "badge-pos"
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 10px; background: #0f172a; border: 1px solid #1e293b; border-radius: 6px; margin-bottom: 6px;">
                <span style="font-size: 0.8rem; color: #f8fafc; font-weight: 500;">{lm['structure_name']}</span>
                <div>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #38bdf8; margin-right: 8px;">{lm['distance_mm']} mm</span>
                    <span class="{alert_badge}">{lm['alert']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
