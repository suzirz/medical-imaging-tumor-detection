"""
Tab: DeepSurv Patient Survival Time Estimation & Prognostic Risk Curve.
Forecasts overall survival trajectories, Kaplan-Meier curves, and chemotherapeutic gain deltas.
"""
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from evaluation.survival_prognosticator import DeepSurvPrognosticator

def render_prognosis_tab(filename: str):
    """
    Renders the DeepSurv Patient Survival Trajectory and Kaplan-Meier curve tab.
    """
    st.markdown("""
    <div style="margin-bottom: 1.25rem;">
        <span class="header-badge">Deep Survival Analysis · Cox Proportional Hazards</span>
        <h2 style="font-size: 1.4rem; font-weight: 700; color: #f8fafc; margin: 0.35rem 0 0.15rem 0;">DeepSurv: Patient Survival Horizon & Prognostic Risk Curve</h2>
        <p style="font-size: 0.85rem; color: #94a3b8; margin: 0;">Multi-parametric survival time forecasting integrating patient age, Karnofsky Performance Scale (KPS), tumor burden, and molecular genetic status (IDH1/MGMT) into 5-year Kaplan-Meier survival curves.</p>
    </div>
    """, unsafe_allow_html=True)

    last_analysis = st.session_state.get("last_analysis", None)
    if last_analysis and last_analysis.get("scan_id") == filename:
        pathology = last_analysis.get("pathology", "Meningioma")
        tumor_area = last_analysis.get("morphometry", {}).get("area_cm2", 4.75)
    else:
        pathology = "Meningioma" if "meningioma" in filename.lower() else (
            "Glioma" if "glioma" in filename.lower() else (
                "Pituitary Adenoma" if "pituitary" in filename.lower() else "Meningioma"
            )
        )
        tumor_area = 4.75 if "normal" not in pathology.lower() else 0.0

    # Clinical Covariate Inputs
    st.markdown("#### Patient Clinical Demographics & Functional Performance Status")
    c1, c2, c3 = st.columns([1, 1, 1.2])

    with c1:
        patient_age = st.slider("Patient Age (Years)", min_value=18, max_value=85, value=54, step=1)
        resection_type = st.selectbox(
            "Anticipated Surgical Resection Extent",
            ["Gross Total Resection (Simpson I/II)", "Subtotal Resection (STR) / Biopsy Only"]
        )

    with c2:
        kps_choice = st.selectbox(
            "Karnofsky Performance Scale (KPS)",
            [
                "KPS 90-100 (Fully functional, normal activity)",
                "KPS 80 (Normal activity with minor symptoms)",
                "KPS 70 (Cares for self, unable to carry active work)",
                "KPS 50-60 (Requires considerable assistance)"
            ],
            index=1
        )
        kps_val_map = {
            "KPS 90-100 (Fully functional, normal activity)": 95,
            "KPS 80 (Normal activity with minor symptoms)": 80,
            "KPS 70 (Cares for self, unable to carry active work)": 70,
            "KPS 50-60 (Requires considerable assistance)": 55
        }
        kps_score = kps_val_map[kps_choice]

    with c3:
        st.markdown("**Molecular Genomic Determinants (From Radiogenomics):**")
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            is_idh_mut = st.checkbox("IDH1/IDH2 Mutant", value=True)
            is_1p19q = st.checkbox("1p/19q Co-deletion", value=False)
        with g_col2:
            is_mgmt_meth = st.checkbox("MGMT Methylated", value=True)

    # Execute DeepSurv Modeling
    prognosticator = DeepSurvPrognosticator()
    surv_res = prognosticator.estimate_survival(
        age=patient_age,
        kps=kps_score,
        pathology=pathology,
        tumor_area_cm2=tumor_area,
        resection_extent=resection_type,
        idh_mutant=is_idh_mut,
        mgmt_methylated=is_mgmt_meth,
        codeleted_1p19q=is_1p19q
    )

    # Key Prognostic Cards
    st.markdown("#### Forecasted Clinical Survival Horizons")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Median Overall Survival (OS)", f"{surv_res['median_os_months']} Mos", delta=surv_res["chemo_benefit_delta"], delta_color="normal")
    m2.metric("Median Progression-Free (PFS)", f"{surv_res['median_pfs_months']} Mos")
    m3.metric("5-Year Survival Probability", f"{surv_res['5_year_survival_pct']}%", delta=f"{surv_res['1_year_survival_pct']}% at 1-Yr", delta_color="off")
    m4.metric("Prognostic Hazard Ratio (HR)", f"{surv_res['hazard_ratio']}", delta=surv_res["risk_stratification"].split('(')[0], delta_color="inverse")

    # Render Kaplan-Meier Survival Curve using Matplotlib
    st.markdown("#### 5-Year Kaplan-Meier Survival Probability Curve")
    fig, ax = plt.subplots(figsize=(12, 3.8), facecolor="#090d16")
    ax.set_facecolor("#0f172a")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e293b")

    t_m = surv_res["timeline_months"]
    s_p = surv_res["survival_probabilities"]

    # Step plot for Kaplan-Meier
    ax.step(t_m, s_p, where="post", color="#38bdf8", linewidth=2.5, label="Predicted Survival Curve (DeepSurv)")
    ax.fill_between(t_m, s_p, step="post", color="#38bdf8", alpha=0.15)

    # Median OS marker
    if surv_res["median_os_months"] <= 60:
        ax.axvline(surv_res["median_os_months"], color="#f87171", linestyle="--", alpha=0.7, label=f"Median OS: {surv_res['median_os_months']} mos")

    ax.set_xlim(0, 60)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Timeline Since Diagnosis / Surgical Resection (Months)", color="#94a3b8", fontsize=10, labelpad=8)
    ax.set_ylabel("Survival Probability (%)", color="#94a3b8", fontsize=10, labelpad=8)
    ax.tick_params(colors="#cbd5e1", labelsize=9)
    ax.grid(True, linestyle=":", alpha=0.2, color="#ffffff")
    ax.legend(facecolor="#111827", edgecolor="#1f2937", labelcolor="#f8fafc", loc="lower left", fontsize=9)

    st.pyplot(fig)
    plt.close()
