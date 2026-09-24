"""
Tab: Native DICOM Medical PACS Ingestion & Radiologist Window-Leveling.
Parses native .dcm hospital scanner files and provides interactive radiologist window leveling presets.
"""
import io
import streamlit as st
import numpy as np
from PIL import Image

from evaluation.dicom_parser import DICOMPACSParser, WINDOW_PRESETS

def render_dicom_tab():
    """
    Renders the DICOM PACS Ingestion and Window-Level Preset Explorer Tab.
    """
    st.markdown("""
    <div style="margin-bottom: 1.25rem;">
        <span class="header-badge">Clinical PACS Engineering · Native DICOM Part 10</span>
        <h2 style="font-size: 1.4rem; font-weight: 700; color: #f8fafc; margin: 0.35rem 0 0.15rem 0;">Native DICOM Medical PACS Ingestion & Window-Leveling</h2>
        <p style="font-size: 0.85rem; color: #94a3b8; margin: 0;">Directly parses native 16-bit hospital scanner files (.dcm), extracts clinical acquisition metadata tags, and applies standardized radiologist window/level contrast curves.</p>
    </div>
    """, unsafe_allow_html=True)

    parser = DICOMPACSParser()

    dcol1, dcol2 = st.columns([1.2, 1], gap="large")

    with dcol1:
        st.markdown("#### DICOM File Acquisition")
        uploaded_dcm = st.file_uploader(
            "Upload Native DICOM Scan (.dcm)",
            type=["dcm"],
            key="dicom_uploader",
            help="Upload raw DICOM file from Siemens, GE, or Philips hospital scanners."
        )

        use_demo = False
        if not uploaded_dcm:
            st.info("No external .dcm file selected. You can test with the built-in clinical DICOM phantom.")
            use_demo = True

    # Parse file or create demonstration
    try:
        if uploaded_dcm:
            dcm_data = parser.parse_dicom(uploaded_dcm)
            dcm_source_name = uploaded_dcm.name
        else:
            demo_bytes = parser.create_demonstration_dicom()
            dcm_data = parser.parse_dicom(demo_bytes)
            dcm_source_name = "SIMULATED_CLINICAL_T1.dcm (Demonstration Phantom)"

        raw_px = dcm_data["raw_pixels"]
        meta = dcm_data["metadata"]

        with dcol2:
            st.markdown("#### Radiologist Window/Level Presets")
            preset_choice = st.selectbox(
                "Clinical Window Preset",
                list(WINDOW_PRESETS.keys())
            )
            preset_vals = WINDOW_PRESETS[preset_choice]

            w_slider = st.slider("Window Width (Contrast)", 10, 2000, preset_vals["width"], 5)
            l_slider = st.slider("Window Level (Brightness)", -200, 1000, preset_vals["level"], 5)

        # Apply Window Level
        windowed_rgb = parser.apply_window_level(raw_px, w_slider, l_slider)

        # Display Comparison View
        st.markdown("#### Radiologist Viewing Console")
        v1, v2 = st.columns(2)
        with v1:
            # Simple min-max for raw display
            min_v, max_v = np.min(raw_px), np.max(raw_px)
            raw_disp = ((raw_px - min_v) / (max_v - min_v + 1e-6) * 255.0).astype(np.uint8)
            st.image(raw_disp, caption=f"Raw Dynamic Range (Min: {min_v:.0f}, Max: {max_v:.0f})", width='stretch')
        with v2:
            st.image(windowed_rgb, caption=f"Active Preset: {preset_choice.split('(')[0]} (W: {w_slider}, L: {l_slider})", width='stretch')

        # DICOM Header Metadata Table
        st.markdown("#### DICOM Header Metadata & Scanner Parameters")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Modality & Sequence", f"{meta['modality']} · {meta['sequence_name'][:14]}")
        mcol2.metric("Magnetic Field", meta["magnetic_field_strength"])
        mcol3.metric("Repetition Time (TR)", meta["repetition_time_tr"])
        mcol4.metric("Echo Time (TE)", meta["echo_time_te"])

        with st.expander("Inspect Full DICOM Header Tags Table", expanded=False):
            st.table(meta)

        # Download sample demonstration DICOM
        demo_dcm_bytes = parser.create_demonstration_dicom()
        st.download_button(
            label="Download Demonstration DICOM File (.dcm)",
            data=demo_dcm_bytes,
            file_name="neuroscan_sample_clinical.dcm",
            mime="application/dicom",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Error parsing DICOM file: {str(e)}")
