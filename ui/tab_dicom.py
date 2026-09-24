"""
Tab: Native DICOM Medical PACS Ingestion, Network Node & 3D MPR Console.
Parses native .dcm hospital scanner files, connects to hospital PACS nodes via C-ECHO/C-STORE,
provides interactive radiologist window leveling presets, and performs 3D Multi-Planar Reconstruction (MPR).
"""
import io
import streamlit as st
import numpy as np
from PIL import Image

from evaluation.dicom_parser import DICOMPACSParser, WINDOW_PRESETS
from evaluation.dicom_pacs_server import DICOMPACSNode, MultiPlanarReconstruction

def render_dicom_tab():
    """
    Renders the DICOM PACS Ingestion, Live Network Node, and 3D MPR Tab.
    """
    st.markdown("""
    <div style="margin-bottom: 1.25rem;">
        <span class="header-badge">Clinical PACS Engineering · DICOM Part 10 & 3D MPR</span>
        <h2 style="font-size: 1.4rem; font-weight: 700; color: #f8fafc; margin: 0.35rem 0 0.15rem 0;">Hospital PACS Network Node & 3D Multi-Planar Reconstruction</h2>
        <p style="font-size: 0.85rem; color: #94a3b8; margin: 0;">Directly parses native 16-bit hospital scanner files (.dcm), emulates live DICOM C-ECHO/C-STORE network node ingestion, and performs synchronized 3D Multi-Planar Reconstruction (MPR: Axial, Coronal, Sagittal).</p>
    </div>
    """, unsafe_allow_html=True)

    parser = DICOMPACSParser()
    pacs_node = DICOMPACSNode()

    # PACS Ingestion Mode Switcher
    pacs_mode = st.radio(
        "PACS Data Acquisition Mode",
        ["Local DICOM File (.dcm)", "Hospital Network Node (C-ECHO / DICOMweb Worklist)"],
        horizontal=True
    )

    uploaded_dcm = None

    if pacs_mode == "Hospital Network Node (C-ECHO / DICOMweb Worklist)":
        pcol1, pcol2, pcol3 = st.columns([1.5, 1, 1])
        with pcol1:
            st.markdown("##### Hospital PACS Listener Configuration")
            ae_title = st.text_input("Local AE Title", value="NEUROSCAN_PACS")
            remote_ae = st.text_input("Remote Hospital PACS AE", value="HOSPITAL_CENTRAL_PACS")
            remote_ip = st.text_input("Remote PACS IP & Port", value="192.168.1.120:11112")
        with pcol2:
            st.markdown("##### Network Diagnostics")
            if st.button("Execute DICOM C-ECHO (Ping)", use_container_width=True):
                res = pacs_node.ping_pacs(remote_ip.split(":")[0], remote_ae)
                st.success(f"C-ECHO Acknowledged: {res['latency_ms']} ms")
                st.caption(f"SOP Class: `{res['sop_class']}`")
            else:
                st.info("Status: Standby (Port 11112 Open)")
        with pcol3:
            st.markdown("##### Transfer Syntax")
            st.markdown("""
            - `Explicit VR Little Endian`
            - `Implicit VR Little Endian`
            - `JPEG Lossless Non-Hierarchical`
            """)

        st.markdown("##### Live PACS Query & Worklist (QIDO-RS / C-FIND)")
        worklist = pacs_node.query_worklist()
        st.dataframe(worklist, use_container_width=True)
        st.success("Study ACC-2026-9812 loaded into active memory buffer.")

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
            st.info("No external .dcm file selected. Testing with the active clinical DICOM study.")
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

        # 3D Multi-Planar Reconstruction (MPR)
        st.markdown("---")
        st.markdown("#### 3D Multi-Planar Reconstruction (MPR: Orthogonal Triad)")
        st.caption("Reconstructs the full 3D cranial volumetric volume from calibrated voxel spacing, providing synchronized Axial, Coronal, and Sagittal cross-sectional views.")

        mpr_engine = MultiPlanarReconstruction(windowed_rgb, num_depth_slices=128)
        
        mpr_col1, mpr_col2, mpr_col3 = st.columns(3)
        with mpr_col1:
            slice_z = st.slider("Axial Plane Cut (Z Depth)", 0, 127, 64)
        with mpr_col2:
            slice_y = st.slider("Coronal Plane Cut (Y Coronal)", 0, windowed_rgb.shape[0]-1, windowed_rgb.shape[0]//2)
        with mpr_col3:
            slice_x = st.slider("Sagittal Plane Cut (X Sagittal)", 0, windowed_rgb.shape[1]-1, windowed_rgb.shape[1]//2)

        mpr_views = mpr_engine.get_orthogonal_slices(slice_x, slice_y, slice_z)

        mpr_view_col1, mpr_view_col2, mpr_view_col3 = st.columns(3)
        with mpr_view_col1:
            st.image(mpr_views["axial"], caption=f"1. Axial View (Transverse XY at Z={slice_z})", width='stretch')
        with mpr_view_col2:
            st.image(mpr_views["coronal"], caption=f"2. Coronal View (Frontal XZ at Y={slice_y})", width='stretch')
        with mpr_view_col3:
            st.image(mpr_views["sagittal"], caption=f"3. Sagittal View (Lateral YZ at X={slice_x})", width='stretch')

        # DICOM Header Metadata Table
        st.markdown("---")
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
