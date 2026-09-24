import streamlit as st

def render_patient_sidebar():
    """Renders the patient session diagnostic log and rekam medis tracking in the sidebar."""
    if "diagnostic_history" not in st.session_state:
        st.session_state.diagnostic_history = []

    with st.sidebar:
        st.markdown("### Patient Diagnostic Log")
        st.caption("Active session rekam medis records")

        history_count = len(st.session_state.diagnostic_history)
        st.markdown(f"**Total Evaluated Scans:** `{history_count}`")

        if history_count == 0:
            st.info("Belum ada scan yang dianalisis dalam sesi ini. Upload citra MRI di workstation utama untuk memulai.")
        else:
            for item in st.session_state.diagnostic_history[:8]:
                badge_class = "badge-pos" if item["finding"] == "POSITIVE" else "badge-neg"
                st.markdown(f"""
                <div class="history-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                        <span style="font-size: 0.8rem; font-weight: 600; color: #f8fafc;">{item['scan_id'][:16]}</span>
                        <span class="{badge_class}">{item['finding']}</span>
                    </div>
                    <div style="font-size: 0.75rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">
                        {item['timestamp']} · {item['confidence']} · {item['engine']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if st.button("Clear Session History", use_container_width=True):
                st.session_state.diagnostic_history = []
                st.rerun()
