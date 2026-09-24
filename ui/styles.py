import streamlit as st

CLINICAL_CSS = """
<style>
    /* Global Base */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #e2e8f0;
    }

    .stApp {
        background-color: #090d16;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0b0f1a;
        border-right: 1px solid #1e293b;
    }

    /* Header Styling */
    .clinical-header {
        border-bottom: 1px solid #1e293b;
        padding-bottom: 1.25rem;
        margin-bottom: 1.5rem;
    }
    .header-badge {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.08);
        border: 1px solid rgba(56, 189, 248, 0.25);
        padding: 3px 8px;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    .header-title {
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #f8fafc;
        margin: 0;
    }
    .header-subtitle {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 0.35rem;
    }

    /* Card Panels */
    .clinical-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    .metric-chip {
        font-family: 'JetBrains Mono', monospace;
        background: #1e293b;
        color: #94a3b8;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        display: inline-block;
        margin-top: 4px;
    }

    /* Result Banners */
    .banner-tumor {
        background: rgba(220, 38, 38, 0.1);
        border: 1px solid rgba(220, 38, 38, 0.4);
        border-left: 4px solid #ef4444;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }
    .banner-normal {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-left: 4px solid #10b981;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }

    .banner-title {
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin-bottom: 0.25rem;
    }
    .banner-desc {
        font-size: 0.85rem;
        color: #cbd5e1;
        margin: 0;
    }

    /* History Table Items */
    .history-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 6px;
        padding: 0.75rem;
        margin-bottom: 0.6rem;
    }
    .badge-pos {
        color: #f87171;
        background: rgba(239, 68, 68, 0.15);
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-neg {
        color: #34d399;
        background: rgba(16, 185, 129, 0.15);
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        border-bottom: 1px solid #1e293b;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 0.25rem;
        font-size: 0.9rem;
        font-weight: 500;
        color: #94a3b8;
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom-color: #38bdf8 !important;
        background: transparent !important;
    }

    /* Buttons */
    .stButton>button {
        background: #0284c7;
        color: #ffffff;
        font-weight: 600;
        border: 1px solid #0369a1;
        border-radius: 6px;
        padding: 0.5rem 1.25rem;
        letter-spacing: -0.01em;
        transition: all 0.15s ease;
    }
    .stButton>button:hover {
        background: #0369a1;
        border-color: #075985;
        color: #ffffff;
    }

    /* Custom Confidence Bar */
    .conf-bar-wrap {
        background: #1e293b;
        border-radius: 4px;
        height: 8px;
        width: 100%;
        overflow: hidden;
        margin: 0.5rem 0 1rem 0;
    }
    .conf-bar-fill-tumor {
        background: #ef4444;
        height: 100%;
        border-radius: 4px;
    }
    .conf-bar-fill-normal {
        background: #10b981;
        height: 100%;
        border-radius: 4px;
    }
</style>
"""

def apply_clinical_theme():
    """Injects high-end clinical workstation CSS and typography into Streamlit."""
    st.markdown(CLINICAL_CSS, unsafe_allow_html=True)
